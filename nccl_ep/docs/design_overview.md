# NCCL EP 设计要点与实现机制

本文档从源码角度总结 NCCL EP（Expert Parallelism）的设计取舍与实现机制，覆盖公开 API/ABI、主机端资源与生命周期、LL/HT 两套设备内核、运行时 JIT、量化 recipe、零拷贝与容错等。文中路径均相对于 `nccl_ep/`，关键位置给出行号以便对照阅读。

## 目录

1. [定位与总体架构](#1-定位与总体架构)
2. [API 与核心抽象](#2-api-与核心抽象)
3. [LL 内核实现机制](#3-ll-内核实现机制)
4. [HT 内核实现机制](#4-ht-内核实现机制)
5. [JIT 机制](#5-jit-机制)
6. [量化 Recipe](#6-量化-recipe)
7. [零拷贝、溢出与流的协同](#7-零拷贝溢出与流的协同)
8. [设计哲学总结](#8-设计哲学总结)
9. [关键源码入口](#9-关键源码入口)

---

## 1. 定位与总体架构

NCCL EP 是构建在 NCCL Device API 之上的 MoE 通信库：

- **LSA**（Load-Store Accessible）：节点内 NVLink load/store；
- **GIN**（GPU-Initiated Networking）：跨节点 RDMA put/signal。

它提供两个原语：

- `ncclEpDispatch`：按 top-k 路由把 token 发送给各专家；
- `ncclEpCombine`：把专家输出按原 token 顺序收回并做加权规约。

核心取舍是**同一套 API、两套完全不同的内核**：

| | LL（Low Latency） | HT（High Throughput） |
|---|---|---|
| 目标 | 小 batch、推理、亚毫秒延迟 | 大 batch、训练 / prefill、高带宽 |
| 通信模式 | 纯点对点 all-to-all | 分层：节点内 LSA 聚合 + 节点间 RDMA |
| 硬件特性 | 直接 P2P store、epoch 双 bank、可选 CFT | warp 专用化流水、TMA、mbarrier、grid barrier |
| Layout | `EXPERT_MAJOR` / `RANK_MAJOR` | `FLAT` / `EXPERT_MAJOR` |
| 特有机制 | `send_only` + `Complete` 分阶段、mask 容错 | push/pull flow、fused metadata、overflow、eager、compact-prob |

代码分层：

- **API/ABI 层**：`include/nccl_ep.h`、`include/ep_enums.h`
- **主机层**：`nccl_ep.cc`（Group/Handle/Ops 主体）、`nccl_ep_cft.cc`（可选 CFT 传输）、`nccl_ep_env.cc`（env 解析）、`ht_chunk_config.hpp` / `ht_sm_policy.hpp`（HT 策略）
- **内核层**：`device/ll_ep.cuh` + `device/ll/*.cuh`、`device/ht_ep.cuh` + `device/ht_ep_configs.cuh`
- **JIT 层**：`device/jit/*.cc/.hpp/.cuh`（运行时编译 kernel）
- **Adapter**：`ll_ep_adapter.cu` / `ht_ep_adapter.cu` 负责 launch planning，再交给 `jit::launch_*`

---

## 2. API 与核心抽象

### 2.1 稳定的 ABI 设计

所有按值传入的公共结构体都以固定前缀 `unsigned size` + `unsigned magic`（`NCCL_EP_MAGIC = 0xC00FFFEE`）开头；`ncclEpGroupConfig_t` 另有 `version`（当前 API v4）。结构体是**只追加（append-only）**的：每个发布版本冻结 `XXX_Vn_LAST_FIELD` / `XXX_Vn_SIZE`，用 `padding_vN` 显式吃掉对齐填充，防止未来字段占用已发布字节。

- 编译期：`NCCL_EP_STATIC_ASSERT_STRUCT_ABI` / `NCCL_EP_STATIC_ASSERT_STRUCT_ABI_BOUNDARY` 断言字段偏移与版本边界；`tests/test_public_struct_abi.cu` 用大量 `static_assert(offsetof(...))` 锁死布局。
- 运行期：`epValidateStruct()` 校验最小尺寸与 magic；`epDecodeStruct()` 以当前库结构为蓝本 `memcpy(min(size, sizeof))`，既接受旧调用者的小前缀（新字段取默认值），也忽略新调用者的未知尾部，实现双向兼容。
- 库版本 `NCCL_EP_MAJOR/MINOR/PATCH` 定义在 `nccl_ep.h`，构建系统解析它生成 `libnccl_ep.so.MAJOR.MINOR.PATCH`（SONAME 为 `libnccl_ep.so.MAJOR`）。

### 2.2 三层句柄生命周期

**Group**（`ncclEpCreateGroup`，集合通信）：从 NCCL communicator + config 创建，持有所有长期资源——workspace、NCCL window、devComm、LL 的 `rdma_buffer` / `ll_epoch_state`、HT 的 IPC mega-buffer / GIN buffer、mask buffer 等；销毁时按逆序释放。

**Handle**（四个入口，各有存在理由）：

- `ncclEpInitHandle`：非集合，只分配路由缓冲（HT 可传 `handle_mem` 由调用方拥有），不绑定路由；
- `ncclEpUpdateHandle`：每步集合操作，绑定 `topk_idx [num_tokens, top_k]`，在 HT 上构建路由图 / 计数 / 偏移；LL 上几乎为空（路由在 dispatch 时现算）；
- `ncclEpCreateHandle` = `Init + Update` 的便利封装；
- `ncclEpImportHandle`：HT-only、纯本地，从已由 UpdateHandle 准备好并**快照的 `handle_mem`** 重建句柄，不发射 kernel、不做扫描——专为 CUDA Graph / checkpoint 场景。

拆分 Init/Update 的根本原因是 **CUDA Graph**：含集合通信的 Init 必须在 capture 之前完成。

**Tensor**：轻量描述符，用户拥有 `data` 与 `sizes`，可栈上声明；`win_hdl` / `win_offset` 用于零拷贝。库分配的用内部 magic 并带 `sizes_shadow` 防篡改。

**Ops**：`ncclEpDispatch` / `ncclEpCombine` / `ncclEpComplete`，以及 AFD 的四个角色包装器（仅校验角色后转发）。

### 2.3 配置驱动的拓扑与资源

`ncclEpGroupConfig_t` 字段直接决定资源规模：

- `num_experts` → warp-group 几何 / 路由表大小；
- `max_dispatch/recv_tokens` → staging 容量（`AUTO` 在 HT 触发 eager 模式）；
- `rdma_buffer_size` → LL 是否延迟分配；
- `zero_copy` → 是否省略 staging 并切换专家复制算法；
- `num_qp_per_rank` / `num_channels` → GIN context；
- `ht_flow` / `meta_exchange_policy` → HT 数据流与元数据路径；
- `enable_mask` → mask buffer 与 host 异步错误标志。

`nccl_ep_env.cc` 集中解析约 35 个 `NCCL_EP_*` 环境变量（每个 group 创建时读一次到自己的副本），只做类型化；范围检查、默认值与优先级由各读取点负责（显式 API 配置通常优先于 env）。

---

## 3. LL 内核实现机制

### 3.1 通信模型与 LSA/GIN 双路径

LL 是**全设备端、无 host 参与**的点对点 all-to-all：每 SM 一个 CTA，token 按 `tokenIdx = smId; tokenIdx += numSms` 条带化。

- **LSA 路径**（同 NVLink 域）：`ncclGetP2pPtr` 解析对端指针，payload 用 split layout（头部区 + payload 区分离），完成信号用 `st_release_sys_global` 直接写对端计数。默认高性能路径是 `k2SidedRmLsaTma`（rank-major + TMA）。
- **GIN 路径**（跨节点）：`ginPut` 用 interleaved layout（`[hdr | data | scales]` 整块 put），0 字节 put 携带 `SignalAdd` 作信号。
- **选择逻辑**：`ll_dispatch_select_algo` / `ll_combine_select_kernel` 根据拓扑（`nvlinkOnly`）、layout、recipe、env（`NCCL_EP_DISPATCH_LSA`、`NCCL_EP_COMBINE_LSA`）选 universal / LSA SIMT / LSA TMA / CFT。AFD 强制走通用内核。

### 3.2 Epoch 双 bank 协议（`send_only` + `Complete` 的基础）

recv/count/flag 缓冲按 `epoch & 1` 分成两个 bank 复用。`LowLatencyEpochState{epoch, pending_epoch, send_in_flight}`：

- 纯 SEND（`send_only=1`）用 `epoch`，由 sm0/t0 记为 `pending_epoch`；`ncclEpComplete()` 走 RECV 阶段用 `pending_epoch`，完成后 `++epoch`；
- 同步调用（SEND|RECV）在 send 边界直接 `++epoch`；
- LSA 专用内核用 **last-CTA 选举**（`atomic_add_release_global(ctaDone)` + release/acquire fence）替代 grid 级 barrier，降低延迟。

发布/消费使用 `fence.release.sys` / `fence.acquire.sys` 等包装，全部集中在 `device/device_primitives.cuh`（仓库编码规范禁止在该文件之外写 inline asm）。

### 3.3 Dispatch 与 Combine 数据流

**Dispatch**：每 warp 负责一个 top-k expert；同一 token 指向同一 rank 的多个 expert 只由最小 topk 槽的 warp 发送（去重）；`atomicAdd(rankSentCnt + dstRank)` 分配对端槽；接收端按 `-numTokensSent - 1` 解码计数。

- **Expert-major**：每专家一段连续的 "zone"（`[localExpert * nRanks * maxTPR, ...)`），用 `atomicAdd(outCnt[le])` 分配槽，需要等齐 `numLocalExperts` 个通道；
- **Rank-major**：直接写 `[srcRank, slot, :]`，同时写 `recv_topk_idx`（LOCAL/GLOBAL/-1）、权重和 `srcInfo`（供 combine 用槽号）。

**Combine**：把专家输出发回源 rank，在 FP32 累加器中规约。

- **Expert-major**：接收侧按 `topkWeights` 加权累加；
- **Rank-major**：权重由调用方在 combine 前预折进 expert 输出，kernel 用 `weight = 1`；
- **1-sided（pull）变体**：本端不发 token，直接从对端 `inData` window 用 TMA 拉取，靠 read-done 标志保护下一轮覆盖；`NCCL_EP_COMBINE_LSA=auto` 由设备端 `want1SidedBuf` 按阈值自选 1/2-sided。

### 3.4 容错与 AFD

**Active-mask（仅 LL）**：`rankMask[r]`（1=活跃）；发送前跳过 masked rank，等待超时则隔离该 rank 并置 host 可见的 `asyncErrorFlag`。`MaskClean` 用独立注册的 `sync_buffer` 做 mask-aware barrier、清零两个 bank 的 signal 区，再全部置 1。注意 **CFT 与 masking 互斥**。

**CFT（Compute Fabric Transport，可选）**：用 CUDA logical endpoints（Blackwell+ / CUDA 13.3+）在单 NVLink 域内传 dispatch payload，靠硬件字节计数器判断到达；driver 函数指针动态解析，避免新符号成为 dlopen 硬依赖；全组必须一致（`LlGroupVote` AllGather），任一 rank 失败则整组报错。

**AFD（Attention-FFN 解耦）**：`num_attn_ranks` 把 communicator 分成 A-rank（拥有 token）和 F-rank（拥有 experts）；角色烧进 JIT 模板常量；A-rank 用 `ncclEpAfDispatch` / `ncclEpAfCombine`，F-rank 用 `ncclEpAfRecv` / `ncclEpAfReturn`；仅支持 LL + rank-major + 未量化。

---

## 4. HT 内核实现机制

### 4.1 分层通信与 warp 专用化

HT 把一次操作拆成节点内（LSA barrier + IPC peer 指针数组）和节点间（GIN put/signal + window）两级。单个 kernel block 内按 `threadIdx` 区间划分专职 warp：

- Dispatch：`GIN_GROUP`（跨节点）、`LSA_G2S_GROUP`、`LSA_S2G_GROUP`（每 pipeline 一 warp）、`MAP_GROUP`（count 模式建 S2D/LERM）、`PAD_GROUP`（EM 对齐填零）；head 3 warp 做 mbarrier 初始化 / RDMA guard / LSA barrier，tail 用 `elect_last_block` 推进轮次；
- Combine：`LSA_RED_GROUP`、`CROSS_LSA_RED_GROUP`（各 4 warp，FP32 寄存器累加，`NCCL_EP_COMBINE_MAX_ACC = 64`）、G2S / GIN warp。

token bundle 全部用 **TMA**（`cp.async.bulk`）在 global↔shared 间搬运，配合每 stage 一对 producer/consumer mbarrier 的 FIFO 流水。

### 4.2 元数据：scan vs fused count

路由输入是稀疏 `topk_idx`，HT 需在 handle 阶段构造 `sparse_to_dense_map (S2D)`、local-expert routing map (LERM)、每专家 `counts/offsets`：

- **scan 路径**（通用）：`scan_kernel.cuh` 做跨 block 前缀扫描，构建 EM 表，在 UpdateHandle 期完成一次独立集合通信；
- **fused count 路径**（单 LSA team、固定预算）：`count_kernel.cuh` 只做 per-rank / per-expert 直方图，S2D/LERM/offset 的构建**融合进 dispatch 内核**（MAP warp 逐 chunk 构造并发布），count 行在 dispatch head 扇出，省掉独立的元数据交换阶段；
- 由 `meta_exchange_policy`（AUTO / REQUIRE_FUSED / FORCE_SCAN）与 env 控制；**eager 和 UpdateHandle 早期计数查询会退化到 scan / unfused**。

### 4.3 两种数据流（flow）

- **push-pull**（默认，多 LSA team）：dispatch = RDMA push 到远端 packed staging → 本节点 G2S/S2G fan-out；combine = 节点内 reduce → RDMA push → 跨节点规约。靠 `expected_gin_flag_val` / `expected_lsa_flag_val` 设备计数器实现 CUDA Graph replay 自定序。
- **pull-push**（单 LSA team，topk∈[1,32]）：接收端直接用 NVLink **拉**源 token 并 scatter 到 EM 槽（`PullTmaStage`），combine 直接 push 回并 reduce。省去 staging 往返，但要求单一 LSA 域。

### 4.4 溢出、eager、compact-prob

**Overflow（仅 HT）**：在元数据期对比真实 recv 数与预算，两处检查——库 staging（去重 token 行）和调用方输出（FLAT 去重；EM 是 token×专家对 + 对齐 pad，因此 EM 更容易先溢出）。`TRAP` 直接 `__trap()`；`DROP` 把越界槽置 -1、clamp 计数，`recv_total_counter` 仍报真实前 drop 值；EM permute 路径会**零填充** phantom row 避免读到脏数据。

**Eager**（`max_recv_tokens = AUTO`）：按每轮实际 recv 数分配输出；EM 需 `num_topk`；禁 CUDA Graph capture、禁 DROP；固定容量时用设备端 TMA/SIMT copy epilogue 避免 host 读回计数，对齐不满足则回退 CE。

**compact-prob**：把 `topk_weights` + `uint16` 专家 ID 打包成 token 级 16B 对齐紧凑行（仅当严格小于 dense 才启用），省 RDMA 字节；接收端 JIT kernel 解码并写 `recv_topk_idx/weights`，同时记录原 top-k 位置供 BWD 散射权重。

### 4.5 分块与 SMEM 自适应

`ht_chunk_config.hpp` / `ht_sm_policy.hpp` 在 host 端解析 `tokens_per_chunk`（RDMA 默认 64；单 team 按 `round_up_32(tokens_per_group * dispatch_sms)`）和 SM 预算（flat 16 / EM 24 / Rubin 32）。`disp_smem_cost_t` / `comb_smem_cost_t` 用双线性 cost 模型按设备 SMEM opt-in 求解 stages/pipelines；显式设置的值不会被自动 tuner 缩减。

---

## 5. JIT 机制

**AOT 只编译 host 胶水 + 两个 adapter；真正的 `__global__` kernel 全部定义在 `.cuh` 中，运行时由 nvcc 编译。** 原因是特化维度爆炸：算法（LL/HT）× layout × dtype × topk × warp 数 × 量化 recipe × LSA 模式 × 架构，AOT 穷举不可行；JIT 用模板参数把 recipe/枚举变为编译期常量，hot loop 无分支。

- **构建期**：Makefile 把 25 个 kernel 头 + 公共头 staged 到 `build/include/nccl_ep/`，并生成 `config.h`（写入源码目录、NCCL 头目录、CUDA 头目录的绝对路径）。
- **运行期**：`jit_runtime.cc` 拼出包含 `device/*_ep.cuh` 与模板实参的 `.cu` 源串 → `nvcc_compiler.cc` fork/exec `nvcc --cubin` → `cuModuleLoadDataEx` 加载 → `cuLaunchKernelEx` 启动。
- **三级缓存**：线程本地 4-slot → 进程级 fast cache（按 identity + runtime_key + CUcontext + device）→ 磁盘 cubin（默认 `/tmp/nccl_ep/jit`）。缓存键是 FNV-1a 指纹，包含 kernel family / variant / source / arch / 编译器 ID + **所有头文件的树指纹 + 编译选项**，改头、换编译器、改 flag 自动失效；原子写 + `flock` 保证并发安全，编译失败写 `compile.failed` 避免重复失败。
- **路径解析**：`NCCL_EP_HOME`（wheel 设成包目录）、`NCCL_EP_JIT_NVCC`、`NCCL_EP_JIT_CACHE_DIR` 等，使 relocated 安装 / wheel 可用。

---

## 6. 量化 Recipe

recipe 由 `config->quant_recipe` 唯一指定，**集中校验**，不从可选张量推断；作为 JIT 模板实参并进入缓存键；未处理的 recipe 在编译期因 traits 主模板未定义而报错。

- **FWD**（`NCCL_EP_DISP_QUANT_FWD`，LL+HT dispatch）：tokens 与 scales 按物理字节转发，不转换；要求 dtype / 行宽匹配、`round_scales = 0`、token/scale 行与基址 16B 对齐；
- **DS_FP8E3M4**（LL-only dispatch）：BF16 → E4M3 + 每 128 元素一个 FP32 scale，hidden 需被 512 整除；
- **MXFP8**（HT-only combine）：BF16 → FP8 E4M3 + 每 32 元素一个 E8M0 block scale（内部生成，输入 scales 必须为 NULL），hidden 需被 512 整除；仅单 LSA team 的 expert-major local-permute + push-pull flow，CUDA 12.8+；
- **NVFP4**（实验性，LL-only combine）：调用方提供 FP32 全局 scale。

16B 对齐（`int4` 传输）与 `max_token_bytes`（单 token 行含 scale 的物理字节上限）是贯穿各 recipe 的硬约束。

---

## 7. 零拷贝、溢出与流的协同

- **零拷贝**：通过 `ncclCommWindowRegister` 把用户 buffer 注册为 NCCL window，再挂到 `ncclEpTensor_t::win_hdl`。HT 支持 dispatch 输出 / combine 输入直接读写；LL 仅 dispatch 直写、combine 始终 staging。`zero_copy = ON` 会省略 HT token staging 区域，并切换到专家复制算法（local-dup / nvlink-dup），是一种性能与内存决策，而非单纯的开关。
- **一致性要求**：所有资源分配、window 注册、Group/Handle 创建、Dispatch/Combine 都是**集合操作**，所有 rank 必须以相同顺序、相同配置调用；跨 stream 使用需调用方自行同步。
- **兼容性**：EP 构建所依赖的 NCCL 版本不能新于运行时版本；2.30.5–2.30.x 构建 + 新 runtime 需 `NCCL_DEV_API_JIT=1`；JIT 头必须来自与 runtime 相同的 NCCL 安装。

---

## 8. 设计哲学总结

> 让 GPU 主导一切通信：host 只负责资源准备、拓扑/策略决策和 launch planning，数据搬运与同步全在设备端用 LSA/GIN 完成；用 **append-only ABI + 拆分 handle 生命周期**换取向后兼容与 CUDA Graph 支持；用**双 bank epoch + flag 字**实现无 host 参与的跨调用流水；用**运行时 JIT + 指纹缓存**消化特化维度爆炸；用**拓扑感知的 LL/HT 双内核 + 零拷贝 / 量化 / 容错开关**覆盖从亚毫秒推理到大规模训练的完整频谱。

---

## 9. 关键源码入口

均在 `nccl_ep/` 下：

| 主题 | 位置 |
|---|---|
| API / ABI | `include/nccl_ep.h`、`include/ep_enums.h` |
| Group 创建 | `nccl_ep.cc:2329` |
| Handle 结构 | `nccl_ep.cc:3330` |
| InitHandle / UpdateHandle | `nccl_ep.cc:4429` / `:4840` |
| Dispatch / Combine / Complete | `nccl_ep.cc:5535` / `:7265` / `:8131` |
| LL dispatch / combine 内核 | `device/ll_ep.cuh:497` / `:1780` |
| HT dispatch / combine 内核 | `device/ht_ep.cuh:5137` / `:5519` |
| JIT 编排 | `device/jit/jit_runtime.cc` |

补充：仓库内 `nccl_ep/docs/agents/` 有两份对 agent 有约束力的内部规范——`CODING_SKILL.md`（C++/CUDA 编码规则，如禁默认参数、禁嵌套三元、inline asm 只能放 `device_primitives.cuh`）和 `PLC_DOCS_SKILL.md`（PLC 设计文档永不编辑旧文档、目录须以 MR 号前缀）。修改该子项目代码或设计文档前应先阅读。

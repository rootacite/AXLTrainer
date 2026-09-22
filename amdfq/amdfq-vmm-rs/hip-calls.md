# amdfq-vmm-rs：hipPointerGetAttributes 与 hipMemGetInfo 的实测

**这份文件只记录 2026-09-19 这一轮的两个问题**：torch 会不会对 `hipMalloc`（被 peralloc 路线偷换）
分配的指针调用 `hipPointerGetAttributes`、路线的替换是否让这个 API 的答案异常；以及
`hipMemGetInfo` 能不能统计到路线分配的内存。路线的设计、布局与不变量在
路线与不变量在 [`DESIGN.md`](DESIGN.md)；更早的路线差异记录（DIFF.md）已移除，这里不重述。

结论先写：**两个 API 对路线服务的块都给出与真 `hipMalloc` 相同的答案**（两处超出请求范围的差异
torch 一处都不读），**路线内存被 `hipMemGetInfo` 如实计入**，系统性代价只有每进程 2 MiB 的共享 pad。

**环境与状态。** `torch 2.13.0+rocm10.0.0`、HIP `7.15.26333`、gfx1201（RX 9070 XT 16 GB）、conda 环境
`axl`；hook 为 `amdfq/amdfq-vmm-rs/target/release/libamdfq_vmm_rs.so`（md5 `45b58da64330e42502b6b2f7760a1091`，
本轮未改动）；`config.toml` 未改动。

**怎么复现。** 工具都在 scratch 目录，未入版本库（`/tmp` 会被清空，本轮开始时上一轮的 `/tmp/exp`
已经不在了）：

- `/tmp/attrprobe/attrprobe.cpp` — `hipcc -O1` 探针，同一支二进制分别加/不加 `LD_PRELOAD` 跑，逐字段
  打印两个 API 的答案与 `hipMemGetInfo` 的记账。输出 `/tmp/attrprobe/{base3,hook3}.txt`。
- `/tmp/exp/hipwatch.c` — 观察者垫片（`gcc -shared -fPIC`）。只经 `RTLD_NEXT` 转发并记录
  `hipPointerGetAttributes` / `hipPointerGetAttribute` / `hipMemGetInfo` 的每次调用：指针、返回码、
  属性、调用栈（`dladdr` 给 module+offset）。纯观察，不改变行为。
- `/tmp/exp/torch_api_touch.py`（逐个触碰可能走到这些 API 的 torch 接口）、`/tmp/exp/mem_acct.py`
  （`torch.cuda.mem_get_info()` 的记账）。
- 两次真实训练运行：`/tmp/exp/base/`（不带 hook）、`/tmp/exp/hooked/`（带 hook），各自
  `PYTHONPATH` 用 `axl` 环境，`HIPWATCH_LOG` 指向该目录，stderr 里混着 hook 的日志。

## 1. torch 会不会对 hipMalloc 的指针调用它

### 1.1 全部调用点（读）

读法：`objdump -h` 取 `.text`，扫 `e8 rel32` 且目标为对应 PLT stub 的调用，再用 `nm -n` 反查所在函数。
`site-packages` 下只有三个库引用属性 API；相对每个 `.so` 的偏移如下。

| 库 | 符号 | 调用点 | 用途 |
| --- | --- | --- | --- |
| `libtorch_hip.so` | `hipPointerGetAttributes@hip_4.2`（复数） | `at::cuda::detail::CUDAHooks::isPinnedPtr(void const*)`（函数 `+0xe92710`，调用 `+0xe92758`） | `+0xe92765` 比较 `attr.type == 1`（`hipMemoryTypeHost`），即 `Tensor.is_pinned()` / `at::native::pin_memory` |
| | | `at::cuda::detail::CUDAHooks::getDeviceFromPtr(void*)`（函数 `+0xe91fa0`，调用 `+0xe91fb4`） | 返回码直接进 `c10_cuda_check_implementation`（失败即抛），结果取 `attr.device`（`+0xe91fd9` 按 16 位读） |
| | | `gloo::getGPUIDForPointer(void const*)` | gloo/NCCL 传输层认设备（本栈不使用分布式） |
| `libtorch_python.so` | `hipPointerGetAttribute@hip_5.0`（单数） | `(anonymous namespace)::parseKernelArgs`（函数 `+0xa12460`，调用 `+0xa1284b`），属性号 `3` = `HIP_POINTER_ATTRIBUTE_DEVICE_POINTER` | 老 JIT fuser（紧邻 `launchKernel`/`launch_kernel`/`load_kernel`），返回值直接当 kernel 参数 |
| `libtorch_rocshmem.so` | `hipPointerGetAttributes` | 1 处 | symmetric memory，本栈不使用 |

### 1.2 真实训练里这些调用落在谁身上（测）

观察者垫片记录，两次运行：

| 运行 | 属性查询 | 全来自 | 返回值 | 落在路线服务的 extent 内 |
| --- | --- | --- | --- | --- |
| 基线训练（10 步，20 worker；exit 134，step 10 `HSA_STATUS_ERROR_MEMORY_FAULT`，已知的 Tensile 越读缺陷，其记录已封存到 `../../archive/`） | 102 次 | `CUDAHooks::isPinnedPtr`（链条 `at::native::pin_memory` → `at::_ops::pin_memory::call` → `at::_ops::is_pinned::call`） | 102 次全是 `type=0 Unregistered, dev=-2`，两个交替的**主机**地址（DataLoader 共享内存） | 不适用（没有 hook） |
| 带 hook 训练（step 0 结束前死于 Tensile `Cijk_Alik_Bljk_BBS_…_MT64x64x64` 的 `ILLEGAL_INSTRUCTION`；742 served / 0 forwarded / 0 WARN） | 82 次 | 同上 | 82 次全是 `type=0 Unregistered, dev=-2` | **0 / 82**（对着当次 742 个 extent 做包含判断） |

带 hook 那次观察到的两个交替地址都不是路线服务的：一个落在 served VA 跨度之外，另一个落在跨度内但
不属于任何 extent（即来自不经 PLT 的映射，例如 DataLoader 的共享内存或运行时内部申请）。
`getDeviceFromPtr` 与 fuser 的单数调用在本训练栈一次都没触发。

另外单测过一次接口面（测，`torch_api_touch.py` 加垫片）：`torch.empty(..., device='cuda').is_pinned()`
**根本不问运行时**（ATen 对非 CPU 张量直接返回 false），会产生调用的是 `pin_memory=True` 的张量
（`type=1 Host`）和可页换出的 CPU 张量（`type=0 Unregistered`）——两者都不是 `hipMalloc` 的内存。

### 1.3 假设真问到了路线服务的块，答案是否异常（测）

同一支探针，分别不加/加 hook（`/tmp/attrprobe/attrprobe.cpp`）：

| 字段 | 真 `hipMalloc(512 MiB)` | 路线服务的 512 MiB 块 |
| --- | --- | --- |
| 复数 API | `ret=0, type=2(Device), device=0, devicePointer=ptr, hostPointer=nil, isManaged=0, flags=0` | 完全相同 |
| 单数 API | `MEMORY_TYPE=2, DEVICE_POINTER=ptr, DEVICE_ORDINAL=0, RANGE_START_ADDR=ptr, MAPPED=1` | 完全相同 |

只有两处不同，都在**请求范围之外**：

1. `RANGE_SIZE`：基线是精确请求，路线是取整后的块。例：`hipMalloc(100 MiB + 1234)` 基线报
   `104858834`，路线报 `106954752`（= 102 MiB，2 MiB 粒度向上取整）。只有请求不是 2 MiB 整数倍时出现。
2. 块尾 pad 那一格（`va + round_up(size, 2 MiB)`）：基线回 `type=0 Unregistered / dev=-2`、单数 API 回
   `hipErrorInvalidValue`；路线回 `type=2(Device) / dev=0 / MAPPED=1`（pad 是真实映射——这正是它存在的
   目的）。`RANGE_SIZE` 之外，`BUFFER_ID` 也不同（路线按映射编号），torch 同样不使用。

torch 的四个调用点分别只读 `type`（`isPinnedPtr`）、`device`（`getDeviceFromPtr`）、
`DEVICE_POINTER`（fuser），没有一处读 `RANGE_SIZE`、`BUFFER_ID`，也没有一处会去查一个不属于自己的地址。

## 2. hipMemGetInfo 能不能统计到路线分配的内存

**能**，按句柄算、不按映射算（测，探针两臂对照）：

| 步骤 | 基线 | 路线 |
| --- | --- | --- |
| `hipMemAddressReserve`（只占 VA，未映射） | +0 | +0 |
| `hipMemCreate` 256 MiB | 立即 +256 MiB | 立即 +256 MiB |
| 同一句柄再 `hipMemMap`（含 200 个地址映射同一个 2 MiB pad 句柄） | +0 | **+0** |
| `hipMemUnmap` + `hipMemRelease` + `hipMemAddressFree` | 归还 | 归还 |

torch 层面同样（测，`mem_acct.py`，每种模式各 3 次）：分配 1×128 MiB → `mem_get_info()` 的 free 正好降
`128.0 MiB`；4 个块 → 正好 `512.0 MiB`；两种模式 3/3 一致。

**唯一的系统性残留是 pad 句柄的 2 MiB**：它每进程创建一次、被所有块共享、永不释放，所以带 hook 的
torch 进程 `mem_get_info()` 的起始值稳定比不带 hook 低 2 MiB（`16074` → `16072`，3/3）。此外运行时报的
数字本身有 **±14 MiB 的偶发抖动**，两种模式都出现过（不带 hook 的 torch 脚本有一次收尾 free 比起点高
14 MiB；带 hook 的探针有一次在 200 个映射处 +14 MiB），不是路线的计费，**未归因**。

消费方（测，训练运行）：基线跑里只有 `libMIOpen` 问过这个 API（2 次，conv 时 sizing workspace；带 hook 跑
1 次），torch 自身的 `torch.cuda.mem_get_info()` 在训练里从未被调用。因为路线内存被如实计入，MIOpen 看到
的是诚实的数字；代价是每进程多算一个 2 MiB 的 pad。

## 3. 边界（本轮没有覆盖的）

- 探针是**单线程**纯 HIP 程序；torch 内部多线程并发下的行为只在真实训练运行里顺带看过，没有专门构造。
- ±14 MiB 抖动未归因，只能说它在两种模式下都会出现。
- `getDeviceFromPtr`、JIT fuser 的单数调用、gloo 的 `getGPUIDForPointer` 在本训练栈未触发；将来若启用
  分布式或 JIT fuser，需要重测（按 §1.1 的调用点，届时被查的会是设备指针，届时按 §1.3 比对即可）。

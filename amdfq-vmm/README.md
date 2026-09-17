# amdfq-vmm — 优化路线一的实现

这是 [`../amdfq.md`](../amdfq.md) §12「一个基于 VMM API 的分配器」那套代码的复刻：接管
`hipMalloc`，用 `hipMemAddressReserve` / `hipMemCreate` / `hipMemMap` / `hipMemSetAccess`
自己排布，并在**每一块后面挂同一个共享 pad granule**，这样防越界的 slack 从"每个活块一颗
granule"降到"整个进程一颗"。

来源是未版本化的备份 `~/Desktop/amdfq/`，不是这份仓库。叙事（第九幕）见
[`../amdfq/doc/gfx1201-overread-story.md`](../amdfq/doc/gfx1201-overread-story.md)；
成因与缓解的权威记录见 [`../amdfq.md`](../amdfq.md) §10.3 与 §12、§13，以及
[`../conclusions/bf16-kernel-overrun.md`](../conclusions/bf16-kernel-overrun.md)。

2026-09-18 改造：这个目录现在**只剩 peralloc 一条路**。arena 与空闲表、shadow、尾部守卫，以及
`AMDFQ_VMM` / `AMDFQ_VMM_PAD` / `AMDFQ_VMM_LAYOUT` / `AMDFQ_VMM_RESERVE` / `AMDFQ_VMM_SHADOW` /
`AMDFQ_TAIL` 六个开关都从代码里删掉了；剩下的是"每笔分配各自 reserve、free 时
`hipMemAddressFree`，块后挂同一个共享 pad granule"。同一天早些时候的复现（§11）与探针记录（§10）
用的是改造前那一版，口径见那两节。

一句话现状：**机制成立，设计不合格**——越界读被堵住了，但它用五种互不相同的死法把进程玩坏了，
五种都没有解释。这个目录存在的理由就是接着查那五种死法（§12.4 / §12.5）。

2026-09-18 用真实 GPU 跑了 18 次训练做复现（路线一 13 次，全部失败；另 5 次是无 preload 的
canary），逐 run 结果、结局分类与三条新发现见 [§11](#11-用真实-gpu-复现路线一的故障2026-09-18)。
本文中形如 `§12.2`、`§10.1` 的编号除特别说明外都指 [`../amdfq.md`](../amdfq.md) 的章节，`§11.x`
指本文自己的第 11 节。

## 1. 编译与使用

```bash
cmake -S amdfq-vmm -B amdfq-vmm/cmake-build-debug
cmake --build amdfq-vmm/cmake-build-debug          # -> cmake-build-debug/libamdfq-vmm.so

bash amdfq-vmm/hook.sh                             # 默认：env 解释器里一次 torch 调用
bash amdfq-vmm/hook.sh bash start_train.sh         # 真实训练
bash amdfq-vmm/hook.sh ./any/hip/program           # 任何调用 hipMalloc 的程序
```

`hook.sh` 会在缺 `.so` 时自己编译，跑完之后把日志打出来。直接用环境变量的写法等价：

```bash
LD_PRELOAD=$PWD/amdfq-vmm/cmake-build-debug/libamdfq-vmm.so bash start_train.sh
```

日志默认写 `/tmp/amdfq-hook-<pid>.log`，`AMDFQ_LOG` 可以覆盖（指向 FIFO 也行）。一行一个调用：
`seq elapsed_s T=tid <fn>(args) -> ret=… vmm=… caller=<object>+offset`，`vmm=` 说明这一笔被路线
怎么处理了（`served block=… pad=…`，或者把它转发的那个原因）。进程退出时 `fini` 打摘要：路线参数、
创建/释放的块数与字节数、每一个 fallback 计数。

**`start_train.sh` 不会自动 preload 它**（沿用 [`../amdfq.md`](../amdfq.md) §10.3 的规矩：
一个会从底层彻底改写内存布局的对象，不该在没人知情的情况下生效）。要用就显式带上 `LD_PRELOAD`。

## 2. 开关

模块自己没有开关了：**preload 就是开关**。2026-09-18 起这个目录只保留路线一的 peralloc 一条路
（每笔分配各自 `hipMemAddressReserve`、free 时 `hipMemAddressFree`，每块后面挂同一个共享 pad
granule），`AMDFQ_VMM`、`AMDFQ_VMM_PAD`、`AMDFQ_VMM_LAYOUT`、`AMDFQ_VMM_RESERVE`、
`AMDFQ_VMM_SHADOW`、`AMDFQ_TAIL` 都已删除——外部照旧传 `AMDFQ_VMM_LAYOUT=peralloc` 不会报错，只是
不再有任何作用。被删掉的那几版仍在未版本化备份 `~/Desktop/amdfq/amdfq_vmm.c` 里；路线二的尾部守卫
另有 `../amdfq/amdfq_tail.c`。

| 环境变量 | 行为 | 对应文档 |
| --- | --- | --- |
| `LD_PRELOAD=…/libamdfq-vmm.so` | 唯一入口：`hipMalloc`/`hipFree` 全部走 peralloc 路线 | §12 |
| `AMDFQ_LAUNCH_LOG=1`、`AMDFQ_LAUNCH_ARGS=1` | 打开 kernel 派发跟踪（默认每笔只做转发） | §12.2 #3 |
| `AMDFQ_PROBE_ALLOW_FAULTS=1` | 允许 `vmm_probe.sh` 跑那三个会故意 fault 的模式 | 见 §9 |

其余开关（`AMDFQ_LOG`、`AMDFQ_PROBE_*`）在各文件头部注释里。

## 3. 文件

| 文件 | 作用 |
| --- | --- |
| `amdfq_vmm.c` / `.h` | 路线一的分配器，peralloc 一条路：每笔分配各自 reserve、各自 free，块后挂同一个共享 pad granule；无 arena、无空闲表 |
| `amdfq_hooks_hip.c` | 三个分配门的 interposer：`hipMalloc`/`hipFree` 走上面的分配器，服务不了的一律按调用方的 size 原样转发（这一版没有第二种缓解），`hipHostMalloc` 只记录 |
| `amdfq_hooks_launch.c` / `_blaslt.c` / `_copy.c` | 诊断用：派发、hipBLASLt、拷贝族的跟踪（默认只转发） |
| `amdfq_log.c` / `.h` | 日志 fd、行长、每函数计数、`dladdr` 求 `caller=` |
| `amdfq.c` / `amdfq.h` / `amdfq_gates.h` | `init` 开日志、`fini` 打摘要；三个原型（自己声明，对象里不含 ROCm 头，`ldd` 干净） |
| `hook.sh` | 唯一的入口 |
| `vmm_probe.c` + `.sh` | go/no-go 与代价探针：一个 handle 能不能映射到多个地址、1000×2 MiB 两种做法各花多少 |
| `vmm_diff.c` + `.sh` | 同一个 runtime 描述一个 `hipMalloc` 块和一个路线块时的差别（§12.5 的依据） |
| `vmm_kernel_test.hip` + `.sh` | 在同一套布局上跑平凡 kernel，旁边放 `hipMalloc` 对照 |
| `gemm_repro.hip` + `.sh` | 经 hipBLASLt 派发的那个 bf16 GEMM——路线一死掉的地方 |

## 4. 它做到了什么

`bash amdfq-vmm/vmm_probe.sh`（本机 2026-09-18 复跑，非 faulting 模式）：

- `multimap`：**一个 handle 同时映射到多个虚拟地址**成立（写第二份映射、从第一份读回来一致）；
  同一个 pad handle 映射 8 次，物理上只记 2 MiB 一次；越过块尾去读，返回数据而不是 fault。
  这是整条路线依赖的那件 HIP 头文件没写死的事。
- `cost`：1000 × 2 MiB —— `hipMalloc(2 MiB + 16)` 花 **4000 MiB**、7.6 µs/次；VMM + 一个共享
  pad granule 花 **2000 MiB + 一颗 granule**、17.5 µs/次；unmap + release 13.0 µs/次。
  （§12.1 记的是 2014 MiB / 18.6 µs / 15.2 µs，同量级。）

## 5. 它坏在哪

§12.2 表里的五种，同一份 `config.toml`、同一个 seed，跨两次 boot（2026-09-17 07:15 前后）。
下表"构建"一列里出现的开关都是改造前的（2026-09-18 之前那一版），现在只剩第 1、5 行的构建
（第 5 行的"无 pad"要改成共享 pad granule，pad 开关已删）：

| # | 构建 | 发生了什么 |
| --- | --- | --- |
| 1 | VMM + 共享 pad granule + host access | 第一个训练步 `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`，发生在 torch 自己的 `vectorized_elementwise_kernel<4, bfloat16_copy_kernel…>` 里；rc 134；`dmesg` **没有页错误** |
| 2 | 同上但 `AMDFQ_VMM_PAD=0` | 同样的 abort，换了一个 kernel（Tensile bf16 GEMM `Cijk_Ailk_Bljk_BBS…MT128x128x16…ISA1201…`）→ **不是共享 pad granule 干的** |
| 3 | VMM + 派发跟踪 | **指令取指** fault：`Faulty UTCL2 client ID: SQC (inst)`，地址 `0x7fa218f2d000` 落在 1 TiB 预留里、但**比最高的已映射块还高 772.1 GiB**（当时 738 块只用了 9.6 GiB）。是一个 wave 的 PC，不是它的数据 |
| 4 | VMM arena + pad（`lllj_20260917_073015`） | step 1 `loss=nan`，别的什么都没有：`dmesg` 里**一个 GPU 事件都没有**；run 结束是因为 `control.write_state` 拒绝序列化 NaN |
| 5 | VMM peralloc、无 pad（§12.3） | `_074240`：step 1 NaN，`dmesg` 干净。`_074714`：step 1 loss 正常（0.9978），然后一次 GPU 页错误，地址 **`0x7f1bd361f000` 是这条路线从没分配过的** |

它们的共同点不在 kernel、不在算子、不在尺寸、不在阶段、不在 boot，而在**损坏的种类**：
坏的值（NaN）、非法的指令、坏的 PC、坏的指针——进程自己的内存与派发状态被四种不同方式打坏，
其中两次内核日志完全沉默。§10.1 那个缺陷正好相反：确定、绑定 bf16 Tensile kernel、每次都留下
`[gfxhub] page fault`。

**一条更正**：2026-09-17 那次 GPU reset（`ring reset failed` → MODE1 reset → `VRAM is lost due to
GPU reset!`）曾经被记在这几个 run 旁边，当天就查清是这个目录自己的测试程序
`vmm_kernel_test.hip` 的 `spin_for_flag`（pid 30224）在普通 `hipMalloc` 的块上自旋，不是路线的证据。
那个自旋 kernel 仍在文件里，它也是这里唯一能故意把设备卡死的东西。

## 6. 已经排除的

§12.3：布局（`peralloc` 把 arena/空闲表整个拿掉，反而**更早**死，连续两次 run 两种死法）、
VMM 调用序列本身（shadow 跑了 42,001 次镜像、累计 572 GiB 的 reserve/create/map，进程健健康康
111 步）、teardown（进程退出后 VRAM 回到空卡读数）。另外路线一自己的内存也不是那个野指针的来源
（`_074714` 的故障地址在路线块的 502 GiB 以下、不是 granule 对齐、1643 次分配与 638 次释放的流水里
根本没有它）。

于是只剩一个所有实验都没动过的变量：**torch 的 kernel 真的读写路线分配的页**。

## 7. 开放问题（= 下一步的实验清单）

§12.4，按最像机制的顺序：

1. **host access 描述符。** `hipMemSetAccess` 必须同时要 device 与 host 两个 location，宿主机代码
   才能在同地址读到设备指针——训练路径真的这么干（`trainer/loop.py`、`trainer/dataset.py`、
   `trainer/sampling.py` 里的 `.item()`），少了它进程会死在 `_local_scalar_dense_cuda`。它也是
   `hipMalloc` 的块天生有、这条路线必须自己造出来的唯一属性，而它从来没有在训练规模上被单独测过。
   要 bisect，得先有第二种让 `.item()` 工作的办法（staging 拷贝，或对相关张量改步长）。
2. **规模。** 失败的 run 常驻 10–12 GiB，而 shadow 只能守住 512 MiB（真实分配还得装进卡里）。
   "VMM 页存在、但小"被排除；"VMM 页存在、且在失败规模上"没有。把 shadow 的上限抬上去是最便宜的
   下一个探针，上限就是这张卡：§10.3 量到 run 未加 pad 的峰值只剩 2.9–3.4 GiB，2 GiB 上限会花掉
   大半，而 OOM 本身不构成任何证据。（shadow 已从本目录删除，做这条得先用备份
   `~/Desktop/amdfq/amdfq_vmm.c` 那版，或者针对 peralloc 重写一份。）
3. **free 路径的顺序。** `hipMemUnmap` / `hipMemRelease` 不是 stream-ordered 的，而路线的 `hipFree`
   在 runtime 一开口就调它们，中间没有任何东西保证 GPU 在这块上的早先工作已经做完。
   `AMDFQ_VMM_NOFREE=1`（只记账、不 unmap 不回收）被提出来过却没跑，代价是会漏显存、只够几步。

另外，如果机制是"逐个访问"而不是全局的，最便宜的 bisect 是**只服务某一类分配**（hipBLASLt 的
workspace、缓存分配器的 2 MiB 段、采样阶段的张量），其余留给 `hipMalloc`。

§12.5 的假说与它的测量（不需要改 hook）：torch 拿到块之后还会把地址交给别的内存管理 API，而这些
API 是**针对块本身**的、不是读写它当数据；`hipMalloc` 的内存与 VMM 内存在这些调用上回答不一定相同，
调用方按答案行事，坏掉的就是进程状态。`vmm_diff.c` 已经把差别量出来了（今天的复跑）：

| 查询 | `hipMalloc` 的块 | 路线的块 |
| --- | --- | --- |
| `hipMemGetAddressRange` | `rc=0`，自己的 base/size | `rc=0`，映射的 range |
| `hipMemGetAccess` | `rc=1`（`hipErrorInvalidValue`） | `rc=0`，`flags=0x3` |
| `hipMemRetainAllocationHandle` | `rc=1` | `rc=0`，有 handle |
| `/proc/self/maps` | `/dev/dri/renderD128` | `/dev/dri/renderD128` |

差别是真的、可观测的；§7 从来没查的是**谁在问**。要落地就做一次 Frida 全量 `hip*` 参数捕获
（`libamdhip64.so.7` 的 549 个导出函数 + HSA 入口点，像 `agent6.js` 那样），把每个调用的参数记下来
与本次 run 早先分配过的地址对照，输出"收到一个曾经分配过的地址"的调用列表——任何从一个设备指针
**推导**出别的东西的调用（range/attribute 查询、access/mapping、advice/prefetch、pool 与 IPC
import、unmap/release、memcpy 家族）都是两种内存可能分道扬镳的地方。

## 8. 记录缺口（读这份目录时要知道的）

1. **`amdfq_hooks_hip.c` 是重建的。** 备份里那份（与 `amdfq_vmm.c` 同为 09:06）已经被改成只挂路线二，
   从不调用 `amdfq_vmm_malloc`；真正服务过的那一版在本机没有任何副本（查过 `/tmp`、六个 JetBrains
   LocalHistory、全盘 `amdfq_*`）。指纹是 `amdfq_vmm_begin()` / `amdfq_vmm_end()` 在备份里已无调用者。
   重建时的接线是"服务优先，拒绝时退 §10.1 的 +16（`AMDFQ_VMM=0` 就是这条），`AMDFQ_TAIL=1` 时改用
   尾部守卫且不加 pad"；2026-09-18 改造后只剩"服务优先，否则按调用方的 size 原样转发"。
2. **服务行日志的字段名（`vmm=`、`block=`、`pad=`）是新定的**，原版那几行的字段在任何副本里都查不到。
   `padded=` 沿用 §9 记过的格式——那一版才有它，改造后 `hipMalloc` 的行只有一种形状（没有多问
   runtime 要过任何字节），§11 存档里 `padded=` 的出现只属于改造前的构建。
3. **shadow 与尾部守卫已删除。** 备份里的 shadow 是 09:06 之后的版本：每个镜像的每一颗 granule 都
   映射同一个共享对象、无上限；而 §12.3 里 `lllj_20260917_075937` 那次跑的是更早的带上限版本
   （常驻 410–512 MiB、26,097 次被上限驱逐）。§12.4 第 2 条说的"把 shadow 上限抬上去"指的是后者，
   需要时得从备份把那套代码拿回来（路线二的尾部守卫在 `../amdfq/amdfq_tail.c` 里另有实现）。
4. 为让两个目录不会互相 preload 错，这个工程的产物叫 `libamdfq-vmm.so`（`../amdfq/` 那份仍叫
   `libamdfq.so`）。CMake 工程名同理。

## 9. 不要随手跑的东西

- `vmm_probe.sh` 的 `access` / `overread` / `hostaccess device` 三个模式**故意**去读未映射内存，
  默认已被 `AMDFQ_PROBE_ALLOW_FAULTS` 挡住。这台机器和显示器共用同一张卡，一次 fault 会把 ring
  一起复位掉。
- `gemm_repro.sh` 单独跑是安全的（操作数是普通 `hipMalloc`），但它的看点正是在 hook 下把操作数换成
  VMM 块——那就是 fault 路径。
- 任何长跑之前先打 canary（[`../conclusions/gfx1201-fault-response-wedge.md`](../conclusions/gfx1201-fault-response-wedge.md)
  §4.3）：fault 响应本身会卡死，而卡死时"没崩"看起来跟"没越界"一模一样。
- 端到端复现 §12.2 的五种死法要真跑训练（`LD_PRELOAD=…libamdfq-vmm.so bash start_train.sh`）：
  已经做过，见 §11。做之前先读那一节的 §11.7——这台机器会在会话中途进入"故障不再上报"的状态，
  那时"没崩"与"没越界"分不开。改造后能重跑的只有 peralloc 一种配置（§2）。

## 10. 本目录的验证记录（2026-09-18）

**改造前那一版**（上午，带 arena/shadow/尾部守卫与全部开关）：干净编译（`-Wall -Wextra -Wformat=2`
零警告）；五种开关各一次（默认、`AMDFQ_VMM_PAD=0`、`AMDFQ_VMM=0`、
`AMDFQ_VMM=0 AMDFQ_VMM_SHADOW=1`、`AMDFQ_TAIL=1`）＋ `AMDFQ_VMM_LAYOUT=peralloc`，每次都用一段
"分配 2 MiB / 64 MiB、释放、再分配"的 torch 负载，日志分别给出 `# vmm: route on … pad one shared
granule`、`pad off (control run)`、`route off (disabled)` + `padded=`、`# vmm shadow:`、
`# vmm tail:`，服务模式下地址被空闲表复用、created == released；`vmm_probe.sh` 五个非 faulting
模式全过；`vmm_diff.sh` 复现出 §12.5 那张表。这些开关与模式此后已删除（§2、§8）。

**peralloc 一条路这一版**（改造后）：`cmake --build` 从干净目标重编，`-Wall -Wextra -Wformat=2`
零警告。冒烟负载一段（`/tmp/axl-vmm-smoke/smoke.py`，env `axl`、torch `2.13.0+rocm10.0.0`）：分配
20 MiB / 64 MiB / 2 MiB → 一次 `.item()` 的宿主读 → `torch.cuda.empty_cache()` 触发真正的
`hipFree` → 重新分配 2 MiB → 再释放。日志给出就绪行 `# vmm: pid … granularity=2097152 B (minimum
4096), layout=peralloc (one hipMemAddressReserve per allocation), pad one shared granule of 2097152 B`、
每笔 `vmm=served block=… pad=2097152`、每笔 free `vmm=served`，收尾 `blocks created 4 == released 4`、
`4 per-allocation reservations`、`pad mappings 4 (0 failed)`、**没有任何 fallback 行**，进程退出后设备
回到空载。真实训练没有用这一版跑过——§11 那 13 个路线一 run 全是改造前的构建。

口径提醒：这一次的读数取自一台开着桌面、并且同时可能有别的 GPU 任务在跑的机器（`../amdfq.md`
§11.7 的"每次跑必须在空闲设备上"是给定量测量定的规矩）。所以 `vmm_probe` 的**微秒数**只是指示性的；
显存量级（4000 → 2000 MiB + 一颗 granule）由 granule 取整的算术决定，不受影响。

## 11. 用真实 GPU 复现路线一的故障（2026-09-18）

### 11.1 目的与配置

拿真实训练把 §12.2 记录的那些死法再撞一遍，逐 run 归档。配置就是仓库自带的
[`../config.toml`](../config.toml) 原样：`lllj`（640 张图）、`seed = 1145141919`、
`network_dim = 48`、`train_batch_size = 2`、20 epoch（`total_steps = 6460`）——即 §12.2 那五种死法
所在的同一份配置与同一个 seed。环境 `conda activate axl`（torch `2.13.0+rocm10.0.0`），
AMD RX 9070 XT（gfx1201）。这一节的 run 全部用的是 2026-09-18 改造前的构建（带 arena/shadow/尾部
守卫与各开关），表里"配置"一列出现的开关现在都已删除——当前构建只能跑 peralloc + 共享 pad 一种，
见 §2。

每个 run 就是 `bash start_train.sh`，带不带 preload 决定它是路线一还是 canary：

```bash
LD_PRELOAD=$PWD/amdfq-vmm/cmake-build-debug/libamdfq-vmm.so bash start_train.sh
```

**canary 纪律**：无 preload 的对照 run 穿插在每批之间。这不是"顺便跑个基准"，而是
[`../conclusions/gfx1201-fault-response-wedge.md`](../conclusions/gfx1201-fault-response-wedge.md)
§4.3 要求的：没有 canary，"这次没崩"与"故障根本不再上报"分不开。本次这条纪律直接救了后半段数据
（见 §11.7）。

### 11.2 方法

两个 harness 都在 `/tmp/axl-vmm-repro/`（scratch，不进仓库）：第一轮 `run-suite.sh`，
第二轮 `run-suite2.sh`（多一个看门狗）。每个 run 采集：

- 退出码与信号、wall clock 时长；
- `state.json`（`status` / `training.step` / `training.loss`）与 tqdm 进度条最后那个 `n/6460`；
- 自该 run 起点以来的内核日志（`journalctl -k --since @<epoch>`），逐条抓
  `[gfxhub] page fault` / `GCVM_L2_PROTECTION_FAULT_STATUS` / `Faulty UTCL2 client ID` / 故障页地址；
- 进程自己的故障文本（`Warning: Queue error …`、`Memory Fault Error [host …, faulting addr …, kernel: …]`、
  ROCr 的 `Callback: Queue … aborting with error`、以及 NaN 时 state writer 抛的那行）；
- preload 对象的日志：路线就绪行、`hipMalloc`/`hipFree` 调用数、每个返回的指针；
- 第二轮额外每 5 s 采一次训练进程的 `/proc/<pid>`（`State`、`wchan`、自愿/非自愿上下文切换、RSS、
  线程数）＋ `gpu_busy_percent` ＋ 显存计数器。

两条硬规矩：每个 run 有 wall clock 上限（第一轮 240 s、第二轮 120 s），到点由 `timeout` 杀掉，
退出码 124/137，记为 **at-cap——到点被杀不等于活着**；run 之间必须等设备回到基线（无 `/dev/kfd`
持有者、显存计数器回到起点 ±128 MiB），等不到就停整轮。本次每个 run 之后都等到了。

故障地址还做了一次归属核对：把每个 run 的故障地址（内核日志那个页地址，以及进程自己报的
`faulting addr`）与同一 run 里 hook 返回过的**全部**指针比对（精确匹配、同 2 MiB 页、以及是否落在
某个块 + pad 之内）。

### 11.3 第一轮：13 个 run（路线一 10 个，canary 3 个）

`exit 124` = 到 240 s 上限被杀（卡死）。"内核日志"一列给出该 run 里 `[gfxhub] page fault` 的条数与签名。

| # | 配置 | 时长 / 退出 | step / loss | 进程自己报的 | 内核日志 | 故障 kernel（截断） |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | canary | 27 s / 134 | 10 / 0.0514 | memory-fault | 4× `TCP (0x8)` `0x00801031` @ `0x7f37eea00000` | Ailk…MT64x128x16 DTV1/1 |
| 02 | 默认 | 241 s / **124 卡死** | 0 / – | illegal-instruction + queue abort + hang analysis | 无 | Alik…MT64x64x32 DTV0/1 |
| 03 | 默认 | 15 s / 134 | 0 / – | memory-fault | 10× `SQC (inst)` `0x008012B1` @ `0x3f800000`/`0x403f800000`/`0x803f800000` | Alik…MT64x64x64 DTV0/1 |
| 04 | 默认 | 15 s / 134 | 0 / – | illegal-instruction | 无 | Ailk…MT64x64x64 DTV1/0 |
| 05 | 默认 | 16 s / 134 | 1 / 0.9978 | illegal-instruction | 无 | Alik…MT64x128x64 DTV0/1 |
| 06 | canary | 26 s / 134 | 10 / 0.0514 | memory-fault | 5× `TCP (0x8)` `0x00801031` @ `0x7fc80bc00000` | Ailk…MT64x128x16 DTV1/1 |
| 07 | `AMDFQ_VMM_PAD=0` | 14 s / 134 | 0 / – | illegal-instruction | 无 | Ailk…MT64x64x32 DTV0/1 |
| 08 | `AMDFQ_VMM_PAD=0` | 15 s / 134 | 0 / – | memory-fault | 10× `TCP (0x8)` `0x00801031` @ `0x88ccfb200000` | Alik…MT64x128x64 DTV0/1 |
| 09 | `AMDFQ_LAUNCH_LOG=1` | 240 s / **124 卡死** | 0 / – | memory-fault | 无 | – |
| 10 | `AMDFQ_LAUNCH_LOG=1` | 24 s / 1 | 1 / **NaN** | `ValueError: … not JSON compliant: nan` | 无 | – |
| 11 | `AMDFQ_VMM_LAYOUT=peralloc` `PAD=0` | 240 s / **124 卡死** | 1 / 0.9978 | memory-fault | 无 | – |
| 12 | `AMDFQ_VMM_LAYOUT=peralloc` `PAD=0` | 240 s / **124 卡死** | 0 / – | illegal-instruction | 无 | Ailk…MT128x128x32 DTV0/0 |
| 13 | canary | 240 s / **124 卡死** | 10 / 0.0514 | memory-fault | 无 | – |

三个 canary 都走到第 10 步、loss 完全相同（0.051359500735998154，avg 0.11108）——也就是说
**故障点之前的轨迹是确定性的**；差别只在第 10 步那一下：run-01/06 被页错误杀死，run-13 卡死。
每个 hook 日志里都有路线就绪行（`granularity=2097152 B`、`reserve=1024 GiB`、`pad handle size=…`，
或 `layout=peralloc`、或 `pad disabled by AMDFQ_VMM_PAD=0`），死前服务了 807–1926 次 `hipMalloc`。

### 11.4 第二轮：5 个 run（路线一 3 个，canary 2 个），按要求在下一步之前停止

上限 120 s。第二轮的意义是看门狗：它把"卡死"从推断变成证据。

| # | 配置 | 时长 / 退出 | step / loss | 进程自己报的 | 内核日志 | 故障 kernel（截断） |
| --- | --- | --- | --- | --- | --- | --- |
| r2-01 | canary | 123 s / **124 卡死** | 10 / 0.0514 | memory-fault | 无 | – |
| r2-02 | 默认 | 22 s / 1 | 2 / **NaN** | `ValueError: … not JSON compliant: nan` | 无 | – |
| r2-03 | 默认 | 122 s / **124 卡死** | 0 / – | memory-fault | 无 | – |
| r2-04 | 默认 | 17 s / 134 | 0 / – | illegal-instruction | 无 | torch `ComputeInternalGradientsCUDAKernel<float>` |
| r2-05 | canary | 122 s / **124 卡死** | 10 / 0.0514 | memory-fault | 无 | – |

卡死时进程是什么样子（看门狗采样，`busy` 是 `gpu_busy_percent`，`vis` 是驱动显存计数器）：

```
# r2-01（canary，GPU 100%、用户态自旋）
1789682329 100 14846607360 wchan=0 State:=R VmRSS:=4388064 Threads:=156 voluntary_ctxt_switches:=12662 nonvoluntary_ctxt_switches:=989
1789682334 100 14846607360 wchan=0 State:=R VmRSS:=4388064 Threads:=156 voluntary_ctxt_switches:=12662 nonvoluntary_ctxt_switches:=993

# r2-03（路线一，GPU 100%、阻塞在 futex）
1789682473 100 10131619840 wchan=__futex_wait State:=S VmRSS:=4131084 Threads:=156 voluntary_ctxt_switches:=12029 nonvoluntary_ctxt_switches:=337
1789682478 100 10131619840 wchan=__futex_wait State:=S VmRSS:=4131112 Threads:=156 voluntary_ctxt_switches:=12029 nonvoluntary_ctxt_switches:=337
```

两条形态不同，但都是"GPU 钉在 100%、显存不放、内核日志干净、进程永不退出"：r2-01/r2-05 是**用户态
自旋**（`State=R`、`wchan=0`、自愿切换冻结而非自愿切换一直在涨——进程一直被调度器抢占，却从不
让出）；r2-03 是**阻塞等待**（`State=S`、`wchan=__futex_wait`、两个计数器都冻结，而 GPU 仍在
满载运行）。这正是 wedge 文档 §1 描述的两种"瞎了"的样子。

### 11.5 结局统计

路线一 13 个 run（第一轮 10 + 第二轮 3）：**0 个存活**。跑得最远的是 r2-02——算到第 2 步的 loss
是 NaN 就结束了（run-11 走完第 1 步、loss 0.9978，然后卡死）；其余全部死在第 0 或第 1 步之内。

| 结局 | 个数 | 是哪些 |
| --- | --- | --- |
| 卡死（到上限被杀） | 5 | 02、09、11、12、r2-03 |
| 自行 abort：illegal instruction | 4 | 04、05、07、r2-04 |
| 自行 abort：指令取指页错误 | 1 | 03 |
| 自行 abort：数据页错误 | 1 | 08 |
| 自行退出：NaN 被 state writer 拒绝 | 2 | 10、r2-02 |

按进程报的错误归类：`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` 6 个（02、04、05、07、12、r2-04）、
`HSA_STATUS_ERROR_MEMORY_FAULT` 5 个（03、08、09、11、r2-03）、NaN 2 个（10、r2-02）。
内核日志里**只有 2 个 run** 留下 `[gfxhub] page fault`（03 的 `SQC (inst)`、08 的 `TCP (0x8)`）；
其余 11 个的内核日志是干净的——而其中 5 个其实卡死在同一类 GPU 事件上。

canary 5 个：3 个卡死（run-13、r2-01、r2-05），2 个按文档死在第 10 步（run-01、run-06）。

### 11.6 三条发现

**一、"卡死"是一个此前没有单列的结局类。** 运行时把队列错误报给了进程
（`Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT`；illegal instruction 还带 ROCr 的
`Callback: Queue … aborting with error`），进程随后**永不退出**，内核日志一个字都没有。这直接影响
§12.2 第 4 条：那里把"`dmesg` 干净"当作"没有 GPU 事件"的证据，而本次 5 个 run 的内核日志干净、
却确实发生了 GPU 事件。同一个错误在不同 run 里结局也不一样：illegal-instruction 6 次里 5 次杀死
进程、1 次卡死；memory-fault 5 次里 2 次杀死、3 次卡死。**fault → abort 这条链本身不稳定。**

**二、指令取指故障里的"地址"根本不是地址，是数值。** run-03 那 10 条页错误的客户端是
`SQC (inst)`（取指），状态 `0x008012B1`、`PERMISSION_FAULTS 0xb`，页地址是 `0x3f800000`、
`0x403f800000`、`0x803f800000`：每个的低 32 位都恰好是 `1.0f` 的 IEEE-754 编码（`0x3f800000`），
高 8 位是 64、128。也就是说一个 wave 的 PC 里装的是"`(1.0f, 64)`"这种打包后的**操作数**。
§12.2 #3 记的是"一个 wave 的 PC，不是它的数据"，这条数据把那个 PC 的内容也钉住了：它是算出来的值。

**三、出事的地址不在路线分配的内存里。** 拿每个 run 的故障地址与该 run 里 hook 返回过的全部指针
（精确匹配、同 2 MiB 页、是否落在某个块 + pad 内）比对：

| run | 进程报的故障地址 | 在 1 TiB 预约区内？ | 与服务过的块匹配数 |
| --- | --- | --- | --- |
| 03（默认） | `0x7e95f83c3000` | 是（距 arena 起点 11.6 GiB），但没有任何映射盖住它 | 0 / 1084 个（它下面最近的一个指针还在 719 MiB 之外） |
| 08（`PAD=0`） | `0x88cd88be3000` | 否（超出约 10.9 TB） | 0 / 1089 个 |

这两条独立于 §12.3 对 `_074714` 的那次观察（"故障地址不是我们的"）。另外 r2-04 的 illegal
instruction 出在 torch 自己的 `at::native::ComputeInternalGradientsCUDAKernel<float>` 里，与
§12.2 #1 的 torch 拷贝 kernel 同类、但换了 kernel——再次说明"不是某一个特定 kernel 的毛病"。

### 11.7 方法学提醒（这次学到的）

1. **机器会在一次会话的中途进入 wedge 状态。** 前半段 canary 老实死在第 10 步
   （`TCP (0x8)` / `0x00801031`，与文档基线逐项吻合）；后半段三次 canary 全部在第 10 步**卡住**，
   loss 曲线逐位相同。所以本次数据必须分"wedge 前后"读，**wedge 之后内核日志干净不能当证据**。
   按 wedge 文档，这个状态要靠重启才能清掉。（本次没有重启机器，所以第二轮后半段没有继续跑。）
2. **到点被杀不是"活着"。** 5 个 run 是卡死被 `timeout` 收走的；只看退出码会把它们误读成"没崩"。
3. **看门狗是必需的。** 没有 `/proc` 状态 + `gpu_busy_percent` 的采样，"卡死"只能靠"240 秒没动静"
   推断；有了它才能区分用户态自旋与 futex 阻塞，并且看到 GPU 与显存一直没被释放。
4. **run 之间必须等设备回基线。** 每次卡死都留着 10–14 GiB 显存不放，进程死了才释放；不等就开下
   一个 run，读数会互相污染。

### 11.8 这些数据确立了什么、没确立什么

- **确立**：这条路线的实现在这套栈上跑不动训练——13/13 失败，最好的情况死在第 1 步、
  最常见死在加载之后的第一步之内。失败种类与 §12.2 的清单一致（illegal instruction、指令取指
  fault、数据页错误、NaN），此外多出一个卡死类。把它们放在一起看：坏的值（NaN）、坏的值当 PC
  （`1.0f`）、坏的值当指针（故障地址不属于本进程分配的内存）——症状全都落在"进程自己的状态被打坏"，
  而不是"某个 kernel 越界读"（后者是 §10.1 的确定性 signature，每次都在 dmesg 里留下 `TCP` 页错误）。
- **没确立**：机制。§12.4 的三问（host access 描述符、规模、free 顺序）与 §12.5 的 Frida 参数捕获
  都还没做，本次只是把复现与分类做实了。
- **用不了**：任何"死在第几步"的定量统计。fault→abort 的链条不稳定（同一错误有时杀死、有时卡死），
  而卡死的 run 一律停在 cap 上，步数只反映我设的上限。

### 11.9 产物

| 内容 | 路径 |
| --- | --- |
| 第一轮 harness / 第二轮 harness / 汇总脚本 | `/tmp/axl-vmm-repro/{run-suite.sh,run-suite2.sh,summarize.py}` |
| 逐 run 产物（两轮同一个目录，第一轮 `run-NN-*`、第二轮 `r2-NN-*`） | `/tmp/axl-vmm-repro/<tag>.{log,hook.log,meta.txt}`，第二轮的卡死采样在 `<tag>.watch.log` |
| 汇总表 | `/tmp/axl-vmm-repro/{runs.tsv,round2.tsv}`；两轮的进度都在 `{suite.log,round2.log}` |
| 训练侧产物（每个 run 一个目录） | `/home/acite/LLM/axltrainer/outputs/lllj_20260918_*`、`/home/acite/LLM/axltrainer/logs/lllj_20260918_*`；`python clean.py --run <run_id>` 可清 |
| 状态机最后一份快照 | `${XDG_RUNTIME_DIR:-/tmp}/axltrainer/state.json` |

表里的 kernel 名是截断形式（完整名字长达 855 字符，在各自 `.log` 与 `.meta.txt` 里）。原始日志全部
在 `/tmp` 下，未版本化；本文与 §12 的数字都是从这里抄出来的。复跑一次整套约 20 分钟（第一轮
240 s 上限）/ 约 15 分钟（第二轮 120 s 上限），前提见 §11.7 的第 1、4 条。



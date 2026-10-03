# amdfq-vmm-rs 设计要求

这份文件是 `amdfq/amdfq-vmm-rs/` 的设计约束。怎么编、怎么 preload、日志怎么读见 `src/lib.rs` 的 crate
文档和 `test.sh`；这里只讲「结构必须长什么样」。

它存在的理由：已删除的 C 版 VMM 树（`amdfq-vmm/`）机制成立、设计不合格——12 个计数器 + 名字表 + 标志位散在
全局，一把非递归全局锁跨着真 runtime 调用，计数在解锁之后自增，最后还要一对重入闩兜底。peralloc
路线就是按下面这套形状搬进 Rust 的，后面再加东西也不会再长出同样的形状。

## D1 分配状态只有一个家

活着的分配只有 `REGISTRY`（`registry.rs`）：`HashMap<Address, HookData>`，**以块首地址为 key**。
每笔分配对应一条记录，记录自己带齐「free 时把这笔分配撤销回去所需的一切」：

| 字段 | 含义 |
| --- | --- |
| `HookData::address` | 块首地址：`hipMalloc` 交给调用方的那个指针，同时是 map 的 key |
| `HookData::size` | 调用方要的字节数，原样保存 |
| `HookData::origin` | 这笔块归谁：`Runtime`（runtime 自己分的，free 原样转回去）、`Extent(…)`（本 crate 单独 reserve + Create 出来的）或 `Pooled(…)`（从池里切出来的，free 把区间还给池，D12） |
| `Extent::block` | 从 `handle` 映射出来的字节数：请求向上取整到分配粒度 |
| `Extent::total` | 该地址上 reserve 的总字节数：`block` + 一颗 pad granule；一旦 Map 过就不再 `hipMemAddressFree`（D10） |
| `Extent::handle` | 块自己的 `hipMemGenericAllocationHandle_t`，free 时 `hipMemRelease` |
| `Extent::pad` | 块后面那颗共享 pad granule 的字节数；映射失败就是 `None` |
| `Extent::device` | 这块属于哪张卡：unmap / release 都作用于「当前设备」，free 先 `hipSetDevice` 切回去（D11） |
| `Pooled::pool` | 这笔区间来自哪个池（池 id，D12）；id 单调递增、永不复用，所以旧记录不可能指到一个新池上 |
| `Pooled::offset` / `Pooled::len` | 区间在池里的起点与长度（长度是请求向上取到 granule）。池自己的 handle、跨度、粒度、设备属于池，不存在记录里 |

布局本身（`\|---- block ----\|---- pad（共享 handle）----\|`）和它的来由写在 `peralloc.rs` 头部；池化块的
布局是池的一整段映射里的一个区间，写在 `pool.rs` 头部（D12）。没有并行的计数器、标志位、名字表——所以也就
不会出现 C 版那种「计数和实际块对不上」。

## D2 并发立场：torch 会并发调的 hip API，就当它并发安全

判据就这一条：如果 torch 层面会并发调用某个 hip API，那这个 API 就是并发安全的。因此

- hook **不给它加自己的锁**，也不序列化对 runtime 的调用——传进来多少并发，原样传给 runtime；
- 我们要保证的只是**自己新增的状态**（`REGISTRY`）没有竞态，不给原本安全的 API 引入新的等待。

推论：C 版那把跨 runtime 调用的全局锁、以及为了绕开它而存在的 `amdfq_vmm_begin/end` 重入闩，
按这条立场都是多余的（而且把并发变成了串行）。这一版没有它们。

## D3 锁只包住自己的数据结构

加锁只包住 crate 自己的数据结构（`REGISTRY` 的 map；池的 `POOLS` 表与空闲表，D5/D12），转发、teardown、
建池、拆池和日志都在锁外：任何 `real::*()` 调用期间都不持有锁。这条是可机械检查的——`src/hooks.rs` 里不
允许出现 `RwLock`/`Mutex`/`.lock(`/`.read(`/`.write(`，`test.sh` 每次都查（池的锁在 `pool.rs`，那里的
runtime 调用同样在锁外）。

## D4 日志走 `log` crate

hook 里只写 `log::info!` / `log::warn!`：不自己建文件、管道、序号表。sink 是 `logging.rs` 里的
stderr 实现，一条记录一次 `write(2)`，无锁。`AMDFQ_LOG_LEVEL`（`off`/`error`/`warn`/`info`/`debug`/
`trace`，默认 `info`）控级别，`warn` 只留异常——地址重复（`duplicate`）、释放一个 registry 里
没有的地址（`untracked`）、VRAM 保留把一笔申请拦成 OOM，以及路线自己没做成的那几步：授权没授上
（含只授到设备本身）、pad 没映射成、撤销的四步里有失败、切设备或 peer 查询没成（D11）、设备序号超出
设备表。宿主已经装了 logger 时不抢：`set_logger` 失败，我们的记录就顺着宿主的 sink 走。

`hipFree` 的行词随结局变：`released`（当场撤销）、`released with failures, see the warnings`。

## D5 全局清单是封闭的

新增任何全局必须先写进这张表，否则算设计回归。

| 全局 | 位置 | 用途 |
| --- | --- | --- |
| `REGISTRY: LazyLock<RwLock<HashMap<Address, HookData>>>` | `registry.rs` | 唯一的分配状态，也是 crate 第一把锁（D3） |
| `real.rs` 的符号缓存，每个符号一个 `LazyLock<Option<F>>` | `real.rs` | 真符号解析：两个分配门 + `hipMemGetInfo`（拦截给上层看的账面，剩余按 amdgpu `mem_info_vram_*` 算）+ 路线的 8 个 VMM 入口（`hipMemSetAccess`、`hipMemCreate`/`Map`/`Unmap`/`Release`、`hipMemAddressReserve`/`Free`、`hipMemGetAllocationGranularity`）+ `hipGetDevice`/`hipSetDevice`/`hipGetLastError`/`hipDeviceSynchronize` + 2 个可选的 peer 查询（`hipGetDeviceCount`、`hipDeviceCanAccessPeer`） |
| `ENTRIES: LazyLock<Option<Entries>>` | `peralloc.rs` | 路线是否就绪：这一组入口全都解析到了才开。VRAM 保留读 sysfs，不依赖 `hipMemGetInfo` |
| `DEVICES: [OnceLock<Option<Device>>; 32]` | `peralloc.rs` | 每张卡一份状态（D11）：分配粒度、该卡共享的 pad granule、能访问它的 peer 表；建失败也记住，序号超出上界直接转发 |
| `MAPPED_SPANS: LazyLock<RwLock<BTreeMap<usize, usize>>>` | `peralloc.rs` | 正在映射的 VA 跨度（首地址 → reserve 长度）。新的 reserve 若与其中任何一段相交，这一笔不 Map，转发给 runtime；复用模式下 free 把该段从表里移除，never-reuse 模式下 free 之后仍留在表里——于是它退化成「曾经 Map 过」的集合（D10） |
| `VRAM_RESERVE: LazyLock<usize>` | `peralloc.rs` | 驱动计数器 free 的地板（`AMDFQ_VRAM_RESERVE` 字节，缺省 `0`，即关闭；非 0 是内核修复前的旧 workaround，D10）。`mem_info_vram_total - mem_info_vram_used < reserve + block` 时 `hipMalloc` 返回 OOM，不 Create、不转发。拦截的 `hipMemGetInfo` 把 `free` 改成同一剩余减 reserve，`total` 改成 sysfs total |
| `VA_NEVER_REUSE: LazyLock<bool>` | `peralloc.rs` | `AMDFQ_VA_NEVER_REUSE`（`1`/`0`，缺省 `0`）：`1` 时 free 不把 VA 还给驱动，跨度留在 `MAPPED_SPANS` 里（D10） |
| `POOL_SIZE: LazyLock<usize>` | `pool.rs` | `AMDFQ_POOL_SIZE`（字节，缺省 `0` 即关闭），clamp 到 [16 MiB, 512 MiB]，非整数按关闭处理（D12） |
| `POOLS: LazyLock<RwLock<Table>>` | `pool.rs` | 活着的池：池 id → `Pool`（跨度、reserve 长度、handle、粒度、尾部 pad、空闲表、`live` 计数）。crate 的第二把锁，只包住池表与空闲表；建池、拆池的 runtime 调用都在锁外（D3/D12） |
| `SIZE_UNAVAILABLE_WARNED: AtomicBool` | `pool.rs` | granule 比池还大（该设备上池不可能成立）时只 warn 一次 |
| `SYSFS_MISSING_WARNED: AtomicBool` | `peralloc.rs` | amdgpu `mem_info_vram_*` 读不到时只打一次 warn，之后仍跳过保留检查 |
| `INSTALL: LazyLock<()>` | `logging.rs` | 装 logger 并设级别 |
| `LOGGER` | `logging.rs` | 无状态 sink |
| `EARLY_TOUCH: unsafe extern "C" fn(i32, *mut *mut c_char, *mut *mut c_char)` | `early.rs` | `.init_array` 项，glibc 在载入本对象时调用一次：一次 `hipGetDeviceCount`（D7 的例外）。是一个只读的代码指针，不是状态，也没有第二处可变全局 |

对照 C 版：12 个计数器 + `g_fb_names` + `g_disabled_reason` + 表 + 一把跨调用的非递归锁 +
一对重入闩。

## D6 一笔分配只走 insert 一次 / remove 一次

`hipFree` 取走即出 map，不留悬挂项。地址重复（`duplicate`）与查不到（`untracked`）都要在日志里
看得见——这两个信号本身就是竞态或生命周期错的证据，不该被静默吞掉。

## D7 没有 init / fini / 退出摘要

惰性初始化交给 `LazyLock`（每张卡那份状态用 `OnceLock`：它是按设备序号取的槽位）。进程被 HIP 从
内部 abort 时不存在「状态没来得及打出来」的问题；没有需要按顺序执行的生命周期，也就没有第二个可以
出错的地方。

唯一的例外是 `early.rs` 的 `.init_array` 构造函数。它为 ROCr 的一次空转而存在：本机的 ROCR 1.21
在任何 GPU 工作之后会让 `AsyncEventsLoop` 占满一个 CPU 核，直到进程结束；唯一能止住它的时机是
**载入期**——早于 torch 自己的 `libtorch_cpu.so` 被映射，也就早于应用能调用的任何东西（实测分离表在
`early.rs` 头部，`import torch` 期间连一次 HIP 调用都没有发生）。它不是生命周期：只跑一次、不记状态、
失败只留一条日志、没有 teardown，所以「按顺序执行的生命周期」这条理由没有被推翻。它同时是 crate 里
唯一一处 `dlopen`（D8）。

## D8 真符号解析只有一处

`real.rs` 是 crate 里唯一出现 `dlsym`、`RTLD_NEXT` 和符号名字符串的文件，每个符号一个
`LazyLock<Option<F>>`（比 C 版 `if (real == NULL)` 的首次调用多线程安全）。想加一个门：`hooks.rs`
里一个函数 + `real.rs` 里一行 `LazyLock`，别的地方不碰符号名。`hipMemGetInfo` 是第三个门：拦截给上层看的账面；保留检查读 `/sys/class/drm/cardN/device/mem_info_vram_{used,total}`。

同一文件里还有唯一一处**不走 `resolve`** 的符号操作：`early_touch`，用 `dlopen` 打开调用方点名的
那个路径、再在这个 handle 上取 `hipGetDeviceCount`（D7 的例外）。它不能走 `resolve`——`RTLD_NEXT`
找的是进程全局作用域里的名字，而载入期那里还没有运行时；路径由 `early.rs` 从正在跑的解释器推出来。

## D9 map 里的值必须 `Send + Sync`

由编译器强制。这是 C 版拿不到的保证：那时「这个结构会不会被两个线程同时改」只能靠人读代码。唯一需要
自己担保的是 `hipMemGenericAllocationHandle_t`（`hip.rs` 的 `Handle`）：它是指针，编译器不会认，所以
那里写明了 `unsafe impl Send/Sync` 并给出理由——它命名的是一个设备对象，不是线程私有的内存。

## D10 路线只动自己 reserve 出来的地址

`peralloc::release` 只对 `origin` 是 `Extent` 的记录生效；`Runtime` 的记录和 registry 里查不到的指针
一律原样交给 `hipFree`，我们不做 unmap / release。反过来，只要记录是 `Extent`，撤销就由我们做完，
绝不让 runtime 看见那个指针——撤销前先把当前设备切回记录里的那一张（`Extent::device`），因为 unmap /
release 都作用于当前设备（D11）。

free 之后那段 VA 归谁，由 `AMDFQ_VA_NEVER_REUSE` 决定，缺省是**还给驱动**：teardown 在
`hipMemUnmap` + `hipMemRelease` 之后调 `hipMemAddressFree`，并把这一段从 `MAPPED_SPANS` 移除，
地址可以再次被 reserve / Map。`MAPPED_SPANS` 于是是「正在映射的跨度」，`range_taken` 拒绝的只是与
**在用**映射相交的那一笔 reserve（相交的那次 reserve 也不释放，以免打穿已经占住的跨度）。

`AMDFQ_VA_NEVER_REUSE=1` 是内核修复前的旧行为，保留着是为了能一键退回：那时拆 mapping 不会作废
compute VM 的 TLB，同址复用会把陈旧翻译读出来，所以 free 只 unmap + release，
`hipMemAddressFree` 不调，跨度留在表里，永不复用；此时 `range_taken`
的判据退化成「曾经 Map 过」。2026-09 的内核已修（拆 mapping 作废 TLB），所以缺省不再需要这一条。

`hipMemCreate` 在 reserve 之前：物理 OOM 不占 VA。从未 Map 成功的 reserve（map 失败）在两种模式下
都仍然 `hipMemAddressFree`。hook 把 `MAPPED_SPANS` 累计的字节数、跨度数与当前模式写到
`<stem>.<pid>.json`（`AMDFQ_VA_STATUS`，缺省则与 `trainer/control.py` 同一 runtime 目录下的
`amdfq_vmm_va`），字段为 `used_bytes` / `spans` / `never_reuse`；Chromatrix Dashboard 用它画「已用 /
总 VA」，并按模式换文案（never-reuse 下这个数是累计占用过的，复用模式下是当前占用的）。

这里曾有一条例外：「fork 继承来的块什么都不做」（旧的 `Outcome::Inherited`），已删。理由不是它多余，
而是它照顾的场景在 HIP 的界外——`hipInit` 的 note（`hip/hip_runtime_api.h:2223`，本机副本在
`_rocm_sdk_core/include/hip/`）写明：子进程在 fork 之后还要继续跑 HIP 代码、又不立刻 `exec()` 时，
进程不应在 fork 之前初始化 HIP runtime，父子应各自在 fork 之后初始化，「跨 fork 继承 runtime state
可能带来未定义行为或初始化失败」。也就是说「子进程 free 掉父进程的块」不是要支持的行为，本文不为
fork 之后定义任何行为（HIP 的 `hipInit` note 就是这条界线；那条线的记录已封存）。

## D11 状态按设备分，授权含 peer

多卡进程里没有「进程级 device」这回事：

- device 在挂钩点取（`hipGetDevice`），不缓存、不记在全局；每张卡一份状态（`DEVICES`），第一次在
  哪张卡上分配就在哪张卡上建，建出来的粒度、pad granule、peer 表都只属于那张卡；
- 每笔分配带自己的 device（`Extent::device`）；撤销它的那次 `hipFree` 先 `hipSetDevice` 切回去再 teardown；
- 授权集合 = 本卡 + 能访问本卡的卡 + HOST。mapping 的权限只来自 `hipMemSetAccess`，
  `hipDeviceEnablePeerAccess` 不会回头给我们建的 mapping 加 grant，所以 peer 在这里一次问清
  （`hipDeviceCanAccessPeer` 的能力判断，顺带覆盖应用之后才打开的 peer）。缺 peer 的后果不是慢：
  对端 kernel、peer copy、RCCL 在页表上没权限——静默错数或 fault，而真 hipMalloc 的块不会这样。

设备序号超过 `DEVICES` 的长度（32）就转发：全局清单是封闭的（D5），表不能在运行时长大。peer 的判据
是「本卡能不能访问对端」这一个方向，取对称；这台机器只有一张卡，这一条没有实测。

一张卡第一次被用到时，该卡的 `OnceLock` 会让并发的 `hipMalloc` 等这一次建立（建好之后是纯读）——
和旧版首次 init 同一性质，多卡只是把这一次挪到每张卡的头一笔分配上。

## D12 小请求先从池里切，池空了就还给驱动

`AMDFQ_POOL_SIZE`（字节，变量缺省 = 关闭，clamp 到 [16 MiB, 512 MiB]，非整数按关闭处理）打开池路线
（`pool.rs`）。「变量缺省即关闭」是刻意的：hook 被手工 preload 时不该凭空拿到池。仓库 shipped 的
`config.toml` 把 `amdfq_pool_mib` 给成 `64`，于是正常训练启动（`start_train.sh`）是带着 64 MiB 的池跑的；
那个值对应的实测代价见 D12 末尾。它存在的理由：一笔 `hipMalloc` 在直连路线上要付一次 `hipMemCreate` +
一次 `hipMemAddressReserve` + 两次 `hipMemMap` + 两次 `hipMemSetAccess`，而一个训练步会打上上百次
`hipMalloc`、其中大头是同一批小尺寸。池把这些驱动侧调用从「每笔请求」降到「每个池」。

规则（按顺序）：

1. `size <= PoolSize / 2` 的请求**优先**从池里切；更大的请求照 D10 的直连路线走（一笔请求一个 Create）。
   阈值取一半，是为了让一个池里至少放得下两笔最大的可切分请求——池不会为一笔请求而存在。
2. 池大小取该设备 granule 的**偶数倍**，于是 `PoolSize / 2` 是整数个 granule，阈值内的任何请求都必然
   放得进一个空池（`pool.rs` 的单元测试覆盖这一条）。
3. **可以同时存在多个池**：本设备已有的池里没有装得下的空闲区间时新建一个；建池在锁外，所以并发线程可能
   各建一个，这是允许的。
4. 池的整段是**一次** `hipMemMap`，尾部一颗共享 pad granule：池内任意块的后面直到池尾都是映射内存，块与块
   之间不需要 per-block pad，尾部那颗保护的是「落在池尾的那一笔」。块的越界**读**因此仍然落在映射内存里
   （与直连落在共享 pad 上同效）；越界**写**则可能落进邻块，而直连落进的是没人读的 pad——这是池化相对直连
   唯一新增的语义风险，也是不建议开大池、不默认开启的技术理由之一。
5. **建池全有或全无**：Create → Reserve(PoolSize + granule) → Map 整段 → SetAccess 整段 → 尾部 pad 的
   Map/SetAccess，任何一步失败就撤销已做的步骤并**回落到直连路线**。于是「池建不起来」的后果就是池关闭时
   的行为，不会更糟。尾部 pad 也纳入构造而不像直连那样可省：省掉它，池尾那一笔会在**每一笔**落在池尾的
   请求上暴露，而不是只影响一笔。
6. **池空即还**：`live` 归零时把池的 handle、整段映射与 VA 一起还给驱动（VA 归不归还看
   `AMDFQ_VA_NEVER_REUSE`，与 D10 同规则）。归还走的是直连拆卸同一段代码——把粒度换成 `PoolSize` 的
   `Extent` 交给 `peralloc::teardown`，设备切回、`hipDeviceSynchronize`、VA 记账都不另开一套。
7. 显存保留（`AMDFQ_VRAM_RESERVE`）只对**新池真正新提交的 `PoolSize`** 生效；池内切分不新提交物理内存。
   保留水位拒绝建池时回落到直连路线，让这笔小请求按它自己的大小重新判定——与池关闭时一致。

代价写在明处：池是别人拿不到的**已提交**显存（OOM 与碎片化的来源），能省的每步驱动调用次数有上界（边际
收益递减）。这就是 Chromatrix 与 `doc/configuration.md` 上那句「池并非越大越好」的来由，也是 hook 一侧在变量
未设时默认关闭的来由。同一把尺子下，`(PoolSize/2, PoolSize]` 区间的请求仍会做一次小于 `PoolSize` 的
Create：本方案的不变量是「被池化的 MemCreate 恒为 `PoolSize`」，不是「任何 MemCreate 都不得小于
`PoolSize`」。

这条路线**没有**被实测成更快。`test/bench_alloc_pool.py` 在同一份 `config.toml` 负载上（128 张子集、每
run 201 步、一档一次运行）量到：池关闭的中位步时 1.070–1.074 s，16/32/64/128 MiB 分别是 −0.1% / ±0.0% /
−0.6% / −1.0%，256 MiB 两次尝试都在第 125 步被 OOM 终止，而池关闭的 9 次运行里只中止过 1 次。每次运行的
损失轨迹与基线逐位相同，所以池本身是对的、只是不快：一个 run 内 13448 次 `hipMalloc` 摊在 215 s 的训练
窗口上，即使每次分配的开销归零也只有 0.1–0.2% 的上限，而建池/拆池有自己的开销。仓库 shipped 的
`amdfq_pool_mib = 64` 因此是维护者的选择，不是性能结论。

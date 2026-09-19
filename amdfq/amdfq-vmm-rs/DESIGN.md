# amdfq-vmm-rs 设计要求

这份文件是 `amdfq/amdfq-vmm-rs/` 的设计约束。怎么编、怎么 preload、日志怎么读见 `src/lib.rs` 的 crate
文档和 `test.sh`；这里只讲「结构必须长什么样」。

它存在的理由：已删除的 C 版 VMM 树（`amdfq-vmm/`）机制成立、设计不合格——12 个计数器 + 名字表 + 标志位散在
全局，一把非递归全局锁跨着真 runtime 调用，计数在解锁之后自增，最后还要一对重入闩兜底。peralloc
路线（`../doc/amdfq.md` §12）就是按下面这套形状搬进 Rust 的，后面再加东西也不会再长出同样的形状。

## D1 分配状态只有一个家

活着的分配只有 `REGISTRY`（`registry.rs`）：`HashMap<Address, HookData>`，**以块首地址为 key**。
每笔分配对应一条记录，记录自己带齐「free 时把这笔分配撤销回去所需的一切」：

| 字段 | 含义 |
| --- | --- |
| `HookData::address` | 块首地址：`hipMalloc` 交给调用方的那个指针，同时是 map 的 key |
| `HookData::size` | 调用方要的字节数，原样保存 |
| `HookData::origin` | 这笔块归谁：`Runtime`（runtime 自己分的，free 原样转回去）或 `Extent(…)`（本 crate reserve 出来的） |
| `Extent::block` | 从 `handle` 映射出来的字节数：请求向上取整到分配粒度 |
| `Extent::total` | 该地址上 reserve 的总字节数：`block` + 一颗 pad granule；一旦 Map 过就不再 `hipMemAddressFree`（D10） |
| `Extent::handle` | 块自己的 `hipMemGenericAllocationHandle_t`，free 时 `hipMemRelease` |
| `Extent::pad` | 块后面那颗共享 pad granule 的字节数；映射失败就是 `None` |
| `Extent::device` | 这块属于哪张卡：unmap / release 都作用于「当前设备」，free 先 `hipSetDevice` 切回去（D11） |

布局本身（`\|---- block ----\|---- pad（共享 handle）----\|`）和它的来由写在 `peralloc.rs` 头部。
没有并行的计数器、标志位、名字表——所以也就不会出现 C 版那种「计数和实际块对不上」。

## D2 并发立场：torch 会并发调的 hip API，就当它并发安全

判据就这一条：如果 torch 层面会并发调用某个 hip API，那这个 API 就是并发安全的。因此

- hook **不给它加自己的锁**，也不序列化对 runtime 的调用——传进来多少并发，原样传给 runtime；
- 我们要保证的只是**自己新增的状态**（`REGISTRY`）没有竞态，不给原本安全的 API 引入新的等待。

推论：C 版那把跨 runtime 调用的全局锁、以及为了绕开它而存在的 `amdfq_vmm_begin/end` 重入闩，
按这条立场都是多余的（而且把并发变成了串行）。这一版没有它们。

## D3 锁只包住自己的数据结构

加锁只包住 crate 自己的数据结构（`REGISTRY` 的 map），转发、teardown 和日志都在
锁外：任何 `real::*()` 调用期间都不持有锁。这条是可机械检查的——`src/hooks.rs` 里不允许出现
`RwLock`/`Mutex`/`.lock(`/`.read(`/`.write(`，`test.sh` 每次都查。

## D4 日志走 `log` crate

hook 里只写 `log::info!` / `log::warn!`：不自己建文件、管道、序号表。sink 是 `logging.rs` 里的
stderr 实现，一条记录一次 `write(2)`，无锁。`AMDFQ_LOG_LEVEL`（`off`/`error`/`warn`/`info`/`debug`/
`trace`，默认 `info`）控级别，`warn` 只留异常——地址重复（`duplicate`）、释放一个 registry 里
没有的地址（`untracked`），以及路线自己没做成的那几步：授权没授上
（含只授到设备本身）、pad 没映射成、撤销的四步里有失败、切设备或 peer 查询没成（D11）、设备序号超出
设备表。宿主已经装了 logger 时不抢：`set_logger` 失败，我们的记录就顺着宿主的 sink 走。

`hipFree` 的行词随结局变：`released`（当场撤销）、`released with failures, see the warnings`。

## D5 全局清单是封闭的

新增任何全局必须先写进这张表，否则算设计回归。

| 全局 | 位置 | 用途 |
| --- | --- | --- |
| `REGISTRY: LazyLock<RwLock<HashMap<Address, HookData>>>` | `registry.rs` | 唯一的分配状态，也是 crate 的锁（D3） |
| `real.rs` 的符号缓存，每个符号一个 `LazyLock<Option<F>>` | `real.rs` | 真符号解析：两个分配门 + 路线的 8 个 VMM 入口（`hipMemSetAccess`、`hipMemCreate`/`Map`/`Unmap`/`Release`、`hipMemAddressReserve`/`Free`、`hipMemGetAllocationGranularity`）+ `hipGetDevice`/`hipSetDevice`/`hipGetLastError`/`hipDeviceSynchronize` + 2 个可选的 peer 查询（`hipGetDeviceCount`、`hipDeviceCanAccessPeer`） |
| `ENTRIES: LazyLock<Option<Entries>>` | `peralloc.rs` | 路线是否就绪：这一组入口全都解析到了才开 |
| `DEVICES: [OnceLock<Option<Device>>; 32]` | `peralloc.rs` | 每张卡一份状态（D11）：分配粒度、该卡共享的 pad granule、能访问它的 peer 表；建失败也记住，序号超出上界直接转发 |
| `EVER_MAPPED: LazyLock<RwLock<BTreeMap<usize, usize>>>` | `peralloc.rs` | 曾经 Map 过的 VA 跨度（首地址 → reserve 长度）。free 之后仍在表里；新的 reserve 若与其中任何一段相交，这一笔不 Map，转发给 runtime（D10） |
| `INSTALL: LazyLock<()>` | `logging.rs` | 装 logger 并设级别 |
| `LOGGER` | `logging.rs` | 无状态 sink |

对照 C 版：12 个计数器 + `g_fb_names` + `g_disabled_reason` + 表 + 一把跨调用的非递归锁 +
一对重入闩。

## D6 一笔分配只走 insert 一次 / remove 一次

`hipFree` 取走即出 map，不留悬挂项。地址重复（`duplicate`）与查不到（`untracked`）都要在日志里
看得见——这两个信号本身就是竞态或生命周期错的证据，不该被静默吞掉。

## D7 没有 init / fini / 退出摘要

惰性初始化交给 `LazyLock`（每张卡那份状态用 `OnceLock`：它是按设备序号取的槽位）。进程被 HIP 从
内部 abort 时不存在「状态没来得及打出来」的问题；没有需要按顺序执行的生命周期，也就没有第二个可以
出错的地方。

## D8 真符号解析只有一处

`real.rs` 是 crate 里唯一出现 `dlsym`、`RTLD_NEXT` 和符号名字符串的文件，每个符号一个
`LazyLock<Option<F>>`（比 C 版 `if (real == NULL)` 的首次调用多线程安全）。想加一个门：`hooks.rs`
里一个函数 + `real.rs` 里一行 `LazyLock`，别的地方不碰符号名。

## D9 map 里的值必须 `Send + Sync`

由编译器强制。这是 C 版拿不到的保证：那时「这个结构会不会被两个线程同时改」只能靠人读代码。唯一需要
自己担保的是 `hipMemGenericAllocationHandle_t`（`hip.rs` 的 `Handle`）：它是指针，编译器不会认，所以
那里写明了 `unsafe impl Send/Sync` 并给出理由——它命名的是一个设备对象，不是线程私有的内存。

## D10 路线只动自己 reserve 出来的地址

`peralloc::release` 只对 `origin` 是 `Extent` 的记录生效；`Runtime` 的记录和 registry 里查不到的指针
一律原样交给 `hipFree`，我们不做 unmap / release。反过来，只要记录是 `Extent`，撤销就由我们做完，
绝不让 runtime 看见那个指针——撤销前先把当前设备切回记录里的那一张（`Extent::device`），因为 unmap /
release 都作用于当前设备（D11）。

曾经 Map 过的 VA 在进程生命周期内不复用：free 只 `hipMemUnmap` 和 `hipMemRelease`，不
`hipMemAddressFree`，那段地址一直占着。`EVER_MAPPED` 记下每一段已经 Map 过的 `[va, va+total)`；
之后 `hipMemAddressReserve` 若交回与其中任何一段相交的范围，这一笔不 Map（相交的那次 reserve
也不释放，以免打穿已经占住的跨度）。从未 Map 成功的 reserve（create / map 失败）仍
`hipMemAddressFree`。

这里曾有一条例外：「fork 继承来的块什么都不做」（旧的 `Outcome::Inherited`），已删。理由不是它多余，
而是它照顾的场景在 HIP 的界外——`hipInit` 的 note（`hip/hip_runtime_api.h:2223`，本机副本在
`_rocm_sdk_core/include/hip/`）写明：子进程在 fork 之后还要继续跑 HIP 代码、又不立刻 `exec()` 时，
进程不应在 fork 之前初始化 HIP runtime，父子应各自在 fork 之后初始化，「跨 fork 继承 runtime state
可能带来未定义行为或初始化失败」。也就是说「子进程 free 掉父进程的块」不是要支持的行为，本文不为
fork 之后定义任何行为（同一段也记在 `../../conclusions/hip-runtime-calls.md` §3）。

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

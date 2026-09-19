# amdfq-tail-rs 设计要求

这份文件是 `amdfq/amdfq-tail-rs/` 的设计约束。怎么编、怎么 preload、日志怎么读见 `src/lib.rs` 的 crate
文档和 `test.sh`；这里只讲「结构必须长什么样」。

它存在的理由：C 版尾守卫（`amdfq/amdfq-tail/`）机制成立、设计不合格——两张开地址表、十几个计数器、一把
跨着真 runtime 调用的全局锁、constructor/destructor 退出摘要、自己的日志文件。路线本身
（`../doc/amdfq.md` §13）按下面这套形状搬进 Rust，后面再加东西也不会再长出同样的形状。

约束编号与 `amdfq/amdfq-vmm-rs/DESIGN.md` 对齐：同一条在两个 crate 里意思相同。D12 是 peralloc 的延时
释放池，本路线不拥有调用方的块，不搬。

## D1 分配状态只有一个家

全局只有 `REGISTRY`（`registry.rs`）：`by_start` 以块首地址为 key，`by_end` 是同一份记录的终点索引
（`hipFree` 之后要给「结束在被释放地址上」的前任补守卫）。每笔分配对应一条记录，记录自己带齐
「free 时把这笔分配的守卫撤销回去所需的一切」：

| 字段 | 含义 |
| --- | --- |
| `HookData::address` | 块首地址：`hipMalloc` 交给调用方的那个指针，同时是 `by_start` 的 key |
| `HookData::size` | 调用方拿到的字节数：原样，或 pad 兜底时的 `size + 16` |
| `HookData::origin` | 这笔块归谁：`Runtime`（runtime 自己分的，没有我们的页）或 `Guarded`（块后面有一页本 crate reserve 出来的） |
| `Runtime` / `Guarded` 共有的 `block` | 运行时 extent 向上取整到分配粒度；`by_end` 的 key 是 `address + block` |
| `Guarded::page` | 守卫页的地址（块尾上取整到的第一个 granule 页，尾已对齐时等于 end）；free 时 `hipMemUnmap` + `hipMemAddressFree` |
| `Guarded::granule` | 那一页的字节数 |
| `…::device` | 这块属于哪张卡：撤销守卫的 VMM 调用作用于当前设备，free 先 `hipSetDevice` 切回去（D11） |

没有并行的计数器、标志位、名字表。`by_end` 不是第二份分配状态，只是同一条记录的索引。

## D2 并发立场：torch 会并发调的 hip API，就当它并发安全

判据就这一条：如果 torch 层面会并发调用某个 hip API，那这个 API 就是并发安全的。因此

- hook **不给它加自己的锁**，也不序列化对 runtime 的调用——传进来多少并发，原样传给 runtime；
- 我们要保证的只是**自己新增的状态**（`REGISTRY`）没有竞态，不给原本安全的 API 引入新的等待。

推论：C 版那把跨 `hipMem*`、甚至跨 pad 兜底的 `hipFree`/`hipMalloc` 的全局锁，按这条立场是多余的。
这一版没有它。

## D3 锁只包住自己的数据结构

加锁只包住 crate 自己的数据结构（`REGISTRY` 的两张 map），转发、reserve/map/unmap 和日志都在锁外：
任何 `real::*()` 调用期间都不持有锁。这条是可机械检查的——`src/hooks.rs` 里不允许出现
`RwLock`/`Mutex`/`.lock(`/`.read(`/`.write(`，`test.sh` 每次都查。

## D4 日志走 `log` crate

hook 里只写 `log::info!` / `log::warn!`：不自己建文件、管道、序号表。sink 是 `logging.rs` 里的
stderr 实现，一条记录一次 `write(2)`，无锁。`AMDFQ_LOG_LEVEL`（`off`/`error`/`warn`/`info`/`debug`/
`trace`，默认 `info`）控级别，`warn` 只留异常——地址重复（`duplicate`）、释放一个 registry 里
没有的地址（`untracked`）、以及路线自己没做成的那几步：授权没授上、hint 没落到要的页、撤销失败、
切设备或 peer 查询没成（D11）、设备序号超出设备表、pad 兜底把原块还回去之后新块没拿到。

`hipMalloc` 的行词随结局变：`guarded`（映射了一页）、`backed`（守卫页起点已被别的块占着）、`unaligned`（块尾无法上取整到一个 granule 页）、
`padded`（还回去再拿一块 +16）、`unguarded`（页没拿到、原块留下）、`off`（`AMDFQ_TAIL=0` 或入口
没齐）。`hipFree`：`released`（拆了我们的页）或没有页时只带 `size`。给前任补上守卫另起
`re-guarded pred=…`。

宿主已经装了 logger 时不抢：`set_logger` 失败，我们的记录就顺着宿主的 sink 走。

## D5 全局清单是封闭的

新增任何全局必须先写进这张表，否则算设计回归。

| 全局 | 位置 | 用途 |
| --- | --- | --- |
| `REGISTRY: LazyLock<RwLock<Registry>>` | `registry.rs` | 唯一的分配状态（`by_start` + `by_end`），也是 crate 的锁（D3） |
| `real.rs` 的符号缓存，每个符号一个 `LazyLock<Option<F>>` | `real.rs` | 真符号解析：三条分配门（`hipMalloc`/`hipFree`/`hipHostMalloc`）+ 路线的 VMM 入口（`hipMemSetAccess`、`hipMemCreate`/`Map`/`Unmap`、`hipMemAddressReserve`/`Free`、`hipMemGetAllocationGranularity`、`hipMemGetAddressRange`）+ `hipGetDevice`/`hipSetDevice`/`hipGetLastError` + 2 个可选的 peer 查询 |
| `ENTRIES: LazyLock<Option<Entries>>` | `tail.rs` | 路线是否就绪：这一组入口全都解析到了才开 |
| `DEVICES: [OnceLock<Option<Device>>; 32]` | `tail.rs` | 每张卡一份状态（D11）：分配粒度、该卡共享的守卫 handle、能访问它的 peer 表；建失败也记住，序号超出上界直接不守卫 |
| `ENABLED: LazyLock<bool>` | `tail.rs` | `AMDFQ_TAIL` 读一次：`"0"` 关掉守卫，分配仍转发、仍登记 |
| `INSTALL: LazyLock<()>` | `logging.rs` | 装 logger 并设级别 |
| `LOGGER` | `logging.rs` | 无状态 sink |

对照 C 版：`g_table` + `g_ends` + 一把跨调用的锁 + `g_state`/`g_off_reason`/`g_owner` + 十几个计数器 +
constructor/destructor。

## D6 一笔分配只走 insert 一次 / remove 一次

`hipFree` 取走即出 map，不留悬挂项。地址重复（`duplicate`）与查不到（`untracked`）都要在日志里
看得见——这两个信号本身就是竞态或生命周期错的证据，不该被静默吞掉。

## D7 没有 init / fini / 退出摘要

惰性初始化交给 `LazyLock`（每张卡那份状态用 `OnceLock`：它是按设备序号取的槽位）。进程被 HIP 从
内部 abort 时不存在「状态没来得及打出来」的问题；没有需要按顺序执行的生命周期，也就没有第二个可以
出错的地方。共享守卫 handle 创建一次、从不 `hipMemRelease`。

## D8 真符号解析只有一处

`real.rs` 是 crate 里唯一出现 `dlsym`、`RTLD_NEXT` 和符号名字符串的文件，每个符号一个
`LazyLock<Option<F>>`。想加一个门：`hooks.rs` 里一个函数 + `real.rs` 里一行 `LazyLock`，别的地方不碰
符号名。

## D9 map 里的值必须 `Send + Sync`

由编译器强制。唯一需要自己担保的是 `hipMemGenericAllocationHandle_t`（`hip.rs` 的 `Handle`）：它是
指针，编译器不会认，所以那里写明了 `unsafe impl Send/Sync` 并给出理由——它命名的是一个设备对象，不是
线程私有的内存。

## D10 路线只动自己 reserve 出来的地址

`tail::unprotect` 只对 `origin` 是 `Guarded` 的记录生效；`Runtime` 的记录和 registry 里查不到的指针
一律原样交给 `hipFree`，我们不做 unmap / address_free。反过来，只要记录是 `Guarded`，那一页的撤销
就由我们做完，绝不让 runtime 去 free 那一页——撤销前先把当前设备切回记录里的那一张（D11）。

调用方的块始终是 runtime 的：本路线从不 `hipMemAddressReserve` 调用方的 extent，也从不把调用方的
指针从 `hipFree` 里藏起来。pad 兜底是唯一一次我们主动还块再拿一块，指针已经是 runtime
给的。

本文不为 fork 之后定义任何行为（与 vmm-rs D10 同一段 HIP `hipInit` note）。C 版的 `g_owner` pid
检查不搬。

## D11 状态按设备分，授权含 peer

多卡进程里没有「进程级 device」这回事：

- device 在挂钩点取（`hipGetDevice`），不缓存、不记在全局；每张卡一份状态（`DEVICES`），第一次在
  哪张卡上分配就在哪张卡上建，建出来的粒度、共享守卫 handle、peer 表都只属于那张卡；
- 每笔分配带自己的 device；撤销它的守卫先 `hipSetDevice` 切回去再 unmap；
- 授权集合 = 本卡 + 能访问本卡的卡 + HOST。mapping 的权限只来自 `hipMemSetAccess`，
  `hipDeviceEnablePeerAccess` 不会回头给我们建的 mapping 加 grant。缺 peer 的后果不是慢：对端
  kernel 在页表上没权限——静默错数或 fault，而真 hipMalloc 的块不会这样。

设备序号超过 `DEVICES` 的长度（32）就不守卫：全局清单是封闭的（D5），表不能在运行时长大。peer 的
判据是「本卡能不能访问对端」这一个方向，取对称；这台机器只有一张卡，这一条没有实测。

C 版是进程级一份 `g_shared` / `g_prop`，授权只有本卡 + HOST。这一版按 D11 拆到每张卡，并补上 peer。

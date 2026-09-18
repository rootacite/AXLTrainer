# amdfq-vmm-rs 设计要求

这份文件是 `amdfq-vmm-rs/` 的设计约束。怎么编、怎么 preload、日志怎么读见 `src/lib.rs` 的 crate
文档和 `test.sh`；这里只讲「结构必须长什么样」。

它存在的理由：`../amdfq-vmm/` 那份 C 实现机制成立、设计不合格——12 个计数器 + 名字表 + 标志位散在
全局，一把非递归全局锁跨着真 runtime 调用，计数在解锁之后自增，最后还要一对重入闩兜底。peralloc
路线（`../amdfq.md` §12）就是按下面这套形状搬进 Rust 的，后面再加东西也不会再长出同样的形状。

## D1 分配状态只有一个家

全局只有 `REGISTRY`（`registry.rs`）：`HashMap<Address, HookData>`，**以块首地址为 key**。
每笔分配对应一条记录，记录自己带齐「free 时把这笔分配撤销回去所需的一切」：

| 字段 | 含义 |
| --- | --- |
| `HookData::address` | 块首地址：`hipMalloc` 交给调用方的那个指针，同时是 map 的 key |
| `HookData::size` | 调用方要的字节数，原样保存 |
| `HookData::origin` | 这笔块归谁：`Runtime`（runtime 自己分的，free 原样转回去）或 `Extent(…)`（本 crate reserve 出来的） |
| `Extent::block` | 从 `handle` 映射出来的字节数：请求向上取整到分配粒度 |
| `Extent::total` | 该地址上 reserve 的总字节数：`block` + 一颗 pad granule，`hipMemAddressFree` 用的就是它 |
| `Extent::handle` | 块自己的 `hipMemGenericAllocationHandle_t`，free 时 `hipMemRelease` |
| `Extent::pad` | 块后面那颗共享 pad granule 的字节数；映射失败就是 `None` |

布局本身（`\|---- block ----\|---- pad（共享 handle）----\|`）和它的来由写在 `peralloc.rs` 头部。
没有并行的计数器、标志位、名字表——所以也就不会出现 C 版那种「计数和实际块对不上」。

## D2 并发立场：torch 会并发调的 hip API，就当它并发安全

判据就这一条：如果 torch 层面会并发调用某个 hip API，那这个 API 就是并发安全的。因此

- hook **不给它加自己的锁**，也不序列化对 runtime 的调用——传进来多少并发，原样传给 runtime；
- 我们要保证的只是**自己新增的状态**（`REGISTRY`）没有竞态，不给原本安全的 API 引入新的等待。

推论：C 版那把跨 runtime 调用的全局锁、以及为了绕开它而存在的 `amdfq_vmm_begin/end` 重入闩，
按这条立场都是多余的（而且把并发变成了串行）。这一版没有它们。

## D3 锁只出现在 registry 内部

加锁只包住 HashMap 操作，转发和日志都在锁外：任何 `real::*()` 调用期间都不持有锁。
这条是可机械检查的——`src/hooks.rs` 里不允许出现 `RwLock`/`Mutex`/`.lock(`/`.read(`/`.write(`，
`test.sh` 每次都查。

## D4 日志走 `log` crate

hook 里只写 `log::info!` / `log::warn!`：不自己建文件、管道、序号表。sink 是 `logging.rs` 里的
stderr 实现，一条记录一次 `write(2)`，无锁。`AMDFQ_LOG_LEVEL`（`off`/`error`/`warn`/`info`/`debug`/
`trace`，默认 `info`）控级别，`warn` 只留两种异常——地址重复（`duplicate`）和释放一个 registry 里
没有的地址（`untracked`）。宿主已经装了 logger 时不抢：`set_logger` 失败，我们的记录就顺着宿主的
sink 走。

## D5 全局清单是封闭的

新增任何全局必须先写进这张表，否则算设计回归。

| 全局 | 位置 | 用途 |
| --- | --- | --- |
| `REGISTRY: LazyLock<RwLock<HashMap<Address, HookData>>>` | `registry.rs` | 唯一的分配状态，也是全 crate 唯一一把锁 |
| `real.rs` 的符号缓存，每个符号一个 `LazyLock<Option<F>>` | `real.rs` | 真符号解析：两个分配门 + 路线用的 10 个 VMM 入口 + 释放路径上的 `hipDeviceSynchronize` |
| `SETUP: LazyLock<Option<Setup>>` | `peralloc.rs` | 路线是否就绪：device id、分配粒度、建这份状态的 pid |
| `PAD: LazyLock<Option<Handle>>` | `peralloc.rs` | 所有块共用那一颗 pad granule 的物理对象，建一次、永不释放 |
| `INSTALL: LazyLock<()>` | `logging.rs` | 装 logger 并设级别 |
| `LOGGER` | `logging.rs` | 无状态 sink |

对照 C 版：12 个计数器 + `g_fb_names` + `g_disabled_reason` + 表 + 一把跨调用的非递归锁 +
一对重入闩。

## D6 一笔分配只走 insert 一次 / remove 一次

`hipFree` 取走即出 map，不留悬挂项。地址重复（`duplicate`）与查不到（`untracked`）都要在日志里
看得见——这两个信号本身就是竞态或生命周期错的证据，不该被静默吞掉。

## D7 没有 init / fini / 退出摘要

惰性初始化交给 `LazyLock`。进程被 HIP 从内部 abort 时不存在「状态没来得及打出来」的问题；没有
需要按顺序执行的生命周期，也就没有第二个可以出错的地方。

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
一律原样交给 `hipFree`，我们不做 unmap / release / address_free。反过来，只要记录是 `Extent`，撤销
就由我们做完，绝不让 runtime 看见那个指针——唯一的例外是 fork 继承来的块：那是父进程还在用的 handle，
我们什么都不做（`peralloc::Outcome::Inherited`），也不把它转给 runtime。

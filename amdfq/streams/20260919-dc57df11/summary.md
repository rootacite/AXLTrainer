# `20260919-dc57df11` — pad 0 的分配/释放流（C hook，采到挂）

## 1. 这次采的是什么

| 项 | 值 |
| --- | --- |
| hook | `amdfq/amdfq-tail/cmake-build-release/libamdfq.so`，sha256 `dc57df118398c4f6d9b0e04e11282d4626d5af3390af26c71bfe87a85511caf6` |
| 守卫 | **关**（`AMDFQ_TAIL=0`），逐笔 0 pad |
| 命令 | `AMDFQ_TAIL=0 AMDFQ_LOG=<此目录>/frozen.log LD_PRELOAD=<hook> bash start_train.sh`（cwd = 仓库根） |
| 采集脚本 | `collect.sh`（编排 + 基线 + 崩溃记录）、`sample-host.sh`（4 Hz vis/gtt） |
| 训练配置 | `config.toml` 现行那份：`train_data_dir = /home/acite/LLM/Character/LLLJ/`、`output_name = lllj`、`seed = 1145141919` |
| 解释器 | `/home/acite/miniconda3/envs/axl/bin/python` (3.14.7) |
| run_id | `lllj_20260919_183641` |
| 基线（采前 / 采后） | vis `59,965,440` / `59,965,440` B，gtt `30,244,864` / `30,244,864` B（卡上无其它进程持有 `renderD128`/`/dev/kfd`） |

## 2. 怎么结束的：挂住，不是 abort

- 训练跑到 step 10 之后**卡死**：`state.json` 的 `training.step = 10`、step-10 `loss = 0.0514`、
  `updated_at = 1789814230.34`（18:37:10）；hook 日志最后一行在 `t = 31.662524 s`；
  此后 12 分钟**没有一行新日志**（`frozen.log` mtime 18:37:11）。
- 卡住期间的取证：进程 state `R`、157 线程（`frozen-snap/ps-L.txt`）、GPU busy 37%、
  显存钉在 `14,832,037,888 B`、`sample.log` 4 Hz 采样一条不变。
- 18:49:14 抓下 `frozen-snap/`（`status`/`smaps_rollup`/`maps`/`numa_maps`/`ps-L`/`fd-count`；
  `VmSize` 37.8 GB、`VmRSS` 4.3 GB），随后 `kill -KILL` → `exit code 137`，卡回到采前基线。
- **所以 §4-d 的判据没命中**：这次既没有 `exit 134`（SIGABRT）也没有 `HSA_STATUS_ERROR_MEMORY_FAULT`
  记录在案（`crash.txt` 的 dmesg 段只有更早一次运行的页错误，pid 378516）。这次故障表现为
  **挂住（hang）**而不是 abort。窗口到挂住为止，仍然是合法的输入序列。

## 3. 日志里有 7 个 `T=`，流只取其中一个进程

同一份日志里，每个继承 `LD_PRELOAD` 的进程都会写行。判据：**同一个进程的线程共享 hook 的原子计数器，
它们的 seq 值两两不交；而 spawn 出来的辅助进程从 1 重新编号（撞号），fork 出来的子进程继续各自那份
拷贝的计数器（也撞号）。** 冻结快照的 `ps-L` 直接证实了被选中的两个 tid 就是训练进程 (405921) 的线程：

| T= | 行数 | M / F / H | 归属 | 依据 |
| --- | --- | --- | --- | --- |
| 405921 | 6,227 | 3,214 / 3,008 / 0 | **流**（训练主线程） | seq 1..8729 |
| 406787 | 2,420 | 1,735 / 685 / 0 | **流**（线程 `pt_autograd_0`） | seq 933..8604，与主线程不交；`ps-L` 里 TGID=405921 |
| 406641 | 84 | 0 / 0 / 84 | **流**（线程 `pt_data_pin`，pinned host 内存） | seq 570..2998，不交；`ps-L` 里 TGID=405921 |
| 405928 / 405929 / 406086 / 406528 | 19 | 0 / 0 / 0 | 不是流（orphans 回收器、其子进程、forkserver） | 各自从 seq 1 开始，只有 banner/注释 |

三个线程合起来正好铺满 seq 1..8729 且互不重叠（`ops.tsv` 的流 = 8726 笔 + 3 行注释）。

## 4. 流规模

| 项 | 值 |
| --- | --- |
| 操作 | **8,726** 笔：`hipMalloc` 4,949、`hipFree` 3,693、`hipHostMalloc` 84 |
| 窗口（日志自身时钟） | `t = 0.000012 .. 31.662524 s`；seq `1 .. 8729` |
| 末态活表 | **1,256 块 / 11,679,039,488 B** |
| 尺寸多重集 sha1（本脚本自己的配方） | `964e9c8d38f2a3ff8fdc5fb138fa9301ff3b1bdc` |
| 表文件 | `ops.tsv`（222 KB，8,726 行 + 4 行头）；原始日志 `frozen.log`（1.1 MB，8,750 行） |

## 5. 校验（§4 六项 + 残差）

| # | 检查 | 结果 |
| --- | --- | --- |
| a | 解析零残留 | 0 行无前缀、0 行无法分类（`parses_clean=True`） |
| b | 与 hook 自己的账对账 | 模拟末态 `1,256 块 / 11,679,039,488 B` == 日志最后一条 `live=`/`live_bytes=`，**逐字节** |
| c | 流纯不纯 | 只有一行 `# vmm tail: disabled by AMDFQ_TAIL=0`，没有 `falling back to pad`（hook 没有自己发过 `hipFree`/`hipMalloc`） |
| d | 崩溃签名 | **未命中**：是 hang（见 §2），`exit 137` 是我们的 `SIGKILL` |
| e | 与旧指纹同源？ | **命中**：把新表切到第 **4,853** 笔，正好是 `M/F/H = 2,994/1,775/84`、`1,219 块 / 9,661,579,264 B` —— 与 `amdfq/doc/live-data.md` §9.1/§9.3 记的旧 step-5 冻结窗口**逐字节相同**。尺寸多重集 sha1 无法比（旧配方没记录），我们的值是 `006d053cb517daab689051711a06434a35bfe671`（切到 4,853） |
| f | 可重复性 | **未做**（要另起一轮采集；见 §8） |
| — | 残差 | 未匹配 free 0、重复指针 0、malloc 失败 0、free 失败 0 |

**时间口径对不上，只能按 op 切。** 旧的 `frozen5t0b` 在 `t = 19.859 s` 时已经做过 2,994 次
`hipMalloc`；我们这次到第 2,994 次 `hipMalloc` 是 `t = 25.074 s`（按 `t <= 19.859` 切只有 993 次
malloc、735 块活表）。两次运行的启动/编码耗时不同，所以**秒不可比，op 序号可比** —— §4-e 用的是
op/malloc 口径。

## 6. HIP 重放程序（`amdfq/replay/`，自动读表）

`replay.cpp` 里没有一笔写死的操作：表路径由 argv 给，运行时读入、按序执行（10 万行与 1 千行走同一份
代码），槽号→真实指针的映射在运行时维护。只用 `hipMalloc`/`hipFree`/`hipHostMalloc`/`hipMemGetInfo`，
没有 `LD_PRELOAD`、没有 hook、没有 torch。

**节奏**按训练自己的时间线走：表里带一列 `t`（hook 在原始运行里记的秒数），第 i 笔在第
`X × (t_i − t_first)` 秒执行，于是簇还是簇、停顿还是停顿。`--pace X` 是那个倍数（默认 `1.0`；
`0` = 全速跑完）。本表 t 从 4.999535 s（模型加载完、第一笔分配）到 31.662524 s，所以 1:1 复现是
**26.67 s**。

**跑完不退出**：默认进入 0.5 s 一跳的循环、**保留全部活分配**，等 Ctrl+C（SIGINT）才释放并结束——
这样外部读数（或 BO dump）能对着"这份流的终态"取，而不是对着一个正在退场的进程。只有 SIGINT /
SIGTERM / SIGQUIT 会结束它：**SIGHUP 与 SIGPIPE 被忽略**，所以关掉终端、或把输出接到 `| head`
都不会把它带走；退出时那行会写明是**哪个信号**（`# interrupted by signal 2: …`），底下没这行就说明
是被 SIGKILL 掉的（例如 OOM）。脚本里用 `--exit` 可以跳过这段。

**两股输出分开走**：逐笔行与收尾汇总走 **stdout**（进 `replay.out`，是给对账用的数据）；心跳与
`# holding …` / `# interrupted by signal N …` 这类状态行走 **stderr**（是给操作者的），所以终端上
只看得到那三行数，几千行逐笔日志不会糊在眼前。

**`run.sh` 前台跑重放，不自己结束它**：脚本把重放放在前台（同一个前台进程组），并忽略自己的
SIGINT——终端按下的 Ctrl+C 由内核直接送给重放，重放释放全部活分配、打印 `# interrupted by signal 2`
后退出，脚本这才继续打印对账。所以既不存在"脚本自己退出"的歧义，也不存在脚本留下的孤儿进程；
每跑一次都顺带验一遍这条路径。开机前会先检查有没有别的 `replay` 在跑（一个就占 11.7 GB），有则拒绝
启动；收尾时再查一次，万一有残留会显式告警。

**每 0.5 s 一行两本账**（重放期间与保留期间都打，写 stderr；`--quiet` 只关 stdout 的逐笔行）：

```
#    26.68s held   live=1256/11679039488 B | ledger free vram 5341446144 B (5094.0 MiB) |
  driver free vram 2749992960 B (2622.6 MiB) | ledger - driver 2591453184 B (2471.4 MiB) |
  driver vram_used 14345990144 B (13681.4 MiB)
```

- `ledger free vram` = `hipMemGetInfo` 说这个进程还能拿到的量；
- `driver free vram` = sysfs `mem_info_vram_total − mem_info_vram_used`（`card1`），即驱动认为设备上还剩多少；
- `ledger − driver` = 两者之差：一开始是 −8.8 MiB（本程序自己的开销），跑到终态变成 **+2,471.4 MiB**
  —— 就是 ROCr 那笔对驱动已记账、却仍能交给自己客户的储备。

`bash amdfq/replay/run.sh amdfq/streams/20260919-dc57df11` 的实测：

| 项 | 结果 |
| --- | --- |
| 构建 | `hipcc --offload-arch=gfx1201 -O2 -Wall -std=c++17`，0 告警 |
| 跑完 | 8,726 笔，`elapsed = 26.669 s`（`pace = 1.00x`），`histogram_failures = 0` |
| 逐笔对账 | hook 8,642 行（M+F；H 行没有 `live=`/`live_bytes=` 字段）vs replay 8,642 行 → **mismatch 0** |
| replay 末态 | `live = 1256`、`live_bytes = 11,679,039,488`（与日志一致）；跑完 `vis_free = 5,341,446,144 B`、`vis_total = 17,095,983,104 B` |
| 空闲段 | 26.67 s 后打印 `# holding 1256 live allocations (11679039488 B); Ctrl+C to end`，`ps` 里 state `SN`；期间 sysfs 显存 14.43 GB（峰值）→ 14.35 GB；收到 SIGINT 后打印 `# interrupted by signal 2: …`、退出码 0、显存回到基线 `59,777,728 B`；共 45 行心跳 |
| 终端噪声 | 在真 pty 里跑 `run.sh` 并按 Ctrl+C：整屏 65 行 = 45 行心跳 + 20 行（脚本 3 行状态、重放 2 行状态、对账 3 行、驱动器自身的输出）；逐笔的 8,642 行全在 `replay.out` |
| 半途 Ctrl+C | 流跑到第 9 秒按 Ctrl+C：重放停在第 417 笔、释放退出；对账打印 `Ctrl+C at op 417 of 8642; the prefix agrees op for op`（不是 "DOES NOT AGREE"），退出码 0，无残留 |
| 残留检查 | 启动前若已有 `replay` 在跑就拒绝（实测拦下一次遗留进程，提示 `kill -INT <pid>`）；跑完再查一次并告警 |
| run.sh 的收尾 | 不再有任何"脚本发信号"的环节：重放在前台跑，Ctrl+C 由终端直接送达（见上文）。`--stay` 已移除——要留下持有态就直接跑 `amdfq/replay/replay <stream>/ops.tsv`。另：`--quiet` 会让逐笔对账拿不到数据，脚本会说明"check skipped"而不是报 `DOES NOT AGREE` |
| 输出 | `replay.out`（逐笔行 + 84 条 host 行 + 收尾汇总，8,7xx 行）；`replay.err`（心跳与状态行的副本） |

## 7. 目录里都有什么

| 文件 | 内容 |
| --- | --- |
| `frozen.log` | 原始 hook 日志（hook 自己逐行 `write`，无缓冲） |
| `ops.tsv` | 重放程序的输入；`seq / kind / size / slot / flags / ret` |
| `gen_ops.py` | 日志 → 表的生成器（含 §5 的全部校验与 §5-e 的切点） |
| `collect.sh`、`sample-host.sh` | 采集编排与 4 Hz 采样器 |
| `baseline.txt`、`crash.txt`、`collect.out`、`train.log`、`sample.log` | 采前基线、结束时取证、采集输出、训练 stderr、计数器采样 |
| `frozen-snap/` | 挂住时那个还活着的进程的快照（status/smaps_rollup/maps/numa_maps/ps-L/fd） |
| `replay.out`、`replay.err` | 重放的数据输出（逐笔 + 汇总）与状态输出（心跳 + `# holding`/`# interrupted`） |

复算：

```bash
cd amdfq/streams/20260919-dc57df11
python3 gen_ops.py --log frozen.log --out ops.tsv --cut 4853 --cut-malloc 2994 --cut-time 19.859
cd ../../.. && bash amdfq/replay/run.sh amdfq/streams/20260919-dc57df11
```

## 8. 这一轮没做的

- **第二次采集**（§4-f 的可重复性）：要另起一轮，串行跑，用来确认「同一份流」这个说法。
- **哨兵**、四档 pad（0/16/64/1 MiB）：按本轮范围不做；`replay --pad N` 已经留好接口。
- **没有提交 git**：`amdfq/streams/**` 与 `amdfq/replay/**` 是新增未跟踪文件，等你看过再说。

# 三个方案的代价：不挂补丁、C 尾守卫、Rust 尾守卫

> 一次实测，2026-09-19。AMD RX 9070 XT（gfx1201，`card1`，设备总量 `17,095,983,104 B = 16,304.0 MiB`），
> 桌面已关，空卡基线 `vis = 59,965,440 B = 57.2 MiB`、`gtt = 30,244,864 B`。
> 同一份 `config.toml`（md5 `bcd1b2362168643638768a4e9bef98e3`，跑完未变）、同一个 seed `1145141919`。
> 三组**串行**，上一组 vis 回到 ~57 MiB 才起下一组。口径沿用 [`doc/eva.md`](doc/eva.md) §1：
> 主数字是训练进程内 `hipMemGetInfo` 的 **free**；驱动 vis 只作参考。
> 结论里每一条都能在 `/tmp/amdfq-tail-bench/analysis.json` 和各组 `freeze-step*.json` 里逐行算出来。

上一轮 [`doc/eva.md`](doc/eva.md) 比的是 pad-16 对 C 尾守卫。本轮变量换成实现：无守卫、C Release、Rust release。pad-16 不再上场。

---

## 0. 一句话结论

| 方案 | 这个进程自己还能用的显存<br>（第 5 步 / 第 100 步 / 第 500 步） | 驱动计数器看到的<br>（同左，used） | 干净步墙钟 | 能跑到哪一步 |
| --- | --- | --- | --- | --- |
| **不挂补丁**（只挂 ledger） | **6,499 / — / —** MiB free | 11,880 / — / — MiB | 1.1365 s 中位数（7 步） | **第 10 步后空转**（SIGKILL；stderr 无 HSA 行） |
| **C 尾守卫**（Release） | **6,289 / 4,539 / 4,015** MiB free | 11,899 / 13,969 / 14,488 MiB | 1.1551 s（478 步）；相对无补丁 **+13.0 ms** | 500 步 SIGTERM |
| **Rust 尾守卫**（release） | **6,493 / 4,525 / 4,015** MiB free | 11,887 / 13,994 / 14,491 MiB | 1.1660 s（478 步）；相对无补丁 **+26.1 ms** | 500 步 SIGTERM |

一句话：**Rust 守卫相对无补丁几乎不占进程内可用显存（第 5 步 −6.0 MiB），C 在第 5 步少 210.0 MiB 但第 500 步与 Rust 逐字节相同；步时 C +1.1%、Rust +2.3%，长跑上 Rust 比 C 再多 9.0 ms/步。`Train/Loss` 三组能对齐的每一步都相等，无 NaN。**

Rust 并不更快：整段墙钟 C 840.9 s、Rust 852.6 s。进度报告里 Rust「一下子到 260」是因为 10 分钟档跨过了 C 收尾、腾卡、Rust 自己开跑，不是并行。

---

## 1. 计量口径

与 [`doc/eva.md`](doc/eva.md) §1 相同，本轮不再复述四本账的来历。本轮用到的是：

| 读数 | 谁在读 | 本轮角色 |
| --- | --- | --- |
| **进程内 ledger** | `libledger.so` 在训练进程里 4 Hz 调 `hipMemGetInfo`（构造时不 `dlopen` HIP，等 runtime 已经在了再 `dlsym`） | **主口径**：这个进程还能要到多少 |
| 外部 ledger | 旁路小进程 `extledger`，同样 `hipMemGetInfo` | 与进程内之差 = ROCr 储备 |
| 驱动 vis / GTT | `/sys/class/drm/card1/device/mem_info_vis_vram_{used}` 与 `mem_info_gtt_used` | 参考 + 落点 |

规矩仍是冻结后落定再读：`state.json` 的 `training.step` 到达目标 → `SIGSTOP` → 1.5 s → vis/GTT/外部 ledger 连读 5 次。本轮每个冻结点 vis 五次读数跨度都是 **0 B**。进程内 ledger 取冻结前最后一条 `rc=0` 样本。两个 ledger 不混用。每次跑从空卡起步，跑完 vis 回到 `59,965,440 B`。GTT 全程 43.0 MiB（空卡 28.9），**没有静默落主机**。

---

## 2. 实验设计

三个组，唯一变量是分配路径。三组都挂同一份 ledger + 外部 HIP 客户端，插桩开销是常数。「每步额外时间」相对 `nopatch`，不是相对完全空手。

| 组 | LD_PRELOAD | 守卫 | 说明 |
| --- | --- | --- | --- |
| `nopatch` | `/tmp/amdfq-tail-bench/libledger.so` | 无 | 无 hook 的显存基线。eva.md 的 `nopatch` |
| `tail-c` | ledger + `amdfq/amdfq-tail/cmake-build-release/libamdfq.so` | 开 | 原 C 树，**Release**（sha256 `dc57df11…`）。不用 `hook.sh` 默认的 debug |
| `tail-rs` | ledger + `amdfq/amdfq-tail-rs/target/release/libamdfq_tail_rs.so` | 开 | Rust release（sha256 `bcf5a536…`） |

日志按各自设计走（C 写 `AMDFQ_LOG` 文件；Rust `AMDFQ_LOG_LEVEL=info` 走 stderr）。冻结点：第 5 步三组都冻（无 hook 走不过第 10 步，这是唯一三组对齐的点）；第 100、500 步只冻 hook。第 500 步冻在 `set_training(500)` 之后、sample 之前。到 500 则 `SIGTERM`。

| 组 | pid | run_id | 墙钟 | 结局 |
| --- | --- | --- | --- | --- |
| nopatch | 397700 | `lllj_20260919_171151` | 185.8 s | max_step=10，exit −9 |
| tail-c | 398369 | `lllj_20260919_171459` | 840.9 s | 500，exit −15 |
| tail-rs | 401031 | `lllj_20260919_172903` | 852.6 s | 500，exit −15 |

---

## 3. 显存损失

单位 MiB。精确字节写在括号里。

### 3.1 第 5 步冻结点

| 组 | **进程内 used / free** | 外部 used / free | 驱动 vis | GTT | 储备（外部 used − 进程 used） | Δfree vs nopatch |
| --- | --- | --- | --- | --- | --- | --- |
| nopatch | **9,805.0 / 6,499.0**（`10,281,287,680` / `6,814,695,424`） | 11,926.0 / 4,378.0 | 11,880.2 | 43.0 | 2,121.0 | — |
| tail-c | **10,015.0 / 6,289.0**（`10,501,488,640` / `6,594,494,464`） | 11,946.0 / 4,358.0 | 11,899.4 | 43.0 | 1,931.0 | **−210.0** |
| tail-rs | **9,811.0 / 6,493.0**（`10,287,579,136` / `6,808,403,968`） | 11,934.0 / 4,370.0 | 11,887.5 | 43.0 | 2,123.0 | **−6.0** |

三条结论：

1. **Rust 守卫的自身成本是几颗 granule。** 相对 nopatch，进程内 free 少 `6,291,456 B = 6.0 MiB`，驱动 vis 高 `7,643,136 B = 7.3 MiB`。和「一颗共享守卫页」同量级。
2. **C 在第 5 步少 210.0 MiB 真实可用显存，驱动计数器只承认 19.2 MiB。** 进程内 free 少 `220,200,960 B`，vis 只高 `20,127,744 B`。少掉的几乎都来自储备（2,121 → 1,931）。只看 vis 会把这笔报成 19 MiB——和 eva.md §1.3 同一类口径陷阱，数字小一个数量级。这是单次冻结上的稳定差（vis 跨度 0 B），**没有第二跑**；不写成「C 的稳态税是 210 MiB」。
3. **第 5 步三组的 loss 相同**：`0.12364877760410309`。活着的计算流对齐。

### 3.2 第 100 步、第 500 步

| 组 | 步 | 进程内 used / free | 外部 used / free | vis | 储备 |
| --- | --- | --- | --- | --- | --- |
| tail-c | 100 | 11,765.0 / **4,539.0** | 14,018.0 / 2,286.0 | 13,968.8 | 2,253.0 |
| tail-rs | 100 | 11,779.0 / **4,525.0** | 14,036.0 / 2,268.0 | 13,993.7 | 2,257.0 |
| tail-c | 500 | 12,289.0 / **4,015.0** | 14,534.0 / 1,770.0 | 14,487.8 | 2,245.0 |
| tail-rs | 500 | 12,289.0 / **4,015.0** | 14,538.0 / 1,766.0 | 14,491.0 | 2,249.0 |

第 100 步 Rust 比 C 少 14.0 MiB free（`14,680,064 B`）。第 500 步进程内 used/free **逐字节相同**（`12,885,950,464` / `4,210,032,640`）；vis 差 `3,297,280 B = 3.1 MiB`。长跑上两条守卫占的进程内可用显存没有差出一颗 granule 以外。

---

## 4. 性能损失

TensorBoard `Train/Loss` 的 `wall_time` 差分。剔除冻结点及其后一步（5、6、100、101、500、501）、`save_every_n_steps = 50` 的存档/出图步、以及 ≥ 3 s 的离群点。剩下叫干净步。

### 4.1 与 nopatch 重叠的干净步（2、3、4、7、8、9、10）

带冻结邻步的格子不在这张表里。同一配置同一 seed。

| 组 | step 2 | step 3 | step 4 | step 7 | step 8 | step 9 | 均值差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nopatch | 1.2579 | 1.1475 | 1.0385 | 1.1353 | 1.0494 | 1.1365 | 基准 |
| tail-c | 1.2805 | 1.1608 | 1.0448 | 1.1520 | 1.0593 | 1.1424 | **+13.0 ms**（+1.1%） |
| tail-rs | 1.2951 | 1.1759 | 1.0584 | 1.1576 | 1.0667 | 1.1554 | **+26.1 ms**（+2.3%） |

单位秒。nopatch 干净步中位数 1.1365 s（n=7，含 step 10）。样本只有这些步，小数点当量级看。次序稳定：**nopatch < C < Rust**。

### 4.2 长跑干净步（11–499，去掉存档步）

| 组 | 走过的步数 | 干净步 | 中位数 s/步 | 相对 C |
| --- | --- | --- | --- | --- |
| nopatch | 10 | 7 | 1.1365 | — |
| tail-c | 500 | 478（长跑 471） | **1.1552** | 基准 |
| tail-rs | 500 | 478（长跑 471） | **1.1665** | 同一步均值 **+9.0 ms**（+0.8%） |

整段墙钟（编码缓存命中、三次冻结、十次 sample）：C 840.9 s，Rust 852.6 s（+11.7 s，+1.4%）。Rust 的日志走 stderr 热路径，C 走文件；`AMDFQ_LOG_LEVEL=warn` 会再削一截 Rust 的步时，本轮没测。

---

## 5. Loss：守卫有没有把计算弄脏

TensorBoard `Train/Loss`，与冻结点上 `state.json` 的 `loss` 交叉。

| 对照 | 步数 | max \|Δ\| | NaN | 结论 |
| --- | --- | --- | --- | --- |
| nopatch vs C | 1–10 | **0** | 无 | 逐值相同 |
| nopatch vs Rust | 1–10 | **0** | 无 | 逐值相同 |
| C vs Rust | 1–500 | **0** | 无 | 逐值相同 |

第 4 步三次都是 `0.10055211186408997`（与 [`doc/live-data.md`](doc/live-data.md) 的 `0.1005521` 一致）。第 5 步三次都是 `0.12364877760410309`。第 100 步 C 与 Rust 都是 `0.024769598618149757`，第 500 步都是 `0.10730841755867004`。

bf16 训练跨进程本来不保证 bit 相同。这次相同，说明至少在这份配置、这个 seed 上，两条守卫都没有把 GEMM 结果改掉。

---

## 6. 「不挂补丁」的代价

它仍然最省：第 5 步进程内 free 6,499.0 MiB，步墙钟也是最快。**它仍然走不过第 10 步。** 本轮第 10 步 loss 已经写入（`0.051359500735998154`），随后约 125 s 进程一直 `R`、vis 停在 ~14.8 GiB，stderr 没有 `HSA_STATUS_ERROR` / `Memory access fault` / `loss=nan`。按「第 10 步后不再前进」杀掉（SIGKILL −9），好让后面两组继续。这与 eva.md 里三次硬崩是同一类「无守卫走不过第 10 步」，但**现场不是那条 HSA 行**——可能是 fault 之后卡在 HIP 同步里。只有这一次，不写成机制变化。

---

## 7. 局限

1. **每个臂一次。** 第 5 步 C 的 −210 MiB 没有复跑。
2. 三组都挂 ledger + 外部 HIP 客户端。相对差可比；绝对值比完全空手的训练略紧。
3. 日志 I/O 算进实现。
4. nopatch 只有崩前那几步可用来比时间，长跑额外时间只能 C vs Rust。
5. 第 500 步的显存不含那一轮出图。

---

## 8. 产物与复现

| 内容 | 路径 |
| --- | --- |
| harness（按 step 冻结、四口径读数） | `/tmp/amdfq-tail-bench/{run_all.py,ledger.c,extledger.c,analyze.py,libledger.so}` |
| 各组冻结快照 | `/tmp/amdfq-tail-bench/{nopatch,tail-c,tail-rs}/freeze-step{5,100,500}.json` |
| 逐步 loss / vis | `…/steps.ndjson`、`…/driver.ndjson`、`…/ledger.log`、`…/extledger.log` |
| 表格来源 | `python3 /tmp/amdfq-tail-bench/analyze.py` → `analysis.json` |
| TensorBoard | `/home/acite/LLM/axltrainer/logs/lllj_20260919_17{1151,1459,2903}` |
| C / Rust 对象 | `amdfq/amdfq-tail/cmake-build-release/libamdfq.so`、`amdfq/amdfq-tail-rs/target/release/libamdfq_tail_rs.so` |

```bash
cmake -S amdfq/amdfq-tail -B amdfq/amdfq-tail/cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build amdfq/amdfq-tail/cmake-build-release
cargo build --release --manifest-path amdfq/amdfq-tail-rs/Cargo.toml
python3 -u /tmp/amdfq-tail-bench/run_all.py
/home/acite/miniconda3/envs/axl/bin/python /tmp/amdfq-tail-bench/analyze.py
```

复现前提：桌面关闭，空卡 vis ~60 MiB，一次只跑一个 GPU 进程，每个组跑完回到基线再起下一组。

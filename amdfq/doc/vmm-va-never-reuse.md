# VA 永不复用：amdfq-vmm-rs 下训练的早期 HSA 故障消失

**这份文件是什么。** 实验室记录：the hook 为什么存在、它实际改了训练进程的哪一段内存、挂上之后原先会怎样坏，以及 2026-09-20 测到的那一刀——**禁止复用已经 Map 过的 VA 之后，那些早期 HSA 故障在已经跑过的窗口里不再出现。** §12 起是同一命题的第二条线：不挂 hook、不用 torch，纯 HIP VMM 高频复用，GPU 写字节标签，看复用后的映射还是不是刚写进去的值。

不替代 `amdfq/amdfq-vmm-rs/DESIGN.md`。产品约束仍是 D10。探针源码在 `amdfq/vmm-cc/`，正文 §15 有一份当时的完整拷贝。

**地位。** 训练对照 2026-09-20 在本机测过。产品路径已经是永不复用。对照臂是同一条路线、同一天更早的 teardown（unmap + release + `hipMemAddressFree`）。两边都已经没有 `AMDFQ_POOL`，也没有 launch 观测门。单变量是：**free 之后这段 VA 能不能再被 Map。** HIP 探针 2026-09-20 同机，单变量同样是 AddressFree。

「驱动 / GPU 页表缺陷」是推断，不是结论。见 §9、§14。

---

## 1. 背景：训练为什么会需要一块「块后面的内存」

本机训练卡是 AMD RX 9070 XT（gfx1201）。不挂任何 hook 时，这条配置会在训练里随机死于 GPU page fault（常见是 step 10 附近、`HSA_STATUS_ERROR_MEMORY_FAULT` / `exit 134`）。

原因已经单独写清，不是本文要再证的：gfx1201 上 hipBLASLt 派发的 **bf16 Tensile kernel 会读出 A/B 操作数末尾**，幅度不超过 4 KiB，读到的值不进结果。操作数后面如果已经是映射着的显存，这件事完全无声；后面如果是一页没映射的显存，GPU 报页错误，进程被杀。权威记录：[`../../conclusions/bf16-kernel-overrun.md`](../../conclusions/bf16-kernel-overrun.md)。叙事：[`gfx1201-overread-story.md`](gfx1201-overread-story.md)。

所以补丁的目标从来不是改 kernel，而是：**让那次越界读落到一块合法的、已经映射的显存上。**

两条实现路线（`amdfq.md` §12 / §13）：

| 路线 | 对象 | 做法 |
| --- | --- | --- |
| **route 1 / the hook** | `amdfq/amdfq-vmm-rs/`（C 树 `amdfq-vmm/` 已删） | 自己 `hipMemAddressReserve` + `hipMemMap`，把 `hipMalloc` 要的块从 VMM 里交出去，块后面再映射一颗**全进程共享**的 pad granule |
| **route 2 / the tail hook** | `amdfq/amdfq-tail-rs/`（测量多用过 C 树 `amdfq/amdfq-tail/`） | `hipMalloc` 仍走 runtime；只在块的独占末端后面，若还没有东西 backing，再 map 一页共享 guard |

route 1 的动机是省显存：runtime 自己给每笔分配做 pad 时，粒度一进位就是每块一颗 2 MiB；共享一颗 pad 映射到每一笔后面，1000 个 2 MiB 块从 4000 MiB 降到约 2014 MiB（`amdfq.md` §12.1，C 树 `vmm_probe.c`）。本文只谈 **the hook**。tail 的覆盖时代（hipMalloc→Async）NaN 是另一条线，不要和这里混。

---

## 2. 这个 hook 实际做了什么

`LD_PRELOAD=amdfq/amdfq-vmm-rs/target/release/libamdfq_vmm_rs.so`。只导出 `hipMalloc` / `hipFree`。`start_train.sh` 默认不预加载，必须显式 `run.sh train`。

**`hipMalloc`（能 serve 时）** 不把请求交给 runtime 的 `hipMalloc`。顺序是：

1. 读当前 device，按该卡粒度把请求向上取整得到 `block`，再加一颗 pad，得到 `total`。
2. `hipMemAddressReserve(total)` 拿到一段 VA。
3. `hipMemCreate(block)` 做出这块自己的物理 handle，`hipMemMap` 到 `va`。
4. `hipMemSetAccess`：本卡 + peer + HOST。授权没授上就把映射拆掉，改走 runtime（不把一块 device-only / 无授权的指针交给 torch——host 上的 `.item()` 会直接读这个地址）。
5. 把**同一张卡、进程里只创建一次**的共享 pad handle，`hipMemMap` 到 `va+block`。pad 映射失败只让这一笔没有越界保护，块本身仍交出。

布局：

```
va                         va + block                    va + block + pad
|---------------- block ----------------|---- pad（共享 handle）----|
         调用方的 tensor 住这里              越界读落到这里
```

`block` 来自这笔请求自己的 handle；`pad` 是同一颗物理 2 MiB 映射到很多 VA 上。kernel 读过块尾会进这颗页，而不是走进未映射地址。

**`hipFree`（这笔是 Extent 时）** runtime 的 `hipFree` **看不见**这个指针（D10）：hook 自己 `hipSetDevice` 切回记录里的卡，`hipDeviceSynchronize`，然后 unmap 块、unmap pad、`hipMemRelease` 块的 handle。D10 改之前最后一步是 `hipMemAddressFree`，VA 还给驱动，下次 `hipMalloc` 可以再拿到同一段——这就是 **reuse**。改之后不再 AddressFree，VA 一直占着——**never-reuse**，也就是本文测的那一刀。

serve 不了的请求（VMM 入口不全、reserve/create/map 失败、设备状态建不起来）原样转发给 runtime 的 `hipMalloc`；那种指针的 free 也原样转发。

---

## 3. 原先出了什么问题

分两层。第一层是 hook **要修**的；第二层是 hook **自己引入**的。本文的测量针对第二层。

### 3.1 不挂 hook：Tensile 越界读 → page fault

越界读本身在。补丁不存在时，训练会在某个 bf16 GEMM 上走到未映射页，典型签名是 `HSA_STATUS_ERROR_MEMORY_FAULT`、`dmesg` 里 gfxhub page fault。这是 kernel 的事，tail 和 VMM 都是在给它垫一块合法内存。

### 3.2 挂上 the hook：进程自己的状态被弄坏，而且坏法不固定

route 1 从 C 树起就不是「垫成功、别的都好」。`amdfq.md` §12 把它从优化路线里关掉，理由写在 §12.2：同一条 `config.toml`、同一个 seed，五种互不共享 kernel / 算子 / 阶段的坏法，共同点是**进程自己的值、指令、PC、指针被弄脏**，而不是 §3.1 那种绑定 Tensile、每次都留下 gfxhub fault 的越界读。

当时记下来的包括（C 树，2026-09-17）：

| 坏法 | 例子 |
| --- | --- |
| 非法指令 | 第一步就 `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`：一次在 torch 的 `bfloat16_copy` kernel，一次（关掉 pad）在 Tensile GEMM。**共享 pad 不是充分原因。** |
| 取指跑飞 | SQC(inst) fault，地址在 1 TiB reservation 里、却在最高已映射块后面 772 GiB——wave 的 PC，不是数据 |
| 静默 NaN | step 1 `loss=nan`，`dmesg` 完全没有 GPU 事件，进程因为 `state.json` 拒写 NaN 结束 |
| 野指针 page fault | peralloc、无 pad：step 1 先给出正常 loss 0.9978，然后 fault 在 `0x7f1bd361f000`——**路线从未分配过的地址**，也不在 1643 笔 malloc / 638 笔 free 的记录里 |

对照：`AMDFQ_VMM=0`（仍是 +16 pad、不 serve）能健康过 step 10；只对镜像块跑同一套 VMM 调用、真正的 tensor 仍来自 `hipMalloc` 的 shadow 跑到 111 步（`amdfq.md` §12.3）。当时的读法：**VMM 调用本身、以及把共享对象 map 在操作数后面，在那个尺度上无害；坏的是把操作数自己从 VMM 内存里交出去。**

Rust 树把 peralloc 搬过来之后，第二层没有消失。中间还测过：把 `hipFree` 的整段 teardown 都注释掉，故障从 NaN 变成 OOM（[`hook-free-path-nan-vs-oom.md`](hook-free-path-nan-vs-oom.md)）——说明「拆掉一块已经交出去的 VMM 内存」和数值错误至少住在同一条因果链上，但四步里哪一步才是扳机，当时没有拆开。

### 3.3 2026-09-20、产品已经是「立刻 teardown」时：开头几步的 HSA，外加花屏

在去掉 `AMDFQ_POOL`、去掉 launch 观测门之后，teardown 仍是 unmap + release + **AddressFree**（reuse）。同一天四次 `run.sh train` 都在 encode 走完后的 **step 0–4** 被 HSA 杀掉，签名每次不一样：非法指令、memory fault、孔径违规。孔径那一次（58453）kernel 是 `layer_norm_grad_input_kernel`，GPU coredump 自己也没写成；维护者同时看到**桌面其它窗口被弄花**。这已经不是「只有训练进程的 tensor 坏了」——这张卡兼做出显示。

具体表在 §7。never-reuse 要回答的就是：这些第二层故障，是不是 **VA 被收回再 Map 上去** 就能充分触发。

不要和这些混：

| 签名 | 这条线 | 另一份 |
| --- | --- | --- |
| hook 下训练早期 HSA 炸 / 花屏，never-reuse 后同一窗口能过 | VA 复用 | 本文 §7 |
| 纯 HIP VMM：reuse 后刚 remap 的槽里出现 `0x00`，`--never-reuse` 1000 次 PASS | 同一单变量，无 torch / hook | 本文 §12–§14 |
| 不挂 hook 的 gfxhub page fault（Tensile bf16 越界读） | kernel 越界 | `conclusions/bf16-kernel-overrun.md` |
| 整段 teardown 都跳过：NaN 变成 OOM | 四步全不拆 | `hook-free-path-nan-vs-oom.md` |

---

## 4. 用词

| 词 | 这里的意思 |
| --- | --- |
| **the hook** | `amdfq/amdfq-vmm-rs/`，peralloc：`LD_PRELOAD` `target/release/libamdfq_vmm_rs.so`。`hipMalloc` 从本 crate `hipMemAddressReserve` 出来的地址上 `hipMemMap`。 |
| **the tail hook** | `amdfq/amdfq-tail-rs/`。不是本文的对象。 |
| **reuse** | free 时 `hipMemAddressFree`。下一次 `hipMemAddressReserve` 可以把同一段 VA 再交出去、再 Map。 |
| **never-reuse** | 一旦 Map 成功，`[va, va+total)` 记入 `EVER_MAPPED`，free 不 `hipMemAddressFree`。新的 reserve 若与已记录跨度相交，这一笔不 Map。 |
| **extent** | 一笔 served 分配：`block` + 一颗 pad granule 的 reserved VA、块自己的 handle、共享 pad 映射、device。 |
| **served** | 日志 `hipMalloc(…) -> ret=0 served va=…`：路线自己交出的指针，runtime 的 `hipMalloc` 没看见它。 |

---

## 5. 环境（除非某行另写）

| 项 | 值 |
| --- | --- |
| GPU | AMD RX 9070 XT，gfx1201，16 GB，兼做出显示 |
| 解释器 | conda `axl`，torch `2.13.0+rocm10.0.0`，HIP `7.15.26333` |
| 工作负载 | `AMDFQ_LOG_LEVEL=off\|info ./amdfq/amdfq-vmm-rs/run.sh train` → `bash start_train.sh` |
| `config.toml` | 未改；验证跑前 md5 `bcd1b2362168643638768a4e9bef98e3` |
| 粒度 | 日志 `granule=2097152`（2 MiB） |

---

## 6. 这次 hook 里改了什么

reuse（对照，D10 改之前）的 teardown 顺序仍是：

`hipDeviceSynchronize` → `hipMemUnmap(block)` → `hipMemUnmap(pad)` → `hipMemRelease` → `hipMemAddressFree`

never-reuse（产品，`peralloc.rs`）在 `hipMemMap` 成功后调用 `remember_mapped`；`teardown` 不再
`hipMemAddressFree`。从未 Map 成功的 reserve（create / map 失败）仍释放地址。

物理显存仍随 handle 释放。废弃的是 GPU VA。长跑会一直占 VA，这是机制本身的代价，不是意外。

---

## 7. 测到的

同一天、同一张卡、同一条 `run.sh train`。reuse 臂四次都在 step 0–4 死于 HSA；never-reuse 臂两次都越过了那个窗口，loss 有限，进程是人停的。

| 臂 | 日志 | 最远 | 签名 |
| --- | --- | --- | --- |
| reuse | `/tmp/amdfq-rs-train.53589.log` | step 4 | `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`，Tensile GEMM kernel |
| reuse | `/tmp/amdfq-rs-train.55217.log` | step 0 | `HSA_STATUS_ERROR_MEMORY_FAULT` |
| reuse | `/tmp/amdfq-rs-train.55747.log` | step 1 | `HSA_STATUS_ERROR_MEMORY_FAULT` |
| reuse | `/tmp/amdfq-rs-train.58453.log` | step 0 | `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`（`code: 0x29`），kernel `at::native::layer_norm_grad_input_kernel<float, float, false>`；GPU coredump 失败；维护者同时看到桌面其它窗口被弄花 |
| never-reuse | `/tmp/amdfq-rs-train.60355.log` | step 76 | 640 张 latent encode 完成；step 50 采样三套 prompt × 35 denoise 跑完；逐步 loss 有限（停时 `0.0626`，`avg_loss` `0.097`）；`KeyboardInterrupt`。无 HSA 行 |
| never-reuse | `/tmp/amdfq-vmm-noreuse-verify/train.stderr.log` | step 45 | 过了预先设定的 step 25 判定后 SIGINT。无 HSA 行。停时 loss `0.214`，`avg_loss` `0.110` |

58453 的判定行（stderr，encode 100% 之后立刻）：

```
Warning: Queue error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address.
…
kernel: void at::native::(anonymous namespace)::layer_norm_grad_input_kernel<float, float, false>(…)
```

花屏只有维护者这一次目击（58453），验证跑没有再开 reuse 臂，也没有第三人看见花屏。把它记成「reuse 臂上出现过」，不当成独立复现。

---

## 8. 操纵检查（验证跑）

验证跑用 `AMDFQ_LOG_LEVEL=info`，所以能数 served VA。60355 是 `off`，没有逐次 malloc 行。

从 `/tmp/amdfq-vmm-noreuse-verify/train.stderr.log` 数到：

| 项 | 值 |
| --- | --- |
| served `hipMalloc` | 15237 |
| 其中不同的起始 VA | 15237 |
| 起始 VA 重复 | 0 |
| reserved extent 相交 | 0 |
| 日志 `overlaps a mapped span` | 0 |
| `untracked` / live `duplicate` | 0 / 0 |
| 解析到的 `step=N loss=` | 90 个，范围 0.0071–0.5906，无 NaN |
| `HSA_STATUS` / `Queue error` / `GPU coredump` | 0 |

`test.sh` 的默认两笔同尺寸 alloc（free 夹在中间）也是两个不同 served VA，检查 `served VA never reused` 通过。

---

## 9. 推断（未证实）

**能当事实的：** 在这台机器、当时的 so、这条训练路径上，**VA 复用是那些早期 HSA 故障的充分触发条件**；禁止复用后，同一路径可以跨过原先的炸点。60355 和验证跑是两个独立样本。HIP 探针（§12）在去掉 torch / hook 之后，同一把「复用就坏、不复用就过」仍在，坏法是刚 remap 的那一槽里出现 `0x00`，不是 HSA 杀进程。

**还不能当结论的：**

1. **「NaN / PF / Hang 从此都消失」。** 两边都没有跑完 6460 步。只能说在已观察到的窗口里（reuse：0–4 步死；never-reuse：45 步和 76 步、含一次采样）没再出现那些签名。
2. **「就是驱动 / GPU 页表缺陷」。** 现象和「同一 VA 被拆掉再铺上」强相关，和页表 / TLB 住得很近，但没有第二证人（没读驱动、没 dump PTE）。同样解释得通、也还没拆开的至少有：
   - 驱动在 `AddressFree` + 再次 `Reserve`/`Map` 同一 VA 时写出坏 PTE（页表缺陷）；
   - 我们自己的 unmap + `AddressFree` 序列在复用时把映射弄坏，never-reuse 只是绕开了那条 teardown；
   - GPU 按 VA 缓存了 PTE，复用打到过期 TLB。
3. **花屏的机制。** 孔径违规可以是 GPU hang / reset 之后 compositor 扫到坏帧，也可以是 VRAM / 页表已经伤到显示客户端。58453 没有把这两种拆开。

`hook-free-path-nan-vs-oom.md` §3.3 问过「四步 teardown 哪一步缺了会移动故障」。本文只跳过了 `hipMemAddressFree`，unmap 和 release 仍在。这是那道题的一个单步答案，不是四步都试过。它也还没有单独重跑 §3.2 里 C 树那五种坏法——只覆盖了 Rust 树、立刻 teardown、2026-09-20 的早期 HSA 窗口。

---

## 10. 怎么复现

对照臂会花屏，不要在桌面会话上为了「再拿一个样本」去跑。never-reuse 臂：

```bash
# 产品 so（D10 永不复用）
cd /path/to/AxlTrainer
AMDFQ_LOG_LEVEL=info bash amdfq/amdfq-vmm-rs/run.sh train
# 操纵检查：served 起始 VA 必须全部唯一
grep -oP 'served va=\K0x[0-9a-f]+' "$AMDFQ_LOG_FILE" | sort | uniq -d
```

判定：encode 走完之后，训练条在 step ≥ 25 且 stderr 没有 `HSA_STATUS` / `Queue error`，即越过了对照臂的炸点。完整 6460 步不是本文件已经付过的账。

---

## 11. 相关

| 文件 | 关系 |
| --- | --- |
| `amdfq.md` §12 | route 1 的成本、五种历史坏法、shadow 对照；C 树上的记录 |
| `amdfq/amdfq-vmm-rs/DESIGN.md` D10 | 产品约束：Map 过的 VA 不 `hipMemAddressFree` |
| `amdfq/amdfq-vmm-rs/src/peralloc.rs` | serve / teardown、`EVER_MAPPED`、`remember_mapped` |
| `hook-free-path-nan-vs-oom.md` | 整段 teardown 都跳过：NaN → OOM；§3.3 是「拆成一步一步跳」 |
| `pool-budget-vs-steps.md` | 当时还有 `AMDFQ_POOL`；产品里已经删掉，和本文不是同一变量 |
| `conclusions/bf16-kernel-overrun.md` | 不挂 hook 的 Tensile 越界读（§3.1） |
| `gfx1201-overread-story.md` | 从对齐误判到越界读、再到两条补丁路线的叙事 |
| `amdfq/vmm-cc/` | 纯 HIP VMM 探针（§12–§15）；不挂 hook |

---

## 12. HIP 探针：方法

训练路径（§7）里 reuse 会在 step 0–4 被 HSA 杀掉，never-reuse 能过那个窗口。那条路径经过 torch、caching allocator、Tensile、hook 的 pad。要回答的下一句是：**只对 VMM 自己做高频 Reserve/Map/Unmap/Release，GPU 写一块带标签的内存，复用 VA 会不会把标签写丢。**

探针在 `amdfq/vmm-cc/`，不 `LD_PRELOAD` hook，不跑 trainer。编译和调用：

```bash
bash amdfq/vmm-cc/run.sh --iters 1000
bash amdfq/vmm-cc/run.sh --iters 1000 --never-reuse
```

`hipcc --offload-arch=gfx1201 -O2 -Wall -std=c++17`。设备 0，单线程，默认 stream。环境同 §5 的卡和 HIP `7.15.26333`；不经过 conda torch。

**存活集。** 最多 256 个槽。槽 `i`（0-based）整块每个字节应为 `i`。默认每块 2 MiB，按 `hipMemGetAllocationGranularity` 的 minimum 向上取整（本机 gran=4096，2 MiB 已对齐）。

**一笔活块。** `hipMemCreate` → `hipMemAddressReserve` → `hipMemMap` → `hipMemSetAccess`（本卡 + HOST，HOST 失败则只授本卡）→ GPU kernel `fill_tag` 把整块写成该槽标签 → `hipDeviceSynchronize` → 按 `--check-every`（默认 1）对每个存活槽起 `check_tag`，统计 `!= tag` 的字节和第一个坏偏移。

**拆。** 满 256 之后 round-robin：拆槽 `v`，立刻在同一槽位再建。默认 teardown 对齐 HIP 7.15 how-to 示例和 CUDA VMM「Releasing the Memory」（`cuMemUnmap` / `cuMemRelease` / `cuMemAddressFree` **按这个顺序**）：

```
hipDeviceSynchronize
hipMemUnmap
hipMemRelease          # 物理 handle；--hold-old-phys 会推迟这一步
hipMemAddressFree      # 默认 reuse；--never-reuse 跳过
```

HIP 同一页的**正文**把 AddressFree / Release 说反了，和它自己的代码不一致。以示例代码为准。本机 `hip_runtime_api.h` 只解释各函数做什么，不规定顺序。

**单变量。** `--never-reuse`：Unmap + Release 仍做，VA 不还。物理上仍是 256 块。`--hold-old-phys` 和 `--no-sync-before-unmap` 这次没跑。

**Reserve 若交回与仍存活槽相交的 VA，直接 FAIL**（「把还活着的地址又发出来」）。失败时若 `got==0x00` 会打印「标签等于 got 的槽」——槽 0 的标签就是 `0x00`，和「新 backing 没写上、仍是 0」分不开，**不当成槽 0 乱写的证据**。

判定：跑完 iters 且一次串味都没有 → 退出 0。发现错误字节或 HIP 调用失败 → 退出 1。

---

## 13. HIP 探针：结果

2026-09-20，同卡。reuse 臂没有一次活过「第一次 AddressFree 之后的那几步」；never-reuse 和「还没拆」都是绿的。

### 13.1 维护者样本（`--iters 1000`）

```
❯ bash amdfq/vmm-cc/run.sh --iters 1000
device AMD Radeon RX 9070 XT (gfx1201) gran=4096 size=2097152 live=256 iters=1000 check-every=1 never-reuse=0 hold-old-phys=0 no-sync-before-unmap=0
FAIL slot=3 expect=0x03 got=0x00 off=0 mismatches=2097152 va=0x7f51e5c00000 size=2097152 bytes: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
     va_hits_for_this=2 unique_va=256 reuse_hits=4 iters=259 live=256
     slot whose tag equals got: slot=0 va=0x7f53ef800000

❯ bash amdfq/vmm-cc/run.sh --iters 1000
device AMD Radeon RX 9070 XT (gfx1201) gran=4096 size=2097152 live=256 iters=1000 check-every=1 never-reuse=0 hold-old-phys=0 no-sync-before-unmap=0
FAIL slot=14 expect=0x0e got=0x00 off=925696 mismatches=1170944 va=0x7f8ab1a00000 size=2097152 bytes: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
     va_hits_for_this=2 unique_va=256 reuse_hits=15 iters=270 live=256
     slot whose tag equals got: slot=0 va=0x7f8cc2400000

❯ bash amdfq/vmm-cc/run.sh --iters 1000 --never-reuse
device AMD Radeon RX 9070 XT (gfx1201) gran=4096 size=2097152 live=256 iters=1000 check-every=1 never-reuse=1 hold-old-phys=0 no-sync-before-unmap=0
hb iters=391 live=256 unique_va=392 reuse_hits=0 checks=392 ok dt=1.0s
hb iters=667 live=256 unique_va=668 reuse_hits=0 checks=668 ok dt=2.0s
hb iters=945 live=256 unique_va=946 reuse_hits=0 checks=946 ok dt=3.0s
hb iters=1000 live=256 unique_va=1000 reuse_hits=0 checks=1000 ok dt=3.2s
PASS iters=1000 unique_va=1000 reuse_hits=0 checks=1000 never-reuse=1 hold-old-phys=0

❯ bash amdfq/vmm-cc/run.sh --iters 1000 --never-reuse
device AMD Radeon RX 9070 XT (gfx1201) gran=4096 size=2097152 live=256 iters=1000 check-every=1 never-reuse=1 hold-old-phys=0 no-sync-before-unmap=0
hb iters=389 live=256 unique_va=390 reuse_hits=0 checks=390 ok dt=1.0s
hb iters=666 live=256 unique_va=667 reuse_hits=0 checks=667 ok dt=2.0s
hb iters=945 live=256 unique_va=946 reuse_hits=0 checks=946 ok dt=3.0s
hb iters=1000 live=256 unique_va=1000 reuse_hits=0 checks=1000 ok dt=3.2s
PASS iters=1000 unique_va=1000 reuse_hits=0 checks=1000 never-reuse=1 hold-old-phys=0
```

### 13.2 同日更早的对照（`--iters 256` / `512`）

`--iters 256` 还没 teardown：2/2 PASS，`reuse_hits=0`。`--iters 512 --never-reuse`：2/2 PASS，`unique_va=512`。默认 reuse、`--iters 512`：11/11 FAIL，形状和 §13.1 一样。

reuse 失败行满足 `iters == 256 + slot`、`reuse_hits == slot + 1`、`va_hits_for_this == 2`、`unique_va == 256`、`got == 0x00`。也就是：**坏的是刚 round-robin 拆掉又 Map 回去的那一槽**，这块 VA 是第二次出现，检查看到的是 0 而不是该槽标签。有的整块 2 MiB 全 0（`off=0, mismatches=2097152`），有的从某个 `off` 起一大段是 0（`off` 都整除 256，对 4096 却常常不对齐）。

| 臂 | n | 次数 | 结果 |
| --- | --- | --- | --- |
| 默认 reuse | 256 | 2 | PASS（还没 AddressFree） |
| 默认 reuse | 512 | 11 | FAIL，slot 3–21，iters 259–277 |
| 默认 reuse | 1000 | 2 | FAIL，slot 3 / 14，iters 259 / 270 |
| `--never-reuse` | 512 | 2 | PASS |
| `--never-reuse` | 1000 | 2 | PASS，`unique_va=1000`，`reuse_hits=0` |

没有一次 reuse 跑到 512/1000。没有一次 never-reuse 在同一 iters 上 FAIL。Reserve 没有交回与存活槽相交的 VA。

---

## 14. HIP 探针：能当事实的 / 还不能当结论的

**能当事实的：**

1. 在这台 gfx1201、HIP 7.15.26333、这个探针上，**`hipMemAddressFree` 之后再次 Map 同一段已用过的 VA，是 GPU 可见的标签丢失的充分条件。** 丢失表现为刚 remap 的那一槽里出现 `0x00`（整块或后缀），不是 HIP API 返回错误、也不是 HSA 杀进程。
2. 同一套 `fill_tag` / `check_tag`、同样 `Unmap`+`Release`、同样 256 活槽，只要不 AddressFree，1000 次申请（1000 个不同 VA）两次都过。还没拆的 256 次两次都过。所以不是「2 MiB fill kernel 在第一次 Map 上就会漏写」，也不是 round-robin 写错槽。
3. 训练路径 §7 和这条探针的单变量是同一个：free 之后 VA 能不能再被 Map。两条线都是 reuse 坏、never-reuse 过。探针没有 torch、没有 hook、没有 pad。

**还不能当结论的：**

1. **「就是 TLB / 页表没刷」。** 没读驱动、没 dump GPU PTE。`0x00` 也符合「新物理页没被这次 fill 写到、检查读到的是新 backing」。fill 走 GPU store、check 也走 GPU load，中间没有再 Unmap，所以和「Map 之后 GPU 翻译还指向别处 / 指向未写的页」住得很近，仍是推断。
2. **`got=0x00` 不是槽 0 往失败槽里乱写。** 槽 0 标签就是 `0x00`；失败槽的 VA 和打印出来的槽 0 VA 不是同一块。槽 0 在 iter 256 被第一个拆掉再 fill `0x00`：若故障就是「remap 后仍是 0」，那一次检查会通过，所以 reuse 臂从不在 slot=0 失败。
3. **没有证明 fill 打到了另一块还活着的映射。** 11+2 次 reuse 失败都是「刚 remap 的那一槽自己不是期望标签」，不是邻居槽变成了当前标签。
4. **`--hold-old-phys` 没跑**，所以「fill 是否落到刚 Release 掉的旧物理页」没有直接证人。
5. 探针默认 2 MiB、256 槽、单 stream；没有复现训练里的非法指令 / 孔径 / 花屏。它只回答字节标签，不替代 §7。

怎么复现：§12 的两条命令。reuse 臂在这张卡上会在 ~260 次申请处 FAIL，不会花屏（和 §7 的训练 reuse 不同）；never-reuse 臂大约 3 s 过 1000 次。

---

## 15. 探针源码（当时完整拷贝）

树里的文件是 [`../vmm-cc/vmm_cc.hip`](../vmm-cc/vmm_cc.hip)，启动器 [`../vmm-cc/run.sh`](../vmm-cc/run.sh)。下面与 2026-09-20 测 §13 时同一份。

```hip
/* High-frequency HIP VMM churn with a 256-slot byte-tag integrity check.
 *
 * Each live slot i holds size bytes, every byte equal to i. The process keeps at most 256
 * mappings. Once full, it tears one down (Unmap -> hipMemRelease -> hipMemAddressFree) and
 * immediately maps a replacement into that slot — the teardown that let the trainer reuse VA.
 *
 *   bash amdfq/vmm-cc/run.sh [--iters N] [--size B] [--check-every K]
 *                            [--never-reuse] [--hold-old-phys] [--no-sync-before-unmap]
 *
 * Do not LD_PRELOAD the hook; this talks to the VMM API itself.
 */
#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <unordered_map>
#include <vector>

namespace {

const int kLive = 256;
const int kThreads = 256;

#define CHECK(expr)                                                                 \
    do {                                                                            \
        hipError_t e_ = (expr);                                                     \
        if (e_ != hipSuccess) {                                                     \
            std::fprintf(stderr, "FAIL %s -> %s (%d)\n", #expr,                     \
                         hipGetErrorString(e_), static_cast<int>(e_));              \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

volatile sig_atomic_t g_stop = 0;

void on_sigint(int) { g_stop = 1; }

__global__ void fill_tag(unsigned char* p, size_t n, unsigned char tag) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        p[i] = tag;
    }
}

__global__ void check_tag(const unsigned char* p, size_t n, unsigned char tag,
                          unsigned long long* mismatches, unsigned long long* first_off) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        if (p[i] != tag) {
            atomicAdd(mismatches, 1ull);
            atomicMin(first_off, static_cast<unsigned long long>(i));
        }
    }
}

int launch_blocks(size_t n) {
    size_t blocks = (n + static_cast<size_t>(kThreads) - 1) / static_cast<size_t>(kThreads);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    return static_cast<int>(blocks);
}

void gpu_fill(void* va, size_t n, unsigned char tag) {
    fill_tag<<<launch_blocks(n), kThreads>>>(static_cast<unsigned char*>(va), n, tag);
    CHECK(hipGetLastError());
}

struct CheckOut {
    unsigned long long mismatches;
    unsigned long long first_off;
};

struct Slot {
    void* va = nullptr;
    hipMemGenericAllocationHandle_t handle = nullptr;
    bool live = false;
};

struct Config {
    uint64_t iters = 1024;
    size_t size = 2ull << 20;
    uint64_t check_every = 1;
    bool never_reuse = false;
    bool hold_old_phys = false;
    bool no_sync_before_unmap = false;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--iters N] [--size B] [--check-every K]\n"
                 "          [--never-reuse] [--hold-old-phys] [--no-sync-before-unmap]\n",
                 argv0);
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            c.iters = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            c.size = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--check-every") == 0 && i + 1 < argc) {
            c.check_every = std::strtoull(argv[++i], nullptr, 10);
            if (c.check_every == 0) {
                c.check_every = 1;
            }
        } else if (std::strcmp(argv[i], "--never-reuse") == 0) {
            c.never_reuse = true;
        } else if (std::strcmp(argv[i], "--hold-old-phys") == 0) {
            c.hold_old_phys = true;
        } else if (std::strcmp(argv[i], "--no-sync-before-unmap") == 0) {
            c.no_sync_before_unmap = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (c.size == 0) {
        std::fprintf(stderr, "FAIL --size must be > 0\n");
        std::exit(2);
    }
    return c;
}

double monotonic_s() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

hipMemAllocationProp device_prop() {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    return prop;
}

void set_access(void* va, size_t size) {
    hipMemAccessDesc desc[2]{};
    desc[0].location.type = hipMemLocationTypeDevice;
    desc[0].location.id = 0;
    desc[0].flags = hipMemAccessFlagsProtReadWrite;
    desc[1].location.type = hipMemLocationTypeHost;
    desc[1].location.id = 0;
    desc[1].flags = hipMemAccessFlagsProtReadWrite;
    hipError_t both = hipMemSetAccess(va, size, desc, 2);
    if (both == hipSuccess) {
        return;
    }
    (void)hipGetLastError();
    CHECK(hipMemSetAccess(va, size, desc, 1));
}

void dump_bad(int slot, void* va, size_t size, unsigned char expect, unsigned long long off,
              unsigned long long mismatches) {
    unsigned char got = 0;
    size_t nbytes = 16;
    if (off >= size) {
        off = 0;
    }
    if (off + nbytes > size) {
        nbytes = size - static_cast<size_t>(off);
    }
    std::vector<unsigned char> buf(nbytes);
    CHECK(hipMemcpy(&got, static_cast<unsigned char*>(va) + off, 1, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(buf.data(), static_cast<unsigned char*>(va) + off, nbytes,
                    hipMemcpyDeviceToHost));
    std::fprintf(stderr,
                 "FAIL slot=%d expect=0x%02x got=0x%02x off=%llu mismatches=%llu va=%p size=%zu "
                 "bytes:",
                 slot, expect, got, off, mismatches, va, size);
    for (size_t i = 0; i < nbytes; i++) {
        std::fprintf(stderr, " %02x", buf[i]);
    }
    std::fprintf(stderr, "\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    Config cfg = parse_args(argc, argv);
    signal(SIGINT, on_sigint);

    CHECK(hipSetDevice(0));
    hipDeviceProp_t dprop{};
    CHECK(hipGetDeviceProperties(&dprop, 0));

    hipMemAllocationProp prop = device_prop();
    size_t gran = 0;
    CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    size_t size = ((cfg.size + gran - 1) / gran) * gran;

    std::fprintf(stderr,
                 "device %s (%s) gran=%zu size=%zu live=%d iters=%" PRIu64
                 " check-every=%" PRIu64 " never-reuse=%d hold-old-phys=%d no-sync-before-unmap=%d\n",
                 dprop.name, dprop.gcnArchName, gran, size, kLive, cfg.iters, cfg.check_every,
                 cfg.never_reuse ? 1 : 0, cfg.hold_old_phys ? 1 : 0,
                 cfg.no_sync_before_unmap ? 1 : 0);

    Slot slots[kLive];
    int filled = 0;
    int victim = 0;

    CheckOut* d_out = nullptr;
    CHECK(hipMalloc(&d_out, sizeof(CheckOut) * static_cast<size_t>(kLive)));

    void* ghost = nullptr;
    if (cfg.hold_old_phys) {
        CHECK(hipMemAddressReserve(&ghost, size, gran, nullptr, 0));
    }

    std::unordered_map<uintptr_t, uint64_t> va_hits;
    uint64_t reuse_hits = 0;
    uint64_t checks = 0;
    uint64_t done = 0;
    double t0 = monotonic_s();
    double t_beat = t0;

    auto record_va = [&](void* va) {
        uintptr_t key = reinterpret_cast<uintptr_t>(va);
        uint64_t& n = va_hits[key];
        if (n > 0) {
            reuse_hits++;
        }
        n++;
    };

    auto ranges_overlap = [&](void* a, void* b) {
        auto aa = reinterpret_cast<uintptr_t>(a);
        auto bb = reinterpret_cast<uintptr_t>(b);
        return aa < bb + size && bb < aa + size;
    };

    auto map_new = [&](int slot) {
        hipMemGenericAllocationHandle_t handle = nullptr;
        CHECK(hipMemCreate(&handle, size, &prop, 0));
        void* va = nullptr;
        CHECK(hipMemAddressReserve(&va, size, gran, nullptr, 0));
        for (int i = 0; i < kLive; i++) {
            if (!slots[i].live) {
                continue;
            }
            if (ranges_overlap(va, slots[i].va)) {
                std::fprintf(stderr,
                             "FAIL Reserve returned va=%p overlapping live slot=%d va=%p size=%zu\n",
                             va, i, slots[i].va, size);
                std::exit(1);
            }
        }
        record_va(va);
        CHECK(hipMemMap(va, size, 0, handle, 0));
        set_access(va, size);
        slots[slot].va = va;
        slots[slot].handle = handle;
        slots[slot].live = true;
    };

    auto teardown = [&](int slot, hipMemGenericAllocationHandle_t* kept) {
        if (!cfg.no_sync_before_unmap) {
            CHECK(hipDeviceSynchronize());
        }
        CHECK(hipMemUnmap(slots[slot].va, size));
        /* HIP how-to sample + CUDA VMM: Unmap, then Release the handle, then AddressFree.
         * --hold-old-phys keeps the handle so old backing can be remapped at the ghost VA. */
        if (cfg.hold_old_phys) {
            *kept = slots[slot].handle;
        } else {
            CHECK(hipMemRelease(slots[slot].handle));
            *kept = nullptr;
        }
        if (cfg.never_reuse) {
            /* VA stays reserved so this span cannot be mapped again. */
        } else {
            CHECK(hipMemAddressFree(slots[slot].va, size));
        }
        slots[slot].va = nullptr;
        slots[slot].handle = nullptr;
        slots[slot].live = false;
    };

    auto verify_one = [&](void* va, unsigned char tag, int slot_for_msg, const char* what) {
        CheckOut init{};
        init.mismatches = 0;
        init.first_off = ~0ull;
        CHECK(hipMemcpy(d_out, &init, sizeof(init), hipMemcpyHostToDevice));
        check_tag<<<launch_blocks(size), kThreads>>>(static_cast<unsigned char*>(va), size, tag,
                                                     &d_out->mismatches, &d_out->first_off);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CheckOut out{};
        CHECK(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
        checks++;
        if (out.mismatches != 0) {
            std::fprintf(stderr, "FAIL %s\n", what);
            dump_bad(slot_for_msg, va, size, tag, out.first_off, out.mismatches);
            std::fprintf(stderr, "     unique_va=%zu reuse_hits=%" PRIu64 " iters=%" PRIu64 "\n",
                         va_hits.size(), reuse_hits, done);
            std::exit(1);
        }
    };

    auto verify_all_live = [&]() {
        CheckOut init[kLive];
        for (int i = 0; i < kLive; i++) {
            init[i].mismatches = 0;
            init[i].first_off = ~0ull;
        }
        CHECK(hipMemcpy(d_out, init, sizeof(init), hipMemcpyHostToDevice));
        int nlive = 0;
        for (int i = 0; i < kLive; i++) {
            if (!slots[i].live) {
                continue;
            }
            nlive++;
            unsigned char tag = static_cast<unsigned char>(i);
            check_tag<<<launch_blocks(size), kThreads>>>(
                static_cast<unsigned char*>(slots[i].va), size, tag, &d_out[i].mismatches,
                &d_out[i].first_off);
            CHECK(hipGetLastError());
        }
        CHECK(hipDeviceSynchronize());
        CheckOut host[kLive];
        CHECK(hipMemcpy(host, d_out, sizeof(host), hipMemcpyDeviceToHost));
        checks++;
        for (int i = 0; i < kLive; i++) {
            if (!slots[i].live) {
                continue;
            }
            if (host[i].mismatches != 0) {
                unsigned char tag = static_cast<unsigned char>(i);
                dump_bad(i, slots[i].va, size, tag, host[i].first_off, host[i].mismatches);
                auto it = va_hits.find(reinterpret_cast<uintptr_t>(slots[i].va));
                uint64_t hits = it == va_hits.end() ? 0 : it->second;
                std::fprintf(stderr,
                             "     va_hits_for_this=%" PRIu64 " unique_va=%zu reuse_hits=%" PRIu64
                             " iters=%" PRIu64 " live=%d\n",
                             hits, va_hits.size(), reuse_hits, done, nlive);
                unsigned char got_tag = 0;
                CHECK(hipMemcpy(&got_tag,
                                static_cast<unsigned char*>(slots[i].va) +
                                    (host[i].first_off < size ? host[i].first_off : 0),
                                1, hipMemcpyDeviceToHost));
                for (int j = 0; j < kLive; j++) {
                    if (slots[j].live && static_cast<unsigned char>(j) == got_tag) {
                        std::fprintf(stderr, "     slot whose tag equals got: slot=%d va=%p\n", j,
                                     slots[j].va);
                    }
                }
                std::exit(1);
            }
        }
    };

    auto heartbeat = [&](bool force) {
        double now = monotonic_s();
        if (!force && now - t_beat < 1.0) {
            return;
        }
        t_beat = now;
        std::fprintf(stderr,
                     "hb iters=%" PRIu64 " live=%d unique_va=%zu reuse_hits=%" PRIu64
                     " checks=%" PRIu64 " ok dt=%.1fs\n",
                     done, filled, va_hits.size(), reuse_hits, checks, now - t0);
    };

    for (done = 0; done < cfg.iters && !g_stop; done++) {
        hipMemGenericAllocationHandle_t old_handle = nullptr;
        unsigned char old_tag = 0;
        int slot;
        if (filled < kLive) {
            slot = filled++;
        } else {
            slot = victim;
            victim = (victim + 1) % kLive;
            old_tag = static_cast<unsigned char>(slot);
            teardown(slot, &old_handle);
            if (cfg.hold_old_phys) {
                CHECK(hipMemMap(ghost, size, 0, old_handle, 0));
                set_access(ghost, size);
            }
        }

        map_new(slot);

        unsigned char tag = static_cast<unsigned char>(slot);
        if (cfg.hold_old_phys && old_handle != nullptr) {
            /* A fill of the same tag into reused VA is invisible on old phys. Poison first. */
            unsigned char poison = static_cast<unsigned char>(tag ^ 0xff);
            gpu_fill(slots[slot].va, size, poison);
            CHECK(hipDeviceSynchronize());
            verify_one(ghost, old_tag, slot, "hold-old-phys: GPU fill of new VA landed on old phys");
            gpu_fill(slots[slot].va, size, tag);
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemUnmap(ghost, size));
            CHECK(hipMemRelease(old_handle));
        } else {
            gpu_fill(slots[slot].va, size, tag);
            CHECK(hipDeviceSynchronize());
        }

        if ((done + 1) % cfg.check_every == 0) {
            verify_all_live();
        }
        heartbeat(false);
    }

    heartbeat(true);

    for (int i = 0; i < kLive; i++) {
        if (!slots[i].live) {
            continue;
        }
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemUnmap(slots[i].va, size));
        CHECK(hipMemRelease(slots[i].handle));
        if (!cfg.never_reuse) {
            CHECK(hipMemAddressFree(slots[i].va, size));
        }
        slots[i].live = false;
    }
    if (ghost) {
        CHECK(hipMemAddressFree(ghost, size));
    }
    CHECK(hipFree(d_out));

    std::printf("PASS iters=%" PRIu64 " unique_va=%zu reuse_hits=%" PRIu64 " checks=%" PRIu64
                " never-reuse=%d hold-old-phys=%d\n",
                done, va_hits.size(), reuse_hits, checks, cfg.never_reuse ? 1 : 0,
                cfg.hold_old_phys ? 1 : 0);
    if (g_stop) {
        return 130;
    }
    return 0;
}
```


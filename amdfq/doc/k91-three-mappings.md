# Lab record: hooked `--driver-frees` down to a 91×91 `mm`, then three VMM mappings

**What this file is.** The method, the words, the knobs, and the numbers for the 2026-09-19
work that started as "ratchet `test/torch-test.py` toward the trainer" and stopped on a 91×91
`torch.mm` whose three operands each sit in their own VMM mapping. It is meant to be readable
without the chat that produced it. The original plan and the running notes live in
`torch-test-toward-trainer.md`; this file does not replace that plan, it archives what was
actually done.

**Status.** Methods and results through the launch-instrumented three-way comparison are
measured. §10 then kills “three mappings” as the cause: packing onto one mapping still FAILs
while `AMDFQ_POOL=0`; the same uncached 91×91 PASSes 3/3 when the deferred-free pool holds
extents. The open question is which unmap races with the GEMM, not how many mappings A,B,C occupy.

Raw stdout of the cited runs: `torch-test-ladder/`. `/tmp/torchtest-ladder/` was the
working directory and is volatile.

---

## 1. Words (use these, not near-synonyms)

| Term | Meaning here |
| --- | --- |
| **the hook** | `amdfq/amdfq-vmm-rs/` in peralloc mode: `LD_PRELOAD` of `target/release/libamdfq_vmm_rs.so`. It serves `hipMalloc` from address ranges it reserved itself. |
| **the C hook** | The deleted C VMM tree (`amdfq-vmm/`). The surviving C tree is `amdfq/amdfq-tail/`. Not used in this work. |
| **uncached / `--driver-frees`** | `PYTORCH_NO_HIP_MEMORY_CACHING=1` is set **before** `import torch`. Every torch free is a `hipFree`. Torch treats env `=0` like unset (caching **on**); only `=1` turns it off. Probed 2026-09-19. |
| **cached** | Caching allocator on. Torch sub-allocates; the hook sees fewer, larger `hipMalloc`s. |
| **caller** | The bytes `hipMalloc` was asked for (`HookData::size`). The tensor lives here. |
| **block** | Those bytes rounded up to the VMM mapping granularity (measured **2 MiB** = 2097152). Mapped from a handle created for this request. |
| **slack** | `block − caller`. Still mapped, not part of the tensor. Uncached 91×91 bf16: 2 MiB − 16562 B. |
| **pad** | One extra granule mapped behind the block from a **process-wide shared** handle, so a kernel that reads past the block does not walk off the GPU page tables. Also 2 MiB. Layout: `va … va+block … va+block+pad`. |
| **extent** | One served allocation: the reserved VA (`block + pad`), the block handle, the pad mapping, the device id. |
| **packed** | A, B and C of the GEMM sit at different offsets of **one** `hipMalloc`. The cached k=91 run is packed (offsets 0 / 16896 / 33792 in a 2 MiB block). |
| **three mappings / split** | A, B and C each came from their own `hipMalloc`, so each has its own 2 MiB-aligned VA, own slack, own pad. The uncached k=90 and k=91 runs are split. |
| **k** | Square side of the mm check: `m1, m2, out` are `k×k`. Flag `--mm-k`. Default in the script is 512; the cut is 90 vs 91. |
| **PASS** | Process exit 0 and every check the script actually ran is green. |
| **FAIL** | A check is red, or the process dies. For the mm check: GPU result is non-finite, or `max abs err` vs CPU fp64 exceeds `rtol=0.05, atol=1e-2`. |
| **err** | `max abs err` of the GPU `k×k` result (promoted to fp64 on the host) against `m1.double().cpu() @ m2.double().cpu()`. |
| **UserArgs extra** | 140-byte packed blob on `hipExtModuleLaunchKernel`. Word 2 holds two `u32` sizes, word 3 holds `1` and K, words 4–7 are D, C, B, A device pointers, words 8–11 are strides (all equal to k), word 12 is `1.0f` alpha. |
| **items / wg** | `hipExtModuleLaunchKernel` grid: global work-items and work-group size. 1152 items / 32-thread groups = 36 groups = 6×6 tiles of MT 16 → a **96×96** covered area even when k is 90 or 91. |
| **hooked** | `AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test …` |
| **unhooked** | The same python line, no `LD_PRELOAD`. |

The two defects that must not be mixed:

| Signature | This work | The other file |
| --- | --- | --- |
| Hooked FAIL, unhooked PASS, often finite wrong numbers or NaN | the hook / VMM layout | this file |
| Unhooked `HSA_STATUS_ERROR_MEMORY_FAULT` (gfxhub page fault) | Tensile bf16 over-read | `conclusions/bf16-kernel-overrun.md` |

---

## 2. Environment (fixed unless a row says otherwise)

| Item | Value |
| --- | --- |
| GPU | AMD RX 9070 XT, gfx1201, 16 GB, otherwise idle (~76 MiB VRAM at rest) |
| Interpreter | conda env `axl`, torch `2.13.0+rocm10.0.0`, HIP `7.15.26333` |
| Hook | `amdfq/amdfq-vmm-rs/target/release/libamdfq_vmm_rs.so` |
| Pool | `AMDFQ_POOL=0` (every `hipFree` of a served pointer tears the extent down: unmap block, unmap pad, release, address-free) |
| Seed | `--seed 0` (script default) |
| cwd | repo root |
| `config.toml` | not read; md5 left `bcd1b2362168643638768a4e9bef98e3` |
| Desktop | not required for these ~0.25 s probes; VRAM was at rest |

---

## 3. The workload

`test/torch-test.py` is not collected by `unittest discover`. After 2026-09-19 it has these flags
(defaults in parentheses):

| Flag | Default | What it changes |
| --- | --- | --- |
| `--driver-frees` | off | sets `PYTORCH_NO_HIP_MEMORY_CACHING=1` **before** `import torch` |
| `--no-stress` | off | skips plateau, stress loop, `verify()`, size/peak checks |
| `--skip-checks N` | 0 | drops the first N of the 7 correctness samples |
| `--max-checks N` | all | runs at most N samples after the skip |
| `--mm-k N` | 512 | square size of sample 4 (the mm) |
| `--mm-dtype` | `bf16` | `bf16` / `fp32` / `fp16` for that mm |
| `--linear` | off | **not used** in any run below |
| `--time-scale` / `--mem-scale` | 1 | only matter if stress is on |

The 7 samples, in order: CPU elemwise; scan ones; scan rows; **bf16/fp32 mm**; autograd x³;
autograd 2048 mm; autocast backward.

The mm sample (the only one that remains in the minimal command) is:

```python
m1 = torch.randn(k, k, device="cuda", dtype=dtype)
m2 = torch.randn(k, k, device="cuda", dtype=dtype)
out = m1 @ m2
got = out.double().cpu()
ref = m1.double().cpu() @ m2.double().cpu()
```

PASS if `got` is finite and `allclose` to `ref` with `rtol=0.05, atol=1e-2`. The script also prints
`mm va m1=… m2=… out=… nbytes=…`.

Hooked launcher:

```bash
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test [flags…]
```

Unhooked:

```bash
python -u test/torch-test.py [flags…]
```

Log: `$AMDFQ_LOG_FILE` (run.sh default `/tmp/amdfq-rs-test.<pid>.log`). `AMDFQ_LOG_LEVEL=info`
records every `hipMalloc` / `hipFree` / launch. Launch gates (added for §7):
`hipModuleGetFunction`, `hipLaunchKernel`, `hipModuleLaunchKernel`, `hipExtModuleLaunchKernel`.
`registry::locate` labels a pointer in a still-live extent as `caller` / `slack` / `pad`. The word
`untracked` is **not** used on launch misses (reserved for a `hipFree` of an unknown pointer;
`test.sh` greps it).

---

## 4. Campaign A — stage 2 (`--driver-frees` on the full script)

Question: does turning every torch free into a `hipFree` make the hooked script fail?

| Arm | n | exit | `hipFree` (hook log) | outcome |
| --- | ---: | --- | ---: | --- |
| unhooked `--driver-frees` | 2 | 0, 0 | — | PASS, 2.07 s / 2.11 s |
| hooked, no flag (caching on) | 1 | 0 | 104 | PASS, 2.07 s |
| hooked `--driver-frees` r1 | 1 | 1 | 3068 | FAIL, process lived |
| hooked `--driver-frees` r2 | 1 | 1 | 3065 | FAIL, process lived |
| hooked `--driver-frees` r3 | 1 | 134 SIGABRT | 2598 (cut short) | FAIL in checks, then GPU page-not-present |

First red check on r1–r3: `matmul: bf16 vs cpu fp64` at **512×512**, `max abs err 2377.967` all
three times (unhooked / hooked-cached: 0.250). r3 then:
`Memory access fault by GPU node-1 … address 0x51bb3000. Reason: Page not present or supervisor privilege.`

`hipFree` count moved ~30× (104 → ~3066). The stage happened. Stage 3 (`--linear`) was not run.

---

## 5. Campaign B — confirm caching on, then subtract

**Caching on, hook on, full script** (the requested confirmation):

```
PYTORCH_NO_HIP_MEMORY_CACHING=0 AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test
```

`caching_allocator enabled=True`, env `'0'`. PASS, 2.05 s, 103 `hipFree`.

Then keep `--driver-frees` (`=1`) and cut work. All hooked unless noted.

| Cut | Command extra | outcome |
| --- | --- | --- |
| drop plateau / loop / verify | `--no-stress` | FAIL, still 512×512 mm err **2377.967** |
| only first 3 samples | `--no-stress --max-checks 3` | PASS |
| only the mm | `--no-stress --skip-checks 3 --max-checks 1` | FAIL, err 2441.124 |
| same mm, unhooked | same flags, no preload | PASS, err 0.250 |

Prefix scans are not required. The mm alone is enough.

Shrink `--mm-k` (hooked, `--driver-frees --no-stress --skip-checks 3 --max-checks 1`, seed 0):

| k | dtype | n | result | err |
| ---: | --- | ---: | --- | ---: |
| 1, 8, 16, 64 | bf16 | 1 / 1 / 1 / **3** | PASS | 0.000–0.105 |
| 65–88 | bf16 | 1 each | PASS | ≤0.103 |
| **90** | bf16 | **3** | **PASS** | **0.118 ×3** |
| **90** | fp32 | 1 | PASS | 0.000 |
| **91** | bf16 | **3** | **FAIL** | **339.957, nan, 37.250** |
| **91** | fp32 | 1 | FAIL | 339.812 |
| 92 | bf16 | 1 hooked FAIL (357.095); 1 unhooked PASS (0.095) | | |
| 94, 95, 96 | bf16 | 1 / 1 / 3 | FAIL | 403 / 413 / 361.960, nan, 361.960 |
| 128–256 | bf16 | 1…**3** at 256 | FAIL | 528 … **1151.451 ×3** |
| 512 | bf16 | 2 | FAIL | **2441.124 ×2** |
| 512 | fp32, fp16 | 1 each | FAIL | 2439.370, 2439.494 |
| 256 | fp32 | 1 | FAIL | 1151.611 |
| 64 | fp32 | 1 | PASS | 0.000 |

Cut is **90 PASS / 91 FAIL**, both dtypes. k≥96 often repeats the same err; k=91 does not.

Minimal FAIL command:

```
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 91
```

Last PASS: the same with `--mm-k 90`.

---

## 6. Campaign C — instrument the 91×91 launch

Gates as in §3. Three hooked runs of the minimal command, k in {90, 91}, plus k=91 **without**
`--driver-frees` (cached). Kernel name in all three (abbreviated):

`Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_…_ISA1201_…_WS32_WG16_2_1`

| | k=90 split (uncached) | k=91 split (uncached) | k=91 packed (cached) |
| --- | --- | --- | --- |
| result | PASS, err 0.118 | FAIL, err 37.250, **finite** | PASS, err 0.117 |
| launch | `items=1152,1,1 wg=32,1,1` | same | same |
| extra M,N,K,lda | `0x5a` = 90 | `0x5b` = 91 | `0x5b` = 91 |
| alpha | `0x3f800000` = 1.0f | same | same |
| hipMalloc for A,B,C | 16200 × 3, three VAs | 16562 × 3, three VAs | **one** 2097152 B block |
| Python VAs | m1, m2, out each 2 MiB aligned | same | m1+0, m2+0x4200, out+0x8400 of the same block (16896 = 16562 rounded to 256) |
| extra words 4–7 | D=C=out, B=m2, A=m1; all `off=0 caller` of **their own** block | same pattern | D=C, B, A all `caller` of **the same** block at those offsets |
| extra hits in slack or pad | none | none | none |
| `hipFree` of A,B,C | after mm and after `.double().cpu()` | same | not in this window (cache holds the 2 MiB) |

Also allocated in the uncached runs, still live at launch: 25 165 760 B and 79 691 776 B (hipBLASLt
code-object / workspace; not the GEMM operands). Five `hipLaunchKernel`s are the `randn` fills and
the later fp64 promotions, not the GEMM.

Decoded extra (k=91 split), 17 × u64 then 4 unused bytes of the 140:

```
[0] 0x2220000100000001
[1] 0x2408010008
[2] 0x5b0000005b        # u32,u32 = 91, 91
[3] 0x5b00000001        # u32,u32 = 1, 91
[4] D = out
[5] C = out
[6] B = m2
[7] A = m1
[8..11] 0x5b            # lda/ldb/ldc/ldd = 91
[12] 0x3f800000         # 1.0f
[13..16] 0
```

k=90 is the same with `0x5a` = 90. Cached k=91 is the same 91s with the packed pointers.

`1152 / 32 = 36` workgroups = 6×6 MT-16 tiles → 96×96 covered for both 90 and 91
(`ceil(90/16) = ceil(91/16) = 6`).

---

## 7. What is measured, and what is not

Measured:

1. Hooked + uncached + this mm is enough to get the script's own numbers wrong. Unhooked uncached
   is not. Hooked cached (full script, and isolated 91×91) is not.
2. The GEMM is **not** launched with a pointer that has already been `hipFree`d. A, B, C, D are
   live, in `caller`, at launch. Operand teardown is afterwards.
3. Split 90×90 and split 91×91 fire the **same** kernel and **same** grid. Extra differs in the
   size fields (correctly 90 vs 91) and the three VAs.
4. Split 91×91 **fails**; packed 91×91 **passes**, same kernel, same grid, same 91s. The variable
   that moved is the layout: three 2 MiB VMM mappings versus one.

Not established:

- That this is use-after-free of anything. The launch-time pointers are live. (A later kernel
  using a VA after a later `hipFree` is a different question, and is how the full-script r3 died.)
- Why three mappings of a 91×91 are wrong and one mapping of the same 91×91 is not. §8.
- Why 90 split passes and 91 split fails, given the same kernel and the same 96×96 coverage.
- That this 91×91 failure is the trainer's NaN / `ILLEGAL_INSTRUCTION` path. It is the smallest
  hooked `--driver-frees` failure found. The trainer still has many more opcodes.

---

## 8. Hypotheses for "three mappings compute the wrong GEMM" (not conclusions)

These were the readings that fit §6–§7 **before** §10. §10 kills H2, H3, H4, and H1 as written
(error in every tile; pack / pack-block / 20 MiB still FAIL). What remains is §10.5: immediate
teardown, not the A,B,C layout.

---

## 10. Investigation (2026-09-19): three mappings are not why it is wrong

The question in the title is answered negatively. Split vs packed was the wrong cut. Immediate
VMM teardown is the cut that actually moves the result.

New script flags used below: `--mm-pack` (one allocation, 512-byte gaps), `--mm-pack-block`
(one 2 MiB allocation), `--mm-pack-mib N`, `--mm-sync` (`torch.cuda.synchronize()` after the mm).
All hooked, `--driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 91`, seed 0, unless
a row says otherwise.

### 10.1 Error is not an edge-tile

Split, pool 0, one FAIL: peak at `[42,50]` tile `(2,3)`, `got=37.25` vs `ref=0.0004`. All 36
MT-16 tiles are wrong (tile max 21.9–37.3). Not “only the 91-mod-16 tail”.

### 10.2 Packing does not fix it (`--driver-frees`, `AMDFQ_POOL=0`)

| Layout | hipMalloc for A,B,C | offsets | n | result | err |
| --- | --- | --- | ---: | --- | ---: |
| split (three mappings) | 16562 × 3 | 2 MiB-aligned bases | 1 | FAIL | 37.250, identical tile map |
| `--mm-pack` | 49920 (then 50688 at 512-align) | 0 / 0x4100 then 0 / 0x4200 / 0x8400 | 3+3 | FAIL | 37.250 ×2, then 1149 / 37.250 ×2 |
| `--mm-pack-block` | **2097152** (caller = block, no slack) | 0 / 0x4200 / 0x8400 | 3 | FAIL | nan, 37.250, 37.250 then 830–1674 at 512-align |
| `--mm-pack-mib 20` | **20971520** | 0 / 0x4200 / 0x8400 | 3 | FAIL | 735, 989, 1111 |

Same kernel, same extra 91s. H2 (three page tables), H3 (2 MiB-aligned bases), H4 (2 MiB
spacing), and “no slack” (H1 as written) are all **killed**: one mapping, 512-aligned packed
offsets matching the cached PASS, 2 MiB or 20 MiB caller, still FAIL while `--driver-frees` and
`AMDFQ_POOL=0` stay on.

### 10.3 Caching on still PASSes with the same 2 MiB packed layout

| Arm | `--driver-frees` | hipMalloc | offsets | result |
| --- | --- | --- | --- | --- |
| cached split (`randn` + `@`) | no | 2097152 | 0 / 0x4200 / 0x8400 | PASS, err 0.117 |
| cached `--mm-pack-block` | no | 20971520 (allocator rounded up) | 0 / 0x4200 / 0x8400 | PASS, err 0.117 |

So “one 2 MiB VMM mapping with those offsets” is **not** sufficient to PASS. The cached PASS and
the uncached pack-block FAIL share that layout. What they do not share is
`PYTORCH_NO_HIP_MEMORY_CACHING=1` and the hook actually tearing extents down.

### 10.4 Holding extents after `hipFree` restores PASS

| Arm | n | result | err |
| --- | ---: | --- | ---: |
| `AMDFQ_POOL=2684354560` (2.5 GiB), `--driver-frees`, split | **3** | **PASS ×3** | **0.117 ×3** |
| `AMDFQ_POOL=0`, `HIP_LAUNCH_BLOCKING=1`, split | 1 | FAIL | 1168.581 |
| `AMDFQ_POOL=0`, `--mm-sync`, split | **3** | FAIL ×3 | 5.00e31, 5.21e26, 1.15e30 |

The 2.5 GiB pool never unmaps in this short process (`hipFree` lines are `pooled`, not
`released`). That is the same knob that bought training steps in `pool-budget-vs-steps.md`.
`HIP_LAUNCH_BLOCKING=1` does not save the pool-0 run (it may not apply to
`hipExtModuleLaunchKernel`). `torch.cuda.synchronize()` after the mm, before the CPU compare,
does not save it either and the err gets *larger*.

Operand `hipFree`s in these logs are still **after** the GEMM launch and after the fp64
conversion buffers; launch-time A,B,C remain live in extra either way. What the pool changes is
whether those later frees (and any the log does not show as `released`) actually unmap.

### 10.5 What this does and does not say

Measured:

- The 91×91 failure is **not** “three independent mappings confuse the kernel”. Packing onto one
  mapping, filling the whole granule, or using a 20 MiB mapping does not reproduce the cached
  PASS while frees still tear extents down.
- The 91×91 failure **does** require immediate teardown (`AMDFQ_POOL=0`). The same uncached
  split PASSes 3/3 when the pool holds the extents. Caching on also PASSes, and it also avoids
  those teardowns.
- Launch-time A, B, C are live in every FAIL that was instrumented. The premature unmap, if that
  is the mechanism, is not of those three pointers at launch.

Inference (one reading, not a verdict): hipBLASLt’s GEMM is still using **some** VMM extent when
a later `hipFree` unmaps **some** extent (possibly a different one, possibly via a GPU VM flush),
and the hook’s `hipDeviceSynchronize` before unmap does not wait for that queue — the same
`cpu_wait` / `streamSet_` gap already read out of CLR for ordinary `hipFree`. The 2.5 GiB pool
and the caching allocator both skip that unmap, so the GEMM finishes on stable page tables.
`HIP_LAUNCH_BLOCKING` and `torch.cuda.synchronize()` not fixing it is consistent with that
queue not being the one those waits cover. This is the original “we unmapped too soon”
hypothesis applied to the 91×91, not a new one. It has two witnesses that skip teardown and
PASS (pool, cache) and many that teardown and FAIL; it does not yet have a witness that waits
the right queue and PASSes with teardown still on.

---

## 11. Reproduce

```bash
# last PASS (split 90)
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test \
  --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 90

# first FAIL (split 91)
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test \
  --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 91

# packed 91 (caching on) — PASS
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test \
  --no-stress --skip-checks 3 --max-checks 1 --mm-k 91

# unhooked uncached 91-neighbour (92) — PASS
python -u test/torch-test.py --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 92

# uncached 91 with the pool holding extents — PASS
AMDFQ_POOL=2684354560 bash amdfq/amdfq-vmm-rs/run.sh test \
  --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 91

# one mapping, still uncached, still pool 0 — FAIL (three mappings are not required)
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test \
  --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 91 --mm-pack-block
```

Hook self-check after the launch gates: `bash amdfq/amdfq-vmm-rs/test.sh` (exit 0, 2026-09-19).

Cited stdout: `torch-test-ladder/` (`k91-instrument.txt`, `inst2-k90.out`,
`inst2-k91.out`, `inst-k91-cacheon.out`, the stage-2 and subtraction outs).

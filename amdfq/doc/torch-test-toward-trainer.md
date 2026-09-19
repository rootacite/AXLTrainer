# Ratchet `test/torch-test.py` toward the trainer until the hook crashes it

**Lab record of methods, terms, knobs and numbers:** `k91-three-mappings.md`. This file is the
plan plus running notes.

**What this is.** A plan, not a result. No stage has been run yet. The pool campaign
(`pool-budget-vs-steps.md`) is enough to say the hook's failure is very likely premature teardown of
physical VRAM. The next question is **who still uses that memory, and why**. Splitting the four
teardown calls, tracing VAs after `hipFree`, and a keep-last-N knob are lower information than this:
`test/torch-test.py` already survives the hook, the trainer does not, so the difference is a set of
operations the script does not do. Add those operations one at a time until the script dies the way
the trainer does; the first failing stage minus the last passing one is the delta to instrument.

**Decided 2026-09-19 (maintainer).** Stage 2 before stage 3. Stages land as flags on
`test/torch-test.py` itself, not a second script. Unittest discover still ignores the file (`torch-test.py`
does not match `test*.py`).

**Status.** Plan only. The baseline (current script, `AMDFQ_POOL=0`, hook on) is already known to
pass; everything below that is unmeasured.

---

## 1. Why this, and what it is not

The script under the hook already does VMM-backed GEMM, a three-layer backward, host↔device hops,
and real `hipFree` (via `empty_cache()`), then checks that every number it produced is the number it
should have. It exits 0. A training run under the same hook, with the pool off, dies at step 0
(`pool-budget-vs-steps.md`: 5/5 at `AMDFQ_POOL=0`, median 0). So "any GPU work on a VMM mapping" is
not the trigger.

This ratchet does **not** claim the script is UAF-free today. `verify()` catches corruption of
tensors the script owns and reads (plateau, loss, grads). It does not catch a use of padding, of
allocator metadata, or of a pointer the script never looks at. A PASS means "the script's own
numbers did not break", not "no use-after-free happened". The first stage that fails `verify()` or
kills the process is still a usable delta: that is when instrumentation starts, not before.

This is also not the bf16 Tensile over-read (`bf16-kernel-overrun.md`). That defect kills an
**unhooked** training run at step 10 on this configuration. Mixing the two would look like "the
script finally crashed" when it had only wandered onto a shape the library already over-reads.

---

## 2. Two failure signatures (keep them apart)

Every stage runs **with the hook and without it**. Only "hook dies, no-hook lives" counts as finding
the delta.

| Signature | Meaning | Already seen |
| --- | --- | --- |
| Hook: NaN / `ILLEGAL_INSTRUCTION` / hang / `verify()` fail. No hook: clean | Premature teardown. This is the target | Hooked training, pool off, step 0 |
| No hook also `HSA_STATUS_ERROR_MEMORY_FAULT` (gfxhub page fault) | Tensile bf16 over-read. Wrong defect | Unhooked training, step 10 |
| `OutOfMemoryError` | The stage does not fit the card, or the pool is holding what the workload needs | Pool at 6 GiB in the budget campaign |

If a stage hits the second row, change the shapes and retry; do not treat it as a successful
finding. If it hits the third, shrink the stage, do not raise the pool.

---

## 3. Controls

- **`AMDFQ_POOL=0`.** Immediate teardown. Unpooled training dies at step 0; the current script
  survives that. The first stage that does not is the delta. The default 2.5 GiB pool pushes the
  trainer's death to steps 4–6, and the script's default loop is only 5 iterations — a slow leak
  and a PASS would look the same.
- **Same hook object, same interpreter.** `bash amdfq/amdfq-vmm-rs/run.sh test …` for the hooked arm;
  the same python line without `LD_PRELOAD` for the control. Conda env `axl`.
- **Repeats.** At least 3 hooked + 3 unhooked per stage. The hook's failure is not stable (NaN vs
  fault vs stall in the pool campaign).
- **Do not change `config.toml`.** This work does not start a training run. The file stays at md5
  `bcd1b2362168643638768a4e9bef98e3`.
- **No GPU work until the maintainer says the plan is enough and to start.** This file is that
  plan.

Pass: process exit 0 and every `verify()` check green.

Hit: hooked arm only, one of the first-row signatures.

---

## 4. Known HIP-level difference (already measured)

`hip-runtime-calls.md` §12 compared one trainer window (`lllj_20260918_073313`, 10 steps) to
`python -u test/torch-test.py --time-scale 1`. Same shape (~94 % device queries), different scale
and mix:

- Steady state: ~785 k HIP calls per training step vs ~6.6 k per script iteration (~119×).
- The script is copy-shaped (`hipMemcpyWithStream` 273/iter vs 4/step). The trainer is GEMM-shaped
  (`hipExtModuleLaunchKernel` 9,141/step vs 9/iter).
- Driver-visible `hipMalloc`/`hipFree`: 342+339 per step vs 12+12 per iteration. The script's
  `churn()` asks for 96 allocations; the caching allocator serves most of them from blocks freed
  moments earlier.
- Trainer-only in that window: `hipModuleLaunchKernel` and `hipStreamGetDevice` (AOTriton / SDPA),
  `hipSetDevice` (MIOpen re-asserting the device), `hipModuleLoadData` / `LoadDataEx`,
  `hipPointerGetAttributes`.
- The script's own comment already refuses `F.linear` / `x @ w.T` because the first transposed
  GEMM costs ~1.6 s of kernel loading on this stack.

Those rows are the order of the ladder: one HIP-visible dimension per stage, not "load the UNet".

---

## 5. The ladder

Baseline (already done, not a stage): current `test/torch-test.py` under the hook, `AMDFQ_POOL=0`.
PASS.

Implementation: flags on `test/torch-test.py`. A stage that is off must not change the baseline
path. Names below are the intended flags; the first edit is stage 2, then stage 3. Later stages
are listed so the stopping rule is defined, not so they get written up front.

| Stage | Flag (intended) | One added dimension | HIP difference it targets |
| --- | --- | --- | --- |
| 2 | `--driver-frees` | Make frees hit the driver: `PYTORCH_NO_HIP_MEMORY_CACHING=1`, or churn that actually produces ~300 `hipFree` per iteration | Count of `hipFree`, not the opcode |
| 3 | `--linear` | Replace `torch.mm` with `F.linear` (transposed-weight GEMM) | The Tensile path the trainer actually uses |
| 4 | `--conv` | `conv2d` + backward, small NCHW, bf16 autocast | MIOpen / `hipSetDevice` |
| 5 | `--sdpa` | `scaled_dot_product_attention` + backward | AOTriton / `hipModuleLaunchKernel` |
| 6 | `--checkpoint` | Small `nn.Module` + `torch.utils.checkpoint` | Activation free/recompute overlapping `hipFree` |
| 7 | `--lora` | Tiny CLIP + PEFT LoRA backward (the shape already in `test/test_vram_gpu.py`) | LoRA GEMM (`aten.mm` with `M = network_dim`). Unhooked arm must not use a known over-read K |
| 8 | (combine 3–6) | One mini block: conv + SDPA + linear + checkpoint | Combination, only after singles all pass |
| 9 | `--unet` | Real SDXL UNet (or one down block) dummy latent, forward+backward, no DataLoader | Full kernel set without the data pipeline |
| 10 | `--optim` | Dual optimizer step | Optimizer-state alloc/free |
| 11 | `--workers` | DataLoader workers | Extra processes / HSA queues. Last, because fork muddies hook logs |

**Stage 1 (`PYTORCH_CUDA_ALLOC_CONF` only) is skipped as a named rung.** It is almost certainly a
PASS and is folded into the environment of stage 2 rather than measured on its own. If stage 2 is
implemented by setting `PYTORCH_NO_HIP_MEMORY_CACHING=1` inside the process, the launcher conf is
irrelevant for that arm anyway.

**Do not start at stage 9.** A pass would not say which dimension mattered; a crash would be too
wide to instrument.

A cheap probe that is **not** this ladder: run existing `test/test_vram_gpu.py` and
`test/test_masked_loss_gpu.py` under the hook with `AMDFQ_POOL=0`. A pass would skip "a small
module with PEFT+checkpoint is already enough"; a crash would shorten the ladder. Not scheduled
until asked.

---

## 6. How a stage is run

Hooked:

```bash
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test --driver-frees
# later:
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test --driver-frees --linear
```

Unhooked: the same python invocation, no `LD_PRELOAD`, same flags.

`run.sh test` does not currently export `start_train.sh`'s `PYTORCH_CUDA_ALLOC_CONF`. Stage 2's
flag is the thing that changes allocator behaviour; do not silently start inheriting the trainer
env from `run.sh` as a side effect of this work.

Log: the hook already tees stderr to `$AMDFQ_LOG_FILE` (default `/tmp/amdfq-rs-test.<pid>.log`).
Keep the log for any failing repeat. `/tmp` is volatile — anything that will be cited later is
copied into this file or a sibling under `conclusions/`.

---

## 7. Stopping rule

Walk the ladder in order. After each stage:

- Both arms PASS → write that fact here (or in a follow-up), implement the next flag, stop at the
  end of that step.
- Hooked HIT, unhooked PASS → **stop adding stages.** The new flag is the delta. The next piece of
  work is instrumentation of that flag's new HIP traffic (entry points, kernel names, pointers
  still live after `hipFree`), not stage N+1.
- Unhooked MEMORY_FAULT → change shapes, same stage, do not advance.
- Either arm OOM → shrink, same stage.

Stage 2 then 3 is the only sequence that is scheduled. Stages 4–11 exist so that "what would come
next" is not invented under a crash; they are not a commitment to write them.

---

## 8. What stage 2 and stage 3 will actually change in the script

**Stage 2 (`--driver-frees`).** Today `churn()` does 96 allocations and `empty_cache()` per
iteration, and the driver sees 12 `hipMalloc` + 12 `hipFree` (`hip-runtime-calls.md` §12 point 5).
The flag must make the driver see a free rate in the same order as a training step (~339
`hipFree`), without adding new opcodes. Preferred mechanism: set
`PYTORCH_NO_HIP_MEMORY_CACHING=1` at the start of `main` when the flag is on, so every torch free
becomes a `hipFree`. Alternative if that OOMs the 3.5 GiB plateau: keep the cache and force
distinct sizes / `empty_cache()` until the hook log shows hundreds of `hipFree` per iteration.
The rest of the script, including `verify()`, stays. Report the observed `hipFree` count from the
hook log; if it did not move, the stage did not happen.

**Stage 3 (`--linear`).** Where the stress loop (and the matching `verify()` references) use
`torch.mm(x, w)`, use `F.linear(x, w)` instead — i.e. `x @ w.T`. First iteration is allowed to be
slow (the ~1.6 s kernel load the comment warns about). Do not add conv, attention, or
checkpointing in the same edit. `verify()`'s analytic grads have to follow the transposed
multiply.

Stage 2 remains in force when stage 3 is on, so a `--linear` crash is "driver-visible frees **and**
transposed GEMM" unless a `--linear` without `--driver-frees` control is added later. The first
stage-3 runs therefore include `--driver-frees`. If that combination hits, the extra control
(linear only, caching allocator on) is the way to split the two dimensions; it is not done
speculatively.

---

## 9. Stage 2 result (2026-09-19): hooked HIT, stop the ladder

`--driver-frees` is enough. The script under the hook with `AMDFQ_POOL=0` fails as soon as every
torch free is a real `hipFree`. The same flag without the hook passes. Hooked without the flag
still passes. Stage 3 (`--linear`) was not run. Stdout of the repeats: `torch-test-ladder/`.

**Commands.** Hooked: `AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test --driver-frees`. Unhooked:
`python -u test/torch-test.py --driver-frees`. Control: `AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test`
(no flag). Env `axl`, GPU idle (~76 MiB VRAM) before and after.

**What the flag actually did.** `PYTORCH_NO_HIP_MEMORY_CACHING=1` is set before `import torch`.
Caching-allocator counters then stay at 0, so the plateau/peak checks size the hold from the live
tensors instead of `memory_allocated()` — that is a script adjustment, not a hook finding. The
driver-visible free count is the stage's own check that it happened:

| Arm | n | exit | `hipFree` in hook log | `verify()` / checks |
| --- | ---: | --- | ---: | --- |
| unhooked `--driver-frees` | 2 | 0, 0 | — | PASS (2.07 s, 2.11 s) |
| hooked, no flag (baseline) | 1 | 0 | **104** | PASS (2.07 s) |
| hooked `--driver-frees` r1 | 1 | 1 | **3068** | FAIL, process lived |
| hooked `--driver-frees` r2 | 1 | 1 | **3065** | FAIL, process lived |
| hooked `--driver-frees` r3 | 1 | **134** (SIGABRT) | **2598** (truncated) | FAIL in `checks()`, then GPU fault |

~30× more `hipFree` than the baseline. Zero `pooled` lines (`AMDFQ_POOL=0`). r1–r3 all `released`.

**How it failed (hooked, `--driver-frees`).** Not a Tensile `HSA_STATUS_ERROR_MEMORY_FAULT` of the
training shape — the unhooked arm of the same flag is clean, and the first broken check is a 512×512
bf16 `torch.mm` inside `checks()`, before the stress loop.

- r1, r2, r3 all fail `matmul: bf16 vs cpu fp64` with the **same** `max abs err 2377.967` (unhooked
  and hooked-baseline: 0.250). Measured, three witnesses.
- r1 also: `d/dx x^3`, plateau-slice round trip, `w2.grad` / `w1.grad` vs chain rule, churn sum
  `nan`. Process exited 1 after `verify()`.
- r2 also: plateau sweep `observed [-5.406, 5.031]` against 0.519531, layer-1 split-K
  `1.82e+34`, `w3.grad` `9.82e+12`, `w2`/`w1` grads `nan`, 8 MiB hops not bit-identical. Loss of
  the chain still matched its own fp64 accumulation and was identical across the five iterations.
- r3 aborted during plateau construction after the same four `checks()` failures:
  `Memory access fault by GPU node-1 … on address 0x51bb3000. Reason: Page not present or supervisor privilege.`
  then `Aborted (core dumped)`. That is a page-not-present on a GPU access, not a host SIGSEGV.

**What this does and does not say.**

- Measured: making every torch free a `hipFree`, with the hook tearing the mapping down immediately,
  is sufficient to corrupt (and sometimes fault) a workload that is otherwise clean under the same
  hook. The script's own tensors are observably wrong; this is not a silent UAF in padding the
  script never reads.
- Measured: `hipFree` volume, not a new opcode. `--linear` / conv / SDPA were not in these runs.
- Not established: *which* live pointer the teardown is stepping on. The identical 2377.967 on the
  first bf16 matmul is a place to start instrumentation; it is not yet a cause.
- Inference (not a verdict): the user of the freed memory can be this process's own subsequent
  kernels, not only some trainer-only library. That is the reading the ratchet was built to test;
  the next work is to instrument the `--driver-frees` path (the new `hipFree`s vs the 512×512
  matmul's operands), not to walk the rest of the ladder.

`--linear` is already parsed in `test/torch-test.py` but was never passed. Leave it; do not run it
until this delta is instrumented.

---

## 10. Subtraction (2026-09-19): last PASS is 90×90, first FAIL is 91×91

Instrumentation was not started. Two things first: (1) `PYTORCH_NO_HIP_MEMORY_CACHING=0` under the
hook restores PASS; (2) with caching still off, the script is cut down until it stops failing. The
minimal crashing command is a single square `torch.mm` of size 91. Stdout: `torch-test-ladder/`.

**Caching on, hook on, full script.**

```
PYTORCH_NO_HIP_MEMORY_CACHING=0 AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test
```

`caching_allocator enabled=True`, env `'0'` (torch treats `0` like unset; `1` is the off switch —
probed separately). PASS, 2.05 s, 103 `hipFree`. That is the requested confirmation.

**Drop the plateau.** `--driver-frees --no-stress` still FAILs. First broken check is still the
512×512 bf16 `mm`, same `max abs err 2377.967`. The 3.5 GiB hold, churn, transfers and `verify()`
are not required.

**Slice `checks()`.** `--max-checks 3` (CPU elem + two scans) PASSes. `--skip-checks 3 --max-checks 1`
(the bf16 `mm` alone) FAILs (`max abs err 2441.124`). Prefix frees from the scans are not required.
Unhooked, same `mm` alone, `--driver-frees`: PASS, err 0.250.

**Shrink the `mm`.** All hooked, `--driver-frees --no-stress --skip-checks 3 --max-checks 1`, seed 0:

| k | dtype | n | result | max abs err |
| ---: | --- | ---: | --- | ---: |
| 1, 8, 16, 64 | bf16 | 1 / 1 / 1 / 3 | PASS | 0.000–0.105 |
| 65, 68, 72, 80, 88 | bf16 | 1 each | PASS | ≤0.103 |
| **90** | bf16 | **3** | **PASS** | **0.118, 0.118, 0.118** |
| **90** | fp32 | 1 | PASS | 0.000 |
| **91** | bf16 | **3** | **FAIL** | **339.957, nan, 37.250** |
| **91** | fp32 | 1 | FAIL | 339.812 |
| 92 | bf16 | 1 hooked FAIL (357.095); 1 unhooked PASS (0.095) | | |
| 94, 95, 96 | bf16 | 1 / 1 / 3 | FAIL | 403 / 413 / 361.960, nan, 361.960 |
| 128, 160, 192, 224, 256 | bf16 | 1 / 1 / 1 / 1 / 3 | FAIL | 528 … **1151.451 ×3** |
| 512 | bf16 | 2 | FAIL | 2441.124, 2441.124 |
| 512 | fp32, fp16 | 1 each | FAIL | 2439.370, 2439.494 |
| 256 | fp32 | 1 | FAIL | 1151.611 |
| 64 | fp32 | 1 | PASS | 0.000 |

Results are finite except when noted `nan`. The GPU does not have to page-fault for the numbers to
be wrong.

**The pair to instrument (not started).**

Last PASS:

```
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 90
```

First FAIL (minimal crashing version found):

```
AMDFQ_POOL=0 bash amdfq/amdfq-vmm-rs/run.sh test --driver-frees --no-stress --skip-checks 3 --max-checks 1 --mm-k 91
```

Same without `LD_PRELOAD` is PASS at k=92 (and at k=512). Caching on (`=0`) is PASS for the full
script.

**What this does and does not say.**

- Measured: the hook + immediate teardown + uncached allocs is enough to corrupt a 91×91 `mm`; a
  90×90 `mm` on the same path is not. Dtype is not the cut (fp32 matches bf16 at 90/91).
- Measured: the result is usually finite and, at a fixed k≥96, often bit-identical across repeats
  (256 → 1151.451 three times, 512 → 2441.124 twice, 90 → 0.118 three times). k=91 is not stable
  in the error value (339 / nan / 37).
- Not established: why 91 and not 90 (kernel choice, tile, mapping size, …). That is inference
  until the two commands are instrumented.
- Not established: that this 91×91 failure is the same mechanism as the trainer's NaN / illegal
  instruction. It is the smallest script that reproduces *a* hooked `--driver-frees` failure.

---

## 11. k=91 instrumentation (2026-09-19): not a freed pointer at launch

Launch gates were added to the hook (`hipModuleGetFunction`, `hipExtModuleLaunchKernel`,
`hipModuleLaunchKernel`, `hipLaunchKernel`) and `registry::locate` classifies a kernel argument as
`caller` / `slack` / `pad` of a still-live mapping. The mm check prints `m1`/`m2`/`out` VAs.
`test.sh` still passes. Extract: `torch-test-ladder/k91-instrument.txt`.

Three hooked runs of `--no-stress --skip-checks 3 --max-checks 1 --mm-k {90|91}`, `AMDFQ_POOL=0`:

| | k=90, uncached (`--driver-frees`) | k=91, uncached | k=91, cached (no flag) |
| --- | --- | --- | --- |
| result | PASS, err 0.118 | FAIL, err 37.250, finite | PASS, err 0.117 |
| kernel | same `Cijk_Ailk_Bljk_…_MT16x16x32_…_WG16_2_1` | same | same |
| launch | `items=1152,1,1 wg=32,1,1` | same | same |
| extra M,N,K,lda | `0x5a` = 90 | `0x5b` = 91 | `0x5b` = 91 |
| hipMalloc for the three matrices | 16200 × 3, each its own 2 MiB VA | 16562 × 3, each its own 2 MiB VA | **one** 2 MiB block; m1/m2/out at off 0 / 16896 / 33792 |
| extra VAs at launch | all four (C=D, B, A) `off=0 caller` of their own block | same | all four `caller` of the **same** block |
| pad / slack hits in extra | none | none | none |
| `hipFree` of A/B/C | after the mm and after the `.double().cpu()` copies | same | not in this window (cache holds the 2 MiB) |

`1152 / 32 = 36` workgroups = 6×6 tiles of MT 16, i.e. a 96×96 covered area for both 90 and 91.

**Measured.**

- The GEMM is not launched with a pointer that has already been `hipFree`d. The four device pointers
  in the 140-byte UserArgs blob are live, at offset 0 of the bytes the caller asked for.
- k=90 and k=91 uncached fire the **same** Tensile kernel with the **same** grid; the extra blob
  differs only in the size fields (90 vs 91) and the three VAs. The sizes are the sizes Python asked
  for.
- k=91 **passes** under the same hook, same kernel, same grid, same 91s in extra, when the caching
  allocator packs A, B and C into one 2 MiB VMM mapping instead of three.

**Not established (inference only).** Why three separate VMM mappings of a 91×91 fail and one packed
mapping of the same 91×91 does not. A kernel tail that over-reads into per-block slack (uncached:
possibly uninitialised; packed: 334 bytes of alignment padding then the next tensor) would fit the
layout difference; it has not been shown. This also has not been shown to be the trainer's NaN /
illegal-instruction path — only that the smallest `--driver-frees` failure is **not** "a `hipFree`
then a kernel using that VA".

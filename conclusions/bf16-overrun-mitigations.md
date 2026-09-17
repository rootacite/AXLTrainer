# Temporary mitigations for the gfx1201 bf16 over-read

**What this is.** The measures that keep a training run alive on this box despite the defect
[`bf16-kernel-overrun.md`](bf16-kernel-overrun.md) establishes: what each one changes, the measurement
behind it, its cost, and what it leaves unaddressed. Not one of them is a fix, and not one of them stops
the read — they attack the *consequence* (does the read meet an unmapped page?), and that is only
defensible because the read itself has been measured inert (§1).

**Where the numbers come from.** The two environment measures were A/B'd in `fixes/fix2`'s campaign on
`torch 2.13.0+rocm10.0.0` / HIP `7.15.26333` — the stack where the packaged reproducer dies on demand in
~25 s (`fixes/fix2/README.md` header), and the stack `environment.yml` names again as of 2026-09-17. It
had been on `2.12.0+rocm7.14.1` since at least 2026-09-15, where those same configurations do not
reproduce, so there was nothing to A/B there. The mechanism
is stack-independent — the over-read lives in the library's precompiled code object, which `fixes/hip1`
faults on this machine with guard-paged operands the harness manufactures itself
(`fixes/hip1/source-trace.md` §3) — which is why the measures carry over even though their numbers come
from elsewhere.

---

## 1. Why a measure that only hides the abort is defensible here

Hiding a memory fault is normally the wrong move: it converts a loud failure into silent corruption.
It is defensible **for this defect** because the read has been measured to have no effect on any number,
three ways:

1. **The numbers do not move when the over-read is allowed to happen.** `fixes/fix2` ran the same
   configuration and seed under a surviving layout that keeps over-reading (guard pages off) against one
   where the operands never end at a hole (`PYTORCH_NO_HIP_MEMORY_CACHING=1`): dim 32 → all 12 losses
   byte-identical and all 2208 LoRA tensors byte-identical; the canonical configuration (seed
   1145141920, dim 36, dies at step 101 without the knob) → **120/120 steps**, 120/120 losses
   byte-identical, and all 2208 tensors at steps 30/60/90/120/final byte-identical. The run that dies and
   the run that survives also agree byte for byte on every step they share.
2. **The loaded values are discarded inside the kernel, by construction.** The tail loop compares
   against the K bound (`v_cmp_ge_i32 s66, v149, s12`) and replaces the loaded registers with zero
   (`v_cndmask_b32_e64 …, 0, s66`), and the generator pre-zeroes that destination for exactly this reason
   (`set to zero to avoid unexpected value`, `KernelWriterAssembly.py:10667-10668`). The address guard
   (`v_min_i32`, `"truncated load: clamp GRO to legal range"`) caps a *start* offset; the read then
   overshoots it by at most one 16-byte access. Bytes read past the operand end can therefore only land
   in registers that are zeroed before use — `fixes/hip1/source-trace.md` §3, `bf16-kernel-overrun.md`
   §3.2.
3. **The overshoot is one page, not a wild pointer.** Measured in `(0, 4096]` bytes past the operand,
   with the kernel's own K-tail arithmetic giving 2048 B for the shape that killed the run
   (`bf16-kernel-overrun.md` §2.4). The killed run had also completed 210 steps on the same shape before
   the allocator happened to place that operand next to a hole.

What the guard page contributes is a **process kill** — the read is the same either way. Removing the
kill restores the behaviour the over-read has everywhere else on this card.

**The caveat that keeps this honest.** Points 1–3 are measured for the kernels those runs executed, at
the level of "the whole run's numbers and tensors", not for all 1830 launchable bf16 kernels (M4 is
still open, `bf16-kernel-overrun.md` §3.4). A knob that removes the fault *detector* also removes it for
any other out-of-bounds access in the same process, including one that would genuinely corrupt
something. That is the real price of §2.1, and it is a price paid in *unknowns*, not in measured damage.

---

## 2. The measures

| # | Measure | Measured effect | Cost | What it leaves |
| --- | --- | --- | --- | --- |
| 2.1 | `HSA_SVM_GUARD_PAGES=0` | fault → 12/12 steps; 120/120 at full scale, numbers byte-identical | none recorded | the read; the fault detector, process-wide |
| 2.2 | `PYTORCH_NO_HIP_MEMORY_CACHING=1` | same 12/12 and 120/120, numbers byte-identical | ~2.2–2.9× slower steps (2.9 s/it vs 1.12 s/it) | the read; allocator behaviour everywhere |
| 2.3 | Retry on the fault signature + checkpoint cadence | how `test/verify_mask_pipeline.py` already survives it | one step of work, when it fires | nothing about the cause |
| 2.4 | *(not a measure)* rolling the stack back to `torch 2.12.0+rocm7.14.1` | falsified: `fixes/fix3` aborts on it | — | — |

### 2.1 `HSA_SVM_GUARD_PAGES=0`

AMD's own reference describes the variable as *"Controls the use of guard pages in Shared Virtual Memory
(SVM) allocations"*, default **1**, with *"0: Disable SVM guard pages (for debugging memory access
patterns)"* (packaged in `fixes/fix2/resources/rocm-10.0.0-environment-variables.pdf`). With the default,
an SVM allocation is followed by a reserved unmapped page; the over-read meets it and the process dies.
With `0` the reservation is gone.

| run | result |
| --- | --- |
| dim 32, baseline (`fixes/fix2`'s env table) | FAULT at step 4 |
| dim 32, `HSA_SVM_GUARD_PAGES=0` | ok, **12/12 steps** |
| canonical config (seed 1145141920, dim 36), guard pages off | **120/120 steps**, byte-identical to the `no-hip` run (§1.1) |

Of the six ROCm environment variables `fixes/fix2` tested, this is the only one that changed the
outcome; the rest are in §3.

- **How to set it.** It belongs in the environment of the training process, so that the runtime sees it at
  initialisation: the shell that runs `bash start_train.sh`, `start_train.sh` itself, or the environment
  Ranko runs in (a launched run inherits it). `fixes/fix2/repro_real.py --env HSA_SVM_GUARD_PAGES=0` is
  how the measurements passed it to the child.
- **Cost.** `fixes/fix3` §8's table records "none"; no timing pair was taken, so that means "nothing was
  observed", not "measured equal".
- **What it costs you.** It removes the detector for *every* SVM over-read in that process: this defect
  becomes invisible afterwards, and so does an unrelated one. Do not leave it on while investigating any
  other memory problem.
- **The guarantee is empirical, not structural.** The reserved page is gone; the read is not. An operand
  that ends at the end of a mapped region whose successor page is still unmapped can still fault. The
  sample behind "it works" is two runs totalling 132 steps, plus the whole-run byte comparisons.
- **It does not blind the verification harnesses.** `fixes/hip1/launch_kernels.hip` and its siblings
  manufacture the hole themselves — a VMM reservation (`hipMemAddressReserve` + `hipMemMap`) whose tail
  they deliberately leave unmapped (`bf16-kernel-overrun.md` §2.3) — so they keep detecting the read with
  this variable set. The variable governs ROCr's own SVM allocations, not a caller's reservation.

### 2.2 `PYTORCH_NO_HIP_MEMORY_CACHING=1`

Must be set **before importing torch** (it changes how the HIP caching allocator is built). Every tensor
gets its own `hipMalloc`, so no operand ends at the boundary of a cached segment.

- **Measured:** ok 12/12 steps at dim 32; 120/120 on the canonical configuration that otherwise dies at
  step 101 — with the same byte-identical losses and tensors as §1.1.
- **Cost:** ~2.9 s/it instead of 1.12 s/it on that configuration (`fixes/fix2` §Consequences); quoted as
  "~2.2–2.9× slower" in `fixes/fix3` §8 and in `doc/troubleshooting.md`.
- **Leaves:** the read, and a changed allocator everywhere in the process (footprint and fragmentation
  behaviour were not measured beyond the step time).
- **Not combined** with §2.1 in any measured row; they were tested separately, each against a faulting
  baseline.

### 2.3 Treat the fault as a restart event

This is what the repo already does, and it is the cheapest measure of the three because it costs nothing
while it does not fire.

- **Recognise it.** `HSA_STATUS_ERROR_MEMORY_FAULT` in the child's log, exit code `-6`, no Python
  traceback, `state.json` left at `training` with a dead PID, and DataLoader forkserver workers that
  outlive the process and must be reaped by matching the run's own working directory
  (`fixes/fix2/README.md` §Consequences). The dashboard then needs a Reset (`train_reset`) to clear the
  stale state — `doc/troubleshooting.md`.
- **Retry, then escalate.** `test/verify_mask_pipeline.py` is the worked example: a child that dies this
  way is relaunched, and from the second retry on it gets `PYTORCH_NO_HIP_MEMORY_CACHING=1`
  (`doc/mask-verification.md`).
- **Bound the loss with the checkpoint cadence.** The run that started this investigation
  (`lllj_20260916_062841`) died inside step 1352 with `lllj_s001350` on disk — two steps of work lost.
  `[training].resume_lora_path` restarts from such a checkpoint, weights only: step and epoch counters
  restart at 0 and no optimizer state is restored (`doc/training.md`, `AGENT.md` §5 Resume).

This whole measure assumes the fault is still *reported*. That is not guaranteed: the driver's fault
response can itself wedge, and then an over-read stalls the GPU instead of killing the process — no
marker to retry on, a live PID, and a status stuck at `training`
([`gfx1201-fault-response-wedge.md`](gfx1201-fault-response-wedge.md)).

### 2.4 What is *not* a measure: rolling the stack back

Running on `torch 2.12.0+rocm7.14.1` instead of `2.13.0+rocm10.0.0` is not a workaround. `fixes/fix3`
reproduced the abort on that stack with the
`config.toml` committed there (`network_dim = 48`, `train_batch_size = 2`), every run, in ~15 s, before
and after a host reboot — "on the pinned stack the trainer *does* abort, it just had not been pointed at
this configuration before" (`fixes/fix3/README.md` header; its §8 closing note says
`doc/troubleshooting.md` should stop claiming immunity). The stack changes *which* shapes and allocator
layouts lose the lottery, not whether these kernels read past their operands, and `environment.yml` is
back on `2.13.0+rocm10.0.0` as of 2026-09-17. Its real content is "the lottery has not been lost in this
configuration yet".

---

## 3. Ruled out by measurement

Recorded so nobody re-tries them. All rows are `fixes/fix2` unless stated.

| Idea | Verdict |
| --- | --- |
| `GPU_SINGLE_ALLOC_PERCENT=50`, `HIP_MEM_POOL_SUPPORT=1`, `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, `HSA_DISABLE_CACHE=1`, `AMD_LOG_LEVEL=4`/`AMD_LOG_MASK=0x60000` | FAULT at step 4, exactly as the baseline — no effect |
| `flush_memory_every_step` (`empty_cache` + gc before each step) | already on and does not help; same route |
| Avoid the rank / batch / chunk shapes that were seen to die | not a rule: the fatal set is a scatter wider than any window tested, and `network_dim = 32` (no partial K tile at all) faults anyway |
| `network_dim` 64, `train_batch_size` 1, `max_token_length` 75, `network_dropout` 0, `gradient_checkpointing_te = true` | survive their windows, each by moving the graph or the shapes — i.e. by moving the address, not the read; `fixes/fix3` §8 |
| `torch.accelerator.prepare()` → `.to(device)` | lets a dim-32 run finish, but drops Accelerate's autocast wrap, so it is not a numeric no-op; not applied in `trainer/` (`doc/troubleshooting.md`) |
| Samples at 256×256 | avoids the fault in `fixes/fix2`'s sampling-driven route and changes what the run does — diagnostic, not a workaround |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | not runnable here: the first step's loss is `NaN` and the trainer dies serialising it |
| A generator-side repair of the guard | measured not viable: candidate A (cap − one access width) and C (operand-relative bound) fix the trainer's shape and make 7 of 10 ladder shapes wrong; B (redirect above-cap reads) keeps every crc and still faults — `fixes/hip1/source-trace.md` §5.1 |
| The vendor's own selection-side rule (`AssertSummationElementMultiple = DepthU` for GLTr solutions) | honoured by the host (the selected kernel visibly changes) and does **not** stop the crash: 12/12 runs faulted, so the family has at least two over-readers — `fixes/hip1/source-trace.md` §5.2, `fixes/hip1/hip1.md` §13.2 |

One idea in neither column: pinning the prompt encoder's chunk count so the GEMM shape stops switching
(`target_num_chunks`), proposed in `fixes/fix2-ex.md` ("Candidate prevention (not yet tested)"). It
changes training semantics — short captions get padded to the pinned chunk count — and was never measured,
so it is mentioned only so it is not mistaken for one of the measures above.

---

## 4. What this leaves you with

A judgement, kept apart from the measurements above:

- **Normal operation: nothing.** Keep the shipped defaults (pinned stack, `bucket_reso_steps = 128`) and
  treat a fault as a restart event (§2.3). This is what the repo does today, and the last long run that
  died this way lost two steps.
- **When a run keeps dying at the same point and restarting is not acceptable:**
  `HSA_SVM_GUARD_PAGES=0` for that run. It is the only measure here with no measured cost, and §1's
  evidence is what makes it defensible despite hiding the abort. If the run still faults, fall back to
  `PYTORCH_NO_HIP_MEMORY_CACHING=1` at roughly 2.2–2.9× the step time.
- **Neither is a fix, and both were measured on the stack the pin names** (`2.13.0+rocm10.0.0`) — see
  the scope note at the top.
- **What would change this verdict:** the M4 population scan finding a kernel whose over-read is *used*
  rather than discarded, or a case where the discard in §1.2 does not hold (another epilogue or ISA
  path). Either would make "let the read happen" stop being an acceptable trade.
- **The proper fix is not trainer-side.** The code-level route is closed by measurement
  (`fixes/hip1/source-trace.md` §5); the open target is the family's second, non-DTV over-reader
  (`fixes/hip1/hip1.md` §13.3).

## 5. Where each number comes from

| Document | Holds |
| --- | --- |
| [`bf16-kernel-overrun.md`](bf16-kernel-overrun.md) | The defect itself: the read, its distance bound, the shape correlation, the population still unmeasured |
| `fixes/hip1/source-trace.md` | Which generator function emits the guarded read (§3), why the ISA assumption behind it fails (§4), why no code-level repair exists (§5) |
| `fixes/hip1/hip1.md` §13 | The paused patch/rebuild/deploy phase: candidate results, the ASEM rule's 12/12, the second over-reader, where to resume |
| `fixes/fix2/README.md` | The ROCm-10.0.0 campaign: the environment-variable table (§8), the byte-for-byte integrity comparisons, the fault's corpse (exit `-6`, dead PID, orphaned workers). Its "do not fix this with `HSA_SVM_GUARD_PAGES=0`" is the verdict §1.1's integrity result weighs against |
| `fixes/fix3/README.md` | The abort on the stack that was pinned until 2026-09-17 (header), and §8's workaround table that this document revises — the same verdict also stands in `fixes/fix2-ex.md` |
| `doc/troubleshooting.md`, `doc/mask-verification.md` | The user-facing workaround text and the retry/escalation automation |

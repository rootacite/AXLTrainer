# `fixes/fix2/` — minimal reproduction attempts for the gfx1201 page fault

Fastest crash that actually aborts (still the trainer process):

```
conda activate axl
python repro_real.py --row "network_dim 32"   # FAULT at step 4, ~25 s
```

`crash.py` is the same path rewritten without `trainer/` (diffusers + peft +
accelerate only). It hits the same `M=462, K=32` GEMMs and **survives** — the
standalone allocator layout never meets an unmapped page.

This directory answers one question: **how much of the training stack do you actually need to
reproduce the page fault described in `fixes/fix2.txt` and `fixes/fix2-ex.md`?**

The short answer, measured on this machine (torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, gfx1201):

- the `[462, 36] @ [36, 2048]` LoRA-backward shape on its own — and even the real SDXL UNet plus
  both CLIP text encoders at exactly that shape — is **not enough**;
- the short repro `fixes/fix2.txt` promises (`train_batch_size = 2` + `network_dim = 36`, dead on
  the first backward) **does not reproduce**: it runs a full epoch, including with every caption
  padded so that file's own derived shape (`seq 231`, `M = 462`, `K = 36`) is used on every step;
- what does reproduce, 100%, is the second sighting (`fixes/fix2-ex.md`): the unmodified
  `trainer/main.py` on the packaged config, dead at **step 101 of 120**, every time, including 6/6
  after a host reboot;
- that 101-step run can be cut to **8 steps (~70 s, or 36 s when the parent stops the child at the
  abort instead of waiting for its teardown)**: what the fault needs is the **sampling cadence**, not
  the number of training steps — `python repro_real.py --row "fast: cadence 2, cheap samples"`;
- a *faster, different* abort exists with two config changes: `network_dim` 12 / 24 / 32 dies at
  **step 4 in ~24 s**, in a plain training step with no sample involved (a different Tensile tile, so
  a sibling of the same bug rather than the same fault);
- give every tensor its own `hipMalloc` (`PYTORCH_NO_HIP_MEMORY_CACHING=1`) and **both** abort routes
  disappear — the caching allocator's layout is the ingredient they share;
- the same fault leaves **no numerical trace**: with one shared latent cache, the run that dies at
  step 101 and the run that survives 120 steps produce byte-identical losses and byte-identical LoRA
  tensors up to the abort;
- `crash.py` (no `trainer/` import) matching the dim-32 *work* does **not** fault. The trainer still
  faults at step 4 when its *step loop* is replaced by `crash.py`'s. The fatal difference is
  `accelerator.prepare()` of the UNet and both text encoders after this process's load/cache
  history — replacing that call with `.to(device)` lets the unmodified rest of `trainer/main.py`
  finish 12/12 steps. That is a measurement, not a shipped fix.

So the shape tells you *where* the Tensile kernel overruns; the surrounding allocation layout — a
real training step, at a particular step number, for a particular seed — decides whether that
overrun meets an unmapped page and aborts the process.

### Why `crash.py` (no `trainer/`) survives, measured

`crash.py` reproduces the *work* of the `network_dim 32` abort without importing the trainer:
same batch, dim, alpha, seed, `M = 462` for steps 1–4 then `M = 231`, and losses agreeing with the
trainer to ~1e-4 (`0.1498 / 0.1525 / 0.1633 / 0.1219` vs `0.1498 / 0.1524 / 0.1632 / 0.1219`; the
loss mask's mean weight is 0.9998, so weighting it changes nothing). It runs 8/8 steps alive.

What the two runs share and what they do not (`diff_probe.py`, `patch_mirror.py`):

| Removed from the faulting trainer | Result |
| --- | --- |
| DataLoader workers (`max_data_loader_n_workers = 0`) | still FAULT at step 4 |
| the in-process VAE pass (latent cache pre-populated) | still FAULT at step 4 |
| the loss mask (`batch["loss_mask"] = None` at the collate) | still FAULT at step 4 |
| the trainer's whole step body (replaced with `crash.py`'s loop) | still FAULT at step 4 |
| `Accelerator` created after the pipeline load | still FAULT at step 4 |
| `load_sdxl_pipeline` called with `crash.py`'s arguments | still FAULT at step 4 |
| the trainer's MIOpen solver switches (copied into `crash.py`) | still no fault |

The kernels are not the difference either: with `AMD_LOG_LEVEL=4 AMD_LOG_MASK=0x60000` on both,
the trainer and `crash.py` touch the **same 67 Tensile kernels**, including the faulting
`MT64x128x16` (9 lookups vs 8); `crash.py` has no kernel the trainer lacks.

What the allocator snapshot does show (`AXL_SNAPSHOT_DIR` + `--snapshot`, dumped at the top of each
step): the faulting address reported by KFD lands **exactly at the end of a 2 MiB caching-allocator
segment**, with the next segment starting 2 MiB later — the overrun crosses the segment boundary
into the unmapped hole. So the two processes differ in *which* segment a GEMM operand sits in at
step 4, not in the arithmetic.

### The fatal difference is `accelerator.prepare()` of UNet + TEs

`import trainer` is not the trigger. `python trainer/main.py` rewritten so `main()` only imports
and then runs `crash.child_main` survives 8/8. Putting `crash.py`'s forward into the trainer
process (row `true-crash` / `crash-body`) still dies at step 4. The arithmetic is the same; the
GPU mapping is not.

Trainer (`trainer/main.py` `_prepare_artifacts`):

```python
prepared = artifacts.accelerator.prepare(
    artifacts.modules.denoise,
    *artifacts.modules.text_encoders,
    artifacts.denoise_optimizer,
    artifacts.te_optimizer,
    artifacts.te_scheduler,
)
```

`crash.py` also calls `acc.prepare(unet, te1, te2, …)` and does **not** die. `Accelerator.prepare`
is not intrinsically poisonous — it is poisonous **after this process's load / VAE-cache /
flush history**, because that is when the weights land next to a 2 MiB hole.

`patch_mirror.py` (copies under `/tmp`, never the repo files):

| row | what changed | 12-step dim-32 result |
| --- | --- | --- |
| `control` | unmodified trainer | FAULT at step 4 |
| `true-crash` / `crash-body` | trainer process, `crash.py` step loop | FAULT at step 4 |
| `run-crash-child` | trainer cwd + imports, `crash.child_main` | no fault |
| `prepare-models` | `prepare` UNet + both TEs; optimizers untouched | FAULT at step 4 |
| `prepare-opts` | `prepare` optimizers only; models `.to(device)` | 12/12 ok |
| `no-prepare` | `.to(device)` for UNet + TEs, no `prepare` on them | 12/12 ok (twice) |
| `crash.py` / `--import-trainer` / `--accel-early` / `--pin` | standalone process | no fault |

Step 1 CUDA snapshots: **5735 `alloc` sizes match** between trainer and `crash.py`. They diverge
on HIP *segments* — around event 2001 the trainer is already `segment_free` (VAE workspace
returned) while `crash.py` is still growing encode workspace. Live tensors are ~7.2 GB both
sides; virtual span is ~16 GB (trainer) vs ~26 GB (`crash.py`). Fatality is whether the Tensile
overrun's next page is a neighbour or a hole.

`Accelerator.prepare_model` is more than `.to(device)`: it wraps `forward` with bf16 autocast
and `convert_outputs_to_fp32`. Dropping `prepare` therefore changes the numeric path. It is
**not** a fix to ship; it only locates the fatal moment at "models going onto the GPU", not at
the later step loop.

Do **not** "fix" this with `HSA_SVM_GUARD_PAGES=0`. That makes the same overrun miss the guard
page and continue. Integrity runs (guard-off vs `PYTORCH_NO_HIP_MEMORY_CACHING=1`) were
byte-identical on this box, but the option still converts a crash into a silent OOB read.

## What is in here

| File | What it does |
| --- | --- |
| `crash.py` | Self-contained dim-32 loop (no `trainer/` import). Same shapes as the step-4 abort; measured **no fault**. `--mask`, `--nonblocking`, `--tb`, `--pin`, `--accel-early`, `--import-trainer`, `--steps`, `--snapshot DIR`, `--env KEY=VALUE` layer trainer behaviours on one at a time. |
| `diff_probe.py` | Ablation ladder over the *config*: removes one trainer ingredient per row from the guaranteed fault. `--list`, `--row <substring>`. Writes `resources/diff-probe.json`. |
| `patch_mirror.py` | Ablation ladder over the *code*: copies the trainer into `/tmp` and replaces one file (never the repo). Rows include `control`, `crash-body`, `true-crash`, `prepare-models`, `prepare-opts`, `no-prepare`, `run-crash-child`. Writes `resources/patch-mirror.json`. |
| `repro_real.py` | **The trainer-side reproducer.** Runs the unmodified `trainer/main.py` on `resources/original-config.toml` and the packaged dataset, and prints the failing-vs-surviving table. `--grid table|mech|fast|dim|dim2|integrity|integrity2` picks the row set, `--row <exact|labels>` and `--only <substrings>` narrow it, `--env KEY=VALUE` adds child environment (ROCm variables), `--list` prints them, `--dry-run` writes the configs without starting anything. |
| `repro.py` + `trial.py` | Model-free. Sweeps PEFT LoRA GEMMs over `M`, `network_dim`, dtype, allocator perturbation, layer count and resident memory. One subprocess per configuration (the fault cannot be caught in-process). |
| `repro_unet.py` + `repro_unet_sweep.py` | Real SDXL UNet + LoRA from `trainer/config.toml`, synthetic cross-attention input that carries gradients — i.e. no CLIP, no VAE, no training loop. |
| `repro_joint.py` | Real UNet **and** both CLIP text encoders with LoRA, a real tokenized caption, joint backward — the case `fixes/fix2.txt` records as faulting. No VAE, no DataLoader, no optimizer. |
| `resources/dataset/` | The six kanae images+captions the faulting run trained on (5 MB). |
| `resources/original-config.toml` | The exact generated config of the run that faulted 6/6 times. |
| `resources/fault-child.log`, `resources/fault-dmesg.log` | The child-side KFD abort and the kernel-side `[gfxhub] page fault` records. |
| `resources/fault-child-after-reboot.log`, `resources/fault-dmesg-after-reboot.log` | The same fault after the host was rebooted, to rule out accumulated machine state. |
| `repro_min.py` | **Standalone.** `diffusers` + `peft` + `torch` only, no `trainer/` import: a real SDXL UNet with LoRA and a synthetic batch at the real shapes, with optional text encoders, checkpointing and optimizer. |
| `integrity.py` | Compares the `--grid integrity` runs byte for byte: per-step losses, LoRA tensors, sample PNGs. |
| `seed_step_shapes.py` | CPU-only walk of the trainer's sampler + caption shuffle: prints the per-step `M` trace for a seed and where two seeds diverge. |
| `merge_results.py` | Glues the `repro_real.py` passes into the single ordered table in `resources/real-results.json`. |
| `resources/*.json` | Raw output: `real-results.json` (the table below) plus the per-pass files it is merged from (`real-baseline-4of4.json`, `real-first.json`, `real-rest.json`), `real-mech*.json` / `real-fast*.json` (the meta-operation and speed grids), `real-dimscan*.json` (the rank scans), `real-seed919.json`, `real-integrity*.json` and `integrity2*.json` (the byte-level comparisons), `seed-shapes.json` (the per-step `M` traces), `minimal-results.json` / `explore.json` (level 1), `unet-sweep.json` (level 2), `env.json` (versions). |
| `resources/session-log.md` | Run matrix: which runs faulted, which survived, with kernel timestamps and PIDs. |
| `resources/rocm-10.0.0-environment-variables.pdf` | Official AMD ROCm 10.0.0 env-var reference (`HSA_SVM_GUARD_PAGES` default 1, etc.). |

Model weights are **referenced**, never copied: `repro_unet.py` / `repro_joint.py` read
`pretrained_model_name_or_path` from `trainer/config.toml` (or `--model`), and fail fast if the path
does not exist. Run everything with the `axl` conda env; `repro_unet*` and `repro_joint` can be started
from anywhere (they `chdir` to the repo root so `trainer/config.toml` resolves).

## Level 1 — synthetic LoRA GEMMs (`repro.py`)

```
python repro.py                       # default grid
python repro.py --quick               # one dtype, one mode
python repro.py --rows 462,231 --layers 70 --lived-gb 6 --alternate 1
```

| parameter | values | why |
| --- | --- | --- |
| `--rows` (M) | 154 / 231 / 308 / 462 / 693 | `train_batch_size * encoder_seq_len`. The four values the real trainer can produce are 154 (batch 2, 1 chunk), 231 (batch 3, 1 chunk), 308 (batch 2, 2 chunks) and 462 (batch 3, 2 chunks = 7×64+**14**, the partial tile `fix2.txt` pins the fault on, and also batch 2 with the 3 chunks that file derives it from) |
| `--ranks` (K) | 36 / 32 / 64 | `network_dim`; 36 = 1×32+4 |
| `--dtypes` | bfloat16 / float32 | the faulting kernel is a BF16 Tensile kernel |
| `--layers` | 1 / 70 | an SDXL UNet repeats this LoRA pattern dozens of times per step |
| `--lived-gb` | 0 / 6 | keeps memory resident so the allocator packs buffers like the real process |
| `--alternate` | 0 / 1 | alternates M each iteration, as the trainer did when the caption's CLIP chunk count changed |
| `--stress` | 0 / 1 | odd-sized live allocations, to shift the caching allocator's packing |

Observed: **0 of 40 configurations faulted** (`resources/minimal-results.json`), i.e. the full cross
product of M ∈ {154, 231, 308, 462, 693} × K ∈ {36, 32} × dtype ∈ {bf16, fp32} × allocator stress, all
through real PEFT LoRA adapters. A second grid that mimics the real process more closely — 70 stacked
LoRA layers, 6 GB kept resident, M alternating between 154 and 308 — is **0 of 12**
(`resources/explore.json`).

## Level 2 — real UNet, no CLIP (`repro_unet_sweep.py`)

```
python repro_unet_sweep.py                 # 7 batch/chunk combinations
python repro_unet_sweep.py --detach 1      # control: detached cross-attention input
```

`M = batch * 77 * chunks`, with the real 1024×768 bucket latents (96×128) and the UNet's LoRA
adapters installed, so the documented GEMM runs inside a real graph with gradient checkpointing on.

Observed: **0 of 7 configurations faulted** (`resources/unet-sweep.json`) — every (batch, chunks)
pair that cover the real range, M = 77 / 154 / 231 / 308 / 462 (twice, as batch 2×3 chunks and
batch 3×2 chunks) / 693, each in ~9 s.

## Level 3 — real UNet + both CLIP encoders (`repro_joint.py`)

```
python repro_joint.py --batch 3 --caption-index 2       # a real 2-chunk caption -> seq 154, M 462
python repro_joint.py --batch 6                          # default short caption -> seq 77, M 462
python repro_joint.py --batch 2 --prompt "$(cat resources/long-prompt-3chunks.txt)"   # 3 chunks -> seq 231, M 462
```

`repro_joint.py` loads both text encoders with LoRA, tokenizes a real caption (one to three CLIP
chunks), encodes it **with** gradients, and backpropagates through the joint TE+UNet graph — the
situation fix2 lists as `page fault` in its isolation table. Measured here: **ok** at M=231 (batch 3, seq 77: 2.85 s, 9.0 GB peak) and at M=462 (batch 6,
seq 77: 6.45 s, 10.9 GB peak).

## The `fix2.txt` triggering size does not reproduce on this box (measured)

`fixes/fix2.txt` pins its first sighting on one combo — `train_batch_size = 2` with
`network_dim = 36` — and derives it from `3 CLIP chunks -> encoder seq 231`:

    [B·231, 36] @ [36, 2048]      B = 2  ->  M = 462 = 7×64 + 14, K = 36 = 1×32 + 4

Both halves of that derivation are testable, and neither holds here
(torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, gfx1201, i.e. the version `fix2.txt` says faults too):

| row | what it runs | result |
| --- | --- | --- |
| `fix2.txt: batch 2, dim 36, kanae` | the file's combo, all 188 kanae images, `max_token_length = 225`, 1024×768 bucket | **ok**, 94 steps (1 epoch), 118 s, no fault at step 1 or later |
| `fix2.txt control: batch 3, dim 36, kanae` | same data, `train_batch_size = 3` | **ok**, 63 steps, 112 s |
| `fix2.txt: batch 2, dim 36, 3-chunk captions` | every caption padded past 150 tokens, so `seq = 231`, `M = 462`, `K = 36` on every step — the `fix2.txt` shape exactly | **ok**, 94 steps, 118 s |

Why `seq 231` is not reachable from this dataset without padding: the chunk size is
`tokenizer.model_max_length - 2 = 75`, and the token counts of the 188 kanae captions, measured with
the real SDXL `CLIPTokenizer`, top out at **145** tokens. Every caption therefore needs one or two
CLIP chunks (`seq 77` or `154`), so at `train_batch_size = 2` the LoRA GEMM runs with `M = 154` or
`M = 308` — never 462. Padding is what turns the file's stated `3 chunks` into an actual test, and
it survives too.

So on this machine the short repro `fix2.txt` promises (first backward, batch 2 + dim 36) is not
available: what still reproduces is the longer second sighting below.

## The shortest repro: 8 steps, ~70 seconds

```
python repro_real.py --only "cadence 2"
```

is the packaged config with one thing changed — `save_every_n_steps` 30 -> 2, so the sample
generation happens every other step instead of every 30 — plus 512x512 samples to keep it short.
It dies at **step 8 of 10** in 72 s wall clock (the 101-step row needs ~180 s), with the identical
fault: `-6` (SIGABRT), the same Tensile kernel
`Cijk_Ailk_Bjlk_..._MT32x32x128_..._ISA1201`, `GCVM_L2_PROTECTION_FAULT_STATUS:0x0080113B`, three
orphaned DataLoader workers.

Where those 72 seconds go: ~15 s loading the pipeline, ~8 s warm-caching the six latents, ~20 s for
8 training steps and 4 sample generations, then ~30 s of KFD/coredump teardown between the fault and
the process exit. The training itself is the small part, which is the whole point. With
`--kill-after-fault 3` the parent stops the child as soon as the abort is in its log instead of waiting
for the teardown, which brings the row down to **36 s** (the entry then records `exit=-9` plus
`killed_after_fault: true`; leave the flag off when the authentic `-6` matters). Measured: 72 s
natural, 36 s with the flag.

## Which meta-operation triggers it

Same seed and shapes throughout; each row moves exactly one knob. Every row that died did so in the
same kernel.

| row | death |
| --- | --- |
| packaged cadence (30 steps) | **FAULT at step 101** (180 s) |
| cadence 30, 512x512 samples | **FAULT at step 90** — the 3rd sample itself |
| cadence 30, one denoise step per sample | **FAULT at step 90** — same step |
| cadence 10, 512x512 samples | **FAULT at step 31** — one step after the 3rd sample |
| cadence 5, 512x512 samples | **FAULT at step 20** — the 4th sample |
| **cadence 2, 512x512 samples** | **FAULT at step 8** — the 4th sample |
| cadence 3, 512x512 samples | ok (12 steps, 4 samples) |
| cadence 1, 512x512 samples, 24 steps | **FAULT at step 12** (an 8-step window was just too short) |
| cadence 1, packaged 768x768 samples, 8 steps | ok — window too short to tell |
| cadence 30, 256x256 one-step samples | ok, 120/120 steps — a sample that small poisons nothing |
| cadence off (`save_every_n_steps = 0`) | ok, 120/120 steps |
| no DataLoader workers (`max_data_loader_n_workers = 0`) | **FAULT at step 101** — the worker processes are not involved |
| `flush_memory_every_step = true` (`empty_cache` + gc every step) | **FAULT at step 101** — the cheap workaround does not work |
| gradient checkpointing off | not runnable — without it the 1024x768 UNet backward OOMs at step 0 on 16 GB |
| `cache_latents = false` | not runnable — the trainer's on-demand VAE encode raises `Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)` (the VAE is left on CPU; a separate defect, unrelated to this fault) |
| `PYTORCH_NO_HIP_MEMORY_CACHING=1` | ok, 120/120 steps |

What that says:

- **The cadence event is necessary.** Turn it off and 120 steps of the identical shapes, seed and
  optimizer run clean. This is the only place in the loop where large blocks are handed back and
  re-allocated mid-run (the models go to CPU, the VAE comes up, the sample is decoded).
- **How much work the sample does is irrelevant.** Eight denoise steps and a single denoise step both
  die at step 90 of a 30-step cadence, so it is the offload/reload plus the VAE pass that matters,
  not the denoising.
- **Cramming the cadence forward is what makes it fast.** For the cadences that do die, the death
  lands on or just after the 3rd or 4th sample, whichever the cadence reaches first — cadence 2 dies
  at step 8, cadence 5 at step 20, cadence 10 at step 31, cadence 30 at step 90/101.
- **But it is not a plain event count.** Cadence 3 reaches its 4th sample at step 12 and survives,
  while cadence 1 — a sample after *every* step — dies at step 12 of a 24-step run (an 8-step window
  was simply too short, which is how that row was first mis-measured). What decides is the pairing of
  a sample pass with the shapes and the allocator state around it: a given configuration is
  reproducible (cadence 2 always dies at step 8, cadence 5 always at step 20) but the death step
  cannot be derived from the cadence alone.
- **Smaller samples move the failure earlier, not away.** 512x512 samples die at the 3rd sample
  itself; the packaged 768x768 samples survive that same sample and die 11 steps later, inside a
  training step. Where the overrun lands moves with the allocation footprint; the fault does not.

## A faster sibling: `network_dim` 24 or 32 dies at step 4 (24 s)

`repro_real.py --grid dim` is the packaged run with the LoRA rank changed and the run cut to 12
steps:

| row | kernel | death |
| --- | --- | --- |
| `network_dim 12` | `..._MT64x128x16_MI16x16x1_..._ISA1201` | **FAULT at step 4**, 24 s total |
| `network_dim 24` | same kernel | **FAULT at step 4**, 24 s total |
| `network_dim 32` | same kernel | **FAULT at step 4**, 24 s total (3 of 3 runs, at 36 s / 24 s / 24 s) |
| `network_dim` 8 / 16 / 20 / 28 / 36 / 40 / 44 / 48 / 56 / 64 | — | ok over 12 steps |
| `network_dim` 24 / 32 **with `no-hip`** | — | **ok over 12 steps** |

The scan that produced this (`--grid dim`, 12 steps each) picks the same `MT64x128x16` kernel for
K = 24 and K = 32 and dies in both, before the epoch-2 batch is half done. Since 12 steps contain no
sample (the first is at step 30), this death has nothing to do with the sampling path.

That is the fastest abort in this directory: `M = 462`, `K = 24` or `32`, no sample involved, and no
partial `K` tile at all (32 is exactly one 32-wide tile) — which also puts a dent in `fix2.txt`'s
"partial tile" explanation. Whatever `K = 12 / 24 / 32` changes, it is small enough to be reproducible
inside four steps of a fresh process.

The `no-hip` rows are the interesting part: giving every tensor its own `hipMalloc`
(`PYTORCH_NO_HIP_MEMORY_CACHING=1`) carries **K = 24 and K = 32 through the same twelve steps**. So the
allocator layout is the one ingredient both routes share — the overrun is a property of these Tensile
kernels, but whether it reaches an unmapped page is a property of the caching allocator, for the
training-path deaths exactly as for the cadence-triggered one. It is, however, a
**different kernel** from the canonical `MT32x32x128`, so it is a sibling of the same ROCm bug class
rather than the same fault; the cadence-2 row above stays the fast version of *this* fault.

## A standalone repro: `repro_min.py`

`repro_min.py` imports nothing from `trainer/`. It is `diffusers` + `peft` + `torch`: a real SDXL UNet
with LoRA adapters, a synthetic batch at the shapes the trainer uses (`M = batch * 77 * chunks`,
`K = network_dim`, `N = 2048`, bf16), and a loop of forward/backward steps with optional gradient
checkpointing, optimizer and text-encoder LoRAs.

```
python repro_min.py                # dim 32, batch 3 x 2 chunks: the rank that aborts at step 4
python repro_min.py --te 1         # put both CLIP encoders in the backward as well
python repro_min.py --dim 36      # the rank that survives in the trainer until step 101
```

It redirects its own file descriptor 2 to `repro_min.stderr.log` before doing anything, because the
KFD abort and the HIP warning are written by the runtime itself: the process can be gone before Python
flushes, and the kernel name plus faulting address are exactly the evidence worth keeping.

**Outcome: it does not reproduce the fault.** Every combination tried runs to completion:

| variant | M | K | steps | peak | result |
| --- | --- | --- | --- | --- | --- |
| UNet + LoRA | 462 | 32 | 8 | 6.9 GB | ok |
| UNet + LoRA | 462 | 12 | 8 | 6.8 GB | ok |
| UNet + LoRA + both text encoders (joint backward) | 462 | 32 | 8 | 8.8 GB | ok |
| … + the trainer's schedule-free optimizer | 462 | 32 | 30 | 8.9 GB | ok |
| UNet + LoRA + schedule-free | 462 | 12 | 30 | 6.8 GB | ok |

Two ranks that abort at step 4 *inside the trainer* (`network_dim` 12 and 32) are therefore fine in a
standalone process with the same shapes, the same dtypes, the same LoRA targets, the same joint
text-encoder graph and the same optimizer. What the trainer adds around the step — the DataLoader and
its workers, the latent cache built by the VAE, the sampler's per-step batch, the run directory and
state machine, the TB writer, and the sampling pass — is where the difference has to live. That is a
useful negative: the fault is a property of the *process's allocation history*, not of the step.

(Two bugs were fixed on the way and are worth knowing for anyone extending this file: gradient
checkpointing must stay on or the 1024x768 UNet backward allocates past 16 GB, and the text encoders
have to be used the SDXL way — concatenated *penultimate* hidden states, not `last_hidden_state`, which
makes the UNet's softmax overflow to NaN in bf16.)

## ROCm 10.0.0 environment variables that matter here

AMD's `ROCm environment variables` reference for 10.0.0 matches this box's stack
(`torch 2.13.0+rocm10.0.0`, HIP `7.15.26333`). Four groups are relevant:

**Consistent with the mechanism.** `HSA_SVM_GUARD_PAGES` defaults to **1**, so SVM allocations carry
guard pages: an overrun past a buffer faults *when the page after it is unmapped*, which is exactly what
every captured entry shows. It also predicts the dichotomy we measure — when the caching allocator carves
the buffer out of a larger mapped segment, the same overrun reads mapped memory and nothing happens.
`RW: 0x0` (a read) is in every entry, which is why the integrity section compares *numbers* rather than
looking for damaged memory.

**Diagnostics for the layer we could not observe.** `AMD_LOG_LEVEL=4` with `AMD_LOG_MASK=0x20000`
(memory allocation) and `0x40000` (memory pool allocation) makes HIP log allocation and pool activity —
a way to see the allocator layout around the abort. `GPU_DUMP_CODE_OBJECT=1` dumps the code objects, i.e.
the Tensile kernel itself. `HSAKMT_DEBUG_LEVEL=6/7` raises driver (libhsakmt) verbosity.
`HSA_DISABLE_FRAGMENT_ALLOCATOR=1` turns off ROCr's internal fragment cache, which the reference
describes as helping "debug tools identify memory faults at their origin by preventing cached memory
blocks from masking out-of-bounds writes".

**Other allocator layers.** `GPU_SINGLE_ALLOC_PERCENT` (100), `HIP_INITIAL_DM_SIZE` (8 MB),
`HIP_MEM_POOL_SUPPORT` (0), `HIP_VMEM_MANAGE_SUPPORT` (1) sit below PyTorch's caching allocator, so they
are different levers from `PYTORCH_NO_HIP_MEMORY_CACHING=1`.

**Next to the kernel line we saw once.** `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM` (default 1) with
`HSA_SCRATCH_SINGLE_LIMIT` / `HSA_NO_SCRATCH_RECLAIM`: scratch memory is assigned per queue, and when an
allocation fails ROCr reclaims scratch across queues and retries. The kernel printed `[drm] *ERROR* Not
enough memory for command submission!` during the rank scans, which is that neighbourhood.

`repro_real.py --env KEY=VALUE` (repeatable) passes these to the child. The `network_dim 32` baseline
dies at step 4 in ~24 s, so each variable costs under a minute to test:

| variable | result (12-step window, same shared latent cache) |
| --- | --- |
| *(nothing)* | FAULT at step 4 |
| **`HSA_SVM_GUARD_PAGES=0`** | **ok, 12/12 steps** |
| `GPU_SINGLE_ALLOC_PERCENT=50` | FAULT at step 4 |
| `HIP_MEM_POOL_SUPPORT=1` | FAULT at step 4 |
| `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` | FAULT at step 4 |
| `HSA_DISABLE_CACHE=1` | FAULT at step 4 |
| `AMD_LOG_LEVEL=4`, `AMD_LOG_MASK=0x60000` | FAULT at step 4 — log kept in `resources/hip-allocation-log.txt` |
| `PYTORCH_NO_HIP_MEMORY_CACHING=1` (PyTorch, for contrast) | ok, 12/12 steps |

That log is worth reading: the runtime spends the last seconds before the abort in a burst of
`Cannot find the function: Cijk_...` lookups — 320 of them over 81 distinct Tensile kernel names, the
last one **0.39 ms** before the `Memory Fault Error`. The crash therefore coincides with a
kernel-selection burst, which fits "the falling kernel had just been picked". Whether those misses are
normal candidate probing or a real lookup failure is not settled; `AMD_LOG_MASK` bits `0x2` (kernel
commands), `0x10` (queue contents) and `0x80` (kernel creation) would show the kernel actually launched.

So the abort is the overrun landing on an **SVM guard page**, and only two things change the outcome:
the guard page (does the overrun meet unmapped memory?) and PyTorch's caching allocator (where do the
buffers sit?). ROCr's own pool, fragment cache and L2 are irrelevant — that is a narrower statement than
"the allocator layout matters", and it points at the layer to fix.

## Did the fault also damage runs that did *not* crash?

Every captured abort is `RW: 0x0` — the Tensile kernel **reads** one page past a buffer. A read
cannot scribble on a neighbouring tensor, so the silent failure mode to worry about is narrower: if
that page happens to be mapped, the kernel reads garbage into its own GEMM result, the step computes
wrong numbers, and nothing crashes and no dmesg line appears. The question then is whether the
arithmetic depends on the address layout, which is measurable.

`repro_real.py --grid integrity` plus `integrity.py` run two probes, each comparing runs byte for
byte (per-step `Train/Loss`, the LoRA tensors at every checkpoint, and the sample PNGs):

- **Allocator layout.** The same config and seed under three layouts: the caching allocator,
  `PYTORCH_NO_HIP_MEMORY_CACHING=1` (every tensor its own `hipMalloc`, addresses reshuffled) and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. If an overread ever fed garbage into a result,
  these runs could not agree exactly.
- **Sampling on / off.** `validation.sample_seed = 1` gives the sample pass its own generator;
  `trainer/sampling.py` otherwise draws its seed from the **global** RNG, which would make the two runs
  diverge for an innocent reason. With a fixed sample seed, a run that samples every 30 steps and a run
  that never samples must produce the same 120 losses unless the sample pass alters training state.

### First pass: the comparison was confounded, and the confound is worth keeping

Comparing those runs byte for byte turned up something that has nothing to do with the fault: **two runs
of the same configuration do not agree on their cached latents.** Hashing `data/.latents_cache/*.pt`
across five runs of the same six images, only 1-4 of the six latents ever match another run's — 
even between two runs that used the same allocator *and* the same training seed. The VAE encode is not
bitwise reproducible on this box (MIOpen picks conv algorithms per process), so every trajectory
diverges from step 1 (~`1e-5` relative) before the fault could possibly show up.

That matters on its own: the pipeline has a run-to-run noise floor in its *inputs*, so "run it twice and
diff" cannot detect silent damage. It also invalidates any claim that a surviving run is bitwise
reproducible without pinning the latents first.

### Second pass, with one shared latent cache

`--data-dir` points every row at one staged dataset, so the latent cache is written once and read by
all of them, and the comparison becomes meaningful:

| pair | question it answers |
| --- | --- |
| same config twice | is the training path reproducible at all once its inputs are fixed? |
| cached vs `no-hip` | does the allocator layout change the numbers? |
| samples vs no samples | does the sample pass alter training state? |

Both seeds are run, under the caching allocator and under `no-hip`, so a trajectory that aborts can be
compared against one that survives. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **not
runnable** here: the first step's loss comes out `NaN` and the trainer dies serialising it.

### What the shared-cache comparison shows

Both seeds, one latent cache, four runs each. Seed 1145141919 (every run survives the full 120 steps):

| pair | per-step loss | saved LoRA tensors | sample PNGs |
| --- | --- | --- | --- |
| `cached` vs `no-hip` | **byte-identical, 120/120 steps** | **all 2208 tensors at steps 30/60/90/120/final byte-identical** | 4 shared, all byte-identical |
| `cached (a)` vs `cached (b, repeat)` | byte-identical, 120/120 steps | all byte-identical | — |
| `cached` vs `sampling off` | identical for **30** steps, differ from step 31 (`~3.6e-6` relative) | 1464 of 2208 differ at the end | — |

Seed 1145141920 (both `cached` runs die at step 101):

| pair | per-step loss | saved LoRA tensors |
| --- | --- | --- |
| `cached` (dies at step 101) vs `no-hip` (survives 120) | **byte-identical over all 101 shared steps** | **all 2208 tensors at steps 30 / 60 / 90 byte-identical** |
| the *first* run of the session vs every later one | differs from step 1 | — |

- **The fault leaves no numerical trace.** The run that dies and the run that survives produced identical
  losses for every step they share, and identical LoRA weights at every checkpoint they both wrote. They
  also ran under two completely different allocator layouts, so this doubles as the layout-independence
  result: no garbage entered a GEMM result in either run before the abort.
- **With the inputs pinned, the training path is bitwise reproducible across a whole 120-step run** — same
  configuration twice, and `cached` vs `no-hip` too: identical losses, identical tensors at every
  checkpoint, identical sample PNGs. The "noise floor" is in the *inputs* (a cold MIOpen cache picks
  different conv algorithms) and in the sample pass, not in the step.
- **The sampling pass perturbs the trajectory slightly** — the first difference appears at step 31/32,
  right after the step-30 sample, at `~4e-6` relative. It is deterministic (both sampling runs agree
  exactly), so it is not damage; it is a second thing worth knowing about sampling: it is not a no-op for
  the training numbers.
- **The second, stronger probe: let the overrun happen and compare.** `HSA_SVM_GUARD_PAGES=0` removes the
  guard page, so the run survives *while still reading past its buffers* — the failure converted from
  fatal into silent, which is exactly the hypothesis to test. Warm `network_dim 32` run vs the `no-hip`
  reference: **all 12 losses byte-identical, all 2208 LoRA tensors byte-identical.** So the overrun's read
  either does not reach the result or is masked out of it; it is not silent corruption.
- **And the same probe at full scale**, on the canonical configuration (seed 1145141920, dim 36, the one
  that dies at step 101): with guard pages off it runs **120/120 steps** instead of dying, and against the
  `no-hip` run of the same configuration and seed it is **120/120 losses byte-identical and all 2208
  tensors at steps 30/60/90/120/final byte-identical**. Two independent allocator arrangements, one of
  them overrunning whenever the shapes line up, and not one number differs.
  (The four sample PNGs *do* differ between those two runs — the VAE decode is not bit-reproducible here,
  same as the VAE encode. It does not affect the training numbers.)
- **The first run of a session is not comparable to later ones.** Its trajectory differs from step 1
  because a cold MIOpen cache selects different conv algorithms — the same effect that made the first
  version of this experiment meaningless.

Two mistakes were made building this and are worth recording, because both look like findings:

- The first version of the grid selected the seed only in the row *label*, never in the row itself, so
  every "seed 1145141919" row actually ran seed 1145141920. That produced a dramatic-looking result
  ("the sample seed flips the surviving seed into a crashing one") which was pure mislabelling.
  `repro_real.py` now refuses to start a row whose label names a seed it would not run.
- Comparing runs byte for byte across separate data directories is meaningless (see above): fix the
  latents first, with `--data-dir`.

## The path that does reproduce it 100%

`repro_real.py` is that path, packaged — no harness, no synthetic tensor, just the trainer:

```
python repro_real.py                                   # the whole table below
python repro_real.py --row "original (packaged 6 images)"   # just the 100% row (~3 minutes)
python repro_real.py --grid fast --list                # the speed/mechanism/rank grids
```

It writes `resources/original-config.toml` and the six packaged kanae images into a throwaway repo
mirror and starts the unmodified `trainer/main.py` there, so nothing under the repository or under a
user dataset directory is written to. Model weights are referenced, never copied:
`trainer/config.toml`'s `pretrained_model_name_or_path`, overridable with `--model`.

| row | data | batch | dim | seed | steps | result |
| --- | --- | --- | --- | --- | --- | --- |
| `original (packaged 6 images)` | packaged kanae | 3 | 36 | 1145141920 | 120 | **FAULT, step 101** |
| `original, run again` | packaged kanae | 3 | 36 | 1145141920 | 120 | **FAULT, step 101** |
| `seed 1145141919 (the surviving seed)` | packaged kanae | 3 | 36 | 1145141919 | 120 | ok |
| `fix2.txt: batch 2, dim 36, kanae` | kanae (188) | 2 | 36 | 1145141920 | 94 | ok |
| `fix2.txt control: batch 3, dim 36, kanae` | kanae (188) | 3 | 36 | 1145141920 | 63 | ok |
| `fix2.txt: batch 2, dim 36, 3-chunk captions` | kanae (188, padded) | 2 | 36 | 1145141920 | 94 | ok |
| `fast: cadence 2, cheap samples` | packaged kanae | 3 | 36 | 1145141920 | 10 | **FAULT, step 8 (72 s)** |

The remaining rows of the same grid are in `resources/real-results.json`: `no HIP memory caching` and
`sampling every 30 steps disabled` both survive 120/120 steps, while `network_dim 32` aborts at step 4
(see the sibling section below).

The config the faulting row runs: SDXL base, six kanae images, `train_batch_size = 3`,
`network_dim = 36`, `max_token_length = 225`, `bucket_reso_steps = 128`, bf16,
`save_every_n_steps = 30`, seed 1145141920. The process dies with `-6` (SIGABRT) at **step 101 of
120** (epoch 51), right after the step-90 sample, in
`Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x128_MI16x16x1_SN_..._ISA1201_..._WGMXCC1`
(`GCVM_L2_PROTECTION_FAULT_STATUS:0x0080113B`, TCP client, 2 MiB-aligned faulting address).

The status word varies between sightings with the same client and kernel: `0x0080113B` (most runs),
`0x0090113B` (the first entry in `fixes/fix2-ex.md` and again in the shared-cache rows here) and
`0x00801031` (the `network_dim` 12/24/32 deaths, where `WALKER_ERROR` and `MAPPING_ERROR` come out
clear). `RW: 0x0` — a read — is constant across all of them, which is what makes "silently wrong
numbers" the only interesting failure mode, and the integrity section below is about exactly that.

It is not accumulated machine state: after a clean boot the same row faulted **6 times in a row**
(`resources/real-baseline-4of4.json`, `resources/fault-*-after-reboot.log`). Each death leaves three
DataLoader forkserver workers behind, which `repro_real.py` reaps by matching that run's own
`mirror` path.

The smallest known difference between a faulting run and a surviving one is `training.seed`
(1145141920 vs 1145141919, everything else byte-identical). That seed changes the caption-shuffle
RNG, hence the tokenized prompt, hence the number of CLIP chunks, hence the encoder sequence length
and `M`:

```
M = train_batch_size * (chunks * 77)      chunks ∈ {1, 2}  ->  M ∈ {231, 462}   (batch 3)
```

Both seeds spend ~85% of their steps at `M = 462`, and the surviving runs therefore use the "killer"
shape constantly — which is exactly why the shape alone cannot be the trigger. The crash lands on a
**shape-switch step** (462↔231), where the GEMM workspace is freed and re-allocated; whether the
overrun then meets an unmapped page is decided by the allocator's layout, which is itself
reproducible for a given config *and* seed.

## Why the seed flips it (measured, not hand-waved)

The seed does not work through tensor *values*: PEFT starts `lora_B` at zero, and a Tensile kernel's
addresses come from shapes and pointers, never from the bytes in the buffers. `seed_step_shapes.py`
walks the trainer's own sampler and dataset on the CPU and shows what the seed really moves:

- which images share a batch — `BucketBatchSampler` shuffles with `random.Random(cfg.seed + epoch)`;
- how each caption is shuffled — `random.Random(cfg.seed + epoch + sha1(path))`, hence how many CLIP
  chunks it needs (one chunk = 74 content tokens), hence `M = batch * chunks * 77`.

For the packaged run the bucket order is identical for both seeds (all six images fall into one
1024×768 bucket), so the chunk count is the only lever left, and it moves `M` from step 5 onwards:

| step | seed 1145141920 (dies) | seed 1145141919 (survives) |
| --- | --- | --- |
| 100 | M 462 | M 231 |
| **101** | **M 231 — dies here** | M 462 |
| 102 | M 462 | M 462 |

The totals are almost identical (101 vs 102 steps at M 462, 19 vs 18 at M 231, 32 vs 31 shape
switches). What differs is *where the transitions sit*: the crashing run switches 462 → 231 exactly
at step 101, the surviving run switches 231 → 462 there and does its own descending switch two steps
later. So the crash lands on a **shape transition**, and the seed decides which step that transition
is.

A transition by itself is not sufficient either: seed 1145141920 makes 32 of them in 120 steps and
survives 31, including a 462 → 231 at step 5. The missing half is the allocator layout the transition
runs into, and the meta-operation grid above pins down what poisons it: **the sampling pass**. Turn
the cadence off and those same 32 transitions at the same seed run 120 clean steps; keep it on and the
fault appears at the 3rd or 4th sample, in whichever GEMM the bad layout happens to be feeding
(with the packaged 768x768 samples that is 11 steps later, at a descending transition; with 512x512
samples it is the sample itself).

So the original run needs the coincidence: a poisoned layout (step-90 sample) *and* a descending
shape transition eleven steps later (step 101, chosen by the seed). `seed_step_shapes.py` reproduces
the shape trace and `repro_real.py --grid mech` reproduces the layout drift; neither alone is the
fault.

## Consequences

- "Avoid `M` with a partial Tensile tile" is not a usable rule: the same shapes run fine in a clean
  process, and `network_dim = 32` (no partial `K` tile at all) faults anyway. The reliable mitigation
  remains the one in `fixes/fix2.txt`: keep `bucket_reso_steps = 128` and fall back to
  `PYTORCH_NO_HIP_MEMORY_CACHING=1` when a run dies this way — measured here it carries the same
  configuration through 120/120 steps at ~2.9 s/it instead of 1.12 s/it.
- Two cheaper mitigations are **ruled out** by measurement: `flush_memory_every_step` (an
  `empty_cache` + gc before every step) still faults at step 101, and neither the DataLoader workers
  nor the latent cache participate. Shrinking the samples to 256x256 does avoid the fault, but that
  changes what the run does, so it is a diagnostic, not a workaround.
- Automation should treat `HSA_STATUS_ERROR_MEMORY_FAULT` in a child's log as this known fault:
  the process exits `-6`, leaves `state.json` at `training` with a dead PID, and its DataLoader
  forkserver workers outlive it and must be reaped by matching the run's own working directory.
- The `fix2.txt` dodge ("avoid `train_batch_size = 2` with `network_dim = 36`") is not needed on
  this machine: that combination trains a whole epoch fine, as does its derived shape on every
  step. What still needs a dodge is the long run.
- Two failures that are *not* this bug, found while building the grid and left alone: turning off
  gradient checkpointing OOMs the 1024x768 UNet backward on 16 GB, and `cache_latents = false` hits
  the trainer's on-demand VAE path with the VAE still on CPU
  (`Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)`).
- What would settle the remaining mechanism question is a *minimal* process that still faults, or a
  way to observe the allocator layout at the moment of the abort. The measurements above narrow it to
  "a sample pass of at least 512x512 paired with the shapes around it", but the quantifier that turns
  that into a prediction is still missing.
</content>

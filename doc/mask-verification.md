# Mask pipeline verification — status and restart runbook

Closed-loop verification for the optional loss-mask pipeline (`{stem}.mask.png` sidecar, or the
training image's alpha channel). The harness is `test/verify_mask_pipeline.py`; run it from the repo root, and it drives
the real `trainer/main.py` and writes `<report-dir>/mask_verify_report.md` + `.json`.

**Current state: see "Where it stands" below.** A local re-run needs a free GPU: the harness refuses
to start while a training run looks live, which is correct, because the child runs need the whole
card. The geometry under test is fit+pad (no cropping): every sample is fitted whole into its
area-budgeted bucket and the letterbox pad carries loss weight 0.

Some of the earlier measurements this document compares against were taken inside the sealed
gfx1201 record (`archive/`, and the `fixes/fix2` grid it came from) — 涉及负责任披露流程，暂不公开.
Where a claim rests on them it says so.

## 1. What each tier asserts

| Tier | Cost | What it proves |
| --- | --- | --- |
| `plumbing` | CPU, seconds | Sidecar discovery and exclusion, `{stem}.mask.png` name rules, mask geometry after fit+pad at a real bucket (including "a tall sample keeps its head and feet"), per-sample pairing through collate + bucket grouping, latent-cache independence (a mask edit does not move the key, a geometry change does), and alpha-derived masks against an independent fit on the real dataset. |
| `loss` | GPU, one pipeline load | Exact identities: all-ones mask is bit-identical to no mask; all-black gives zero loss, zero weight grads and zero latent grad; the loss is linear in the mask (no area renormalization); coverage scales the loss and the gradient norm; a real alpha mask reaches the loss and lowers it. |
| `train` | GPU, 8 child runs | Masked vs unmasked vs scale-matched training through `trainer/main.py` on copies of the configured dataset: run completion, the trainer's own `Loss masks:` line, TensorBoard scalars, a bitwise duplicate run as the repeatability floor, the loss-weighting law, masked-vs-unmasked weight deltas, per-region probes, and a weight-only resume. |
| `stand` | GPU, ~5 child runs | The same on copies of the untagged stand dataset, where every mask comes from the image's own alpha channel (no sidecars are written). |

Run matrix of a training tier, for `--steps 120 --seeds 2`: `masked` (spatial sidecar mask),
`control` (no mask), `scale` (constant gray weight with the same coverage as `masked`, so a
difference between them is geometry rather than loss volume), one duplicate `masked` floor run, and
one resume sub-run. Checkpoint cadence is set to `steps // 4` so samples and step checkpoints land
inside the window.

## 2. Where it stands

**Complete after the fit+pad (no-crop) geometry change.** Full run 2026-09-15 12:23, `--tiers all`,
in the `axl_rocm_7_14` env (torch `2.12.0+rocm7.14.1`, HIP `7.14.60850`): **54 checks, 0 failed,
2846 s** (~47 min), 24 observations. Report:
`/home/acite/LLM/axltrainer/mask-verify/fitpad_20260915_122315/mask_verify_report.md` + `.json`. All
13 child training runs finished on their **first attempt** with `gpu_memory_fault=false` — the page
fault never appeared under this stack, including on the new portrait buckets
(`512x1920`, `640x1664`, `640x1792`, `384x2304`, …) whose kernel shapes did not exist before.

The run before this one (2026-09-15 07:05, crop geometry) was 52 checks / 2482 s; it is superseded by
this one but remains the reference for the *pre-change* behaviour:

| Run | Geometry | Result |
| --- | --- | --- |
| `20260915_0705` | centre crop | 52/52, 2482 s |
| `fitpad_20260915_122315` | area-budgeted buckets + fit+pad | **54/54, 2846 s** |

One plumbing check was strengthened *after* that report was written (it now also asserts that an
independent centre crop of the same sample loses both end bands). The strengthened check was re-run
on its own — `--tiers plumbing --report-dir /tmp/axl-mask-verify/plumbing-final`, `DONE`, 0 failed,
`crop_keeps_head=False crop_keeps_feet=False` — rather than re-running the 13 GPU children for an
assertion that already passed.

An earlier attempt on 2026-09-14 was interrupted by the fault before `write_report` ran, so nothing
from it survives except the `plumbing`-only report in `/tmp/axl-mask-verify/plumbing-only/`.

What the run establishes:

| Claim | Evidence |
| --- | --- |
| Unmasked is untouched | all-ones mask bit-identical to no mask (loss `0.148071` three times, grad diff `0.0`) |
| A fully ignored sample contributes nothing | all-black mask → loss `0`, weight-grad norm `0`, latent-grad max `0` |
| The loss is a plain spatial weight, no area renormalization | `right40 + left60 = 0.148071` = the all-ones loss exactly; half-gray = half the loss |
| The pad is really zero-weight, and the geometry is really a fit | pad rows/columns are exactly `0` while the content is covered (plumbing: `pad_weight_max=0`, `agreement=1` against an independent fit); the mask survives fit+pad pixel-for-pixel |
| A tall sample keeps its head and feet | a 1:3 source in a `256x512` bucket keeps both end bands at weight `0.999` with 33 % pad — the case the old crop rule deleted |
| The trainer's reported loss follows the mask | masked/control `0.0704`/`0.1190` (seed 1) and `0.0651`/`0.1103` (seed 2) against coverage `0.588963`; law error `1.3e-4` / `1.1e-5`. The pad scales this law just like a content mask, so the reference is the content-covered control run |
| Masks reach real runs | `Loss masks: 12/12` masked, `0/12` control, `12/12` scale-matched; same for the stand tier (`3/3`, `0/3`) |
| Masked checkpoints stay usable | resume loads `1472` tensors, `0` skipped; kohya metadata intact (`network_dim 64`, `alpha 32`) |
| A geometry change moves the cache key, a mask edit does not | plumbing checks the documented key text and shows the same path under two fit geometries hashing differently |

Two results are deliberately **not** resolved, and should not be read as support for anything:

- **Masked vs unmasked LoRA weights at 120 steps.** `signal_over_floor` is `0.72`–`0.86` across the
  comparisons (two seeds × unmasked/scale-matched): the mask's effect on the final weights
  (`~0.071`–`0.080` relative L2) is *smaller* than the duplicate run's own difference
  (`0.083`–`0.111`). The floor run is not bitwise identical here (`duplicate_is_identical=false`,
  1464 of 2208 tensors changed at bf16 scale), which is the same "first run of a session is not
  comparable to later ones" effect the sealed fix2 record documents — MIOpen picks its algorithms per
  process, and the first masked run of the session ran with a cold cache. Resolving the weight
  question needs a warm-up run before the measured ones; it does not need a different mask
  implementation.
- **The per-region probe is directionally right but not separable from noise at 120 steps**
  (`masked_minus_unmasked_ignored_region` positive, `effect_over_noise` `1.8` against the 3× bar,
  `resolved: false` in both tiers). Same reading as the weight question: more steps, a lower LR, or
  deterministic kernels would be needed to resolve it.
- **Mask coverage scales the gradient norm harder than the loss** (`0.40`–`0.63` gradient ratios
  against `0.23`–`0.49` loss ratios): recorded as measured, not asserted — the mask removes loss
  where the error is smallest, so the remaining gradient is not simply a scaled copy.

Masked loss is *not* inpainting: masking suppresses the objective for the ignored pixels, it does not
insulate the network from them. The `loss` tier measures that directly — repainting only the
masked-out 40% still moves the LoRA gradient, by `share_of_trained_side = 1.23` of the same repaint on
the trained side.

What `plumbing` established, in short: masks pair with their own image and never appear as training
samples; the mask survives fit+pad pixel-for-pixel against an independent implementation (a sidecar at
`1280x768`, alpha on real transparent art at `512x1920` / `640x1792` / `640x1664`); a mask edit does
not invalidate the documented latent-cache key while a geometry change does (the key carries the fit
geometry); and on the then-configured dataset
(`/storage/Games/AVG/LimeLight Lemonade Jam/dataset/mix/bg`, 130 images) 4 files carry alpha and the
alpha is **not** effectively opaque (mean 0.55), so alpha does act as a mask there.

## 3. Resources it needs

| Need | Current value |
| --- | --- |
| Model | `/opt/models/diffusers/waillu_170` (exists; read from `config.toml`, never copied) |
| Training dataset | `train_data_dir` from `config.toml` — **currently `/home/acite/LLM/Character/LLLJ/`** (640 images). Only the first `--images` files are copied out. |
| Stand dataset | `--stands-dir` default `/storage/Games/AVG/LimeLight Lemonade Jam/dataset/stands/杏珠` (241 files, no `.txt` captions → the trainer falls back to the file stem) |
| Interpreter | `/home/acite/miniconda3/envs/axl/bin/python` — torch `2.13.0+rocm10.0.0` / HIP `7.15.26333`. (The 2026-09-15 run in §2 used the `axl_rocm_7_14` env, which was the pin at the time.) The harness reads the expected env **name** from `environment.yml`'s `name:` and refuses any other prefix (`--allow-foreign-env` to override), so it follows a rename of the env. |
| GPU | One RX 9070 XT, exclusively. Child runs load SDXL at bf16; the earlier tiers peaked around 10 GB. |
| Wall clock | **2846 s (~47 min) measured** on 2026-09-15 (fit+pad geometry) for `--tiers all`, of which the `train` tier is 8 child runs of 120 steps at ~4.5 min each and `stand` is 5 runs of 60 steps. The crop-geometry run before it took 2482 s. |
| Scratch | Several GB. **The default `--report-dir` is under `/tmp`, which is tmpfs (16 GB, 6.3 GB free right now).** Point it at a disk. |

Each child run gets a generated config in a throwaway repo mirror. It **inherits** everything from the
live `config.toml` (model, `network_dim`, dataset, bucket settings, `max_token_length`,
precision, `sample_seed`, …) and **overrides** only: `seed`, `epoch`, `save_every_n_steps`,
`sampling_enabled = true` (the checks count sample images, so the verifier asks for them whatever
the live config was left at), `resume_lora_path`, `train_batch_size = 3` (pinned in `launch_run`, so the live config's batch size
does not apply), `te_warmup_steps`, both learning rates (`--unet-lr`/`--te-lr`, deliberately higher
than the shipped values so a few minutes move the LoRA measurably), DataLoader workers = 2, and cheap
samples (`768×768`, 8 steps, 1 repeat). So `network_dim` and `sample_seed` come from the config file
as it stands at launch — check them before starting.

Arrays of tables are the one exception: `[[validation.samples]]` cannot be expressed by the flat
line writer, so the mirror carries only the `[validation]` scalars and the child resolves **one**
prompt set from them (written as a `samples = "…"` string it aborted the child instead).

Nothing is written into a source dataset directory: images are copied first and every child run gets
its own `AXL_RUNTIME_DIR`.

## 4. Re-run checklist

1. **Confirm nothing is training.** The harness refuses to start when a live training process or a
   live `state.json` exists (checks `LIVE_STATUSES`, so `sampling` counts). Do not pass `--force`
   while another run is live — the children need the whole GPU.
   ```bash
   pgrep -af 'trainer/main.py'
   cat "${XDG_RUNTIME_DIR:-/tmp}/axltrainer/state.json"
   ```
2. **Use the project env and put the report on disk, not `/tmp`**, and prune step checkpoints:
   ```bash
   conda activate axl
   cd /home/acite/Deeppin/AxlTrainer
   python test/verify_mask_pipeline.py --tiers all \
     --report-dir "/home/acite/LLM/axltrainer/mask-verify/$(date +%Y%m%d_%H%M%S)" \
     --prune-step-checkpoints
   ```
   The directory is created by the harness; keep the timestamped name so a later run does not
   overwrite this one.
3. **Expect the retry path to be load-bearing again.** The 2026-09-14 attempt died to the gfx1201
   fault because its children ran on `torch 2.13.0+rocm10.0.0`; both 2026-09-15 runs used the stack then
   pinned (`axl_rocm_7_14`, `2.12.0+rocm7.14.1`) and finished all 13 child runs on their first attempt
   with no fault — the fit+pad run on buckets whose shapes (`512x1920`, `640x1792`, `384x2304`) did not
   exist before (see `doc/troubleshooting.md`). `environment.yml` pins `2.13.0+rocm10.0.0` again — the
   stack the 2026-09-14 children faulted on — so the default `--retries 2` matters: a child that does
   die is relaunched, and from the second retry on the child gets `PYTORCH_NO_HIP_MEMORY_CACHING=1`
   (~2.2× slower). If retries do fire, a tier takes correspondingly longer and the affected run is
   marked in the report; `--no-hip-memory-caching` for the whole run is only worth it if the faults
   are constant.
4. **Read the run as one report.** `DONE` means zero failed checks; `FAILED` exits 1 and the failing
   rows are listed in the report. Interrupting the harness leaves no report — that is what happened
   last time.
5. Optional narrowing, useful for a second pass: `--tiers plumbing,loss` is CPU + one pipeline load
   (minutes, no child training runs, so no page-fault exposure) and `--no-resume-check` drops the
   resume sub-run.

## 5. The finished masked run as a reference

The masked loss feature has been exercised by a real training run, which is an independent reference
for the `train` tier. It finished at 07:04 on 2026-09-15: step 3210/3210, epoch 10/10, no error.
**It ran under the old centre-crop geometry**, so its per-step loss levels and its per-image
buckets are not comparable with runs made after the fit+pad change — use it for the *plumbing*
claims (mask discovery, alpha-only masking, checkpoint handling), not for loss magnitudes.

| Field | Value |
| --- | --- |
| Run | `output_name = lllj`, `run_id = lllj_20260915_053826`, PID 170386, started 2026-09-15 05:38 |
| Status | `finished`, step 3210/3210, epoch 10/10, run time ~86 min |
| Interpreter | conda `axl_rocm_7_14` → torch `2.12.0+rocm7.14.1`, HIP `7.14.60850` |
| Data | `/home/acite/LLM/Character/LLLJ/` — 640 images, **0** `{stem}.mask.png` sidecars, 518 alpha-capable |
| Mask source | alpha channel only (a sidecar would win, but none exist) |
| Config | `train_batch_size = 2`, `network_dim = 64`, `seed = 1145141919`, `save_every_n_steps = 50`, `sample_seed = 0` |
| Artifacts | `outputs/lllj_20260915_053826/lllj_s000050 … lllj_s003200`, `lllj_final`, `lllj_samples/` (195 PNGs), TB scalars in `logs/lllj_20260915_053826/` |

How much of it is actually masked: `plumbing` samples the first 12 records plus a spread across the
whole dataset (61 images) and finds `alpha_mean = 0.79`, `mask_mean = 0.855`, and **20 of the sampled
alpha-carrying images partially transparent** — so roughly a third of the 640 images carry a real
loss weight and the rest are opaque, where the mask is a no-op. The reference is a **mixed** masked
run, not a fully masked one. (The first 60 records are all opaque — the event CGs sort before the
transparent character art — which is why the scan samples across the dataset rather than a prefix.)

Two caveats when using it as a reference:

- **Its mask count is not in its log.** `start_train.sh` ends with
  `exec python -u trainer/main.py 1> >(grep -Ev "grid_desc|CandidateSelectionModel|metadata" >> /dev/null)`,
  so the child's stdout is discarded — the launch prints (`Loss masks: N/M samples`,
  `Checking/Generating latents cache...`) never reach `train.log`, which only carries stderr (tqdm,
  tracebacks). The harness does not have this problem: it launches `trainer/main.py` itself and
  captures stdout to the child log, which is how the `train` tier's "reports mask usage" check reads
  the line. For the live run, the count has to be derived from the dataset, as above.
- **Its numbers are one stack away from everything measured since.** The reference run used the stack
  then pinned (`axl_rocm_7_14`, torch `2.12.0+rocm7.14.1`, HIP `7.14.60850`); the harness children now
  follow `environment.yml` (`axl`, torch `2.13.0+rocm10.0.0`), which is also the stack the
  sealed measures behind this document were taken on — different kernels, different allocator behaviour. The
  *plumbing* claims above do not depend on the stack; the loss levels do.

## 6. Interpretation rules for the results

- **Loss levels are only comparable within one geometry.** Every sample now carries the letterbox pad
  at weight 0, so a run's reported loss is scaled by the mean mask weight (content coverage × mask
  coverage), and the letterboxed buckets differ per dataset. Compare runs made with the same
  `[bucketing]` settings and the same dataset; a pre-fit+pad run's loss curve is a different quantity.
- **A retried child is still comparable.** If a run was relaunched under
  `PYTORCH_NO_HIP_MEMORY_CACHING=1`, its allocator layout differs from its untried sibling's, but the
  numbers do not: the sealed integrity pass showed the caching allocator and `no-hip` produce
  byte-identical losses (120/120 steps) and byte-identical LoRA tensors (all 2208 at every
  checkpoint) once the latent cache is shared. That is also why `train` reports its retry count.
- **Pin the latents if you want to diff runs by hand.** The VAE encode is not bitwise reproducible
  here (MIOpen picks conv algorithms per process), so two runs in separate data directories diverge
  from step 1 at ~1e-5 even with the same seed. The harness gives every variant its own copy, so its
  masked-vs-control deltas are comparisons of *different* data directories — read them as
  "clearly larger than the noise floor", not as bitwise identities. Only the duplicate `floor` run,
  which reuses the same copy, is a bitwise claim.
- **The first run of a session is not comparable to later ones** (cold MIOpen cache selects different
  conv algorithms). This is what made the first version of the fix2 integrity grid meaningless.
- **`sample_seed` is inherited, and the live config has it at `0`.** That means the sample pass draws
  from the global RNG, and per that same sealed integrity pass a sampling run and a non-sampling
  run diverge from step 31 at ~4e-6 relative. Every child run samples on the same cadence, so the
  masked/control/scale comparison is still like-for-like, but nothing in this verification should be
  read as "sampling is a no-op". Set `sample_seed` to a fixed non-zero value in `config.toml`
  before the run if you want the trajectory to be independent of the sample pass.
- **Sample PNGs are not comparable.** The VAE decode is not bitwise reproducible here, so the sample
  images of two runs differ even when the training numbers are identical. Compare losses and LoRA
  tensors, not pictures.

# Mask pipeline verification — status and restart runbook

Closed-loop verification for the optional loss-mask pipeline (`{stem}.mask.png` sidecar, or the
training image's alpha channel). The harness is `verify_mask_pipeline.py` at the repo root; it drives
the real `trainer/main.py` and writes `<report-dir>/mask_verify_report.md` + `.json`.

**Current state: complete** — 52/52 checks on 2026-09-15 (see "Where it stands"). A local re-run
needs a free GPU: the harness refuses to start while a training run looks live, which is correct,
because the child runs need the whole card.

## 1. What each tier asserts

| Tier | Cost | What it proves |
| --- | --- | --- |
| `plumbing` | CPU, seconds | Sidecar discovery and exclusion, `{stem}.mask.png` name rules, mask geometry after crop/resize at a real bucket, per-sample pairing through collate + bucket grouping, latent-cache independence, and alpha-derived masks against an independent crop on the real dataset. |
| `loss` | GPU, one pipeline load | Exact identities: all-ones mask is bit-identical to no mask; all-black gives zero loss, zero weight grads and zero latent grad; the loss is linear in the mask (no area renormalization); coverage scales the loss and the gradient norm; a real alpha mask reaches the loss and lowers it. |
| `train` | GPU, 8 child runs | Masked vs unmasked vs scale-matched training through `trainer/main.py` on copies of the configured dataset: run completion, the trainer's own `Loss masks:` line, TensorBoard scalars, a bitwise duplicate run as the repeatability floor, the loss-weighting law, masked-vs-unmasked weight deltas, per-region probes, and a weight-only resume. |
| `stand` | GPU, ~5 child runs | The same on copies of the untagged stand dataset, where every mask comes from the image's own alpha channel (no sidecars are written). |

Run matrix of a training tier, for `--steps 120 --seeds 2`: `masked` (spatial sidecar mask),
`control` (no mask), `scale` (constant gray weight with the same coverage as `masked`, so a
difference between them is geometry rather than loss volume), one duplicate `masked` floor run, and
one resume sub-run. Checkpoint cadence is set to `steps // 4` so samples and step checkpoints land
inside the window.

## 2. Where it stands

**Complete.** Full run 2026-09-15 07:05, `--tiers all`, in the `axl_rocm_7_14` env
(torch `2.12.0+rocm7.14.1`, HIP `7.14.60850`): **52 checks, 0 failed, 2482 s** (~41 min). Report:
`/home/acite/LLM/axltrainer/mask-verify/20260915_0705/mask_verify_report.md` + `.json`, raw log
`run.log` next to it. All 11 child training runs finished on their **first attempt** with
`gpu_memory_fault=false` — the page fault never appeared under this stack.

An earlier attempt on 2026-09-14 was interrupted by the fault before `write_report` ran, so nothing
from it survives except the `plumbing`-only report in `/tmp/axl-mask-verify/plumbing-only/`.

What the run establishes:

| Claim | Evidence |
| --- | --- |
| Unmasked is untouched | all-ones mask bit-identical to no mask (loss `0.15557` three times, grad diff `0.0`) |
| A fully ignored sample contributes nothing | all-black mask → loss `0`, weight-grad norm `0`, latent-grad max `0` |
| The loss is a plain spatial weight, no area renormalization | `right40 + left60 = 0.15557` = the all-ones loss exactly; half-gray = half the loss |
| The trainer's reported loss follows the mask | masked/control `0.598` and `0.600` against coverage `0.6087`, law error `1.2e-4` / `2.2e-4`, both seeds |
| Masks reach real runs | `Loss masks: 12/12` masked, `0/12` control, `12/12` scale-matched; same for the stand tier (`3/3`, `0/3`) |
| Masked checkpoints stay usable | resume loads `1472` tensors, `0` skipped; kohya metadata intact (`network_dim 64`, `alpha 32`) |
| The mask changes behaviour where it is applied | per-region probe: masked-trained error is worse in the ignored region and better in the trained region, `consistent_sign` true and `resolved` true in both tiers |

Two results are deliberately **not** resolved, and should not be read as support for anything:

- **Masked vs unmasked LoRA weights at 120 steps.** `signal_over_floor` is `0.72`–`0.83` across all
  four comparisons (two seeds × unmasked/scale-matched): the mask's effect on the final weights
  (`~0.070` relative L2) is *smaller* than the duplicate run's own difference (`0.0887`). The floor
  run is not bitwise identical here (`duplicate_is_identical=false`, 1464 of 2208 tensors changed at
  bf16 scale), which is the same "first run of a session is not comparable to later ones" effect
  documented in `fixes/fix2/` — MIOpen picks its algorithms per process, and the first masked run of
  the session ran with a cold cache. Resolving the weight question needs a warm-up run before the
  measured ones; it does not need a different mask implementation.
- **Mask coverage scales the gradient norm harder than the loss** (`0.63` and `0.53` gradient ratios
  against `0.49` and `0.24` loss ratios): recorded as measured, not asserted — the mask removes loss
  where the error is smallest, so the remaining gradient is not simply a scaled copy.

Masked loss is *not* inpainting: masking suppresses the objective for the ignored pixels, it does not
insulate the network from them. The `loss` tier measures that directly — repainting only the
masked-out 40% still moves the LoRA gradient, by `share_of_trained_side = 1.47` of the same repaint on
the trained side.

What `plumbing` established, in short: masks pair with their own image and never appear as training
samples; the mask survives crop + bucket resize pixel-for-pixel against an independent implementation
(both as a sidecar at 1152×768 and as alpha on real transparent art at 768×1280); a mask edit does
not invalidate the documented latent-cache key; and on the then-configured dataset
(`/storage/Games/AVG/LimeLight Lemonade Jam/dataset/mix/bg`, 130 images) 4 files carry alpha and the
alpha is **not** effectively opaque (mean 0.55), so alpha does act as a mask there.

## 3. Resources it needs

| Need | Current value |
| --- | --- |
| Model | `/opt/models/diffusers/waillu_170` (exists; read from `trainer/config.toml`, never copied) |
| Training dataset | `train_data_dir` from `trainer/config.toml` — **currently `/home/acite/LLM/Character/LLLJ/`** (640 images). Only the first `--images` files are copied out. |
| Stand dataset | `--stands-dir` default `/storage/Games/AVG/LimeLight Lemonade Jam/dataset/stands/杏珠` (241 files, no `.txt` captions → the trainer falls back to the file stem) |
| Interpreter | `/home/acite/miniconda3/envs/axl_rocm_7_14/bin/python` — torch `2.12.0+rocm7.14.1` / HIP `7.14.60850`. The harness reads the expected env **name** from `environment.yml`'s `name:` and refuses any other prefix (`--allow-foreign-env` to override), so it follows a rename of the env. |
| GPU | One RX 9070 XT, exclusively. Child runs load SDXL at bf16; the earlier tiers peaked around 10 GB. |
| Wall clock | **2482 s (~41 min) measured** on 2026-09-15 for `--tiers all`, of which the `train` tier is 8 child runs of 120 steps at ~4 min each and `stand` is 5 runs of 60 steps. |
| Scratch | Several GB. **The default `--report-dir` is under `/tmp`, which is tmpfs (16 GB, 6.3 GB free right now).** Point it at a disk. |

Each child run gets a generated config in a throwaway repo mirror. It **inherits** everything from the
live `trainer/config.toml` (model, `network_dim`, dataset, bucket settings, `max_token_length`,
precision, `sample_seed`, …) and **overrides** only: `seed`, `epoch`, `save_every_n_steps`,
`resume_lora_path`, `train_batch_size = 3` (pinned in `launch_run`, so the live config's batch size
does not apply), `lr_warmup_steps`, both learning rates (`--unet-lr`/`--te-lr`, deliberately higher
than the shipped values so a few minutes move the LoRA measurably), DataLoader workers = 2, and cheap
samples (`768×768`, 8 steps, 1 repeat). So `network_dim` and `sample_seed` come from the config file
as it stands at launch — check them before starting.

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
   conda activate axl_rocm_7_14
   cd /home/acite/Deeppin/AxlTrainer
   python verify_mask_pipeline.py --tiers all \
     --report-dir "/home/acite/LLM/axltrainer/mask-verify/$(date +%Y%m%d_%H%M%S)" \
     --prune-step-checkpoints
   ```
   The directory is created by the harness; keep the timestamped name so a later run does not
   overwrite this one.
3. **Keep the retry path, but do not expect it to fire.** The 2026-09-14 attempt died to the gfx1201
   fault because its children ran on `torch 2.13.0+rocm10.0.0`; the 2026-09-15 run on the pinned
   `2.12.0+rocm7.14.1` stack finished all 11 child runs on their first attempt with no fault (see
   `doc/troubleshooting.md`). The default `--retries 2` stays as a safety net: a child that does die
   is relaunched, and from the second retry on the child gets `PYTORCH_NO_HIP_MEMORY_CACHING=1`
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
- **Same stack as the harness now.** The reference run and the harness children both use the
  `axl_rocm_7_14` env (torch `2.12.0+rocm7.14.1`, HIP `7.14.60850`), so the reference is directly
  comparable on that axis. The one thing it is *not* comparable to is the `fixes/fix2/` tables, which
  were measured on `2.13.0+rocm10.0.0` — different kernels, different allocator behaviour.

## 6. Interpretation rules for the results

- **A retried child is still comparable.** If a run was relaunched under
  `PYTORCH_NO_HIP_MEMORY_CACHING=1`, its allocator layout differs from its untried sibling's, but the
  numbers do not: `fixes/fix2/`'s integrity pass showed the caching allocator and `no-hip` produce
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
  from the global RNG, and per the integrity pass in `fixes/fix2/` a sampling run and a non-sampling
  run diverge from step 31 at ~4e-6 relative. Every child run samples on the same cadence, so the
  masked/control/scale comparison is still like-for-like, but nothing in this verification should be
  read as "sampling is a no-op". Set `sample_seed` to a fixed non-zero value in `trainer/config.toml`
  before the run if you want the trajectory to be independent of the sample pass.
- **Sample PNGs are not comparable.** The VAE decode is not bitwise reproducible here, so the sample
  images of two runs differ even when the training numbers are identical. Compare losses and LoRA
  tensors, not pictures.

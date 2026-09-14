# Mask pipeline verification — status and restart runbook

Closed-loop verification for the optional loss-mask pipeline (`{stem}.mask.png` sidecar, or the
training image's alpha channel). The harness is `verify_mask_pipeline.py` at the repo root; it drives
the real `trainer/main.py` and writes `<report-dir>/mask_verify_report.md` + `.json`.

**Current state: not finished, and not safe to restart while a training run is live.** `plumbing`
passed on 2026-09-14 and is saved; `loss`, `train` and `stand` have no surviving results. The last
attempt was interrupted by the gfx1201 Tensile page fault (`doc/troubleshooting.md`), which kills the
child training runs the `train`/`stand` tiers depend on. Read "Restart checklist" before launching.

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

| Tier | Status | Evidence |
| --- | --- | --- |
| `plumbing` | **PASS, 8/8 checks**, 9.2 s | `/tmp/axl-mask-verify/plumbing-only/mask_verify_report.md` + `.json` (2026-09-14 22:01, conda `axl`, torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, RX 9070 XT) |
| `loss` | no results | — |
| `train` | no results | — |
| `stand` | no results | — |

`plumbing` is also the only report on disk: searching `/tmp` and `/run/user/1000` for
`mask_verify*` returns that one directory and nothing else. A later full run was interrupted before
`write_report` ran, so there is no partial `mask_verify_report.*` for the GPU tiers to inspect — the
tiers have to be re-run, they cannot be resumed from a report. The scratch dir of that attempt
(datasets, mirrors, child runs) is gone too.

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
| Interpreter | `/home/acite/miniconda3/envs/axl/bin/python` — the harness refuses to run outside conda env `axl` (`--allow-foreign-env` to override) |
| GPU | One RX 9070 XT, exclusively. Child runs load SDXL at bf16; the earlier tiers peaked around 10 GB. |
| Wall clock | About an hour for `--tiers all` per `AGENT.md`. Estimate from the run matrix: `train` is 8 child runs × 120 steps ≈ 25 min, `stand` is 5 × 60 steps ≈ 10 min, plus one pipeline load and the `loss` probes. Not measured end-to-end here. |
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

## 4. Restart checklist

1. **Confirm nothing is training.** The harness refuses to start when a live training process or a
   live `state.json` exists (checks `LIVE_STATUSES`, so `sampling` counts). Do not pass `--force`
   while another run is live — the children need the whole GPU.
   ```bash
   pgrep -af 'trainer/main.py'
   cat "${XDG_RUNTIME_DIR:-/tmp}/axltrainer/state.json"
   ```
2. **Put the report on disk, not `/tmp`**, and prune step checkpoints:
   ```bash
   conda activate axl
   cd /home/acite/Deeppin/AxlTrainer
   python verify_mask_pipeline.py --tiers all \
     --report-dir "/home/acite/LLM/axltrainer/mask-verify/$(date +%Y%m%d_%H%M%S)" \
     --prune-step-checkpoints
   ```
   The directory is created by the harness; keep the timestamped name so a later run does not
   overwrite this one.
3. **Expect the gfx1201 fault and let the retry path absorb it.** Keep the default `--retries 2`: a
   child that dies from the Tensor page fault is relaunched, and from the second retry on the child
   gets `PYTORCH_NO_HIP_MEMORY_CACHING=1` (the documented dodge, ~2.2× slower). A tier with many
   retries takes correspondingly longer; `--no-hip-memory-caching` on the whole run is only worth it
   if the faults are constant.
4. **Read the run as one report.** `DONE` means zero failed checks; `FAILED` exits 1 and the failing
   rows are listed in the report. Interrupting the harness leaves no report — that is what happened
   last time.
5. Optional narrowing, useful for a second pass: `--tiers plumbing,loss` is CPU + one pipeline load
   (minutes, no child training runs, so no page-fault exposure) and `--no-resume-check` drops the
   resume sub-run.

## 5. The live masked run as a reference

There is a real masked training run in flight, which is a useful independent reference for the
`train` tier. Snapshot taken while checking (it keeps moving):

| Field | Value |
| --- | --- |
| Run | `output_name = lllj`, `run_id = lllj_20260915_053826`, PID 170386, started 2026-09-15 05:38 |
| Status at snapshot | `sampling`, step 700 / 3210, epoch 3 / 10 |
| Interpreter | conda `axl_rocm_7_14` → torch `2.12.0+rocm7.14.1`, HIP `7.14.60850` |
| Data | `/home/acite/LLM/Character/LLLJ/` — 640 images, **0** `{stem}.mask.png` sidecars, 518 alpha-capable |
| Mask source | alpha channel only (a sidecar would win, but none exist) |
| Config | `train_batch_size = 2`, `network_dim = 64`, `seed = 1145141919`, `save_every_n_steps = 50`, `sample_seed = 0` |
| Artifacts | `outputs/lllj_20260915_053826/lllj_s000050 … lllj_s000700` (every 50 steps), `lllj_samples/` (42 PNGs), TB scalars in `logs/lllj_20260915_053826/` |

How much of it is actually masked: of 30 sampled images, 28 carry alpha, and 46% of those have a mean
alpha below 0.99 (min 0.036, median 1.0). So roughly 240 of the 640 images get a genuinely non-opaque
loss weight and the rest are effectively unmasked — the reference is a **mixed** masked run, not a
fully masked one. That is a realistic workload and a reasonable thing for the `train` tier's geometry
probes to be read against, but it is not a clean all-masked control.

Two caveats when using it as a reference:

- **Its mask count is not in its log.** `start_train.sh` ends with
  `exec python -u trainer/main.py 1> >(grep -Ev "grid_desc|CandidateSelectionModel|metadata" >> /dev/null)`,
  so the child's stdout is discarded — the launch prints (`Loss masks: N/M samples`,
  `Checking/Generating latents cache...`) never reach `train.log`, which only carries stderr (tqdm,
  tracebacks). The harness does not have this problem: it launches `trainer/main.py` itself and
  captures stdout to the child log, which is how the `train` tier's "reports mask usage" check reads
  the line. For the live run, the count has to be derived from the dataset, as above.
- **Different ROCm stack.** The reference runs on HIP `7.14.60850` (`axl_rocm_7_14`); the harness
  children run `2.13.0+rocm10.0.0` / HIP `7.15.26333` (`axl`). Byte-level agreement with the reference
  is not expected across those two stacks; use it for shape and direction, not for bitwise comparison.

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

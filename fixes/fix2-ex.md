### gfx1201 page fault, second sighting: the faulting LoRA GEMM is chosen by the caption tokenizer

Same abort as `fix2.txt`, hit again on 2026-09-14 by an automated verification session
(`verify_mask_pipeline.py`) that trains real `trainer/main.py` subprocesses on 6 kanae images.
Same Tensile OOB, but a trigger condition fix2 did not cover: with `max_token_length = 225`
and captions of varying length, **the encoder sequence length changes from step to step**, so
the LoRA backward GEMM's `M` is not fixed by batch size and `network_dim` alone. Two runs whose
only config difference is `seed` behave differently: one never faults, the other faults at the
same step (101 of 120) in six out of six attempts, in three separate sessions and processes.

#### Problem

Child log (`torch 2.13.0+rocm10.0.0`, RX 9070 XT / gfx1201, driver 7.2.4), process exits with
`-6` (SIGABRT) and no Python traceback:

```text
epoch=51/60 step=101 loss=0.0371:  84%|████████▍ | 101/120 [02:05<00:21,  1.12s/it]Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
VGPU(0x...) hang analysis:
 grid=[6144, 1, 1], workgroup=[128, 1, 1], group_seg_size=25088, kernel_obj=0x..., ...
:0:rocdevice.cpp :3905: 53892590555 us:  Memory Fault Error [host: acitehost, GPU index: 0,
 faulting addr: 0x7fb3edc00000, kernel: Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x128_MI16x16x1_SN_..._ISA1201_..._WGMXCC1]
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x7fb3edc00000. Reason: Page not present or supervisor privilege.
GPU coredump: handler exited with error (status: 1)
```

Kernel log (six events, three sessions; timestamps are kernel uptime):

```text
[53548.227659] amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:157 vmid:9 pasid:1071)
[53548.227665] amdgpu 0000:03:00.0:  Process python pid 231641 thread python pid 231641
[53548.227666] amdgpu 0000:03:00.0:   in page starting at address 0x00007f8cba400000 from client 10
[53548.227668] amdgpu 0000:03:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x0090113B
[53548.227669] amdgpu 0000:03:00.0:     Faulty UTCL2 client ID: TCP (0x8)
[53548.227670] amdgpu 0000:03:00.0:     MORE_FAULTS: 0x1
[53548.227671] amdgpu 0000:03:00.0:     WALKER_ERROR: 0x5
[53548.227672] amdgpu 0000:03:00.0:     PERMISSION_FAULTS: 0x3
[53548.227673] amdgpu 0000:03:00.0:     MAPPING_ERROR: 0x1
[53548.227673] amdgpu 0000:03:00.0:     RW: 0x0

[53891.491110] ... pid 234258 ... address 0x00007fb3edc00000 ... GCVM_L2_PROTECTION_FAULT_STATUS:0x0090113B
[54080.526779] ... pid 234857 ... address 0x00007fd902200000 ... GCVM_L2_PROTECTION_FAULT_STATUS:0x0090113B
```

Every faulting address is 2 MiB aligned (`0x…79400000`, `0x…6be00000`, `0x…8a000000`,
`0x…ba400000`, `0x…3edc00000`, `0x…d902200000`) and every fault is on the TCP client, i.e. one
page past the end of a large (2 MiB-backed) allocation — the same fingerprint as fix2
(`0x7f4d79400000`). Five events report `0x0090113B`, one reports `0x0080113B`; the decoded
subfields are identical in all six.

#### Setup that produced it

| Key | Value |
| --- | --- |
| base model | `/opt/models/diffusers/waillu_170` (sdxl_base_v1-0) |
| data | 6 kanae images, bucket 1024×768, `bucket_reso_steps = 128` |
| `train_batch_size` / steps | 3 / 120 (60 epochs × 2), `gradient_accumulation_steps = 1` |
| `network_dim` / `alpha` / `dropout` | 36 / 18 / 0.2 |
| `max_token_length` | 225 (3 CLIP chunks cap), `clip_skip = 1` |
| precision | bf16, `gradient_checkpointing_unet = true`, `..._te = false` |
| optimizers | UNet schedule-free AdamW, TE AdamW cosine |
| save/sample cadence | `save_every_n_steps = 30` (samples at 30/60/90/120) |
| `seed` | **1145141919** vs **1145141920** — the only difference between the two groups |

This is not the fix1 bucket bug: `bucket_reso_steps` was 128 and the latents were 96×128
(kanae) / 96×160 (the stand dataset), both divisible by 16.

#### Repro pattern

| Runs | Config | Result |
| --- | --- | --- |
| 3 (session 1) + 3 (session 2) | `seed = 1145141919`, everything else as above | **ok**, 120/120 steps |
| 1 (session 4, the duplicate-run floor) | `seed = 1145141919` | **ok** |
| 3 (session 1: masked / gray-mask / unmasked) | `seed = 1145141920` | **page fault**, all at step 101 |
| 1 (session 2) | `seed = 1145141920` | **page fault**, step 101 |
| 2 attempts (session 3, the same run twice) | `seed = 1145141920` | **page fault**, both at step 101 |

So the abort is deterministic for a given (data, config, seed) — six for six, each in a fresh
process, each ~2 min in, each at step 101, always right after the step-90 sample generation.
It is *not* a random per-step lottery, which is what made it look "intermittent" at first.

#### Why the seed changes the GEMM shape

`shuffle_caption` picks its RNG from `cfg.seed + epoch + sha1(image_path)`, so the tag order —
and therefore the tokenization — depends on the run's seed. `text_processing.tokenize_long_prompt`
then chunks the prompt at 75 tokens per CLIP chunk, up to `ceil(225/75) = 3`:

```
M = train_batch_size × (chunks × 77) = 3 × {77, 154} = 231 or 462
```

The 6 kanae captions have 53 / 74 / 103 / 32 / 57 / 91 content tokens → chunk counts
1 / 1 / 2 / 1 / 1 / 2 → per-batch `M` is 231 or 462 depending on which images share a batch.
`462 = 7×64 + 14` is exactly the partial-tile remainder fix2 pins the fault on
(`K = 36 = 1×32 + 4`).

| seed | `M` histogram over 120 steps | first differing step |
| --- | --- | --- |
| 1145141919 (no fault) | `{462: 102, 231: 18}` | step 96: 231, step 97: 462 |
| 1145141920 (faults) | `{462: 101, 231: 19}` | step 101: **231** (a 462→231 switch) |

Two caveats, both verified in this session:

- The shape alone is not sufficient: `M = 462` occupies ~85% of all batches for *both* seeds,
  and the surviving runs spend most of their time in it. What differs is the *sequence*, and
  the abort lands on a shape-switch step (231↔462 forces the GEMM workspace to be freed and
  reallocated, which is when the overrun meets an unmapped page).
- The layout is reproducible: because the shape sequence is fixed by the seed, so is the
  allocator's packing, which is why the same step fails every time in a fresh process.

#### What got through it

`PYTORCH_NO_HIP_MEMORY_CACHING=1` (fix2's dodge) let the same `seed = 1145141920` run proceed
past step 101: it reached **119/120** before being stopped, at ~2.8 s/it instead of 1.12 s/it.

#### Operating an automated harness around it

- Detect the signature in the child's log (`HSA_STATUS_ERROR_MEMORY_FAULT`,
  `Memory access fault by GPU node`, `GCVM_L2_PROTECTION_FAULT`) — the process dies with
  `-6`, so a runner that only checks exit codes cannot tell this apart from a crash in the
  trainer.
- The abort kills the parent but **not** its DataLoader forkserver workers: they survive and
  keep holding memory. Reap them by matching the dead run's own working directory on their
  command line.
- `state.json` of the aborted run is left at `training` with a dead PID; `train_status`
  reconciles it to `error` on the next poll.
- The harness in `verify_mask_pipeline.py` now does all three, plus a retry that switches to
  `PYTORCH_NO_HIP_MEMORY_CACHING=1` on the second attempt instead of burning another identical
  one.

#### Candidate prevention (not yet tested)

Make the encoder sequence length constant so the GEMM shape never switches:
`family_sdxl.encode_prompts` calls `encode_prompt_batch` without `target_num_chunks`, so the
chunk count is per batch. Passing `target_num_chunks = ceil(cfg.max_token_length / 75)` would
pin every batch to 3 chunks (`M = 693 = 10×64 + 53`) at the cost of padding short captions.
That changes training semantics slightly and needs its own measurements before shipping.

Until then the practical rules from fix2 still hold: avoid `train_batch_size = 2` with
`network_dim = 36`, keep `bucket_reso_steps = 128`, and fall back to
`PYTORCH_NO_HIP_MEMORY_CACHING=1` when a run dies this way. Do **not** set
`HSA_SVM_GUARD_PAGES=0`: that only hides the abort.

#### Update 2026-09-14 — the trigger is the sampling pass, and the run can be cut to 8 steps

`fix2/` now packages a reproducer (`fix2/repro_real.py`) and the measurements behind it. What that
work adds to the above:

- The abort survives a host reboot (6/6 again on a clean boot), so it is not accumulated machine
  state, and it is reproducible from the packaged config in three minutes.
- **It is not the shapes.** With the cadence disabled (`save_every_n_steps = 0`) the identical run —
  same seed, same shapes, same optimizer — finishes 120/120 steps. The sampling pass is what leaves
  the allocator in the state where the overrun lands on an unmapped page; where the abort then
  surfaces (inside the sample itself vs. a descending shape transition 11 steps later) depends on how
  big the samples are.
- **It can be shortened from 101 steps to 8**: `repro_real.py --only fast` uses the same config with
  the cadence packed to every 2 steps and 512x512 samples, and dies at step 8 in ~70 s, same kernel,
  same `0x0080113B`, same three orphaned workers.
- The `seed` dependence in the table above is a *timing* effect, not a data effect: the seed picks
  which step carries the descending shape transition (`repro_real.py --grid` + `seed_step_shapes.py`
  print the per-step `M` trace), and the sample at step 90 supplies the poisoned layout.
- `fix2.txt`'s `batch = 2` + `network_dim = 36` combination does **not** fault here: it trains a full
  epoch, including with every caption padded so that file's derived shape (`seq 231`, `M = 462`,
  `K = 36`) is used on every step. Its `3 CLIP chunks` premise is unreachable on the kanae captions —
  measured, the longest is 145 tokens, i.e. two chunks.

#### Update 2026-09-15 — `crash.py` vs trainer: the fatal moment is `accelerator.prepare()`

A self-contained rewrite of the fastest abort (`network_dim = 32`, dies at step 4 through
unmodified `trainer/main.py`) lives in `fixes/fix2/crash.py`. It does **not** import `trainer/`
or spawn `trainer/main.py`. It matches the crashing row's shapes (`M = 462, 462, 462, 462, 231, …`),
dual optimizers, Accelerator bf16, real CLIP chunking, and the six packaged kanae images. Losses
agree with the trainer to ~1e-4. It survives **8/8 steps**, three times.

That is not "Python `import` causes a GPU fault". The two processes do different GPU work
before the first backward:

- Putting `crash.py`'s step loop inside the trainer process (`patch_mirror.py` rows
  `crash-body` / `true-crash`) **still dies at step 4**.
- Running `crash.child_main` from a `trainer/main.py` that only imports (`run-crash-child`)
  **survives**.
- `accelerator.prepare(UNet, TE1, TE2)` in the trainer, with optimizers left raw
  (`prepare-models`): **still dies at step 4**.
- `.to(device)` for UNet + TEs, `prepare` only the optimizers (`prepare-opts`), or
  `.to(device)` for everything (`no-prepare`): **12/12 steps, twice**.

`crash.py` also calls `acc.prepare(...)`. The API is not the poison; the trainer's
load → VAE cache → `empty_cache` → `prepare` sequence is. Step-1 CUDA alloc *sizes*
match (5735 events); HIP *segment* layout does not (trainer virtual span ~16 GB,
`crash.py` ~26 GB). The Tensile kernel still reads past a buffer (AMD). Whether that
read hits an unmapped page is this process's caching-allocator layout after
`prepare_model` (which also wraps `forward` in autocast — dropping `prepare` is
therefore **not** a numeric no-op and is not a shipped workaround).

Repo `trainer/` was not left patched: `patch_mirror.py` edits copies under `/tmp`.

Practical rules that *are* measured:

- Fastest 100% abort: `cd fixes/fix2 && python repro_real.py --row "network_dim 32"` (~25 s).
- `PYTORCH_NO_HIP_MEMORY_CACHING=1` avoids both abort routes (~2.2× slower).
- `network_dim` 12 / 24 / 32 abort at step 4 with no sampling; 8 / 16 / 20 / 28 / 36+ survive
  a 12-step window (dim 36 still dies at step 101 with sampling on).
- `HSA_SVM_GUARD_PAGES=0` is **not** an acceptable fix.

#### 2026-09-15 — the stack matters as much as the shapes

Everything above was measured on `torch 2.13.0+rocm10.0.0` (HIP `7.15.26333`). On
`torch 2.12.0+rocm7.14.1` (HIP `7.14.60850`, `environment.yml`'s pin at the time) the author reports the
packaged repros no longer abort. Local corroboration: the last attempt of the `network_dim 32` row
ran under that env and was stopped by hand at step 2 with no fault marker, where it used to die at
step 4; and the author's live run on that stack has trained 700+ steps with sampling every 50 without
faulting. `fixes/fix2/README.md` carries the same note, and its tables stay stack-specific.

The downgrade does not remove the overrun — the same Tensile kernels are still reading past their
buffers — so it is a stopgap. It is also a different rollback from the one `fixes/fix2.txt` tested:
that one was `2.12.0+rocm7.2` (HIP `7.2.53211`), which still faulted.

#### Related

- `fixes/fix2/README.md` — the reproducer, the failing-vs-surviving tables, `crash.py`, and
  the `prepare` ablation.
- `fixes/fix2.txt` — first sighting, `batch = 2` + `network_dim = 36` (kanae, seq 231).
- `fixes/fix1.txt` — bucket step 64 / latents not divisible by 16. Different bug, same dmesg family.
- ROCm/rocm-libraries#7992 — gfx1201 Tensile MT64x64x* OOB, status `0x0080113B`.
- Environment: python 3.14.7, torch 2.13.0+rocm10.0.0 (hip 7.15.26333), diffusers 0.40.0,
  transformers 5.16.1, peft 0.20.0, accelerate 1.14.0, driver 7.2.4-zen2-1-zen.
</content>

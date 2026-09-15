# `fixes/fix3/` — the first-step abort: `network_dim = 48` with `train_batch_size = 2`

> **Reproduced 2026-09-15 23:16 through 2026-09-16 00:30, before and after a host reboot.** `config.toml`
> as committed dies inside the **first** training step — encoding finishes, the epoch progress bar
> appears, and the process is killed by the gfx1201 Tensile page fault. Every run, in ~15 s. This is not
> the stopgap `doc/troubleshooting.md` describes: on the pinned stack (`torch 2.12.0+rocm7.14.1`, HIP
> `7.14.60850`) the trainer *does* abort, it just had not been pointed at this configuration before.

This directory is the reproducer, the evidence, and the analysis. The sibling packs:
[`fixes/fix2/README.md`](../fix2/README.md) (ROCm 10.0.0 stack, sampling-driven step-101 abort and a
step-4 rank abort), [`fixes/fix1.txt`](../fix1.txt) (the `bucket_reso_steps = 64` alignment bug — a
different bug with the same dmesg family).

## The path, in one table

The config is the repository's own `config.toml`: SDXL base, the author's 640-image dataset, warm latent
cache, `train_batch_size = 2`, `network_dim = 48`, `seed = 1145141919`, `max_token_length = 225`,
`bucket_reso_steps = 128`, bf16, UNet gradient checkpointing on.

| what ran | result |
| --- | --- |
| the live run (`lllj_20260915_231601`, `./start_train.sh`) | **FAULT** in the first step |
| `repro_step1.py`, before the reboot | **FAULT** (2/2) |
| `repro_step1.py`, after the reboot | **FAULT** (4/4, two configs) |
| same config, `network_dim = 64` | ok (that rank trained 3230 steps earlier the same day) |

Fault text (child side, `resources/live-run-fault.txt`):

```
Encoding Latents: 100%|██████████| 640/640 [00:03<00:00, 160.40it/s]
  0%|          | 0/3230 [00:00<?, ?it/s]
Dispatch Header =0xd02 (type=2, barrier=1, acquire=2, release=1), setup=3
grid=[2560, 1, 1], workgroup=[128, 1, 1]
:0:rocdevice.cpp :3678: 21989940725 us:  Memory Fault Error [host: acitehost, GPU index: 0,
    faulting addr: 0x7f1665600000, kernel: Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…_ISA1201_…]
Memory access fault by GPU node-1 (Agent handle: 0x55b8e004a9b0) on address 0x7f1665600000.
    Reason: Page not present or supervisor privilege.
```

That kernel is launched by a **text-encoder LoRA weight-gradient GEMM** — `[48, 308] @ [308, 1280]`, i.e.
`network_dim` x `train_batch_size * sequence length` x the text encoder's hidden size. §7 identifies it.

Kernel side (`journalctl -k`):

```
amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:101)
amdgpu 0000:03:00.0:  Process python pid 7407 thread python pid 7407
amdgpu 0000:03:00.0:   in page starting at address 0x00007f9278800000 from client 10
amdgpu 0000:03:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
amdgpu 0000:03:00.0:          Faulty UTCL2 client ID: TCP (0x8)
```

## 1. Reproduce it

```bash
conda activate axl_rocm_7_14
python fixes/fix3/repro_step1.py --only "step-1 abort" --in-place     # ~15 s, FAULT
```

`--in-place` runs against the data directory `config.toml` names, whose `.latents_cache/` is already
warm, so the run reads latents and writes nothing. Without it the harness stages a copy. The child is
the **unmodified** `trainer/main.py` in a throwaway mirror under `/tmp/axl-fix3-repro/` that symlinks
`trainer/` and holds a generated `config.toml`; model weights are referenced, never copied.

The packaged variant (no dependency on the author's dataset, ~19 MB in `resources/dataset/`):

```bash
python fixes/fix3/repro_step1.py --grid packaged --dataset fixes/fix3/resources/dataset
python fixes/fix3/repro_step1.py --grid packaged --dataset fixes/fix3/resources/dataset --keep-data
```

Both invocations abort. The first stages the six images and encodes their latents in-process; the second
reuses that directory and its cache. Running both is what rules out the in-process VAE encode as an
ingredient (§6).

## 2. What the fault is

A Tensile GEMM reads past the end of a buffer and the read lands on an unmapped page, so ROCr aborts the
process (`SIGABRT`, no Python traceback). Everything about the mechanism matches `fixes/fix2`:

- **The kernel is picked by the shape, and almost always the same one.** Every faulting row records the
  Tensile solution it died in. 44 of the 46 are the same 855-character string,
  `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…_ISA1201_…_WS32_WG64_2_1`, which
  is why this reads as one bug. The two exceptions are `network_dim = 36` (`MT64x64x64` instead of
  `MT64x128x16`) — a different rank and sequence length select a different member of the same Tensile
  family, and that member overruns too. The launch geometry recorded in the child log for the common case
  is `grid=[2560,1,1]`, `workgroup=[128,1,1]`, `private_seg_size=0`, `group_seg_size=1638`.
- **The address is not fixed.** `0x7f1665600000` (live), `0x7ff310400000`, `0x7f8534400000`,
  `0x7f1a43000000`, `0x7f9278800000` — every one of them **2 MiB-aligned**, i.e. the overrun crosses into
  the start of an unmapped 2 MiB segment. The address is the allocator's choice, not the kernel's.
- **The status word varies with the address, not with the kernel.** The same configuration produced
  `0x00801031` (WALKER_ERROR `0x0`, MAPPING_ERROR `0x0`) three times and `0x0080113B` (WALKER_ERROR `0x5`,
  MAPPING_ERROR `0x1`) once — both families `fixes/fix2` documents. `RW: 0x0` (a read) and client
  TCP `0x8` are constant, as they are in every fix2 sighting.
- **Two environment knobs flip the outcome**, both of which act on the address side and neither of which
  touches the kernel selection:

  | knob | effect |
  | --- | --- |
  | `PYTORCH_NO_HIP_MEMORY_CACHING=1` | aborts disappear — every tensor gets its own `hipMalloc`, so no buffer sits at the end of a cached segment |
  | `HSA_SVM_GUARD_PAGES=0` | aborts disappear — the overrun finds mapped memory instead of a guard page |

  Guard pages off is a **diagnostic, not a fix**: it converts the abort into a silent out-of-bounds read.
  `fixes/fix2` measured that this silent read changes no number (byte-identical losses and LoRA tensors
  against a `no-hip` run), which is why it is safe to use as evidence here and still wrong to ship.

So the overrun belongs to this Tensile family on gfx1201: the shapes pick which member runs, and the
allocator decides whether the read past the buffer reaches unmapped memory. This pack's contribution is
that a **configuration can pin that layout hard enough to make the abort 100% reproducible on the first
backward** — and that the shapes which control it are two specific numbers, not the dataset's geometry.

### It is the backward pass

`probe_op.py` logs every GEMM with the autograd state at the time it was dispatched (`probe_op.gemms.log`
in the row's mirror). In the faulting run the log ends inside the backward: the step's forward is 2480
GEMMs, one `BACKWARD START` marker follows it, and the process dies after 5413 further GEMMs — all of
them tagged `backward`, including the checkpoint-recomputation passes (which run with grad enabled
inside `loss.backward`). No forward-only GEMM is near the end of the log. This is the same phase
`fixes/fix2.txt` isolated (`loss.backward()`, not the forward).

## 3. Rule out the neighbours

- **Not the `bucket_reso_steps` bug** (`fixes/fix1.txt`): it is 128, and the offending bucket is
  1280x768 → latent 160x96, both divisible by 16.
- **Not the sampling pass** (`fixes/fix2`'s step-101 route): the abort happens at step 0, long before the
  first sample (`save_every_n_steps = 50`). `save_every_n_steps = 0` aborts identically.
- **Not the DataLoader**: `max_data_loader_n_workers = 0` + `persistent_workers = false` aborts
  identically, and no worker process is ever involved.
- **Not accumulated machine state**: the reboot changed nothing (kernel string identical, see §1).
- **Not the seed's numeric effect**: `lora_B` starts at zero and the fault needs no particular values;
  the seed moves *which* images form the first batch and how their captions shuffle, i.e. shapes.

## 4. The first step's shapes

`probe_batches.py` replays the trainer's own `LoraImageDataset` + `BucketBatchSampler` on the CPU, so the
first batch — the only one that matters for a step-0 abort — is known before anything starts:

| seed | batch 0 bucket | latent | CLIP chunks | encoder seq | `M` (LoRA GEMM) | runs |
| --- | --- | --- | --- | --- | --- | --- |
| 1145141919 | **1280x768** | 160x96 | [2, 2] | 154 | 308 | **9 FAULT / 9** |
| 1145141922 | **1280x768** | 160x96 | [2, 2] | 154 | 308 | **3 FAULT / 3** |
| 1145141924 | **1280x768** | 160x96 | [1, 1] | 77 | **154** | 0 FAULT / 2 |
| 1145141920 | 768x1408 | 96x176 | [2, 2] | 154 | 308 | 3 FAULT / 9 |
| 1145141921 | 1408x768 | 176x96 | [2, 2] | 154 | 308 | 2 FAULT / 3 |
| 1145141923 | 512x1792 | 64x224 | [2, 2] | 154 | 308 | 0 FAULT / 2 |

Read the first three rows together: **1919 and 1922 put the batch in the same bucket with the same `M`
and abort every time; 1924 puts it in the same bucket with a one-chunk caption, so `M = 154`, and never
aborts.** Comparing 1919 with 1920/1921/1923 shows the other half: the same `M` in a different bucket is
a coin flip rather than a certainty — and 1920 also demonstrates that a seed is not a property of the
config but a probability, since its nine runs split 3–6.

So neither the seed nor the bucket is a clean predictor on its own. What the rows that never fault share
is `M = 154`; what every row that always faults shares is `M = 308` *and* bucket `1280x768`. §7 explains
why those two numbers matter: `M` and `K` of the offending GEMM are `network_dim` and
`batch x sequence`, and the bucket only moves the allocations around it.

The dataset contributes to that set entirely through the bucket: 640 images fall into 12 buckets, from
`1280x768` (121 images) and `1408x768` (258) up to `512x2304`, and each caption needs 1–3 CLIP chunks
(143 / 432 / 65 images respectively).

## 5. How narrow the trigger is

Every row below changes exactly one thing in the packaged config, and the scans are ordered from the
coarsest question to the finest: which single knobs matter, then which shapes, then which rank. Because a
single "ok" can be allocator luck, the stability rows were run three times and the two most interesting
seeds nine. The tables are generated from the recorded JSON by `summarize.py`, so they cannot drift from
the runs; §5.1 and §5.2 say what they mean.

### Repeat runs: which rows hold their verdict

| row | pass 1 | pass 2 | pass 3 | peak VRAM |
| --- | --- | --- | --- | --- |
| stability: dim 48, batch 2 (the abort) | **FAULT** | **FAULT** | **FAULT** | 9.91, 9.79, 9.91 GB |
| stability: dim 64, batch 2 | ok | ok | ok | 10.14, 10.98, 10.34 GB |
| stability: dim 48, batch 1 | ok | ok | ok | 9.37, 9.65, 10.59 GB |
| stability: dim 48, batch 4 | vram-kill | ok | vram-kill | 12.87, 12.1, 13.02 GB |
| stability: dim 48, dropout 0 | ok | ok | ok | 10.4, 10.3, 10.29 GB |
| stability: dim 48, te checkpointing on | ok | ok | ok | 9.88, 9.9, 10.0 GB |
| stability: dim 48, hipMalloc per tensor | ok | ok | ok | 10.53, 10.56, 10.55 GB |
| stability: dim 48, guard pages off | ok | ok | ok | 10.49, 10.57, 10.43 GB |

### Geometry rows (whole dataset, one knob changed)

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| shape: no bucketing (train_resolution 1024) | 0 | **FAULT** | `…MT64x128x16…` | 137.4 | 10.09 GB |
| shape: bucket_reso_steps 64 (fixes/fix1 territory) | 1 | **FAULT** | `…MT64x128x16…` | 140.3 | 11.29 GB |
| shape: max_token_length 75 (one CLIP chunk) | 2 | ok |  | 16.5 | 10.69 GB |
| shape: max_token_length 150 (two CLIP chunks) | 0 | **FAULT** | `…MT64x128x16…` | 14.9 | 9.91 GB |

### Rank scan over three steps (buckets 1280x768, 1408x768, 1408x768)

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| dim short scan: network_dim 8 | 3 | ok |  | 16.9 | 11.33 GB |
| dim short scan: network_dim 12 | 3 | ok |  | 17.3 | 11.32 GB |
| dim short scan: network_dim 16 | 3 | ok |  | 17.3 | 11.47 GB |
| dim short scan: network_dim 20 | 3 | ok |  | 17.3 | 11.58 GB |
| dim short scan: network_dim 24 | 1 | **FAULT** | `…MT64x128x16…` | 16.1 | 11.23 GB |
| dim short scan: network_dim 28 | 3 | ok |  | 17.3 | 12.0 GB |
| dim short scan: network_dim 32 | 3 | ok |  | 17.3 | 12.16 GB |
| dim short scan: network_dim 36 | 2 | **FAULT** | Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA1_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA0_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB32_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT2_2_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA2_NLCB1_ONLL0_PAP0_PGL0_PGR2_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS0_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1] | 17.3 | 12.37 GB |
| dim short scan: network_dim 40 | 3 | ok |  | 17.3 | 12.54 GB |
| dim short scan: network_dim 44 | 3 | ok |  | 17.3 | 12.93 GB |
| dim short scan: network_dim 48 | 0 | **FAULT** | `…MT64x128x16…` | 14.9 | 9.58 GB |
| dim short scan: network_dim 52 | 3 | ok |  | 17.8 | 13.12 GB |
| dim short scan: network_dim 56 | 3 | ok |  | 17.7 | 13.21 GB |
| dim short scan: network_dim 60 | 3 | ok |  | 17.7 | 13.71 GB |
| dim short scan: network_dim 64 | 3 | ok |  | 17.7 | 14.18 GB |

### Rank scan over twelve steps

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| dim scan: network_dim 8 | 12 | ok |  | 26.9 | 12.28 GB |
| dim scan: network_dim 12 | 12 | ok |  | 26.9 | 12.63 GB |
| dim scan: network_dim 16 | 12 | ok |  | 26.9 | 12.87 GB |
| dim scan: network_dim 20 | 12 | ok |  | 27.3 | 13.19 GB |
| dim scan: network_dim 24 | 1 | **FAULT** | `…MT64x128x16…` | 16.1 | 10.69 GB |
| dim scan: network_dim 28 | 6 | **FAULT** | `…MT64x128x16…` | 21.7 | 12.89 GB |
| dim scan: network_dim 32 | 11 | vram-kill |  | 26.6 | 13.97 GB |
| dim scan: network_dim 36 | 2 | **FAULT** | Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA1_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA0_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB32_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT2_2_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA2_NLCB1_ONLL0_PAP0_PGL0_PGR2_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS0_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1] | 17.7 | 12.37 GB |
| dim scan: network_dim 40 | 5 | vram-kill |  | 20.2 | 13.68 GB |
| dim scan: network_dim 44 | 5 | vram-kill |  | 20.1 | 13.98 GB |
| dim scan: network_dim 48 | 0 | **FAULT** | `…MT64x128x16…` | 14.9 | 9.91 GB |
| dim scan: network_dim 52 | 5 | vram-kill |  | 20.1 | 14.18 GB |
| dim scan: network_dim 56 | 5 | vram-kill |  | 20.1 | 14.23 GB |
| dim scan: network_dim 60 | 2 | vram-kill |  | 17.3 | 13.71 GB |
| dim scan: network_dim 64 | 2 | vram-kill |  | 16.9 | 13.88 GB |

### `train_batch_size` x `network_dim`

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| batch 1: dim 24 | 2 | ok |  | 0.0 | 0.0 GB |
| batch 1: dim 32 | 2 | ok |  | 15.3 | 10.15 GB |
| batch 1: dim 48 | 2 | ok |  | 15.7 | 10.98 GB |
| batch 1: dim 64 | 2 | ok |  | 15.7 | 11.66 GB |
| batch 2: dim 24 | 1 | **FAULT** | `…MT64x128x16…` | 16.1 | 10.51 GB |
| batch 2: dim 32 | 2 | ok |  | 16.1 | 10.9 GB |
| batch 2: dim 48 | 0 | **FAULT** | `…MT64x128x16…` | 14.9 | 9.91 GB |
| batch 2: dim 64 | 2 | ok |  | 16.5 | 12.07 GB |
| batch 3: dim 24 | 2 | ok |  | 20.5 | 12.69 GB |
| batch 3: dim 32 | 2 | ok |  | 17.3 | 12.04 GB |
| batch 3: dim 48 | 2 | ok |  | 17.3 | 11.62 GB |
| batch 3: dim 64 | 2 | ok |  | 17.8 | 12.73 GB |
| batch 4: dim 24 | 2 | ok |  | 19.8 | 12.81 GB |
| batch 4: dim 32 | 2 | ok |  | 17.7 | 12.76 GB |
| batch 4: dim 48 | 2 | ok |  | 18.1 | 13.05 GB |
| batch 4: dim 64 | 2 | ok |  | 18.1 | 13.7 GB |

### Seed scan (three runs per seed)

| row | pass 1 | pass 2 | pass 3 | peak VRAM |
| --- | --- | --- | --- | --- |
| seed scan: 1145141919 | **FAULT** | **FAULT** | **FAULT** | 9.68, 9.62, 9.93 GB |
| seed scan: 1145141920 | ok | **FAULT** | **FAULT** | 0.0, 10.29, 10.06 GB |
| seed scan: 1145141921 | ok | **FAULT** | **FAULT** | 10.06, 9.72, 10.06 GB |
| seed scan: 1145141922 | **FAULT** | **FAULT** | **FAULT** | 9.62, 9.91, 9.91 GB |
| seed scan: 1145141923 | ok | ok | ok | 9.47, 9.78, 9.55 GB |
| seed scan: 1145141924 | ok | ok | ok | 9.62, 9.61, 9.64 GB |

### The committed seed, six more runs

| row | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed scan: 1145141919 | **FAULT** | **FAULT** | **FAULT** | **FAULT** | **FAULT** | **FAULT** | 9.91, 9.91, 9.91, 9.91, 9.91, 9.68 GB |

### A seed that is only sometimes fatal, six more runs

| row | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed scan: 1145141920 | **FAULT** | ok | ok | **FAULT** | ok | ok | 10.31, 0.0, 10.3, 10.77, 10.31, 10.31 GB |

### Single-bucket datasets: bucket x rank

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| bucket 1280x768: dim 24 | 1 | ok |  | 10.9 | 9.34 GB |
| bucket 1280x768: dim 48 | 0 | **FAULT** | `…MT64x128x16…` | 10.9 | 10.13 GB |
| bucket 1280x768: dim 64 | 1 | ok |  | 11.3 | 10.34 GB |
| bucket 1408x768: dim 24 | 1 | ok |  | 10.9 | 10.0 GB |
| bucket 1408x768: dim 48 | 1 | ok |  | 11.3 | 10.22 GB |
| bucket 1408x768: dim 64 | 1 | ok |  | 11.3 | 10.55 GB |
| bucket 512x1920: dim 24 | 1 | ok |  | 10.9 | 9.68 GB |
| bucket 512x1920: dim 48 | 0 | **FAULT** | `…MT64x128x16…` | 10.9 | 9.67 GB |
| bucket 512x1920: dim 64 | 1 | ok |  | 12.1 | 9.85 GB |
| bucket 512x2176: dim 24 | 0 | **FAULT** | `…MT64x128x16…` | 10.9 | 9.93 GB |
| bucket 512x2176: dim 48 | 0 | **FAULT** | `…MT64x128x16…` | 10.9 | 10.06 GB |
| bucket 512x2176: dim 64 | 1 | ok |  | 11.3 | 10.26 GB |

### First pass over the whole dataset (superseded by the rows above)

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| step-1 abort: repo config (dim 48, batch 2) | 0 | **FAULT** | `…MT64x128x16…` | 22.1 | 10.13 GB |
| step-1 abort, repeat | 0 | **FAULT** | `…MT64x128x16…` | 14.9 | 9.81 GB |
| dim 64 (the rank that trained 3230 steps on 2026-09-15) | 1 | ok |  | 15.3 | 10.36 GB |
| dim 48, batch 1 | 1 | ok |  | 14.9 | 9.37 GB |
| dim 48, batch 4 | 1 | ok |  | 16.1 | 12.25 GB |
| dim 48, hipMalloc per tensor | 1 | ok |  | 16.9 | 10.58 GB |
| dim 48, guard pages off | 1 | ok |  | 15.3 | 10.01 GB |
| dim 48, no dataloader workers | 0 | **FAULT** | `…MT64x128x16…` | 13.3 | 10.48 GB |
| dim 48, no sampling cadence | 0 | **FAULT** | `…MT64x128x16…` | 14.9 | 9.75 GB |
| dim 48, te checkpointing on | 1 | ok |  | 15.3 | 9.79 GB |
| dim 48, dropout 0 | 1 | ok |  | 15.3 | 10.28 GB |
| dim 48, seed 1145141920 | 1 | ok |  | 15.3 | 9.86 GB |
| dim 48, 6 packaged images | 1 | ok |  | 12.9 | 10.48 GB |
| dim 48, 2 packaged images | 1 | ok |  | 12.5 | 9.46 GB |
| dim 48, 12 packaged images | 1 | ok |  | 13.7 | 10.27 GB |
| dim 48, no loss mask (alpha stripped) | 1 | ok |  | 12.1 | 9.45 GB |

### Packaged mini-dataset, cold latent cache

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| packaged mini-dataset (6 images, bucket 1280x768, 2 chunks) | 0 | **FAULT** | `…MT64x128x16…` | 12.1 | 10.16 GB |

### Packaged mini-dataset, warm latent cache

| row | steps | verdict | kernel | seconds | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| packaged mini-dataset (6 images, bucket 1280x768, 2 chunks) | 0 | **FAULT** | `…MT64x128x16…` | 10.9 | 10.05 GB |

### 5.1 The fatal quantity is `M`, not the rank and not the bucket

The scans below are read through `M = train_batch_size * 77 * (CLIP chunks the batch's captions need)` —
the row count of the cross-attention LoRA GEMM, and, in the GEMM that actually faults (§7), its `K`
dimension. Two rows change `M` and nothing else — same seed, same images, same first bucket, same rank:

| row | first bucket | chunks | `M` | result |
| --- | --- | --- | --- | --- |
| `max_token_length = 75` (one chunk) | 1280x768 | [1, 1] | **154** | ok over 2 steps |
| `max_token_length = 150` (two chunks) | 1280x768 | [2, 2] | **308** | **FAULT at step 0** |

Forty-four of the forty-six faults happen on a step whose `M` is `308`; the two exceptions are
`network_dim = 36` on a step whose `M` is `462`, and those are also the only two that die in the other
Tensile solution (`MT64x64x64`). No row anywhere in this pack has faulted on a step whose `M` is `154`.

### 5.2 Which bucket shapes kill which ranks

Single-bucket datasets isolate the bucket: each row trains on four images that all fall into one bucket,
so every step has the same spatial shape and the only things varying between rows are the bucket and the
rank. The `M` column was measured on each staged directory with `probe_batches.py --data-dir` afterwards,
because the caption shuffle is keyed on the image path and a staged copy is not the original.

| bucket | latent | `M` at batch 2 | rank 24 | rank 48 | rank 64 |
| --- | --- | --- | --- | --- | --- |
| **1280x768** | 160x96 | 308 | ok | **FAULT** | ok |
| **512x1920** | 64x240 | 308 | ok | **FAULT** | ok |
| **512x2176** | 64x272 | 308 | **FAULT** | **FAULT** | ok |
| 1408x768 | 176x96 | 462 | ok | ok | ok |

In this grid every fatal cell is on an `M = 308` step, and the one bucket whose captions give `M = 462`
is fine at all three ranks tested there — though rank 36 does abort on an `M = 462` step in the real
sequence, where it picks the other kernel (§7). `M = 308` is not sufficient on its own either:
`(1280x768, 24)`, `(1280x768, 64)` and `(512x1920, 24/64)` share the row count with the fatal cells and
survive.

The rank axis on the real dataset (three steps: bucket 1280x768, then 1408x768 twice) adds the ranks the
single-bucket grid did not cover — `24`, `36` and `48` abort at steps 1, 2 and 0 respectively, every other
rank survives all three steps (table above, `grid-dim3.json`). Over the longer twelve-step window
(`grid-dim.json`) rank `28` also aborts, at step 6 — all three of those steps are bucket `1408x768`.

`correlate.py --only dim` prints that scan against the step→bucket map, which is how a "died at step N"
row is turned into a bucket:

| bucket | latent | rank 24 | rank 28 | rank 36 | rank 48 |
| --- | --- | --- | --- | --- | --- |
| 1280x768 | 160x96 | ok | ok | ok | **FAULT** `M=308` step 0 |
| 1408x768 | 176x96 | **FAULT** `M=308` step 1 | **FAULT** `M=308` step 6 | **FAULT** `M=462` step 2 | — |
| 512x1920 | 64x240 | — | ok | — | — |

The dash in the last cell is the shape of the evidence, not a survivor: rank 48 aborts at step 0, so it
never reaches the 1408x768 steps. Filling that cell needs a run that starts in `1408x768`, which is what
the single-bucket grid does — and there `(1408x768, 48)` survives (`M = 462`).

## 6. The packaged dataset: what reproduces and what does not

Two results here are as useful as the positive one, and both were surprises.

**(a) The first six images of the data directory do not reproduce.** Staging the first N images by name
takes `0001–0006`, which are 3.3k–3.7k-pixel-tall crops: they bucket to `640x1664`, `512x2048`,
`512x1920`, and the sampler's first batch is a **single** image (`M = 154`) in `640x1664`. Every one of
those rows survives — including the row with the loss mask stripped, which is why the mask is not a
suspect. The mini-dataset therefore has to be chosen by *measured shape*, not by name.

**(b) The packaged mini-dataset reproduces it, cold cache included.** `resources/dataset/` holds six
`2120x1280` images from the 1280x768 bucket, each with a caption that tokenizes into exactly two CLIP
chunks; `probe_batches.py` reports bucket `1280x768` / seq 154 / `M = 308` on **every** batch, versus
the real run's alternating 308/308/462:

```
{"batch": 0, "bucket": "1280x768", "latent": "160x96", "chunks": [2, 2], "M_lora_gemm": 308}
{"batch": 1, "bucket": "1280x768", "latent": "160x96", "chunks": [2, 2], "M_lora_gemm": 308}
```

Both packaged runs abort — the one that stages the images and encodes their latents in-process, and the
`--keep-data` repeat that reads the cache the first one wrote:

| row | data | latent cache | result |
| --- | --- | --- | --- |
| `packaged mini-dataset` | 6 staged images | cold (encoded by this run) | **FAULT at step 0** |
| `packaged mini-dataset` | the same directory, `--keep-data` | warm | **FAULT at step 0** |

That is a useful negative for the hypothesis §6 started from: the in-process VAE encode is *not*
required to reach the bad layout, so the write-up of the mechanism does not have to lean on it. It also
makes the pack self-contained — 19 MB of images, no dependency on the author's dataset.

## 7. Which operation overruns

`probe_op.py` replaces the trainer's entry point (`--child-script`) and logs every GEMM, convolution and
attention call with its operand shapes, both to a file and to its own stdout — the stream MIOpen writes
its kernel selection to. Run under `HIP_LAUNCH_BLOCKING=1`, every launch is synchronous, so the last
line before the fault is the operation whose kernel overran:

```bash
python fixes/fix3/repro_step1.py --row "step-1 abort: repo config (dim 48, batch 2)" --in-place \
    --child-script fixes/fix3/probe_op.py --env HIP_LAUNCH_BLOCKING=1
```

**The faulting operation is a text-encoder LoRA A-weight gradient**, `grad_A = grad_outᵀ @ input`:

```
OP 8344 backward aten.mm.default [[48, 308], [308, 1280]]
VGPU=0x5645c6184410 SWq=… HWq=…, id=1
Dispatch Header =0xd02 (type=2, barrier=1, acquire=2, release=1), setup=3
grid=[2560, 1, 1], workgroup=[128, 1, 1]
:0:rocdevice.cpp :3678: 3194366185 us:  Memory Fault Error [host: acitehost, GPU index: 0,
    faulting addr: 0x7f2d68800000, kernel: Cijk_…_MT64x128x16_…IS A1201_…]
```

Two independent probe runs end on that same shape. Reading the operands:

| dimension | value | where it comes from |
| --- | --- | --- |
| `M` | **48** | `network_dim` |
| `K` | **308** | `train_batch_size` (2) x encoder sequence length (154) |
| `N` | **1280** | the text encoder's hidden size (`text_encoder_2`) |

The op sits in a repeating per-layer backward pattern, one layer per group of four GEMMs — base dgrad,
`grad_B`, `lora_A` dgrad, `grad_A` — and `N = 1280` puts it in `text_encoder_2`'s attention projections
(`q_proj`/`k_proj`/`v_proj`/`out_proj`). The UNet's cross-attention LoRA has `N = 2048` (its input
width) and the UNet's spatial layers have `K = batch x latent pixels`, so neither matches.

**This is why the rank is the strongest knob in this bug.** `network_dim` is literally this GEMM's `M`,
so changing the rank changes which Tensile kernel is selected for it: `M = 48` is three quarters of the
`MT64` tile (`64` exactly, one full tile, is the one rank that never aborted in any scan), and
`K = 308 = 19x16 + 4` is a partial `K` tile as well. The bucket, by contrast, **does not appear in this
GEMM at all** — the text encoder only ever sees `batch x sequence` tokens. That is the correlation the
two shape axes really have:

- `network_dim` and `batch x chunks` *are* the offending GEMM's dimensions, so they decide which kernel
  is launched and whether it overruns;
- the bucket decides nothing about that GEMM, and only sets how the rest of the step's allocations are
  laid out around it — which is why the same `(rank, M)` cell aborts in one bucket and not in another.

A caveat worth keeping: without `HIP_LAUNCH_BLOCKING`, the *same* configuration surfaces the fault at a
different point in the stream (the plain runs' logs end on MIOpen's candidate-selection dump for a
3x3 convolution, `M = 30720 / N = 384 / K = 2880`, i.e. the level-0 feature map of this bucket). With
launches serialized, the fault is attributable to the GEMM that launched it; asynchronous, it is
attributed to whatever the queue reached next. Both sightings are in the kernel `MT64x128x16`, so they
are the same overrun, reported at different points — which is itself a statement about how the overrun
and the report point decouple.

## 8. Workarounds, measured

| workaround | cost | verdict |
| --- | --- | --- |
| `network_dim` 64 (the rank that trained 3230 steps on 2026-09-15) | none on this card | works, but the fatal set is a scatter, not a rule — `M = 64` is exactly one `MT64` tile, and every other rank tested is a partial tile (§7) |
| `train_batch_size` 1 | ~2x slower per epoch | works, and it halves the offending GEMM's `K` (154 instead of 308) — but it also changes which bucket the first batch draws, so the two effects are not separated by this row alone |
| `network_dim` in {8, 12, 16, 20, 28, 32, 40, 44, 52, 56, 60} | none | survived the tests run here; the twelve-step scan shows the scatter is wider than three steps can see (rank 28 is fine over 3 steps and fatal at step 6) |
| `network_dropout = 0.0` | changes training | survived 3/3; it changes the LoRA forward's graph, i.e. the layout |
| `gradient_checkpointing_te = true` | slower TE backward | survived 3/3; same reason |
| `max_token_length = 75` | truncates captions | works, and is the cheapest way to keep `K = 154` — but it silently drops caption content |
| `PYTORCH_NO_HIP_MEMORY_CACHING=1` | ~2.2–2.9x slower (fixes/fix2's measurement) | works, and is the only knob that addresses the overrun's *fatality* rather than its address |
| `HSA_SVM_GUARD_PAGES=0` | none | **do not use**: it turns the abort into a silent overread |
| smaller samples | — | irrelevant here: the abort needs no sample |
| `flush_memory_every_step` | small | already on in the config, and does not help (fixes/fix2's step-101 route, and this one) |

The honest summary: **this is a Tensile/ROCm defect on gfx1201 that no trainer-side knob fixes.** The
trainer can stay off the rank/batch/chunk combinations that put this GEMM's operands next to a hole, and
`doc/troubleshooting.md` should stop claiming that this stack is immune.

## 9. The harness, and the OOM that froze the machine

`repro_step1.py` grew safety rails after an accident worth recording, because the neighbourhood of this
bug is out-of-memory rather than faults:

- **Disabling UNet gradient checkpointing at this batch/bucket does not fit on a 16 GiB card.** Measured
  2026-09-15: `torch.OutOfMemoryError: Tried to allocate 20.00 MiB … 14.99 GiB is allocated by PyTorch`
  inside `gelu`. That run exhausted the GPU and took the desktop session down with it; the row is now
  marked `skip` and only runs behind the watchdog.
- **`--vram-limit-gb` (default 13.5)** polls `/sys/class/drm/card*/device/mem_info_vram_used` every
  400 ms and `SIGKILL`s the row's process group when it crosses the limit, and each row waits for the
  previous one's VRAM to drain below 1.5 GiB before starting. `--row-timeout` (default 240 s) bounds a
  hung row. Rows record `peak_vram_gb`, so a "survives" verdict can always be read together with how
  close to the card's limit it ran.
- Rows that fault are killed the moment the KFD message reaches the log, instead of waiting out ROCm's
  ~30 s teardown; orphaned forkserver workers are reaped by matching the run's own mirror path.

Two calibration notes, because both limits were wrong at first and the wrong values are visible in the
recorded JSON:

- The limit started at 12.5 GiB and clipped a row that was fine: `dim scan: network_dim 44` was killed at
  12.7 GiB **in its second training step** (the step is what reaches ~12.7 GiB here — `peak_vram_gb` and
  the row's log both say so; the first guess that it had died during latent encoding was wrong). The
  default is now 13.5 GiB.
- The watchdog is armed only once encoding reports done. That change is defensive, not evidence-driven:
  encoding is the pass every successful run on this box already performs, so a row killed there would be
  a false positive, but no such kill was ever observed.
- The twelve-step rank scan still loses seven ranks to the watchdog (`vram-kill` rows in
  `grid-dim.json`) because twelve steps of the large buckets accumulate past 14 GiB. Those ranks get
  their verdicts from the three-step scan (`grid-dim3.json`) and the single-bucket grid instead.

## 10. Files

| File | What it does |
| --- | --- |
| `repro_step1.py` | The reproducer. Builds a mirror, writes a config, stages a dataset, runs the unmodified `trainer/main.py`, reports fault/ok per row. `--grid default\|stability\|shape\|bucket\|dim\|dim3\|batch\|seed\|packaged`, `--row`, `--only`, `--repeats`, `--env`, `--child-script`, `--in-place`, `--keep-data`, `--prewarm`, the VRAM watchdog and the row timeout. |
| `probe_batches.py` | CPU-only. Replays the trainer's dataset + bucket sampler to print each bucket, caption chunk count, encoder sequence length and `M`, per seed — the step→bucket map the scans are read against (`--json`), and the per-bucket image names the single-bucket datasets are built from. |
| `probe_op.py` | Runs inside the mirror as `trainer/main.py`'s replacement: logs every GEMM, convolution and attention call with its operand shapes, to `probe_op.gemms.log` and to stdout (where MIOpen's kernel selection lands), tagged `forward`/`backward`, plus `BACKWARD START` markers. |
| `correlate.py` | Joins a scan with the step→bucket map: prints the knob × bucket matrix and the list of fatal `(bucket, knob)` cells. |
| `summarize.py` | Renders the scan JSONs as this file's tables, so the numbers cannot drift from the runs. |
| `resources/config.toml` | The failing configuration (the repository's `config.toml`; author paths, rewritten per row). |
| `resources/dataset/` | The six-image packaged mini-dataset (§6). |
| `resources/step-buckets.json` | The first 24 steps of the real run as `(step, bucket, latent, tokens, chunks, seq, M)` — the map that turns "died at step 1" into "bucket 1408x768". |
| `resources/grid-*.json`, `resources/packaged-*.json`, `resources/seed-*.json`, `resources/probe-op*.json` | Raw per-row records: verdict, exit code, step reached, peak VRAM, kernel, log path. |
| `resources/live-run-fault.txt` | The production run's own log tail, including the faulting dispatch header. |
| `resources/post-reboot-dmesg.txt` | The kernel-side records of the post-reboot faults: timestamps, PIDs, addresses, status words. |
| `resources/probe-op.md` | What `probe_op.py` found: the faulting operation, its three dimensions, and why the async log points somewhere else. |

## 11. Open questions

- **Why *this* kernel and not its neighbours.** The offending GEMM's shape is now known (`[r, B*seq] @
  [B*seq, 1280]`), and so is the fact that some `(M, K)` pairs abort while others do not — but which of
  the Tensile solutions picked for those pairs drops the bounds check is still a ROCm-side question.
  `AMD_LOG_LEVEL=4 AMD_LOG_MASK=0x60000` (as in `fixes/fix2`) dumps the lookups around the abort.
- **Why the plain (non-blocking) run reports the fault at a convolution.** §7's caveat measures it, but
  whether the conv's kernel is also overrunning (and merely reported second) or is just the next sync
  point is not settled. A kernel-name trace with `AMD_LOG_MASK=0x2` would show both launches.
- **A minimal process that still aborts.** Everything in `fixes/fix2/crash.py`-style attempts survives;
  this pack adds a *config* that does not, but the abort still needs the real trainer's allocation
  history — the standalone attempts load the same modules and never sit this GEMM's operand next to a
  hole. Confirming that would turn the workaround table in §8 into a rule.
- **Whether the 19 MB packaged dataset should shrink further.** Six images were chosen to make every
  batch `M = 308`; fewer images would still fault, but the set was not minimised beyond that.

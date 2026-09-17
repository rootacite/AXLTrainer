# bf16 hipBLASLt kernels read past their operands

**The claim.** On gfx1201 with ROCm 7.14.1, the bf16 Tensile kernels that hipBLASLt dispatches
read past the end of their A/B operands on shapes whose K is not a whole number of `DepthU` tiles or
whose free dimensions are not whole macro tiles. The over-read is small and bounded (measured at
most 4 KiB past the operand; the shape that killed the training run works out to 2 KiB), it does not
change the result, and it is invisible while the operand's end sits comfortably inside a mapped
page — but when the allocation ends within that distance of an unmapped page the read faults, and a
GPU page fault kills the process. That is what killed the training run at step 1352. The fault is
reproducible on demand from the library's own code object on the production shape and the production
kernel.

**Status.** This is the conclusion of the method-2 work in `fixes/hip1/` (hand-launch every bf16
kernel with guard-paged operands). The full population scan — how many of the 1830 launchable
kernels do it — was **not** run; the milestone that would produce it is M4 in the plan and is still
open.

**Environment.** HIP 7.14.60850, `torch 2.12.0+rocm7.14.1`, conda env `axl_rocm_7_14`, AMD RX 9070 XT
(gfx1201). The argument-block conventions below were read out of a checkout of the matching ROCm
source, `/home/acite/Deeppin/rocm-libraries` at tag `therock-7.14.1`
(commit `cd9574023093742434e8c992d13b89ab9a6c1cf8`); paths like `projects/hipblaslt/...` are relative
to that checkout.

---

## 1. What is established, and how firmly

| # | Statement | How it is known | Firmness |
| --- | --- | --- | --- |
| 1 | A kernel from the library reads past the end of its operand | The code object is loaded directly and launched with operands whose last byte is the last mapped byte before a guard page; the read faults on the guard page | measured |
| 2 | It is the production kernel on the production shape | Same 855-char kernel name as the running process had loaded; the kernel's own free/sum sizes on the trainer's shape make a 20-workgroup launch, matching the live fault | measured, matched |
| 3 | The overrun is a tail over-read, not a wild pointer | Overrun distance lies in `(0, 4096]` bytes; the K-tail arithmetic for that kernel (`DepthU 16 × ld_B 64 elements × 2 B = 2048 B`) fits inside the interval | distance measured, arithmetic inferred |
| 4 | It happens where a boundary check would be needed and not elsewhere | Of the kernel's 10 ladder shapes, the one that fills its macro tiles exactly is clean at pad 0; 6 of the edge/K-tail shapes fault | measured |
| 5 | The crash additionally needs an unlucky allocator layout | The same shape ran 210 times before the fatal step; with 4 KiB of mapped, zeroed slack after the operand the same launch is clean | measured |
| 6 | The over-read data does not reach the result | With 4 KiB (and 64 KiB, 256 KiB) of zeroed slack the D crc equals the exact reference | measured |
| 7 | None of this is an artefact of the harness | Every non-faulting shape returns the exact reference D; 132 control jobs spanning all four bf16 layout families, 3 runs: 132 clean, 0 wrong / 0 noop / 0 badcopy | measured |

---

## 2. The evidence chain

### 2.1 The crashed run

Run `lllj_20260916_062841` (`/home/acite/LLM/axltrainer/outputs/lllj_20260916_062841`), last
checkpoint `lllj_s001350`, samples through step 1350 written 07:06:10 and 07:06:37, TensorBoard last
write 07:06:39, kernel log page fault at 07:06:40 — so it died inside **step 1352**.

Reconstructing the batch order from the repo's own `BucketBatchSampler`/`LoraImageDataset` plus both
tokenizers (same code, same seed, dataset unchanged since 2026-09-15) gives step 1352 = epoch 4,
batch index 59 = **bucket 1280x768**, images `0248.png` + `0171.png`, chunks `[1, 2]` → sequence 154
→ **K = 308**; the checkpoint's own metadata says `ss_network_dim = 64` (alpha 48).

The GPU log (`journalctl -k -b -1`) matches the live fault of `fixes/fix3` field for field:

```
[gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:18578)
Process python pid 167085
page 0x7f7007400000
GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
Faulty UTCL2 client ID: TCP (0x8)
```

The operation is `aten.mm([[64, 308], [308, 1280]])` — the text_encoder_2 LoRA A-weight gradient, i.e.
the GEMM whose M is `network_dim`.

### 2.2 The kernel

`fixes/hip1/resources/bf16-kernel-inventory.json` decodes the four bf16 Tensile libraries hipBLASLt
installs for gfx1201 (`.co` code objects + `.dat.zlib` solution libraries):

- 2105 `_Type_BB_` (plain bf16) kernels; 1830 launchable at all; 275 skipped because they need the
  split-U workspace (`globalSplitU > 1`).
- Per library: 372 `Ailk_Bjlk`, 406 `Ailk_Bljk`, 303 `Alik_Bjlk`, 749 `Alik_Bljk`.

The kernel the trainer faulted in is an 855-character name:

```
Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…_GRVWA8_GRVWB8_…_WS32_WG64_2_1
```

in `TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co`
(macro tile 64x128, `DepthU` 16, 128 threads, 1638 B group segment). Its argument block is 144 B.

### 2.3 The reproducer

`fixes/hip1/launch_kernels.hip` loads the library's code object itself
(`hipModuleLoad` + `hipModuleGetFunction`), resolves one kernel by name, fills the argument block
the code object's own metadata declares, places each operand so that its **last byte is the last
mapped byte** of a VMM allocation (guard page immediately after, `--pad` adds mapped, zeroed
slack), and launches with a 1-D flat grid. A read past the operand lands on an unmapped page, the
kernel dies with signal 13, and ROCr's log names the kernel.

`--pad` is the measuring stick: a job that faults at pad N but is clean at pad M overruns by more
than N and at most M bytes.

### 2.4 The distance

The trainer's shape, expressed in the kernel's own terms (see 2.6 for why the kernel's M is the
plugin's N):

```
launch_kernels --jobs resources/bf16-overrun-jobs.tsv --kernel <855-char name> \
               --shape 1280 64 308 --dcrc 0x4ec0d5e7 --tier packed --pad P
```

| pad | result |
| --- | --- |
| 0 | FAULT at the first byte of **B**'s guard page (operand 64x308 = 39424 B) |
| 4096 | CLEAN, crc `0x4ec0d5e7` = exact reference |
| 65536 | CLEAN, crc exact |
| 262144 | CLEAN, crc exact |

So the over-read is more than 0 and at most 4096 bytes past B, and the data it reads does not change
D. For this kernel B is addressed as `(n, k, batch)` with `n` contiguous, so `ld_B = 64` elements =
128 B and one extra `DepthU` block of rows would be `16 × 128 = 2048 B` — inside the interval,
though the disassembled kernel reads this operand with 16-byte transposed loads rather than whole
blocks, so 2048 B is an upper-bound arithmetic, not the measured distance (3.1). (The other operand,
A, has `ld_A = 1280` and would overrun by 40960 B if it were the one; it is not, which is consistent
with the fault landing on B's page.)

The faulting address is not evidence of where the access started. The harness places each operand so
that its last byte is the last mapped byte, which makes B's end page-aligned (`0x…a72000`); the
reported address is the first byte of the guard page exactly because of that. The pad table bounds
the distance, not the instruction — the instruction comes from the binary
(`fixes/hip1/source-trace.md` §3).

### 2.5 Shape correlation

The ladder the inventory builds around each kernel's own macro tile, at pad 0, for the fix3 kernel:

```
== summary (tier packed) == clean 4  wrong 0  error 0  fault 6  skipped 0
```

The clean ones include `control-full-tiles` (128x256x64 — exactly two macro tiles by two, and K =
4 × `DepthU`), and the faulting ones include `min` (56x120x24), `edges-minus-4+k-tail` (60x124x72)
and `multi-tile-edges` (136x264x72). Reads past the operand appear exactly where a clamp would be
needed — K not a multiple of `DepthU`, or a free dimension not a multiple of the macro tile — and
not on the shape that fills its tiles exactly.

### 2.6 Why the argument block is trustworthy — the convention, from the sources

Everything above rests on filling the kernel's argument block the way hipBLASLt fills it. That was
settled from the ROCm source rather than by calibration, after an earlier, wrong convention produced
misleading results (see 2.8):

| Question | Answer | Where |
| --- | --- | --- |
| Which of the two free dimensions is the kernel's M | `SizesFree0` = D's dimension 0 = m; the library does not swap A and B. Which layout family serves which transposes is fixed by the library's own problem type: `Ailk_Bjlk` has `IndexAssignmentsA: [0, 3, 2]`, `IndexAssignmentsB: [1, 3, 2]`, `TransposeA: false`, `TransposeB: true` | `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tensile_host.cpp:1796-1884`; `.../Tensile/Logic/asm_full/gfx1200/GridBased/gfx1200_Cijk_Ailk_Bjlk_*.yaml` |
| What `strideA0` / `strideA1` mean | Dimension 0 is the contiguous one, its stride is 1 and it is **not passed**; the arguments carry the strides of dimensions 1.., in **elements** | `projects/hipblaslt/tensilelite/src/ContractionSolution.cpp` `singleCallArgs` (the stride loops); `tensile_host.cpp` problem constructor (`row_stride = 1`, `col_stride = ld`) |
| Stride units | Elements: the kernel shifts to bytes itself (`s_lshl … log2(bpe)`) | `tensilelite/Tensile/KernelWriterAssembly.py` (the D store address chain) |
| D's memory layout | `(i, j, batch)` with i contiguous ⇒ column-major over (i, j): element (i, j) at `i + j*m` | `tensile_host.cpp:1879-1884` |
| Launch grid | One flat id: `calculateGrid` computes `ceil(m/MT0)`, `ceil(n/MT1)`, and `numWorkGroups.x *= y*z` flattens it once `internalArgsSupport.version >= 1`; the kernel's `remapWgSerial` takes wg0 as the fastest index and reconstructs `numWG0/1` from `SizesFree` and the macro tile | `ContractionSolution.cpp` `calculateGrid` / `generateSingleCall`; `KernelWriterAssembly.py` `remapWgSerial` |
| The packed scalars | `Gemm info` = `gemmCount | argType << 30`; `kernel info0` = GSU, GSUWGMRR, GSUC, staggerU/staggerStrideShift/staggerUMapping; `kernel info1` = WGM, WGMXCC, WGMXCCG | `ContractionSolution.cpp` `kernelArgs` |
| `ESMRuntimeSupported` | Not a solution property: the device's `hipDeviceAttributeExpertSchedMode`, appended on gfx1200/gfx1201 only | `ContractionSolution.cpp` (`if(sizeMapping.expertSchedulingMode > 0)`) |

A consequence worth stating plainly: for `Ailk_Bjlk` the kernel's **A operand is the plugin's B
tensor and its B operand is the plugin's A**, and its M is the plugin's N. The trainer's
`64×1280` GEMM is therefore launched as a 1280×64 problem — which is what the observed 20
workgroups (`1280 / 64`) say, and that is why the audit's B operand is the 64×308 tensor.

### 2.7 Independent consistency: the earlier sweep

`fixes/hip1/sweep_gemm_shapes.hip` takes the other route — it calls hipBLASLt with real shapes and
lets the library choose the kernel, then reads the fault out of the crash. Its three rounds
(wide 2800 runs, fine 1936, trainer 192) found 234 FAULTs over **15 distinct kernels**, all of them
in the two `Bjlk` library files. Those numbers are from that run's stdout and were not archived
here; re-run the program to regenerate them. They matter because that route never touches this
program's argument block: it goes through hipBLASLt, so the two routes agree only if the kernel
itself is at fault.

### 2.8 Harness defects that produced false readings

Recorded so that no number from the earlier calibration is reused:

1. **Wrong convention.** The first launcher took `strideX0` to be the stride over M (it is over
   dimension 1), assumed byte strides, used a 2-D grid, and hashed D as row-major while accepting a
   transposed hash as an equally good result. The last of those hid the other three: results looked
   "clean" on shapes that were in fact wrong. All four are fixed; the transposed-crc acceptance is
   gone, so a mismatch is now a mismatch.
2. **A stream race in the harness.** Operand fills ran on the null stream while the launch went to a
   freshly created stream, with no ordering between them. About 3 % of jobs came back with a D of
   all zeros and were reported as WRONG; the same jobs were clean on the next run. Now every device
   operation of one job goes on one stream, the fills are verified to have landed before the launch
   (`BADCOPY` if they did not), and D is primed with a sentinel no exact integer accumulation can
   produce (`NOOP` if the launch left it untouched). Three runs of the 132-job control sample after
   the fix: 132 clean, 0 wrong, 0 noop, 0 badcopy.
3. **Not a defect, but slower than it needed to be:** the operand fills were one 2-byte synchronous
   `hipMemcpy` per element. Building them on the host and uploading once per operand made a control
   sample run go from 82 s to about 1.2 s.

The route-2 sweep numbers in 2.7 are unaffected by any of this.

---

## 3. What the evidence does not establish, and what has since been pinned

Items 1 to 3 were open when this document was written and are now settled from the ROCm source and
the disassembled kernel (`fixes/hip1/source-trace.md`); what stays open is noted inside item 1, and
item 4 (the population) still needs the full scan.

1. **How far, exactly.** Only the interval `(0, 4096]` is measured, and the fault address cannot
   narrow it: the harness places each operand so its last byte is the last mapped byte, which makes
   the reported address the page of the faulting access by construction (2.4). The *instruction* is
   no longer open. Every A/B read in this kernel is `global_load_tr_b128`, a 16-byte transposed
   global load; the tail block's reads are the only ones preceded by the generator's address clamp
   (`v_min_i32`, `"truncated load: clamp GRO to legal range"`, `KernelWriterAssembly.py:10665`).
   That clamp caps a *start* offset, derived from the K remainder and the macro tile and never from
   the operand's size, so an access starting at the cap can reach up to 14 bytes past it — and
   whether the cap itself is also placed past the operand when the free dimension is partial is what
   a finer `--pad` ladder (16, 64, 256, 1024) or the generator's annotated assembly would settle.
   Source trace, fix sites and the two candidate patches: `fixes/hip1/source-trace.md`.
2. **Whether the over-read data is masked out or merely multiplied by zeros.** Neither, at the
   address level: the kernel issues the read and then discards the *values* — the tail loop compares
   against the K bound (`v_cmp_ge_i32 s66, v149, s12`) and replaces the loaded registers with zero
   (`v_cndmask_b32_e64 …, 0, s66`), and the generator pre-zeroes the destination before the load
   (`set to zero to avoid unexpected value`, `KernelWriterAssembly.py:10667-10668` for bf16/fp16
   WMMA, `:10662` for the int8/`isM` cases). Nothing downstream
   distinguishes "masked" from "multiplied by zero"; the read happens either way, which is all the
   claim needs.
3. **Whether the library considers this legal.** The generator's own assumption is the opposite of a
   licence to read past the operands: it keeps the wide tail load only because the ISA claims to
   drop the part of an access that sticks out of its bounds (`asmCaps["HasPartialOOB"]`,
   `KernelWriterAssembly.py:10214-10215`, defined as "every ISA except gfx1250" in
   `rocisa/rocisa/include/hardware_caps.hpp:96` — so true on gfx1201), and it pre-zeroes the
   destination for exactly that reason. That assumption is carried by the descriptor: a MUBUF
   `buffer_load` has a record count, while the `global_load_tr_b128` form used here has a 2-SGPR
   address operand and no bound field at all, so nothing enforces it on this path. The earlier note
   in this slot about `needsPadFreeDim` was wrong and is withdrawn: that flag pads MX scale tensors
   and is enabled on gfx950 only (`tensile_host.cpp:1761-1779`, reached from the `setMXScaleA/B`
   paths), so it is not a general "kernels may read past their operands" precedent.
4. **The population.** How many of the 1830 launchable kernels over-read, and how many of those
   over-read far enough to matter, needs the full scan (M4). The plan's own cost estimate was wrong:
   the dominant cost is not the launch but the ~7 s device reset ROCr performs after each fault, so a
   pad-0 pass over all 18300 jobs at the fix3 fault rate would be hours.

---

## 4. Reproduction

```bash
cd /home/acite/Deeppin/AxlTrainer
export LIBDIR=/home/acite/miniconda3/envs/axl_rocm_7_14/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201

# population, ladder and job list (torch-free; system python is fine)
/usr/bin/python3 fixes/hip1/kernel_inventory.py --libdir "$LIBDIR" --out-dir fixes/hip1/resources
/usr/bin/python3 fixes/hip1/kernel_inventory.py --check --libdir "$LIBDIR"   # fix3 kernel vs hip1.md §12.9

# driver
source ~/miniconda3/etc/profile.d/conda.sh && conda activate axl_rocm_7_14
hipcc --offload-arch=gfx1201 -O2 -Wall -std=c++17 fixes/hip1/launch_kernels.hip -o /tmp/launch_kernels

K=$(/usr/bin/python3 -c "import sys;sys.path.insert(0,'fixes/hip1');import kernel_inventory as ki;print(ki.fix3_kernel_name())")

# the kernel's exact-tile control: clean, crc 0xe4a70c8f
/tmp/launch_kernels --jobs fixes/hip1/resources/bf16-overrun-jobs.tsv --kernel "$K" --pad 0 --libdir "$LIBDIR"

# the trainer's shape in the kernel's own terms: faults at pad 0, clean and exact from pad 4096
/tmp/launch_kernels --jobs fixes/hip1/resources/bf16-overrun-jobs.tsv --kernel "$K" \
                    --shape 1280 64 308 --dcrc 0x4ec0d5e7 --pad 0 --libdir "$LIBDIR"
```

---

## 5. Files

| Path | What it is |
| --- | --- |
| `fixes/hip1/kernel_inventory.py` | Decodes the 4 bf16 `.co` + `.dat.zlib` pairs, writes the inventory and the job list, builds each kernel's shape ladder, computes the reference D crc, `--check` against hip1.md §12.9 |
| `fixes/hip1/launch_kernels.hip` | Loads one kernel from the library and launches it with guard-paged operands; driver mode restarts children after a fault and classifies every job CLEAN / WRONG / NOOP / BADCOPY / FAULT |
| `fixes/hip1/sweep_gemm_shapes.hip` | The other route: real hipBLASLt calls over sweeps of shapes, one child per run |
| `fixes/hip1/launch_fault_kernel.hip`, `fixes/hip1/launch_fault_via_rocblas.hip` | The minimal single-shape reproducers (standalone kernel / hipBLASLt) |
| `fixes/hip1/resources/bf16-kernel-inventory.json` | 2105 kernels with their metadata, layout, size mapping, packed scalars and ladders |
| `fixes/hip1/resources/bf16-overrun-jobs.tsv` | 18300 jobs: 1830 launchable kernels × 10 shapes |
| `fixes/hip1/hip1.md` | The investigation this conclusion rests on (§12.9 the standalone launcher, §12.10 the rocBLAS route and the grid, §12.12 the withdrawn fp16 sibling) |
| `fixes/hip1/source-trace.md` | Where the kernel comes from (logic file, solution index, runtime record), which generator function emits the guarded read, and where a code-level fix would go |
| `conclusions/bf16-overrun-mitigations.md` | What keeps a run alive anyway: the working measures, each one's measured cost, and what they leave unaddressed |
| `conclusions/gfx1201-fault-response-wedge.md` | What the GPU and driver do with this over-read when it works, and the state observed on 2026-09-16 in which the read stopped being reported at all (a hang instead of a kill) |
| `/home/acite/LLM/axltrainer/outputs/lllj_20260916_062841` | The crashed run |

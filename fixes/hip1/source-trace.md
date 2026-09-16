# Source trace: where the gfx1201 bf16 over-reading kernel comes from

`conclusions/bf16-kernel-overrun.md` establishes *what* happens (a bf16 Tensile kernel reads
past its operand, and the crash needs the operand to end near an unmapped page). This document
establishes *where it comes from*: which file in the ROCm source generates the kernel, which
generator function emits the read that over-runs, and where a code-level fix would go.

Everything below is static — ROCm sources plus the installed binary. No GPU run was made for this
document. **No source fix has been applied**; §5 specifies it.

Sources used:

| What | Where |
| --- | --- |
| ROCm checkout | `/home/acite/Deeppin/rocm-libraries`, tag `therock-7.14.1`, commit `cd9574023093742434e8c992d13b89ab9a6c1cf8` |
| Generator (authoritative for hipBLASLt) | `projects/hipblaslt/tensilelite/Tensile/` (19625 lines of `KernelWriterAssembly.py`, not the `shared/tensile` mirror, which differs) |
| Installed library audited | `<conda env axl_rocm_7_14>/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201/` |
| ROCm LLVM for the disassembly | `/opt/rocm/core/lib/llvm/bin/llvm-objdump` (AMD LLVM 23.0.0git) |

---

## 1. The kernel: file, solution, runtime record, binary

The failing kernel is the 855-character
`Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…_SVW8_…_WS32_WG64_2_1`
(the one `fixes/fix3`, `fixes/hip1` and the inventory all point at). Its four identities line up
exactly:

| Layer | Value | How it was checked |
| --- | --- | --- |
| Source logic file | `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1201/GridBased/gfx1201_Cijk_Ailk_Bjlk_BBS_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs.yaml`, **entry 210 of 473** | `SolutionIndex: 210` in the entry; its `KernelNameMin`/`SolutionNameMin` expand to this exact name; `DepthU 16`, `ASEM 1`, `DirectToVgprA/B True`, `MatrixInstruction [16,16,16,1]`, `MIWaveTile [1,8]`, `WorkGroup [64,2,1]`, `WorkGroupMapping 8`, `ScheduleIterAlg 3`, `StaggerU 32`, `1LDSBuffer 0`, `EdgeType ShiftPtr`, `StoreVectorWidth 8`, `SourceSwap False` |
| Runtime record | `<library>.dat.zlib`, entry **index 107020** | `libraryLogicIndex` = 107020; `TensileLiteLibrary_lazy_gfx1201_Mapping.dat.zlib[106810] = TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201`, covering `106810..107282` = 473 solutions, so 107020 is offset **210** |
| Code object | `<library>.co`, one text symbol at `0x56E400`; body `0x56E400..0x582FDC` (its `s_endpgm`), next kernel at `0x583000` | exactly one text symbol carries the full 855-character name |
| Sibling | entry **355** (index 107165) differs in six fields only: `SourceSwap True`, `StoreVectorWidth 1`, and the three name strings / `SolutionIndex` | those two entries are why 473 solutions carry only 427 distinct kernel names, and the library does hold both bodies — the sibling at `0x5F3400` is the `…_SS1_…_SVW1_…` name (79104 B) |

Two traps this file sets, both worth remembering:

- **The file name does not name the library.** This file is `…_SHB_HA_S_SAB_SCD_SAV_UserArgs.yaml`
  but the library built from it is `…_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_…`
  (no `SAB`, no `SCD`). Selecting "sibling" yaml files by name would have missed it; the mapping
  above is by index and parameters instead.
- **A 120-character prefix of the kernel name is not unique** (`SVW8` vs `SVW1` differ at
  character ~700). Match on the full name when resolving a kernel.

## 2. The generator chain that turns that file into the installed `.co`

| Stage | Code | What it decides |
| --- | --- | --- |
| Pick the logic file | `Tensile/CustomYamlLoader.py:147-148` (`GFX_ARCH_IDX = 2`, arch read *inside* the yaml) + `TensileCreateLibrary/Run.py:985-990` (`archMatch`, `--logic-filter`) | the `asm_full/gfx1201/GridBased/*.yaml` files are consumed for gfx1201; directory names are organisational only |
| Merge, assign indices | `TensileCreateLibrary/Run.py:838-875` (master library re-indexed per architecture, then per lazy library) | the runtime `index` is assigned by the generator, per library, continuing from the start index the mapping file records; for a single-file library the position in the file equals the yaml's `SolutionIndex`, which is why 107020 = 106810 + 210 |
| Compose names | `SolutionStructs/Naming.py:101` (abbreviation = the capitals of a parameter), `:144` `_getName`, `:230-238` `getKernelNameMin`/`getSolutionNameMin`/`getSolutionNameFull`; parameter sets in `Common/RequiredParameters.py:29-34`, `Common/ValidParameters.py` | the 855-character name is a function of the parameters — never type it by hand |
| Problem-type name | `SolutionStructs/Problem.py:1256-1362` | `HA` = `ActivationType: hipblaslt_all` + `S` (f32 compute) → the `HA_S` in the name |
| Library name | `SolutionLibrary.py:376-503`, `Contractions.py:346-370` | the `<library>.co` / `<library>.dat.zlib` basename |
| Emit assembly | `KernelWriter.py` + `KernelWriterAssembly.py` (+ `rocisa`, a nanobind C++ extension) | every instruction, including the one in §3 |
| Assemble | `Toolchain/Component.py:152-160` (`amdclang++ -x assembler --target=amdgcn-amd-amdhsa -mcode-object-version=4 -c`), `:172-174` (`-Xclangas -target-feature +real-true16` on gfx12) | the AMDGPU object; `-g` only when built with debug |
| Bundle | `Toolchain/Component.py:300-302` (`clang-offload-bundler --compress --bundle-align=4096`) | the `CCOB` container that is the `.co` |

The practical consequence for a rebuild: the `.co` and its `.dat.zlib` must be produced together
(the runtime picks the kernel by the name recorded in the `.dat`, and the argument block it fills is
described in the same record), and the installed library takes its directory from
`HIPBLASLT_TENSILE_LIBPATH` when set.

## 3. The read that over-runs

### 3.1 What the binary contains

The whole kernel contains 120 loads; they split cleanly by descriptor:

| Form | Count | Descriptor | Role |
| --- | --- | --- | --- |
| `global_load_tr_b128` (16 B/thread, transposed feed for `v_wmma`) | 36 | `s[48:49]` = A, `s[52:53]` = B (2 SGPRs = a 64-bit address) | **the A/B operand reads** |
| `buffer_load_b128` / `buffer_load_b32` / `buffer_load_d16_b16` | 16 / 3 / 65 | `s[20:23]`, `s[32:35]`, `s[40:43]` (4 SGPRs = MUBUF v#) | bias and activation parameters (`label_Load_Biasf32_0`, `label_Load_Biasbf16_0`, `label_ActivationSetPCAddrEnd*`) |

So every read of the A/B operands is a **16-byte transposed global load**, and the kernel's operand
traffic has no other form.

The 36 A/B loads sit at four sites (`label_staggerInputEnd`, `label_skipEnableESM`,
`label_unrolledLoop_lc1`, `label_MultiplyDone_1XRC0SXCY4YIESXR`, 9 loads each); only the last site —
the tail block, which ends at `label_TailLoopBeginL` — carries a guard. Immediately before each of
its loads:

```
0x56FFF0  v_min_i32_e32 v64, v148, v64                     // the guard
0x56FFF4  global_load_tr_b128 v[76:79], v64, s[48:49]      // A, 16 bytes
0x570060  v_min_i32_e32 v65, v148, v65
0x570064  global_load_tr_b128 v[84:87], v65, s[52:53]      // B  (x8, up to 0x5700D4)
```

and the block ends at `label_TailLoopBeginL` (0x570100). The three other sites issue the same
16-byte load with **no** `v_min_i32` in front of it. `label_SkipTailLoopL` is branched to at
0x56FED0, so when `K % DepthU == 0` neither the guarded block nor the tail loop runs at all — which
is why the clean control shape in the ladder is exactly the K = 4·`DepthU` one.

### 3.2 Where the guard comes from

`KernelWriterAssembly.globalReadGuardK` (`:10207`) is the tail-block read generator. For
`enableGLTrA/B` kernels (`isTr`, `:10212`) it computes a "max read address offset"
(`:10223-10271`) from: the wave id (`Serial / WavefrontSize`, for B divided by
`MIWaveGroup[0]`), `MatrixInstK / glvw`, the right-half bit (`VBfeU32(Serial, bpeGR+1, 1)`),
`glvw`, `lsc * (NumReadsIterCoalesced - 1)`, and — the K term — `SizesSum % DepthU`:

```
:10262  VMovB32(dst=vgpr(tmp), src=sgpr("SizesSum+%u" …))         # K
:10264  vectorStaticRemainder(tmp, tmp2, tmp, DepthU, …)          # tmp2 = K % DepthU
:10267  VSubU32(dst=vgpr(tmp2), src0=vgpr(tmp2), src1=1,
                comment="GLTr%s: unroll idx - 1")
        MacroInstruction("GLOBAL_OFFSET_%s")                      # expansion: KernelWriterAssembly.py:1798
:10270  vectorMultiplyBpe(maxGroVgpr, maxGroVgpr, tP["bpeGR"])     # → bytes
```

The disassembly matches that sequence instruction for instruction (the `(s27 & 15) - 1` times a
stride, `+8` for the right-half bit, `×2` for bytes, then the `v_min_i32`).

The guard itself is `globalReadGuardKBody` (`:10323`), which for `isTr` emits (`:10665`):

```python
module.add(VMinI32(dst=vgpr(offsetVgpr), src0=vgpr(maxGroVgpr), src1=vgpr(offsetVgpr),
                   comment="truncated load: clamp GRO to legal range"))
```

and immediately before it, for the bf16/WMMA case, zeroes the destination
(`:10662`, `:10667-10668`, `VMovB32(dst=loadVgpr, src=0, comment="set to zero to avoid unexpected
value")`). The load itself is emitted at `:10676-10699` via `chooseGlobalRead` (`:16123`).

Three properties of that guard matter here:

1. **It clamps the address, not the access window.** `VMinI32(offsetVgpr, maxGroVgpr, offsetVgpr)`
   bounds the byte offset at which the 16-byte access *starts*; the access width does not enter the
   computation anywhere, so a cap that is right about the last element the tile needs still lets the
   access end up to 14 bytes past it.
2. **The cap is derived from the problem's K remainder and the tile geometry, not from the
   operand's size.** Nothing in the `isTr` block mentions the number of bytes allocated, and its
   free-dimension terms come from the macro tile (`glvw`, `lsc`, wave geometry), so a partial free
   dimension makes the cap a tile-shaped quantity rather than an operand-shaped one.
3. **The out-of-range *data* is discarded after the fact, not prevented.** The tail loop
   (`label_TailLoopBeginL` 0x570100 … `label_TailLoopEndL` 0x57080C) zeroes register values under a
   comparison against the K bound (`v_cmp_ge_i32 s66, v149, s12` → `v_cndmask_b32_e64 …, 0, s66`),
   the `TailLoop_SkipZeroOutMask` label of `KernelWriterAssembly.mfmaIter` (`:9043`, `:9263`). The
   read has already happened by then. This also answers the open question in
   `conclusions/bf16-kernel-overrun.md` §3.2: out-of-range values are dropped at the *value* level,
   not at the address level.

### 3.3 What is measured

From the archived run of the trainer's own shape (kernel `M=1280 N=64 K=308`, `--tier packed`):
`pad 0` faults, `pad 4096 / 65536 / 262144` are CLEAN with the exact reference crc. In the
`pad 0` placement B ran `0x…a68600..0x…a72000` (39424 B) and the kernel faulted at
`0x…a72000` — the first byte after B. That byte is page-aligned **because the harness places each
operand so its last byte is the last mapped byte**, so the reported address identifies the *page*
of the faulting access, not the offset it started from. The over-read is therefore bounded by
`(0, 4096]` bytes and not localised further, exactly as the conclusion document says.

## 4. Why a 16-byte load survives at all: `HasPartialOOB`

`globalReadGuardK` keeps the wide tail load when the ISA claims to tolerate an access that sticks
partly out of its bounds (`:10214-10215`):

```python
# For example, it replaces buffer_load_b128 with 4 buffer_load_b32 if HasPartialOOB is false.
tailGRVW = kernel["GlobalReadVectorWidth%s"%tc] if self.states.asmCaps["HasPartialOOB"] else 1
```

`HasPartialOOB` is not a hardware query: `projects/hipblaslt/tensilelite/rocisa/rocisa/include/
hardware_caps.hpp:96` defines it as `checkNotInList(isaVersion, {{12,5,0}})` — true for **every ISA
except gfx1250**, so true on gfx1201. The rest of the machine agrees with that assumption: the
destination is pre-zeroed before the load (`set to zero to avoid unexpected value`; `:10662` for the
int8/`isM` cases, `:10667-10668` for half/bf16 WMMA, which is the one that applies here), so the
part of an access the hardware drops leaves a deterministic 0 behind.

Where the assumption can be discharged differs by load form:

| Form | Descriptor | Can the hardware drop a partially out-of-bounds access? |
| --- | --- | --- |
| `buffer_load_*` (MUBUF) | 4 SGPRs: base, stride/record shape, **record count** | yes — the record count is the bound the capability refers to |
| `global_load_tr_b128` (this kernel's A/B path) | 2 SGPRs: a 64-bit address (`chooseGlobalRead` passes `2 if isTr else 4` at `:10676`, `:10691`) | no record count exists in the operand form (`hardware_caps.hpp:346-349` probes it as `global_load_tr_b128 v[0:1], v0, s[0:1], offset:0`) |

So on this path the capability the generator relies on has nothing to enforce it: the only bound is
the software clamp of §3.2, which caps a start offset without accounting for the 16-byte access, and
whose cap is a tile-shaped quantity. The vendor's own remedy for an ISA *without* the capability
cannot be borrowed either: the TR path has exactly two encodings,
`GlobalLoadTR8B64` (8 B, `bpl==8`) and `GlobalLoadTR16B128` (16 B, `bpl==16`), selected in
`chooseGlobalRead` (`:16239-16249`). Forcing `tailGRVW = 1` there would ask for a 2-byte load and
emit nothing.

## 5. What a fix can and cannot do (measured)

### 5.1 No address-side change is both safe and data-preserving

Three generator-side candidates were built and measured against the harness — each one patched,
rebuilt in ~15 s, then run on the trainer's shape and on the kernel's ten-shape ladder. "Wrong" means
D no longer matches the CPU reference, i.e. the patch moved data the tile uses:

| Candidate | Edit | trainer shape (kernel M=1280 N=64 K=308) | ten-shape ladder |
| --- | --- | --- | --- |
| A | cap := cap − one access width (14 B) | CLEAN, crc exact | **7/10 wrong** |
| B | reads above the cap read the block start instead | still faults | 10/10 clean, crc exact |
| C | cap := min(cap, bytes left in the operand − one access) | CLEAN, crc exact | 7/10 wrong, crcs identical to A |

A and C move the same reads and produce the same wrong crcs; B moves a strictly smaller set and keeps
every crc exact. Together those two facts close the space: the reads that cross the operand's end are
*partly inside it*, and the part inside is data the tail block needs. The access cannot be narrowed
either — the transposed global load has exactly two encodings, 8 and 16 bytes
(`chooseGlobalRead`, `:16239-16249`) — and it cannot be bounded, because its operand is a 2-SGPR
address with no record count (§4).

So the GLTr guard cannot be repaired in the generator. Either the operand has slack, or the solution
must not be selected for that problem.

### 5.2 The selection-level rule works, and is still not enough

The vendor's own idiom (`AssertSummationElementMultiple` raised to `DepthU`, so the runtime's
`BoundSizeMultiple` predicate refuses the solution for problems that need a tail loop) was applied to
the layouts its two existing rules miss — `SolutionStructs/Solution.py`, 10 lines, §5.3 — and measured
through the real host API with `ltcheck.cpp`: a bf16 GEMM with bias and activation on the plugin's
64×1280×308, four epilogue variants, three library configurations.

| Configuration | Selected kernel (from the fault log) | Result |
| --- | --- | --- |
| installed library | the family's DTV kernels | **faults, every variant** |
| rebuilt `.co`, no predicate | `MT32x32x64 … ASEM64 … DTVA0_DTVB1` — the *renamed* kernel | faults |
| rebuilt `.co` + `BoundSizeMultiple` | `MT32x32x64 … ASEM1 … DTVA0_DTVB0 … GRVWA1_GRVWB1` — a different one | faults |

All twelve runs (four epilogue variants × three configurations) faulted. Three things follow. The rule
is honoured: the host stops choosing the DTV solution once the predicate is there, deterministically
and visibly in the selected kernel's name. The crash survives it: the family holds at least one more
over-reading kernel, and that one reads with 1-element buffer loads (`GRVWA1_GRVWB1`), not with
16-byte transposed loads — the GLTr guard of §3 is not the only mechanism behind this family's page
faults. And the rule does not change the code at all: the renamed kernel still carries its tail block
(36 transposed loads, 5 tail labels, same as the installed one), so `ASEM → NoTailLoop` does not
remove the guarded read.

**Retraction.** An earlier version of this document read a ladder run as evidence that the rule does
remove the over-read ("10/10 no faults"). That build also carried the candidate-C generator
experiment, and the zero faults came from *that*, not from the rule; with the generator restored to
upstream the renamed kernel faults on the same six shapes. The claim is withdrawn.

Nothing was installed into the environment on the strength of this: the rule removes one class of
faults and leaves the crash in place, so enabling it would cost selection choices without removing
the failure. The wiring exists and is off (`/home/acite/axl-hipblaslt/enable.sh`, `--persist` writes
the env's `activate.d` hook; `disable.sh` undoes it).

What the cost would be, as far as it can be measured without a timing run: the rule refuses 157 of
this family's 473 solutions (its DTV ones) whenever the summation dim is not a multiple of their own
`DepthU`, and the file's own tuning table says none of the 7768 problems it covers is affected —
6345 of them are won by a DTV solution, and every one of those has `K % DepthU == 0`. The K-tail DTV
case the rule excludes was never in the vendor's benchmark set at all, which is also why the shape
that faults here was never tuned away.

### 5.3 Staging and testing a patched library (kept, reusable)

The rebuild is cheap and the swap is reversible, so the loop is worth keeping even though this
particular fix is not deployable:

1. **Rebuild one logic file** — `python3 -m Tensile.TensileCreateLibrary --architecture=gfx1201
   --logic-filter='gfx1201/GridBased/<file>' <rocblaslt library dir> <out dir> HIP`, with
   `PYTHONPATH=<src>/Tensile:<src>/rocisa`; the prebuilt `_rocisa.abi3.so` from
   `_rocm_sdk_libraries/share/hipblaslt/tensilelite/rocisa/` can be symlinked into the source package
   instead of building the nanobind extension. 15 s for the family's 427 kernels, and the rebuilt
   code object carries exactly the installed kernel names when nothing is changed.
2. **Stage a tree** — copy the installed `gfx1201/` (≈170 MB) and overwrite the family's
   `.co` and `.dat.zlib`. Pair the two files by the kernel name with the `_ASEM<digits>_` token
   normalised (a rule can change `ASEM1` into `ASEM16/32/64/128` depending on the solution's own
   `DepthU`), carry the new name and the new `BoundSizeMultiple` value into the installed `.dat`, and
   leave every `index` alone: the installed master library and its mapping file are keyed on
   106810..107282 for this family.
3. **Point the loader at it** — `HIPBLASLT_TENSILE_LIBPATH` takes the directory that *holds* the
   files, not its parent (`hipModuleLoad failed: /nonexistent/Kernels.so-000-gfx1201-xnack-.hsaco`
   from a probe with a bogus path shows the file name being concatenated onto the variable). Pointing
   it one level too high makes hipBLASLt fail with `HIPBLAS_STATUS_INVALID_VALUE`.
4. **Re-verify** — the harness (`launch_kernels.hip`) forces a kernel by name and manufactures the
   page-boundary placement, so it checks the *code*; `ltcheck.cpp` goes through the real hipBLASLt
   entry points, so it checks *selection* and *results* together.

## 6. What is still open

1. **The other over-reader in this family.** The host-level A/B (§5.2) ends on a second kernel —
   `MT32x32x64_MI16x16x1_…_DTVA0_DTVB0_…_GRVWA1_GRVWB1_…_LDSB1_…_EPS1` — which reads its operands
   with 1-element buffer loads, not with transposed 16-byte loads, and faults all the same. Which
   address it over-reads and why its own guard does not stop it is not established: it is a second
   mechanism, and the GLTr guard of §3 does not cover it.
2. **The exact byte distance in the GLTr case, and which of the two mechanisms produces it.** The
   measurements bound the over-read to `(0, 4096]` and the `pad 0` fault address is page-aligned by
   construction (§3.3), so they cannot separate (M1) "the cap is computed from the macro tile, and a
   partial free dimension puts it past the operand's end" from (M2) "the cap is at the last needed
   element and the 16-byte access extends 14 bytes past it". The disassembly shows the clamp and its
   inputs but not the numeric value of `maxGroVgpr`, which needs the `GLOBAL_OFFSET_x` macro
   expansion (`:1798`) and the register map for the B operand. §5.1 now makes this a curiosity
   rather than a blocker: whatever the split, no address-side change can be made safe.
3. **The ladder's per-shape partition** — settled. The ten-shape ladder at `pad 0` faults on
   `k-tail`, `m-edge+k-tail`, `n-edge`, `m+n-edges+k-tail`, `min`, `edges-minus-4+k-tail`, and is
   clean on `control-full-tiles`, `n-edge+k-tail`, `m-edge`, `multi-tile-edges` (`clean 4  fault 6`).
   Every faulting shape has `K % DepthU != 0`; whether the free dimension fills its tile is not what
   decides it.
4. **Population.** Unchanged from the conclusion document: how many of the 1830 launchable kernels
   over-read this way still needs the full scan.

## 7. Reproducing the trace

```bash
# 1. the source entry and its runtime record
python3 - <<'PY'
import yaml, zlib, sys
sys.path.insert(0, "fixes/hip1"); import kernel_inventory as ki
d = yaml.safe_load(open("<rocm-libraries>/projects/hipblaslt/library/src/amd_detail/rocblaslt/"
                        "src/Tensile/Logic/asm_full/gfx1201/GridBased/"
                        "gfx1201_Cijk_Ailk_Bjlk_BBS_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs.yaml"))
print(len(d[5]), d[5][210]["SolutionIndex"])
m = ki.msgpack_loads(zlib.decompress(open("<libdir>/"
      "TensileLiteLibrary_lazy_gfx1201_Mapping.dat.zlib", "rb").read()))
print(m[106810])                      # library that owns index 107020
PY

# 2. the device object, then the kernel's own text
python3 /tmp/axl-hip1/extract_one.py \
  TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co \
  /tmp/axl-hip1/ailk_bjlk_gfx1201.o
/opt/rocm/core/lib/llvm/bin/llvm-objdump -d --triple=amdgcn --mcpu=gfx1201 \
  --start-address=0x56E400 --stop-address=0x583000 /tmp/axl-hip1/ailk_bjlk_gfx1201.o > /tmp/axl-hip1/tail_kernel.asm
grep -n "v_min_i32\|global_load_tr_b128\|label_TailLoop\|label_SkipTailLoop" /tmp/axl-hip1/tail_kernel.asm
```

`extract_one.py` unwraps the `CCOB` container and the clang offload bundle; the device object keeps
the generator's own labels (`label_TailLoopBeginL`, `label_SkipTailLoopL`,
`label_MultiplyDone_…`, `label_RegularSrdInitializationD`, …), which is what makes the
source↔binary mapping above checkable rather than asserted.

## 8. Files

| File | Role |
| --- | --- |
| `fixes/hip1/source-trace.md` | this document |
| `conclusions/bf16-kernel-overrun.md` | the measured claim and its evidence chain |
| `fixes/hip1/kernel_inventory.py` | kernel/solution inventory, job generation, `--check` |
| `fixes/hip1/launch_kernels.hip` | the guard-page launcher that produced the pad table |
| `fixes/hip1/resources/bf16-kernel-inventory.json`, `bf16-overrun-jobs.tsv` | generated inventory and job list |
| `/home/acite/axl-tensile-build/ltcheck.cpp` | host-level hipBLASLt probe (§5.2): bf16 GEMM with bias/activation/alpha-vector, reference check, timing, `--only <variant>`; built against the repo's own headers plus a two-file shim for the generated `hipblaslt-export.h`/`hipblaslt-version.h` |
| `/home/acite/axl-tensile-build/{build.sh,stage_deploy.py,stage_nopred.py,check_gemm.py,slack_sweep.py}` | the rebuild (15 s), the ASEM rename + predicate staging, the no-predicate control tree, the torch numeric check, and the slack sweep |
| `/home/acite/axl-hipblaslt/` | the staged patched library tree plus `enable.sh` / `disable.sh` for `HIPBLASLT_TENSILE_LIBPATH` (off by default) |

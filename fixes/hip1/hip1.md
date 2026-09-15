# `fixes/hip1/hip1.md` — the hand-triggered page fault and the trainer's page fault are the same kernel event

> Written 2026-09-16, on the machine the other packs were measured on (RX 9070 XT, gfx1201), inside the
> `axl_rocm_7_14` environment (torch 2.12.0+rocm7.14.1, HIP 7.14.60850). Every number below is from a
> log captured in that session; the ring-buffer table in §3 and §7 is a *single* kernel log, so the
> trainer's faults and the hand-made one are compared line by line, not across sessions.

This closes a loop `fixes/fix2` and `fixes/fix3` left implicit: the abort those packs chase is an
ordinary GPU page fault — a read from a warp that lands on a virtual address with no GPU page-table
entry — and a small HIP program can produce that same event on demand, with the same kernel message,
the same status bit-field, the same client, and the same process-side death.

## 1. Verdict

**Same class, different cause.** The kernel-side record and the process-side abort are indistinguishable
from the trainer's, apart from the process name and one status bit. What differs is only *why* the
address is unmapped:

| | trainer (`fixes/fix2`, `fixes/fix3`) | `tools/hip/hip_vmm_tail_read.hip` |
| --- | --- | --- |
| what touched the address | a precompiled Tensile GEMM (`Cijk_Ailk_Bjlk_…ISA1201…`) reading past the end of its buffer, in the backward pass | one thread issuing one load at an address that was reserved and deliberately never mapped |
| who chose the address | the allocator (a guard page / the end of a cached segment) | the probe (`hipMemAddressReserve` + a 2 GiB offset) |
| kernel event | `[gfxhub] page fault … GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031`, client `TCP (0x8)`, `RW: 0x0` | identical, with `0x00801030` |
| process death | ROCr `Memory Fault Error …` + `Memory access fault by GPU node-1 … Reason: Page not present or supervisor privilege.`, then `SIGABRT` | identical text, `SIGABRT` too once SIGPIPE is ignored (§6) |

So: the trainer's abort needs no exotic explanation at the driver level. It is the generic "warp read an
unmapped VA" path, which this probe reproduces at will. §2–§6 are the evidence; §5 is the one difference
that is real (how many reads hit the page before the queue is torn down); §8 adds observations that are
not conclusions, and §10 says what this does **not** settle.

## 2. The program

`tools/hip/hip_vmm_tail_read.hip` (see `HIP.md` §5 for the VMM API it uses) does five things:

1. `hipMemAddressReserve` 4 GiB of device VA — a `---p` anonymous VMA appears in `/proc/self/maps`, and
   VRAM free does not move (15.84 GiB before and after).
2. `hipMemCreate` 64 MiB — the only physical allocation; VRAM free drops 15.84 → 15.78 GiB.
3. `hipMemMap` that allocation into the first 64 MiB + `hipMemSetAccess`; a kernel writes, the host reads
   back, so the mapped part is proven live.
4. Print the PROT of the tail from `/proc/self/maps` (authoritative: `---p`, i.e. `PROT_NONE`) and from
   `hipMemGetAccess` (`hipErrorInvalidValue`).
5. Read the tail (`base + 2 GiB`) from a one-thread kernel on purpose.

```bash
conda activate axl_rocm_7_14
hipcc -O3 tools/hip/hip_vmm_tail_read.hip -o /tmp/hip_vmm_tail_read
before=$(sudo dmesg | wc -l); /tmp/hip_vmm_tail_read > /tmp/tail.log 2>&1; echo "exit=$?"
sudo dmesg | tail -n +$((before+1))
```

## 3. Kernel records, side by side

Trainer: this boot's ring buffer at uptime 405 s — the same event `fixes/fix3/resources/post-reboot-dmesg.txt`
records as `Sep 15 23:34:38`, pid 7407, address `0x00007f9278800000`, status `0x00801031`, `ring 24`,
`vmid 8`, client `TCP (0x8)`: 

```
amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:101)
amdgpu 0000:03:00.0:  Process python pid 7407 thread python pid 7407
amdgpu 0000:03:00.0:   in page starting at address 0x00007f9278800000 from client 10
amdgpu 0000:03:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
amdgpu 0000:03:00.0:          Faulty UTCL2 client ID: TCP (0x8)
amdgpu 0000:03:00.0:          MORE_FAULTS: 0x1
amdgpu 0000:03:00.0:          WALKER_ERROR: 0x0
amdgpu 0000:03:00.0:          PERMISSION_FAULTS: 0x3
amdgpu 0000:03:00.0:          MAPPING_ERROR: 0x0
amdgpu 0000:03:00.0:          RW: 0x0
```

Probe, `sudo dmesg` delta of the run in §2:

```
[11343.306306] amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:1237)
[11343.306311] amdgpu 0000:03:00.0:  Process hip_vmm_tail_re pid 95604 thread hip_vmm_tail_re pid 95604
[11343.306313] amdgpu 0000:03:00.0:   in page starting at address 0x00007f7d37c00000 from client 10
[11343.306315] amdgpu 0000:03:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801030
[11343.306317] amdgpu 0000:03:00.0:  Faulty UTCL2 client ID: TCP (0x8)
[11343.306319] amdgpu 0000:03:00.0:  MORE_FAULTS: 0x0
[11343.306320] amdgpu 0000:03:00.0:  WALKER_ERROR: 0x0
[11343.306321] amdgpu 0000:03:00.0:  PERMISSION_FAULTS: 0x3
[11343.306322] amdgpu 0000:03:00.0:  MAPPING_ERROR: 0x0
[11343.306323] amdgpu 0000:03:00.0:  RW: 0x0
```

The process-side text (ours, from `/tmp/tail.log`; the trainer's is
`fixes/fix3/resources/live-run-fault.txt`):

```
:0:rocdevice.cpp            :3678: 11343484349 us:  Memory Fault Error [host: acitehost, GPU index: 0,
    faulting addr: 0x7f7d37c00000, kernel: (anonymous namespace)::read_unmapped(int const*, int*)]
Memory access fault by GPU node-1 (Agent handle: 0x55bc2dc19da0) on address 0x7f7d37c00000.
    Reason: Page not present or supervisor privilege.
```

## 4. Field by field

| field | trainer (fix3, this boot) | our probe |
| --- | --- | --- |
| event | `[gfxhub] page fault` | same |
| `src_id` / `ring` | `0` / `24` | `0` / `24` |
| `vmid` | `8` | `8` |
| process | `python pid 7407` | `hip_vmm_tail_re pid 95604` |
| faulting address | `0x00007f9278800000` | `0x00007f7d37c00000` — both **2 MiB aligned** |
| status | `0x00801031` | `0x00801030` |
| UTCL2 client | `TCP (0x8)` | `TCP (0x8)` |
| `MORE_FAULTS` | `0x1` | `0x0` |
| `WALKER_ERROR` | `0x0` | `0x0` |
| `PERMISSION_FAULTS` | `0x3` | `0x3` |
| `MAPPING_ERROR` | `0x0` | `0x0` |
| `RW` | `0x0` (a read) | `0x0` (a read) |
| records printed for that address | 4 | 1 |
| userspace | `rocdevice.cpp:3678`, `Memory access fault by GPU node-1 … Page not present or supervisor privilege` | identical, same source line |

`PERMISSION_FAULTS: 0x3` and `RW: 0x0` are constant in **every** sighting across the packs — `fixes/fix1.txt`
quotes them for its `ring:24 vmid:8 pasid:144` event, every fix2 and fix3 record carries them, and so does
this probe: a read that the translation tables refuse. `client TCP (0x8)` is the texture-cache path, which
is where every GEMM's global loads go.

## 5. The status word, and the single bit that differs

The status word is printed by the driver already decoded. Checking the decode against the raw value over
the whole ring buffer (58 bursts, §7) and the packaged fix2 logs (7 bursts) gives one identity that holds
without exception: **bit 0 of `GCVM_L2_PROTECTION_FAULT_STATUS` is the `MORE_FAULTS` flag.**

That makes `0x00801031` and `0x00801030` the same fault report with `MORE_FAULTS` set or clear, which
matches the one other difference — the number of records:

- the trainer's GEMM has thousands of threads in flight; several of them read the same unmapped page
  before ROCr tears the queue down, so the driver logs the full decode once and then repeats the
  three-line address block (`MORE_FAULTS: 0x1`; measured 4–7 records per burst over the campaign — 17
  bursts of 4, 17 of 5, 13 of 6, 3 of 7, `fixes/fix2`'s rows all 4);
- our probe has exactly one thread issuing one load, so exactly one record and `MORE_FAULTS: 0x0`.

Nothing else in the status word separates the two.

## 6. The process-side death path is the same, including the coredump lines

`hip_vmm_tail_read` exits **141 (SIGPIPE)** by default and prints no coredump message. The trainer prints

```
Failed to write segment data to pipe: Broken pipe
GPU coredump: handler exited with error (status: 1)
GPU core dump failed
```

and exits **-6 (SIGABRT)** (measured natural exit; the fix3 packaged grid records only the harness's own
teardown signals, `-9`/`-15`). The difference is the SIGPIPE disposition, not the fault: CPython starts
with `SIGPIPE` set to `SIG_IGN`, a plain C++ program does not. Making the probe ignore it produces the
trainer's output verbatim:

```bash
bash -c 'trap "" PIPE; exec /tmp/hip_vmm_tail_read'   # exit=134, prints the three lines above
```

fix2's child log is the same text with `Failed to write program header: Broken pipe` instead, and
`rocdevice.cpp:3905` instead of `:3678` — that pack ran the ROCm 10.0.0 stack; the file/line pair is a
property of the runtime version, not of the fault.

## 7. The whole ring buffer, grouped

One boot (2026-09-15 23:27 → 2026-09-16 02:40) contains the fix3 campaign, the `hip_vmm_probe` draft that
faulted, and our runs. Grouping every `[gfxhub] page fault` block into bursts (a burst = consecutive
records carrying the same process and address; only the first carries the decode):

| process | status | ring | walker | mapping | bursts | records |
| --- | --- | --- | --- | --- | --- | --- |
| `python` | `0x0080113B` | 157 | `0x5` | `0x1` | 42 | 212 |
| `python` | `0x00801031` | 24 | `0x0` | `0x0` | 8 | 40 |
| `hip_vmm_tail_re` | `0x00801030` | 24 | `0x0` | `0x0` | 7 | 7 |
| `hip_vmm_probe` | `0x00801031` | 24 | `0x0` | `0x0` | 1 | 2 |

58 bursts; `bit 0 == MORE_FAULTS` consistent in all 58. All 58 faulting addresses are 2 MiB aligned, as
are all 7 packaged fix2 addresses. The `hip_vmm_probe` row is the throwaway first draft of the §5 probe
in `HIP.md`, whose host-location mapping was fed to `hipMemcpy`; that is the copy-engine sighting.

Grouping script (throwaway, kept out of the repo; feed it `dmesg` or `journalctl -k` text on stdin — it
prints one line per fault record, and a burst is the consecutive run of records with the same `proc` and
`addr`, of which only the first carries `status` and `more`):

```python
import re, sys
FAULT = re.compile(r"\[gfxhub\] page fault \(src_id:\d+ ring:(\d+) vmid:(\d+) pasid:(\d+)\)")
FIELDS = (("Process ", "proc"), ("in page starting at address ", "addr"),
          ("FAULT_STATUS:", "status"), ("MORE_FAULTS:", "more"))
cur = None
for line in sys.stdin:
    m = FAULT.search(line)
    if m:
        if cur:
            print(" ".join(f"{k}={v}" for k, v in cur.items()))
        cur = {"ring": m.group(1)}
    if cur is None:
        continue
    for key, name in FIELDS:
        i = line.find(key)
        if i >= 0 and name not in cur:
            cur[name] = line[i + len(key):].split()[0]
if cur:
    print(" ".join(f"{k}={v}" for k, v in cur.items()))
```

Its output on this boot's log (`sudo dmesg | python3 …`), shortened — a burst's repeated records each get
their own line and only the first of them carries `status`/`more`: 

```
ring=24 proc=python addr=0x00007ff310400000 status=0x00801031 more=0x1
ring=24 proc=python addr=0x00007ff310400000
ring=24 proc=python addr=0x00007ff310400000
ring=157 proc=python addr=0x00007f1a43000000 status=0x0080113B more=0x1
ring=24 proc=hip_vmm_tail_re addr=0x00007f7d37c00000 status=0x00801030 more=0x0
```

## 8. What the numbers say about the trainer's fault

- The status families are not two bugs: `0x00801031` vs `0x0080113B` differ in `WALKER_ERROR` /
  `MAPPING_ERROR` only, i.e. whether the page-table walk stopped before or at the PTE. Both appear for the
  same configuration in the same campaign (fix3 §2 says the address, not the kernel, moves the status).
- In this boot, **ring and status are perfectly correlated**: all 42 `ring 157` bursts carry
  `WALKER_ERROR 0x5` / `MAPPING_ERROR 0x1`, and all 16 `ring 24` bursts (python + both probes) carry a
  clean walk. fix2's packaged logs — the original capture and the after-reboot one, both on the ROCm 10.0.0
  stack — are `ring 157` for all 7 bursts. This is an
  observation, not a conclusion — the packets that fault on ring 24 in the trainer are the same Tensile
  GEMMs as on ring 157, and nothing here explains why the queue differs.
- A 2 MiB-aligned faulting address is not a property of the fault: the driver hands out 2 MiB-aligned
  device VA, so the first unmapped byte after any allocation boundary is 2 MiB aligned. Our run shows the
  same alignment for an address that was never part of a GEMM's buffer.

## 9. Cost, and the health of the card afterwards

Eight induced faults in the session — seven from this probe (four started by me, three by the author) and
one from the `hip_vmm_probe` draft. After them: `rocm-smi` edge/junction/memory 50/54/54 °C, a 4096²
bf16 matmul correct, 15.41 GiB free of 15.92 GiB. A page fault kills the faulting process's context; it
does not reset the device, and the kernel log of the whole boot contains no ring timeout, no GPU reset
and no `amdgpu: [drm] *ERROR*` line related to the faults.

## 10. What this does not settle

- It does not say why the Tensile kernel reads past its buffer. That remains `fixes/fix3` §2–§4: a
  precompiled gfx1201 Tensile solution selected by shape, whose overrun lands in unmapped VA depending on
  the allocator's layout (§7 of `HIP.md` lists the knobs that change *which* solution runs).
- It does not turn the guard-page workaround into a fix. `HSA_SVM_GUARD_PAGES=0` removes the abort by
  making the overrun land on mapped memory; this probe shows what the abort *is*, not that the overrun is
  harmless.
- It does not reproduce the fault with the trainer's own kernel; the probe uses its own `read_unmapped`
  kernel. The equivalence established here is at the driver/ROCr boundary, which is the layer both share.

## 11. Files

| Path | Role |
| --- | --- |
| `tools/hip/hip_vmm_tail_read.hip` | The probe (reserve 4 GiB → map 64 MiB → read the tail) |
| `HIP.md` §5 | The VMM API it uses, plus `/proc/self/maps`, `hipMemGetAccess` behaviour |
| `fixes/fix2/README.md`, `fixes/fix3/README.md` | The two campaigns whose kernel records are compared here |
| `fixes/fix3/resources/post-reboot-dmesg.txt`, `fixes/fix2/resources/fault-dmesg*.log` | The packaged kernel records |
| `fixes/fix3/resources/live-run-fault.txt`, `fixes/fix2/resources/fault-child.log` | The packaged process-side abort text |
| `tools/hip/hip_vmm_probe.hip` | The §5 probe; its first draft produced the `hip_vmm_probe` row in §7 |
| §12 below | The same fault traced to the one binary file that provides the kernel |

## 12. Which binary file holds the faulting kernel

**Answer: exactly one file in the conda stack.**

```
/home/acite/miniconda3/envs/axl_rocm_7_14/lib/python3.14/site-packages/_rocm_sdk_libraries/
  lib/hipblaslt/library/gfx1201/
  TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co
```

1,297,415 bytes, sha256 `d4def22c4a36a57ce33c7f5d1c2f0c709fc14e459b5b1df9051d0a61d9c04a07`. It is one
member of **hipBLASLt's** gfx1201 Tensile library; the solution metadata that selects it lives in the
sibling `…_l_Ailk_Bjlk_…gfx1201.dat.zlib`. rocBLAS, MIOpen and the torch kpacks do not provide this
kernel (§12.7).

### 12.1 The operation, re-measured

`fixes/fix3` §7 attributed the abort with `probe_op.py` under `HIP_LAUNCH_BLOCKING=1`; re-run on
2026-09-16 the last line before the abort is the same operation and the same shape:

| item | value |
| --- | --- |
| op | text-encoder LoRA A-weight gradient `grad_A = grad_outᵀ @ input` (`aten.mm`) |
| operands | `[[48, 308], [308, 1280]]` |
| `M` | 48 — `network_dim` (the rank) |
| `K` | 308 — `train_batch_size` (2) x encoder sequence length (154) |
| `N` | 1280 — `text_encoder_2` hidden size (the only 1280-wide module in the graph) |
| dispatch | `grid=[2560, 1, 1]`, `workgroup=[128, 1, 1]`, `group_seg_size=1638` |
| kernel | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…_ISA1201_…_WS32_WG64_2_1` (855 chars) |

### 12.2 What that file is

The `library/gfx1201/` directory holds hipBLASLt's Tensile library for one device, cut into one file
per problem type. Three kinds of file sit there:

| file | role |
| --- | --- |
| `…_l_<layout>_…gfx1201.dat.zlib` | **solution metadata**: zlib-compressed msgpack — every candidate kernel for that problem type, with its `name`, `kernelName`, `index`, size predicates and the library logic that picks between them |
| `…_l_<layout>_…gfx1201.co` | **the code**: the code object that actually contains those kernels |
| `TensileLibrary_lazy_gfx1201.dat.zlib`, `TensileLiteLibrary_lazy_gfx1201_Mapping.dat.zlib`, `Kernels.so-000-gfx1201.hsaco` | the handful of kernels hipBLASLt always needs, shipped uncompressed/inline instead of lazily |

The file name encodes what is inside: `BB_BB` (bf16 A and B), `HA_Bias_SAV_UA` (high-precision
accumulate, bias, the `SAV`/`UserArgs` variants) and the contraction layout `l_Ailk_Bjlk`, whose
letters attach `strideA0`/`strideB0` to the dimensions they name — a binding §12.9 and §12.10 read off
two actual calls, which is why the same kernel takes an `i`-fast and an `i`-slow operand without
changing its name. The directory also carries the same family with other layouts
(`l_Alik_Bljk`, `l_Ailk_Bljk`, `l_Alik_Bjlk`), which is why "the name is in this directory" is not
enough and the runtime had to be observed (§12.5).

`.co` is an AMD container, not an ELF: **CCOB** = a zstd-compressed code-object bundle. The compression
is why a plain `grep` over the ROCm tree finds neither this kernel name nor any other: the strings are
inside a zstd stream, and only the `.dat.zlib` (zlib) and the `.kpack` files (another container) are
decodable at all. hipBLASLt imports no zstd entry point of its own — only `inflate`, for the
`.dat.zlib` — so it hands the CCOB to `hipModuleLoad` and the runtime decompresses it:
`libamdhip64.so.7` links `librocm_sysdeps_zstd.so.1` and carries the `CCOB` magic.

### 12.3 How to unpack and read it

The steps below are packaged as `fixes/hip1/extract_kernel_codeobject.py`, which reads the copy of the
container that sits next to it:

```bash
python fixes/hip1/extract_kernel_codeobject.py               # unwrap + verify the kernel is inside
python fixes/hip1/extract_kernel_codeobject.py --list        # just the bundle's targets
python fixes/hip1/extract_kernel_codeobject.py --load-check  # + hipModuleLoadData/hipModuleGetFunction
```

It writes the code object to `/tmp/axl-hip1/…codeobject.elf` (29 MB, so it stays out of the repository
unless `--out` says otherwise). What it does, by hand:

**(a) The container header.** 32 bytes, then one zstd stream:

| offset | size | field | this file |
| --- | --- | --- | --- |
| 0 | 4 | magic | `CCOB` |
| 4 | 2 | container version | 3 |
| 6 | 2 | flags | 1 |
| 8 | 8 | compressed size | 1,297,415 |
| 16 | 8 | original size | 29,162,488 |
| 24 | 8 | checksum word | `603052fc31cc868f` |
| 32 | … | zstd frame | decompresses to the original size |

```python
import zstandard
p = ('/home/acite/miniconda3/envs/axl_rocm_7_14/lib/python3.14/site-packages/_rocm_sdk_libraries'
     '/lib/hipblaslt/library/gfx1201/'
     'TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co')
b = open(p, 'rb').read()
d = zstandard.ZstdDecompressor().decompress(b[32:], max_output_size=int.from_bytes(b[16:24], 'little') + 64)
open('/tmp/co_payload.bin', 'wb').write(d)
print(len(b), d[:24], d.find(b'\x7fELF'))     # 1297415 b'__CLANG_OFFLOAD_BUNDLE__' 4096
```

**(b) The payload is a clang offload bundle, not a bare code object.** Use the clang that ships with
the same stack instead of parsing the bundle by hand:

```bash
LLVM=/home/acite/miniconda3/envs/axl_rocm_7_14/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/bin
$LLVM/clang-offload-bundler --list --type=o --input=/tmp/co_payload.bin
#   hipv4-amdgcn-amd-amdhsa--gfx1201
#   host-x86_64-unknown-linux-gnu-
$LLVM/clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
      --input=/tmp/co_payload.bin --output=/tmp/co.o
```

`/tmp/co.o` is 29,158,392 bytes: an AMDGPU ELF (`Type: DYN`, `Machine: AMD GPU`, `OS/ABI: AMD HSA`),
15 sections, 8 program headers, three `LOAD` segments — read-only at vaddr 0 (`.note`, `.dynsym`,
`.dynstr`, `.rodata`), RX at `0x288f00` (`.text`), and a small RW one. It holds **427 kernels** (854
`.kd` entries, one set per symbol table): the whole problem-type library in one code object.

**(c) Read the symbol table.** Every kernel appears twice: a `FUNC` symbol for its entry point in
`.text`, and a 64-byte `OBJECT` symbol in `.rodata` — the *kernel descriptor* (`.kd`) that the runtime
hands to the hardware:

```bash
KEN=Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1
readelf -sW /tmp/co.o | grep -F "$KEN" | grep -F '.kd'
```

**(d) The metadata half** (`…_l_Ailk_Bjlk_…gfx1201.dat.zlib`, zlib + msgpack):

```python
import zlib, msgpack
P_DAT = ('…/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201/'
         'TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.dat.zlib')
KEN = 'Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1'
m = msgpack.unpackb(zlib.decompress(open(P_DAT, 'rb').read()), raw=False, strict_map_key=False)
print(len(m['solutions']))                                                 # 473 candidates
print([s['index'] for s in m['solutions'] if s['kernelName'] == KEN])      # [107020, 107190]
print([s['libraryLogicIndex'] for s in m['solutions'] if s['kernelName'] == KEN])   # [210, 380]
```

**(e) The other container in the tree** is the `.kpack` (`blas_lib_gfx1201.kpack`, torch/torchvision
and the other math libraries): magic `KPAK`, a 64-byte header whose second u64 is the manifest offset,
one zstd frame per member from `0x48` to that offset, and a msgpack `toc` at the manifest offset
mapping each member to its `ordinal`/`original_size`. Decoding it is what ruled out the rocBLAS
members as a source (§12.7).

### 12.4 Evidence 1 — the kernel is in that code object (static)

**(a) The symbols.** `readelf -sW` on the unbundled code object, filtered to the exact 855-character
name:

```
   665: 000000000056e400     0 FUNC    GLOBAL PROTECTED    8 <name>          <- entry point, .text
   813: 0000000000282380    64 OBJECT  GLOBAL PROTECTED    7 <name>.kd       <- kernel descriptor, .rodata
  (the same pair repeats in .symtab as 123859 / 123860)
```

**(b) The code.** The entry point disassembles to real gfx1201 ISA (RDNA4 `s_load_b32` /
`s_wait_kmcnt` forms), and the next kernel symbol sits at `0x583000`, so this kernel's code occupies
`0x14c00` bytes:

```bash
LLVM=…/_rocm_sdk_core/lib/llvm/bin
$LLVM/llvm-objdump -d --start-address=0x56e400 --stop-address=0x56e428 /tmp/co.o
#   000000000056e400 <label_ASM_Start>:
#   s_load_b32 s20, s[0:1], 0x0     // 00000056E400: F4000500 F8000000
#   s_load_b32 s22, s[0:1], 0x4     // 00000056E408: F4000580 F8000004
#   s_wait_kmcnt 0x0                // 00000056E420: BFC70000
#   s_lshr_b32 s21, s20, 30         // 00000056E424: 85159E14
```

**(c) The counts** that make the match specific rather than a coincidence:

| quantity | value |
| --- | --- |
| occurrences of the exact 855-char `kernelName` in the 29,162,488 B payload | 6 — twice in the AMDGPU metadata `.note`, twice in `.dynstr`, twice in `.strtab` |
| occurrences of the tuning-family prefix `…_MT64x128x16…` in that payload | 72 (the same tile with other `SN_` options) |
| kernels in the code object (`.kd` descriptors) | 427 (854 entries: `.dynsym` + `.symtab`) |
| solutions in the paired metadata | 473, of which **2** have exactly this `kernelName` (`index` 107020 / 107190, `libraryLogicIndex` 210 / 380) |
| files in the conda ROCm tree that contain this `kernelName` | 2 — this `.co` and its `.dat.zlib` |

**(d) The runtime accepts it.** `extract_kernel_codeobject.py --load-check` hands the extracted object
to `hipModuleLoadData` and looks the kernel up by name:

```
hipModuleLoadData   : ok      module=0x55c9065f7f80 (29158392 bytes)
hipModuleGetFunction: ok      function=0x55c9070cbd60
```

So this is not a look-alike: the HIP runtime loads the file and resolves the faulting kernel inside it.
Nothing is launched, so the check puts no work on the queue.

The same search over the Arch stack (`/opt/rocm/core/lib/hipblaslt/library/gfx1200|gfx1201/`) finds
the same name in the sibling `.dat.zlib` files there too, which is consistent with `fixes/fix2`
crashing on that stack with the same kernel string.

### 12.5 Evidence 2 — that file is what the trainer loads (runtime, eBPF)

The loading is invisible to a naive probe: hipBLASLt does it on **PyTorch's autograd thread**, whose
`comm` is `pt_autograd_0`, so a `comm == "python"` filter (bpftrace's `comm` is the *thread* name)
reports nothing at all — which is exactly what the first two attempts of this campaign did. With the
filter removed, one run of `fixes/fix3/repro_step1.py` produced this chain:

```
hipModuleLoad        comm=pt_autograd_0  file=…/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co
  -> err=0 module=0x7f68fa2e6d40
hipModuleGetFunction comm=pt_autograd_0  module=0x562d5010b840  name=Cijk_Ailk_Bjlk_…_MT64x128x16_…  -> err=500   (hipErrorNotFound)
hipModuleGetFunction comm=pt_autograd_0  module=0x562d63a45eb0  name=…                                -> err=500
hipModuleGetFunction comm=pt_autograd_0  module=0x562d5fa0d410  name=…                                -> err=500
hipModuleGetFunction comm=pt_autograd_0  module=0x562d649de640  name=…                                -> err=500
hipModuleGetFunction comm=pt_autograd_0  module=0x7f68fa2e6d40  name=…                                -> err=0 function=0x562d59dc3110
hipExtModuleLaunchKernel f=0x562d59dc3110 grid=2560,1,1 -> 488 dispatches
```

That is hipBLASLt's lazy loading pattern: ask every module already loaded for the kernel by name, and
only when none has it load the library that does — the file appears late in the run, and only once.
The other four modules are the `Kernels.so-000-gfx1201.hsaco` and the three sibling Tensile libraries
with different layouts. A second run with `HIP_LAUNCH_BLOCKING=1` (the run that also re-attributes the
op of §12.1) reproduced the same chain, `err=0` on the same library, 481 dispatches, `grid=2560,1,1`
again.

The probe that produced this — 6 uprobes, no `comm` filter — is §12.8.

**Unit correction (2026-09-16).** `grid=2560,1,1` here, and in the dispatch dump of §12.1, is
**work-items**, not workgroups: `hipExtModuleLaunchKernel` takes its global sizes in work-items
(`hip_ext.h`), and ROCr prints the same quantity. The standalone launcher of §12.9 settled it — it
launches 10 workgroups of 128 threads and its dump reads `grid=[1280, 1, 1]`. So this dispatch is
**20 workgroups**, not 2560, and the earlier reading of it as an over-provisioned grid (and the
HBM-table argument type that reading implied) was wrong. What the 20 are is settled in §12.10: 20 M
tiles of 64, i.e. hipBLASLt's m was 1280 and its n 48, not the 48/1280 the shape was assumed to have.

### 12.6 Evidence 3 — the fault log alone says so

The fault message that ROCr prints is preceded by a dump of the faulting dispatch packet, including
`kernel_obj`, the address of the kernel descriptor the hardware was executing. The descriptor in this
code object has vaddr `0x282380`; subtracted from each recorded `kernel_obj` it leaves a 2 MiB-aligned
load base, i.e. the descriptor *is* this code object's:

| run | `kernel_obj` | `kernel_obj` − `0x282380` | 2 MiB aligned |
| --- | --- | --- | --- |
| `fixes/fix3` live run, 2026-09-15 | `0x7f2cd2682380` | `0x7f2cd2400000` | yes |
| this session, trace run 2 | `0x7f3f92082380` | `0x7f3f91e00000` | yes |
| this session, trace run 4 | `0x7f77f5882380` | `0x7f77f5600000` | yes |
| this session, standalone launcher (§12.9), 2026-09-16 | `0x7f8556a82380` | `0x7f8556800000` | yes |

No eBPF needed for this one — it falls out of the kernel/ROCr logs already packaged in `fixes/fix2`
and `fixes/fix3` plus the symbol address in the code object. The last row is the strongest form of the
argument: a process that only ever loaded this one code object, launched this one kernel, and still
faulted on the descriptor at `0x282380`.

### 12.7 What this rules out

| candidate | how it was excluded |
| --- | --- |
| rocBLAS's Tensile library | `lib/rocblas/library/TensileLibrary_lazy_gfx1201.dat` (msgpack) contains `"solutions": []` — no gfx1201 solutions at all; the 67 rocBLAS code objects in `blas_lib_gfx1201.kpack` contain no `Cijk_`/`UserArgs` strings (§12.3e) |
| MIOpen | `~/.cache/miopen` (MIOpen's `MIOPEN_CUSTOM_CACHE_DIR`, per `start_train.sh`) contains no such name, and the kernels MIOpen does compile at runtime in this step are named `Im2d2Col_v2` / `batched_transpose_*` (visible in the same trace as `comgr_do_action` + `hipModuleGetFunction`) |
| torch / torchvision device code | `torch/.kpack/torch_gfx1201.kpack` (246 members) and `torchvision/.kpack/torchvision_gfx1201.kpack` (7 members) contain no such name |
| a JIT / cache artifact | the name exists in a shipped `.co` before the run starts; the run only reads it |
| tensors, the dataset or the mask path | unchanged from `fixes/fix3` §3 — the shape of the *operand* GEMM is what selects the kernel |

Why the trainer gets there at all: this torch build's BLAS backend is hipBLASLt by default
(`torch.backends.cuda.preferred_blas_library()` → `_BlasBackend.Cublaslt`), so `aten.mm` on
`text_encoder_2`'s attention projections is dispatched by the library above.

### 12.8 Reproduce

Runtime half — save the 6-uprobe script below as `/tmp/which_library.bt` (root; the point of it is
that there is *no* `comm` filter), then run one fault row under it:

```bash
sudo bpftrace /tmp/which_library.bt &
conda activate axl_rocm_7_14
python fixes/fix3/repro_step1.py --row "step-1 abort: repo config (dim 48, batch 2)" --in-place
```

```
uprobe:…/libamdhip64.so.7:hipModuleLoad
{ @out[tid] = arg0; printf("hipModuleLoad comm=%s file=%s\n", comm, str(arg1)); }
uretprobe:…/libamdhip64.so.7:hipModuleLoad
{ printf("  -> err=%d module=%p\n", retval, *(uint64 *)@out[tid]); }
uprobe:…/libamdhip64.so.7:hipModuleGetFunction
/strcontains(str(arg2), "Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16")/
{ @fnout[tid] = arg0; @ours[tid] = 1;
  printf("hipModuleGetFunction comm=%s module=%p name=%s\n", comm, arg1, str(arg2)); }
uretprobe:…/libamdhip64.so.7:hipModuleGetFunction
{ if (@ours[tid] == 1) {
      printf("  -> err=%d function=%p\n", retval, retval == 0 ? *(uint64 *)@fnout[tid] : 0);
      if (retval == 0) { @fname[*(uint64 *)@fnout[tid]] = 1; }
      delete(@ours[tid]); } }
uprobe:…/libamdhip64.so.7:hipExtModuleLaunchKernel
/@fname[arg0] == 1/
{ @launches = count(); @grid[arg1, arg2, arg3] = count(); }
END { print(@grid); printf("dispatches of that kernel: %lld\n", (int64)@launches); }
```

Static half — §12.3 (a)–(d): CCOB header → zstd → `clang-offload-bundler --unbundle` → `readelf -sW`
→ `.kd` symbol, with the `.dat.zlib` decoded for the solution identity.

Five faults were induced for this section (four plain runs, one with `HIP_LAUNCH_BLOCKING=1`), all in
the same kernel, `ring 157`, status `0x0080113B`. Afterwards: `rocm-smi` 48/51/50 °C, 15.23 GiB free,
a 4096² bf16 matmul correct, no ring timeout and no GPU reset in the boot.

### 12.9 The standalone launcher: what it does, and the layout it uses

`fixes/hip1/launch_fault_kernel.hip` takes the torch/hipBLASLt path out of the picture entirely: it
loads the code object from §12.3 itself, resolves the 855-char kernel with `hipModuleGetFunction`,
places the four operands itself and fires the same GEMM on it with `hipModuleLaunchKernel`. It is the
shortest program that still reproduces the fault — no command line, no reference check, no data
initialisation — because the two things that decide the outcome are *where the operands are* (below)
and the shape that is passed to the kernel.

```bash
bash fixes/hip1/repro_standalone.sh   # one command: extract, compile, run (fixes/hip1/hip1.md §12.11)

conda activate axl_rocm_7_14          # or any other ROCm environment; the steps the script runs:
python fixes/hip1/extract_kernel_codeobject.py                     # /tmp/axl-hip1/…codeobject.elf
hipcc --offload-arch=gfx1201 -O2 fixes/hip1/launch_fault_kernel.hip -o /tmp/launch_fault_kernel
/tmp/launch_fault_kernel
```

`repro_standalone.sh` runs those three steps in order and pins no environment: any `hipcc` on `PATH`
is accepted, the `hipcc`/`HIP_PATH`/`ROCM_PATH` it resolved and the `libamdhip64` the binary will load
are printed, and the exit status is inverted so that *the fault* is status 0 and a clean completion is
status 1 — a batch of runs across stacks can be read from those statuses alone.

`--offload-arch` keeps the build to about a second; the program contains no `__global__` kernel, so
there is no device code to generate. `conda activate` matters here for the same reason it does in
`HIP.md` §3: without it `hipcc` resolves its clang through `/opt/rocm` (the 7.15 stack) and the binary
loads that stack's `libamdhip64` at run time. That is a stack choice, not a mistake, and it is what the
printed lines are for: a run says which stack produced its fault.

What it does, in order:

| # | Step | Detail |
| --- | --- | --- |
| 1 | read the code object | `OBJECT_PATH` = the 29 MB ELF of §12.3 (b); the `.co` container itself is not accepted here, `hipModuleLoadData` wants the unbundled object |
| 2 | `hipModuleLoadData` + `hipModuleGetFunction` | only this file is loaded, and only this kernel is looked up — no other module is asked for it (contrast §12.5) |
| 3 | place the operands | four VMM placements, each end-aligned against an unmapped page — no data is written into them, the kernel only has to touch the wrong address, not the wrong value; 29 KB + 788 KB + 2 × 123 KB |
| 4 | fill the argument block | 144 bytes, one `Args` struct, table below |
| 5 | `hipModuleLaunchKernel` | `grid 10×1×1`, `block 128×1×1`, `sharedMemBytes 0`, arguments through the `extra` buffer — the raw-buffer form hipBLASLt itself uses, so the ABI layout is ours rather than the runtime's guess |
| 6 | `hipStreamSynchronize` | where a GPU fault surfaces; the program's last act, and its only diagnostic beyond ROCr's own line |

`sharedMemBytes` is 0 because the kernel's LDS is static: the descriptor declares
`group_segment_fixed_size = 1638` and the kernel sets `s_mov_b32 m0, 0x666` (= 1638) for its LDS
instructions, which is the `group_seg_size=1638` of the fault dump in §12.1. The block size is 128 for
the same reason (`max_flat_workgroup_size = 128`, and the dispatch packet of the real fault says
`workgroup=[128, 1, 1]`).

#### Operand layout

The letters of `Cijk_Ailk_Bjlk` are Tensile's index assignments: they name the **logical** operands and
list each tensor's dimensions from the slowest- to the fastest-varying, so `Cijk` is an M×N matrix with
`j` contiguous, `Ailk` an M×K matrix with `l` (= K) contiguous, and `Bjlk` an N×K matrix with `l`
contiguous. The sibling `.dat.zlib` solution then declares `transB: True` — the tensor the caller
*stores* is the transpose of that logical B. Torch stores row-major, so the stored operands are A =
[48, 308] with `l` contiguous, B = that transpose = [308, 1280] with `j` (= N) contiguous, and C = D =
[48, 1280] with `j` contiguous. The strides handed to the kernel are those of the logical dims *in that
stored memory*, which is why B's two strides look swapped next to its printed shape:

| Tensor | Shape | Contiguous dim | Strides passed (elements) | Bytes |
| --- | --- | --- | --- | --- |
| A | [48, 308] | `l` (K) | `strideA0 = 308` (i), `strideA1 = 1` (l) | 29,568 |
| B | [308, 1280] | `j` (N) | `strideB0 = 1` (j), `strideB1 = 1280` (l) | 788,480 |
| C, D | [48, 1280] | `j` (N) | `strideC0 = strideD0 = 1280` (i), `strideC1 = strideD1 = 1` (j) | 122,880 each |

With `i` = M = 48, `j` = N = 1280 and `l` = K = 308 (the shapes §12.1 measured), A is M×K row-major and
the stored B is K×N row-major — the pair of buffers torch handed hipBLASLt for
`grad_A = grad_outᵀ @ input`: `aten.mm([[48, 308], [308, 1280]])`, row-major, no transposes and no
batches (`SizesFree2 = 1`). The *roles* the library gives those two buffers are the other way round,
and §12.10 measures that: hipBLASLt's m is 1280 — this table's B — and its n is 48, this table's A.

#### Why the operands are placed with guard pages

The kernel's bug is a *read* past the end of an operand, and a read past the end of an operand only
faults if the bytes after it are not mapped. `hipMalloc` gives no say in that, and on this machine it
gives the worst case: measured on 2026-09-16 with a standalone allocation probe that dispatches nothing,
ROCclr's device heap bumped out **one 2 MiB `rw-s` arena** for all four buffers —

```
A 0x…600000 .. 0x…607380   (29,568 B)
B 0x…608000 .. 0x…6c8800   (788,480 B)   A ends 3,200 B before B
C 0x…6c9000 .. 0x…6e7000   (122,880 B)   B ends 2,048 B before C
D 0x…6e7000 .. 0x…705000   (122,880 B)   C ends exactly at D
7fa5d5600000-7fa5d5800000 rw-s … /dev/dri/renderD128     ← all four inside this one mapping
```

with roughly another megabyte of that same mapped arena left over after the last buffer. Every
candidate overrun is far smaller than that:

| overrun site | why | worst case |
| --- | --- | --- |
| A's K tail | K = 308 = 19×16 + 4, so l = 308…319 is fetched for the last row | 12 elements, 24 B |
| A's M tail | M = 48 < MT64, so rows i ≥ 48 are fetched | 16 rows, 9.6 KB |
| B's K tail | B is stored [K,N], so l = 308…319 is 12 whole rows past the end | 30 KB |

All three fit inside the arena with room to spare, so the `hipMalloc` version of this program would
have read a neighbour and finished silently — exactly the failure mode to avoid, and exactly why
`fixes/fix3`'s scan sees the same shape die in one bucket and survive in another: the *allocator*
decides, not the shape. The program therefore places each operand itself with the VMM API (`HIP.md`
§5, proven on this card): reserve `mapped + 4096`, create and map `mapped = ceil(size/4096)×4096`
bytes of it, grant read-write access, and hand the kernel `base + mapped - size` — so the operand's
**last byte is the last byte of the mapping**, and the guard page right after it is reserved but never
mapped. One byte past the end now faults no matter how large the overrun is. Verified without
dispatching, by compiling the same source with the launch cut off: the mappings in `/proc/self/maps`
end exactly where the operands end, and nothing is mapped for some pages after —

```
A  0x7f31469c0c80 .. 0x7f31469c8000  29568 bytes, 3200 bytes of lead-in, guard page at 0x7f31469c8000
7f31469c0000-7f31469c8000 ---s … /dev/dri/renderD128     ← A's mapping, 32 KiB, ends at A's last byte
```

The lead-in (`4096 - size % 4096`, e.g. 3,200 B for A) is mapped but unused, so an overrun below the
operand would still be silent for that many bytes; the geometry above is upward (past the K end, past
the M end), and ROCr reports the direction of a fault in its status word — `RW 0x0` for a read, as in
the fault log of §12.1. Guarding all four operands is deliberate: C and D are guarded too, so if the
kernel's store masking is also wrong the fault will say so, at C's or D's guard page, with `RW 0x1`.

#### Result — the fault, and where it came from

Run on 2026-09-16 in the environment's own stack (`conda activate axl_rocm_7_14`,
`hipcc --offload-arch=gfx1201 -O2 launch_fault_kernel.hip -o /tmp/launch_fault_kernel`):

```
device            : AMD Radeon RX 9070 XT (gfx1201, 32 CUs)
module            : 0x55c49d93ef80, function 0x55c4a02693c0
A  0x7f8773500c80 .. 0x7f8773508000  29568 bytes, 3200 bytes of lead-in, guard page at 0x7f8773508000
B  0x7f8765900800 .. 0x7f87659c1000 788480 bytes, 2048 bytes of lead-in, guard page at 0x7f87659c1000
C  0x7f87734e0000 .. 0x7f87734fe000 122880 bytes,    0 bytes of lead-in, guard page at 0x7f87734fe000
D  0x7f87734c0000 .. 0x7f87734de000 122880 bytes,    0 bytes of lead-in, guard page at 0x7f87734de000
problem           : A[48,308] x B[308,1280] -> D[48,1280], bf16, grid 10, block 128
VGPU=0x55c4a02a5d80 SWq=0x7f87734be000, HWq=0x7f875d500000, id=1
    Dispatch Header =0xd02 (type=2, barrier=1, acquire=2, release=1), setup=3
    grid=[1280, 1, 1], workgroup=[128, 1, 1]
    private_seg_size=0, group_seg_size=1638
    kernel_obj=0x7f8556a82380, kernarg_address=0x0x7f875d000000
    completion_signal=0x0, correlation_id=0
    rptr=0, wptr=2
:0:rocdevice.cpp            :3678: 17170533569 us:  Memory Fault Error [host: acitehost, GPU index: 0,
    faulting addr: 0x7f8773509000, kernel: Cijk_Ailk_Bjlk_…_WS32_WG64_2_1]
Memory access fault by GPU node-1 (Agent handle: 0x55c49d987640) on address 0x7f8773509000.
    Reason: Page not present or supervisor privilege.
```

What it establishes:

- **The kernel faults on its own.** Same kernel name, same `group_seg_size=1638`, and
  `kernel_obj` − `0x282380` = `0x7f8556800000`, 2 MiB aligned — this run is the fourth row of the
  §12.6 fingerprint table. One module, four buffers, one dispatch: no torch, no hipBLASLt, no MIOpen,
  no dataset, and no allocator neighbours.
- **The overrun is past A**, and the guard pages are what makes that unambiguous. The faulting address
  `0x7f8773509000` sits one page past where A's mapping ends (`0x7f8773508000`, A's guard page), while
  B, C and D have their guard pages at `0x7f87659c1000`, `0x7f87734fe000` and `0x7f87734de000` — none
  of them anywhere near. A is the M×K operand (`Ailk`), i.e. this is the edge of A's `i` dimension:
  `M = 48 < MT64`, and rows 48…63 span A's end plus 9,856 B = 2.4 pages. The 24-byte K tail of the last
  row is not excluded by anything in this log, but only the `i` tail reaches a second page past the
  end — which is the page that was reported. Which of A's dimensions that tail belongs to depends on
  the argument block, and §12.10 shows the library sends M and N the other way round (m = 1280,
  n = 48), so under the real descriptors the same overrun is this operand's `j` tail.
- **The dispatch dump's `grid` is in work-items.** This launch asked for 10 workgroups × 128 threads
  and ROCr printed `grid=[1280, 1, 1]`. That corrects §12.5: the `grid=[2560, 1, 1]` of the real
  dispatch is **20 workgroups**, not 2560 — and §12.10 shows they are 20 M tiles of 64, the count a
  library-side m of 1280 asks for.

This transcript is ROCr's user-space output. The classification of the fault — the GCVM status word,
whose read/write bit distinguishes a load from a store — is in the kernel log, the same `0x0080113B`
of §12.1; `fixes/fix3/resources/live-run-fault.txt` and `fixes/fix2/resources/fault-dmesg*.log` are
the two examples of that side by side with their runs.

#### The argument block, and where every byte of it comes from

The 144-byte block is not reconstructed from the compiled kernel's behaviour: the code object carries
its own declaration. `.note` (file offset `0x200`, size `0x1c52f4`) holds **427** notes, all with name
`AMDGPU` and type 32 (`NT_AMDGPU_METADATA`), one per kernel, each a msgpack map:

```python
import struct
b = open('/tmp/axl-hip1/…codeobject.elf', 'rb').read()
note = b[0x200:0x200 + 0x1c52f4]        # walk the notes, take the one whose '.name' is the kernel
# {'amdhsa.kernels': [{'.name': 'Cijk_Ailk_Bjlk_…_WG64_2_1',
#                      '.kernarg_segment_size': 144, '.group_segment_fixed_size': 1638,
#                      '.max_flat_workgroup_size': 128, '.sgpr_count': 70, '.vgpr_count': 164,
#                      '.args': [{'.name': 'Gemm info', '.offset': 0, '.size': 4,
#                                 '.value_kind': 'by_value', '.value_type': 'u32'}, … 30 in all]}]}
```

Every `Args` field in the program is one `.args` entry, at the `.offset` the metadata gives:

| Offset | Size | Name | Value in the launch | Note |
| --- | --- | --- | --- | --- |
| 0 | 4 | Gemm info | `0x00000001` | arg type 0, 1 GEMM |
| 4 | 4 | kernel info0 | `0x23200001` | GSU 1, staggerU 32 / shift 3 / mapping 1 |
| 8 | 4 | kernel info1 | `0xffc10008` | WGM 8, WGMXCC 1, WGMXCCG -1 |
| 12 | 4 | numWG | 10 | the grid size, `NUM_WG == GRID` |
| 16 | 4 | SizesFree0 | 48 | M |
| 20 | 4 | SizesFree1 | 1280 | N |
| 24 | 4 | SizesFree2 | 1 | the batch dim: not a batched problem |
| 28 | 4 | SizesSum0 | 308 | K |
| 32, 40, 48, 56 | 8 each | D, C, A, B | `hipMalloc` device pointers | bf16 in the object's metadata |
| 64, 68, 72, 76 | 4 each | strideD0, D1, C0, C1 | 1280, 1, 1280, 1 | |
| 80, 84, 88, 92 | 4 each | strideA0, A1, B0, B1 | 308, 1, 1, 1280 | |
| 96, 100 | 4 each | alpha, beta | `1.0f`, `0.0f` | f32, the metadata's `useBeta: True` |
| 104 | 4 | ESMRuntimeSupported | 0 | |
| 108, 116 | 8 each | AddressScaleAlphaVec, bias | null | **never read** |
| 124, 128 | 4 each | biasType, StrideBias | 0 | never read |
| 132, 136, 140 | 4 each | activationAlpha, Beta, Type | 0 | never read; type 0 = no activation |

The tail is dead weight, not an oversight: the kernel's last kernarg load is
`s_load_b32 s47, s[0:1], 0x58` (0x56e460) off a base of `kernarg + 16`, i.e. offset **104**, so nothing
past `ESMRuntimeSupported` is touched in this path (`useBias: 1` and `activationType: Hipblaslt_all`
in the `.dat` describe what the *solution family* supports, not what this tuning reads).
`sizeof(Args) == 144` is asserted at compile time so the block keeps matching
`.kernarg_segment_size`.

#### The four packed scalars

These are the only values in the block that are not simply M, N, K or a pointer. Each one's bit layout
is read out of the kernel's own prologue (the second table column is the instruction that consumes it):

| Field | Bits | Value | Read by |
| --- | --- | --- | --- |
| Gemm info | 31:30 argument type | 0 | `s_lshr_b32 s21, s20, 30` (0x56e424), then `s_cmp_eq_u32 s21, 3` (0x56e430) and `s_cmp_eq_u32 s21, 0` (0x56e438): 0 and 3 fall into the inline-argument path that loads the block from `kernarg+0x10` (`label_Bypass_ArgType3_to_ArgType0_Instance1`, 0x56e440), anything else into `label_HBMArgs` (0x56e46c), which loads a *pointer* at `kernarg+0x10` instead |
| Gemm info | 29:0 GEMM count | 1 | read only in the multi-GEMM path (`s_mul_i32 s54, s20, 4`, 0x56e858); unreachable for argument type 0 |
| kernel info0 | 13:0 GSU | 1 | `s_and_b32 s17, s46, 0x3fff` (0x56e734) multiplied into the tile count, and `s_cmp_eq_u32 s16, 1` (0x56ec24) → `label_GSU` (0x56ed50), which zeroes the GSU workspace pointer |
| kernel info0 | 14 GSUWGMRR | 0 | `s_and_b32 s16, s46, 0x4000` (0x56ec2c) → `label_GSUWGMRR` (0x56ecc4) |
| kernel info0 | 15 GSUC | 0 | `s_and_b32 s16, s46, 0x8000` (0x56f0e8, inside `label_WGM`) → `label_GSUC_A` (0x56f100) |
| kernel info0 | 23:16 staggerU | 32 | `s10 = kernel info0 >> 16`, then `s_and_b32 s10, s10, 0xff` (0x56f5f8) |
| kernel info0 | 28:24 staggerUStrideShift | 3 | `s_and_b32 s18, s10, 0x1f00` (0x56f5e0) + `s_lshr_b32 s18, s18, 8` (0x56f5ec), used as `s_lshl_b32 s17, s16, s18` (0x56f60c) = 32 << 3 = 256 |
| kernel info0 | 31:29 staggerUMapping | 1 | `s_and_b32 s19, s10, 0xe000` (0x56f5f0), compared against 0x2000/0x4000/0x6000/0x8000 = mappings 1–4 (0x56f63c) |
| kernel info1 | 15:0 WGM (signed) | 8 | `s_mov_b32 s16, s11` (0x56ed5c) + `s_sext_i32_i16 s16, s16` + `s_cmp_gt_i32 s16, 1` (0x56ed6c) → `label_WGMPositive` (0x56eecc); that register is then the divisor of the workgroup-mapping remap |
| kernel info1 | 21:16 WGMXCC | 1 | `s_lshr_b32 s52, s11, 16` (0x56e4b4) + `s_ctz_i32_b32 s52, s52` (0x56e4bc) + `s_cmp_gt_i32 s52, 0` (0x56e4c4): a field of 1 gives `ctz = 0` and skips the whole XCC remap |
| kernel info1 | 31:22 WGMXCCG | -1 (0x3ff) | `s_lshr_b32 s53, s11, 22` (0x56e4c0) |
| numWG | all 32 | 10 | the grid size, passed so the kernel can partition workgroups between XCDs; moot while WGMXCC is 1 |

The *values* are not invented either: the sibling `.dat.zlib` solution for this kernel (index 107020,
§12.3 d) spells its runtime-knob suffix out as `GSU1_GSUAMB_GSUC0_GSUWGMRR0_…_SU32_SUM1_SUS256_…_
WGM8_WGMXCC1_WGMXCCGn1`, and its `internalArgsSupport` is `{version: 2, gsu: True, wgm: True,
staggerU: True, useUniversalArgs: True}` — i.e. exactly these quantities are the ones the kernel is
built to take at run time rather than at compile time (`SU32_SUM1_SUS256` is the 32 / 1 / 256 decoded
in the table above). A `-1` in those names means "not pinned", which is why WGMXCCG is signed and
`stride shift` is a shift rather than a stride.

Two of them are not free to zero: the tile arithmetic of the normal path divides by GSU
(`s4 = wgid / (tiles_i × tiles_j × GSU)`), and a WGM of 0 or 1 takes a different branch than 8. The
others only reorder work or pick a remap, which is why the launch is still *correct* with them
mismatched — the shape, the strides and the pointers are what decide which bytes the kernel touches.

#### What is verified, and what is not

- **Build**: clean with `hipcc --offload-arch=gfx1201 -O2 -Wall` (no warnings) against the
  environment's 7.14 stack; 226 lines, one `main`.
- **Everything up to the dispatch** has been run, most recently as an identical copy of the source with
  the launch cut off: it prints the device (`AMD Radeon RX 9070 XT (gfx1201, 32 CUs)`), the module and
  function pointers, the four placements, and leaves `/proc/self/maps` showing each mapping ending
  exactly at its operand's last byte with the guard pages absent — the placement transcript above.
- **The dispatch has been run, and it faults** — §12.9 "Result". If a future run prints `completed
  without a fault`, no overrun reached a guard page, which would rule this shape and this kernel out
  as the fault's cause.
- `hipFuncGetAttributes` is deliberately absent: it rejects a kernel that came from a raw code object
  (`hipErrorInvalidDeviceFunction`, "invalid device symbol"), so the runtime's own idea of the kernarg
  size is not available as an independent check; the 144 comes from the metadata note.
- One fidelity caveat, which does not change the shape: the four packed scalars and `numWG` are what
  the kernel's prologue and the solution metadata imply, not a recording of what hipBLASLt sent. The
  fault did not depend on them: this run passed `numWG = 10`, the grid it actually launched, and the
  argument type the kernel's own prologue uses for inline arguments.

### 12.10 The rocBLAS route: what the library sends, and why the grid is 20

§12.9 takes hipBLASLt out of the picture by loading the code object and driving the kernel by hand.
This route does the opposite: it keeps the library and takes away everything else — torch, the
dataset, the allocator — by calling `rocblas_gemm_ex` with the same operands as §12.9 and the same VMM
guard-page placement around them (`fixes/hip1/launch_fault_via_rocblas.hip`). That is what makes it a
second route rather than a repeat: the 144-byte block of §12.9 is *our* reconstruction of what
hipBLASLt would have sent, and here hipBLASLt sends it.

```bash
bash fixes/hip1/repro_rocblas.sh      # one command: locate rocBLAS, compile, run (fixes/hip1/hip1.md §12.11)

conda activate axl_rocm_7_14          # or any other ROCm environment; the steps the script runs:
hipcc --offload-arch=gfx1201 -O2 fixes/hip1/launch_fault_via_rocblas.hip \
    -o /tmp/launch_fault_via_rocblas -lrocblas
/tmp/launch_fault_via_rocblas
```

`repro_rocblas.sh` compiles and runs the same program under the same convention as §12.9's script, and
adds the one thing this route needs on a stack it has not seen: it looks for `librocblas.so` along
`LD_LIBRARY_PATH`, then under `$HIP_PATH`/`$ROCM_PATH`, then in `/opt/rocm/lib`, and passes that
directory to the compiler — the ROCm wheels keep rocBLAS in `_rocm_sdk_libraries/lib`, a system install
in `<root>/lib` (`HIP.md` §1, §2).

No `extract_kernel_codeobject.py` step: the library loads its own container.

**The library follows the transpose flags.** Which of the four Tensile contraction libraries a call
reaches is decided by the two transpose flags, and the mapping is not the one the names suggest. One
run per row, legal leading dimensions throughout, the library read out of an `strace -e openat`:

| transA | transB | library opened |
| --- | --- | --- |
| none | none | `…_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1201.co` |
| **none** | **transpose** | `…_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co` |
| transpose | none | `…_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201.co` |
| transpose | transpose | `…_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx1201.co` |

A's letter follows `transA` (none → `Ailk`) and B's follows `transB` (transpose → `Bjlk`), so the
library §12.3 identified — the one the faulting kernel is in — is the `none/transpose` pair. rocBLAS
never enters its own Tensile path for this call: `strace` shows `hipblaslt/library/gfx1201/` — the
`Kernels.so-000-gfx1201.hsaco`, the two lazy `.dat.zlib` indexes, and one contraction's `.co` — and
nothing under `rocblas/library/`, which is the positive form of §12.7's exclusion.

**A first attempt transcribes the report's labels literally, and faults on the wrong kernel.**
Written as m = 48, n = 1280, k = 308 with `transA=transpose`, `transB=none`, `lda=k`, `ldb=n` — the two
stored tensors called the transposes of the operands §12.1 names — the call does fault, but on

```
kernel: Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x32_…_WS32_WG32_4_1
        grid=[2560, 1, 1], group_seg_size=26624
```

and the fault is a property of the arguments, not of the kernel. That call is not the column-major
form of this `aten.mm` at all, and its `ldb` is not a stride either buffer has: with `transB=none` the
operand is declared a column-major 308×1280 matrix, whose last element lies at 1,637,427 while the
buffer holds 394,240 — 2.4 MB past the end, far enough to reach unmapped memory, edge-tile overrun or
no edge-tile overrun. It is worth recognising as a second way to get a `Page not present` on this
card, and it says nothing about §12.1.

**The reproducing call is the faithful column-major form of that GEMM, not a rearrangement.** Each
descriptor is one of the trainer's tensors with its *own* strides, read column-major:

| hipBLASLt descriptor | the tensor | why it is that tensor |
| --- | --- | --- |
| A, m×k = 1280×308, `lda` 1280, `opA` none | `input`, row-major [308, 1280] | element (j, l) at `l·1280 + j` — a row-major tensor read column-major is exactly this, no transpose in memory |
| B, n×k = 48×308, `ldb` 48, `opB` transpose | `grad_out`, row-major [308, 48] | stored as its own transpose, (q, l) at `q + 48·l`; supplying it transposed is what makes it the k×n operand |
| D, m×n = 1280×48, `ldd` 1280 | `grad_A`, row-major [48, 1280] | the column-major image of the row-major output, so the result needs no transpose on the way out |
| m 1280, n 48, k 308 | | `D(j, q) = Σ_l input(l, j)·grad_out(l, q)`, i.e. `grad_outᵀ @ input` |

So the m/n exchange is not a trick played on the library: it is what `aten.mm([[48, 308], [308, 1280]])`
*is* when its operands are described the way a column-major API describes them, and the leading
dimensions are the tensors' own strides. This is the call the trainer made: the byte layouts above are
`grad_out`'s and `input`'s rather than layouts picked to steer the selection, and the dispatch it
produces is identical to the recorded fault on every field the runtime prints (table below). It selects

```
getBestSolutions(): sol-idx = 107020
kernel  : Cijk_Ailk_Bjlk_…_MT64x128x16_…_WS32_WG64_2_1
```

`107020` is the `index` §12.3 (d) read out of the paired `.dat.zlib` for exactly this `kernelName`, and
the one whose runtime-knob suffix §12.9 decodes (the `.dat` holds two such solutions, 107020 and
107190). The library's own log and the static read of the metadata name the same solution from
opposite directions.

Changing only the two sizes, to m = 48, n = 1280 with the same flags and legal leading dimensions,
reaches the same library but a different solution — `sol-idx 107062`,
`…_MT32x32x64_…_WS32_WG32_4_1` — which with the guard pages in place completes without faulting. The
sizes are part of the solution's selection, not only of its grid.

The dispatch then matches §12.1 on the same fields §12.9 matches:

| field | §12.1 live fault | this route | §12.9 launcher |
| --- | --- | --- | --- |
| `grid` | `[2560, 1, 1]` | `[2560, 1, 1]` | `[1280, 1, 1]` |
| `workgroup` | `[128, 1, 1]` | `[128, 1, 1]` | `[128, 1, 1]` |
| `private_seg_size` | 0 | 0 | 0 |
| `group_seg_size` | 1638 | 1638 | 1638 |
| `kernel_obj` − `0x282380` | 2 MiB aligned | 2 MiB aligned | 2 MiB aligned |

The launcher's `grid` differs on purpose: its hand-written argument block says `SizesFree0 = 48`, so it
asks for the 10 workgroups that problem needs.

**What this settles.** The `grid = 2560` of §12.1 is **20 workgroups, and they are 20 M tiles**:
`MT64x128` covers the 1280 dimension 20 times and the 48 dimension once, so the library's m was 1280,
not 48. That is the factor of two §12.5 left open. It also means the hand-built argument block of
§12.9 passes M and N the other way round from what the library sends, so the overrun it measured is the
`i` tail of the [48, 308] operand while the real one is that operand's `j` tail — the same operand, the
same mechanism (a 48-long free dimension against a 64- or 128-wide tile), a different axis label. The
guard-page fault addresses show the difference in size: 9,854 B past the operand under §12.9's strides,
158 B under the library's.

Run on 2026-09-16, in the environment's own stack:

```
device            : AMD Radeon RX 9070 XT (gfx1201, 32 CUs)
A  0x7f7701f00800 .. 0x7f7701fc1000 788480 bytes, 2048 bytes of lead-in, guard page at 0x7f7701fc1000
B  0x7f7726fa0c80 .. 0x7f7726fa8000  29568 bytes, 3200 bytes of lead-in, guard page at 0x7f7726fa8000
D  0x7f77266e0000 .. 0x7f77266fe000 122880 bytes,    0 bytes of lead-in, guard page at 0x7f77266fe000
problem           : A[1280,308] x B[48,308] -> D[1280,48], bf16, transA none, transB transpose,
                    lda 1280, ldb 48, ldd 1280
VGPU=0x55d154323fb0 SWq=0x7f77266dc000, HWq=0x7f76f8c00000, id=1
    Dispatch Header =0xd02 (type=2, barrier=1, acquire=2, release=1), setup=3
    grid=[2560, 1, 1], workgroup=[128, 1, 1]
    private_seg_size=0, group_seg_size=1638
    kernel_obj=0x7f74e9682380, kernarg_address=0x0x7f76f8e00200
:0:rocdevice.cpp            :3678: 18126037308 us:  Memory Fault Error [host: acitehost, GPU index: 0,
    faulting addr: 0x7f7726fa8000, kernel: Cijk_Ailk_Bjlk_…_WS32_WG64_2_1]
Memory access fault by GPU node-1 (Agent handle: 0x55d154323fb0) on address 0x7f7726fa8000.
    Reason: Page not present or supervisor privilege.
```

Three operands, not four: C and D are the same buffer, which a `beta = 0` GEMM allows. The fault lands
on **B**, the 29,568-byte [48, 308] operand, at the **first byte of its guard page** — B's mapping ends
exactly at `0x7f7726fa8000` — while A's and D's guard pages (`0x7f7701fc1000`, `0x7f77266fe000`) are
nowhere near. Two bf16 values per element and `j = 48…127` over `l ≤ 307` put the furthest read
79 elements past the operand, 158 B, so the overrun stays inside the first guard page rather than
reaching the second as §12.9's does.

**The same call on the other stack faults in the same place, in a differently tuned build of the same
solution.** Run on 2026-09-16 under the system ROCm of `HIP.md` §1 — `HIP_PATH=/opt/rocm/core`, HIP
7.15.26333, no activation at all — the script compiles against that stack and the fault returns:

```
hipcc     : /opt/rocm/bin/hipcc
HIP_PATH  : /opt/rocm/core
rocBLAS   : /opt/rocm/core/lib
librocblas.so.5 => /opt/rocm/core/lib/librocblas.so.5
libamdhip64.so.7 => /opt/rocm/core/lib/libamdhip64.so.7
A  0x7f80ff300800 .. 0x7f80ff3c1000 788480 bytes, 2048 bytes of lead-in, guard page at 0x7f80ff3c1000
B  0x7f80ffe58c80 .. 0x7f80ffe60000  29568 bytes, 3200 bytes of lead-in, guard page at 0x7f80ffe60000
D  0x7f80ffe30000 .. 0x7f80ffe4e000 122880 bytes,    0 bytes of lead-in, guard page at 0x7f80ffe4e000
problem           : A[1280,308] x B[48,308] -> D[1280,48], bf16, transA none, transB transpose,
                    lda 1280, ldb 48, ldd 1280
    grid=[2560, 1, 1], workgroup=[128, 1, 1]
    private_seg_size=0, group_seg_size=1638
:0:rocdevice.cpp            :3905: 18674934042 us:  Memory Fault Error [host: acitehost, GPU index: 0,
    faulting addr: 0x7f80ffe60000, kernel: Cijk_Ailk_Bjlk_…_WS32_WG64_2_1]
Memory access fault by GPU node-1 on address 0x7f80ffe60000. Reason: Page not present or supervisor privilege.
```

Every number the selection produces is unchanged — `grid=[2560, 1, 1]`, `workgroup=[128, 1, 1]`,
`group_seg_size=1638` — and the fault is again the **first byte of B's guard page**, so the 158-byte
overrun is the same distance in both builds. What the stack changes is which build of the solution
answers, and ROCr prints the difference in the name: at 885 characters against the 855 of §12.1, the
7.15 build's `kernelName` carries four tunings the 7.14.1 one does not — `BL1_BS1` after `ASEM1`,
`SKWS0` after `SKFDPO0`, `UDFMAC0` after `USLMX0`, and `WGMXCC1` at the end — around an identical
`MT64x128x16_MI16x16x1` solution and an identical `WS32_WG64_2_1` tail. ROCr itself differs between the
stacks too (`rocdevice.cpp:3905` against `:3678`), so the comparison is between two independently built
libraries, not two builds of one file.

The two routes answer different cross-stack questions, and the difference is worth keeping in view:

| Route | What a run on a second stack measures |
| --- | --- |
| §12.9, `repro_standalone.sh` | the **fixed 7.14-extracted object** on that stack's runtime: the bytes are the same everywhere, both stacks print the same `kernelName`, and both fault in the same place — A's guard page + 4096, which is 9,854 B past the operand |
| §12.10, `repro_rocblas.sh` | that **stack's own hipBLASLt**: the program is recompiled and the library chooses its own build of the solution, which is how the four extra tunings above appear |

#### What is verified, and what is not

- **Build**: clean under `hipcc --offload-arch=gfx1201 -O2 -Wall`, 193 lines, one `main`.
- **The rocBLAS API is declared in the file, not included.** No ROCm wheel in this environment ships
  rocBLAS headers, and the only copy on the machine is `/opt/rocm`'s, part of the HIP 7.15 stack;
  adding `-I /opt/rocm/core/include` would shadow the environment's HIP 7.14 headers with 7.15 ones —
  the pitfall `HIP.md` §6 describes. The three prototypes and the enum values are transcribed from
  `…/rocblas/internal/rocblas-types.h`.
- **The dispatch is the library's, not ours.** This program passes four descriptors and two scalars;
  the solution, the strides, the packed scalars, `numWG` and the grid all come out of hipBLASLt. That
  the result is bit-identical to §12.1 in `group_seg_size`, grid, workgroup, kernel name and
  `kernel_obj` offset is the evidence that §12.9's reconstruction was faithful.
- **The argument block is not visible from here.** §12.9's table is the only place those 144 bytes are
  enumerated; this route confirms the four scalars that matter by reproducing their effects.
- **Two stacks.** The same call, run un-activated against the system ROCm 7.15.26333 as well as in the
  environment's 7.14.1, faults in both — same grid, same `group_seg_size=1638`, the fault again on B's
  first guard byte — while each stack names its own build of the solution. The cross-stack comparison
  above is from those two runs; the program is recompiled by each stack, so no 7.14 object is carried
  into the 7.15 run.
- Not verified: the fault's read/write bit, which like §12.9 lives in the kernel log (`0x0080113B` of
  §12.1) and not in ROCr's user-space output.

### 12.11 Files

| Path | Role |
| --- | --- |
| `_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co` | The file the kernel is in (CCOB, 1,297,415 B) |
| `…same name….dat.zlib` | Its selection metadata (zlib + msgpack, 473 solutions) |
| `fixes/hip1/TensileLibrary_BB_BB_…_gfx1201.co` | A copy of that container in this directory, so §12 is reproducible from the repository alone |
| `fixes/hip1/extract_kernel_codeobject.py` | Unwraps the container (§12.3 a–c), checks the kernel's symbols in the result, and with `--load-check` proves the runtime accepts it |
| `fixes/hip1/launch_fault_kernel.hip` | The standalone launcher of §12.9: loads the code object, places A/B/C/D with VMM guard pages, and fires the 48×1280×308 bf16 GEMM on this kernel |
| `fixes/hip1/launch_fault_via_rocblas.hip` | The rocBLAS launcher of §12.10: same operands and the same guard-page placement, one `rocblas_gemm_ex` — the route where hipBLASLt chooses the solution instead of us |
| `fixes/hip1/repro_standalone.sh` | Route 1 in one command (§12.9): extracts the code object, compiles `launch_fault_kernel.hip`, runs it, and prints the `hipcc`/`HIP_PATH`/`libamdhip64` it resolved. Pins no environment, and exits 0 when the program faults and 1 when it does not |
| `fixes/hip1/repro_rocblas.sh` | Route 2 in one command (§12.10): locates `librocblas.so` on that stack, compiles `launch_fault_via_rocblas.hip` against it, runs it, same reporting and same exit convention |
| `fixes/fix3/probe_op.py`, `fixes/fix3/repro_step1.py` | The op attribution and the reproducer used here |
| `fixes/fix3/README.md` §7, `fixes/fix3/resources/probe-op.md` | The operation and shapes re-confirmed in §12.1 |
| `fixes/fix2/resources/fault-dmesg*.log` | Contains a `kernel_obj` used in §12.6 |
| §12.8 runtime probe | Inline in §12.8; the bpftrace script itself is not yet a file in the repository |

### 12.12 The fp16 sibling: measured, then withdrawn

For one revision, both programs took a second element type and route 2 a third — `fp16`, plus `f32` as a
control — to answer a question neither route answers on its own: is the overrun a property of *this*
kernel (its bf16 loads, its 1,638-byte LDS, its `MT64x128x16` tile), or of the geometry it was tuned for,
a 48-long free dimension against a 64- or 128-wide tile? The programs and both scripts have since been
returned to their single-variant form, so no fp16 code path is left in the tree; this section is its
record. What differed in the runs below is the element type, the kernel name and the code object only —
the shapes, the strides and the leading dimensions, the guard-page placement and the grid are those of
§12.9 for route 1 and §12.10 for route 2.

**The fp16 sibling of the faulting kernel.** hipBLASLt ships one Tensile contraction library per element
type. The faulting kernel lives in `…_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co`; the fp16
library of the same contraction is `…_Type_HH_HPA_…`, and its kernel of the same tuning chain is the bf16
one with the dtype letters of its global loads changed:

```
bf16  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…
fp16  Cijk_Ailk_Bjlk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…
```

Both are 855 characters and identical outside `BBS_BH`→`HHS_BH`, and the metadata of the two `.kd`
symbols agrees in every field the argument block of §12.9 fills in — so that block, the launch and the
placement carry over unchanged and the element type is the only variable:

| | bf16 (the faulting kernel) | fp16 sibling |
| --- | --- | --- |
| library | `TensileLibrary_BB_BB_…_Type_BB_HPA_…_gfx1201.co`, 1,297,415 B | `TensileLibrary_HH_HH_…_Type_HH_HPA_…_gfx1201.co`, 1,441,712 B |
| code object | 29,158,392 B, 427 kernels | 31,447,456 B, 500 kernels |
| `.kd` / entry point | `0x282380` / `0x56e400` | `0x2ee600` / `0x40f200` |
| metadata of that `.kd` | kernarg 144, group_seg 1638, 128 threads, sgpr 70, vgpr 164, 30 args | identical in all of those |

**Route 1 — the hand-loaded object faults in both element types, on both stacks.**

| stack | dtype | exit | faulting address | kernel named |
| --- | --- | --- | --- | --- |
| 7.14.60850 (env) | bf16 | 141 | A's guard page + 4096 | 855 chars, `BBS_BH` |
| 7.14.60850 (env) | fp16 | 141 | A's guard page + 4096 | 855 chars, `HHS_BH` |
| 7.15.26333 (system) | bf16 | 141 | A's guard page + 4096 | 855 chars, `BBS_BH` |
| 7.15.26333 (system) | fp16 | 141 | A's guard page + 4096 | 855 chars, `HHS_BH` |

All four runs print `grid=[1280, 1, 1]` (this program's 10 workgroups) and `group_seg_size=1638`. The
object is the 7.14 build in every row, so the 7.15 rows say what §12.10's table says for route 1: that
runtime loads and runs these bytes the same way, not that it would build the kernel the same way.

**Route 2 — the dtype also changes what hipBLASLt selects, and the fp16 selection overruns too.**

| stack | operands | exit | solution chosen | group_seg | fault |
| --- | --- | --- | --- | --- | --- |
| 7.14.60850 | bf16 | 141 | `BBS_BH … MT64x128x16 …`, 855 chars | 1638 | B's guard page, +0 |
| 7.14.60850 | fp16 | 141 | `HHS_BH … MT64x128x32 …`, 859 chars | 12416 | B's guard page, +0 |
| 7.14.60850 | f32 | 0 | rocBLAS's own fp32 kernels | — | completed without a fault |
| 7.15.26333 | bf16 | 141 | `BBS_BH … MT64x128x16 …`, 885 chars | 1638 | B's guard page, +0 |
| 7.15.26333 | fp16 | 141 | `HHS_BH … MT64x128x32 …`, 889 chars | 12416 | B's guard page, +0 |
| 7.15.26333 | f32 | 0 | rocBLAS's own fp32 kernels | — | completed without a fault |

The fp16 call is *not* answered with the sibling tuning. Its solution differs from the bf16 one in seven
of its 144 tokens — the dtype, the macro tile (`MT64x128x16` → `MT64x128x32`), `DTVA1`→`DTVA0`,
`EPS0`→`EPS1`, `LBSPPA0`→`LBSPPA1024`, `LPA0`→`LPA16`, `PLR0`→`PLR1` — i.e. a K tile twice as deep and a
12,416-byte LDS footprint against 1,638, on the same `grid=[2560, 1, 1]` and
`workgroup=[128, 1, 1]`. It faults at the **first byte of B's guard page**, exactly where the bf16
selection does, so the 158-byte overrun (§12.10) is the free-dimension edge of that operand, not the
bf16 loads and not the size of the bf16 tile's LDS.

The `f32` rows are the control that separates "the geometry" from "the 16-bit hipBLASLt path": the same
problem, the same strides, each operand sized 4 bytes per element and still end-aligned against its guard
page, completes cleanly in rocBLAS's own fp32 kernels on both stacks.

**The 7.15 names add five tokens, and the fp16 pair adds the same five.** Both route-2 names on the system
stack carry `BL1_BS1` after `ASEM1`, `SKWS0` after `SKFDPO0`, `UDFMAC0` after `USLMX0` and `WGMXCC1` at the
end, and nothing else: 855 → 885 for bf16, 859 → 889 for fp16, 144 equal tokens in each pair once the
insertions are removed. The selected tile and `group_seg_size` are unchanged across stacks
(`64x128x16` / 1638 and `64x128x32` / 12416), so the cross-stack difference is the library build, not the
selection.

**The fault page itself drifts, which corrects §12.9's and §12.10's "guard page + 4096".** Six repetitions
of route 1 with bf16 reported A's guard page + 4096 five times and A's guard page itself once; six
repetitions with fp16 reported + 4096 six times. Every repetition places its operands in a fresh VMM
reservation, so the addresses all differ and both pages are unmapped; the M-edge overrun spans 9,856 B ≈
2.4 pages, and ROCr names whichever unmapped page the warp retired against first. §12.9's transcript is one
of the + 4096 runs. The reproducible claim is therefore "past the end of A, within the first pages after
it" — a run that reports the guard page itself is the same fault one page earlier, not a different one.
(The sentence in §12.10's route table that fixes the offset to "A's guard page + 4096" for both stacks is
over-specific for the same reason; §12.9's and §12.10's fault transcripts are each accurate for the run
they show.)

**What this settles, and what it does not.**

- Settled: the overrun does not belong to the bf16 element type, to one tuning or to one library build. A
  differently tuned fp16 solution, out of a different library file, overruns the same operand at the same
  byte, and the fp32 path through rocBLAS does not overrun this shape at all — so the arithmetic that runs
  off the end is the edge-tile tiling, which the dtype only gets to re-tune.
- Not settled by this section: nothing here changes the trainer's own path, which is the bf16 call of
  §12.1 and §12.10. The fp16 runs also add no second fault classification: the read/write bit lives in the
  kernel log (`0x0080113B`), and none of these runs collected one — the same gap §12.9 and §12.10 note.

**Re-running it, now that the variants are gone.** Both runs need the two substitutions back:

- route 1: point `OBJECT_PATH` and `KERNEL` in `launch_fault_kernel.hip` at the fp16 pair, and extract
  that object with the extractor's generic flags —
  `python fixes/hip1/extract_kernel_codeobject.py --co <the Type_HH .co> --kernel '<the HHS_BH name>'`.
  Nothing else in the program changes (both element types are 2 bytes).
- route 2: pass `rocblas_datatype_f16_r` (`150`) and keep the 2-byte element size where the file now
  passes `rocblas_datatype_bf16_r`; the control is `rocblas_datatype_f32_r` (`151`) with 4-byte operands,
  the one run of the six that must not fault.
- the fp16 container is not in the repository: it was a copy of the environment's shipped file,
  `…/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201/TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co`
  (1,441,712 B, md5 `4f410302f705e14f6f47077380b7f28b`), which `--co` can read from that path directly.
  The system stack ships a file of the same name whose bytes differ, so a cross-stack comparison has to
  say which one it copied.

The per-run logs of the tables above are the `/tmp/axl-hip1/repro_standalone_<dtype>.log`,
`repro_rocblas_<dtype>.log` and `rep_<dtype>_<n>.log` files of the 2026-09-16 session, and the two sweep
transcripts the tables were read from.

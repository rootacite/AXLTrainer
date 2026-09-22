# HIP development on this machine

How to compile and run HIP code against the ROCm stack in the `axl`
conda environment (the environment `environment.yml` names). The trainer and
Ranko never build HIP code — this exists for probes, kernel experiments, and
anything that needs to touch the driver directly.

The probe sources and the fault-reproduction routes this document was written against are sealed in
`archive/` — 涉及负责任披露流程，暂不公开. What stays here is the knowledge: the stack layout, the
environment fixes, the VMM API notes and the pitfalls.

## 1. What the stack actually is

ROCm is installed as **PyPI wheels**, not as a distribution package:

| Component | Version | Location |
| --- | --- | --- |
| `rocm` metapackage | 10.0.0 | `pip` metadata only |
| `rocm-sdk-core` → `_rocm_sdk_core` | 10.0.0 | `$CONDA_PREFIX/lib/python3.14/site-packages/_rocm_sdk_core` |
| `rocm-sdk-libraries` → `_rocm_sdk_libraries` | 10.0.0 | `…/site-packages/_rocm_sdk_libraries` |
| `rocm-sdk-device-gfx1201` | 10.0.0 | kpack archives + gfx1201 Tensile DBs |
| HIP | 7.15.26333 | `_rocm_sdk_core/lib/libamdhip64.so.7` |
| clang | 23.0.0git (`8f497e09`) | `_rocm_sdk_core/lib/llvm/bin/clang-23` |
| torch | 2.13.0+rocm10.0.0 | `torch.version.hip == 7.15.26333` |

Paths inside `_rocm_sdk_core`:

| What | Path |
| --- | --- |
| HIP headers | `include/hip/` (`hip_runtime.h`, `hip_version.h`, …) + `include/hip/amd_detail/` |
| HSA headers | `include/hsa/` |
| Device bitcode | `lib/llvm/amdgcn/bitcode/` (incl. `oclc_isa_version_1201.bc`) |
| `hipcc` / `hipconfig` | `bin/` (also reachable as `$CONDA_PREFIX/bin/hipcc`, a Python trampoline that execs the real binary) |
| MIOpen / rocBLAS / hipBLASLt | `_rocm_sdk_libraries/lib/` |
| Compressed BLAS kernels | `_rocm_sdk_libraries/.kpack/*.kpack` |

There is a **second, unrelated ROCm** on this machine: `/opt/rocm/core`, owned by
the Arch package `rocm-gfx120x-bin 10.0.0-2`. It is the same ROCm release as the
wheels above — HIP **7.15.26333**, same clang revision — but a separate
installation (its `libamdhip64.so.7` is a different build: `4ac6ac42…` against the
wheel's `1a8bcd00…`). `~/.zshrc` exports `HIP_PATH`/`ROCM_PATH=/opt/rocm/core`,
`/etc/profile.d/rocm-bin.sh` appends `/opt/rocm/core/{bin,lib/llvm/bin}` to
`PATH`, and `/etc/ld.so.conf.d/rocm-bin.conf` puts `/opt/rocm/core/lib` on the
default library search path. Those three are why an unmodified environment mixes
the two stacks — and since both now report HIP 7.15.26333, the version a program
prints no longer tells you which library it loaded; `ldd` does. The trainer does
not care: torch dlopens the environment's libraries by absolute path at import
(`torch/_rocm_init.py` → `rocm_sdk.initialize_process`, `RTLD_GLOBAL`), so
`LD_LIBRARY_PATH` is irrelevant to it.

## 2. One-time setup

The setup helper that applied this is sealed with the rest of the probes (see the note at the top);
the four changes it made, spelled out so they can be redone by hand:

| # | Change | Why |
| --- | --- | --- |
| 1 | Unversioned `libfoo.so → libfoo.so.N` symlinks for every library in `_rocm_sdk_core/lib` and `_rocm_sdk_libraries/lib` (44 of them in the 10.0.0 wheels) | The linker looks for `libfoo.so`; the wheels ship only the versioned name, so `hipcc` fails with `cannot open …/lib/libamdhip64.so` |
| 2 | `_rocm_sdk_core/amdgcn → lib/llvm/amdgcn` | hipcc derives the device bitcode path from `ROCM_PATH`; in the system layout it is `<root>/amdgcn` |
| 3 | `_rocm_sdk_core/include/{thrust,rocprim,hipcub} → /opt/rocm/core/include/<same>` | These header-only libraries are in no ROCm wheel, and torch's headers include `<thrust/complex.h>`, so extension builds need them |
| 4 | `$CONDA_PREFIX/etc/conda/activate.d/axl_rocm_hip_toolchain.sh` + the matching `deactivate.d` script | Activating points `HIP_PATH`/`ROCM_PATH`/`HIP_CLANG_PATH` at the environment and prepends its `lib` directories to `LD_LIBRARY_PATH`; deactivating restores the previous values |

Nothing outside the conda environment is touched; `/opt/rocm` is left as it is.

## 3. Compile and run

```bash
conda activate axl
hipcc -O3 my.hip -o /tmp/my
/tmp/my
```

`hip_smoke.hip` prints the device, the HIP version the headers were compiled
against, the HIP version loaded at run time, and the result of one kernel. The
two versions agreeing tells you the toolchain and the runtime are the same
stack:

```
device        : AMD Radeon RX 9070 XT
arch          : gfx1201 (12.0, 32 CUs, warp 32)
HIP header    : 7.15.26333
HIP runtime   : 7.15.26333
kernel        : 0/1024 wrong
PASS
```

Without activation the same binary prints the same two versions, because both
stacks are HIP 7.15.26333 — so this output no longer says which library was
loaded; `ldd /tmp/hip_smoke | grep amdhip64` does. Built in an environment that
has not had §2's setup script applied, it resolves to
`/opt/rocm/core/lib/libamdhip64.so.7`: the wheels ship only the versioned
`libamdhip64.so.7`, so the loader falls back to the system search path. Run the
script once per environment before trusting a HIP measurement.

Useful variants:

```bash
# one architecture instead of every device the compiler can see
hipcc --offload-arch=gfx1201 -O3 ... -o /tmp/hip_smoke

# what the toolchain actually resolved: clang, include dir, device bitcode
hipcc -v -x hip -c my.hip -o /dev/null 2>&1 |
    grep -E 'idirafter|builtin-bitcode|resource-dir'

# which HIP runtime a binary will load, without running it
ldd /tmp/hip_smoke | grep amdhip64
```

A PyTorch extension build is the same toolchain, driven by torch:
`torch.utils.cpp_extension.load_inline(..., cuda_sources=<hip kernel>)` uses
`ROCM_PATH` (see §4), so inside an activated environment it compiles with the
environment's `hipcc`.

## 4. Who resolves what

| Variable | Consumed by | Must point at |
| --- | --- | --- |
| `HIP_PATH` | `hipcc`/`hipconfig`: include dir, host link of `libamdhip64.so` | `_rocm_sdk_core` |
| `ROCM_PATH` | `hipcc`: device bitcode path; `torch.utils.cpp_extension._find_rocm_home()` | `_rocm_sdk_core` |
| `HIP_CLANG_PATH` | `hipcc`: which `clang` frontend, and `clang-offload-bundler` next to it | `_rocm_sdk_core/lib/llvm/bin` |
| `LD_LIBRARY_PATH` | the dynamic loader: which `libamdhip64.so.7` your own binary gets | `_rocm_sdk_core/lib` and `_rocm_sdk_libraries/lib` first |
| `ROCM_HOME` | `_find_rocm_home()`, checked before `ROCM_PATH` | not set; set it only to override |
| `ROCM_SDK_PRELOAD_LIBRARIES` | `rocm_sdk.initialize_process()`: extra libraries to dlopen | not set |

If `ROCM_HOME`/`ROCM_PATH` are both unset, `_find_rocm_home()` falls back to
`which hipcc` and then to `/opt/rocm` — two more ways to silently leave the
environment.

## 5. Virtual memory management (`hipMemAddressReserve`, `hipMemMap`)

Both exist in both stacks: declared in `_rocm_sdk_core/include/hip/hip_runtime_api.h`
(doxygen group `Virtual`), exported by `libamdhip64.so` with symbol version `hip_5.1`
in both — the environment's 10.0.0 wheels and `/opt`'s 10.0.0 package are both HIP
7.15.26333. They are the driver-level VMM API — the
HIP spelling of `cuMemAddressReserve`/`cuMemMap` — and they decouple *where* a buffer
lives in the address space from *which physical allocation* backs it:

| Call | Role |
| --- | --- |
| `hipMemGetAllocationGranularity` | alignment a property set requires |
| `hipMemAddressReserve` | reserve a range of virtual addresses; no physical memory behind it |
| `hipMemCreate` | create a physical allocation described by `hipMemAllocationProp` (type, location, shareable handle type) |
| `hipMemMap` | map a handle into a reserved range, at a chosen offset |
| `hipMemSetAccess` / `hipMemGetAccess` | grant / query read-write access for a location on a mapped range |
| `hipMemRetainAllocationHandle` | handle of the allocation backing an address |
| `hipMemExportToShareableHandle` / `hipMemImportFromShareableHandle` | hand the allocation to another process as a POSIX fd |
| `hipMemUnmap`, `hipMemAddressFree`, `hipMemRelease` | teardown, in that order |

The point of the split is that a buffer can grow without moving: keep the reserved
address, map another physical block into the next subrange, and no pointer is
invalidated. Subranges can come from different allocations, and an exported handle lets
a second process map the same physical memory. (`hipMemMapArrayAsync` is documented as
not implemented.)

```bash
hipcc -O3 my_vmm_probe.hip -o /tmp/my_vmm_probe && /tmp/my_vmm_probe
```

That probe ran the full device cycle (granularity → create → reserve → map →
set access → kernel writes the mapping → unmap/free/release), a POSIX-fd export/import
round trip, and a host-location cycle. On gfx1201 under the 7.14.1 stack it printed
`PASS` (the probe has not been re-run on 10.0.0):

- granularity is 4096 B; `hipMemGetAccess` reports 3 (read-write);
  `hipMemRetainAllocationHandle` returns a handle equal to the created one.
- Export to a POSIX fd and import it back into a second reservation: a write through
  one mapping is visible through the other, i.e. the same physical allocation.
- Mapped ranges show up in `/proc/self/maps` as `---s` on
  `/dev/dri/renderD128`; a device location stays `---s` after `hipMemSetAccess`
  (the CPU is not granted access to device memory), a host location turns `rw-s`.

**Host locations behave differently, and one of the differences is a GPU fault:**

- `hipMemSetAccess` is not optional. Until it is called the mapping is `PROT_NONE`, so
  the first CPU touch is a **SIGSEGV in your process**, not an error return — the probe
  prints the `/proc/self/maps` line before and after to make that visible.
- `hipMemGetAccess` with a host location returns `hipErrorInvalidValue` (flags stay 0)
  even though `hipMemSetAccess` succeeded. Cosmetic, but do not use it as a check.
- `hipPointerGetAttributes` classifies the host mapping as **`hipMemoryTypeDevice`**, so
  passing that pointer to `hipMemcpy(..., hipMemcpyHostToDevice)` sends the copy engine
  after an address with no GPU page-table entry: `Memory access fault by GPU node-1 …
  faulting addr: <the mapping>`, kernel `__amd_rocclr_copyBuffer`. CPU access to the
  mapping is fine; going to the device has to be staged through a normal host buffer.

### torch already uses this API

`torch/lib/libc10_hip.so` imports `hipMemCreate`, `hipMemMap`, `hipMemAddressReserve`
and `hipMemSetAccess`, and the loader binds them to the environment's `libamdhip64.so`:

```bash
nm -DC "$CONDA_PREFIX"/lib/python3.14/site-packages/torch/lib/libc10_hip.so |
    grep -E 'hipMem(AddressReserve|Create|Map|SetAccess)'
LD_DEBUG=bindings python -c 'import torch' 2>&1 |
    grep -E "libc10_hip.*hipMem(AddressReserve|Create)"
```

The consumer is PyTorch's ROCm port of the `expandable_segments` allocator
(`HIPCachingAllocator.cpp` carries that name). An `LD_PRELOAD` interposer around those
four functions, on a 4096² matmul, counts:

| Allocator setting | VMM calls |
| --- | --- |
| default | none |
| `expandable_segments:True` | one `hipMemAddressReserve` of 19,251,855,360 B (17.9 GiB, i.e. about everything the 15.9 GiB card can hold) plus 12 `hipMemCreate`/`hipMemMap` pairs of 20 MiB and 4 `hipMemSetAccess` |

Any of `PYTORCH_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF` or `PYTORCH_HIP_ALLOC_CONF`
switches it on; the shipped `start_train.sh` sets none of them to
`expandable_segments`, so the trainer runs the plain native allocator.

## 6. Pitfalls

- **Do not put the system include dir in `CPATH`.** `CPATH` is searched *before*
  the `-idirafter` that hipcc uses for `HIP_PATH/include`, so `/opt/rocm`'s HIP
  headers would win over the environment's. Link the three header-only
  directories (§2 item 3) instead; that is precise and does not affect HIP.
- **Reinstalling the ROCm wheels wipes items 1 and 2** (they are files the wheel
  owns, and the wheels ship no such links). Re-run the setup script after any
  `pip install`/`uv` operation that touches `rocm-sdk-core`/`rocm-sdk-libraries`.
- **`thrust`/`rocprim`/`hipcub` come from `/opt/rocm`.** Removing the
  `rocm-gfx120x-bin` package breaks torch extension builds; plain HIP code is
  unaffected. cmake projects that need `find_package(hip)` also still rely on
  `/opt/rocm`, because no ROCm wheel ships cmake configs.
- **`rocm-sdk path --root|--bin|--cmake` fails** with "Could not load the
  `rocm[devel]` package". That package is the upstream home of everything §2 does
  by hand; install `rocm[devel]==10.0.0` if you want the full set (cmake configs,
  `$CONDA_PREFIX/bin/clang*` links) instead of this script.
- **`hipconfig` prints `llc: No such file or directory`** — cosmetic, hipcc drives
  clang directly. Its `HIP_PATH`/`ROCM_PATH` lines report its own resolution, not
  necessarily what a compile used; `hipcc -v` is the authority.
- **The `-x hip` warnings about ignored `hipError_t` are `[[nodiscard]]` noise**;
  wrap calls in a check macro.

## 7. Why a local toolchain change cannot fix that abort

The kernel that faults is a **precompiled Tensile solution**, not something the
local toolchain generates: `_rocm_sdk_libraries` ships no plaintext kernel names
anywhere (the loose gfx1201 Tensile `.dat`/`.hsaco` files contain only metadata
strings), the BLAS kernels live compressed in `.kpack/blas_lib_gfx1201.kpack`
(40 MB, `KPAK` v1 container, zstd, loaded by `librocm_kpack.so.0` via
`kpack_open`/`kpack_get_kernel`/`kpack_load_code_object`), and MIOpen wrote no
compiled artefacts to `~/.cache/miopen/3.6.0.8d1ae90e` during those runs.

So a different clang cannot change that kernel's codegen; only *solution
selection* can. Handles worth knowing:

- `ROCM_KPACK_DEBUG=1` — verbose kpack logging (which archive is opened/loaded).
- `ROCM_KPACK_PATH`, `ROCM_KPACK_PATH_PREFIX`, `ROCM_KPACK_DISABLE` — redirect or
  replace the kpack search path, e.g. to feed a different BLAS kernel set.
- The kpack C API (`kpack.h`, shipped at `/opt/rocm/core/include/rocm_kpack/`)
  can enumerate the binaries and dump a gfx1201 code object from the archive,
  which would name the owning library unambiguously.
- `MIOPEN_FIND_MODE` / `MIOPEN_*` and rocBLAS layer variables change which
  solution is picked, which is the axis the abort actually depends on.

## 8. Files

| Path | Role |
| --- | --- |
| sealed in `archive/` | The setup helper, the smoke test, the VMM probe and the two fault-reproduction routes (§7) — 涉及负责任披露流程，暂不公开 |
| `HIP.md` | This document |

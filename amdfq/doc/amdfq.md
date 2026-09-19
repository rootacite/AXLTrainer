# Where the ROCm memory-allocation APIs live in a training run

**What this is.** A process-level map of one real AxlTrainer training run on the stack `environment.yml`
pins (`torch 2.13.0+rocm10.0.0`, HIP `7.15.26333`, conda env `axl`): every ROCm shared object the
process loads, which of them *define* the memory-allocation APIs, which of them *call* them, and the
exact signatures. It is the layer the gfx1201 over-read is placed against
([`../../conclusions/bf16-kernel-overrun.md`](../../conclusions/bf16-kernel-overrun.md)).

**How it was measured.** `bash start_train.sh` under the current `config.toml` (`network_dim = 48`,
`train_batch_size = 2`, `gradient_checkpointing_te = true`, 640 images), with `/proc/<pid>/maps` copied
every 0.2 s for the process's whole life — 26 snapshots, one per distinct shared-object set. The run died
the way this configuration is known to: exit code `134` (SIGABRT) at **step 10**,
`HSA_STATUS_ERROR_MEMORY_FAULT`. Every symbol statement below is `readelf --dyn-syms` over the union of
those 26 snapshots, and was taken on this machine, not from documentation. Every symbol statement in
§1–§6 rests on it. Raw evidence (scratch, not versioned): `/tmp/so-analysis/` — `run/maps_*.txt`,
`run/train.stderr`, `syms.tsv`.

**How the call paths were measured.** The sections above answer "who *could* call these"; §7 answers "who
*does*", by hooking the runtime in a live trainer process: all 549 exported `hip*` functions of
`libamdhip64.so.7` plus 28 HSA entry points plus `dlsym`/`dlvsym`, installed at `dlopen` time, every call
counted. It confirms §4's branch by measurement, and it corrects two claims made earlier in this work
(§7.5). Raw evidence (scratch): `/tmp/frida-gates/` — `agent6.js`, `tracer6.py`,
`run6/snapshot_final.json`, `run6/maps_latest.txt`, `report6.txt`, `rebased.txt`.

**How to watch it yourself.** `amdfq/amdfq-tail-rs/` is the living `LD_PRELOAD` object that logs the three
allocation gates of §7.1 in an ordinary run, with nothing to attach to and no module list to distrust:
build, use and self-test in §9; §10 is what that object was built for and what the pad costs; §11 measures
that cost four rounds deep and sorts out which of the three accounts is telling the truth; §12 is the first
optimisation tried against it and why it is on the shelf; §13 is the tail guard — the route that keeps
`hipMalloc` allocating and buys the same protection for one granule per process instead of one per block —
and what it is worth. The numbers in §9–§13 were taken with the original C tree (`amdfq/amdfq-tail/`). A Chinese narrative of the whole investigation, from the first `fix1` guess about bucket
alignment to the tail guard that ships, is [`gfx1201-overread-story.md`](gfx1201-overread-story.md);
the four rounds of §11 in their original Chinese working notes are
[`live-data.md`](live-data.md). Both are narrative; §11 below stays the record of the
numbers.

**What the object does today** (route 2 in the tree). The three gates still log every call.
`hipMalloc` forwards the caller's size unchanged, then the tail guard of §13 maps one shared page behind
the block when the runtime backs nothing there; `hipFree` unmaps that page and re-guards the predecessor.
The `+16` byte pad of §10.1 is no longer the default path — it remains as the fallback for the ~0.2 % of
allocations whose end page cannot be taken. `AMDFQ_TAIL=0` turns the guard off so the original step-10
fault can be reproduced. The living object is `amdfq/amdfq-tail-rs/` (`bash amdfq/amdfq-tail-rs/run.sh`,
design constraints in `amdfq/amdfq-tail-rs/DESIGN.md`). The original C tree (`amdfq/amdfq-tail/`) is what
§9–§13 measured. Route 1's allocator, the shadow of §12.3, and the extra HIP hooks that diagnosed them
stay in the backup at `~/Desktop/amdfq/`; §9–§12 stay as the record of what was measured while each
earlier strategy was in place.

---

## 1. The objects the process loads

280 file-backed objects are mapped: 275 real shared objects, hipBLASLt's `Kernels.so-000-gfx1201.hsaco`,
and 4 `.co` Tensile code objects (276 of the 280 paths contain `.so`). 57 of the shared objects are
ROCm's, plus the 5 code objects — and **all of them come from the environment's wheels**: `/opt/rocm/core`
appears in the mappings **zero** times, although `HIP_PATH` and `ROCM_PATH` both point there.

| Source | Count | Contents |
| --- | --- | --- |
| `<env>/_rocm_sdk_core/lib` | 35 | `libamdhip64.so.7` (HIP runtime), `libhsa-runtime64.so.1` (ROCr), `libamd_comgr.so.3`, `libhiprtc.so.7`, `librocm-core.so.1`, `librocm_kpack.so.0`, `librocm_smi64.so.1`/`libamd_smi.so.27`, `librocprofiler-register.so.0`/`librocprofiler-sdk.so.1`, `libroctracer64.so.4`/`libroctx64.so.4`, `llvm/lib/libLLVM.so.23.0git`, `llvm/lib/libclang-cpp.so.23.0git`, `host-math/*` (SuiteSparse family + `librocm-openblas.so.0`), `rocm_sysdeps/*` (incl. `librocm_sysdeps_drm.so.2`, `librocm_sysdeps_drm_amdgpu.so.1`, `…numa.so.1`, `…elf.so.1`, `…zstd.so.1`) |
| `<env>/_rocm_sdk_libraries/lib` | 18 | `librocblas.so.5`, `libhipblaslt.so.1`, `libMIOpen.so.1`, `libMIOpenCKGroupedConv_gfx1201.so`, `libhipblas.so.3`, `libhipfft.so.0`, `libhiprand.so.1`, `libhipsolver.so.1`, `libhipsparse.so.4`, `libhipsparselt.so.0`, `librocfft.so.0`, `librocrand.so.1`, `librocsolver.so.0`, `librocsparse.so.1`, `librccl.so.1`, `liborigami.so.1`, `librocroller.so.1`, `libhipdnn_backend.so` |
| `<env>/torch/lib` (of 11 mapped, the HIP-specific 4) | 4 | `libtorch_hip.so`, `libc10_hip.so`, `libtorch_rocshmem.so`, `libaotriton_v2.so.0.13.0` |
| GPU code objects (file-backed maps) | 5 | `…/hipblaslt/library/gfx1201/Kernels.so-000-gfx1201.hsaco` plus 4 `TensileLibrary_*_gfx1201.co` |

`<env>` = `/home/acite/miniconda3/envs/axl/lib/python3.14/site-packages`. The process also holds
`/dev/kfd` and `/dev/dri/renderD128` open.

**Loading order matters for any future capture.** Everything above except the code objects is in place by
snapshot 02 (~0.24 s in, during `import torch`). `Kernels.so-000-gfx1201.hsaco` first appears at snapshot
20, and the Tensile library holding this run's faulting kernel only at snapshot 26 — 19 s before the
abort. A capture taken during the encoding phase would miss the latter.

## 2. Which `.so` defines the allocation APIs

Every symbol below has exactly **one** exporter in the process. Scanning all 280 mapped objects:

| API | Defined in |
| --- | --- |
| `hipMalloc`, `hipFree`, `hipMallocManaged`, `hipMallocAsync`, `hipMallocFromPoolAsync`, `hipExtMallocWithFlags`, `hipMemPoolCreate`, `hipGetProcAddress` | `_rocm_sdk_core/lib/libamdhip64.so.7` |
| `hipMemAddressReserve`, `hipMemAddressFree`, `hipMemCreate`, `hipMemMap`, `hipMemUnmap`, `hipMemRelease`, `hipMemRetainAllocationHandle`, `hipMemSetAccess`, `hipMemGetAccess`, `hipMemGetAllocationGranularity` | the same `_rocm_sdk_core/lib/libamdhip64.so.7` |
| `hsa_amd_memory_pool_allocate`, `hsa_amd_vmem_address_reserve`, `hsa_amd_vmem_address_free`, `hsa_amd_vmem_handle_create`, `hsa_amd_vmem_map`, `hsa_amd_vmem_set_access` | `_rocm_sdk_core/lib/libhsa-runtime64.so.1` |

No math library, no `torch/lib` object, and no MIOpen/RCCL object exports any of them — those are
consumers only. So the answer to "where does `hipMalloc` live in this process" is a single file, and the
file is the wheel's, not `/opt`'s: `libamdhip64.so.7` there hashes
`sha256 1a8bcd00d05cce36bebcf4acf30289f36f6562ec532e9c618c42e9bd2d90ed05`, while `/opt/rocm/core`'s
`libamdhip64.so.7` (the same HIP version, `7.15.26333`) hashes
`4ac6ac420d354e279b68c18bc426ab336ccf6d72834d2ea4e915d93cf269c646` — see
[`HIP.md`](HIP.md) §1. `libamdhip64.so.7` itself needs `libhsa-runtime64.so.1`, `libamd_comgr.so.3`,
`librocm_kpack.so.0` and `librocprofiler-register.so.0`.

## 3. Who calls them

From the same scan (`UND` symbols per mapped object). This is a static statement about which objects
*declare* a dependency on these APIs; which of them actually call anything in this configuration is
measured in §7:

| Caller | Allocation APIs it references |
| --- | --- |
| `torch/lib/libc10_hip.so` (the caching allocator's HIP backend) | `hipMalloc`, `hipFree`, `hipMallocManaged`, `hipMallocAsync`, and the whole VMM set: `hipMemAddressReserve`/`hipMemAddressFree`, `hipMemCreate`, `hipMemMap`/`hipMemUnmap`, `hipMemRelease`, `hipMemSetAccess` |
| `torch/lib/libtorch_hip.so` | `hipMalloc`, `hipFree`, `hipMemAddressReserve`, `hipMemCreate`, `hipMemMap`, `hipMemUnmap`, `hipMemRelease`, `hipMemSetAccess`, `hipMemGetAllocationGranularity` |
| `torch/lib/libtorch_rocshmem.so` (rocSHMEM) | `hipMalloc`, `hipExtMallocWithFlags`, the VMM set, `hipMemGetAccess`, `hipMemRetainAllocationHandle` |
| `_rocm_sdk_libraries/lib/librccl.so.1` | `hipMalloc`, `hipExtMallocWithFlags`, `hipMallocFromPoolAsync`, `hipMemPoolCreate`, the VMM set |
| rocBLAS, hipBLASLt, MIOpen (+ its CK gfx1201 module), hipBLAS, hipFFT, hipSOLVER, hipSPARSE, hipSPARSELt, rocFFT, rocRAND, rocSOLVER, rocSPARSE | `hipMalloc`/`hipFree` for workspaces; `hipMallocAsync` (rocBLAS, hipSPARSE, rocSPARSE), `hipMallocManaged` (hipFFT, rocFFT) |
| `_rocm_sdk_core/lib/libamdhip64.so.7` | `hsa_amd_memory_pool_allocate` and the `hsa_amd_vmem_*` set — the runtime is the **only** consumer of the HSA layer |

## 4. The chain, and which branch this configuration takes

```
torch (Python) → libc10_hip.so  ─┬─ hipMalloc            → libamdhip64.so.7 ─→ hsa_amd_memory_pool_allocate ─┐
                                 └─ hipMemAddressReserve  → libamdhip64.so.7 ─→ hsa_amd_vmem_address_reserve ─┤ libhsa-runtime64.so.1
                                    hipMemCreate         →      "           ─→ hsa_amd_vmem_handle_create   ─┤        ↓
                                    hipMemMap            →      "           ─→ hsa_amd_vmem_map            ─┤  /dev/kfd ioctl
                                    hipMemSetAccess      →      "           ─→ hsa_amd_vmem_set_access     ─┘  (amdgpu KFD)
```

`start_train.sh` sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8`.
That is the **native caching allocator**, so the branch taken at runtime is `hipMalloc` →
`hsa_amd_memory_pool_allocate`. The VMM entry points are linked into `libc10_hip.so`/`libtorch_hip.so`
because PyTorch supports `expandable_segments:True` (and a stream-ordered backend via `hipMallocAsync`
and `hipMemPoolCreate`), but neither is enabled here. This started as code-path reasoning from the
configuration plus the symbol references above; **§7 turned it into a measurement** — the VMM, pool and
async entry points are hooked, and every one of them counts zero calls.

Two consequences worth keeping in view. First, what `hipMalloc` returns is one pool allocation mapped
into the process's address space, and the caching allocator then hands out sub-ranges of it: a read past
an operand's end usually lands in *another live tensor's* pages, which is why the over-read only kills
when the allocator happens to place that operand against a hole, and why the outcome is a lottery rather
than a deterministic crash. Second, the mapping machinery that decides whether such a hole exists is the
same one the `HSA_SVM_GUARD_PAGES=0` measure targets
([`conclusions/bf16-overrun-mitigations.md`](../../conclusions/bf16-overrun-mitigations.md) §2.1).

## 5. Signatures and meaning

Header of record: `<env>/_rocm_sdk_core/include/hip/hip_runtime_api.h` (VMM group, "Virtual Memory
Management", lines 9963–10161) and `<env>/_rocm_sdk_core/include/hsa/hsa_ext_amd.h`.

### 5.1 The classic path (what this configuration runs on)

```c
hipError_t hipMalloc(void** ptr, size_t size);
```
Allocates `size` bytes on the default accelerator and returns a plain pointer. `size == 0` allocates
nothing, writes `nullptr` and returns `hipSuccess`. Returns `hipErrorOutOfMemory` /
`hipErrorInvalidValue` (bad context, `ptr == NULL`).

```c
hipError_t hipFree(void* ptr);                     /* implicit device synchronize */
hipError_t hipMallocManaged(void** dev_ptr, size_t size,
                            unsigned int flags /* = hipMemAttachGlobal */);
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
```
`hipExtMallocWithFlags` selects the memory type: `hipDeviceMallocDefault`, `…Finegrained`,
`…Uncached`, or `hipMallocSignalMemory`; any other flag is `hipErrorInvalidValue`. This is the closest
HIP gets to asking for a mapping whose neighbour behaviour differs — fine-grained (SVM) memory is what
RCCL and rocSHMEM ask for.

```c
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props);
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream);
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream);
```
The stream-ordered path: allocation and free are ordered against a stream instead of the device, with a
pool object owning the caching. PyTorch reaches it only through the `cudaMallocAsync`/`hipMallocAsync`
allocator backend.

### 5.2 The VMM family (a reservation is not an allocation)

```c
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option);
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags);
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags);
hipError_t hipMemMap(void* ptr, size_t size, size_t offset,
                     hipMemGenericAllocationHandle_t handle, unsigned long long flags);
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr);
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr);
hipError_t hipMemUnmap(void* ptr, size_t size);
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
hipError_t hipMemAddressFree(void* devPtr, size_t size);
```

The five-step lifetime, in the order the docs require:

1. `hipMemGetAllocationGranularity` — the minimum/recommended granularity for a given `prop`; sizes and
   alignments must be rounded to it.
2. `hipMemAddressReserve` — reserves a **virtual address range only**, no physical memory. `alignment`
   and a requested `addr` let the allocator pick where; `flags` must be 0.
3. `hipMemCreate` — creates the physical allocation and returns a handle. `prop.allocType` must be
   `hipMemAllocationTypePinned` or `…Uncached`, `prop.location.type` must be `…Device` or `…Host`;
   anything else is `hipErrorInvalidValue`.
4. `hipMemMap` — maps that handle into the reserved range (`offset` currently must be 0).
   `hipMemSetAccess` — grants the listed locations access; a mapped range is not usable before this.
5. Teardown is `hipMemUnmap` → `hipMemRelease` → `hipMemAddressFree`, in that order.

`hipMemRetainAllocationHandle` runs the lookup backwards (address → handle), which is what a consumer
needs to export or re-map an allocation it did not create — RCCL and rocSHMEM both reference it.

The meaning for this defect: VMM is the only path in which the *virtual* placement of an operand is
chosen deliberately (fixed reservations, chosen granularity, explicit unmapping), so it is the path in
which a neighbouring hole is a design decision rather than allocator luck. `expandable_segments` uses it
for exactly that reason — it is the third layout `fixes/fix2/repro_real.py`'s integrity grid tests
(`"integrity: seed 1145141919, expandable segments"` → `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
It is not active in this configuration.

### 5.3 The HSA layer underneath

```c
hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                          uint32_t flags, void** ptr);
hsa_status_t hsa_amd_vmem_address_reserve(void** va, size_t size, uint64_t address, uint64_t flags);
hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
                                        hsa_amd_memory_type_t type, uint64_t flags,
                                        hsa_amd_vmem_alloc_handle_t* memory_handle);
hsa_status_t hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                              hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags);
hsa_status_t hsa_amd_vmem_set_access(void* va, size_t size,
                                     const hsa_amd_memory_access_desc_t* desc, size_t desc_cnt);
hsa_status_t hsa_amd_vmem_address_free(void* va, size_t size);
```

This is where the two HIP families land; `libhsa-runtime64.so.1` is the last shared object before the
kernel, and it reaches the hardware through KFD (`/dev/kfd`, amdgpu). Anything that changes what a
stray read meets — pool placement, page mapping, access flags — is decided here, not in `libamdhip64`.

### 5.4 A note on the manual

`/home/acite/Downloads/Memory management — HIP 7.15.0 Documentation.pdf` is HIP's **"Memory management"**
page (62 pages: `hipMalloc`/`hipFree`, host and pitched variants, memcpy/memset, pointer attributes,
arrays, `hipMemGetInfo`). It does **not** document `hipMemAddressReserve`, `hipMemCreate`, `hipMemMap`,
`hipMemSetAccess` or the rest of the VMM group — those live on a separate HIP docs page. The signatures
quoted above therefore come from the runtime headers the wheels ship, which are the same version as the
libraries being loaded.

## 6. What this run died of

The capture caught the fault, and the object set pins its provenance:

- Exit `134` (SIGABRT), `state.json` frozen at `status=training / step=10`, `HSA_STATUS_ERROR_MEMORY_FAULT`,
  faulting address `0x7f0d18e00000`, dispatched grid `[2560,1,1]`, workgroup `[128,1,1]`.
- The faulting kernel is `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…_ISA1201_…`.
  That full name appears **exactly** among the entries declared in
  `_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.dat.zlib`
  (27 names share the `…MT64x128x16` prefix) — and the matching `.co` is one of the four the process had
  mapped. rocBLAS's own gfx1201 libraries under `…/rocblas/library/` contain no such kernel.
- The `.co` files are not ELF: they start with the magic `CCOB` and hold no plaintext kernel names; the
  Tensile metadata that names the kernels is the compressed `.dat.zlib` beside them. That is why a
  `strings`/`readelf` search over `.co` finds nothing and the library had to be identified through its
  metadata.
- The Tensile library carrying the kernel was mapped at snapshot 26 (02:48:49), at the
  encoding→training boundary, and no further code object appeared before the abort 19 s later — so the
  fatal dispatch used a library loaded earlier in the run, not one loaded for it. The reason the timing
  matters at all: of the 144 `.co` libraries in that gfx1201 directory, the process ever mapped **4**,
  which is hipBLASLt selecting code objects per shape rather than loading the set. Which shape first
  pulled this one in cannot be told from 0.2 s snapshots.

## 7. Measured: which entry points are used, and who enters them

**Method.** A tracer spawns `bash start_train.sh`, finds the trainer PID within ~3 ms (`start_train.sh`
`exec`s the interpreter, so the shell PID *is* the trainer PID) and attaches Frida 17.7.3. Hooks are
installed from `Process.attachModuleObserver`, i.e. at the moment each library is `dlopen`ed; the process's
first HIP call is at t≈1.6 s, so no library and no call was missed. Three mechanisms carry the result:

- **Gate stack.** Every hooked `hip*` function pushes its name on a per-thread stack in `onEnter` and pops
  it in `onLeave`. When an `hsa_*` hook fires, the name on top of that thread's stack is recorded as that
  call's *gate*; `<no-hip-gate>` means it fired with no hooked `hip*` frame above it.
- **Exhaustive gate set.** This is what decides whether the answer means anything. Hooking only the
  memory family (57 symbols) left **11667 of 14607** `hsa_agent_get_info` calls looking ungated, which
  reads as "torch reaches HSA directly". Hooking *every function `libamdhip64.so.7` exports whose name
  starts with `hip`* — 549 of them; 579 hooks in total with the 28 HSA symbols and `dlsym`/`dlvsym`; 0
  missing, 0 errors — drops the ungated count to **0**. A partial hook set manufactures a finding.
- **Full census, sampled stacks.** Every call is counted; a backtrace is captured once per distinct call
  site. Attribution is *not* taken from Frida's module list — see §7.4.

In one run of this configuration (the step-10 fault reproduced in it, as in every run of it) the process
calls 31 of the 549 exported HIP entry points and 11 HSA functions.

### 7.1 The gates that allocate

| Gate | Calls | Bytes | Runtime caller (absolute address → `/proc/maps` → symbol) |
| --- | --- | --- | --- |
| `hipMalloc` | 4801 | 66.8 GiB | `libc10_hip.so+0x41efa` `CUDACachingAllocator::Native::DeviceCachingAllocator::alloc_block` **4793**; `libhipblaslt.so.1` `hipblasLtCreate+0x50` 6; `librocblas.so.5` `_rocblas_handle::_rocblas_handle()` 2 |
| `hipFree` | 3693 | — | `libc10_hip.so+0x26918` `…DeviceCachingAllocator::release_block` ← `release_blocks` ← `release_cached_blocks` ← `NativeCachingAllocator::emptyCache` ← `THCPModule_emptyCache` (`libtorch_python.so`) ← CPython |
| `hipHostMalloc` | 84 | 0.49 GiB | `libtorch_hip.so+0xdab94d` `at::cuda::(anonymous namespace)::CUDACachingHostAllocatorImpl::allocate_host_memory_slowpath` — torch's host (pinned) allocator, on a thread other than the trainer's |
| `hipMemcpyWithStream` | 4408 | — | `libtorch_hip.so+0x1b78194` `at::native::copy_kernel_cuda` |
| `hipMemcpyAsync` | 1143 | — | `libtorch_hip.so+0x1b77b2c` `at::native::copy_device_to_device` 1121, `copy_kernel_cuda` 22 |
| `hipGetDeviceCount` | 5 | — | `libc10_hip.so` `c10::cuda::device_count()`, `device_count_ensure_non_zero()`, `CUDAKernelLaunchRegistry::CUDAKernelLaunchRegistry()`; `libhipblaslt.so.1` 2 |

The first three are the allocator gates proper: torch's device caching allocator, torch's host (pinned)
allocator, and the math libraries' workspaces. `hipGetDeviceCount` allocates nothing itself but is the
boot gate — it is the single entry through which HSA comes up at all (§7.2).

`hipMemcpyWithStream` and `hipMemcpyAsync` are the half-hidden pair. They come from
`at::native::copy_kernel_cuda` / `copy_device_to_device`, and *inside* HIP they pin host buffers and
classify pointers: 2693 `hsa_amd_memory_lock_to_pool` and 4532 `hsa_amd_pointer_info` calls happen under
them. At this layer nothing in the torch configuration separates "copy" from "memory management": a copy
out of a buffer that is not already pinned takes a pool path inside the runtime.

The remaining called entry points are queries, launches and code-object loads. They allocate nothing from
the HIP/HSA memory API, but they are what drives the HSA *query* traffic, so they are listed for
completeness:

| Gate | Calls | Runtime caller |
| --- | --- | --- |
| `hipGetDevice` | 6678924 | `libc10_hip.so` `c10::cuda::GetDevice` 2.94 M, `SetDevice` 1.47 M; MIOpen CK 1.58 M; hipBLASLt `rocblaslt_internal_get_arch_name` 0.38 M |
| `hipGetDevicePropertiesR0600` | 1970222 | MIOpen CK 1.58 M; hipBLASLt 0.38 M; `TensileLite::hip::GetDevice` 2138; MIOpen 1399 |
| `hipGetLastError` | 384804 | `at::native::copy_device_to_device` 128605 and `gpu_kernel_impl_nocast<…>` 67519/59473/30340 |
| `hipLaunchKernel` | 250299 | `libtorch_hip.so` `gpu_kernel_impl_nocast<…>` (bf16 copy, `CUDAFunctor_add<BFloat16>`, `AUnaryFunctor<…MulFunctor<float>>`) and `copy_device_to_device` |
| `hipDeviceGetAttribute` | 177060 | hipBLASLt `TensileLite::ContractionSolution::singleCallArgs` 71621, MIOpen CK; 587 sites |
| `hipExtModuleLaunchKernel` | 99534 | hipBLASLt `TensileLite::hip::SolutionAdapter::launchKernel` 95814; MIOpen 3720 |
| `hipDevicePrimaryCtxGetState` | 50705 | `at::cuda::detail::_hasPrimaryContext` |
| `hipStreamGetCaptureInfo` | 46052 | `at::CUDAGeneratorImpl::philox_cuda_state` 16024, `CUDAGeneratorState::increase` 16024 |
| `hipStreamGetDevice` | 20851 | aotriton `getGpuFromStream` 19934, `getMultiProcessorCount` 917 |
| `hipStreamIsCapturing` | 9255 | `cudaMallocMaybeCapturing` 4793, `at::native::copy_kernel_cuda` 4430 |
| `hipModuleLaunchKernel` | 9050 | `libaotriton_v2.so.0.13.0` (1 site) |
| `hipSetDevice` | 3725 | `libMIOpen.so.1` `miopen::set_device` (1 site) |
| `hipRuntimeGetVersion` | 2138 | `TensileLite::hip::GetDevice` |
| `hipModuleGetFunction` | 1156 | MIOpen 813, hipBLASLt `TensileLite::hip::SolutionAdapter::getKernel` 327 |
| `hipMemsetAsync` | 140 | MIOpen CK grouped conv 133; `at::native::gpu_reduce_kernel` 7 |
| `hipPointerGetAttributes` | 102 | `at::cuda::detail::CUDAHooks::isPinnedPtr` (1 site) |
| `hipModuleLoadDataEx` | 16 | aotriton |
| `hipStreamSynchronize` | 10 | `libtorch_hip.so` |
| `hipMemset` | 6 | hipBLASLt |
| `hipModuleLoadData` | 5 | MIOpen |
| `hipModuleLoad` | 5 | hipBLASLt |
| `hipMemGetInfo` | 2 | MIOpen |
| `hipInit` | 2 | MIOpen |
| `hipDeviceGetStreamPriorityRange` | 1 | `libc10_hip.so` |

**Dead entry points.** Hooked, enumerated, and never called once in the whole run: the entire VMM family
(`hipMemAddressReserve`, `hipMemAddressFree`, `hipMemCreate`, `hipMemMap`, `hipMemUnmap`, `hipMemSetAccess`,
`hipMemGetAccess`, `hipMemRelease`, `hipMemRetainAllocationHandle`, `hipMemGetAllocationGranularity`), the
memory-pool family (`hipMemPoolCreate`/`…Destroy`/`…TrimTo`/`…SetAttribute`/`…GetAttribute`/`…ExportPointer`/
`…ImportPointer`, `hipMemGetDefaultMemPool`, `hipMemGetMemPool`, `hipMemSetMemPool`), `hipMallocAsync`,
`hipMallocFromPoolAsync`, `hipMallocManaged`, `hipMallocPitch`, `hipMalloc3D`/`3DArray`/`Array`/
`MipmappedArray`, `hipExtMallocWithFlags`, `hipHostRegister`, `hipHostUnregister`, `hipMemAdvise`,
`hipMemPrefetchAsync` and the batch prefetch/discard pair, `hipExternalMemory*`, and `hipGetProcAddress`.
In this configuration the allocator is `hipMalloc`/`hipFree` and nothing else.

### 7.2 Every HSA call happens from inside a HIP call

| `hsa_*` | Calls | Gate(s) |
| --- | --- | --- |
| `hsa_agent_get_info` | 14607 | `hipLaunchKernel` 11632 (79.6 %), `hipModuleLoad` 2794 (19.1 %), `hipModuleLoadData` 91, `hipGetDeviceCount` 53, `hipMemcpyWithStream` 19, `hipModuleLoadDataEx` 16, `hipMemGetInfo` 2 |
| `hsa_amd_memory_pool_allocate` | 4889 | `hipMalloc` **4801** (98.2 %), `hipHostMalloc` 84, `hipGetDeviceCount` 2, `hipMemcpyWithStream` 2 |
| `hsa_amd_pointer_info` | 4532 | `hipMemcpyWithStream` 4408, `hipPointerGetAttributes` 102, `hipMemcpyAsync` 22 |
| `hsa_amd_memory_pool_free` | 3693 | `hipFree` (100 %) |
| `hsa_amd_memory_lock_to_pool` | 2693 | `hipMemcpyWithStream` 2671, `hipMemcpyAsync` 22 |
| `hsa_amd_memory_pool_get_info` | 16 | `hipGetDeviceCount` (100 %) |
| `hsa_amd_agent_memory_pool_get_info` | 7 | `hipGetDeviceCount` (100 %) |
| `hsa_system_get_info` | 6 | `hipGetDeviceCount` 5, `hipMemcpyWithStream` 1 |
| `hsa_amd_agent_iterate_memory_pools` | 3 | `hipGetDeviceCount` (100 %) |
| `hsa_init` / `hsa_iterate_agents` | 1 / 1 | `hipGetDeviceCount` |

For all eleven functions the gate-stack depth histogram is `{1: n}`: one HIP frame on the stack, never zero,
never deeper. The pairs balance one for one where they must — `hipMalloc` 4801 ↔ 4801 `pool_allocate`,
`hipFree` 3693 ↔ 3693 `pool_free`, `hipHostMalloc` 84 ↔ 84 allocations from the host pool.

**Every immediate caller is inside the runtime.** The callers are `libamdhip64.so.7` offsets; the file has
no `.symtab`, so `.eh_frame` ranges identify the internal function without naming it:

| `hsa_*` | Caller | `.eh_frame` function | Calls |
| --- | --- | --- | --- |
| `hsa_amd_memory_pool_allocate` | `+0x5f3b78` | `0x5f3aa0..0x5f3cf7` (device pool; under `hipMalloc`) | 4801 |
| `hsa_amd_memory_pool_allocate` | `+0x5f2ca5`, `+0x5f2c74` | `0x5f2c10..0x5f2e3b` (host pool; under `hipHostMalloc`) | 87, 1 |
| `hsa_amd_memory_pool_free` | `+0x5f310f` | `0x5f3100..0x5f31c8` | 3693 |
| `hsa_amd_pointer_info` | `+0x5f95bd` | `0x5f9570..0x5f977d` | 4532 |
| `hsa_amd_memory_lock_to_pool` | `+0x5f3001` | `0x5f2fa0..0x5f30ee` | 2693 |
| `hsa_agent_get_info` | `+0x601bf4` | `0x601710..0x601d5c` | 14551 |
| init-time queries (`pool_get_info`, `agent_memory_pool_get_info`, `system_get_info`, `iterate_agents`, `hsa_init`) | `+0x5ed0d8`…`+0x5f1960` | `0x5ed0b0..0x5ed377`, `0x5ed5d0..0x5ee9fb`, `0x5ef780..0x5f1614`, `0x5f18a0..0x5f1a91` | ≤ 4 each |

One exception, and it is the runtime's own: a single `hsa_amd_agent_iterate_memory_pools` call comes from
`libhsa-runtime64.so.1+0x1c9832` `rocr::image::ImageRuntime::CreateImageManager` — `libhsa-runtime64` is
the only other object in the process that ever calls into the HSA memory API.

`hsa_init`'s stack is the entire boot sequence in one frame chain:

```
[0] libamdhip64.so.7+0x5ed5ef
[1] libc.so.6+0x9d879            pthread_once
[2] libamdhip64.so.7+0x15273e
[3] libamdhip64.so.7+0x49ab3b    hipGetDeviceCount
[4] libc10_hip.so / libtorch_hip.so   (c10::cuda::device_count() and friends, §7.1)
```

So: `hipGetDeviceCount` → `pthread_once` → `hsa_init` → agent and memory-pool enumeration, all inside the
one HIP call. Nothing above it in the stack is torch code calling HSA; torch's stack frame is two levels
*below* the HIP entry point, where it belongs.

**Corroboration from the static side.** §2 and §3 say who *may* reach the HSA layer; they agree with the
measurement: the HSA allocation symbols have exactly one referent in the process (`libamdhip64.so.7`),
`libc10_hip.so` and `libtorch_hip.so` reference no `hsa_` symbol at all, and `dlsym`/`dlvsym` were called
376 times with no `hip*`/`hsa*` name among the requests — so no module resolves an HSA entry point by
string either. The one torch-side object that does reference `hsa_amd_*` — `libtorch_rocshmem.so`
(`hsa_amd_memory_lock_to_pool`, `hsa_amd_pointer_info`, `hsa_amd_memory_pool_get_info`,
`hsa_amd_agent_iterate_memory_pools`) — is mapped but never initialized in this configuration, and
contributed zero calls. It is the one place in the tree where a *direct* entry into the HSA memory API
from non-HIP code exists, and it is dormant unless rocSHMEM is brought up.

The plain statement of the result: **`hsa_amd_*` is not an entry point for this training process, it is
runtime-internal.** Torch enters ROCm memory management through `hip*` and only through `hip*`.

### 7.3 The fault address lies in allocator-owned space

Every successful `hipMalloc`/`hipHostMalloc` return value was recorded. In the analysed run the faulting
address `0x7f0b06600000` is exactly the *start* address a `hipMalloc` returned (twice: a 44 MiB and a
42 MiB block) and lies inside several other blocks' ranges. The ledger is a flat list over the whole run
and the allocator recycles freed ranges, so containment is not proof that the block was live when the
fault happened — only that the address the over-read walked onto belongs to the allocator's arena rather
than to a library mapping or a random hole. That is the shape §4 predicts.

### 7.4 Why every attribution went through `/proc/maps`

Frida's `Process.findModuleByAddress` is not usable for this question in this process. The same `hipMalloc`
call chain came back as three different modules in four runs:

| Run | Reported | Why it is wrong |
| --- | --- | --- |
| 1 | `libc10_hip.so+0x41efa` | correct — `DeviceCachingAllocator::alloc_block`, symbol and `.eh_frame` agree |
| 2 | `libtorch_hip.so+0xfe64efa` | 0xfe64efa is past the end of a 156 MB file, and neither a symbol nor an `.eh_frame` range covers it |
| 3 | `librocsolver.so.0+0x1d16efa` | past the end of a 28.5 MB file, same |

The mechanism: these libraries are mapped twice — once as a whole-file read-only mapping and once as the
real ELF segments. For `libamdhip64.so.7` the extra mapping is 27.6 MB and its start sits exactly
`0x2200000` below the real base, which is enough to move both Frida's notion of a module's address range
and any naive `min(start − file_offset)` base computation. Every caller quoted in §7.1 and §7.2 is
therefore produced as: absolute address as recorded by the hook → the file `/proc/<pid>/maps` says backs
that address → that file's real base found by matching the ELF's first `LOAD` segment in *both* file
offset and rounded size → symbol. The `libamdhip64` virtual offsets were cross-checked two independent
ways (maps-derived base and Frida's base agree once the extra mapping is excluded).

### 7.5 What this supersedes

- §4 described the VMM/pool/async branches as "not enabled here" from configuration plus symbol
  references. It is now measured: they are hooked, and they count zero.
- Intermediate reports made while establishing this claimed "only 8 APIs are called", "`dlsym` is never
  called" and "11667 `hsa_agent_get_info` calls have no HIP gate". All three were artifacts of narrower
  hook sets — the first two from a build that hooked 73 symbols over two modules (`libc.so.6` was not
  among them, so a `dlsym` count could not exist), the third from the 57-symbol memory-only gate list.
  With the exhaustive set the same quantities read 31 entry points, 376 `dlsym` calls, and 0 ungated HSA
  calls.
- The counts reproduce across independent runs with different hook sets: `hsa_agent_get_info` 14607 in
  three of them, `dlsym` 374–376, `hsa_amd_memory_pool_allocate`/`_free` 4879–5037 / 3693. The spread is
  step timing, not behaviour.

## 8. Limits of this note

- Which allocator branch is *taken* (§4) is no longer inferred: §7 hooks both families and the VMM/pool/
  async entry points count zero calls. What remains inference is the *reverse* direction — a configuration
  that did enable `expandable_segments` or the async backend would need its own run; the hooks, not the
  reasoning, would answer it.
- §7's call counts are a complete census; its *symbolization* is not uniformly deep. `libc10_hip.so`,
  `libtorch_python.so` and `python3.14` ship symbol tables, so those frames name real functions.
  `libamdhip64.so.7` ships no `.symtab`, so its HSA call sites are anonymous internal functions located by
  `.eh_frame` range only; `libtorch_hip.so` and the MIOpen CK object are stripped, so their deeper frames
  are nearest-export approximations. Every *file* attribution, by contrast, is a `/proc/maps` lookup and is
  exact — which is the part §7.2's conclusion rests on.
- Frida's module list cannot be trusted in this process (§7.4): the same frame was reported against three
  different libraries, two of them at offsets past the end of the file. Any future capture must resolve
  addresses through `/proc/<pid>/maps` and recompute each library's real base from its first `LOAD`
  segment. The rule is written down because ignoring it produced two wrong attributions in this work
  before they were caught.
- The scan covers the objects mapped during one run of this configuration. Other configurations would
  change the code objects (`expandable_segments` would add VMM call traffic) but not the defining
  libraries in §2.
- The dead run's leftovers are the same phenomenon `fixes/fix2/README.md` records as "orphaned DataLoader
  workers", one level up: the run uses a DataLoader **forkserver**, so what survives the SIGABRT is the
  forkserver plus its 20 `pt_data_worker` children per killed generation — no `atexit` runs on an abort,
  and the workers are forked from the server, not from the trainer. Three dead generations had left
  three such groups (~11 GB RSS and one `/dev/kfd` handle each) on this box before they were killed.

## 9. Watching the three gates from a preload object

`amdfq/amdfq-tail-rs/` builds `libamdfq_tail_rs.so`: an `LD_PRELOAD` object that logs the three gates §7.1
says are the whole allocation path of this configuration — `hipMalloc`, `hipFree`, `hipHostMalloc` — one
line per call. It is the allocation half of §7 reduced to a run you do not have to instrument: no Frida, no
module list to distrust (§7.4), no process to attach to before the first HIP call. It is also not
read-only: after each `hipMalloc` the tail guard of §13 maps one shared page behind the block when the
runtime backs nothing there. The `+16` pad that *The pad* below and §10 measure is what the object used to
do on every request, and is now the fallback for the ~0.2 % of allocations whose end page cannot be taken.

The original C tree (`amdfq/amdfq-tail/`) is what this section's sample lines and §10–§13's numbers were
taken with. The living object follows the same route under the constraints in
`amdfq/amdfq-tail-rs/DESIGN.md` (no lock across a runtime call, `log` crate to stderr, one registry).

| Path | Role |
| --- | --- |
| `amdfq/amdfq-tail-rs/src/hooks.rs` | The interposers: log the call, forward it, then the tail guard on `hipMalloc`/`hipFree` |
| `amdfq/amdfq-tail-rs/src/tail.rs` | The tail guard (§13): one shared page behind every block the runtime does not already back |
| `amdfq/amdfq-tail-rs/src/registry.rs` | Live-allocation map keyed by the pointer `hipMalloc` returned, plus the end-address index |
| `amdfq/amdfq-tail-rs/src/real.rs` | The only `dlsym` in the crate |
| `amdfq/amdfq-tail-rs/run.sh` | Builds the object and starts `test` or `train` under it |
| `amdfq/amdfq-tail-rs/test.sh` | Export-face / style / one-allocation self-check |
| `amdfq/amdfq-tail/` | Original C sources; `hook.sh` still builds `libamdfq.so` |

`amdfq/amdfq-tail-rs/run.sh` and `test.sh` are the interface: they build the object if it is missing.

```bash
bash amdfq/amdfq-tail-rs/test.sh                       # default: one torch call in the env's interpreter
bash amdfq/amdfq-tail-rs/run.sh train                  # a real run
bash amdfq/amdfq-tail-rs/test.sh ./any/hip/program     # anything that calls hipMalloc
```

Underneath that is `LD_PRELOAD`. The hook logs to stderr (`AMDFQ_LOG_LEVEL` filters; default `info`):

```bash
LD_PRELOAD=$PWD/amdfq/amdfq-tail-rs/target/release/libamdfq_tail_rs.so bash start_train.sh
```

A line is `LEVEL T=tid amdfq_tail_rs::hooks <fn>(args) -> ret=…` (the C object's lines were
`seq elapsed_s T=tid <fn>(args) -> ret=… caller=<object>+offset live=…`, which is what the sample
below is). The living object writes one `write(2)` per line to stderr, so the SIGABRT of §6 still
leaves every line up to the fatal kernel on the terminal / `run.sh` log. Nothing is ever written to
stdout, which in a trainer process is `api.py`'s NDJSON channel. `AMDFQ_TAIL=0` disables the guard
(the three gates keep logging). There is no destructor summary (DESIGN.md D7).

```
000003     0.884444 T=92116  hipMalloc(size=2097152) -> ptr=0x7f9caa400000 ret=0 live=1 live_bytes=2097152 caller=libc10_hip.so+0x41efa
000007     1.105912 T=92116  hipFree(ptr=0x7f9caa400000) -> ret=0 live=0 live_bytes=0 caller=libc10_hip.so+0x26918
```

The C `caller=` field was resolved through the loader's `link_map` base — the one §7.4 says to trust —
so `libc10_hip.so+0x41efa` is §7.1's `alloc_block` frame. The torch path is the one measured end to
end so far; for a full run of this configuration §7.1's counts put the log at ~8.6 k lines
(4801 + 3693 + 84 calls).

**The pad** (historical: this was the object's default from the first mitigation through the four-round
measurement of §11; it is now the fallback of §13, not the path every `hipMalloc` takes). `hipMalloc`
used to add 16 bytes to every request: the caller kept its own size, the runtime was asked for 16 bytes
more, and the log carried both (`size=… padded=…`). The real effect is not 16 bytes, because the runtime
reserves in 2 MiB granules. Measured 2026-09-17 by reading the size of the mapping that holds the
returned pointer (`/proc/self/maps`) and the free VRAM before and after the call (`hipMemGetInfo`;
scratch `/tmp/amdfq-pad/vma.c`, `cost.c`):

| Request | Without the pad | With it |
| --- | --- | --- |
| 1 MiB | 2 MiB reserved | 2 MiB |
| 2 MiB | 2 MiB | **4 MiB** |
| 4 MiB | 4 MiB | **6 MiB** |
| 8 MiB | 8 MiB | **10 MiB** |
| 32 MiB | 32 MiB | **34 MiB** |
| 44 MiB (§7.3's fault-adjacent block) | 44 MiB | **46 MiB** |

A request that is not a multiple of the granule already carries slack and costs nothing; one that is an
exact multiple gains a whole extra granule of address space *and* of VRAM — and exact multiples are what
torch's caching allocator and hipBLASLt's workspaces ask for, and what §7.3's 44 MiB and 42 MiB blocks
were. That this is enough to move the step-10 fault, and what it costs, is measured in §10; the log's
closing summary line is the run's own account of the bytes (`# hipMalloc bytes: callers asked … , the
runtime was asked …`).

**What a run shows.** Measured 2026-09-17 under the pad: the default command — one `torch.zeros` on the
GPU — logged `hipMalloc(size=2097152 padded=2097168)` attributed to `libc10_hip.so+0x41efa`, and a
program that allocates all three ways logged `hipMalloc`, `hipHostMalloc` and `hipFree` with the pointers
it received. Under the tail guard the same command logs `hipMalloc(size=2097152)` (no `padded=` field)
plus a `# vmm tail:` ready line. A command that never touches HIP still writes a log, saying 0 calls; a
command that fails writes one too, and `hook.sh` exits with that command's status.

**What it cannot see.** `LD_PRELOAD` interposes calls that cross a PLT. A call the runtime makes to its
own entry points *inside* `libamdhip64.so.7` need not, so these counts are a lower bound on what a Frida
session (§7) records; the attribution of each logged call is unaffected. The reverse direction is
bounded the same way: nothing here reports on the VMM or async families, which §7 measures as dead in
this configuration.

## 10. The fix

The pad §9's `hipMalloc` interposer carries is the answer to §6's step-10 abort: raise every device
allocation by `AMDFQ_VRAM_PAD` (16) bytes, so that a bf16 Tensile kernel which reads past the end of its
operand lands in the allocator's own slack instead of in an unmapped page. This part of the section gives
the principle, the evidence that it is the pad — and not the act of interposing — that removes the fault,
and what the pad costs.

### 10.1 The principle

The defect stays where §6 and `../../conclusions/bf16-kernel-overrun.md` put it: in the kernel, not in the
allocator. The gfx1201 Tensile bf16 kernels over-read the A/B operands they were handed. What turns that
into a fatal abort is arithmetic. §7.1 shows the process's whole allocation path is `hipMalloc`, and the
runtime reserves both address space and VRAM in 2 MiB granules, so an allocation whose requested size is
an exact multiple of the granule ends exactly where its mapping ends: the byte after its last byte is the
first byte of an unmapped granule. An over-read of a few bytes therefore walks onto a page that does not
exist, and the GPU raises the `[gfxhub] page fault` of §6.

The pad inverts that arithmetic without touching the kernel. `hipMalloc` asks the runtime for `size + 16`
and still hands back a pointer whose usable extent the caller believes is `size`:

- the caller sees no change — same pointer, same `size`, and `hipFree` needs nothing but the pointer;
- the runtime can no longer use the granule the request ends in, because `size + 16` does not fit it, so
  it reserves the next one as well: the bytes past the end of the requested block are now mapped, in VA
  *and* in VRAM;
- an over-read of anything up to about a granule — 2 MiB, not 16 bytes — lands in memory that belongs to
  the process.

That granule of slack is the whole trick, and it is also the limit of it. The kernel still over-reads: the
pad buys somewhere harmless for the read to land, it does not stop the read. It cannot help a read that
walks further than a granule, and it protects only what it is applied to — device allocations through
`hipMalloc`. `hipHostMalloc` (§7.1's third gate) is left alone. Every log line carries both sizes, the
caller's and the runtime's, so the pad is visible in the transcript rather than implied.

### 10.2 The verification

The claim to test is narrow, and it has to be separated from the confound sitting next to it: the run also
gains an `LD_PRELOAD` object, and a different allocator layout could by itself move where the over-read
lands. All runs below are 2026-09-17, the same `config.toml`, the same seed and the same launcher; only the
preloaded object differs.

| Run | Preloaded | Pad | Outcome |
| --- | --- | --- | --- |
| `lllj_20260917_041147` | — | — | aborted at step 10; `[gfxhub] page fault` 04:12:31, pid 71023 |
| `lllj_20260917_045346` | — | — | aborted at step 10; fault 04:54:10, pid 85072 |
| `lllj_20260917_050031` | `libamdfq.so` | 16 | 14 steps, no fault, stopped by hand |
| `lllj_20260917_050412` | `libamdfq.so` | 16 | 41 steps, exit 0, `status=finished` |
| `lllj_20260917_050950` | `libamdfq.so`, pad-0 build | **0** | **aborted at step 10**; fault 05:10:16, pid 96771 |

The last row is the control, and it is what makes the first four rows an attribution rather than a
coincidence. It is built from a copy of `amdfq/amdfq-tail/` that differs in exactly one line —
`amdfq_log.h`: `#define AMDFQ_VRAM_PAD 16` → `0` (`diff -r` against the project is that line and nothing
else; copy and build under `/tmp/amdfq-pad0/`). That it interposes without padding anything is in its own
transcript (`/tmp/amdfq-run/pad0.log`): the header says `hipMalloc requests raised by 0 bytes`, and all
4949 `hipMalloc` lines it managed to write have `padded` equal to `size` (0 exceptions).

The rest of the comparison is what rules out the other explanations:

- **Same run, not a similar one.** The control's first 4949 `hipMalloc` sizes match the padded run's first
  4949 sizes one for one, and diverge only at the 4950th — which the control never reached, because it was
  dead by then. Same seed, same data order, same allocation sequence, same code path.
- **Same speed.** Median 1.160 s/step over the control's 10 steps against 1.154 over the padded run's 40
  (and 1.159 for a hook-free run over the same steps): the interposer is not perturbing timing, which is
  the other way a preload could have influenced the fault.
- **The same fault comes back.** Same step (10, where both hook-free runs also died), same signature
  (`[gfxhub] page fault`, `GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031`, `Faulty UTCL2 client ID: TCP`), and
  the aborted run's log carries no `fini` summary for the trainer — the SIGABRT of §6, not the clean exit
  the padded runs produced.
- **Kernel side.** `dmesg` shows a `[gfxhub] page fault` for every aborted run in the table and none for
  either padded run: the last fault before them is 04:54:10, the last one in the whole log is 05:10:16, the
  control's.

So the pad is what removes the fault, in this configuration and over these steps. That is the extent of
the claim: one padded run of 41 steps and one shorter one on one seed, against two hook-free aborts and one
pad-0 abort. Nothing here is a stability result — no long run, no repeat, no second seed — and no second
configuration.

### 10.3 The cost

Measured from the padded run's own log (`/tmp/amdfq-run/hooked.log`, 26928 lines, 14038 `hipMalloc` calls
written over its 41 steps and the encoding phase before them) by replaying its malloc/free stream and
reserving `round_up(size, 2 MiB)` per live block, which is what §10.1 says the runtime does:

| | Without the pad | With it | Cost |
| --- | --- | --- | --- |
| Peak resident, whole run | 12,244 MiB | 14,778 MiB | **+2,534 MiB (1.207×)** |
| Peak resident, steady-state training (5 s windows, 15–55 s) | 11.2–12.2 GiB | 13.6–14.7 GiB | median **+2.49 GiB** |
| Device VRAM in use at that peak (`mem_info_vram_used`, sampled every 0.2 s) | — | 15,925 MiB of 16,304 MiB total | **379 MiB left**; 654 MiB of the usage is not the trainer |

The reason is the granule. **14032 of the 14038 requests — 100.0 % — asked for an exact multiple of it**
(10972 of them the caching allocator's 2 MiB small-pool segments), so each one bought a whole extra 2 MiB
of address space and VRAM instead of 16 bytes of slack. About 1255 allocations were live at the peak
moment, and 1255 × 2 MiB is the +2.5 GiB. The only size class that was off-boundary — 25 MiB, 6 calls —
cost nothing. Cumulative bytes say nothing about this: 172.15 GB asked for, 224,608 bytes of it the pad.
The cost is entirely in the granules.

Time costs nothing measurable: 1.154 s/step over 40 steps against 1.159 for a hook-free run over the same
steps, with 26908 log lines written in 60 s.

Two properties of that curve matter for whatever comes next: the peak is not in the training step but in
the end-of-run checkpoint save, and with the pad the card was measured within 379 MiB of full at that
peak. Without the pad the same moment would have had roughly 3 GiB free (2.9 GiB recomputed from the device
counter, 3.4 GiB from the log's own peak), so the pad spends about 2.5 GiB of the headroom this
configuration has at its lowest point.

Two bounds on these figures: the run was stopped at step 41, so the epoch-end sampling phase a full run
would add is not in them, and the "without the pad" column is a replay of the same log with the granule
rule applied, not a second run.

**The cost as it stands is not acceptable.** A mitigation that pays one granule per live allocation —
about 2.5 GiB, 20 % above the unpadded peak, which is most of a 16 GB card's remaining headroom, to buy a
mapped granule behind every block — is too crude to keep. What has been established here is a mechanism,
not a design: the fault is provably caused by the last bytes of a granule-aligned allocation meeting an
unmapped page, and provably removed by giving that boundary a mapped neighbour. Something has to bring the
price down before this can stand, and what that something is is deliberately left open in this note — the
measurements above are the constraint any candidate has to be judged against.

The first candidate tried against them is §12: it meets the cost constraint, it removes the fault, and it
is closed anyway, because it breaks the run in other ways that are not yet explained.

---

## 11. The two byte accounts, four rounds deep — what the pad costs, and what is real

§10.3 priced the pad from the padded run's own log, by replaying its malloc/free stream and charging
`round_up(size + 16, 2 MiB)` per live block: **+2,534 MiB** at the peak (14,778 vs 12,244 MiB, 1.207×;
1,255 live blocks) on a 16,304.0 MiB card. That is the number routes 1 and 2 have to beat, and it is a
*model* of three accounts that disagree with each other about the same process. This section measures them.
The same four rounds in their original Chinese working notes, including the intermediate readings this
section compresses, are kept verbatim at [`live-data.md`](live-data.md);
[`gfx1201-overread-story.md`](gfx1201-overread-story.md) narrates them alongside the rest
of the investigation.

| Instrument | How it is read | What it covers |
| --- | --- | --- |
| driver counter | `/sys/class/drm/card1/device/mem_info_vis_vram_used`, from outside the process | every BO the driver charged to VRAM, whoever made it |
| HIP ledger | `hipMemGetInfo` inside the process | what the runtime reports used/free to its own client |
| external HIP ledger | the same call from a bare C process that allocates nothing | the same ledger through a second HIP process |
| hook live table | `amdfq_live.c`'s table (`live=`, `live_bytes=`) | the bytes *requested* through the three interposed gates |
| BO dump | `amdgpu_vm_info`, per process (§11.5) | the buffer objects themselves — sizes, states, domains |
| third-party client | a Vulkan process with no ROCr in it (§11.4) | how much VRAM someone who is not ROCr's client can get |

Idle baseline for all of §11: `vis = 75,657,216 B = 72.1 MiB`; device total
`17,095,983,104 B = 16,304.0 MiB`; granule 2 MiB.

### 11.1 Round one: four frozen runs of the training process

Four runs of the real trainer under `amdfq`'s `LD_PRELOAD` object, identical except for `AMDFQ_PAD_BYTES`
(0 / 16 / 64 / 1,048,576). A stub at the end of step 5 in `trainer/loop.py` raised `SIGSTOP`, so all four
freeze in the same place (`state.json` at `training.step = 4`, step-4 `loss = 0.1005521` in all four) and
both accounts can be read on a stopped process. The stub was removed afterwards; `trainer/loop.py` matches
HEAD.

| Item (MiB) | 0 pad | 16 pad | 64 pad | 1 MiB pad |
| --- | --- | --- | --- | --- |
| hook live table: live allocations | 1,219 | 1,219 | 1,219 | 1,219 |
| hook live table: live bytes (requested) | 9,214.0 | 9,214.0 | 9,214.0 | 9,214.0 |
| Σ round_up(size, 2 MiB) over the live set | 9,220.0 | 9,220.0 | 9,220.0 | 9,220.0 |
| of which exact multiples of the granule | 1,213 | 1,213 | 1,213 | 1,213 |
| Σ(padded − size) over the live set | 0 | 0.02 | 0.07 | **1,219.0** |
| the process's own ledger (last sample before the freeze) | 9,933.0 | 12,183.0 | 12,091.0 | 12,183.0 |
| driver counter (frozen, stable over 30 s) | **11,895.3** | **12,210.9** | **12,211.0** | **12,211.0** |
| external ledger (frozen, 77 identical samples) | **11,942.00** | **12,258.00** | **12,258.00** | **12,258.00** |
| renderD128 mapped bytes | 11,766.2 | 12,082.2 | 12,082.2 | 12,082.2 |
| frozen-window spread of the counter | 0.051 (112 samples) | 0.047 | 0.055 | 0.043 |
| difference from 0 pad: counter / ext. ledger / renderD128 | — | +315.6 / +316.0 / +316.0 | +315.7 / +316.0 / +316.0 | **+315.7 / +316.0 / +316.0** |
| last `padded` field in the hook log | `2,097,152 → 2,097,152` | `→ 2,097,168` | `→ 2,097,216` | `→ 3,145,728` |

Exact bytes: counter 0 pad `12,473,077,760`, the three non-zero pads `12,804,075,520` /
`12,804,116,480` / `12,804,145,152` (69,632 B apart); external ledger `12,522,094,592` at 0 pad and
`12,853,444,608` — byte for byte — at all three non-zero pads; renderD128 2,359 maps / 11,766.2 MiB at
0 pad and 1,304 maps / 12,082.2 MiB at the others.

The allocation history is byte-identical in all four runs:

| Item | Value (same in all four) |
| --- | --- |
| `hipMalloc` / `hipFree` calls up to the freeze | 2,994 / 1,775 |
| live blocks / requested bytes | 1,219 / `9,661,579,264 B` |
| allocating threads (log lines) | 3 (3,300 / 1,473 / 84) |
| run ids | `lllj_20260917_202744`, `_203050`, `_203412`, `_204138` |
| freeze instant (`CLOCK_MONOTONIC`) | 47527.2 / 47713.1 / 47914.9 / 48360.9 |

The address space at the freeze:

| Item | 0 pad | 16 pad | 64 pad | 1 MiB pad |
| --- | --- | --- | --- | --- |
| mappings / total VA (MiB) | 4,344 / 34,663.0 | 3,295 / 34,978.9 | 3,288 / 34,979.0 | 3,288 / 34,967.6 |
| `/dev/dri/renderD128` | 2,359 = 11,766.2 MiB | 1,304 = 12,082.2 MiB | 1,304 = 12,082.2 MiB | 1,304 = 12,082.2 MiB |
| mappings of exactly 2 MiB | 1,931 | 7 | 9 | 6 |
| `PROT_NONE` (MiB) | 9,575.0 | 9,574.8 | 9,565.1 | 9,590.4 |
| `VmRSS` / `VmHWM` (MiB) | 4,322.5 / 6,951.7 | 4,322.0 / 6,950.7 | 4,332.1 / 6,961.9 | 4,295.2 / 6,945.1 |
| threads / `VmSwap` | 157 / 0 | 157 / 0 | 157 / 0 | 157 / 0 |

**The "rounded up to a fixed granule" explanation dies here.** For a pad rounded up to a granularity X,
the live set should cost this much:

| Assumed X | pad 16 / 64 expected | **pad 1 MiB expected** |
| --- | --- | --- |
| 4 KiB | 4.8 MiB | **1,219.0 MiB** |
| 64 KiB | 76.2 MiB | **1,219.0 MiB** |
| 256 KiB | 304.8 MiB | **1,219.0 MiB** |
| 512 KiB | 609.5 MiB | **1,219.0 MiB** |
| 1 MiB | 1,219.0 MiB | **1,219.0 MiB** |
| 2 MiB | 2,426.0 MiB | **2,426.0 MiB** |

Measured: 16 B → +315.6 (counter) / +316.0 (external ledger) / +316.0 (renderD128); 64 B → +315.7 /
+316.0 / +316.0; **1 MiB → +315.7 / +316.0 / +316.0**. So (1) only the 256 KiB row explains the *size* of
the 16/64 step, (2) that model demands **at least +1,219.0 MiB** for a 1 MiB pad — a lower bound, since
`Σ(padded − size) = 1,278,214,144 B` — and (3) it was measured at **+0.066 MiB**. The whole family is
dead: what is real is a **+316.000 MiB step that appears as soon as the pad is non-zero and then
saturates**, independent of the pad's size.

The hook log's own tail carries the same shape as a side observation: the address stride between the last
three `caller=libc10_hip.so+0x41efa` returns is 4 MiB at 0 pad and 6 MiB at 16 / 64 B / 1 MiB.

While the run is alive both accounts oscillate hard (counter 7,595.4–13,055.3 MiB over the 8 s before the
0-pad freeze; the process's ledger 7,552–11,767 MiB) and close to within 47 MiB the moment the process
stops. The "own ledger" row above is a last-sample-before-freeze number and must not be compared with a
frozen value (§11.7.1).

What round one left open: (a) is the counter's excess real memory or a counting artifact, (b) what is the
256 KiB-looking step, (c) why does a 1 MiB pad cost no more than 16 B, (d) whose memory is the excess.
Rounds two to four answer all four, and §11.5 names it.

### 11.2 Round two: the same stream through a pure HIP program

Round one cannot separate the pad's effect from the hook's own cost or from torch's runtime, so this round
drops both: `gen_replay.py` turns the frozen hook log of the 0-pad re-run (`/tmp/amdfq-frozen5t0b.log`,
cut at the freeze instant) into a C table, `replay-frozen5t0b.cpp` replays it, and the program has no
`LD_PRELOAD`, no hook and no torch — only `hipMalloc`/`hipFree`/`hipHostMalloc`, `hipMemGetInfo` and sysfs
reads. It `SIGSTOP`s itself when the stream ends. Four runs differ only in `--pad`: 0 / 16 / 64 /
1,048,576 (same binary, sha256 `8d3b2b0f8848aec6`, same starting baseline).

**Is the stream complete?** `completeness.py` replays five frozen hook logs (0-pad original, 0-pad re-run,
16 / 64 B / 1 MiB) with the same cut-off rule (the hook line copied into `/tmp/stop<tag>-report.txt`
carries process-relative seconds, the same clock as the log):

| Log | cut (s) | M | F | H | unmatched/double free | duplicate ptr | unparsed | final live set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `frozen5t0` | 19.923 | 2,994 | 1,775 | 84 | 0 | 0 | 0 | 1,219 / 9,661,579,264 B |
| `frozen5t0b` | 19.859 | 2,994 | 1,775 | 84 | 0 | 0 | 0 | same |
| `frozen5t16` | 19.927 | 2,994 | 1,775 | 84 | 0 | 0 | 0 | same |
| `frozen5t64` | 19.887 | 2,994 | 1,775 | 84 | 0 | 0 | 0 | same |
| `frozen5t1m` | 19.951 | 2,994 | 1,775 | 84 | 0 | 0 | 0 | same |

All five land on the same live set, and only the three interposed line shapes appear, so the three gates
cover every allocation and free in the window. **Limit:** the hook records requests, not reservations and
not placement.

**The four pads** (MiB; each running from `vis = 75,657,216 B`):

| Item | pad 0 | pad 16 B | pad 64 B | pad 1 MiB |
| --- | --- | --- | --- | --- |
| `Σ(padded − size)` | 0 | 19,504 | 78,016 | 1,278,214,144 |
| `live_padded` (bytes) | 9,661,579,264 | 9,661,598,768 | 9,661,657,280 | 10,939,793,408 |
| process's own ledger used | **9,302.0** | **11,728.0** | **11,728.0** | **11,728.0** |
| driver counter (frozen) | **11,402.0** | **11,718.0** | **11,718.0** | **11,718.0** |
| external ledger used | **11,412.0** | **11,728.0** | **11,728.0** | **11,728.0** |
| counter − ledger | **+2,100.4** | **−9.66** | **−9.67** | **−9.66** |
| mappings of exactly 2 MiB | 1,925 | **1** | **1** | **1** |
| renderD128 bytes / maps | 11,334.0 / 2,517 | 11,650.0 / 1,464 | 11,650.0 / 1,461 | 11,650.0 / 1,461 |
| total VA | 20,427.9 | 20,743.6 | 20,743.6 | 20,743.6 |

Exact bytes: own ledger `9,753,853,952` (pad 0) against `12,297,699,328` at all three non-zero pads, byte
for byte; counter `11,956,224,000` / `12,287,569,920` / `12,287,561,728` / `12,287,569,920` (the 8,192 B
between 16 and 64 B is the counter's ~15 MiB quantisation); external ledger `11,966,349,312` /
`12,297,699,328`, identical to the in-process reading at every non-zero pad. GTT does not move within any
run.

**Three findings, all exact.**

1. **`ledger used = Σ round_up(request, 2 MiB) + 82.0 MiB`.** Pad 0: `9,220.0 + 82.0 = 9,302.0`
   (measured 9,302.0). Pad 16 B / 64 B / 1 MiB: `9,220.0 + 2,426.0 + 82.0 = 11,728.0` (measured 11,728.0
   in all three). The 1,213 live blocks that were exact multiples of the granule each gain a whole granule
   from **any** non-zero pad, which is why 16 B costs exactly what 1 MiB costs. The 82.0 MiB is this
   program's floor with no kernels run (the sentinel program of §11.3, which allocates one scratch block,
   reads 84.0 MiB).
2. **The counter's excess over the ledger exists only while the requests are granule-aligned.** At pad 0
   the process has 1,925 mappings of exactly 2 MiB and `counter − ledger = +2,202,370,048 B =
   +2,100.4 MiB`; at 16 B / 64 B / 1 MiB there is 1 such mapping and the difference collapses to
   `−10,129,408 B = −9.66 MiB` (64 B: −10,137,600 B). The training runs show the same on/off shape
   (1,931 → 7 / 9 / 6). The step decomposes exactly:
   `Δcounter = Δledger + Δgap = 2,543,845,376 + (−2,212,499,456) = 331,345,920 B = +316.000 MiB`. In
   words: a non-zero pad does two things at once — it makes the runtime charge a granule for every aligned
   live block (ledger +2,426.0 MiB), and it destroys the granule alignment, which removes ~2.1 GiB of
   "unowned" exactly-2-MiB mappings from the counter side (gap −2,110.0 MiB). The two nearly cancel, and
   that is the whole of the saturating step §11.1 could not explain. §11.5 says what those mappings are.
3. **The training process differs from the pure replay by a constant.** Same stream, four pads: the
   external-ledger offset is `555,745,280 B = 530.00 MiB` and the renderD128 offset 432.2 MiB at *every*
   pad, byte for byte; the counter offset is 492.57–492.88 MiB (only pad 0 is 0.3 MiB above the rest).
   Whatever else the training process holds does not move with the pad.

**Audit** (`audit.py`, checks A–D): A — the log stream re-parsed against the emitted C table: all 4,853
operations match in kind, size, order and free target, slots 0..3077. B — the C table simulated in Python
against the hook's own frozen table: `live=1219`, `Σ=9,661,579,264 B`, size-multiset sha1
`6d7c7d7804352d0e` — identical. C — what the runs printed against what the pad implies:
`live_padded = live_bytes + 1219 × pad` (0 → 9,661,579,264; 1 MiB → 10,939,793,408), with the 16 / 64 B
runs checked against the same identity separately. D — same binary and baseline for both runs; its last
step reads `/proc/<pid>/exe` and needs the processes alive, so it was not re-run today.

**The prediction written before the runs.** Findings 1–2 say pad 16 B / 64 B must give the same two
columns as pad 1 MiB. Measured afterwards: all three non-zero pads give own ledger `12,297,699,328 B`,
counter `12,287,569,920` / `12,287,561,728` / `12,287,569,920`, external ledger `12,297,699,328 B`, and 1
exactly-2-MiB mapping. Byte-for-byte hit.

### 11.3 Round three: sentinels — write every byte, then take what is left

Rounds one and two still trust the ledger's *numbers*. Round three tests the memory itself: a program
(`/tmp/amdfq-sentinel/replay_sentinel.cpp`, again no hook and no torch) replays the same frozen stream at
pad 0, writes a sentinel pattern into **every** allocation (`seed(slot) + i·0x9E3779B97F4A7C15`), asks the
ledger for the remaining free bytes, requests exactly that, fills it, and finally reads every live
allocation back and compares.

| Item (MiB) | sent0 (one-shot) | sentA (2 MiB chunks, then free-back) | sentB (one-shot, then free-back) |
| --- | --- | --- | --- |
| own ledger used / free at the replay stop | 9,452.0 / 6,852.0 | 9,452.0 / 6,852.0 | 9,452.0 / 6,852.0 |
| requested / obtained | 6,852.0 / **6,848.0** | 6,852.0 / **6,842.0** (3,421 chunks; the 3,422nd failed) | 6,852.0 / **6,846.0** |
| after giving it back: ledger | — | **9,452.0 / 6,852.0** | **9,452.0 / 6,852.0** |
| after giving it back: counter | — | 12,914.6 (1,368.0 *above* the pre-grab value) | 9,434.5 (2,112.0 *below*) |
| sentinel words written / verified | 5,750,915,072 written | 1,207,697,408 verified | 1,207,697,408 verified |
| mismatches | **0** | **0** | **0** |
| GTT / `VmSwap` / host `SwapFree` | 34.938 / 0 / unchanged | same | same |

Two results. First, **the ledger's free is real to the ledger's own client**: the whole 6,852.0 MiB was
handed over (6,842–6,848 MiB after the runtime's own overhead), written with sentinels, read back with
zero mismatches, and the ledger returned byte-exactly to `9,452.0 / 6,852.0` on release. Second, **the
counter does not follow the ledger**: after the release it landed 1,368.0 MiB above its pre-grab value in
one run and 2,112.0 MiB below it in the other, accounted for one-for-one by exactly-2-MiB mappings
(`+684 × 2 MiB`, `−1,056 × 2 MiB`).

Across eleven checkpoints, ten satisfy

```
driver counter − own ledger = 2 MiB × (mappings of exactly 2 MiB − live blocks of exactly 2 MiB) − 21.4 MiB
```

with a residual of −21.4/−21.5 MiB; the exception is the one checkpoint taken before the settle delay
(§11.7.1). This is where the counter's excess first becomes countable: it is the pool of exactly-2-MiB
mappings, and the ledger never counts them. When the device is filled they disappear (sentA's fill leaves
4,292 mappings, of which 4,290 are live 2 MiB blocks).

### 11.4 Round four: a process with no ROCr in it

Rounds two and three both ask ROCr for memory, so neither separates "the driver counter is wrong" from
"ROCr holds driver-level memory it does not attribute to the app and hands back to its own client". The
deciding experiment is a **third process that never loads ROCr** and tries to take the memory ROCr claims
is free.

*Why Vulkan.* HIP is the same stack; ROCm's OpenCL (`libamdocl64`) sits on ROCclr→HSA→ROCr as well; Mesa's
rusticl does not, but its kernel compiler is broken on this machine (`clinfo` building a one-line kernel
prints `Attribute set does not match Module context!` then `LLVM ERROR: Broken module found` and aborts).
**Vulkan/RADV** goes libdrm→amdgpu ioctl, needs no shader compiler to allocate and map memory, and exposes
exactly the driver's two heaps: heap 1 (device-local) `17,095,983,104 B`, byte-identical to
`mem_info_vis_vram_total`; heap 0 `16,696,004,608 B`, equal to `mem_info_gtt_total`; memory types 3/4 are
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, so a VRAM buffer can be written from the CPU (5.5 GB/s).

*Independence, self-proved.* The third process's `/proc/<pid>/maps` contains `libvulkan_radeon.so` and
`libdrm_amdgpu` and **no** `libhsa-runtime64`, `libamdhip64` or `libamdocl64`; the trainer's contains
`libhsa-runtime64` and `libamdhip64`.

*The criterion, fixed in advance.* With `M_idle` the most a non-ROCr process can take on an idle card and
`M_withA` the most it can take while the HIP process is frozen holding its state (the third party's own
cost cancels in the difference):

| Hypothesis | Prediction for `M_idle − M_withA` |
| --- | --- |
| 1 the counter overcounts (the ledger is honest) | **9.151 GiB** (= ledger delta 9,911,140,352 − 85,983,232) |
| 2 the counter is honest (ROCr holds the rest) | **11.235 GiB** (= counter delta 12,138,266,624 − 75,657,216) |

*Method.* `b_vk.c` runs in three phases, `SIGSTOP`ping itself at each end: P0 builds its Vulkan instance
and device while the card is still empty (its own cost is 122,880 B of VRAM plus a 256 MiB GTT staging
buffer), P1 asks first for exactly what the HIP ledger claims is free and then walks a descending ladder
(1 GiB → 256 → 64 → 16 → 4 → 2 → 1 MiB) until the card says no, filling every block with a sentinel
pattern, P2 reads everything back through a staged GPU copy (BAR reads run at ~10 MB/s, the copy path at
~6.5 GB/s). A block counts as VRAM **only on positive evidence**: the counter must have risen by the
block's size (±256 KiB) and GTT must not have moved. `RADV_DEBUG=nogttspill` is required — without it RADV
silently satisfies a failed VRAM allocation from host memory. The sentinel constant differs from the HIP
side's (`0xD1B54A32D192ED03` vs `0x9E3779B97F4A7C15`), so cross-contamination would identify itself.

*Three runs.* `base3`: the third party alone (`M_idle`). `r0`: the HIP process alone, as a same-session
reference for its own grab. `dec0`: the decisive one — third party stopped at P0, HIP process replayed and
frozen, **30 s of nobody touching anything**, then the third party takes what it can, then the HIP process
wakes and grabs again, then both read their memory back.

| Checkpoint | HIP ledger used / free | driver counter | driver GTT | bare HIP reader used / free | third party holds |
| --- | --- | --- | --- | --- | --- |
| `base3` third party alone | — | 17,086,889,984 | 283,975,680 | 85,983,232 / 17,009,999,872 | **17,011,048,448 B (15.843 GiB)** |
| `r0` HIP alone, after replay | 9,911,140,352 / **7,184,842,752** | 12,107,448,320 | 21,889,024 | 12,125,732,864 / 4,970,250,240 | — |
| `r0` after its own grab | 17,095,983,104 / 0 | 17,077,714,944 | 21,889,024 (**unmoved**) | 17,095,983,104 / 0 | — |
| `dec0` b0 (third party ready) | — | 75,804,672 | 283,975,680 | 85,983,232 / 17,009,999,872 | 0 |
| `dec0` a1 (HIP frozen) | 9,911,140,352 / **7,184,842,752** | **12,138,266,624** | 290,328,576 | **12,125,732,864 / 4,970,250,240** | — |
| `dec0` a1 + 30 s hold | same | 12,138,307,584 → **12,138,266,624** | 290,373,632 → 290,328,576 | same | — |
| `dec0` b1 (third party done) | same | **17,087,578,112** | 290,328,576 | 12,125,732,864 / 4,970,250,240 (**unmoved**) | **4,949,278,720 B** |
| `dec0` a2 (HIP grabs again) | 17,095,983,104 / 0 | 17,086,160,896 | **5,231,304,704** | 17,095,983,104 / 0 | — |
| `dec0` a3 / b2 (both read back) | same as a2 | 17,086,771,200 | 5,242,228,736 | same as a2 | same as b1 |

The third party's ladder in `dec0`: 1 GiB ×0 (rejected first try) → 256 MiB ×2 → 64 ×1 → 16 ×2 → 4 MiB ×2
→ 2 MiB ×1 → 1 MiB ×6, i.e. 18 blocks = 4,609.0 MiB; all 18 passed the positive-evidence test (their
counter deltas sum to 4,949,307,392 B against 4,949,278,720 B obtained), and 7 attempts judged to be host
memory plus 1 with no charge were released.

**The resolution.**

| | Value |
| --- | --- |
| `M_idle` (third party alone) | 17,011,048,448 B = 15.843 GiB |
| `M_withA` (third party against the frozen HIP process) | **4,949,278,720 B = 4.609 GiB** |
| `M_idle − M_withA` | **12,061,769,728 B = 11.235 GiB** |
| hypothesis 1 predicts | 9,825,157,120 B = 9.151 GiB → **off by 2,236,612,608 B (2.083 GiB)** |
| hypothesis 2 predicts | 12,062,609,408 B = 11.235 GiB → **off by 839,680 B (0.007 %)** |

Plainly: the HIP process claimed **6,852.0 MiB** was still free, the third party got **4,720.0 MiB**, the
two differ by **2,132.0 MiB**, and after the third party finished the counter stood at
`17,087,578,112 / 17,095,983,104` — **8.0 MiB left**. Neither side's data was disturbed (HIP read back
2,105,802,752 words, the third party 618,659,840, both zero mismatches).

1. **Hypothesis 2 holds: the driver counter is honest, and ROCr really does hold ~2.07 GiB of driver-level
   VRAM that its own ledger does not attribute.**
2. The counter's honesty has a second, independent proof: every accepted block had to show a matching
   counter rise, and the counter ended 8.0 MiB below the device total.
3. **ROCr can recycle that holding for its own client.** In `r0` the HIP process got its entire
   7,184,842,752 B claim while the counter rose only 4,970,266,624 B and GTT did not move: 2,214,576,128 B
   (2,112.0 MiB) of the grab came out of ROCr's own holding. So "the ledger's free is real" (§11.3) is
   true for ROCr's client and false for everyone else.
4. **When VRAM really runs out, ROCr substitutes host memory and does not say so.** In `dec0`, after the
   third party took 4.95 GiB, the HIP process still got its full 7,184,842,752 B in one call; the counter
   did not move (−1,417,216 B) while **GTT went from 290,328,576 B to 5,231,304,704 B (+4.94 GiB)**. The
   read-back found no mismatches: the data is real, it simply is not in VRAM, and `hipMemGetInfo` reports
   it as VRAM throughout.
5. **The bare HIP reader sides with the counter.** Frozen, it reports `used 11.293 GiB / free 4.629 GiB`
   (the process itself says free 6.690 GiB), within 21 MiB of what the third party actually got; and after
   the third party took 4.95 GiB its numbers do not move at all — HIP's `free` cannot see non-HIP
   allocations (§11.7.3).
6. **The 2.07 GiB is a stable holding, not bookkeeping lag.** In the 30 s hold the counter moved 40,960 B
   and GTT 45,056 B.
7. **Nothing aliases.** Different sentinel constants on the two sides, both read their own memory back with
   zero mismatches, so the HIP process's 7.18 GB and the third party's 4.95 GB are not the same pages.

### 11.5 The BO-level reconciliation: what the "unowned mappings" are

Rounds one to four measure memory *accounts*. Four `amdgpu_vm_info` dumps (`/tmp/amdfq/`,
`{0,16,64,1m}pad-vm-info.txt`) measure the objects themselves: per process, every BO with its size and
state. Summing the trainer's VM at each pad's freeze:

| | 0 pad | 16 pad | 64 pad | 1 MiB pad |
| --- | --- | --- | --- | --- |
| trainer VM pid | 100391 | 101237 | 102083 | 103108 |
| BO count | **3,618** | 2,477 | 2,487 | 2,493 |
| BO bytes | 12.267 GiB | 12.576 GiB | 12.576 GiB | 12.576 GiB |
| of which VRAM | 12.260 GiB | 12.568 GiB | 12.568 GiB | 12.568 GiB |
| `Kernel PT/PDs` (4 KiB BOs) | 1,086 = 4.24 MiB | 1,000 = 3.91 MiB | 1,010 = 3.95 MiB | 1,016 = 3.97 MiB |
| BOs of exactly 2 MiB | **1,926** (= 3,852 MiB) | **2** | **2** | **2** |
| BOs of exactly 4 MiB | none in the top list | **872** (= 3,488 MiB) | 872 | 872 |
| child-process VMs | 23 BOs / 10 MiB | 23 / 10 MiB | 23 / 10 MiB | 24 / 10 MiB |

The dominant size classes, pad 0 → padded: 2 MiB ×1,926 → 4 MiB ×872; 20 MiB ×149 → 22 MiB ×149;
26 MiB ×66 → 28 MiB ×60; 14 MiB ×107 → 16 MiB ×125; 8 MiB ×24, 256 KiB ×23, 512 KiB ×19, 24 KiB ×12 and
the 4 KiB PTs are the same in both. Four things follow, and they close every question round one opened.

1. **It is not page tables.** `Kernel PT/PDs` is 3.9–4.2 MiB at every pad — three orders of magnitude
   below the ~2.1 GiB at issue.
2. **The "unowned 2 MiB mappings" are a pool of real 2 MiB BOs.** Pad 0 holds **1,926** BOs of exactly
   2 MiB, the padded runs hold **2**. The difference splits as ~869 live blocks of exactly 2 MiB (the hook
   table's own count) plus **~1,055 that belong to no live allocation** — the same number the address-space
   count produced in §11.3 (1,058) by an unrelated route. Those ~1,055 BOs are `1,055 × 2 MiB = 2,110 MiB`,
   against the 2,112.0 MiB ROCr handed back to its own client in round four. **The holding is a pool of
   about a thousand and fifty 2 MiB BOs.**
3. **The padded runs have no pool, and their histogram carries the pad's fingerprint.** Every
   granule-aligned size class moves up exactly one granule (2 → 4, 14 → 16, 20 → 22, 26 → 28 MiB), and the
   pool is gone (2 exactly-2-MiB BOs left). The pad converts a *reusable* ~2.1 GiB pool into ~2.4 GiB of
   granules permanently attached to live blocks.
4. **The BO totals reproduce the counter's difference independently.** 12.267 vs 12.576 GiB = **+316 MiB**
   for the padded state, against the counter's +315.66 MiB (training runs) and +316.000 MiB (pure replay).
   Two instruments sharing no code agree to within 1 MiB.
   One caveat: the BO totals sit ~660 MiB above the counter at *every* pad (12,554 vs 11,895 MiB at
   pad 0), a constant offset that cancels in the differences and is not explained here.

### 11.6 What the four rounds establish

The pad's price, measured three ways on the same frozen live set of 1,219 blocks / 9,214.0 MiB:

| Instrument | 0 pad | padded | Difference |
| --- | --- | --- | --- |
| HIP ledger (what its client can still ask for) | 9,302.0 used / 7,002.0 free | 11,728.0 used / 4,576.0 free | **+2,426.0 MiB used / −2,426.0 MiB free** |
| driver counter (real VRAM, proven truthful by §11.4) | 11,402.0 | 11,718.0 | **+316.0 MiB** |
| BO dump (the objects themselves) | 12,267 MiB | 12,576 MiB | **+316–317 MiB** |

The three are not in conflict, and §11.5 says why:

- at pad 0 the live blocks carry **no** extra granules; instead ROCr keeps **~1,055 two-MiB BOs
  (2,110 MiB) as a pool**, which the driver counts as used but which ROCr hands out on demand (§11.4 `r0`)
  — so the process's usable memory is `4,902 MiB of device free + 2,114 MiB of pool ≈ 7,016 MiB`, against
  its own ledger's 7,002.0 MiB;
- with the pad, each of the ~1,213 aligned live blocks owns one more granule (**+2,426 MiB**, permanent)
  and the pool is gone, so usable memory is just the device's free VRAM:
  `16,304 − 11,718 = 4,586 MiB`, against its ledger's 4,576.0 MiB;
- the *net* device-level difference is `2,426 − 2,110 = 316 MiB`, exactly what the counter and the BO
  totals report.

So padding and pooling are two ways of spending granule-sized memory behind live blocks: **the pad spends
it on granules nobody can reuse; the unpadded state spends it on a pool ROCr hands straight back.** For a
process that is ROCr's own client — which the trainer is — the difference is the full **2,426.0 MiB
(2.37 GiB)** of usable memory, not the 316 MiB the device counter shows. §10.3's `+2,534 MiB` is the same
phenomenon measured at the run's peak (1,255 live blocks, end-of-run checkpoint save) instead of at the
frozen step-5 state.

Two consequences to keep:

- **`hipMemGetInfo` cannot decide how much headroom a run has.** For the process itself it is optimistic
  by ~2.1 GiB (the pool is charged to the driver yet still handed over), and for anyone else it is
  optimistic by the same 2.1 GiB in the other direction: §11.4 measured a non-ROCr process getting
  4,720.0 MiB where the ledger advertised 6,852.0 MiB.
- **VRAM exhaustion does not produce a failure, it produces host memory.** `dec0` shows `hipMalloc`
  succeeding on a full card with the difference silently landing in GTT (+4.94 GiB) and no error returned.

### 11.7 Methodological warnings (they apply to every number above)

1. **The counter spikes during an allocation burst and settles over hundreds of milliseconds.** One run
   read 14,673.1 MiB at the freeze instant and 11,605.1 MiB 0.25 s later, with a further 58.6 MiB of slow
   decay. Every program here synchronises and sleeps 0.5 s before reading (`--settle 500`); a "frozen
   instant" number taken without that is 0.05–2.5 GiB high.
2. **The external ledger reader carries its own footprint.** The first version was a torch process holding
   2,132.0 MiB of its own, which lands in its `used` reading — hence §11.4's bare C reader, whose floor
   (82.0 MiB) is measured separately, and hence round one's external-ledger column must not be compared
   with its in-process column.
3. **`hipMemGetInfo` is device-wide between HIP processes and blind across APIs.** Four fresh HIP processes
   each saw the trainer's usage (round one); in round four a fresh HIP process saw nothing of the third
   party's 4.95 GiB.
4. **Every run starts from an idle card.** A leftover process holding 164 MiB moved one checkpoint by that
   much, and one holding 9.4 GiB made the next run fail at its 267th allocation. Baselines used here:
   `vis = 75,657,216 B`, and for round four `gtt = 15,536,128 B` (round one's `gtt = 30,281,728 B` is the
   same idle card in a different session; what matters is that GTT does not move *within* a run).
5. **Placement must be proved per allocation, not inferred.** A size threshold cannot do it: 4 MiB blocks
   landing in host memory slipped past an 8 MiB threshold. The rule used here is positive evidence —
   counter rise ≈ block size, GTT rise ≈ 0 — and RADV must be started with `RADV_DEBUG=nogttspill`, or it
   will silently satisfy a failed VRAM allocation from host memory. `VK_EXT_memory_budget`'s per-heap usage
   is no help: RADV books a host-memory block under the heap that was *requested*.
6. **`kill -CONT` pollutes a hook log.** Killing a `SIGSTOP`ped process requires `CONT`, and that second of
   running appends new lines. Frozen-state analysis must cut the log by the *log's own* clock
   (process-relative seconds) — filtering by `CLOCK_MONOTONIC` silently reads the whole file.
7. **Accessing a `SIGSTOP`ped GPU process is safe; leaving one behind is not.** Every round ends with
   `kill -CONT` then `KILL`, then a check that the counters are back at the idle baseline.

### 11.8 Artifacts

| Content | Path |
| --- | --- |
| Round one: hook logs, ledger samples, snapshots, freeze reports | `/tmp/amdfq-frozen5t{0,16,64,1m}.log`, `/tmp/ledger-frozen5t*.log`, `/tmp/extledger-frozen5t*.log`, `/tmp/samp-frozen5t*.log`, `/tmp/frozen5t*-snap/`, `/tmp/stop5t*-report.txt` |
| Round one: the `amdgpu_vm_info` dumps analysed in §11.5 | `/tmp/amdfq/{0,16,64,1m}pad-vm-info.txt` |
| Round two: generator, C stream, binary, harness, audit, completeness | `/tmp/amdfq-replay/{gen_replay.py, replay-frozen5t0b.cpp, replay, run-replay.sh, measure.py, audit.py, completeness.py}`, reports `report-replay{0,16,64,1m}.txt`, snapshots `/tmp/replay-replay{0,16,64,1m}-snap/` |
| Round three: sentinel program and harness | `/tmp/amdfq-sentinel/{replay_sentinel.cpp, gen_ops.py, run.sh, sample-host.py, report.py, audit.py, cleanup.py}`, `/tmp/sent-sent{0,A,B}.{out,err}`, `/tmp/sent-sent{0,A,B}-snap{1..4}/` |
| Round four: third party, probe, bare reader, orchestrator | `/tmp/amdfq-vk/{b_vk.c, vkprobe.c, hipfree.cpp, drive.py, sample-host.py, report.py, cleanup.py, memmon.py}`, `/tmp/vk-{base3,r0,dec0}.*`, `/tmp/vk-{base3,r0,dec0}-snap*/` |
| Compilers | `hipcc --offload-arch=gfx1201 -O2 -Wall -std=c++17` (HIP), `gcc -O2 -Wall -lvulkan` (Vulkan) |
| Every round | start from an idle card; `ulimit -c 0`; one GPU process at a time; kill and confirm the baseline afterwards |

---

## 12. Optimization route 1: an allocator on the VMM API — closed 2026-09-17, kept on the books

**Status.** Closed as an optimisation. It does what §10.3 asks for — the pad costs *one* 2 MiB granule for
the whole process instead of one per live allocation — and it removes the step-10 fault it was built for,
but it damages the process in several other ways, none of them the fault it was built to remove, and none
of them explained. What follows is what was measured before it was set aside. **The route keeps its
research value**: an allocator whose memory makes two unrelated kernels raise an illegal shader
instruction, sends the GPU to a wild pointer, puts a wave's PC in unmapped space and turns a training
step's loss into NaN is not a finished argument, and the phenomenon (§12.2) is not yet placed — it has the
shape of random corruption of the process's own state rather than of a kernel defect.

### 12.1 What it was, and what was established before any training ran

The mechanism is §10.1's, moved out of the runtime's hands. Instead of asking `hipMalloc` for `size + 16`
so that the runtime's own granule rounding leaves a mapped neighbour behind the block, take over the layout
with the VMM API of §5.2 — `hipMemAddressReserve`, `hipMemCreate`, `hipMemMap`, `hipMemSetAccess` — and map
**one** physical granule behind *every* block, so the slack costs one granule in total rather than 1255 of
them (the peak count of §10.3).

The whole route rests on one thing the HIP headers do not state either way: that one
`hipMemGenericAllocationHandle_t` can be mapped at many virtual addresses at once. `amdfq-vmm/vmm_probe.c` (retired C VMM tree, deleted)
answers it, and measures the price:

| | pad-16 (§10.1) | VMM + one shared granule |
| --- | --- | --- |
| VRAM for 1000 × 2 MiB blocks | 4000 MiB | **2014 MiB** (1000 blocks + the single pad granule) |
| one `hipMalloc` | 7.6 µs | 18.6 µs (reserve + create + 2 × map + set-access) |
| one `hipFree` | — | 15.2 µs |

`multimap` shows the aliasing is real (one handle at 8 VAs; 1000 mappings of one pad handle cost the
physical 2 MiB **once**) and that a read across a block's end lands in the shared granule instead of
faulting; a 1 TiB reservation costs address space only. So the cost constraint of §10.3 is met, with the
call overhead as the only visible price.

The hook's implementation is `amdfq-vmm/amdfq_vmm.c` (retired C VMM tree, deleted): one `hipMemAddressReserve` for the process
(`AMDFQ_VMM_RESERVE`, default 1 TiB), a first-fit free list over the extents a run gives back,
`hipMemCreate` + `hipMemMap` + `hipMemSetAccess` per block, the shared pad granule behind it. Knobs:
`AMDFQ_VMM=0` turns serving off (falling back to §10.1's +16), `AMDFQ_VMM_PAD=0` keeps the layout and maps
no pad, `AMDFQ_VMM_LAYOUT=peralloc` reserves per allocation instead of carving an arena (§12.3).

### 12.2 What a training run does with it: five different failures, and not the step-10 abort

All of these are the same route on the same `config.toml` and seed, in two boots (before and after
2026-09-17 07:15). The first three are recorded in this machine's session notes with the runtime's own
abort text; the last two have their logs and run directories on disk.

| # | Build | What happened |
| --- | --- | --- |
| 1 | VMM, shared pad granule, host access | `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` in the first training step, in torch's own `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda…>`; rc 134; **no page fault in `dmesg`** |
| 2 | same, `AMDFQ_VMM_PAD=0` | the same abort in a different kernel — Tensile bf16 GEMM `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x16_…_ISA1201_…_WS32_WG16_8_1_WGMXCC1` — so **the shared pad granule is not the cause** |
| 3 | VMM + the dispatch tracer | an **instruction fetch** fault: `Faulty UTCL2 client ID: SQC (inst)`, address `0x7fa218f2d000` inside the 1 TiB reservation but **772.1 GiB past the highest mapped block** (738 blocks had used 9.6 GiB). A wave's PC, not its data |
| 4 | VMM, arena + pad (`lllj_20260917_073015`) | `loss=nan` at step 1 and nothing else: **no GPU event in `dmesg` at all**; the run ended because `control.write_state` refuses to serialise NaN |
| 5 | VMM, per-allocation, no pad (§12.3) | `lllj_20260917_074240`: NaN at step 1, `dmesg` clean. `lllj_20260917_074714`: a sane step-1 loss (0.9978), then a GPU page fault on **0x7f1bd361f000**, an address never allocated by the route (see §12.3) |

The reading is in what the five do *not* have in common. They do not share a kernel (a torch copy kernel,
a Tensile GEMM, an instruction fetch, no kernel at all), nor an operator, a size, a phase or a boot. What
they share is the kind of damage — a bad *value* (NaN), an illegal *instruction*, a bad *PC*, a bad
*pointer* — the process's own memory and dispatch state degraded in four different kinds of way, and twice
with the kernel log completely silent. §10.1's defect is the opposite of that: deterministic, bound to the
bf16 Tensile kernels, and it leaves a `[gfxhub] page fault` behind every time.

**One correction so it does not get counted twice.** A 2026-09-17 GPU reset (`ring reset failed` → MODE1
reset → `VRAM is lost due to GPU reset!`, device wedged) that was first written down beside these runs was
traced the same day to this directory's own test program: `vmm_kernel_test.hip`'s `spin_for_flag`, pid
30224, spinning on a plain **`hipMalloc`** block. It is not evidence about the route. The spin kernel is
still in that file, and it remains the one thing here that can wedge the device on purpose.

### 12.3 What the last round ruled out

**The layout.** `AMDFQ_VMM_LAYOUT=peralloc` removes the arena, the bump pointer and the free list from the
route entirely: every `hipMalloc` reserves exactly its own extent and every `hipFree` gives it back with
`hipMemAddressFree`. With the pad also off, the route is the smallest it can be — and it fails *earlier*
than the arena did, in two different ways in two consecutive runs of the same build:

| Run | Result |
| --- | --- |
| `lllj_20260917_074240` | 1420 allocations served, **0 fallbacks**; step 1 `loss=nan`, `dmesg` clean |
| `lllj_20260917_074714` | 1643 allocations served, **0 fallbacks**; step 1 loss 0.9978, then the page fault of §12.2 #5 |
| `lllj_20260917_074459` (control, `AMDFQ_VMM=0`) | step **66**, losses 0.0862 / 0.0528 / 0.1023 / 0.0067 / 0.1074 — past step 10, healthy |

Two facts fall out of the same logs. The faulting address `0x7f1bd361f000` is **not ours**: the route's
blocks that run lie in `0x7f9967e00000 … 0x7f9d8de00000`, and the fault sits 502 GiB below the lowest of
them, is not granule-aligned, and appears nowhere in a transcript of 1643 allocations and 638 frees. And
the teardown path itself works: VRAM returns to the idle card's reading when the process dies, so
unmap/release/address-free are all running.

**The VMM calls themselves.** `AMDFQ_VMM=0 AMDFQ_VMM_SHADOW=512` serves nothing — every allocation comes
from the real `hipMalloc` plus the §10.1 pad — and additionally drives the whole VMM sequence for a mirror
block that no kernel ever touches: reserve, create, map, set-access, then unmap, release, address-free when
the pointer is freed. The mapped total is capped (oldest mirror dropped first), and the call sequence is
the same one the served route runs, at the same sizes and the same rate.

| Shadow run (`lllj_20260917_075937`, 200 s) | |
| --- | --- |
| allocations mirrored | 42,001 |
| cumulative extents reserved | 572 GiB |
| held mapped continuously | 410–512 MiB (at the cap) |
| evicted by the cap / torn down normally | 26,097 / 15,855 |
| reserve, create and map failures | 0, 0, 0 |
| outcome | **111 steps**, losses 0.012–0.24, two sampling phases, no GPU fault in `dmesg` |

And the losses at steps 5 / 21 / 39 / 66 (0.1236 / 0.0750 / 0.0192 / 0.1074) are identical — to every digit
the progress bar prints — to the same run with no shadow at all. Forty-two thousand extra VMM calls, 572 GiB
of them, changed nothing.

So the calls, the bookkeeping, the free list, the arena, the pad granule and the teardown path are all
excluded, and the two routes' *own* memory is not where the wild pointer comes from. What is left is the
one variable none of these experiments moved: **torch's kernels actually reading and writing the pages the
route allocated.**

### 12.4 What a next attempt has to settle first

Three questions, in the order they look most likely to be the mechanism:

1. **The host access descriptor.** `hipMemSetAccess` is called with *two* locations, device **and** host,
   because that is what makes a device pointer readable from host code at the same address — which the
   training path does for real (`.item()` in `trainer/loop.py`, `trainer/dataset.py`,
   `trainer/sampling.py`), and without which the process dies in `_local_scalar_dense_cuda` instead. It is
   also the only property `hipMalloc`'s blocks have that this route has to manufacture, and it was never
   measured at training scale without it (device-only cannot run the training path at all). A bisect here
   needs a second way to make `.item()` work — a staging copy, or a stride over the affected tensors —
   before it can be posed.
2. **Scale.** The failing runs held 10–12 GiB live; the shadow could hold only 512 MiB, because the real
   allocations still had to fit the card. "VMM pages present, small" is excluded; "VMM pages present at the
   failing scale" is not. Raising the shadow's cap is the cheap next probe, and its limit is the card:
   §10.3 measures only 2.9–3.4 GiB free at the run's unpadded peak, so a 2 GiB cap spends most of what is
   left, and an OOM would not be evidence of anything.
3. **The order of the free path.** `hipMemUnmap` and `hipMemRelease` are not stream-ordered, and the
   route's `hipFree` calls them the moment the runtime asks — with nothing in between that would make the
   GPU's earlier work on that block have finished. Whether the `hipFree` it replaces gives its callers that
   ordering, and whether torch's or hipBLASLt's frees depend on it, is exactly what this experiment would
   answer: `AMDFQ_VMM_NOFREE=1` (accounting only, nothing unmapped or recycled) was proposed and never run,
   the price being a VRAM leak that allows only a few steps.

Also untouched, and the cheapest bisect of all if the mechanism turns out to be per-access rather than
global: serving only *some* classes of allocation through the route (hipBLASLt's workspace, the caching
allocator's 2 MiB segments, the sample-phase tensors) and leaving the rest to `hipMalloc`.

### 12.5 A hypothesis for the corruption, and the measurement it needs

**The hypothesis (maintainer, 2026-09-17, not yet tested).** torch does not stop at `hipMalloc`: after a
block comes back it hands that address to further memory-management APIs, which operate *on the block*
rather than reading and writing it as data. `hipMalloc`'s memory and VMM memory need not answer those calls
the same way — and where the answers differ, the caller acts on the difference, and what breaks after that
is process state, not the block's contents. That would explain what §12.2 lists and what §12.3 could not
attribute to any one cause: five unrelated failure shapes, none of them the step-10 fault, none of them
reproducible from the layout, the pad, the call sequence or the teardown, all of them downstream of "torch
asked the runtime something about a pointer".

The hypothesis is not free-floating: the two kinds of memory **are** distinguishable through the runtime's
own queries, measured 2026-09-17 with the three calls the current hook uses. A live `hipMalloc` block
answers `hipMemGetAddressRange` with its own base and size (`rc=0`) and appears in `/proc/self/maps` as a
`/dev/dri/renderD128` mapping, while `hipMemGetAccess` and `hipMemRetainAllocationHandle` fail on it
(`rc=1`, hipErrorInvalidValue) — the runtime's own pool is not VMM. A VMM block we mapped answers
`hipMemGetAccess` (`rc=0`, `flags=0x3`, device read-write) and `hipMemRetainAllocationHandle` with a handle,
and its range query returns the mapped range. Reserved-but-unmapped vault space answers nothing (all three
fail, no mapping). So the difference is real and observable; what §7 never looked for is **who asks**.

**The measurement that would settle it.** §7's method once more, but recording arguments instead of counts:
a broad Frida pass over the whole exported `hip*` surface (549 functions of `libamdhip64.so.7` plus the HSA
entry points, the way `agent6.js` does), in a live training run, with every call's arguments captured —
then the addresses in those arguments correlated against the set of addresses `hipMalloc` /
`hipHostMalloc` handed out earlier in the same run, and against the route's blocks when the route serves.
The output to want is a list of calls that received a *previously allocated* address, ordered by how often,
with their arguments. Cross-checked against §5's signatures and §7.1's measured gates, that list is the
candidate set: anything that takes a device pointer and *derives* something from it (range and attribute
queries, access and mapping calls, advice/prefetch, pool and IPC import, unmap/release, the memcpy family)
is a place where the two kinds of memory can part company, and the earlier attribution work has tested only
the handful of calls `vmm_diff.c` covers. The same pass should record the *sequence* around the first
failure — what torch asks about a block between receiving it and the kernel that dies on it — because the
hypothesis is about a caller acting on an answer, not about a single call.

**Why it is worth doing rather than another probe.** §12.4's three questions are all mechanism-shaped and
each needs code changes to pose. This one needs no change to the hook at all: it measures the *caller*, and
it is the only route-1 item left that can name a culprit instead of excluding one.

**What is on disk for whoever picks it up** (all unversioned scratch; the VMM sources named here are in the
`~/Desktop/amdfq/` backup — the tree's `amdfq/amdfq-tail/` does not build them, see the status paragraph at the top of
this note): `vmm_probe.c` (the
go/no-go and cost probe; the two `isolated` modes that fault on purpose need
`AMDFQ_PROBE_ALLOW_FAULTS=1`), `vmm_diff.c` (one `hipMalloc` block and one route block, queried through
every runtime call that describes them), `vmm_kernel_test.hip` (the hook's layout call for call, with a
trivial kernel on it and a `hipMalloc` control — and the `spin_for_flag` hazard above), `amdfq_vmm.c` /
`amdfq_vmm.h`, `amdfq_hooks_hip.c`. Run directories: `lllj_20260917_073015`, `_074240`, `_074459`,
`_074714`, `_075553`, `_075937`.

---

## 13. Optimization route 2: the tail guard — one shared page behind every block

Route 1 replaced the allocator and broke the process in ways still unexplained (§12). Route 2 keeps
`hipMalloc` doing the allocating and only *adds* what the pad was buying: a mapped page immediately past
the end of an allocation. That difference is also the experiment §12 asked for — is the damage from the VMM
calls themselves, or from serving *operands* out of VMM memory?

### 13.1 The idea

§10.1's fault arithmetic is about one address: the first byte after a granule-aligned allocation's last
byte. The pad handles it by making the runtime reserve one more granule per block, which is where §10.3's
+2.4 GiB comes from. Route 2 handles the same address directly:

1. after the real `hipMalloc` returns, take the block's extent (the runtime's own answer from
   `hipMemGetAddressRange`, rounded up to the granule);
2. ask what backs the page holding the 16 bytes past that extent: if `hipMemGetAddressRange` resolves
   inside another live allocation, a read there cannot fall off the end of anything and there is **nothing
   to do**;
3. if nothing backs it — the case that faults — take that page with `hipMemAddressReserve(hint = addr)` and
   map **one shared object**, created once at start with `hipMemCreate`, into it
   (`hipMemMap` + `hipMemSetAccess`);
4. on `hipFree`, unmap the block's own guard, then ask the same question again about the page the freed
   block *was* guarding for the allocation before it, so a neighbour that just lost its protection gets a
   new one.

The reservation uses `addr` as a *hint* and only counts when the driver puts it exactly there; a hint that
lands elsewhere means the page is held by something else (in practice the runtime's own reserved address
space), and then — and only then — the block is given back and re-allocated with `AMDFQ_TAIL_PAD` (16)
bytes added, i.e. §10.1's pad for that one block. Measured: **about 0.2 % of allocations take that path**.
Nothing else about what the caller gets changes — same pointer, same size, same layout.

Because every guard is a mapping of the *same* `hipMemCreate`d object, the scheme costs one physical
granule for the whole process. That sharing is a design requirement, not tidiness: the runtime books every
`hipMemCreate` object as 2 MiB of VRAM even when the object behind it is a 4 KiB page, so one object per
block would inflate `hipMemGetInfo` by 2 MiB per block while using almost nothing.

### 13.2 The implementation

The living code is `amdfq/amdfq-tail-rs/` (on by default; `AMDFQ_TAIL=0` disables it). `tail::after_malloc`,
`tail::unprotect`, `tail::reprotect`, `address_backed` and `protect_page` are the whole path;
`hooks.rs`'s `hipMalloc` calls the first, its `hipFree` brackets the real free with the other two. The
original C sources (`amdfq/amdfq-tail/amdfq_tail.c`) are what the heartbeat counters below were taken
with: examined, ends already backed, unaligned (an end that does not start a page of its own), pages
protected (and how many of those came after a free), released, protected now, re-taken with a pad (and
failures), and reserve/map/unmap failures plus hint misses. The pad fallback is the compile-time
constant 16. Route 1's allocator, the shadow, and the hint-miss `/proc/self/maps` dump that diagnosed
them stay in the backup at `~/Desktop/amdfq/`.

### 13.3 What it costs, measured

Probe (`vmm_probe.c`), and the same numbers §12.1 quotes for route 1's layout:

| | §10.1's pad-16 | route 2, one shared guard |
| --- | --- | --- |
| VRAM for 1,000 × 2 MiB blocks | 4,000 MiB | **2,014 MiB** (1,000 blocks + the one shared guard) |
| VRAM for 400 protected blocks | — | **+1.6 MiB total** |
| 12,000 protected pages | — | **4 MiB total** |
| one allocation / one free | 7.6 µs / 15.2 µs | 18.6 µs / 15.2 µs |

The tail run of 2026-09-17 (240 s, reached **step 150**, 112,688 allocations, the same allocation sequence
as the pad-16 run of the same length) against that run, both sampled at the same live set:

| (MiB) | pad 16 | tail guard | difference |
| --- | --- | --- | --- |
| live blocks at the sample | 856 | 856 | — |
| `gran` = Σ round_up(request, 2 MiB) | 7,466 | 7,466 | 0 |
| `rgran` = Σ round_up(runtime extent, 2 MiB) | 9,166 | **7,468** | **−1,698** |
| the runtime's own ledger | 9,799 | **8,103** | **−1,696** |
| driver counter | 9,787 | 10,178 | **+391** |
| counter − ledger | −12 | **+2,075** | — |

The last two rows are the trap that made an earlier reading of this table wrong: at the same live set the
tail run's *counter* is 391 MiB **higher**, which looks like "the guard is not free after all". §11.5
explains it — a run that does not spend a granule per block keeps a pool of driver-charged,
still-recyclable memory, here `counter − ledger = +2,075 MiB` (`base`: +2,454; pad-16: −12), and §11.4
`r0` shows that kind of holding being handed straight back to its own client. The guard's *own* physical
cost is the 2 MiB by which `rgran` exceeds `gran`.

Heartbeat from that run: 56,000 allocations examined, **55,902 pages protected**, 54,919 released, 983
live, 98 re-taken with the pad (0.18 %), 0 reserve / map / unmap failures.

### 13.4 Conclusion: route 2 is worth ~2 GiB, and the memory is real

- **The saving versus the unoptimised state** — §10.1's pad-16, which is what the object has carried since
  the fix — is one granule per granule-aligned live block: **1,986 MiB** for 1,000 blocks (the probe's own
  4,000 → 2,014 MiB), **1,698 MiB** at the tail run's 856 live blocks, and **2,426.0 MiB** at the step-5
  live set of 1,213 aligned blocks (§11.2's ledger law, four pads, exact to the byte).
- **It is real VRAM, not an accounting artifact.** §11.5 counts the objects: the pad's granules are real
  BOs in the trainer's VM (872 × 2 MiB become 872 × 4 MiB, and every other granule-aligned size class
  shifts up by exactly one granule), while the ~2.1 GiB the unpadded accounting keeps instead is a pool of
  ~1,055 real 2 MiB BOs — real to the driver, handed back to ROCr's own client on demand (§11.4 `r0`),
  and untouchable by anyone else (the third party got 4,720.0 MiB where the ledger advertised 6,852.0 MiB).
- **Route 2 also removes the fault it was built for**, and does so without route 1's damage: the tail run
  passed step 10 and reached step 150 with a guard in place for 99.8 % of allocations. That is also the
  answer to §12's open question — the VMM calls, and a shared object mapped past live operands, are
  harmless at this scale; what broke route 1 was serving the operands *themselves* out of VMM memory.
- **Where the tree stands.** `amdfq/amdfq-tail-rs/` is the tail guard of this section, on by default.
  `bash amdfq/amdfq-tail-rs/run.sh test` / `train` (or
  `LD_PRELOAD=amdfq/amdfq-tail-rs/target/release/libamdfq_tail_rs.so`) is the interface; `start_train.sh`
  does not preload it. The pad of §10.1 remains as the fallback for ~0.2 % of allocations, and
  `AMDFQ_TAIL=0` restores an unpadded transcript so the original step-10 fault can be reproduced.
  The original C tree remains at `amdfq/amdfq-tail/`.

**Still open for route 2.** One 240 s run, one seed, no repeat and no long run — the same caveat §10.2
carries. The pad fallback and the "hint missed" path are known only by their counters. And the ~2.1 GiB
pool that the unpadded accounting keeps is real memory held by the runtime for its own reuse: on a 16 GB
card that is headroom the run cannot lend to anything else, which is a reason to think about route 2 and
about what that pool is for.

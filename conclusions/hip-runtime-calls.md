# HIP runtime APIs the trainer actually calls

**What this is.** The HIP C API the training process (`start_train.sh` → `python -u trainer/main.py`)
was observed to enter, and what those entries mean on this box's ROCm 10.0 stack. Only the
functions a Frida intercept recorded as having a non-zero count are listed. Symbols the process
*links* but never called — hipBLAS/hipFFT/hipSOLVER/hipSPARSE family APIs, VMM
(`hipMemAddressReserve` / `hipMemCreate` / `hipMemMap` / …), graphs, events, IPC — are out of
scope here.

**Status.** Semantics are taken from the headers this process loads
(`/opt/rocm/include/hip/hip_runtime_api.h`, `hip_ext.h`, `driver_types.h`; HIP 7.15.26333). Counts
and callers are from one instrumented run that died at training step 10 of the documented gfx1201
bf16 over-read (`bf16-kernel-overrun.md`); they describe that run, not a full epoch.

**Environment.**

| Item | Value |
| --- | --- |
| ROCm | 10.0.0 (`/opt/rocm/core/.info/version`) |
| HIP | 7.15.26333 (`HIP_VERSION_*` in `hip_version.h`) |
| torch | `2.13.0+rocm10.0.0`, conda env `axl` |
| GPU | AMD RX 9070 XT, gfx1201 |
| HIP library actually mapped | `_rocm_sdk_core/lib/libamdhip64.so.7` (same soname as `/opt/rocm/lib/libamdhip64.so.7`) |
| Run | pid 17127, `lllj_20260918_073313`, 28.2 s wall, encoding done, training step 10 |

HIP on AMD is implemented by CLR (`hipamd` + `rocclr`) sitting on ROCr (`libhsa-runtime64`). The
C API below is the surface torch / MIOpen / hipBLASLt / AOTriton call; they do **not** talk to
HSA directly from Python.

---

## 1. How the call set was established

Frida 17.7.3 spawned the same interpreter and environment `start_train.sh` `exec`s, attached a
module observer, and hooked every `hip[A-Z]*` export of `libamdhip64.so.7` plus the `hip*` exports
of the eight hip-family libraries the process maps (hiprtc, hipblas, hipblaslt, hipfft, hiprand,
hipsparse, hipsparselt, hipsolver): 2934 functions, 0 attach failures. A snapshot every second
recorded the cumulative count and the first 64 resolved callers of each function.

Three independent instrumented runs agreed on the same 30 names. A fourth spawn with the same
driver but **no** hooks died at the same step of the same Tensile kernel, so the intercept is not
what selected the set.

The first HIP call is `hipGetDeviceCount` at t ≈ 2 s (`starting`). `hipGetDevice` is already in
the thousands by the time encoding starts.

Callers in the tables below are **the first 64 resolved return addresses**, not a distribution.
`Process.findModuleByAddress` is blind for a few seconds after a library maps; a local module
table filled from the observer is what produced the names.

---

## 2. Index of the 30, with counts

Window: process start through training step 10. "Startup" is everything up to the first snapshot
that reports `step >= 1` (t = 17.15 s, 2,388,638 calls). "Per step" is the remaining 9 steps
(11.1 s). Totals 9,460,399.

| Function | Startup | Per step | Total | Share | Header group |
| --- | ---: | ---: | ---: | ---: | --- |
| `hipGetDevice` | 1,448,713 | 558,396 | 6,474,280 | 68.44% | device |
| `hipGetDevicePropertiesR0600` | 738,326 | 135,011 | 1,953,423 | 20.65% | device |
| `hipGetLastError` | 55,621 | 34,957 | 370,231 | 3.91% | error |
| `hipLaunchKernel` | 35,273 | 22,824 | 240,692 | 2.54% | launch |
| `hipDeviceGetAttribute` | 59,030 | 12,765 | 173,915 | 1.84% | device |
| `hipExtModuleLaunchKernel` | 13,032 | 9,141 | 95,297 | 1.01% | launch (AMD ext) |
| `hipDevicePrimaryCtxGetState` | 10,961 | 4,232 | 49,049 | 0.52% | context |
| `hipStreamGetCaptureInfo` | 6,409 | 4,221 | 44,396 | 0.47% | stream capture |
| `hipStreamGetDevice` | 2,893 | 1,915 | 20,130 | 0.21% | stream |
| `hipStreamIsCapturing` | 6,016 | 351 | 9,173 | 0.10% | stream capture |
| `hipModuleLaunchKernel` | 1,142 | 826 | 8,574 | 0.09% | launch |
| `hipMalloc` | 1,643 | 342 | 4,719 | 0.05% | memory |
| `hipMemcpyWithStream` | 4,372 | 4 | 4,408 | 0.05% | copy |
| `hipFree` | 641 | 339 | 3,693 | 0.04% | memory |
| `hipSetDevice` | 577 | 338 | 3,615 | 0.04% | device |
| `hipRuntimeGetVersion` | 2,138 | 0 | 2,138 | 0.02% | init |
| `hipModuleGetFunction` | 500 | 72 | 1,150 | 0.01% | module |
| `hipMemcpyAsync` | 1,124 | 2 | 1,143 | 0.01% | copy |
| `hipMemsetAsync` | 20 | 13 | 135 | — | memset |
| `hipPointerGetAttributes` | 84 | 2 | 102 | — | memory |
| `hipHostMalloc` | 82 | 0 | 84 | — | memory |
| `hipModuleLoadDataEx` | 14 | 0 | 16 | — | module |
| `hipStreamSynchronize` | 1 | 1 | 10 | — | stream |
| `hipMemset` | 6 | 0 | 6 | — | memset |
| `hipModuleLoad` | 5 | 0 | 5 | — | module |
| `hipGetDeviceCount` | 5 | 0 | 5 | — | device |
| `hipModuleLoadData` | 5 | 0 | 5 | — | module |
| `hipInit` | 2 | 0 | 2 | — | init |
| `hipMemGetInfo` | 2 | 0 | 2 | — | memory |
| `hipDeviceGetStreamPriorityRange` | 1 | 0 | 1 | — | stream |

No hipBLAS / hipFFT / hipSOLVER / hipSPARSE *entry point* is in this list. hipBLASLt still
launches kernels: it does so with `hipExtModuleLaunchKernel` against a module it loaded itself,
not through `hipblasLtMatmul`.

The three launch APIs are not interchangeable. They take different handles and different grid
units; mixing them up is a common source of "the kernel ran with the wrong size".

---

## 3. Initialization and version

### `hipInit`

```c
hipError_t hipInit(unsigned int flags);
```

Explicitly initializes the HIP runtime. `flags` must be 0. Most HIP APIs initialize lazily, so
this exists to pin *when* that happens.

The header warns that a process which `fork()`s must not initialize HIP before the fork if the
child will keep running HIP without an immediate `exec()`: the runtime state is not fork-safe.
The trainer's DataLoader workers are a forkserver of the already-initialized interpreter; they
are not supposed to touch the GPU, which is why a HIP abort that skips Python `atexit` leaves
them holding `/dev/kfd` (`trainer/orphans.py`).

Returns `#hipSuccess`, `#hipErrorInvalidValue`.

Observed: 2 calls, both from `libMIOpen.so.1`, both during startup.

### `hipRuntimeGetVersion`

```c
hipError_t hipRuntimeGetVersion(int* runtimeVersion);
```

Writes the HIP runtime version as
`HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH`.
On this box that is `7 * 10000000 + 15 * 100000 + 26333 = 701526333`.

The header's own warning: this number is **not** a CUDA runtime revision, and there is no
mapping between the two. On NVIDIA builds of HIP the same function returns the CUDA runtime
version instead.

Returns `#hipSuccess`, `#hipErrorInvalidValue`.

Observed: 2,138 calls, all during startup, sampled caller `libhipblaslt.so.1`. hipBLASLt uses it
as a one-time capability probe, then never again (0 per training step).

---

## 4. Device and thread-local default device

HIP keeps a **default device per host thread** in TLS. `hipMalloc`, stream/event creation, and
`hipLaunchKernel` (when no stream is given) all read that TLS slot. The default is **not**
process-wide.

### `hipGetDeviceCount`

```c
hipError_t hipGetDeviceCount(int* count);
```

Writes the number of devices that can run compute. Zero devices is `#hipErrorNoDevice`, not a
zeroed `*count` plus success. One or more devices is `#hipSuccess`.

Observed: 5 calls, all startup. First HIP call the trainer makes. Sampled callers
`libc10_hip.so`, `libhipblaslt.so.1`.

### `hipSetDevice`

```c
hipError_t hipSetDevice(int deviceId);
```

Sets the calling thread's default device to `deviceId` in `0 … hipGetDeviceCount()-1`. No
synchronization with the previous or new device; the header calls the overhead "very little"
and recommends calling it at the start of any HIP sequence so thread-pool workers do not inherit
a stale TLS value.

After a successful call from this thread:

- `hipMalloc` allocates on `deviceId`
- streams and events created from this thread belong to `deviceId`
- `hipLaunchKernel` without a stream runs on `deviceId`; a non-null stream uses **that stream's**
  device instead

Returns `#hipSuccess`, `#hipErrorInvalidDevice`, `#hipErrorNoDevice`.

Observed: 3,615 total, ~338/step, sampled caller `libMIOpen.so.1`. MIOpen is the one that
keeps re-asserting the device, not c10.

### `hipGetDevice`

```c
hipError_t hipGetDevice(int* deviceId);
```

Reads the calling thread's current default device into `*deviceId`. This is the hottest function
in the run (68% of all HIP calls, ~558k/step) because c10 asks "which device am I on?" around
every tensor op rather than caching the answer.

Returns `#hipSuccess`, `#hipErrorInvalidDevice`, `#hipErrorInvalidValue`.

Observed: sampled caller `libc10_hip.so`.

### `hipGetDevicePropertiesR0600` / `hipGetDeviceProperties`

```c
#define hipGetDeviceProperties hipGetDevicePropertiesR0600
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);
```

Fills `hipDeviceProp_t` (`hipDeviceProp_tR0600`) for `deviceId`. The `R0600` suffix is the
ROCm 6.0-era struct layout; the public name is a macro onto the versioned ELF export, which is
why Frida sees `hipGetDevicePropertiesR0600` and not `hipGetDeviceProperties`.

Documented HIP-Clang bugs in this header, still present in 7.15: `maxThreadsPerMultiProcessor`,
`regsPerBlock`, and `l2CacheSize` are always 0. `major`/`minor` on AMD are not CUDA compute
capability — the comment uses Navi III / RDNA3 as the example for `major == 11`. Portable
feature tests are the `hipDeviceArch_t` flags and `hipDeviceGetAttribute`, not these two
integers.

Returns `#hipSuccess`, `#hipErrorInvalidDevice`.

Observed: 1.95 M calls, ~135k/step. Sampled callers `libhipblaslt.so.1` (majority of the first
64), `libtorch_hip.so`. hipBLASLt re-queries the full property block rather than caching it.

### `hipDeviceGetAttribute`

```c
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
```

One integer attribute, not the whole property struct. `hipDeviceAttribute_t` is a large enum
(CUDA-compatible names first, then AMD-only). Typical queries: clock rate, max block/grid dims,
shared memory, cooperative launch, whether host pointers can be mapped.

Returns `#hipSuccess`, `#hipErrorInvalidDevice`, `#hipErrorInvalidValue`.

Observed: ~12.8k/step. Sampled callers `libhipblaslt.so.1`, `liborigami.so.1` (origami is the
hipBLASLt kernel-selection helper).

### `hipDevicePrimaryCtxGetState`

```c
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);
```

Driver-API leftover: writes the primary context flags and whether it is active (`0` inactive,
`1` active). **Deprecated on AMD** — the header says it exists for cuCtx compatibility on
NVIDIA; on HIP/HIP-Clang the matching `hipDevicePrimaryCtxRelease` is documented to return
success without actually releasing the primary context.

Returns `#hipSuccess`.

Observed: ~4.2k/step, sampled caller `libtorch_hip.so`. Torch is polling "is the primary context
up?" around launches.

---

## 5. Error handling

### `hipGetLastError`

```c
hipError_t hipGetLastError(void);
```

Returns the last error any HIP runtime call stored **on this host thread**, then resets the
stored error to `#hipSuccess`. `hipPeekAtLastError` is the non-resetting twin; it was never
called.

There is no argument. The return value *is* the sticky error, not a status-of-the-query.

Observed: ~35k/step, sampled caller `libc10_hip.so`. c10 drains the sticky error after almost
every launch/copy so a later call cannot see a stale code. Combined with `hipGetDevice` this is
why "device query" is 95% of the intercept volume and is not evidence of 35k real failures per
step.

---

## 6. Memory

All of these operate against the calling thread's default device unless a pointer's own device
is implied (pointer attributes, free of a device pointer).

### `hipMalloc`

```c
hipError_t hipMalloc(void** ptr, size_t size);
```

Device allocation on the default device. Size 0: no allocation, `*ptr = nullptr`, `#hipSuccess`.
Failure: `#hipErrorOutOfMemory` or `#hipErrorInvalidValue` (bad context, null `ptr`).

This is the synchronous, default-stream allocation. The async / pool variants
(`hipMallocAsync`, `hipMallocFromPoolAsync`, VMM `hipMemCreate`+`hipMemMap`) are referenced by
`libc10_hip` / `libtorch_hip` / `libtorch_rocshmem` / `librccl` and were **not** called. The
caching allocator in this run is therefore the classical `hipMalloc`/`hipFree` pair plus
torch's own reuse, not HIP memory pools.

Observed: 4,719 total, **342/step**. Sampled caller `libc10_hip.so`.

### `hipFree`

```c
hipError_t hipFree(void* ptr);
```

Frees a `hipMalloc` (HIP-Clang) allocation. **Implicit `hipDeviceSynchronize()`** before the
free — the header is explicit. `ptr == NULL` is success and still initializes the runtime.
Passing a `hipHostMalloc` pointer is `#hipErrorInvalidDevicePointer`.

Observed: 3,693 total, **339/step**, sampled caller `libc10_hip.so`. Per-step malloc/free are
almost paired (342 vs 339); the caching allocator is absorbing nearly all tensor traffic, which
is why a 4 GiB plateau does not produce millions of `hipMalloc`s.

### `hipHostMalloc`

```c
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
```

Page-locked (pinned) host memory, mapped into every GPU's address space on the system. The
runtime tracks these allocations and can skip setup that pageable `malloc` memory needs, which
is why H2D/D2H of pinned buffers is the fast path. Granularity is 4 KiB. Size 0: same as
`hipMalloc` (null, success).

Flags (from the same header):

| Flag | Value | Meaning on AMD |
| --- | ---: | --- |
| `hipHostMallocDefault` | 0x0 | Default pinned |
| `hipHostMallocPortable` | 0x1 | Visible to all contexts |
| `hipHostMallocMapped` | 0x2 | Also mapped into the current device; device pointer via `hipHostGetDevicePointer` |
| write-combined | 0x4 | **CUDA source compatibility only; not functional on AMD** |

Zero-copy (device loads host memory over the interconnect, no explicit copy) is the
infrequent-access case the header recommends coherent mapped allocations for.

Observed: 84 calls, all startup, sampled caller `libtorch_hip.so`. No per-step host pins — torch
reuses the ones it created while loading.

### `hipPointerGetAttributes`

```c
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
```

Classifies `ptr`. `attributes->type` is a `hipMemoryType` (`Host` / `Device` / `Managed` /
`Array` / `Unified`). Unrecognized types return `#hipErrorInvalidValue` rather than a new enum
value, to keep ABI. The header notes the behaviour matches CUDA **before** 11.0.

Also returns `device`, `devicePointer`, `hostPointer`, `isManaged`, `allocationFlags`.

Observed: 102 total, ~2/step, sampled caller `libtorch_hip.so`. Used to decide "is this already
on device?" before a copy.

### `hipMemGetInfo`

```c
hipError_t hipMemGetInfo(size_t* free, size_t* total);
```

Free and total **allocatable** bytes on the **current** device. On ROCm this is the actual free
memory left, so it is honest under multiple processes / threads / GPUs. (The Windows path is
documented as process-local and optimistic.)

Returns `#hipSuccess`, `#hipErrorInvalidDevice`, `#hipErrorInvalidValue`.

Observed: 2 calls, both startup, sampled caller `libMIOpen.so.1`. Not on the training hot path;
torch does not poll VRAM through this.

---

## 7. Copies and fills

`hipMemcpyKind` (`driver_types.h`):

| Kind | Value |
| --- | ---: |
| `hipMemcpyHostToHost` | 0 |
| `hipMemcpyHostToDevice` | 1 |
| `hipMemcpyDeviceToHost` | 2 |
| `hipMemcpyDeviceToDevice` | 3 |
| `hipMemcpyDefault` | 4 (infer from virtual addresses) |
| `hipMemcpyDeviceToDeviceNoCU` | 1024 (D2D without compute units) |

`kind` that does not match the pointer kinds is undefined behaviour (`hipMemcpy` docs). The
async variants add: if host memory is **not** pinned, the copy is performed **synchronously**
anyway.

### `hipMemcpyWithStream`

```c
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream);
```

A copy bound to `stream`, documented as **`hipMemcpyAsync` + `hipStreamSynchronize`**: the host
does not return until that copy is done. Because it is host-synchronous it is **illegal during
graph capture**.

Observed: 4,408 total, **4,372 of them during startup**, 4/step afterwards. Sampled caller
`libtorch_hip.so`. Weight load / latent cache populate is this API; the train loop barely copies.

### `hipMemcpyAsync`

```c
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream /* default 0 */);
```

Enqueue a copy on `stream` and return. The copy is always performed by the device attached to
that stream. For peer copies the header recommends a stream on the **source** device and
`hipDeviceEnablePeerAccess`; without peer access the runtime still copies, via a host staging
buffer.

Unpinned host pointers → the copy degrades to synchronous. `hipHostMalloc` is the documented
way to keep it async.

Observed: 1,143 total, ~2/step, sampled caller `libtorch_hip.so`. Almost entirely startup.

The blocking `hipMemcpy` (no stream) was never called.

### `hipMemset`

```c
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
```

Synchronous fill of `sizeBytes` with the constant **byte** `value` (the `int` is truncated to a
byte, CUDA-style). Host-blocking.

Observed: 6 calls, all startup, sampled caller `libhipblaslt.so.1`.

### `hipMemsetAsync`

```c
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream /* 0 */);
```

Same fill, queued on `stream`. Returns before the fill completes; a non-zero stream may overlap
with other streams.

Observed: 135 total, ~13/step. Sampled callers `libMIOpenCKGroupedConv_gfx1201.so` (majority of
the first 64), `libtorch_hip.so`. This is the gfx1201 grouped-conv code object zeroing workspace,
not the trainer itself.

The width-specific variants (`hipMemsetD8/D16/D32[Async]`) were not called.

---

## 8. Streams

A `hipStream_t` of `0` is the null / default stream, with the usual "wait for all other streams
on this device" semantics on synchronizing calls.

### `hipStreamSynchronize`

```c
hipError_t hipStreamSynchronize(hipStream_t stream);
```

Host blocks until every command on `stream` (and its device) is done. Host-synchronous. The
stream already knows its device — no `hipSetDevice` needed beforehand. Null stream: wait for
**other** streams on the same device as well. Honours `hipDeviceScheduleBlockingSync` (active
spin vs blocking wait).

Returns `#hipSuccess`, `#hipErrorInvalidHandle`.

Observed: 10 total, ~1/step, sampled caller `libtorch_hip.so`. Torch almost never host-syncs on
the train path; the implicit sync inside `hipFree` and `hipMemcpyWithStream` is doing that work
instead.

### `hipStreamGetDevice`

```c
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device);
```

Writes the device the stream was created against. `hipDevice_t` here is the driver-style handle,
not the ordinal `hipSetDevice` takes.

Returns `#hipSuccess`, `#hipErrorInvalidValue`, `#hipErrorContextIsDestroyed`,
`#hipErrorInvalidHandle`, `#hipErrorNotInitialized`, `#hipErrorDeinitialized`,
`#hipErrorInvalidContext`.

Observed: ~1.9k/step, sampled caller `libaotriton_v2.so`. AOTriton (flash-attn) checks which
device a stream is on before launching.

### `hipDeviceGetStreamPriorityRange`

```c
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
```

Writes the inclusive numeric range of stream priorities. **Lower number = higher priority.**
Priorities passed to `hipStreamCreateWithPriority` outside the range are clamped.

**On AMD this API is under development and currently just returns `#hipSuccess`.** The values
written are not a real gfx1201 priority range.

Observed: 1 call, startup, sampled caller `libc10_hip.so`.

### `hipStreamIsCapturing`

```c
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
```

Writes whether `stream` is in a graph-capture sequence:

| Status | Meaning |
| --- | --- |
| `hipStreamCaptureStatusNone` | not capturing |
| `hipStreamCaptureStatusActive` | capturing |
| `hipStreamCaptureStatusInvalidated` | capture sequence broken, not yet ended |

Returns `#hipSuccess`, `#hipErrorInvalidValue`, `#hipErrorStreamCaptureImplicit`.

Libraries call this before doing something illegal during capture (the header flags
`hipMemcpyWithStream` as one such thing). The trainer never starts a capture
(`hipStreamBeginCapture` was not called), so the answer is always "none"; the call is still
made.

Observed: 9,173 total, 351/step. Sampled callers `libc10_hip.so`, `libtorch_hip.so`.

### `hipStreamGetCaptureInfo`

```c
hipError_t hipStreamGetCaptureInfo(hipStream_t stream,
                                   hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId);
```

Same status, plus a unique capture id. `_v2` also returns the graph and its dependency nodes;
it was not called.

Returns `#hipSuccess`, `#hipErrorStreamCaptureImplicit`.

Observed: ~4.2k/step, sampled caller `libtorch_hip.so`. Same "am I capturing?" check as above,
with an id.

No `hipStreamCreate` / `Destroy` / `WaitEvent` / `Query` appears in the set: the streams in use
are the ones torch created earlier (or the null stream), and they outlive this 10-step window.

---

## 9. Modules: loading a code object and taking a function out of it

A HIP **module** is a loaded GPU code object (a file, a fatbin in host memory, or a raw image).
A **function** (`hipFunction_t`) is one kernel extracted from it. This is the driver-style path;
the runtime-style path (`hipLaunchKernel`) takes a host stub pointer instead and does not show
up as `hipModule*`.

Resources from `hipModuleLoad*` are released only by `hipModuleUnload`. `hipModuleUnload` was
**not** called in this window — every module the run loaded is still live at the fault.

### `hipModuleLoad`

```c
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
```

Load a code-object **file** into a module on the current context.

Returns `#hipSuccess`, `#hipErrorInvalidValue`, `#hipErrorInvalidContext`, `#hipErrorFileNotFound`,
`#hipErrorOutOfMemory`, `#hipErrorSharedObjectInitFailed`, `#hipErrorNotInitialized`.

Observed: 5 calls, all startup, sampled caller `libhipblaslt.so.1`.

### `hipModuleLoadData`

```c
hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
```

Same, but `image` is a pointer to a code object or fatbin already in host memory. Default
fatbin production is `amdclang++ -O3 -c --offload-device-only --offload-arch=<GPU_ARCH> …`.

Observed: 5 calls, all startup, sampled caller `libMIOpen.so.1`. MIOpen's gfx1201 grouped-conv
code object (`libMIOpenCKGroupedConv_gfx1201.so`) is loaded this way.

### `hipModuleLoadDataEx`

```c
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                               unsigned int numOptions, hipJitOption* options,
                               void** optionValues);
```

Documented as: options are **not used**; this calls `hipModuleLoadData`. The extra arguments
exist for CUDA source compatibility (cuModuleLoadDataEx JIT options).

Observed: 16 calls, all startup, sampled caller `libaotriton_v2.so`. AOTriton ships
precompiled kernels and loads them as in-memory images.

### `hipModuleGetFunction`

```c
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
```

Looks up kernel `kname` in `module`. `#hipErrorNotFound` if the name is missing.

Observed: 1,150 total, ~72/step. Sampled callers `libhipblaslt.so.1`, `libaotriton_v2.so`,
`libMIOpen.so.1`. The per-step traffic is hipBLASLt resolving GEMM kernels as shapes change
(different buckets), not a reload of the module.

---

## 10. Kernel launch — three APIs, three contracts

HIP does not allow a launch whose per-dimension work-item count `gridDim.[xyz] * blockDim.[xyz]`
is `>= 2^32`.

### `hipLaunchKernel`

```c
hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                           void** args, size_t sharedMemBytes, hipStream_t stream);
```

**Runtime** launch. `function_address` is the host stub of a `__global__` compiled into the
calling library (torch's own kernels). `numBlocks` is the **grid in blocks**; `dimBlocks` is
threads per block. `args` is an array of **pointers to** the kernel arguments, one pointer per
parameter. `stream == 0` uses the null stream.

This is what `hipLaunchKernelGGL` in `amd_hip_runtime.h` lowers to.

Returns `#hipSuccess`, `#hipErrorInvalidValue`.

Observed: 240,692 total, **~22.8k/step**. Sampled caller `libtorch_hip.so`. This is torch's own
elementwise / reduction / autograd kernels, not hipBLASLt.

### `hipModuleLaunchKernel`

```c
hipError_t hipModuleLaunchKernel(hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra);
```

**Driver** launch of a `hipFunction_t` from a module. Grid is in **blocks**, same as
`hipLaunchKernel`. Arguments are either `kernelParams` (array of pointers, like
`hipLaunchKernel`) **or** `extra` (a packed, naturally-aligned blob whose address is a multiple
of each argument's size). The header points at `hip_porting_driver_api.md` for the `extra`
layout.

`stream == 0` → null stream.

Returns `#hipSuccess`, `#hipErrorNotInitialized`, `#hipErrorInvalidValue`.

Observed: 8,574 total, **~826/step**. Sampled caller `libaotriton_v2.so`. Flash-attn on this
stack.

### `hipExtModuleLaunchKernel`

```c
hipError_t hipExtModuleLaunchKernel(hipFunction_t f,
    uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
    uint32_t localWorkSizeX,  uint32_t localWorkSizeY,  uint32_t localWorkSizeZ,
    size_t sharedMemBytes, hipStream_t hStream,
    void** kernelParams, void** extra,
    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags);
```

AMD extension (`hip_ext.h`). Same `hipFunction_t` as `hipModuleLaunchKernel`, but the grid is
specified in **work-items** (`globalWorkSize*`), not in blocks. `localWorkSize*` is the block
(work-group) size; the runtime divides. This is the OpenCL/HSA shape, which is why hipBLASLt
and MIOpen prefer it — their kernel metadata is in work-items.

`startEvent` / `stopEvent`, if non-null, must already have been created. The header's AMD-specific
timing caveat: HIP actually updates the **start** event when the kernel **completes**, and the
interval does **not** include the system-scope release / cache flush, only the time to issue
writes to cache. Do not treat those events as CUDA-accurate start/stop.

`flags`: `hipExtAnyOrderLaunch` (0x01) = kernel may run in any order relative to others. **Not
supported on GFX9xx**; gfx1201 is RDNA4, so the restriction does not apply here, but nothing in
the intercept records the flag.

Returns `#hipSuccess`, `#hipInvalidDeviceId`, `#hipErrorNotInitialized`, `#hipErrorInvalidValue`.

`hipHccModuleLaunchKernel` is the deprecated spelling of the same function; it was not called.

Observed: 95,297 total, **~9.1k/step**. Sampled caller `libhipblaslt.so.1`. This is the launch
path of the Tensile bf16 GEMM that dies
(`Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_…_ISA1201` in
`bf16-kernel-overrun.md`). The public `hipblasLtMatmul` symbol is never entered; hipBLASLt
loads the `.co`, `hipModuleGetFunction`s the kernel, and fires it here.

---

## 11. What this set implies for the trainer

1. **Almost all HIP traffic is not GPU work.** 95% of calls are `hipGetDevice` /
   `hipGetDevicePropertiesR0600` / `hipGetLastError` / `hipDeviceGetAttribute` /
   `hipDevicePrimaryCtxGetState` / capture queries. A hook that is expensive on those four will
   dominate a trace even if it is cheap on `hipLaunchKernel`.
2. **The allocator you would intercept to pad GEMM operands is `hipMalloc`/`hipFree`**, ~340
   pairs per step, via `libc10_hip.so`. VMM and `hipMallocAsync` are unused on this run, so a
   hipMem* interceptor would see nothing. `hipFree`'s implicit device-wide sync is the hidden
   cost of those 339 frees.
3. **Copies are a load-time event.** `hipMemcpyWithStream` + `hipMemcpyAsync` together are ~5.5 k
   calls, 99% before step 1. A train-step intercept of memcpy will not see the weight load.
4. **Three launchers, three libraries.** torch kernels → `hipLaunchKernel`; AOTriton →
   `hipModuleLaunchKernel`; hipBLASLt (and MIOpen's ext path) → `hipExtModuleLaunchKernel`. The
   faulting GEMM is the third. Grid units differ (blocks vs work-items); an interceptor that
   wants the launch geometry has to decode the right API.
5. **Modules are loaded once and never unloaded** in this window. A code-object swap (the
   hip1-style "load a guarded clone") has to happen at `hipModuleLoad` / `LoadData` /
   `LoadDataEx`, not at launch, unless it also wraps `hipModuleGetFunction`.
6. **Stream capture is queried continuously and never started.** Anything that is illegal during
   capture (`hipMemcpyWithStream`) is still safe here; the queries are defensive.

---

## 12. `trainer/main.py` vs `test/torch-test.py`

The same agent (2934 `hip*` entry points, module observer, caller sampling) was pointed at
`python -u test/torch-test.py`, and the trainer's run was re-used as the other column. The two are
not the same size of workload, so every row is given as a total **and** as a rate per unit of work:
one training step for the trainer, one stress iteration for the script.

**How the per-iteration column was obtained.** `torch-test.py --time-scale N` runs `5N`
iterations of the same loop, so the loop's own traffic is the difference between two scales:

| Run | Iterations | Calls (last poll) |
| --- | ---: | ---: |
| `--time-scale 0` (checks only, loop and verification skipped) | 0 | 7,118 |
| `--time-scale 1` | 5 | 112,051 |
| `--time-scale 2` | 10 | 145,086 |
| `--time-scale 4` | 20 | 211,140 |

`(211,140 − 112,051) / 15 = 6,606` calls per iteration, and `(145,086 − 112,051) / 5 = 6,607` —
the two independent differences agree to one call in 6,600. The part that does not scale with the
loop (checks + plateau allocation + the whole verification pass) is
`112,051 − 5 × 6,606 = 79,021`.

**Run-level comparison.**

| | trainer (`lllj_20260918_073313`) | `test/torch-test.py --time-scale 1` |
| --- | --- | --- |
| Counted window | 28.2 s, 10 training steps | 3.2 s, 5 iterations + verification |
| HIP calls in that window | 9,460,399 | 112,051 |
| Distinct `hip*` functions called | 30 | 24 |
| Calls/s | 335,119 | 34,691 |
| One-time part (init, weights, cache check / checks + plateau + verification) | 2,388,638 (25.2%) | 79,021 (70.5%) |
| Steady-state part | 785,751 per step | 6,606 per iteration |
| Ratio of the two steady states | 1 step = 119 script iterations | |
| Environment | `start_train.sh` exports | same exports (re-run without them differs by 5 calls in 112 k) |
| Process lifetime | died at 35.2 s in the gfx1201 fault | exited normally at 7.4 s |

**Per API.** "—" means the function was not called by that target at all.

| API | trainer total | share | trainer /step | torch-test total | share | torch-test /iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hipGetDevice` | 6,474,280 | 68.44% | 558,396 | 43,422 | 38.75% | 5,024 |
| `hipGetDevicePropertiesR0600` | 1,953,423 | 20.65% | 135,011 | 15,174 | 13.54% | 40 |
| `hipGetLastError` | 370,231 | 3.91% | 34,957 | 1,627 | 1.45% | 252 |
| `hipLaunchKernel` | 240,692 | 2.54% | 22,824 | 1,442 | 1.29% | 235 |
| `hipDeviceGetAttribute` | 173,915 | 1.84% | 12,765 | 29,815 | 26.61% | 4 |
| `hipExtModuleLaunchKernel` | 95,297 | 1.01% | 9,141 | 59 | 0.05% | 9 |
| `hipDevicePrimaryCtxGetState` | 49,049 | 0.52% | 4,232 | 1,425 | 1.27% | 274 |
| `hipStreamGetCaptureInfo` | 44,396 | 0.47% | 4,221 | 26 | 0.02% | 0 |
| `hipStreamGetDevice` | 20,130 | 0.21% | 1,915 | — | — | — |
| `hipStreamIsCapturing` | 9,173 | 0.10% | 351 | 1,491 | 1.33% | 285 |
| `hipModuleLaunchKernel` | 8,574 | 0.09% | 826 | — | — | — |
| `hipMalloc` | 4,719 | 0.05% | 342 | 110 | 0.10% | 12 |
| `hipMemcpyWithStream` | 4,408 | 0.05% | 4 | 1,383 | 1.23% | 273 |
| `hipFree` | 3,693 | 0.04% | 339 | 106 | 0.09% | 12 |
| `hipSetDevice` | 3,615 | 0.04% | 338 | — | — | — |
| `hipRuntimeGetVersion` | 2,138 | 0.02% | 0 | 14,893 | 13.29% | 0 |
| `hipModuleGetFunction` | 1,150 | 0.01% | 72 | 27 | 0.02% | 0 |
| `hipMemcpyAsync` | 1,143 | 0.01% | 2 | 8 | 0.01% | 1 |
| `hipMemsetAsync` | 135 | 0.00% | 13 | 494 | 0.44% | 86 |
| `hipPointerGetAttributes` | 102 | 0.00% | 2 | — | — | — |
| `hipHostMalloc` | 84 | 0.00% | 0 | 4 | 0.00% | 0 |
| `hipModuleLoadDataEx` | 16 | 0.00% | 0 | — | — | — |
| `hipStreamSynchronize` | 10 | 0.00% | 1 | 524 | 0.47% | 98 |
| `hipMemset` | 6 | 0.00% | 0 | 2 | 0.00% | 0 |
| `hipModuleLoadData` | 5 | 0.00% | 0 | — | — | — |
| `hipModuleLoad` | 5 | 0.00% | 0 | 6 | 0.01% | 0 |
| `hipGetDeviceCount` | 5 | 0.00% | 0 | 5 | 0.00% | 0 |
| `hipInit` | 2 | 0.00% | 0 | — | — | — |
| `hipMemGetInfo` | 2 | 0.00% | 0 | — | — | — |
| `hipDeviceGetStreamPriorityRange` | 1 | 0.00% | 0 | 1 | 0.00% | 0 |
| `hipModuleUnload` | — | — | — | 1 | 0.00% | 0 |
| `hipDeviceSynchronize` | — | — | — | 6 | 0.01% | 1 |

**By category.**

| Category | trainer total | share | trainer /step | torch-test total | share | torch-test /iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| device query | 8,977,712 | 94.90% | 741,468 | 104,943 | 93.66% | 5,321 |
| kernel launch | 344,563 | 3.64% | 32,791 | 1,501 | 1.34% | 244 |
| stream/event | 73,709 | 0.78% | 6,488 | 2,041 | 1.82% | 383 |
| other (`PrimaryCtxGetState`) | 49,051 | 0.52% | 4,232 | 1,425 | 1.27% | 274 |
| alloc/free | 8,496 | 0.09% | 681 | 220 | 0.20% | 24 |
| copy | 5,551 | 0.06% | 6 | 1,391 | 1.24% | 274 |
| module/loader | 1,176 | 0.01% | 72 | 34 | 0.03% | 0 |
| memset | 141 | 0.00% | 13 | 496 | 0.44% | 86 |

### What the comparison shows

1. **The shape is the same, the scale is not.** Both are ~94 % device queries and under 4 %
   kernel launches. Everything that differs is a difference of proportion, not of kind: the two
   targets call the same 22 functions and each adds a handful the other does not.
2. **A training step is ~119 script iterations.** The script's loop is 6.6 k HIP calls against the
   trainer's 786 k per step. The trainer's steady state is ~24× the script's *whole* 3.2 s run.
3. **The script is transfer-shaped; the trainer is GEMM-shaped.**
   - `hipMemcpyWithStream`: 273 per iteration for the script, against 4 per step for the trainer.
     Copy is 1.24 % of the script's traffic and 0.06 % of the trainer's. The count is the script's
     own hop list: 4 reps × 2 payload hops + 128 × 2 small hops + the four in-chain round trips
     (plateau slice, scan-slice bias, bf16 operand, `w3.grad`) × 2 = 272, measured 273 — **each
     `.to()` costs exactly one `hipMemcpyWithStream`**, so the script's transfer volume is real at
     the HIP level.
   - `hipExtModuleLaunchKernel`: 9,141 per step for the trainer (hipBLASLt firing Tensile GEMMs),
     9 per iteration for the script. The script's matmuls are five `torch.mm` calls per iteration.
   - `hipDeviceGetAttribute` is the mirror image: 29,815 in the script but only 4 per iteration —
     it is a torch *startup* fingerprint (the same 2:1 ratio against `hipRuntimeGetVersion` shows
     up in a trivial `torch.randn` probe), while the trainer pays 12,765 *per step* because
     hipBLASLt/origami re-runs kernel selection as buckets change shape.
4. **Per-op overhead is identical, so per-unit cost tracks op count.** `hipGetDevice`,
   `hipGetLastError`, `hipStreamIsCapturing`, `hipStreamGetCaptureInfo`,
   `hipDevicePrimaryCtxGetState` all appear at a few hundred calls per iteration in the script and
   scale up with the trainer's much larger op count. 5,024 `hipGetDevice` per iteration for a loop
   of ~50 torch ops + 264 transfers + 96 `.item()` reductions is ~13 per operation; the trainer's
   558,396 per step is the same discipline applied to a full UNet+TE step.
5. **The script's alloc/free churn reaches the driver only partly.** `churn()` performs 96
   allocations and ~88 frees per iteration, and then `empty_cache()`. The driver sees
   **12 `hipMalloc` and 12 `hipFree` per iteration**. The reason is not the allocator hiding
   everything: `CHURN_LIVE = 8` means at most 8 blocks are ever live at once, so 8–12 allocations
   are all the loop can need from a cold pool — the other ~84 requests are satisfied from blocks
   freed moments earlier. Removing the launcher's `PYTORCH_CUDA_ALLOC_CONF` changes the per-iteration
   rate by 0 (`12` either way).
6. **The script's pinned-memory churn does not reach HIP at all.** It builds 128 pinned tensors per
   iteration, and `hipHostMalloc` is called 4 times in the entire process. torch's pinned-memory
   cache serves the rest. The *transfers* still reach HIP (point 3); the pinned *allocations* do
   not.
7. **`hipStreamSynchronize` per iteration = 98** — the 96 `.item()` calls in `churn()`
   (`float(block.sum().item())`), one implicit sync each, plus one per `torch.cuda.synchronize()`.
   The trainer has 1 per step. This is the clearest single illustration of the two workloads'
   difference: the script's cost is in moving and reading data, the trainer's is in launching.
8. **Only-in-one functions are all startup or shutdown artefacts.** The trainer's exclusives are
   `hipSetDevice` (MIOpen re-asserting the device), `hipStreamGetDevice` (AOTriton),
   `hipModuleLaunchKernel` (AOTriton), `hipPointerGetAttributes`, `hipInit`, `hipMemGetInfo`,
   `hipModuleLoadData(Ex)`. The script's exclusives are `hipDeviceSynchronize` (its explicit
   `torch.cuda.synchronize()`) and `hipModuleUnload` (a clean exit; the trainer died before it
   could unload anything).

### Caveats

- **The exit tail is not in either count.** The agent's RPC stops being served while the target's
  interpreter tears down; the polls resume only as the process dies. For the script that window is
  4.1 s (the script's own work ends at 3.2 s, the process exits at 7.4 s) and for the trainer 7.0 s
  (its step 10 ends at 28.2 s, the process dies at 35.2 s in the fault handler). A control run with
  **zero hooks** blocks in exactly the same window, so this is the target's shutdown parking the
  agent, not the instrumentation. Everything the *scripts themselves* do is inside the counts; what
  is missing is whatever the runtime asks of HIP on the way out.
- The per-iteration column is a difference of two runs, so it is only as good as the assumption
  that the non-loop part is scale-independent — the agreement between the `/5` and `/15` estimates
  (6,607 vs 6,606) is the evidence for that.
- Caller attribution is the first resolved samples per function, not a distribution.

---

## 13. Sources

- `/opt/rocm/include/hip/hip_runtime_api.h` (HIP 7.15.26333) — every signature and comment above
  that is not in `hip_ext.h` / `driver_types.h`
- `/opt/rocm/include/hip/hip_ext.h` — `hipExtModuleLaunchKernel`
- `/opt/rocm/include/hip/driver_types.h` — `hipMemcpyKind`
- `/opt/rocm/include/hip/hip_version.h`, `/opt/rocm/core/.info/version`
- Instrument: `/tmp/hipfrida/` (`hip_hook.js`, `driver.py`, `compare.py`, `timeline_final.json`
  for the trainer, `timeline_torchtest.json` + `timeline_tt_s{0,2,4}.json` for the script); method
  in the session that produced this file, 2026-09-18

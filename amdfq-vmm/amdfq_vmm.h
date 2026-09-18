/*
 * The VMM route (amdfq.md §12, its per-allocation layout): hipMalloc's requests are served from an
 * address range we reserve and populate ourselves, so the slack behind a block costs *one* physical
 * granule for the whole process instead of one granule of VRAM per live allocation — which is what
 * raising every request by 16 bytes costs, once the runtime rounds that up to its granule.
 *
 * The layout of one served allocation:
 *
 *     va                    va + block                     va + block + pad
 *     |---- block ----|---- pad (shared handle) ----|
 *
 * `block` is the request rounded up to the mapping granularity, mapped from a handle created for it;
 * `pad` is mapped from a single handle created once per process and mapped behind every block, so a
 * kernel reading past the end of its operand (conclusions/bf16-kernel-overrun.md) lands in memory
 * that exists. Measured 2026-09-17 (amdfq-vmm/vmm_probe.c): one handle maps at many addresses at once,
 * 1000 mappings of one pad handle cost the physical 2 MiB once, and a read across the boundary
 * returns rather than faulting.
 *
 * One hipMemAddressReserve per hipMalloc, given back by hipMemAddressFree when the pointer is freed:
 * no arena base, no bump pointer and no free list. That layout is the only one this module builds —
 * the arena and the two modes that served nothing (the shadow, and the tail guard of §13) were
 * removed on 2026-09-18, after a run of them on real hardware (README §11) and a run of them as
 * smoke tests (README §10); route 2's tail guard still lives in ../amdfq/amdfq_tail.c.
 *
 * State is per process: whichever process first calls hipMalloc owns it. A process that inherited
 * another's state through fork never touches it — its allocations go to the real runtime, and a
 * pointer it frees that belongs to its parent is left alone (releasing the parent's handle would
 * free memory the parent is still using).
 */
#ifndef AMDFQ_VMM_H
#define AMDFQ_VMM_H

#include <stddef.h>

/* The HIP types this module needs, copied from the ROCm 10.0.0 headers so this object still builds
 * with no ROCm headers present:
 *   hip/hip_runtime_api.h — hipMemAddressReserve, hipMemCreate, hipMemMap, hipMemUnmap,
 *                           hipMemRelease, hipMemSetAccess, hipMemGetAllocationGranularity
 *   hip/driver_types.h    — hipMemLocation, hipMemAllocationProp, hipMemAccessDesc
 */
typedef int hipError_t;
/* The two hipError_t values this module names. */
#define HIP_SUCCESS 0
#define HIP_ERROR_NOT_FOUND 500
typedef struct ihipMemGenericAllocationHandle *hipMemGenericAllocationHandle_t;

typedef struct {
    int type;
    int id;
} hipMemLocation;

typedef struct {
    int type;
    int requestedHandleTypes;
    hipMemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
    } allocFlags;
} hipMemAllocationProp;

typedef struct {
    hipMemLocation location;
    int flags;
} hipMemAccessDesc;

enum {
    AMDFQ_MEM_ALLOCATION_TYPE_PINNED = 0x1,
    AMDFQ_MEM_HANDLE_TYPE_NONE = 0x0,
    AMDFQ_MEM_LOCATION_DEVICE = 1,
    AMDFQ_MEM_LOCATION_HOST = 2,
    AMDFQ_MEM_ACCESS_PROT_READWRITE = 3,
    AMDFQ_GRANULARITY_MINIMUM = 0x0,
    AMDFQ_GRANULARITY_RECOMMENDED = 0x1,
};

/* What a served allocation looks like to the caller. */
struct amdfq_vmm_result {
    void *va;
};

/* Guard against the runtime re-entering the interposer from inside our own VMM calls: the caller
 * only reaches the module while it holds this, and a nested call is forwarded to the real function. */
int amdfq_vmm_begin(void);
void amdfq_vmm_end(void);

/* 0 = served, `*result` filled. -1 = not served; the caller forwards the call to the runtime. */
int amdfq_vmm_malloc(size_t size, struct amdfq_vmm_result *result);
/* 0 = this module's block and released here (the runtime must not see the pointer). -1 = not ours. */
int amdfq_vmm_free(void *ptr);

/* The real implementation of `name`, via dlsym(RTLD_NEXT): NULL when this runtime lacks it. */
void *amdfq_symbol(const char *name);

#endif /* AMDFQ_VMM_H */

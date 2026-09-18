/*
 * LD_PRELOAD interposers for the allocation gates (amdfq.md §7.1): hipMalloc and hipFree are the
 * whole allocation path of a training run in this configuration, so hooking these two is enough to
 * take over every allocation and release the process makes — the HSA calls §7.2 measures all fire
 * from inside them, and the other gates §7 lists allocate nothing themselves.
 *
 * hipMalloc is served from the VMM route (amdfq.md §12), in its per-allocation layout:
 * amdfq_vmm_malloc reserves an extent per request and maps the block plus one shared pad granule
 * behind it, so the slack a bf16 kernel over-reading its operand walks into costs one granule for
 * the whole process instead of one granule of VRAM per live allocation. What the route declines to
 * serve — a fork child, a symbol this runtime lacks, a failed reserve/create/map — is forwarded to
 * the runtime with the caller's size unchanged: this build carries no second mitigation.
 */
#include "amdfq_vmm.h"

/* Resolve the real implementation once per interposing function. RTLD_NEXT starts its search after
 * this object, so it can never hand back the interposer itself. */
#define AMDFQ_REAL(name)                     \
    static __typeof__(name) *real;           \
    if (real == NULL) real = (__typeof__(name) *)amdfq_symbol(#name);

hipError_t hipMalloc(void **ptr, size_t size) {
    AMDFQ_REAL(hipMalloc);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;

    struct amdfq_vmm_result served;
    int outer = amdfq_vmm_begin();
    hipError_t result;

    if (outer && amdfq_vmm_malloc(size, &served) == 0) {
        *ptr = served.va;
        result = HIP_SUCCESS;
    } else {
        result = real(ptr, size);
    }
    if (outer) amdfq_vmm_end();
    return result;
}

hipError_t hipFree(void *ptr) {
    AMDFQ_REAL(hipFree);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;

    int outer = amdfq_vmm_begin();
    /* A block this module served is unmapped, released and its address given back here; the runtime
     * never saw it. Anything else — a fork child's parent block, a pointer allocated before the
     * route existed — is the runtime's to free. */
    int served = outer && amdfq_vmm_free(ptr) == 0;
    hipError_t result = served ? HIP_SUCCESS : real(ptr);
    if (outer) amdfq_vmm_end();
    return result;
}

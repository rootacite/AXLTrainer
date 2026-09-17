/*
 * LD_PRELOAD interposers for the HIP allocation gates (amdfq.md §7.1): hipMalloc, hipFree and
 * hipHostMalloc are the whole allocation path of a training run in this configuration, so hooking
 * these three is enough to see every allocation and release the process makes — the HSA calls §7.2
 * measures all fire from inside them.
 *
 * hipMalloc is served from route 1's allocator (amdfq.md §12), in its per-allocation layout:
 * amdfq_vmm_malloc reserves an extent per request, maps the block and one shared pad granule behind
 * it, so the slack a bf16 kernel over-reading its operand walks into costs one granule for the whole
 * process instead of one granule of VRAM per live allocation. What the route declines to serve — a
 * fork child, a symbol this runtime lacks, a failed create or map — is forwarded to the runtime with
 * the caller's size unchanged: this build carries no second mitigation.
 *
 * Logging happens after the call, so the line carries the returned pointer as well as the arguments,
 * and a run that dies in a later kernel still has its full allocation history on disk. `vmm=` names
 * what the route did with the call: `served block=<bytes> pad=<bytes>`, or which reason forwarded it.
 */
#include "amdfq_gates.h"
#include "amdfq_log.h"
#include "amdfq_vmm.h"

#include <stdio.h>

hipError_t hipMalloc(void **ptr, size_t size) {
    AMDFQ_REAL(hipMalloc);
    amdfq_tick(AMDFQ_FN_HIP_MALLOC);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;

    struct amdfq_vmm_result served;
    const char *why = NULL;
    char note[64];
    int outer = amdfq_vmm_begin();
    hipError_t result;

    if (outer && amdfq_vmm_malloc(size, &served, &why) == 0) {
        *ptr = served.va;
        result = HIP_SUCCESS;
        snprintf(note, sizeof note, "served block=%zu pad=%zu", served.block, served.pad);
    } else {
        result = real(ptr, size);
        snprintf(note, sizeof note, "forwarded(%s)",
                 outer ? (why != NULL ? why : "unresolved") : "nested");
    }
    if (outer) amdfq_vmm_end();

    amdfq_logf("hipMalloc(size=%zu) -> ptr=%p ret=%d vmm=%s caller=%s", size,
               ptr != NULL ? *ptr : NULL, (int)result, note, AMDFQ_CALLER());
    return result;
}

hipError_t hipFree(void *ptr) {
    AMDFQ_REAL(hipFree);
    amdfq_tick(AMDFQ_FN_HIP_FREE);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;

    const char *why = NULL;
    int outer = amdfq_vmm_begin();
    /* A block this module served is unmapped, released and its address given back here; the runtime
     * never saw it. Anything else — a fork child's parent block, a pointer allocated before the
     * route existed — is the runtime's to free. */
    int served = outer && amdfq_vmm_free(ptr, &why) == 0;
    hipError_t result = served ? HIP_SUCCESS : real(ptr);
    if (outer) amdfq_vmm_end();

    amdfq_logf("hipFree(ptr=%p) -> ret=%d vmm=%s caller=%s", ptr, (int)result,
               served ? "served" : (outer ? (why != NULL ? why : "unresolved") : "nested"),
               AMDFQ_CALLER());
    return result;
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
    AMDFQ_REAL(hipHostMalloc);
    amdfq_tick(AMDFQ_FN_HIP_HOST_MALLOC);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;

    hipError_t result = real(ptr, size, flags);

    amdfq_logf("hipHostMalloc(size=%zu flags=0x%x) -> ptr=%p ret=%d caller=%s", size, flags,
               ptr != NULL ? *ptr : NULL, (int)result, AMDFQ_CALLER());
    return result;
}

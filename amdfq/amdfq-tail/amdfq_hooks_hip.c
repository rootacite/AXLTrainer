/*
 * LD_PRELOAD interposers for the HIP allocation gates (amdfq/doc/amdfq.md §7.1): hipMalloc, hipFree and
 * hipHostMalloc are the whole allocation path of a training run in this configuration, so hooking
 * these three is enough to see every allocation and release the process makes — the HSA calls §7.2
 * measures all fire from inside them.
 *
 * hipMalloc forwards the caller's size unchanged, then the tail guard (amdfq/doc/amdfq.md §13) maps one
 * shared page behind the block when the runtime backs nothing there. hipFree unmaps that page
 * before the real free and re-guards the predecessor after it. hipHostMalloc is a transcript.
 * AMDFQ_TAIL=0 disables the guard so the original step-10 fault can be reproduced.
 *
 * Logging happens after the call, so the line carries the returned pointer as well as the arguments,
 * and a run that dies in a later kernel still has its full allocation history on disk.
 */
#include "amdfq_gates.h"
#include "amdfq_live.h"
#include "amdfq_log.h"
#include "amdfq_tail.h"

hipError_t hipMalloc(void **ptr, size_t size) {
    AMDFQ_REAL(hipMalloc);
    amdfq_tick(AMDFQ_FN_HIP_MALLOC);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;

    hipError_t result = real(ptr, size);
    if (result == HIP_SUCCESS && ptr != NULL) {
        size_t granted = size;
        int error = 0;
        if (amdfq_tail_after_malloc(ptr, &granted, &error) == AMDFQ_TAIL_FAILED) {
            result = (hipError_t)error;
            *ptr = NULL;
        }
    }

    size_t live_bytes = 0;
    size_t live = result == HIP_SUCCESS && ptr != NULL ? amdfq_live_add(*ptr, size, &live_bytes)
                                                      : amdfq_live_state(&live_bytes);

    amdfq_logf("hipMalloc(size=%zu) -> ptr=%p ret=%d live=%zu live_bytes=%zu caller=%s", size,
               ptr != NULL ? *ptr : NULL, (int)result, live, live_bytes, AMDFQ_CALLER());
    return result;
}

hipError_t hipFree(void *ptr) {
    AMDFQ_REAL(hipFree);
    amdfq_tick(AMDFQ_FN_HIP_FREE);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;

    amdfq_tail_before_free(ptr);
    hipError_t result = real(ptr);
    if (result == HIP_SUCCESS) amdfq_tail_after_free(ptr);

    size_t live_bytes = 0;
    size_t live = result == HIP_SUCCESS ? amdfq_live_del(ptr, &live_bytes)
                                        : amdfq_live_state(&live_bytes);

    amdfq_logf("hipFree(ptr=%p) -> ret=%d live=%zu live_bytes=%zu caller=%s", ptr, (int)result, live,
               live_bytes, AMDFQ_CALLER());
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

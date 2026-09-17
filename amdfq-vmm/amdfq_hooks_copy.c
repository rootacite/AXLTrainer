/*
 * The copy and fill paths. Torch moves every tensor through these, and the runtime implements the
 * small or unaligned ones with its own blit kernels (`__amd_rocclr_copyBuffer`,
 * `__amd_rocclr_fillBufferUnAligned`), which no launch hook of ours can see — so the *call* that
 * carried the address is the thing to log, and the address from the kernel log can be looked up in
 * the line.
 *
 * Under the same switch as the dispatch trace (`AMDFQ_LAUNCH_LOG=1`). One line per call.
 */
#include "amdfq_log.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

typedef int hipError_t;
typedef void *hipStream_t;

#define HIP_ERROR_NOT_FOUND 500

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, int kind);
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, int kind, hipStream_t stream);
hipError_t hipMemset(void *dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream);

static _Atomic int g_enabled = -1;

static int enabled(void) {
    int state = atomic_load_explicit(&g_enabled, memory_order_relaxed);
    if (state >= 0) return state;
    const char *env = getenv("AMDFQ_LAUNCH_LOG");
    state = env != NULL && env[0] == '1';
    atomic_store_explicit(&g_enabled, state, memory_order_relaxed);
    return state;
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, int kind) {
    AMDFQ_REAL(hipMemcpy);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    hipError_t result = real(dst, src, sizeBytes, kind);
    if (enabled())
        amdfq_logf("copy hipMemcpy(dst=%p src=%p size=%zu kind=%d) -> ret=%d caller=%s", dst, src,
                   sizeBytes, kind, (int)result, AMDFQ_CALLER());
    return result;
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, int kind, hipStream_t stream) {
    AMDFQ_REAL(hipMemcpyAsync);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    hipError_t result = real(dst, src, sizeBytes, kind, stream);
    if (enabled())
        amdfq_logf("copy hipMemcpyAsync(dst=%p src=%p size=%zu kind=%d stream=%p) -> ret=%d caller=%s",
                   dst, src, sizeBytes, kind, stream, (int)result, AMDFQ_CALLER());
    return result;
}

hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
    AMDFQ_REAL(hipMemset);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    hipError_t result = real(dst, value, sizeBytes);
    if (enabled())
        amdfq_logf("copy hipMemset(dst=%p value=%d size=%zu) -> ret=%d caller=%s", dst, value,
                   sizeBytes, (int)result, AMDFQ_CALLER());
    return result;
}

hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream) {
    AMDFQ_REAL(hipMemsetAsync);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    hipError_t result = real(dst, value, sizeBytes, stream);
    if (enabled())
        amdfq_logf("copy hipMemsetAsync(dst=%p value=%d size=%zu stream=%p) -> ret=%d caller=%s", dst,
                   value, sizeBytes, stream, (int)result, AMDFQ_CALLER());
    return result;
}

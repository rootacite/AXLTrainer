/*
 * The gate APIs: the HIP entry points through which this training process reaches the ROCm memory
 * manager. Prototypes are copied from the ROCm 10.0.0 headers that ship with the `axl` env —
 *   _rocm_sdk_core/include/hip/hip_runtime_api.h
 * — and declared here instead of included, so this object builds with no ROCm headers at all and
 * `ldd` stays clean.
 *
 * Deliberately only the three functions that allocate: `hipMalloc`, `hipFree` and `hipHostMalloc`
 * are the whole allocation path of a run in this configuration (amdfq.md §7.1, §7.3), and the HSA
 * layer §7.2 measures is reached from inside them.
 *
 * Each prototype below is *defined* by this shared object as an LD_PRELOAD interposer, which logs
 * and then forwards to the real implementation via dlsym(RTLD_NEXT, ...).
 */
#ifndef AMDFQ_GATES_H
#define AMDFQ_GATES_H

#include <stddef.h>

/* hipError_t is an enum in the real header; int-sized, and never passed by value. */
typedef int hipError_t;

#define HIP_SUCCESS 0
#define HIP_ERROR_NOT_FOUND 500 /* hipErrorNotFound */

hipError_t hipMalloc(void **ptr, size_t size);
hipError_t hipFree(void *ptr);
hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags);

#endif /* AMDFQ_GATES_H */

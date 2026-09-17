/*
 * Kernel-dispatch interposers: what the process *executes*, against amdfq_hooks_hip.c's what it
 * allocates. The question they answer is whether two runs of the same configuration execute the same
 * kernels in the same order — the VMM route (amdfq.md §10.3) changes where a buffer lives, and if the
 * kernel sequence moves with it, that is a different bug than the one being investigated.
 *
 * Off by default: a training step dispatches thousands of kernels and one line each is a lot of log.
 * `AMDFQ_LAUNCH_LOG=1` turns it on; with it off every interposer is a bare forward.
 * `AMDFQ_LAUNCH_ARGS=1` adds each dispatch's first eight argument words, which is what identifies the
 * dispatch behind a GPU fault: the kernel log prints the address it faulted on, and that address can
 * then be looked up in these words.
 *
 * Names come from two places, because launches do: hipModuleGetFunction names every function handle
 * the process loads (torch's precompiled kernels, hipBLASLt's Tensile kernels, MIOpen), and a launch
 * through a host stub is resolved with dladdr to its mangled symbol. Streams are logged so a
 * comparison can tell which stream a dispatch went to.
 */
#define _GNU_SOURCE /* Dl_info */

#include "amdfq_log.h"

#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;
typedef void *hipFunction_t;
typedef void *hipModule_t;
typedef void *hipStream_t;
typedef void *hipEvent_t;

/* dim3 as the ABI passes it: three uint32 by value. */
typedef struct {
    uint32_t x, y, z;
} amdfq_dim3;

#define HIP_SUCCESS 0
#define HIP_ERROR_NOT_FOUND 500

/* The escape hatch of hipModuleLaunchKernel's `extra` argument, as hip_runtime_api.h defines it. */
#define AMDFQ_PARAM_BUFFER_POINTER ((void *)0x01)
#define AMDFQ_PARAM_BUFFER_SIZE ((void *)0x02)
#define AMDFQ_PARAM_END ((void *)0x03)

#define AMDFQ_ARGS_MAX 8

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ, unsigned int sharedMemBytes,
                                 hipStream_t stream, void **kernelParams, void **extra);
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY,
                                    uint32_t globalWorkSizeZ, uint32_t localWorkSizeX,
                                    uint32_t localWorkSizeY, uint32_t localWorkSizeZ,
                                    size_t sharedMemBytes, hipStream_t hStream, void **kernelParams,
                                    void **extra, hipEvent_t startEvent, hipEvent_t stopEvent,
                                    uint32_t flags);
hipError_t hipLaunchKernel(const void *function_address, amdfq_dim3 numBlocks, amdfq_dim3 dimBlocks,
                           void **args, size_t sharedMemBytes, hipStream_t stream);
hipError_t hipDrvLaunchKernelEx(const void *config, hipFunction_t f, void **params, void **extra);

/* Tensile names carry the whole solution: the ones this trace has to tell apart are 888 characters,
 * and a truncated name cannot distinguish the sibling that died from the one next to it. */
#define AMDFQ_NAME_MAX 1024

static struct {
    hipFunction_t handle;
    char name[AMDFQ_NAME_MAX];
} *g_functions;
static size_t g_function_count;
static size_t g_function_cap;
static _Atomic unsigned long long g_dispatches;
static _Atomic int g_enabled = -1; /* -1 until the environment has been read */
static _Atomic int g_args = -1;

static int enabled(void) {
    int state = atomic_load_explicit(&g_enabled, memory_order_relaxed);
    if (state >= 0) return state;
    const char *env = getenv("AMDFQ_LAUNCH_LOG");
    state = env != NULL && env[0] == '1';
    atomic_store_explicit(&g_enabled, state, memory_order_relaxed);
    return state;
}

static int args_enabled(void) {
    int state = atomic_load_explicit(&g_args, memory_order_relaxed);
    if (state >= 0) return state;
    const char *env = getenv("AMDFQ_LAUNCH_ARGS");
    state = env != NULL && env[0] == '1';
    atomic_store_explicit(&g_args, state, memory_order_relaxed);
    return state;
}

static void remember(const char *kname, hipFunction_t handle) {
    for (size_t i = 0; i < g_function_count; i++)
        if (g_functions[i].handle == handle) return;
    if (g_function_count == g_function_cap) {
        size_t cap = g_function_cap == 0 ? 256 : g_function_cap * 2;
        void *grown = realloc(g_functions, cap * sizeof *g_functions);
        if (grown == NULL) return;
        g_functions = grown;
        g_function_cap = cap;
    }
    g_functions[g_function_count].handle = handle;
    snprintf(g_functions[g_function_count].name, AMDFQ_NAME_MAX, "%s", kname);
    g_function_count++;
}

static const char *name_of(hipFunction_t f) {
    static __thread char buf[AMDFQ_NAME_MAX];
    for (size_t i = 0; i < g_function_count; i++)
        if (g_functions[i].handle == f) return g_functions[i].name;

    Dl_info info;
    if (f != NULL && dladdr((void *)f, &info) != 0) {
        /* A host stub: its mangled symbol. Where the object has no symbol for the address, the
         * object plus offset still identifies the same kernel in another run, which is the point of
         * the trace — raw addresses are not comparable across runs (ASLR). */
        if (info.dli_sname != NULL) {
            snprintf(buf, sizeof buf, "%s", info.dli_sname);
        } else {
            const char *file = info.dli_fname != NULL ? strrchr(info.dli_fname, '/') : NULL;
            file = file != NULL ? file + 1 : (info.dli_fname != NULL ? info.dli_fname : "?");
            snprintf(buf, sizeof buf, "%s+0x%lx", file,
                     (unsigned long)((const char *)f - (const char *)info.dli_fbase));
        }
        return buf;
    }
    snprintf(buf, sizeof buf, "<unknown %p>", (void *)f);
    return buf;
}

/* The first AMDFQ_ARGS_MAX words of an argument block, in a caller-provided buffer. Reading eight
 * bytes where the argument is narrower may pick up a neighbour's bytes; that is fine for a trace and
 * is why the words are matched against a fault address rather than interpreted. */
static const char *words_of(const void *buffer, size_t size, char *out, size_t out_size) {
    size_t used = 0;
    unsigned count = (unsigned)(size / 8);
    if (count > AMDFQ_ARGS_MAX) count = AMDFQ_ARGS_MAX;
    for (unsigned i = 0; i < count && used + 20 < out_size; i++) {
        uint64_t word;
        memcpy(&word, (const char *)buffer + (size_t)i * 8, sizeof word);
        used += (size_t)snprintf(out + used, out_size - used, " %#llx", (unsigned long long)word);
    }
    out[used < out_size ? used : out_size - 1] = '\0';
    return out;
}

/* The two array forms (hipLaunchKernel's `args`, hipModuleLaunchKernel's `kernelParams`) carry no
 * count: reading an entry past the kernel's arity dereferences whatever the caller had next on its
 * stack, which is how an earlier revision of this file crashed the process it was tracing. They are
 * reported but not read — only the packed form below has a size to trust. */
static const char *unread_form(const char *const *array, const char *which) {
    (void)array;
    return which;
}

/* The packed-argument form: alternating key/value pairs ending in HIP_LAUNCH_PARAM_END. */
static const char *extra_words(const void *const *extra, char *out, size_t out_size) {
    const void *buffer = NULL;
    size_t size = 0;
    for (int i = 0; i < 32 && extra != NULL && extra[i] != NULL; i += 2) {
        if (extra[i] == AMDFQ_PARAM_BUFFER_POINTER) {
            buffer = extra[i + 1];
        } else if (extra[i] == AMDFQ_PARAM_BUFFER_SIZE) {
            if (extra[i + 1] != NULL) size = *(const size_t *)extra[i + 1];
        } else if (extra[i] == AMDFQ_PARAM_END) {
            break;
        }
    }
    if (buffer == NULL || size == 0) {
        out[0] = '\0';
        return out;
    }
    return words_of(buffer, size, out, out_size);
}

static void log_launch(const char *api, const char *name, uint32_t gx, uint32_t gy, uint32_t gz,
                       uint32_t bx, uint32_t by, uint32_t bz, hipStream_t stream,
                       const char *args) {
    amdfq_logf("launch[%llu] %s name=%s grid=%ux%ux%u block=%ux%ux%u stream=%p caller=%s%s",
               atomic_fetch_add(&g_dispatches, 1) + 1, api, name, gx, gy, gz, bx, by, bz, stream,
               AMDFQ_CALLER(), args != NULL ? args : "");
}

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname) {
    AMDFQ_REAL(hipModuleGetFunction);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    hipError_t result = real(function, module, kname);
    if (enabled() && result == HIP_SUCCESS && function != NULL)
        remember(kname != NULL ? kname : "<null>", *function);
    return result;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ, unsigned int sharedMemBytes,
                                 hipStream_t stream, void **kernelParams, void **extra) {
    AMDFQ_REAL(hipModuleLaunchKernel);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    if (enabled()) {
        char words[AMDFQ_ARGS_MAX * 20];
        const char *args = NULL;
        if (args_enabled()) {
            if (extra != NULL) args = extra_words((const void *const *)extra, words, sizeof words);
            else if (kernelParams != NULL) args = unread_form((const char *const *)kernelParams, " kernelParams");
        }
        log_launch("hipModuleLaunchKernel", name_of(f), gridDimX, gridDimY, gridDimZ, blockDimX,
                   blockDimY, blockDimZ, stream, args);
    }
    return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                stream, kernelParams, extra);
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY,
                                    uint32_t globalWorkSizeZ, uint32_t localWorkSizeX,
                                    uint32_t localWorkSizeY, uint32_t localWorkSizeZ,
                                    size_t sharedMemBytes, hipStream_t hStream, void **kernelParams,
                                    void **extra, hipEvent_t startEvent, hipEvent_t stopEvent,
                                    uint32_t flags) {
    AMDFQ_REAL(hipExtModuleLaunchKernel);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    if (enabled()) {
        char words[AMDFQ_ARGS_MAX * 20];
        const char *args = NULL;
        if (args_enabled()) {
            if (extra != NULL) args = extra_words((const void *const *)extra, words, sizeof words);
            else if (kernelParams != NULL) args = unread_form((const char *const *)kernelParams, " kernelParams");
        }
        log_launch("hipExtModuleLaunchKernel", name_of(f),
                   globalWorkSizeX / (localWorkSizeX ? localWorkSizeX : 1),
                   globalWorkSizeY / (localWorkSizeY ? localWorkSizeY : 1),
                   globalWorkSizeZ / (localWorkSizeZ ? localWorkSizeZ : 1), localWorkSizeX,
                   localWorkSizeY, localWorkSizeZ, hStream, args);
    }
    return real(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
                localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent,
                flags);
}

hipError_t hipLaunchKernel(const void *function_address, amdfq_dim3 numBlocks, amdfq_dim3 dimBlocks,
                           void **args, size_t sharedMemBytes, hipStream_t stream) {
    AMDFQ_REAL(hipLaunchKernel);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    if (enabled()) {
        const char *w = args_enabled() ? unread_form((const char *const *)args, " args") : NULL;
        log_launch("hipLaunchKernel", name_of((hipFunction_t)function_address), numBlocks.x, numBlocks.y,
                   numBlocks.z, dimBlocks.x, dimBlocks.y, dimBlocks.z, stream, w);
    }
    return real(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

hipError_t hipDrvLaunchKernelEx(const void *config, hipFunction_t f, void **params, void **extra) {
    AMDFQ_REAL(hipDrvLaunchKernelEx);
    if (real == NULL) return HIP_ERROR_NOT_FOUND;
    if (enabled()) {
        char words[AMDFQ_ARGS_MAX * 20];
        const char *args = NULL;
        if (args_enabled()) {
            if (extra != NULL) args = extra_words((const void *const *)extra, words, sizeof words);
            else if (params != NULL) args = unread_form((const char *const *)params, " params");
        }
        log_launch("hipDrvLaunchKernelEx", name_of(f), 0, 0, 0, 0, 0, 0, NULL, args);
    }
    return real(config, f, params, extra);
}

/* The VMM route behind hipMalloc: see amdfq_vmm.h and amdfq.md §10.3. */
#define _GNU_SOURCE

#include "amdfq_vmm.h"
#include "amdfq_log.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AMDFQ_VMM_TABLE_MIN 1024

enum vmm_state { VMM_UNINIT = 0, VMM_READY, VMM_DISABLED };

enum vmm_fallback {
    FB_NOSYM,
    FB_GRANULARITY,
    FB_PAD_CREATE,
    FB_REGISTRY,
    FB_CREATE,
    FB_MAP,
    FB_RESERVE_PER,
    FB_FORK,
    FB_SIZE0,
    FB_TOO_BIG,
    FB_NOT_OURS,
    FB_COUNT
};

static const char *const g_fb_names[FB_COUNT] = {
    "nosym", "granularity", "pad-create", "registry", "create", "map",
    "reserve-per", "fork",  "size0",      "too-big",  "not-ours",
};

struct rec {
    void *va; /* REC_TOMB marks a deleted slot; a real mapping never sits at address 1 */
    size_t block;
    size_t pad;
    void *handle;
};

/* Open addressing, keyed by the pointer hipMalloc returned. */
struct table {
    struct rec *slots;
    size_t cap;
    size_t used;
    size_t tombs;
};

#define REC_TOMB ((void *)(uintptr_t)1)

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local int g_depth; /* non-zero while inside our own VMM work */

static enum vmm_state g_state = VMM_UNINIT;
static const char *g_disabled_reason; /* why the route is off, for the summary */
static pid_t g_owner;                 /* the process that built this state */

static hipError_t (*p_get_device)(int *device);
static hipError_t (*p_get_granularity)(size_t *granularity, const hipMemAllocationProp *prop,
                                       int option);
static hipError_t (*p_address_reserve)(void **ptr, size_t size, size_t alignment, void *addr,
                                       unsigned long long flags);
static hipError_t (*p_address_free)(void *ptr, size_t size);
static hipError_t (*p_create)(hipMemGenericAllocationHandle_t *handle, size_t size,
                              const hipMemAllocationProp *prop, unsigned long long flags);
static hipError_t (*p_map)(void *ptr, size_t size, size_t offset,
                           hipMemGenericAllocationHandle_t handle, unsigned long long flags);
static hipError_t (*p_unmap)(void *ptr, size_t size);
static hipError_t (*p_release)(hipMemGenericAllocationHandle_t handle);
static hipError_t (*p_set_access)(void *ptr, size_t size, const hipMemAccessDesc *desc, size_t count);
/* hipGetLastError: the runtime keeps a per-thread sticky error that every other call's failure sets
 * and that the caller's own error check reads. One of our calls is allowed to fail (asking for the
 * host access of a range the device refuses), so we consume that state after it. */
static hipError_t (*p_get_last_error)(void);

static size_t g_granule; /* block rounding and pad size, both the recommended granularity */
static hipMemAllocationProp g_prop;
/* The one physical object every served block's pad is a mapping of: created once, released never. */
static hipMemGenericAllocationHandle_t g_pad;

static struct table g_table; /* keyed by the pointer hipMalloc returned */

static unsigned long long g_blocks_created;
static unsigned long long g_blocks_released;
static unsigned long long g_bytes_created;
static unsigned long long g_bytes_released;
static unsigned long long g_live_bytes;
static unsigned long long g_peak_live_bytes;
static unsigned long long g_pad_maps;
static unsigned long long g_pad_map_failures;
static unsigned long long g_forked_frees;
static unsigned long long g_reserves;      /* hipMemAddressReserve calls that returned */
static unsigned long long g_reserve_bytes; /* bytes those reservations cover */
static unsigned long long g_fallbacks[FB_COUNT];

int amdfq_vmm_begin(void) {
    if (g_depth != 0) return 0;
    g_depth = 1;
    return 1;
}

void amdfq_vmm_end(void) { g_depth = 0; }

static int fallback(enum vmm_fallback reason, const char **why) {
    g_fallbacks[reason]++;
    *why = g_fb_names[reason];
    return -1;
}

/* Consume the runtime's sticky error state after a call of ours that was allowed to fail. Nothing can
 * set that state back, so a failure the caller had left pending before entering our hook is lost with
 * it; that is the price of asking the runtime a question whose answer is an error. */
static void clear_error(void) {
    if (p_get_last_error != NULL) p_get_last_error();
}

static void set_access(void *ptr, size_t size) {
    /* Two locations, because hipMalloc's memory is readable by the CPU at the same address and a
     * device-only mapping is not: measured 2026-09-17 by reading a mapped block from host code
     * (amdfq-vmm/vmm_probe.c `hostaccess`), after a training run died in
     * at::native::_local_scalar_dense_cuda doing exactly that read — torch's `.item()` reads a
     * device pointer from the host on this stack. A range with an unmapped gap in it is rejected
     * outright, so this is per region. */
    hipMemAccessDesc desc[2];
    memset(desc, 0, sizeof desc);
    desc[0].location = g_prop.location;
    desc[0].flags = AMDFQ_MEM_ACCESS_PROT_READWRITE;
    desc[1].location.type = AMDFQ_MEM_LOCATION_HOST;
    desc[1].location.id = 0;
    desc[1].flags = AMDFQ_MEM_ACCESS_PROT_READWRITE;
    if (p_set_access(ptr, size, desc, 2) == 0) return;
    clear_error();
    if (p_set_access(ptr, size, desc, 1) == 0) {
        amdfq_logf("# vmm: host access refused at %p; the GPU can use the block, host reads of it"
                   " (torch's .item() among them) will fault", ptr);
        return;
    }
    clear_error();
    amdfq_logf("# vmm: hipMemSetAccess(%p, %zu) failed; the region may be unreachable", ptr, size);
}

/* Hand a reserved extent back to the OS. */
static void release_extent(void *va, size_t extent) {
    if (p_address_free(va, extent) != 0)
        amdfq_logf("# vmm: hipMemAddressFree(%p, %zu) failed", va, extent);
}

/* --- allocation registry ----------------------------------------------------------------------- */

static size_t slot_of(void *va) {
    uint64_t key = (uint64_t)(uintptr_t)va >> 12;
    key *= 0x9E3779B97F4A7C15ull;
    return (size_t)(key >> 40);
}

static void table_insert(struct table *t, void *va, size_t block, size_t pad, void *handle);

static int table_grow(struct table *t) {
    size_t old_cap = t->cap;
    struct rec *old = t->slots;
    size_t cap = old_cap == 0 ? AMDFQ_VMM_TABLE_MIN : old_cap * 2;
    struct rec *fresh = calloc(cap, sizeof *fresh);
    if (fresh == NULL) return -1;
    t->slots = fresh;
    t->cap = cap;
    t->used = 0;
    t->tombs = 0;
    for (size_t i = 0; i < old_cap; i++)
        if (old != NULL && old[i].va != NULL && old[i].va != REC_TOMB)
            table_insert(t, old[i].va, old[i].block, old[i].pad, old[i].handle);
    free(old);
    return 0;
}

static void table_insert(struct table *t, void *va, size_t block, size_t pad, void *handle) {
    if ((t->used + t->tombs + 1) * 2 > t->cap && table_grow(t) != 0) {
        amdfq_logf("# vmm: registry grow failed; freeing this pointer will fall through to the runtime");
        return;
    }
    size_t mask = t->cap - 1;
    size_t i = slot_of(va) & mask;
    size_t tomb = SIZE_MAX;
    for (;;) {
        struct rec *slot = &t->slots[i];
        if (slot->va == NULL) {
            if (tomb != SIZE_MAX) {
                slot = &t->slots[tomb];
                t->tombs--;
            }
            slot->va = va;
            slot->block = block;
            slot->pad = pad;
            slot->handle = handle;
            t->used++;
            return;
        }
        if (slot->va == REC_TOMB) {
            if (tomb == SIZE_MAX) tomb = i;
        } else if (slot->va == va) {
            slot->block = block;
            slot->pad = pad;
            slot->handle = handle;
            return;
        }
        i = (i + 1) & mask;
    }
}

static struct rec *table_find(struct table *t, void *va) {
    if (t->cap == 0) return NULL;
    size_t mask = t->cap - 1;
    size_t i = slot_of(va) & mask;
    for (;;) {
        struct rec *slot = &t->slots[i];
        if (slot->va == NULL) return NULL;
        if (slot->va == va) return slot;
        i = (i + 1) & mask;
    }
}

static void table_erase(struct table *t, void *va) {
    struct rec *slot = table_find(t, va);
    if (slot == NULL) return;
    slot->va = REC_TOMB;
    slot->handle = NULL;
    t->used--;
    t->tombs++;
}

/* --- start-up ---------------------------------------------------------------------------------- */

static int resolve_all(void) {
    const char *missing = NULL;
    p_get_device = (__typeof__(p_get_device))amdfq_symbol("hipGetDevice");
    p_get_granularity = (__typeof__(p_get_granularity))amdfq_symbol("hipMemGetAllocationGranularity");
    p_address_reserve = (__typeof__(p_address_reserve))amdfq_symbol("hipMemAddressReserve");
    p_address_free = (__typeof__(p_address_free))amdfq_symbol("hipMemAddressFree");
    p_create = (__typeof__(p_create))amdfq_symbol("hipMemCreate");
    p_map = (__typeof__(p_map))amdfq_symbol("hipMemMap");
    p_unmap = (__typeof__(p_unmap))amdfq_symbol("hipMemUnmap");
    p_release = (__typeof__(p_release))amdfq_symbol("hipMemRelease");
    p_set_access = (__typeof__(p_set_access))amdfq_symbol("hipMemSetAccess");
    p_get_last_error = (__typeof__(p_get_last_error))amdfq_symbol("hipGetLastError");

    if (p_get_device == NULL) missing = "hipGetDevice";
    else if (p_get_granularity == NULL) missing = "hipMemGetAllocationGranularity";
    else if (p_address_reserve == NULL) missing = "hipMemAddressReserve";
    else if (p_address_free == NULL) missing = "hipMemAddressFree";
    else if (p_create == NULL) missing = "hipMemCreate";
    else if (p_map == NULL) missing = "hipMemMap";
    else if (p_unmap == NULL) missing = "hipMemUnmap";
    else if (p_release == NULL) missing = "hipMemRelease";
    else if (p_set_access == NULL) missing = "hipMemSetAccess";
    else if (p_get_last_error == NULL) missing = "hipGetLastError";
    if (missing != NULL) {
        amdfq_logf("# vmm: %s is missing from this runtime; hipMalloc is forwarded unchanged", missing);
        return -1;
    }
    return 0;
}

/* Build the per-process state on the first hipMalloc this process makes, when the real HIP entry
 * points are already resolvable. */
static int ensure_ready(const char **why) {
    if (g_state == VMM_READY) return 1;
    if (g_state == VMM_DISABLED) {
        /* Already failed once: the reason was logged and counted where it happened. */
        *why = g_disabled_reason != NULL ? g_disabled_reason : "not-ready";
        return 0;
    }

    if (resolve_all() != 0) {
        g_state = VMM_DISABLED;
        g_disabled_reason = "nosym";
        *why = g_disabled_reason;
        g_fallbacks[FB_NOSYM]++;
        return 0;
    }

    int device = 0;
    p_get_device(&device);
    memset(&g_prop, 0, sizeof g_prop);
    g_prop.type = AMDFQ_MEM_ALLOCATION_TYPE_PINNED;
    g_prop.requestedHandleTypes = AMDFQ_MEM_HANDLE_TYPE_NONE;
    g_prop.location.type = AMDFQ_MEM_LOCATION_DEVICE;
    g_prop.location.id = device;

    size_t minimum = 0, recommended = 0;
    if (p_get_granularity(&minimum, &g_prop, AMDFQ_GRANULARITY_MINIMUM) != 0 || minimum == 0) {
        g_state = VMM_DISABLED;
        g_disabled_reason = "granularity";
        *why = g_disabled_reason;
        g_fallbacks[FB_GRANULARITY]++;
        return 0;
    }
    if (p_get_granularity(&recommended, &g_prop, AMDFQ_GRANULARITY_RECOMMENDED) != 0 ||
        recommended < minimum)
        recommended = minimum;
    /* The pad keeps the magnitude §10.1 measured as sufficient — one granule — and the recommended
     * granularity is also what the runtime's own pool charges, so block rounding costs nothing that
     * hipMalloc would not have cost anyway (measured: 1/2/4/8/32/44 MiB request -> the same size). */
    g_granule = recommended;

    if (p_create(&g_pad, g_granule, &g_prop, 0) != 0) {
        g_state = VMM_DISABLED;
        g_disabled_reason = "pad-create";
        *why = g_disabled_reason;
        g_fallbacks[FB_PAD_CREATE]++;
        amdfq_logf("# vmm: hipMemCreate(%zu) failed for the shared pad granule; nothing is served",
                   g_granule);
        return 0;
    }

    if (table_grow(&g_table) != 0) {
        g_state = VMM_DISABLED;
        g_disabled_reason = "registry";
        *why = g_disabled_reason;
        g_fallbacks[FB_REGISTRY]++;
        return 0;
    }

    g_owner = getpid();
    g_state = VMM_READY;
    amdfq_logf("# vmm: pid %d device %d granularity=%zu B (minimum %zu), layout=peralloc (one"
               " hipMemAddressReserve per allocation), pad one shared granule of %zu B",
               (int)g_owner, device, g_granule, minimum, g_granule);
    return 1;
}

/* --- the two entry points ---------------------------------------------------------------------- */

int amdfq_vmm_malloc(size_t size, struct amdfq_vmm_result *result, const char **why) {
    *why = NULL;
    pthread_mutex_lock(&g_lock);
    if (!ensure_ready(why)) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (getpid() != g_owner) {
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_FORK, why);
    }
    if (size == 0) {
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_SIZE0, why);
    }
    /* The block rounds up to the granularity and the pad is one granule behind it, so the request has
     * to leave room for both. */
    if (size > SIZE_MAX - 2 * g_granule) {
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_TOO_BIG, why);
    }

    size_t block = (size + g_granule - 1) / g_granule * g_granule;
    size_t pad = g_granule;
    size_t extent = block + pad;

    void *va = NULL;
    if (p_address_reserve(&va, extent, g_granule, NULL, 0) != 0) {
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_RESERVE_PER, why);
    }
    g_reserves++;
    g_reserve_bytes += extent;

    hipMemGenericAllocationHandle_t handle = NULL;
    if (p_create(&handle, block, &g_prop, 0) != 0) {
        release_extent(va, extent);
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_CREATE, why);
    }
    if (p_map(va, block, 0, handle, 0) != 0) {
        p_release(handle);
        release_extent(va, extent);
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_MAP, why);
    }
    set_access(va, block);

    if (p_map((char *)va + block, pad, 0, g_pad, 0) == 0) {
        set_access((char *)va + block, pad);
        g_pad_maps++;
    } else {
        /* The block is valid and usable without the slack behind it; only the protection is
         * missing, so this is logged and the allocation is kept. */
        g_pad_map_failures++;
        amdfq_logf("# vmm: pad mapping at %p failed; block at %p served unprotected",
                   (char *)va + block, va);
    }

    table_insert(&g_table, va, block, pad, (void *)handle);
    g_blocks_created++;
    g_bytes_created += block;
    g_live_bytes += block;
    if (g_live_bytes > g_peak_live_bytes) g_peak_live_bytes = g_live_bytes;

    result->va = va;
    result->block = block;
    result->pad = pad;
    result->handle = (void *)handle;
    pthread_mutex_unlock(&g_lock);
    return 0;
}

int amdfq_vmm_free(void *ptr, const char **why) {
    *why = NULL;
    pthread_mutex_lock(&g_lock);
    if (g_state != VMM_READY) {
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_NOT_OURS, why);
    }
    struct rec *record = table_find(&g_table, ptr);
    if (record == NULL) {
        pthread_mutex_unlock(&g_lock);
        return fallback(FB_NOT_OURS, why);
    }
    if (getpid() != g_owner) {
        /* Inherited through fork: the handle belongs to the parent, which is still using it, and
         * this process's copy of the mapping dies with the process. */
        g_forked_frees++;
        size_t block = record->block;
        pthread_mutex_unlock(&g_lock);
        amdfq_logf("# vmm: hipFree(%p) belongs to pid %d, this is %d; block of %zu bytes left to exit",
                   ptr, (int)g_owner, (int)getpid(), block);
        return 0;
    }

    hipMemGenericAllocationHandle_t handle = (hipMemGenericAllocationHandle_t)record->handle;
    size_t block = record->block;
    size_t pad = record->pad;
    size_t extent = block + pad;

    if (p_unmap(ptr, block) != 0)
        amdfq_logf("# vmm: hipMemUnmap(%p, %zu) failed", ptr, block);
    if (pad != 0 && p_unmap((char *)ptr + block, pad) != 0)
        amdfq_logf("# vmm: hipMemUnmap(%p, %zu) failed (pad)", (char *)ptr + block, pad);
    if (p_release(handle) != 0)
        amdfq_logf("# vmm: hipMemRelease(%p) failed", (void *)handle);
    table_erase(&g_table, ptr);
    release_extent(ptr, extent);

    g_blocks_released++;
    g_bytes_released += block;
    g_live_bytes -= g_live_bytes >= block ? block : g_live_bytes;

    pthread_mutex_unlock(&g_lock);
    return 0;
}

/* --- summary ----------------------------------------------------------------------------------- */

void amdfq_vmm_report(void) {
    if (g_state == VMM_UNINIT) {
        amdfq_logf("# vmm: never used");
        return;
    }
    if (g_state == VMM_DISABLED) {
        amdfq_logf("# vmm: route off (%s)",
                   g_disabled_reason != NULL ? g_disabled_reason : "not-ready");
    } else {
        amdfq_logf("# vmm: route on, layout=peralloc, one hipMemAddressReserve per allocation, one"
                   " shared %zu B granule mapped behind every block", g_granule);
        amdfq_logf("# vmm: blocks created %llu (%llu bytes), released %llu (%llu bytes), live peak %llu"
                   " bytes",
                   g_blocks_created, g_bytes_created, g_blocks_released, g_bytes_released,
                   g_peak_live_bytes);
        amdfq_logf("# vmm: %llu per-allocation reservations, %llu bytes, pad mappings %llu (%llu"
                   " failed)", g_reserves, g_reserve_bytes, g_pad_maps, g_pad_map_failures);
        if (g_forked_frees != 0)
            amdfq_logf("# vmm: %llu frees of a parent's blocks were left alone (this process is a fork"
                       " child)", g_forked_frees);
    }
    for (int i = 0; i < FB_COUNT; i++)
        if (g_fallbacks[i] != 0)
            amdfq_logf("# vmm fallback %-12s %llu", g_fb_names[i], g_fallbacks[i]);
}

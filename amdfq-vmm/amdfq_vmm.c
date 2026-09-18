/* The VMM route behind hipMalloc: see amdfq_vmm.h and amdfq.md §10.3. */
#define _GNU_SOURCE

#include "amdfq_vmm.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AMDFQ_VMM_TABLE_MIN 1024

enum vmm_state { VMM_UNINIT = 0, VMM_READY, VMM_DISABLED };

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
static pid_t g_owner; /* the process that built this state */

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

void *amdfq_symbol(const char *name) {
    dlerror();
    void *symbol = dlsym(RTLD_NEXT, name);
    const char *error = dlerror();
    return symbol == NULL || error != NULL ? NULL : symbol;
}

int amdfq_vmm_begin(void) {
    if (g_depth != 0) return 0;
    g_depth = 1;
    return 1;
}

void amdfq_vmm_end(void) { g_depth = 0; }

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
    if (p_set_access(ptr, size, desc, 1) == 0) return;
    clear_error();
}

/* Hand a reserved extent back to the OS. */
static void release_extent(void *va, size_t extent) { p_address_free(va, extent); }

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
    if ((t->used + t->tombs + 1) * 2 > t->cap && table_grow(t) != 0) return;
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

    return p_get_device == NULL || p_get_granularity == NULL || p_address_reserve == NULL ||
                   p_address_free == NULL || p_create == NULL || p_map == NULL || p_unmap == NULL ||
                   p_release == NULL || p_set_access == NULL || p_get_last_error == NULL
               ? -1
               : 0;
}

/* Build the per-process state on the first hipMalloc this process makes, when the real HIP entry
 * points are already resolvable. A failure disables the route for the rest of the process: later
 * calls are rejected without asking the runtime again. */
static int ensure_ready(void) {
    if (g_state == VMM_READY) return 1;
    if (g_state == VMM_DISABLED) return 0;

    if (resolve_all() != 0) {
        g_state = VMM_DISABLED;
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
        return 0;
    }
    if (table_grow(&g_table) != 0) {
        g_state = VMM_DISABLED;
        return 0;
    }
    g_owner = getpid();
    g_state = VMM_READY;
    return 1;
}

/* --- the two entry points ---------------------------------------------------------------------- */

int amdfq_vmm_malloc(size_t size, struct amdfq_vmm_result *result) {
    pthread_mutex_lock(&g_lock);
    if (!ensure_ready()) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    /* A fork child never touches state it inherited. */
    if (getpid() != g_owner) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (size == 0) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    /* The block rounds up to the granularity and the pad is one granule behind it, so the request has
     * to leave room for both. */
    if (size > SIZE_MAX - 2 * g_granule) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    size_t block = (size + g_granule - 1) / g_granule * g_granule;
    size_t pad = g_granule;
    size_t extent = block + pad;

    void *va = NULL;
    if (p_address_reserve(&va, extent, g_granule, NULL, 0) != 0) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    hipMemGenericAllocationHandle_t handle = NULL;
    if (p_create(&handle, block, &g_prop, 0) != 0) {
        release_extent(va, extent);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (p_map(va, block, 0, handle, 0) != 0) {
        p_release(handle);
        release_extent(va, extent);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    set_access(va, block);

    /* The block is valid and usable without the slack behind it, so a pad mapping that fails costs
     * the protection of this allocation only and the allocation itself is kept. */
    if (p_map((char *)va + block, pad, 0, g_pad, 0) == 0) set_access((char *)va + block, pad);

    table_insert(&g_table, va, block, pad, (void *)handle);
    result->va = va;
    pthread_mutex_unlock(&g_lock);
    return 0;
}

int amdfq_vmm_free(void *ptr) {
    pthread_mutex_lock(&g_lock);
    if (g_state != VMM_READY) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    struct rec *record = table_find(&g_table, ptr);
    if (record == NULL) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (getpid() != g_owner) {
        /* Inherited through fork: the handle belongs to the parent, which is still using it, and
         * this process's copy of the mapping dies with the process. */
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    hipMemGenericAllocationHandle_t handle = (hipMemGenericAllocationHandle_t)record->handle;
    size_t block = record->block;
    size_t pad = record->pad;
    size_t extent = block + pad;

    p_unmap(ptr, block);
    if (pad != 0) p_unmap((char *)ptr + block, pad);
    p_release(handle);
    table_erase(&g_table, ptr);
    release_extent(ptr, extent);

    pthread_mutex_unlock(&g_lock);
    return 0;
}

/* The tail guard behind hipMalloc: see amdfq_tail.h and amdfq.md §13. */
#define _GNU_SOURCE

#include "amdfq_tail.h"
#include "amdfq_log.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* HIP types this module needs, copied from the ROCm 10.0.0 headers the way amdfq_gates.h copies
 * its three prototypes, so this object still builds with no ROCm headers present. */
typedef int hipError_t;
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

#define AMDFQ_TAIL_PAD 16
#define AMDFQ_TABLE_MIN 1024
#define AMDFQ_HEARTBEAT 2000u

enum tail_state { TAIL_UNINIT = 0, TAIL_ON, TAIL_OFF };

struct rec {
    void *va; /* REC_TOMB marks a deleted slot; a real mapping never sits at address 1 */
    size_t block;
    void *alt; /* g_table: the protection page, or NULL; g_ends: the owning allocation */
};

struct table {
    struct rec *slots;
    size_t cap;
    size_t used;
    size_t tombs;
};

#define REC_TOMB ((void *)(uintptr_t)1)

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static enum tail_state g_state = TAIL_UNINIT;
static const char *g_off_reason;
static pid_t g_owner;

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
static hipError_t (*p_set_access)(void *ptr, size_t size, const hipMemAccessDesc *desc, size_t count);
static hipError_t (*p_get_range)(void **pbase, size_t *psize, void *ptr);
/* The runtime keeps a per-thread sticky error that every other call's failure sets. Asking what
 * backs an address that nothing backs is the question, so we consume that state after each of them. */
static hipError_t (*p_get_last_error)(void);
static hipError_t (*p_real_malloc)(void **ptr, size_t size);
static hipError_t (*p_real_free)(void *ptr);

static size_t g_granule;
static hipMemAllocationProp g_prop;
static hipMemGenericAllocationHandle_t g_shared;

static struct table g_table; /* keyed by the pointer hipMalloc returned */
static struct table g_ends;  /* keyed by the address a live allocation's extent ends at */

static unsigned long long g_examined;
static unsigned long long g_backed;
static unsigned long long g_created;
static unsigned long long g_neighbour;
static unsigned long long g_released;
static unsigned long long g_protected;
static unsigned long long g_padded;
static unsigned long long g_pad_failures;
static unsigned long long g_unaligned;
static unsigned long long g_elsewhere;
static unsigned long long g_reserve_failures;
static unsigned long long g_map_failures;
static unsigned long long g_unmap_failures;

static void clear_error(void) {
    if (p_get_last_error != NULL) p_get_last_error();
}

static void set_access(void *ptr, size_t size) {
    /* Two locations: hipMalloc's memory is readable by the CPU at the same address, and torch's
     * .item() does that read. A range with an unmapped gap in it is rejected outright. */
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
        amdfq_logf("# vmm tail: host access refused at %p; GPU can use the page, host reads of it"
                   " (torch's .item() among them) will fault",
                   ptr);
        return;
    }
    clear_error();
    amdfq_logf("# vmm tail: hipMemSetAccess(%p, %zu) failed; the page may be unreachable", ptr, size);
}

static size_t slot_of(void *va) {
    uint64_t key = (uint64_t)(uintptr_t)va >> 12;
    key *= 0x9E3779B97F4A7C15ull;
    return (size_t)(key >> 40);
}

static void table_insert(struct table *t, void *va, size_t block, void *alt);

static int table_grow(struct table *t) {
    size_t old_cap = t->cap;
    struct rec *old = t->slots;
    size_t cap = old_cap == 0 ? AMDFQ_TABLE_MIN : old_cap * 2;
    struct rec *fresh = calloc(cap, sizeof *fresh);
    if (fresh == NULL) return -1;
    t->slots = fresh;
    t->cap = cap;
    t->used = 0;
    t->tombs = 0;
    for (size_t i = 0; i < old_cap; i++)
        if (old != NULL && old[i].va != NULL && old[i].va != REC_TOMB)
            table_insert(t, old[i].va, old[i].block, old[i].alt);
    free(old);
    return 0;
}

static void table_insert(struct table *t, void *va, size_t block, void *alt) {
    if ((t->used + t->tombs + 1) * 2 > t->cap && table_grow(t) != 0) {
        amdfq_logf("# vmm tail: registry grow failed; freeing this pointer will skip the guard");
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
            slot->alt = alt;
            t->used++;
            return;
        }
        if (slot->va == REC_TOMB) {
            if (tomb == SIZE_MAX) tomb = i;
        } else if (slot->va == va) {
            slot->block = block;
            slot->alt = alt;
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
    slot->alt = NULL;
    t->used--;
    t->tombs++;
}

static int resolve_all(void) {
    const char *missing = NULL;
    p_get_device = (__typeof__(p_get_device))amdfq_symbol("hipGetDevice");
    p_get_granularity = (__typeof__(p_get_granularity))amdfq_symbol("hipMemGetAllocationGranularity");
    p_address_reserve = (__typeof__(p_address_reserve))amdfq_symbol("hipMemAddressReserve");
    p_address_free = (__typeof__(p_address_free))amdfq_symbol("hipMemAddressFree");
    p_create = (__typeof__(p_create))amdfq_symbol("hipMemCreate");
    p_map = (__typeof__(p_map))amdfq_symbol("hipMemMap");
    p_unmap = (__typeof__(p_unmap))amdfq_symbol("hipMemUnmap");
    p_set_access = (__typeof__(p_set_access))amdfq_symbol("hipMemSetAccess");
    p_get_range = (__typeof__(p_get_range))amdfq_symbol("hipMemGetAddressRange");
    p_get_last_error = (__typeof__(p_get_last_error))amdfq_symbol("hipGetLastError");
    p_real_malloc = (__typeof__(p_real_malloc))amdfq_symbol("hipMalloc");
    p_real_free = (__typeof__(p_real_free))amdfq_symbol("hipFree");

    if (p_get_device == NULL) missing = "hipGetDevice";
    else if (p_get_granularity == NULL) missing = "hipMemGetAllocationGranularity";
    else if (p_address_reserve == NULL) missing = "hipMemAddressReserve";
    else if (p_address_free == NULL) missing = "hipMemAddressFree";
    else if (p_create == NULL) missing = "hipMemCreate";
    else if (p_map == NULL) missing = "hipMemMap";
    else if (p_unmap == NULL) missing = "hipMemUnmap";
    else if (p_set_access == NULL) missing = "hipMemSetAccess";
    else if (p_get_range == NULL) missing = "hipMemGetAddressRange";
    else if (p_get_last_error == NULL) missing = "hipGetLastError";
    else if (p_real_malloc == NULL) missing = "hipMalloc";
    else if (p_real_free == NULL) missing = "hipFree";
    if (missing != NULL) {
        amdfq_logf("# vmm tail: %s is missing from this runtime; allocations go unguarded", missing);
        return -1;
    }
    return 0;
}

static void ensure_ready(void) {
    if (g_state != TAIL_UNINIT) return;

    const char *env = getenv("AMDFQ_TAIL");
    if (env != NULL && env[0] == '0') {
        g_state = TAIL_OFF;
        g_off_reason = "AMDFQ_TAIL=0";
        amdfq_logf("# vmm tail: disabled by AMDFQ_TAIL=0");
        return;
    }

    if (resolve_all() != 0) {
        g_state = TAIL_OFF;
        g_off_reason = "nosym";
        return;
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
        g_state = TAIL_OFF;
        g_off_reason = "granularity";
        amdfq_logf("# vmm tail: hipMemGetAllocationGranularity failed; allocations go unguarded");
        return;
    }
    if (p_get_granularity(&recommended, &g_prop, AMDFQ_GRANULARITY_RECOMMENDED) != 0 ||
        recommended < minimum)
        recommended = minimum;
    g_granule = recommended;

    if (table_grow(&g_table) != 0 || table_grow(&g_ends) != 0) {
        g_state = TAIL_OFF;
        g_off_reason = "registry";
        amdfq_logf("# vmm tail: registry grow failed; allocations go unguarded");
        return;
    }
    if (p_create(&g_shared, g_granule, &g_prop, 0) != 0) {
        g_state = TAIL_OFF;
        g_off_reason = "shared-create";
        amdfq_logf("# vmm tail: hipMemCreate(%zu) failed; no page can be protected", g_granule);
        return;
    }

    g_owner = getpid();
    g_state = TAIL_ON;
    amdfq_logf("# vmm tail: pid %d device %d granularity %zu B; the page holding the 16 bytes after"
               " an allocation's end is reserved and mapped from one shared object whenever nothing"
               " else backs it — at hipMalloc, and again at hipFree for the allocation that ends"
               " where the freed block started",
               (int)g_owner, device, g_granule);
}

static int address_backed(uintptr_t addr) {
    void *base = NULL;
    size_t size = 0;
    if (p_get_range(&base, &size, (void *)addr) != 0) {
        clear_error();
        return 0;
    }
    return (uintptr_t)base <= addr && addr + 16 <= (uintptr_t)base + size;
}

static void *protect_page(uintptr_t addr) {
    void *va = NULL;
    if (p_address_reserve(&va, g_granule, g_granule, (void *)addr, 0) != 0) {
        clear_error();
        g_reserve_failures++;
        amdfq_logf("# vmm tail: hipMemAddressReserve(hint=%p, %zu) failed", (void *)addr, g_granule);
        return NULL;
    }
    if ((uintptr_t)va != addr) {
        g_elsewhere++;
        amdfq_logf("# vmm tail: %p not takeable (reserve gave %p); falling back to pad",
                   (void *)addr, va);
        p_address_free(va, g_granule);
        clear_error();
        return NULL;
    }
    if (p_map(va, g_granule, 0, g_shared, 0) != 0) {
        clear_error();
        g_map_failures++;
        amdfq_logf("# vmm tail: hipMemMap(%p, %zu) failed", va, g_granule);
        p_address_free(va, g_granule);
        clear_error();
        return NULL;
    }
    set_access(va, g_granule);
    return va;
}

static void unprotect_page(void *va) {
    if (p_unmap(va, g_granule) != 0) {
        clear_error();
        g_unmap_failures++;
        amdfq_logf("# vmm tail: hipMemUnmap(%p, %zu) failed", va, g_granule);
    }
    if (p_address_free(va, g_granule) != 0) {
        clear_error();
        amdfq_logf("# vmm tail: hipMemAddressFree(%p, %zu) failed", va, g_granule);
    }
}

static void heartbeat(void) {
    if (g_examined % AMDFQ_HEARTBEAT != 0) return;
    amdfq_logf("# vmm tail: examined %llu allocations, %llu ends already backed, %llu unaligned,"
               " %llu pages protected (%llu of them after a free), %llu released, %llu protected now,"
               " %llu re-taken with a pad (%llu failures), failed reserve/map/unmap %llu/%llu/%llu,"
               " hint missed %llu",
               g_examined, g_backed, g_unaligned, g_created, g_neighbour, g_released, g_protected,
               g_padded, g_pad_failures, g_reserve_failures, g_map_failures, g_unmap_failures,
               g_elsewhere);
}

static size_t extent_of(void *ptr, size_t size) {
    size_t block = (size + g_granule - 1) / g_granule * g_granule;
    void *base = NULL;
    size_t range = 0;
    if (p_get_range(&base, &range, ptr) == 0) {
        if (base == ptr && range > block) block = (range + g_granule - 1) / g_granule * g_granule;
    } else {
        clear_error();
    }
    return block;
}

static void register_block(void *ptr, size_t block, void *protect) {
    table_insert(&g_table, ptr, block, protect);
    table_insert(&g_ends, (void *)((uintptr_t)ptr + block), block, ptr);
}

int amdfq_tail_after_malloc(void **ptr, size_t *granted, int *error) {
    if (ptr == NULL || *ptr == NULL || granted == NULL || *granted == 0) return AMDFQ_TAIL_KEEP;
    pthread_mutex_lock(&g_lock);
    if (g_state == TAIL_UNINIT) ensure_ready();
    if (g_state != TAIL_ON || getpid() != g_owner) {
        pthread_mutex_unlock(&g_lock);
        return AMDFQ_TAIL_KEEP;
    }
    size_t block = extent_of(*ptr, *granted);
    uintptr_t end = (uintptr_t)*ptr + block;

    g_examined++;
    void *protect = NULL;
    int guarded = 1;
    if (end % g_granule != 0) {
        g_unaligned++;
    } else if (address_backed(end)) {
        g_backed++;
    } else if ((protect = protect_page(end)) != NULL) {
        g_created++;
        g_protected++;
    } else {
        guarded = 0;
    }

    if (guarded) {
        register_block(*ptr, block, protect);
        heartbeat();
        pthread_mutex_unlock(&g_lock);
        return AMDFQ_TAIL_KEEP;
    }

    /* The page behind this block is held by something we cannot move. Give it back and take a
     * padded one: the +16 bytes of §10.1 make the runtime reserve a granule more than the request. */
    size_t padded = *granted <= SIZE_MAX - AMDFQ_TAIL_PAD ? *granted + AMDFQ_TAIL_PAD : *granted;
    void *old = *ptr;
    hipError_t rc = p_real_free != NULL ? p_real_free(old) : HIP_ERROR_NOT_FOUND;
    if (rc != HIP_SUCCESS) {
        clear_error();
        g_pad_failures++;
        amdfq_logf("# vmm tail: %p could not be given back for a padded block (rc=%d); it stays"
                   " unguarded",
                   old, (int)rc);
        register_block(old, block, NULL);
        heartbeat();
        pthread_mutex_unlock(&g_lock);
        return AMDFQ_TAIL_KEEP;
    }

    void *fresh = NULL;
    rc = p_real_malloc != NULL ? p_real_malloc(&fresh, padded) : HIP_ERROR_NOT_FOUND;
    if (rc != HIP_SUCCESS || fresh == NULL) {
        clear_error();
        g_pad_failures++;
        amdfq_logf("# vmm tail: %p (block %zu B) was given back and the padded hipMalloc(%zu) failed:"
                   " rc=%d ptr=%p; the caller is told the allocation failed",
                   old, block, padded, (int)rc, fresh);
        if (error != NULL) *error = (int)rc;
        *ptr = NULL;
        pthread_mutex_unlock(&g_lock);
        return AMDFQ_TAIL_FAILED;
    }

    size_t fresh_block = extent_of(fresh, padded);
    *ptr = fresh;
    *granted = padded;
    g_padded++;
    amdfq_logf("# vmm tail: %p (block %zu B) had no page to take at %p; gave it back and took %p with"
               " a +%d B pad (%zu B asked, extent %zu B)",
               old, block, (void *)end, fresh, AMDFQ_TAIL_PAD, padded, fresh_block);
    register_block(fresh, fresh_block, NULL);
    heartbeat();
    pthread_mutex_unlock(&g_lock);
    return AMDFQ_TAIL_REPLACED;
}

void amdfq_tail_before_free(void *ptr) {
    if (ptr == NULL) return;
    pthread_mutex_lock(&g_lock);
    if (g_state != TAIL_ON) {
        pthread_mutex_unlock(&g_lock);
        return;
    }
    struct rec *record = table_find(&g_table, ptr);
    if (record == NULL) {
        pthread_mutex_unlock(&g_lock);
        return;
    }
    if (record->alt != NULL) {
        unprotect_page(record->alt);
        g_released++;
        g_protected--;
    }
    table_erase(&g_ends, (void *)((uintptr_t)ptr + record->block));
    table_erase(&g_table, ptr);
    pthread_mutex_unlock(&g_lock);
}

void amdfq_tail_after_free(void *ptr) {
    if (ptr == NULL) return;
    pthread_mutex_lock(&g_lock);
    if (g_state != TAIL_ON || g_granule == 0 || (uintptr_t)ptr % g_granule != 0) {
        pthread_mutex_unlock(&g_lock);
        return;
    }
    struct rec *end = table_find(&g_ends, ptr);
    if (end == NULL) {
        pthread_mutex_unlock(&g_lock);
        return;
    }
    struct rec *owner = table_find(&g_table, end->alt);
    if (owner == NULL || owner->alt != NULL || address_backed((uintptr_t)ptr)) {
        pthread_mutex_unlock(&g_lock);
        return;
    }
    void *protect = protect_page((uintptr_t)ptr);
    if (protect != NULL) {
        owner->alt = protect;
        g_neighbour++;
        g_protected++;
    }
    pthread_mutex_unlock(&g_lock);
}

void amdfq_tail_report(void) {
    if (g_state == TAIL_UNINIT) {
        amdfq_logf("# vmm tail: never used");
        return;
    }
    if (g_state == TAIL_OFF) {
        amdfq_logf("# vmm tail: off (%s)", g_off_reason != NULL ? g_off_reason : "disabled");
        return;
    }
    amdfq_logf("# vmm tail: one shared %zu B object behind every allocation end with nothing else"
               " behind it",
               g_granule);
    amdfq_logf("# vmm tail: examined %llu allocations, %llu ends already backed, %llu unaligned,"
               " %llu pages protected (%llu of them after a free), %llu released, %llu protected"
               " now, %llu re-taken with a +%d B pad (%llu failures)",
               g_examined, g_backed, g_unaligned, g_created, g_neighbour, g_released, g_protected,
               g_padded, AMDFQ_TAIL_PAD, g_pad_failures);
    amdfq_logf("# vmm tail: failed reservations %llu, maps %llu, unmaps %llu, hint missed %llu",
               g_reserve_failures, g_map_failures, g_unmap_failures, g_elsewhere);
}

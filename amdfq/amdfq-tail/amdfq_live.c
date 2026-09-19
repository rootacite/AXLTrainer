/* The live-allocation table. See amdfq_live.h. */
#include "amdfq_live.h"
#include "amdfq_log.h"

#include <pthread.h>
#include <stdlib.h>
#include <uthash.h>

struct amdfq_block {
    void *ptr;   /* the pointer hipMalloc handed the caller: the key */
    size_t size; /* what the caller asked for; the +16 pad sits on top of it */
    UT_hash_handle hh;
};

static struct amdfq_block *g_blocks;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t g_bytes;     /* sum of size over the table */
static size_t g_peak_count;
static size_t g_peak_bytes;
static size_t g_unmatched; /* frees of a pointer the table does not hold */
static size_t g_duplicate; /* inserts of a pointer it already holds */

size_t amdfq_live_state(size_t *bytes) {
    pthread_mutex_lock(&g_lock);
    size_t count = HASH_COUNT(g_blocks);
    if (bytes != NULL) *bytes = g_bytes;
    pthread_mutex_unlock(&g_lock);
    return count;
}

size_t amdfq_live_add(void *ptr, size_t size, size_t *bytes) {
    if (ptr == NULL) return amdfq_live_state(bytes);

    struct amdfq_block *block = malloc(sizeof *block);
    if (block == NULL) return amdfq_live_state(bytes);

    pthread_mutex_lock(&g_lock);
    block->ptr = ptr;
    block->size = size;

    struct amdfq_block *found = NULL;
    HASH_FIND_PTR(g_blocks, &ptr, found);
    if (found != NULL) {
        g_duplicate++;
        free(block);
    } else {
        HASH_ADD_PTR(g_blocks, ptr, block);
        g_bytes += size;
    }
    size_t count = HASH_COUNT(g_blocks);
    if (count > g_peak_count) g_peak_count = count;
    if (g_bytes > g_peak_bytes) g_peak_bytes = g_bytes;
    if (bytes != NULL) *bytes = g_bytes;
    pthread_mutex_unlock(&g_lock);
    return count;
}

size_t amdfq_live_del(void *ptr, size_t *bytes) {
    if (ptr == NULL) return amdfq_live_state(bytes);

    pthread_mutex_lock(&g_lock);
    struct amdfq_block *found = NULL;
    HASH_FIND_PTR(g_blocks, &ptr, found);
    if (found != NULL) {
        g_bytes -= found->size;
        HASH_DEL(g_blocks, found);
        free(found);
    } else {
        g_unmatched++;
    }
    size_t count = HASH_COUNT(g_blocks);
    if (bytes != NULL) *bytes = g_bytes;
    pthread_mutex_unlock(&g_lock);
    return count;
}

void amdfq_live_report(void) {
    pthread_mutex_lock(&g_lock);
    size_t count = HASH_COUNT(g_blocks);
    size_t bytes = g_bytes;
    size_t peak_count = g_peak_count;
    size_t peak_bytes = g_peak_bytes;
    size_t unmatched = g_unmatched;
    size_t duplicate = g_duplicate;
    pthread_mutex_unlock(&g_lock);

    /* A block still here at exit is one the process never released — the tail the caching allocator
     * keeps for reuse, or a leak. */
    amdfq_logf("# live blocks %zu (%zu bytes) at exit, peak %zu (%zu bytes),"
               " %zu frees with no entry, %zu duplicate inserts",
               count, bytes, peak_count, peak_bytes, unmatched, duplicate);
}

/*
 * The live-allocation table: one entry per block the caller still holds, keyed by the pointer
 * hipMalloc returned. uthash makes the *set* exact rather than inferred — a running count can only
 * say how many inserts and removes have happened, while the table can say which blocks are alive,
 * so a free of a pointer that was never allocated (or a second free of the same one) is visible
 * instead of being silently absorbed.
 *
 * Only hipMalloc is tracked: hipFree is the matching release. hipHostMalloc allocates host memory
 * whose release (hipHostFree) is not a gate this object hooks, so entries for it could never be
 * removed.
 *
 * The lock is held inside these calls only; none of them calls another with it held.
 */
#ifndef AMDFQ_LIVE_H
#define AMDFQ_LIVE_H

#include <stddef.h>

/* Insert (ptr, size) / remove ptr. Both return the number of live blocks after the change and write
 * their total size to *bytes. A NULL ptr changes nothing. */
size_t amdfq_live_add(void *ptr, size_t size, size_t *bytes);
size_t amdfq_live_del(void *ptr, size_t *bytes);

/* The same two numbers without touching the table. */
size_t amdfq_live_state(size_t *bytes);

void amdfq_live_report(void);

#endif /* AMDFQ_LIVE_H */

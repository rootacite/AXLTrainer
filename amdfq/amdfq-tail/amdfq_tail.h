/*
 * The tail guard (amdfq/doc/amdfq.md §13): after every hipMalloc, the page holding the 16 bytes past the
 * allocation's end is queried, and reserved and mapped from one shared object only where the
 * runtime backs nothing, so a bf16 kernel that over-reads its operand lands in memory that exists.
 *
 * Nothing here changes what the caller gets — same pointer, same size, same layout — except on the
 * path where that page cannot be had (in practice: the runtime's own reserved pool address space).
 * There the block is given back and re-allocated with AMDFQ_TAIL_PAD bytes added, which is §10.1's
 * pad for that block alone (~0.2 % of allocations).
 *
 * On by default. AMDFQ_TAIL=0 turns it into a no-op so the original step-10 fault can be reproduced.
 */
#ifndef AMDFQ_TAIL_H
#define AMDFQ_TAIL_H

#include <stddef.h>

enum { AMDFQ_TAIL_KEEP = 0, AMDFQ_TAIL_REPLACED = 1, AMDFQ_TAIL_FAILED = -1 };

/* Returns AMDFQ_TAIL_KEEP when the caller's pointer stands, AMDFQ_TAIL_REPLACED when *ptr is a
 * padded block that must be used instead, and AMDFQ_TAIL_FAILED when the block was given back and
 * the padded re-allocation did not come back — then *ptr is NULL and *error holds the hipError_t
 * to report. *granted is the size the runtime was asked for the block it gets. */
int amdfq_tail_after_malloc(void **ptr, size_t *granted, int *error);
void amdfq_tail_before_free(void *ptr);
void amdfq_tail_after_free(void *ptr);

void amdfq_tail_report(void);

#endif /* AMDFQ_TAIL_H */

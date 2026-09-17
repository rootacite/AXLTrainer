/*
 * LD_PRELOAD hook on the ROCm memory-management gates of a training run. `init` opens the log in
 * /tmp (never stdout: the trainer's stdout is the NDJSON control channel in `api.py`), `fini` writes
 * the live-allocation summary, the tail-guard summary, then the per-function call summary, and
 * closes the log. Hooked calls are interposed in amdfq_hooks_hip.c and described in amdfq_gates.h.
 */
#include "amdfq_live.h"
#include "amdfq_log.h"
#include "amdfq_tail.h"

__attribute__((constructor)) static void init(void) {
    amdfq_log_open();
}

__attribute__((destructor)) static void fini(void) {
    amdfq_live_report();
    amdfq_tail_report();
    amdfq_log_close();
}

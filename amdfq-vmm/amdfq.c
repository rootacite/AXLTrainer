/*
 * LD_PRELOAD hook on the ROCm memory-management gates of a training run. `init` opens the log in
 * /tmp (never stdout: the trainer's stdout is the NDJSON control channel in `api.py`), `fini` writes
 * the VMM module's closing summary, then the per-function call summary, and closes the log. Hooked
 * calls are interposed in amdfq_hooks_hip.c and described in amdfq_gates.h; each one is logged and
 * forwarded unchanged.
 */
#include "amdfq.h"
#include "amdfq_log.h"
#include "amdfq_vmm.h"

__attribute__((constructor)) static void init(void) {
    amdfq_log_open();
}

__attribute__((destructor)) static void fini(void) {
    amdfq_vmm_report();
    amdfq_log_close();
}

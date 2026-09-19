/*
 * Logging and symbol resolution, shared by every interposer.
 *
 * Nothing here ever touches stdout: the log is a file (or FIFO) opened in the library constructor
 * and closed in the destructor. One line per hooked call, written with a single write(2), so a run
 * that dies on a SIGABRT still leaves everything up to the fatal call on disk.
 */
#ifndef AMDFQ_LOG_H
#define AMDFQ_LOG_H

#include <stddef.h>

/* One id per hooked function, in the order the closing summary prints them. */
enum amdfq_fn {
    AMDFQ_FN_HIP_MALLOC,
    AMDFQ_FN_HIP_FREE,
    AMDFQ_FN_HIP_HOST_MALLOC,
    AMDFQ_FN_UNRESOLVED, /* a gate whose real implementation dlsym could not find */
    AMDFQ_FN_COUNT
};

void amdfq_log_open(void);
void amdfq_log_close(void);
void amdfq_logf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void amdfq_tick(enum amdfq_fn fn);

void *amdfq_symbol(const char *name);
const char *amdfq_caller(void *return_address);

/* Caller of the *hooked* function: inside the hook body, level 0 of the return-address stack is the
 * frame that called us, which is the code the log wants to attribute the call to. */
#define AMDFQ_CALLER() amdfq_caller(__builtin_return_address(0))

/* Resolve the real implementation once per interposing function. RTLD_NEXT starts its search after
 * this object, so it can never hand back the interposer itself. */
#define AMDFQ_REAL(name)                                                                     \
    static __typeof__(name) *real;                                                           \
    if (real == NULL)                                                                        \
        real = (__typeof__(name) *)amdfq_symbol(#name);

#endif /* AMDFQ_LOG_H */

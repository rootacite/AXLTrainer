/* Logging and symbol resolution for the amdfq preload. See amdfq_log.h. */
#define _GNU_SOURCE

#include "amdfq_log.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define AMDFQ_FD_UNINIT (-2) /* constructor has not run yet */
#define AMDFQ_FD_RETRY (-3)  /* FIFO with no reader attached yet */
#define AMDFQ_FD_DEAD (-1)
#define AMDFQ_LINE_MAX 512

static _Atomic int g_fd = AMDFQ_FD_UNINIT;
static _Atomic unsigned long g_seq;
static _Atomic unsigned long g_dropped;
static _Atomic unsigned long long g_hits[AMDFQ_FN_COUNT];
static struct timespec g_t0;
static char g_path[256];

static const char *const g_fn_names[AMDFQ_FN_COUNT] = {
    [AMDFQ_FN_HIP_MALLOC] = "hipMalloc",
    [AMDFQ_FN_HIP_FREE] = "hipFree",
    [AMDFQ_FN_HIP_HOST_MALLOC] = "hipHostMalloc",
    [AMDFQ_FN_UNRESOLVED] = "<unresolved>",
};

static const char *log_path(void) {
    if (g_path[0] == '\0') {
        const char *env = getenv("AMDFQ_LOG");
        if (env != NULL && env[0] != '\0')
            snprintf(g_path, sizeof g_path, "%s", env);
        else
            snprintf(g_path, sizeof g_path, "/tmp/amdfq-hook-%d.log", (int)getpid());
    }
    return g_path;
}

/* O_RDWR on a FIFO never blocks for a reader and can never raise SIGPIPE in the trainer when the
 * reader goes away; with nobody attached the buffer fills up instead and lines are dropped. */
static int open_log_file(void) {
    const char *path = log_path();
    struct stat st;
    if (stat(path, &st) == 0 && S_ISFIFO(st.st_mode))
        return open(path, O_RDWR | O_NONBLOCK);
    return open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
}

/* Once, on stderr (never stdout: that is api.py's NDJSON channel), so a run that produced no log
 * says why instead of looking like the hooks never fired. */
static void warn_once(int error) {
    char buf[320];
    int n = snprintf(buf, sizeof buf, "amdfq: cannot open log %s: %s; hooking continues unlogged\n",
                     log_path(), strerror(error));
    if (n > 0) {
        size_t len = (size_t)n < sizeof buf ? (size_t)n : sizeof buf - 1;
        ssize_t ignored = write(STDERR_FILENO, buf, len);
        (void)ignored;
    }
}

static int log_fd(void) {
    int fd = atomic_load_explicit(&g_fd, memory_order_acquire);
    if (fd >= 0 || fd == AMDFQ_FD_DEAD) return fd;

    int opened = open_log_file();
    if (opened < 0) {
        int error = errno;
        int want = error == ENXIO ? AMDFQ_FD_RETRY : AMDFQ_FD_DEAD;
        int expected = fd;
        if (atomic_compare_exchange_strong(&g_fd, &expected, want) && want == AMDFQ_FD_DEAD)
            warn_once(error);
        return -1;
    }
    int expected = fd;
    if (!atomic_compare_exchange_strong(&g_fd, &expected, opened)) {
        close(opened);
        return expected >= 0 ? expected : -1;
    }
    return opened;
}

static double elapsed_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)(t.tv_sec - g_t0.tv_sec) + (double)(t.tv_nsec - g_t0.tv_nsec) / 1e9;
}

void amdfq_logf(const char *fmt, ...) {
    char buf[AMDFQ_LINE_MAX];

    int prefix = snprintf(buf, sizeof buf, "%06lu %12.6f T=%-6u ",
                          atomic_fetch_add(&g_seq, 1) + 1, elapsed_s(), (unsigned)gettid());
    if (prefix < 0) return;
    size_t used = (size_t)prefix < sizeof buf - 2 ? (size_t)prefix : sizeof buf - 2;

    va_list ap;
    va_start(ap, fmt);
    int body = vsnprintf(buf + used, sizeof buf - used - 1, fmt, ap);
    va_end(ap);
    if (body > 0) {
        size_t room = sizeof buf - 2 - used;
        used += (size_t)body < room ? (size_t)body : room;
    }
    buf[used++] = '\n';

    int fd = log_fd();
    if (fd < 0) {
        atomic_fetch_add(&g_dropped, 1);
        return;
    }
    size_t off = 0;
    while (off < used) {
        ssize_t n = write(fd, buf + off, used - off);
        if (n > 0) {
            off += (size_t)n;
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            atomic_fetch_add(&g_dropped, 1);
            return;
        }
    }
}

void amdfq_tick(enum amdfq_fn fn) {
    if ((unsigned)fn < AMDFQ_FN_COUNT) atomic_fetch_add(&g_hits[fn], 1);
}

void *amdfq_symbol(const char *name) {
    dlerror();
    void *symbol = dlsym(RTLD_NEXT, name);
    const char *error = dlerror();
    if (symbol == NULL || error != NULL) {
        amdfq_tick(AMDFQ_FN_UNRESOLVED);
        amdfq_logf("%s: dlsym(RTLD_NEXT) failed: %s", name, error != NULL ? error : "no symbol");
        return NULL;
    }
    return symbol;
}

/* dli_fbase comes from the loader's link_map, i.e. the real ELF base — the one amdfq.md §7.4 says to
 * trust, unlike a base recomputed from a raw module list. */
const char *amdfq_caller(void *return_address) {
    static __thread char buf[192];
    Dl_info info;

    if (return_address != NULL && dladdr(return_address, &info) != 0 && info.dli_fname != NULL) {
        const char *slash = strrchr(info.dli_fname, '/');
        unsigned long offset = (unsigned long)((const char *)return_address - (const char *)info.dli_fbase);
        snprintf(buf, sizeof buf, "%s+0x%lx", slash != NULL ? slash + 1 : info.dli_fname, offset);
    } else {
        snprintf(buf, sizeof buf, "0x%lx", (unsigned long)return_address);
    }
    return buf;
}

void amdfq_log_open(void) {
    clock_gettime(CLOCK_MONOTONIC, &g_t0);
    amdfq_logf("# amdfq preload, pid %d ppid %d, log %s", (int)getpid(), (int)getppid(), log_path());
    amdfq_logf("# %d allocation gates hooked; hipMalloc is forwarded unchanged, then the page past"
               " each block is guarded (AMDFQ_TAIL=0 disables); one line per"
               " call, fields: seq elapsed_s T=tid <fn>(args) -> ret=... caller=<object>+offset, where"
               " the offset is into the object the loader says backs the calling frame",
               AMDFQ_FN_UNRESOLVED - AMDFQ_FN_HIP_MALLOC);
}

void amdfq_log_close(void) {
    unsigned long long total = 0;
    unsigned long dropped = atomic_load(&g_dropped);

    for (int i = 0; i < AMDFQ_FN_COUNT; i++) {
        unsigned long long hits = atomic_load(&g_hits[i]);
        if (hits == 0) continue;
        total += hits;
        amdfq_logf("# calls %-32s %llu", g_fn_names[i], hits);
    }
    amdfq_logf("# total hooked calls %llu, %lu lines dropped, log closed", total, dropped);

    int fd = atomic_exchange(&g_fd, AMDFQ_FD_DEAD);
    if (fd >= 0) close(fd);
}

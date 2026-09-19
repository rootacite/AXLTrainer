/* Replays an amdfq allocation stream in plain HIP: no LD_PRELOAD, no hook, no torch.
 *
 * The stream is not compiled in.  The program reads the op table gen_ops.py writes, so the same
 * binary replays a log of a thousand allocations or a million — the table path is the only thing
 * that has to change.
 *
 *   ./replay ops.tsv [--pace X] [--limit N] [--pad N] [--exit] [--hold]
 *
 *     --pace X    put the calls on the trainer's own timeline: op i happens X × (t_i - t_first)
 *                 seconds after the replay starts, so bursts stay bursts and the pauses stay
 *                 pauses.  Default 1.0 (the run's real rhythm); 0 runs flat out.  Needs the table's
 *                 `t` column; without it the replay runs flat out and says so.
 *     --limit N   run the first N ops only (the old stream's step-5 window is op 4853)
 *     --pad N     ask the runtime for size + N on every hipMalloc (0 = the stream as collected)
 *     --exit      return as soon as the stream is done (for scripts); without it the replay idles,
 *                 holding the allocations, until Ctrl+C
 *     --hold      SIGSTOP instead of idling — a stopped process for outside readers
 *     --quiet     no per-op lines (the heartbeat and the summary still print)
 *
 * After the last op the allocations stay live on purpose: an outside reader can then take frozen
 * readings (or a driver-side BO dump) against exactly the state the trainer's stream ends in.  The
 * process only gives them back when the user ends it with Ctrl+C.
 *
 * Twice a second it prints the two accounts this whole investigation is about, to *stderr* so a run
 * can be watched while it happens without the per-op lines getting in the way (stdout is the op log
 * and the closing summary; status lines go to stderr too):
 *
 *     ledger free vram   what hipMemGetInfo says this process can still get — resolved per 2 MiB
 *     driver free vram   sysfs mem_info_vram_total - mem_info_vram_used — what the driver thinks is
 *                        still free on the device, ROCr's ~2 GiB reserve included in "used"
 *     the difference     ledger minus driver: negative while ROCr holds its reserve against the
 *                        driver's account, equal once a non-zero pad breaks 2 MiB alignment
 *
 * Every op prints the same line shape the hook wrote, with `caller=replay`, so the replay's own
 * accounting (live / live_bytes at each op) can be diffed against frozen.log line by line.
 */

#include <hip/hip_runtime.h>

#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <unistd.h>
#include <time.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const double HEARTBEAT_S = 0.5;

struct Op {
    unsigned long seq;
    char kind;
    size_t size;
    long slot;
    unsigned flags;
    int ret;
    double t;          /* the hook's clock, when the table carries it */
    bool has_t;
};

struct Account {
    size_t live = 0;
    size_t live_bytes = 0;
};

/* The driver's own counters, read from sysfs: no HIP context of its own, so reading them does not
 * move what they measure. */
struct DriverCounters {
    char used_path[256] = {0};
    char total_path[256] = {0};
    char label[64] = {0};

    bool open() {
        DIR *dir = opendir("/sys/class/drm");
        if (dir == nullptr) return false;
        bool found = false;
        while (struct dirent *entry = readdir(dir)) {
            if (strncmp(entry->d_name, "card", 4) != 0) continue;
            char used[256], total[256];
            snprintf(used, sizeof used, "/sys/class/drm/%s/device/mem_info_vram_used", entry->d_name);
            snprintf(total, sizeof total, "/sys/class/drm/%s/device/mem_info_vram_total", entry->d_name);
            if (access(used, R_OK) == 0 && access(total, R_OK) == 0) {
                snprintf(used_path, sizeof used_path, "%s", used);
                snprintf(total_path, sizeof total_path, "%s", total);
                snprintf(label, sizeof label, "%s", entry->d_name);
                found = true;
                break;
            }
        }
        closedir(dir);
        return found;
    }

    bool read(unsigned long long *used, unsigned long long *total) const {
        if (used_path[0] == '\0') return false;
        FILE *u = fopen(used_path, "r");
        FILE *t = fopen(total_path, "r");
        if (u == nullptr || t == nullptr) {
            if (u) fclose(u);
            if (t) fclose(t);
            return false;
        }
        bool ok = fscanf(u, "%llu", used) == 1 && fscanf(t, "%llu", total) == 1;
        fclose(u);
        fclose(t);
        return ok;
    }
};

volatile sig_atomic_t g_signal = 0;   /* 0 while running; the signal that asked us to stop */

void on_signal(int signo) { g_signal = signo; }

double elapsed_since(const struct timespec &t0) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - t0.tv_sec) + (double)(now.tv_nsec - t0.tv_nsec) / 1e9;
}

void wait_until(double target, const struct timespec &t0) {
    double wait = target - elapsed_since(t0);
    if (wait <= 0.0001) return;               /* inside a burst: no syscall, no drift */
    struct timespec nap;
    nap.tv_sec = (time_t)wait;
    nap.tv_nsec = (long)((wait - (double)nap.tv_sec) * 1e9);
    while (nanosleep(&nap, &nap) == -1 && errno == EINTR && g_signal == 0) {
    }
}

/* Same prefix as amdfq/amdfq-tail/amdfq_log.c, so the two logs line up in a diff. */
void emit(unsigned long seq, double elapsed, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

void emit(unsigned long seq, double elapsed, const char *fmt, ...) {
    char body[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof body, fmt, ap);
    va_end(ap);
    printf("%06lu %12.6f T=%-6u %s\n", seq, elapsed, (unsigned)gettid(), body);
}

const char *mib(long long bytes, char *scratch, size_t size) {
    snprintf(scratch, size, "%.1f", (double)bytes / (1024.0 * 1024.0));
    return scratch;
}

void heartbeat(double elapsed, const char *phase, const Account &account,
               const DriverCounters &drivers) {
    size_t ledger_free = 0, ledger_total = 0;
    if (hipMemGetInfo(&ledger_free, &ledger_total) != hipSuccess) ledger_free = 0;
    unsigned long long used = 0, total = 0;
    bool have_driver = drivers.read(&used, &total);
    long long driver_free = have_driver ? (long long)total - (long long)used : 0;
    long long diff = have_driver ? (long long)ledger_free - driver_free : 0;
    char a[32], b[32], c[32], d[32];
    fprintf(stderr, "# %8.2fs %-6s live=%zu/%zu B | ledger free vram %zu B (%s MiB) | "
            "driver free vram %lld B (%s MiB) | ledger - driver %lld B (%s MiB) | "
            "driver vram_used %llu B (%s MiB)%s\n",
           elapsed, phase, account.live, account.live_bytes,
           ledger_free, mib((long long)ledger_free, a, sizeof a),
           driver_free, have_driver ? mib(driver_free, b, sizeof b) : "n/a",
           diff, have_driver ? mib(diff, c, sizeof c) : "n/a",
           used, have_driver ? mib((long long)used, d, sizeof d) : "n/a",
           have_driver ? "" : " (driver counters unreadable)");
    fflush(stderr);
}

const char *usage =
    "usage: replay ops.tsv [--pace X] [--limit N] [--pad N] [--exit] [--hold] [--quiet]\n";

std::vector<Op> read_table(const char *path) {
    FILE *file = fopen(path, "r");
    if (file == nullptr) {
        fprintf(stderr, "replay: cannot open %s: %s\n", path, strerror(errno));
        exit(2);
    }
    std::vector<Op> ops;
    char line[256];
    while (fgets(line, sizeof line, file) != nullptr) {
        if (line[0] == '#' || line[0] == '\n') continue;
        Op op{};
        char kind = 0;
        int got = sscanf(line, "%lu\t%c\t%zu\t%ld\t%x\t%d\t%lf",
                         &op.seq, &kind, &op.size, &op.slot, &op.flags, &op.ret, &op.t);
        if (got == 6) {
            op.has_t = false;
        } else if (got == 7) {
            op.has_t = true;
        } else {
            fprintf(stderr, "replay: unparsable table line: %s", line);
            exit(2);
        }
        op.kind = kind;
        ops.push_back(op);
    }
    fclose(file);
    return ops;
}

}  // namespace

int main(int argc, char **argv) {
    const char *path = nullptr;
    long limit = -1;
    size_t pad = 0;
    double pace = 1.0;
    bool hold = false;
    bool exit_after = false;
    bool quiet = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pace") == 0 && i + 1 < argc) pace = strtod(argv[++i], nullptr);
        else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc) limit = strtol(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "--pad") == 0 && i + 1 < argc) pad = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "--exit") == 0) exit_after = true;
        else if (strcmp(argv[i], "--hold") == 0) hold = true;
        else if (strcmp(argv[i], "--quiet") == 0) quiet = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { fputs(usage, stdout); return 0; }
        else if (argv[i][0] == '-') { fprintf(stderr, "replay: unknown option %s\n%s", argv[i], usage); return 2; }
        else if (path == nullptr) path = argv[i];
        else { fprintf(stderr, "replay: more than one table given\n%s", usage); return 2; }
    }
    if (path == nullptr) { fputs(usage, stderr); return 2; }

    std::vector<Op> ops = read_table(path);
    if (limit >= 0 && (size_t)limit < ops.size()) ops.resize((size_t)limit);

    long max_slot = 0;
    for (const Op &op : ops) if (op.slot > max_slot) max_slot = op.slot;
    std::vector<void *> slot_ptr((size_t)max_slot + 1, nullptr);
    std::vector<bool> slot_live((size_t)max_slot + 1, false);

    bool timeable = false;
    double t_first = 0.0;
    for (const Op &op : ops) {
        if (op.has_t) { timeable = true; t_first = op.t; break; }
    }
    if (pace > 0 && !timeable) {
        printf("# no `t` column in this table: running flat out (no pacing)\n");
        pace = 0;
    }

    DriverCounters drivers;
    bool have_drivers = drivers.open();

    /* The hold is the point of this program, so a closing terminal or a reader that goes away must
     * not end it: only SIGINT (Ctrl+C) and SIGTERM do.  SIGHUP is ignored and SIGPIPE is left
     * ignored as well, so a `| head` on the output cannot kill the process mid-hold. */
    signal(SIGHUP, SIG_IGN);
    signal(SIGPIPE, SIG_IGN);
    struct sigaction stop_action{};
    stop_action.sa_handler = on_signal;
    sigaction(SIGINT, &stop_action, nullptr);
    sigaction(SIGTERM, &stop_action, nullptr);
    sigaction(SIGQUIT, &stop_action, nullptr);

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    Account account;
    size_t failures = 0;
    double next_beat = 0.0;

    printf("# replay: %zu ops, pace=%.2fx, pad=%zu; driver counters %s; SIGHUP/SIGPIPE ignored, "
           "Ctrl+C to end\n", ops.size(), pace, pad,
           have_drivers ? drivers.label : "unreadable");
    fflush(stdout);
    heartbeat(elapsed_since(t0), "start", account, drivers);
    next_beat = HEARTBEAT_S;

    for (const Op &op : ops) {
        if (g_signal != 0) break;                     /* Ctrl+C also stops the replay itself */
        if (pace > 0 && op.has_t) wait_until((op.t - t_first) * pace, t0);
        const double t = elapsed_since(t0);
        if (op.kind == 'M') {
            size_t want = op.size + pad;
            void *ptr = nullptr;
            hipError_t rc = hipMalloc(&ptr, want);
            if (rc != hipSuccess) {
                failures++;
                ptr = nullptr;
            } else if (ptr != nullptr) {
                slot_ptr[(size_t)op.slot] = ptr;
                slot_live[(size_t)op.slot] = true;
                account.live++;
                account.live_bytes += op.size;   /* the hook charges the request, not the pad */
            }
            if (!quiet)
                emit(op.seq, t, "hipMalloc(size=%zu) -> ptr=%p ret=%d live=%zu live_bytes=%zu caller=replay",
                     op.size, ptr, (int)rc, account.live, account.live_bytes);
        } else if (op.kind == 'H') {
            void *ptr = nullptr;
            hipError_t rc = hipHostMalloc(&ptr, op.size, op.flags);
            if (rc != hipSuccess) { failures++; ptr = nullptr; }
            else if (ptr != nullptr) {
                slot_ptr[(size_t)op.slot] = ptr;
                slot_live[(size_t)op.slot] = true;
            }
            if (!quiet)
                emit(op.seq, t, "hipHostMalloc(size=%zu flags=0x%x) -> ptr=%p ret=%d caller=replay",
                     op.size, op.flags, ptr, (int)rc);
        } else if (op.kind == 'F') {
            void *ptr = nullptr;
            if (op.slot >= 0 && (size_t)op.slot < slot_ptr.size() && slot_live[(size_t)op.slot])
                ptr = slot_ptr[(size_t)op.slot];
            hipError_t rc = hipFree(ptr);
            if (rc != hipSuccess) failures++;
            if (op.slot >= 0 && (size_t)op.slot < slot_live.size() &&
                slot_live[(size_t)op.slot]) {
                slot_live[(size_t)op.slot] = false;
                if (rc == hipSuccess && account.live > 0) {
                    account.live--;
                    account.live_bytes -= op.size;
                }
            }
            if (!quiet)
                emit(op.seq, t, "hipFree(ptr=%p) -> ret=%d live=%zu live_bytes=%zu caller=replay",
                     ptr, (int)rc, account.live, account.live_bytes);
        }
        if (t >= next_beat) {
            heartbeat(t, "replay", account, drivers);
            next_beat = t + HEARTBEAT_S;
        }
    }

    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        free_bytes = total_bytes = 0;
    }
    printf("# replay done: ops=%zu live=%zu live_bytes=%zu histogram_failures=%zu "
           "vis_free=%zu vis_total=%zu elapsed=%.3f s pad=%zu pace=%.2fx\n",
           ops.size(), account.live, account.live_bytes, failures, free_bytes, total_bytes,
           elapsed_since(t0), pad, pace);
    fflush(stdout);
    heartbeat(elapsed_since(t0), "held", account, drivers);

    if (hold) {
        fprintf(stderr, "# stopping (SIGSTOP) with the allocations live\n");
        fflush(stderr);
        raise(SIGSTOP);
    }
    if (!exit_after) {
        fprintf(stderr, "# holding %zu live allocations (%zu B); Ctrl+C to end\n",
                account.live, account.live_bytes);
        fflush(stderr);
        while (g_signal == 0) {
            struct timespec tick{0, (long)(HEARTBEAT_S * 1e9)};
            while (nanosleep(&tick, &tick) == -1 && errno == EINTR && g_signal == 0) {
            }
            heartbeat(elapsed_since(t0), "held", account, drivers);
        }
        fprintf(stderr, "# interrupted by signal %d: exiting, the %zu allocations go back with"
                " the process\n", (int)g_signal, account.live);
        fflush(stderr);
    }
    return failures == 0 ? 0 : 1;
}

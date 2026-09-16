/*
 * Which kernels overrun their operands?  The rocBLAS call of
 * launch_fault_via_rocblas.hip, run once per matrix shape, with every operand's last
 * byte made the last mapped byte, so any access at or past the end of an operand hits
 * an unmapped page and ROCr names the kernel that made it.
 *
 * One shape per child process, because the fault kills the process that takes it
 * (hip1.md §12.1: ROCr prints its line, then the process aborts).  The driver never
 * touches the GPU beyond asking for the device name; it forks and execs itself with
 * `--shape M N K --trans ab`, reads the child's own output from a pipe, and records why
 * the child stopped.  A faulting shape therefore costs one process, not the sweep.
 *
 * What a fault here means, and what it does not.  The guard page sits behind the whole
 * operand, and the operands are laid out column-major with lda = m / ldb = n exactly,
 * so what reaches it is an overrun in the operand's *last* column.  An edge overrun in
 * an interior column lands in the next column's first rows — still inside the buffer —
 * which is why this is a detector for the class routes 1 and 2 faulted on (their two
 * faults are that last-column case) and not a proof that a kernel never reads out of
 * bounds anywhere else.  Nothing here proves a fault in production either: on the real
 * tensors these overruns read the neighbouring allocation instead, silently.  The
 * distance reported below is where the first access that left the operand landed, which
 * is a lower bound on how far the kernel goes: a fault on the guard page's own first
 * byte says the kernel's first access after the end was that byte, not that it stayed
 * within one page of the operand.
 *
 * The child prints what the driver needs to read a fault address against:
 *
 *     OPERAND A size=788480 base=0x…  end=0x…  reserved_end=0x…
 *
 * `end` is the first byte past the operand, always the first byte of the guard page, and
 * the reservation runs GUARD_BYTES further, unmapped, so an overrun deeper than one page
 * still lands in our own address space and is measured instead of being reported as a
 * stray address.  `CLEAN` means the sync returned; `STATUS n` is a rocBLAS status, i.e.
 * this shape found no kernel at all.
 *
 * Transposes: `--trans` takes the two flags as n/t.  Each pair selects a different
 * Tensile library (Ailk/Bljk versus Alik/Bjlk — the table in launch_fault_via_rocblas.hip
 * has the mapping), so the sweep defaults to all four and visits four kernel populations
 * per shape.
 *
 * bf16 only, like both repro programs: the fp16 and f32 variants were withdrawn
 * (hip1.md §12.12).
 *
 * Build and run (see launch_fault_via_rocblas.hip for the rocBLAS link line — the three
 * entry points are declared below for the same reason, no ROCm wheel ships the headers):
 *
 *     hipcc --offload-arch=gfx1201 -O2 fixes/hip1/sweep_gemm_shapes.hip \
 *         -o /tmp/sweep_gemm_shapes -L<rocBLAS lib dir> -lrocblas
 *     /tmp/sweep_gemm_shapes                       # the default list, all four transposes
 *     /tmp/sweep_gemm_shapes --list                # the run list, without running it
 *     /tmp/sweep_gemm_shapes --m 1:64:1 --n 8:256:8 --k 1:512:8
 *
 * The driver forks; it must be started from a writable cwd for nothing in particular,
 * but /proc/self/exe has to resolve, so do not run it from a deleted directory.
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

extern "C" {
typedef struct _rocblas_handle* rocblas_handle;
typedef int32_t rocblas_int;
typedef enum {
    rocblas_operation_none = 111,
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
} rocblas_operation;
typedef enum {
    rocblas_datatype_f32_r = 151,
    rocblas_datatype_bf16_r = 168,
} rocblas_datatype;
typedef enum {
    rocblas_status_success = 0,
} rocblas_status;
typedef enum {
    rocblas_gemm_algo_standard = 0,
} rocblas_gemm_algo;
rocblas_status rocblas_create_handle(rocblas_handle* handle);
rocblas_status rocblas_destroy_handle(rocblas_handle handle);
rocblas_status rocblas_gemm_ex(rocblas_handle handle, rocblas_operation trans_a,
                               rocblas_operation trans_b, rocblas_int m, rocblas_int n, rocblas_int k,
                               const void* alpha, const void* a, rocblas_datatype a_type,
                               rocblas_int lda, const void* b, rocblas_datatype b_type,
                               rocblas_int ldb, const void* beta, const void* c,
                               rocblas_datatype c_type, rocblas_int ldc, void* d,
                               rocblas_datatype d_type, rocblas_int ldd,
                               rocblas_datatype compute_type, rocblas_gemm_algo algo,
                               int32_t solution_index, uint32_t flags);
}

#define HIPCHECK(call)                                                                     \
    do {                                                                                   \
        hipError_t e_ = (call);                                                            \
        if (e_ != hipSuccess) {                                                            \
            fprintf(stderr, "%s:%d  %s -> %s\n", __FILE__, __LINE__, #call,                \
                    hipGetErrorString(e_));                                                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// Unmapped address space reserved behind the guard page.  One page is enough to fault;
// the rest is so that a deeper overrun stays inside a reservation we can name.
static const size_t GUARD_BYTES = 1u << 20;

struct Operand {
    char name;
    size_t size;
    unsigned char* base;
    unsigned char* end;  // first byte of the guard page == first byte past the operand
    unsigned char* reserved_end;
};

// Reserve `size` bytes so that the last one is the last byte of a mapping.  The lead-in is
// (gran - size % gran) % gran bytes, a multiple of 16 for every operand here, which keeps
// the 16-byte vector loads the kernels make aligned.
static Operand guarded(char name, size_t size) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;

    size_t gran = 0;
    HIPCHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    const size_t mapped = ((size + gran - 1) / gran) * gran;

    hipMemGenericAllocationHandle_t handle;
    HIPCHECK(hipMemCreate(&handle, mapped, &prop, 0));
    void* base = nullptr;
    HIPCHECK(hipMemAddressReserve(&base, mapped + GUARD_BYTES, gran, nullptr, 0));
    HIPCHECK(hipMemMap(base, mapped, 0, handle, 0));

    hipMemAccessDesc desc = {};
    desc.location = prop.location;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIPCHECK(hipMemSetAccess(base, mapped, &desc, 1));

    Operand op;
    op.name = name;
    op.size = size;
    op.base = (unsigned char*)base + (mapped - size);
    op.end = (unsigned char*)base + mapped;
    op.reserved_end = op.end + GUARD_BYTES;
    printf("OPERAND %c size=%llu base=0x%llx end=0x%llx reserved_end=0x%llx\n", op.name,
           (unsigned long long)op.size, (unsigned long long)(uintptr_t)op.base,
           (unsigned long long)(uintptr_t)op.end, (unsigned long long)(uintptr_t)op.reserved_end);
    return op;
}

struct Trans {
    rocblas_operation a;
    rocblas_operation b;
    const char* name;
};

static const Trans TRANS[] = {
    {rocblas_operation_none, rocblas_operation_none, "nn"},
    {rocblas_operation_none, rocblas_operation_transpose, "nt"},
    {rocblas_operation_transpose, rocblas_operation_none, "tn"},
    {rocblas_operation_transpose, rocblas_operation_transpose, "tt"},
};
static const int TRANS_COUNT = (int)(sizeof(TRANS) / sizeof(TRANS[0]));

static const Trans* find_trans(const char* name) {
    for (int i = 0; i < TRANS_COUNT; i++)
        if (strcmp(TRANS[i].name, name) == 0) return &TRANS[i];
    return nullptr;
}

// --------------------------------------------------------------------------- one shape

static int run_one(int m, int n, int k, const Trans& t) {
    int device = 0;
    HIPCHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIPCHECK(hipGetDeviceProperties(&prop, device));
    printf("DEVICE %s | %s | %d CUs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);

    // Column-major operands, the byte sizes the shape has: A is lda x <k or m>, B is
    // ldb x <n or k>, D is m x n.  lda/ldb are the exact leading dimensions, so each
    // operand ends at a column boundary — the layout the two routes faulted on.
    const rocblas_int lda = (t.a == rocblas_operation_none) ? m : k;
    const rocblas_int ldb = (t.b == rocblas_operation_none) ? k : n;
    const rocblas_int ldd = m;
    const size_t a_cols = (t.a == rocblas_operation_none) ? (size_t)k : (size_t)m;
    const size_t b_cols = (t.b == rocblas_operation_none) ? (size_t)n : (size_t)k;

    printf("SHAPE m=%d n=%d k=%d trans=%s lda=%d ldb=%d ldd=%d\n", m, n, k, t.name, lda, ldb, ldd);

    Operand a = guarded('A', (size_t)lda * a_cols * 2);
    Operand b = guarded('B', (size_t)ldb * b_cols * 2);
    Operand d = guarded('D', (size_t)ldd * n * 2);

    // Zeros: no NaN or denormal can make the selection or the kernel's arithmetic differ
    // between runs, and the values are irrelevant — only the addresses are measured.
    HIPCHECK(hipMemset(a.base, 0, a.size));
    HIPCHECK(hipMemset(b.base, 0, b.size));
    HIPCHECK(hipMemset(d.base, 0, d.size));

    float alpha = 1.0f;
    float beta = 0.0f;

    rocblas_handle handle = nullptr;
    if (rocblas_create_handle(&handle) != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle failed\n");
        return 2;
    }

    printf("LAUNCH\n");
    fflush(stdout);

    rocblas_status st =
        rocblas_gemm_ex(handle, t.a, t.b, m, n, k, &alpha, a.base, rocblas_datatype_bf16_r, lda,
                        b.base, rocblas_datatype_bf16_r, ldb, &beta, d.base,
                        rocblas_datatype_bf16_r, ldd, d.base, rocblas_datatype_bf16_r, ldd,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (st != rocblas_status_success) {
        printf("STATUS %d\n", (int)st);
        return 3;
    }

    // The fault lands here; ROCr names the kernel and aborts.
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("HIPERR %d %s\n", (int)err, hipGetErrorString(err));
        return 4;
    }

    printf("CLEAN\n");
    rocblas_destroy_handle(handle);
    return 0;
}

// ------------------------------------------------------------------- driver: one child

struct Outcome {
    std::string output;  // everything the child and ROCr wrote
    bool exited = false; // false: killed by a signal
    int code = 0;        // exit status, or the signal number
};

// Precomputed before the fork so that the child does nothing but dup2/execv, the only
// calls that are safe between fork and exec.
static Outcome run_child(const std::vector<std::string>& argv) {
    int fds[2];
    if (pipe(fds) != 0) {
        perror("pipe");
        exit(1);
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);
    }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        alarm(600);  // a hung child dies as signal 14 instead of stalling the sweep
        std::vector<char*> raw;
        for (const std::string& s : argv) raw.push_back(const_cast<char*>(s.c_str()));
        raw.push_back(nullptr);
        execv(raw[0], raw.data());
        _exit(127);
    }

    close(fds[1]);
    Outcome out;
    char buf[4096];
    ssize_t got;
    while ((got = read(fds[0], buf, sizeof buf)) > 0) out.output.append(buf, (size_t)got);
    close(fds[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        out.exited = true;
        out.code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        out.exited = false;
        out.code = WTERMSIG(status);
    }
    return out;
}

static bool hex_after(const std::string& text, const char* key, unsigned long long* value) {
    size_t at = text.find(key);
    if (at == std::string::npos) return false;
    at = text.find("0x", at + strlen(key));
    if (at == std::string::npos) return false;
    char* stop = nullptr;
    const unsigned long long parsed = strtoull(text.c_str() + at, &stop, 16);
    if (stop == text.c_str() + at) return false;
    *value = parsed;
    return true;
}

static std::string text_after(const std::string& text, const char* key, const char* stops) {
    size_t at = text.find(key);
    if (at == std::string::npos) return "";
    at += strlen(key);
    size_t end = text.find_first_of(stops, at);
    if (end == std::string::npos) end = text.size();
    std::string value = text.substr(at, end - at);
    const size_t first = value.find_first_not_of(" \t");
    const size_t last = value.find_last_not_of(" \t");
    return first == std::string::npos ? "" : value.substr(first, last - first + 1);
}

struct Fault {
    char operand = '?';
    unsigned long long delta = 0;  // bytes from the operand's end to the faulting access
    std::string description;       // what the driver managed to work out, for the row
};

// Read the fault address against the operands the child printed: the reservation tells us
// which operand the access belonged to even when it went past the guard page.
static Fault classify(const std::string& output) {
    Fault fault;
    unsigned long long addr = 0;
    if (!hex_after(output, "faulting addr", &addr)) {
        fault.description = "no faulting address in the child's output";
        return fault;
    }

    struct Range {
        char name;
        unsigned long long base, end, reserved_end;
    };
    std::vector<Range> ranges;
    size_t at = 0;
    while ((at = output.find("OPERAND ", at)) != std::string::npos) {
        size_t line_end = output.find('\n', at);
        if (line_end == std::string::npos) line_end = output.size();
        const std::string line = output.substr(at, line_end - at);
        char name = '?';
        unsigned long long size = 0, base = 0, end = 0, reserved_end = 0;
        if (sscanf(line.c_str(),
                   "OPERAND %c size=%llu base=0x%llx end=0x%llx reserved_end=0x%llx", &name, &size,
                   &base, &end, &reserved_end) == 5)
            ranges.push_back({name, base, end, reserved_end});
        at += 8;
    }

    for (const Range& r : ranges) {
        if (addr >= r.end && addr < r.reserved_end) {
            fault.operand = r.name;
            fault.delta = addr - r.end;
            char buf[192];
            if (fault.delta == 0)
                snprintf(buf, sizeof buf,
                         "past %c: the first byte after the operand (a read or write there, so "
                         "the overrun is at least 1 B)",
                         r.name);
            else
                snprintf(buf, sizeof buf,
                         "past %c: %llu B past the end, skipping %llu whole guard page%s, so the "
                         "overrun is at least that far",
                         r.name, fault.delta, fault.delta / 4096, fault.delta / 4096 == 1 ? "" : "s");
            fault.description = buf;
            return fault;
        }
    }
    for (const Range& r : ranges) {
        if (addr >= r.base && addr < r.end) {
            fault.operand = r.name;
            fault.description = "inside an operand — not an overrun";
            return fault;
        }
    }

    char buf[96];
    snprintf(buf, sizeof buf, "0x%llx, in no operand's reservation", addr);
    fault.description = buf;
    return fault;
}

// How the child stopped, as one of the two things ROCr and the exit status distinguish:
// a signal is the fault itself, a status above 2 is the rocBLAS call refusing the shape.
static std::string stop_reason(const Outcome& out) {
    return (out.exited ? "exit=" : "signal=") + std::to_string(out.code);
}

// ---------------------------------------------------------------------- the shape list

static std::vector<int> parse_list(const char* spec, const char* what) {
    std::vector<int> values;
    std::string s(spec);
    if (s.find(':') != std::string::npos) {
        int from = 0, to = 0, step = 1;
        if (sscanf(spec, "%d:%d:%d", &from, &to, &step) < 2 || step == 0)
            { fprintf(stderr, "--%s: bad range '%s'\n", what, spec); exit(1); }
        for (long long v = from; step > 0 ? v <= to : v >= to; v += step) values.push_back((int)v);
    } else {
        size_t at = 0;
        while (at <= s.size()) {
            const size_t comma = s.find(',', at);
            const std::string item = s.substr(at, comma == std::string::npos ? std::string::npos
                                                                             : comma - at);
            if (!item.empty()) values.push_back(atoi(item.c_str()));
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    if (values.empty()) {
        fprintf(stderr, "--%s: '%s' lists no value\n", what, spec);
        exit(1);
    }
    for (int v : values)
        if (v < 1) {
            fprintf(stderr, "--%s: %d is not a positive dimension\n", what, v);
            exit(1);
        }
    return values;
}

struct Shape {
    int m, n, k;
    const Trans* trans;
};

#define DEFAULT_M "1,16,32,48,64,128,256,512,1024,2048"
#define DEFAULT_K "1,8,32,64,128,308,512"

static void usage(const char* prog) {
    printf("usage: %s [options]                         sweep, one child process per run\n"
           "       %s --shape M N K --trans ab           one shape in this process (internal)\n"
           "\n"
           "  --m LIST   --n LIST   --k LIST   values (1,2,48) or a range (1:64:8)\n"
           "  --trans L    nn, nt, tn, tt — comma separated, or all (default all)\n"
           "  --start N    skip the first N runs of the list\n"
           "  --limit N    stop after N runs\n"
           "  --faults-only  print only the runs that faulted\n"
           "  --list       print the run list and exit without running it\n"
           "\n"
           "defaults: --m %s --n %s --k %s --trans all\n",
           prog, prog, DEFAULT_M, DEFAULT_M, DEFAULT_K);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    const char* m_spec = DEFAULT_M;
    const char* n_spec = DEFAULT_M;
    const char* k_spec = DEFAULT_K;
    std::string trans_spec = "all";
    long start = 0, limit = -1;
    bool faults_only = false, list_only = false;
    int child_m = 0, child_n = 0, child_k = 0;
    const Trans* child_trans = nullptr;
    bool child = false;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const char* next = i + 1 < argc ? argv[i + 1] : nullptr;
        auto need = [&](void) {
            if (!next) {
                fprintf(stderr, "%s needs a value\n", arg.c_str());
                exit(1);
            }
        };
        if (arg == "--m") { need(); m_spec = next; i++; }
        else if (arg == "--n") { need(); n_spec = next; i++; }
        else if (arg == "--k") { need(); k_spec = next; i++; }
        else if (arg == "--trans") { need(); trans_spec = next; i++; }
        else if (arg == "--start") { need(); start = atol(next); i++; }
        else if (arg == "--limit") { need(); limit = atol(next); i++; }
        else if (arg == "--faults-only") faults_only = true;
        else if (arg == "--list") list_only = true;
        else if (arg == "--shape") {
            if (i + 3 >= argc) {
                fprintf(stderr, "--shape needs M N K\n");
                exit(1);
            }
            child = true;
            child_m = atoi(argv[++i]);
            child_n = atoi(argv[++i]);
            child_k = atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (child) {
        child_trans = find_trans(trans_spec.c_str());
        if (!child_trans) {
            fprintf(stderr, "--trans '%s' is not one of nn, nt, tn, tt\n", trans_spec.c_str());
            return 1;
        }
        return run_one(child_m, child_n, child_k, *child_trans);
    }

    std::vector<const Trans*> trans_list;
    if (trans_spec == "all") {
        for (int i = 0; i < TRANS_COUNT; i++) trans_list.push_back(&TRANS[i]);
    } else {
        size_t at = 0;
        while (at <= trans_spec.size()) {
            const size_t comma = trans_spec.find(',', at);
            const std::string item = trans_spec.substr(
                at, comma == std::string::npos ? std::string::npos : comma - at);
            const Trans* t = find_trans(item.c_str());
            if (!t) {
                fprintf(stderr, "--trans '%s' is not one of nn, nt, tn, tt (or all)\n",
                        trans_spec.c_str());
                return 1;
            }
            trans_list.push_back(t);
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }

    const std::vector<int> ms = parse_list(m_spec, "m");
    const std::vector<int> ns = parse_list(n_spec, "n");
    const std::vector<int> ks = parse_list(k_spec, "k");

    std::vector<Shape> shapes;
    for (const Trans* t : trans_list)
        for (int m : ms)
            for (int n : ns)
                for (int k : ks) shapes.push_back({m, n, k, t});

    if (list_only) {
        for (size_t i = 0; i < shapes.size(); i++)
            printf("%6zu  m=%d n=%d k=%d %s\n", i, shapes[i].m, shapes[i].n, shapes[i].k,
                   shapes[i].trans->name);
        printf("%zu runs\n", shapes.size());
        return 0;
    }

    int device = 0;
    HIPCHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIPCHECK(hipGetDeviceProperties(&prop, device));

    std::string self = "/proc/self/exe";
    char resolved[4096];
    const ssize_t len = readlink(self.c_str(), resolved, sizeof resolved - 1);
    if (len > 0) {
        resolved[len] = '\0';
        self = resolved;
    }

    printf("device    : %s | %s | %d CUs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("sweep     : m %s  n %s  k %s\n", m_spec, n_spec, k_spec);
    printf("            trans %s, %zu shapes, from run %ld%s\n", trans_spec.c_str(), shapes.size(),
           start, limit >= 0 ? ", capped by --limit" : "");
    printf("            one child process per run; a run that faults is the finding\n\n");

    struct KernelStat {
        long count = 0;
        unsigned long long min_delta = 0, max_delta = 0;
        std::string operands;
        std::string example;
    };
    std::map<std::string, KernelStat> kernels;
    long clean = 0, faulted = 0, other = 0;
    if (start < 0) start = 0;
    if ((size_t)start > shapes.size()) start = (long)shapes.size();
    const size_t stop = limit >= 0 ? std::min(shapes.size(), (size_t)(start + limit)) : shapes.size();

    for (size_t i = (size_t)start; i < stop; i++) {
        const Shape& s = shapes[i];
        char mb[16], nb[16], kb[16];
        snprintf(mb, sizeof mb, "%d", s.m);
        snprintf(nb, sizeof nb, "%d", s.n);
        snprintf(kb, sizeof kb, "%d", s.k);
        const Outcome out = run_child({self, "--shape", mb, nb, kb, "--trans", s.trans->name});

        const bool fault_line = out.output.find("Memory Fault Error") != std::string::npos;
        if (fault_line) {
            const Fault fault = classify(out.output);
            const std::string kernel = text_after(out.output, "kernel:", "]\n");
            faulted++;
            KernelStat& stat = kernels[kernel];
            stat.count++;
            if (stat.count == 1 || fault.delta < stat.min_delta) stat.min_delta = fault.delta;
            if (fault.delta > stat.max_delta) stat.max_delta = fault.delta;
            if (stat.operands.find(fault.operand) == std::string::npos)
                stat.operands += fault.operand;
            char example[64];
            snprintf(example, sizeof example, "m=%d n=%d k=%d %s", s.m, s.n, s.k, s.trans->name);
            if (stat.example.empty()) stat.example = example;

            printf("[%5zu/%zu] m=%-5d n=%-5d k=%-5d %s  FAULT %-10s  %s\n  kernel=%s\n", i + 1, stop,
                   s.m, s.n, s.k, s.trans->name, stop_reason(out).c_str(), fault.description.c_str(),
                   kernel.c_str());
        } else if (out.exited && out.code == 0 && out.output.find("CLEAN") != std::string::npos) {
            clean++;
            if (!faults_only)
                printf("[%5zu/%zu] m=%-5d n=%-5d k=%-5d %s  clean\n", i + 1, stop, s.m, s.n, s.k,
                       s.trans->name);
        } else {
            other++;
            const std::string status = text_after(out.output, "\nSTATUS ", "\n");
            const std::string hip = text_after(out.output, "\nHIPERR ", "\n");
            printf("[%5zu/%zu] m=%-5d n=%-5d k=%-5d %s  %-10s  rocblas status=%s  hiperr=%s\n", i + 1,
                   stop, s.m, s.n, s.k, s.trans->name, stop_reason(out).c_str(),
                   status.empty() ? "-" : status.c_str(), hip.empty() ? "-" : hip.c_str());
        }
    }

    printf("\n== summary ==\n");
    printf("runs            : %zu\n", stop - (size_t)start);
    printf("clean           : %ld\n", clean);
    printf("faulted         : %ld\n", faulted);
    printf("no kernel/error : %ld\n", other);
    printf("distinct faulting kernels: %zu\n", kernels.size());

    std::vector<std::pair<std::string, KernelStat>> ordered(kernels.begin(), kernels.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& l, const auto& r) {
        return l.second.count > r.second.count;
    });
    for (const auto& entry : ordered) {
        printf("\n  %ld shape%s, operand%s %s, first access past the end at %llu..%llu B, e.g. %s\n",
               entry.second.count, entry.second.count == 1 ? "" : "s",
               entry.second.operands.size() > 1 ? "s" : "", entry.second.operands.c_str(),
               entry.second.min_delta, entry.second.max_delta, entry.second.example.c_str());
        printf("  %s\n", entry.first.c_str());
    }
    return 0;
}

// Demonstrates the address-space split from HIP.md §5: reserve a large range of device
// virtual addresses with nothing behind it, map one small subrange, then read the
// still-unmapped tail on purpose.
//
//   hipcc -O3 tools/hip/hip_vmm_tail_read.hip -o /tmp/hip_vmm_tail_read && /tmp/hip_vmm_tail_read
//
// The last step is meant to fault the GPU and kill this process; everything before it is
// printed unbuffered so the log survives. See HIP.md §5 (the API) and §6 (the pitfalls).
#include <hip/hip_runtime.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr size_t kReserveBytes = 4ull << 30;  // virtual addresses only, no physical memory
constexpr size_t kMapBytes = 64ull << 20;     // the one subrange that gets real VRAM
constexpr size_t kTailOffset = 2ull << 30;    // where the deliberate read happens

int step_no = 0;

void step(const char* what) {
    std::printf("\n[step %d] %s\n", ++step_no, what);
    std::fflush(stdout);
}

void say(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("        ");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
    std::fflush(stdout);
}

void fatal(hipError_t e, const char* expr) {
    std::printf("        FAIL %s: %s\n", expr, hipGetErrorString(e));
    std::fflush(stdout);
    std::exit(1);
}

#define CHECK(expr)                                            \
    do {                                                       \
        hipError_t e_ = (expr);                                \
        if (e_ != hipSuccess) fatal(e_, #expr);                \
        std::printf("        ok   %s\n", #expr);                \
        std::fflush(stdout);                                   \
    } while (0)

// /proc/self/maps is the authority on the PROT bits of a range: the kernel writes one line
// per VMA with the exact rwxp + p/s mode, and an address with no line has no mapping at all.
const char* vma_line(unsigned long addr, char* out, size_t out_size) {
    FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return nullptr;
    char line[512];
    const char* found = nullptr;
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long lo = 0, hi = 0;
        if (std::sscanf(line, "%lx-%lx", &lo, &hi) != 2) continue;
        if (addr >= lo && addr < hi) {
            std::snprintf(out, out_size, "%s", line);
            found = out;
            break;
        }
    }
    std::fclose(f);
    return found;
}

void describe_mode(const char* mode, char* out, size_t out_size) {
    std::snprintf(out, out_size, "%s  [%s, %s, %s, %s]", mode,
                  mode[0] == 'r' ? "PROT_READ" : "no read",
                  mode[1] == 'w' ? "PROT_WRITE" : "no write",
                  mode[2] == 'x' ? "PROT_EXEC" : "no exec",
                  mode[3] == 's' ? "MAP_SHARED" : "MAP_PRIVATE");
}

// The PROT answer for one address, from the OS, plus what HIP says about the same address.
void report_prot(const char* label, const void* addr, const hipMemLocation& loc) {
    unsigned long a = reinterpret_cast<unsigned long>(addr);
    char line[512] = {};
    const char* hit = vma_line(a, line, sizeof(line));
    std::printf("        %s %p\n", label, addr);
    if (hit) {
        char mode[8] = {};
        if (std::sscanf(line, "%*s %7s", mode) == 1) {
            char desc[128];
            describe_mode(mode, desc, sizeof(desc));
            std::printf("          /proc/self/maps: PROT = %s\n", desc);
        }
        std::printf("          %s", line);
    } else {
        std::printf("          /proc/self/maps: no VMA covers this address "
                    "-> unmapped, effective PROT_NONE\n");
    }
    unsigned long long flags = 0;
    hipError_t e = hipMemGetAccess(&flags, &loc, const_cast<void*>(addr));
    std::printf("          hipMemGetAccess  : %s (flags %llu)\n", hipGetErrorString(e), flags);
    hipPointerAttribute_t attr = {};
    e = hipPointerGetAttributes(&attr, addr);
    std::printf("          hipPointerGetAttributes: %s%s\n", hipGetErrorString(e),
                e == hipSuccess
                    ? (attr.type == hipMemoryTypeDevice ? " (type hipMemoryTypeDevice)"
                                                        : " (type host)")
                    : "");
    std::fflush(stdout);
}

// Every VMA that overlaps [lo, hi), so the reserved range can be read off directly.
void show_range(unsigned long lo, unsigned long hi) {
    std::printf("        VMAs overlapping %p..%p:\n", reinterpret_cast<void*>(lo),
                reinterpret_cast<void*>(hi));
    FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[512];
    bool any = false;
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long a = 0, b = 0;
        if (std::sscanf(line, "%lx-%lx", &a, &b) != 2) continue;
        if (a < hi && b > lo) {
            std::printf("          %s", line);
            any = true;
        }
    }
    if (!any) std::printf("          (none)\n");
    std::fclose(f);
    std::fflush(stdout);
}

void show_vram(const char* when) {
    size_t free_bytes = 0, total = 0;
    CHECK(hipMemGetInfo(&free_bytes, &total));
    std::printf("        %-24s free %6.2f GiB / total %6.2f GiB\n", when,
                free_bytes / 1073741824.0, total / 1073741824.0);
    std::fflush(stdout);
}

__global__ void write_pattern(int* p, int n, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// The deliberate read. One thread, one load, at an address with no GPU page-table entry.
__global__ void read_unmapped(const int* p, int* sink) {
    *sink = *reinterpret_cast<const volatile int*>(p);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // this process is expected to die mid-log
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    hipDeviceProp_t prop_dev = {};
    CHECK(hipGetDeviceProperties(&prop_dev, 0));
    int rt = 0;
    CHECK(hipRuntimeGetVersion(&rt));
    std::printf("device        : %s (%s)\n", prop_dev.name, prop_dev.gcnArchName);
    std::printf("HIP header    : %d.%d.%d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR,
                HIP_VERSION_PATCH);
    std::printf("HIP runtime   : %d.%d.%d\n", rt / 10000000, (rt / 100000) % 100, rt % 100000);
    std::printf("plan          : reserve %.1f GiB of device VA, map %.0f MiB of it, read at "
                "+%.1f GiB\n",
                kReserveBytes / 1073741824.0, kMapBytes / 1048576.0,
                kTailOffset / 1073741824.0);

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;

    step("VRAM before anything");
    show_vram("baseline");

    step("hipMemGetAllocationGranularity");
    size_t gran = 0;
    CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    say("granularity: %zu B", gran);

    step("hipMemAddressReserve: 4 GiB of address space, no physical memory");
    void* base = nullptr;
    CHECK(hipMemAddressReserve(&base, kReserveBytes, gran, nullptr, 0));
    say("reserved range : %p .. %p (%zu B)", base,
        static_cast<void*>(static_cast<char*>(base) + kReserveBytes), kReserveBytes);
    show_vram("after reserve");
    show_range(reinterpret_cast<unsigned long>(base),
               reinterpret_cast<unsigned long>(base) + kReserveBytes);

    step("hipMemCreate: the only physical allocation, 64 MiB");
    hipMemGenericAllocationHandle_t handle = nullptr;
    CHECK(hipMemCreate(&handle, kMapBytes, &prop, 0));
    say("allocation handle: %p", reinterpret_cast<void*>(handle));
    show_vram("after hipMemCreate");

    step("hipMemMap: map that allocation into the first 64 MiB of the reservation");
    CHECK(hipMemMap(base, kMapBytes, 0, handle, 0));
    say("mapped subrange: %p .. %p", base,
        static_cast<void*>(static_cast<char*>(base) + kMapBytes));
    say("tail stays reserved and unmapped: %p .. %p",
        static_cast<void*>(static_cast<char*>(base) + kMapBytes),
        static_cast<void*>(static_cast<char*>(base) + kReserveBytes));
    show_range(reinterpret_cast<unsigned long>(base),
               reinterpret_cast<unsigned long>(base) + kReserveBytes);

    step("hipMemSetAccess on the mapped subrange only");
    hipMemAccessDesc desc = {};
    desc.location = prop.location;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    CHECK(hipMemSetAccess(base, kMapBytes, &desc, 1));
    unsigned long long mapped_flags = 0;
    hipError_t ge = hipMemGetAccess(&mapped_flags, &desc.location, base);
    say("hipMemGetAccess(base= %p): %s, flags %llu", base, hipGetErrorString(ge),
        mapped_flags);

    step("prove the mapped subrange is live: kernel writes, host reads back");
    int words = static_cast<int>(kMapBytes / sizeof(int));
    write_pattern<<<(words + 255) / 256, 256>>>(static_cast<int*>(base), words, 0x2b);
    CHECK(hipDeviceSynchronize());
    int host_read = 0;
    CHECK(hipMemcpy(&host_read, base, sizeof(int), hipMemcpyDeviceToHost));
    say("read back at %p: %d (expect 43)", base, host_read);
    if (host_read != 0x2b) {
        std::printf("        FAIL the mapped subrange does not hold the written value\n");
        return 1;
    }

    void* tail = static_cast<char*>(base) + kTailOffset;

    step("PROT of the mapped subrange, for contrast");
    report_prot("mapped address", base, prop.location);
    report_prot("last mapped word",
                static_cast<char*>(base) + kMapBytes - sizeof(int), prop.location);

    step("PROT of the unmapped tail, before reading it");
    report_prot("tail address", tail, prop.location);
    report_prot("first byte past the mapping", static_cast<char*>(base) + kMapBytes,
                prop.location);

    step("read the unmapped tail on purpose");
    int* sink = nullptr;
    CHECK(hipMalloc(&sink, sizeof(int)));
    say("reading %p from a device kernel; sink at %p", tail, sink);
    std::printf("        expected: a GPU memory access fault at %p, this process dies\n", tail);
    std::fflush(stdout);
    read_unmapped<<<1, 1>>>(static_cast<const int*>(tail), sink);
    CHECK(hipDeviceSynchronize());
    int sink_value = 0;
    CHECK(hipMemcpy(&sink_value, sink, sizeof(int), hipMemcpyDeviceToHost));
    say("survived: the read returned %d -- the tail was not unmapped after all", sink_value);
    return 0;
}

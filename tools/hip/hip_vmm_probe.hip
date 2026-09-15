// Probe for HIP's virtual memory management API (hipMemAddressReserve / hipMemCreate /
// hipMemMap / hipMemSetAccess) and for POSIX-fd export/import of an allocation.
//
//   hipcc -O3 tools/hip/hip_vmm_probe.hip -o /tmp/hip_vmm_probe && /tmp/hip_vmm_probe
//
// Prints PASS/FAIL and exits non-zero on failure. See HIP.md §5.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;

// Fatal: further steps would be meaningless.
#define REQUIRE(expr)                                                              \
    do {                                                                           \
        hipError_t e_ = (expr);                                                    \
        std::printf("%-4s %-58s %s\n", e_ == hipSuccess ? "ok" : "FAIL", #expr,    \
                    hipGetErrorString(e_));                                        \
        if (e_ != hipSuccess) std::exit(1);                                        \
    } while (0)

// Non-fatal: the call is itself the thing under observation.
#define TRY(expr)                                                                  \
    do {                                                                           \
        hipError_t e_ = (expr);                                                    \
        std::printf("%-4s %-58s %s\n", e_ == hipSuccess ? "ok" : "FAIL", #expr,    \
                    hipGetErrorString(e_));                                        \
        if (e_ != hipSuccess) failures++;                                          \
    } while (0)

// The kernel mapping for a VMM range is the one the runtime installed in the process;
// its flags are how the CPU-side access state becomes visible.
static void show_mapping(const void* ptr) {
    unsigned long addr = reinterpret_cast<unsigned long>(ptr);
    FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long lo = 0, hi = 0;
        if (std::sscanf(line, "%lx-%lx", &lo, &hi) != 2) continue;
        if (addr >= lo && addr < hi) {
            std::printf("     %s", line);
            break;
        }
    }
    std::fclose(f);
}

static size_t align_up(size_t bytes, const hipMemAllocationProp& prop, size_t* gran) {
    REQUIRE(hipMemGetAllocationGranularity(gran, &prop, hipMemAllocationGranularityMinimum));
    return ((bytes + *gran - 1) / *gran) * *gran;
}

__global__ void fill(float* p, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

static void device_cycle() {
    std::printf("\n=== device allocation, 256 MiB ===\n");
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;

    size_t gran = 0, size = align_up(256ull << 20, prop, &gran);
    std::printf("     granularity %zu, rounded size %zu\n", gran, size);

    hipMemGenericAllocationHandle_t handle;
    REQUIRE(hipMemCreate(&handle, size, &prop, 0));

    void* ptr = nullptr;
    REQUIRE(hipMemAddressReserve(&ptr, size, gran, nullptr, 0));
    REQUIRE(hipMemMap(ptr, size, 0, handle, 0));
    std::printf("     mapping after hipMemMap (CPU access is not granted yet):\n");
    show_mapping(ptr);

    hipMemAccessDesc desc = {};
    desc.location = prop.location;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    REQUIRE(hipMemSetAccess(ptr, size, &desc, 1));

    unsigned long long flags = 0;
    REQUIRE(hipMemGetAccess(&flags, &desc.location, ptr));
    std::printf("     access flags %llu (3 = read-write)\n", flags);

    hipMemGenericAllocationHandle_t back = nullptr;
    REQUIRE(hipMemRetainAllocationHandle(&back, ptr));
    std::printf("     retained handle is %s the created one\n", back == handle ? "equal to" : "NOT");
    if (back != handle) failures++;
    REQUIRE(hipMemRelease(back));

    int n = static_cast<int>(size / sizeof(float));
    fill<<<(n + 255) / 256, 256>>>(static_cast<float*>(ptr), n, 3.0f);
    REQUIRE(hipDeviceSynchronize());
    float host_v = 0.0f;
    REQUIRE(hipMemcpy(&host_v, static_cast<float*>(ptr) + 12345, sizeof(float),
                      hipMemcpyDeviceToHost));
    std::printf("     kernel wrote %f at element 12345\n", host_v);
    if (host_v != 3.0f) failures++;

    REQUIRE(hipMemUnmap(ptr, size));
    REQUIRE(hipMemAddressFree(ptr, size));
    REQUIRE(hipMemRelease(handle));
}

static void host_location() {
    std::printf("\n=== host allocation, 4 MiB ===\n");
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeHost;

    size_t gran = 0, size = align_up(4ull << 20, prop, &gran);
    hipMemGenericAllocationHandle_t handle;
    REQUIRE(hipMemCreate(&handle, size, &prop, 0));

    void* ptr = nullptr;
    REQUIRE(hipMemAddressReserve(&ptr, size, gran, nullptr, 0));
    REQUIRE(hipMemMap(ptr, size, 0, handle, 0));
    std::printf("     mapping after hipMemMap (PROT_NONE: touching it would fault):\n");
    show_mapping(ptr);

    hipMemAccessDesc desc = {};
    desc.location = prop.location;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    REQUIRE(hipMemSetAccess(ptr, size, &desc, 1));
    std::printf("     mapping after hipMemSetAccess (CPU read-write):\n");
    show_mapping(ptr);

    // Observation, not a requirement: this query returns hipErrorInvalidValue for a host
    // location on 7.14.60850 even though hipMemSetAccess succeeded.
    unsigned long long flags = 0;
    std::printf("     hipMemGetAccess with a host location: %s\n",
                hipGetErrorString(hipMemGetAccess(&flags, &desc.location, ptr)));

    std::memset(ptr, 0xAB, 4096);
    auto* b = static_cast<unsigned char*>(ptr);
    std::printf("     CPU wrote and read back: %02x %02x (expect ab ab)\n", b[0], b[4095]);
    if (b[0] != 0xAB || b[4095] != 0xAB) failures++;

    // No hipMemcpy/hipMemset on this pointer from here: the runtime classifies the mapping
    // as hipMemoryTypeDevice (see HIP.md §6), so a copy faults the GPU at that address.
    hipPointerAttribute_t attr = {};
    TRY(hipPointerGetAttributes(&attr, ptr));
    std::printf("     hipPointerGetAttributes reports %s\n",
                attr.type == hipMemoryTypeDevice ? "hipMemoryTypeDevice"
                                                 : "something other than Device");

    REQUIRE(hipMemUnmap(ptr, size));
    REQUIRE(hipMemAddressFree(ptr, size));
    REQUIRE(hipMemRelease(handle));
}

static void ipc_export() {
    std::printf("\n=== POSIX fd export/import, 8 MiB ===\n");
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;

    size_t gran = 0, size = align_up(8ull << 20, prop, &gran);
    hipMemGenericAllocationHandle_t handle;
    REQUIRE(hipMemCreate(&handle, size, &prop, 0));

    int fd = -1;
    TRY(hipMemExportToShareableHandle(&fd, handle, hipMemHandleTypePosixFileDescriptor, 0));
    std::printf("     exported fd: %d\n", fd);
    if (fd < 0) {
        REQUIRE(hipMemRelease(handle));
        return;
    }

    hipMemGenericAllocationHandle_t imported = nullptr;
    TRY(hipMemImportFromShareableHandle(&imported, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
                                        hipMemHandleTypePosixFileDescriptor));
    if (imported) {
        void* ptr = nullptr;
        void* ptr2 = nullptr;
        REQUIRE(hipMemAddressReserve(&ptr, size, gran, nullptr, 0));
        REQUIRE(hipMemAddressReserve(&ptr2, size, gran, nullptr, 0));
        REQUIRE(hipMemMap(ptr, size, 0, handle, 0));
        REQUIRE(hipMemMap(ptr2, size, 0, imported, 0));

        hipMemAccessDesc desc = {};
        desc.location = prop.location;
        desc.flags = hipMemAccessFlagsProtReadWrite;
        REQUIRE(hipMemSetAccess(ptr, size, &desc, 1));
        REQUIRE(hipMemSetAccess(ptr2, size, &desc, 1));

        // Write through one mapping, read through the other: same physical allocation.
        REQUIRE(hipMemset(ptr, 0x5A, 4096));
        REQUIRE(hipDeviceSynchronize());
        char probe = 0;
        REQUIRE(hipMemcpy(&probe, ptr2, 1, hipMemcpyDeviceToHost));
        std::printf("     the imported mapping sees %02x (expect 5a)\n",
                    static_cast<unsigned char>(probe));
        if (static_cast<unsigned char>(probe) != 0x5A) failures++;

        REQUIRE(hipMemUnmap(ptr2, size));
        REQUIRE(hipMemAddressFree(ptr2, size));
        REQUIRE(hipMemRelease(imported));
        REQUIRE(hipMemUnmap(ptr, size));
        REQUIRE(hipMemAddressFree(ptr, size));
    }
    REQUIRE(hipMemRelease(handle));
}

int main() {
    hipDeviceProp_t p = {};
    REQUIRE(hipGetDeviceProperties(&p, 0));
    std::printf("device        : %s (%s)\n", p.name, p.gcnArchName);
    int rt = 0;
    REQUIRE(hipRuntimeGetVersion(&rt));
    std::printf("HIP header    : %d.%d.%d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR,
                HIP_VERSION_PATCH);
    std::printf("HIP runtime   : %d.%d.%d\n", rt / 10000000, (rt / 100000) % 100, rt % 100000);

    device_cycle();
    host_location();
    ipc_export();

    std::printf("\n%s%s\n", failures ? "FAIL: " : "PASS", failures ? "counted failures above" : "");
    return failures ? 1 : 0;
}

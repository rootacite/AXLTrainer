/*
 * hipMalloc's block against the VMM sequence amdfq_vmm.c builds, as the runtime describes them.
 *
 * Both are device memory the CPU can read; kernels that run fine on the first raise an illegal
 * shader instruction on the second. So the difference is somewhere the queries below can see it — or
 * somewhere they cannot, which is also an answer. Every call is made on one pointer from each route,
 * at the same size, in the same process.
 *
 * The VMM sequence is exactly the hook's: reserve, hipMemCreate with a pinned/device property,
 * hipMemMap at offset 0, then hipMemSetAccess for the device *and* the host location, plus a second
 * shared handle mapped behind the block as the pad.
 *
 * Plain C with the HIP entry points declared here, the way amdfq_gates.h declares its three, so it
 * links against the env's libamdhip64.so.7 and needs no ROCm toolchain.
 */
#define _GNU_SOURCE

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* --- the HIP surface this probe needs ---------------------------------------------------------- */
typedef int hipError_t;
typedef void *hipDeviceptr_t;
typedef struct ihipMemGenericAllocationHandle *hipMemGenericAllocationHandle_t;

typedef struct {
    int type;
    int device;
    void *devicePointer;
    void *hostPointer;
    int isManaged;
    unsigned int allocationFlags;
} hipPointerAttribute_t;

typedef struct {
    int type;
    int id;
} hipMemLocation;

typedef struct {
    int type;
    int requestedHandleTypes;
    hipMemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
    } allocFlags;
} hipMemAllocationProp;

typedef struct {
    hipMemLocation location;
    int flags;
} hipMemAccessDesc;

enum {
    MEM_ALLOCATION_TYPE_PINNED = 0x1,
    MEM_HANDLE_TYPE_NONE = 0x0,
    MEM_LOCATION_DEVICE = 1,
    MEM_LOCATION_HOST = 2,
    MEM_ACCESS_PROT_READWRITE = 3,
    GRANULARITY_MINIMUM = 0x0,
    GRANULARITY_RECOMMENDED = 0x1,
};

hipError_t hipInit(unsigned int flags);
hipError_t hipGetDevice(int *device);
hipError_t hipMemGetInfo(size_t *free, size_t *total);
hipError_t hipMalloc(void **ptr, size_t size);
hipError_t hipFree(void *ptr);
hipError_t hipMemset(void *dst, int value, size_t size);
hipError_t hipDeviceSynchronize(void);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *ptr);
hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize, hipDeviceptr_t dptr);
hipError_t hipMemGetAccess(unsigned long long *flags, const hipMemLocation *location, void *ptr);
hipError_t hipMemAddressReserve(void **ptr, size_t size, size_t alignment, void *addr,
                                unsigned long long flags);
hipError_t hipMemAddressFree(void *ptr, size_t size);
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t *handle, size_t size,
                        const hipMemAllocationProp *prop, unsigned long long flags);
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
hipError_t hipMemMap(void *ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags);
hipError_t hipMemUnmap(void *ptr, size_t size);
hipError_t hipMemSetAccess(void *ptr, size_t size, const hipMemAccessDesc *desc, size_t count);
hipError_t hipMemGetAllocationGranularity(size_t *granularity, const hipMemAllocationProp *prop,
                                          int option);
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp *prop,
                                                   hipMemGenericAllocationHandle_t handle);
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t *handle, void *addr);

/* --- reporting --------------------------------------------------------------------------------- */
#define MiB (1024ULL * 1024ULL)
#define GiB (1024ULL * 1024ULL * 1024ULL)

static int g_device;
static hipMemAllocationProp g_prop;

static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
}

static const char *memory_type_name(int type) {
    switch (type) {
    case 1: return "Host";
    case 2: return "Device";
    case 3: return "Managed";
    case 10: return "Array";
    case 11: return "Unified";
    default: return "?";
    }
}

static void print_maps(const void *ptr) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (f == NULL) return;
    char line[512];
    unsigned long long addr = (unsigned long long)(uintptr_t)ptr;
    int found = 0;
    while (fgets(line, sizeof line, f) != NULL) {
        unsigned long long start = 0, end = 0;
        if (sscanf(line, "%llx-%llx", &start, &end) != 2) continue;
        if (addr >= start && addr < end) {
            char *newline = strchr(line, '\n');
            if (newline != NULL) *newline = '\0';
            if (found == 0) say("  /proc/self/maps:        %s", line);
            found++;
        }
    }
    if (found == 0) say("  /proc/self/maps:        no VMA covers this address");
    fclose(f);
}

/* Everything the runtime will say about one pointer. */
static void describe(const char *label, void *ptr, const void *pad_ptr) {
    say("  -- %s at %p", label, ptr);

    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof attr);
    hipError_t ret = hipPointerGetAttributes(&attr, ptr);
    say("  hipPointerGetAttributes:            ret=%d type=%d(%s) device=%d devicePointer=%p"
        " hostPointer=%p isManaged=%d allocationFlags=0x%x",
        (int)ret, attr.type, memory_type_name(attr.type), attr.device, attr.devicePointer,
        attr.hostPointer, attr.isManaged, attr.allocationFlags);

    hipDeviceptr_t base = NULL;
    size_t size = 0;
    ret = hipMemGetAddressRange(&base, &size, (hipDeviceptr_t)ptr);
    say("  hipMemGetAddressRange:              ret=%d base=%p size=%llu (%llu MiB)", (int)ret, base,
        (unsigned long long)size, (unsigned long long)size / MiB);

    hipMemLocation location;
    unsigned long long flags = 0;
    location.type = MEM_LOCATION_DEVICE;
    location.id = g_device;
    ret = hipMemGetAccess(&flags, &location, ptr);
    say("  hipMemGetAccess(device %d):         ret=%d flags=0x%llx", g_device, (int)ret, flags);
    location.type = MEM_LOCATION_HOST;
    location.id = 0;
    ret = hipMemGetAccess(&flags, &location, ptr);
    say("  hipMemGetAccess(host):              ret=%d flags=0x%llx", (int)ret, flags);

    hipMemGenericAllocationHandle_t handle = NULL;
    ret = hipMemRetainAllocationHandle(&handle, ptr);
    say("  hipMemRetainAllocationHandle:       ret=%d handle=%p", (int)ret, (void *)handle);
    if (ret == 0) {
        hipMemAllocationProp prop;
        memset(&prop, 0, sizeof prop);
        if (hipMemGetAllocationPropertiesFromHandle(&prop, handle) == 0)
            say("  hipMemGetAllocationProperties:      type=%d handleTypes=%d location=%d/%d"
                " compression=%u rdma=%u usage=0x%x",
                prop.type, prop.requestedHandleTypes, prop.location.type, prop.location.id,
                prop.allocFlags.compressionType, prop.allocFlags.gpuDirectRDMACapable,
                prop.allocFlags.usage);
        hipMemRelease(handle);
    }

    print_maps(ptr);
    say("  host read of byte 0:                0x%02x", *(volatile unsigned char *)ptr);
    if (pad_ptr != NULL) {
        say("  -- the pad granule behind it at %p", pad_ptr);
        memset(&attr, 0, sizeof attr);
        ret = hipPointerGetAttributes(&attr, pad_ptr);
        say("  hipPointerGetAttributes:            ret=%d type=%d(%s) devicePointer=%p"
            " hostPointer=%p", (int)ret, attr.type, memory_type_name(attr.type), attr.devicePointer,
            attr.hostPointer);
        base = NULL;
        size = 0;
        ret = hipMemGetAddressRange(&base, &size, (hipDeviceptr_t)pad_ptr);
        say("  hipMemGetAddressRange:              ret=%d base=%p size=%llu", (int)ret, base,
            (unsigned long long)size);
    }
    say("");
}

static void set_access(void *ptr, size_t size) {
    hipMemAccessDesc desc[2];
    memset(desc, 0, sizeof desc);
    desc[0].location = g_prop.location;
    desc[0].flags = MEM_ACCESS_PROT_READWRITE;
    desc[1].location.type = MEM_LOCATION_HOST;
    desc[1].location.id = 0;
    desc[1].flags = MEM_ACCESS_PROT_READWRITE;
    hipError_t ret = hipMemSetAccess(ptr, size, desc, 2);
    if (ret != 0) say("  hipMemSetAccess(device+host) -> ret=%d", (int)ret);
}

int main(void) {
    hipInit(0);
    hipGetDevice(&g_device);
    memset(&g_prop, 0, sizeof g_prop);
    g_prop.type = MEM_ALLOCATION_TYPE_PINNED;
    g_prop.requestedHandleTypes = MEM_HANDLE_TYPE_NONE;
    g_prop.location.type = MEM_LOCATION_DEVICE;
    g_prop.location.id = g_device;

    size_t minimum = 0, recommended = 0;
    hipMemGetAllocationGranularity(&minimum, &g_prop, GRANULARITY_MINIMUM);
    hipMemGetAllocationGranularity(&recommended, &g_prop, GRANULARITY_RECOMMENDED);
    say("device %d, granularity minimum %zu B, recommended %zu B, page size %ld",
        g_device, minimum, recommended, sysconf(_SC_PAGESIZE));

    size_t sizes[] = {2 * MiB, 44 * MiB};
    void *base = NULL;
    size_t reserved = 0;
    if (hipMemAddressReserve(&base, 16 * GiB, 0, NULL, 0) != 0) {
        say("hipMemAddressReserve failed, nothing to compare");
        return 1;
    }
    hipMemGenericAllocationHandle_t pad = NULL;
    hipMemCreate(&pad, recommended, &g_prop, 0);

    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        size_t size = sizes[i];
        say("========================================================");
        say(" size %llu MiB", (unsigned long long)size / MiB);

        size_t before = 0, total = 0;
        hipMemGetInfo(&before, &total);
        void *p = NULL;
        hipError_t ret = hipMalloc(&p, size);
        hipMemset(p, 0x11, 4096);
        hipDeviceSynchronize();
        size_t after = 0;
        hipMemGetInfo(&after, &total);
        say("== hipMalloc: ret=%d, charged %llu MiB",
            (int)ret, (unsigned long long)(before - after) / MiB);
        describe("hipMalloc block", p, NULL);
        hipFree(p);

        void *va = base;
        hipMemGetInfo(&before, &total);
        hipMemGenericAllocationHandle_t handle = NULL;
        hipError_t r_create = hipMemCreate(&handle, size, &g_prop, 0);
        hipError_t r_map = hipMemMap(va, size, 0, handle, 0);
        set_access(va, size);
        hipError_t r_pad = hipMemMap((char *)va + size, recommended, 0, pad, 0);
        set_access((char *)va + size, recommended);
        hipMemset(va, 0x22, 4096);
        hipDeviceSynchronize();
        hipMemGetInfo(&after, &total);
        say("== VMM (the hook's sequence): create=%d map=%d pad_map=%d, charged %llu MiB",
            (int)r_create, (int)r_map, (int)r_pad, (unsigned long long)(before - after) / MiB);
        describe("VMM block", va, (char *)va + size);

        hipMemUnmap((char *)va + size, recommended);
        hipMemUnmap(va, size);
        hipMemRelease(handle);
    }

    hipMemRelease(pad);
    hipMemAddressFree(base, reserved);
    return 0;
}

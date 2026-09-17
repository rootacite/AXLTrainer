/*
 * Go/no-go probe for route 1 of amdfq.md §10.3: instead of raising every hipMalloc by 16 bytes (one
 * granule of VRAM per live allocation), take over the layout with the VMM API and map *one* physical
 * granule behind every block.
 *
 * The whole route rests on one assumption the HIP headers do not state either way: that a single
 * hipMemGenericAllocationHandle_t can be mapped at many virtual addresses at once. `multimap` answers
 * it, and everything else here only measures what that answer would cost.
 *
 * Plain C with the HIP entry points declared the way amdfq_gates.h declares its three, so the probe
 * links straight against the env's libamdhip64.so.7 and needs no ROCm toolchain. Modes marked
 * "isolated" can fault the GPU; run them in their own process (vmm_probe.sh does).
 *
 *   ./vmm_probe info       granularity, what can be reserved, VRAM counters
 *   ./vmm_probe multimap   one handle at 8 VAs, aliasing, a read past a block end, release path
 *   ./vmm_probe cost       1000 x 2 MiB: hipMalloc+pad16 vs VMM+one shared pad, VRAM and time
 *   ./vmm_probe access     isolated: is hipMemSetAccess required?
 *   ./vmm_probe overread   isolated: read past the end with nothing mapped there
 */
#define _GNU_SOURCE

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* --- the HIP surface this probe needs ---------------------------------------------------------- */
typedef int hipError_t;
typedef struct ihipMemGenericAllocationHandle *hipMemGenericAllocationHandle_t;

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
    MEMCPY_HOST_TO_DEVICE = 1,
    MEMCPY_DEVICE_TO_HOST = 2,
};

hipError_t hipInit(unsigned int flags);
hipError_t hipGetDevice(int *device);
hipError_t hipGetDeviceCount(int *count);
hipError_t hipMemGetInfo(size_t *free, size_t *total);
hipError_t hipMalloc(void **ptr, size_t size);
hipError_t hipFree(void *ptr);
hipError_t hipMemset(void *dst, int value, size_t size);
hipError_t hipMemcpy(void *dst, const void *src, size_t size, int kind);
hipError_t hipDeviceSynchronize(void);
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

/* --- reporting --------------------------------------------------------------------------------- */
#define MiB (1024ULL * 1024ULL)
#define GiB (1024ULL * 1024ULL * 1024ULL)

static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/* Device-wide VRAM in use, straight from the kernel's counter: hipMemGetInfo reports the runtime's
 * own accounting, which may or may not include VMM-created memory. Both are printed wherever a
 * number matters. */
static unsigned long long dev_used(void) {
    for (int card = 0; card < 4; card++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_used", card);
        FILE *f = fopen(path, "r");
        if (f == NULL) continue;
        unsigned long long bytes = 0;
        int ok = fscanf(f, "%llu", &bytes);
        fclose(f);
        if (ok == 1) return bytes;
    }
    return 0;
}

static size_t g_granule;
static hipMemAllocationProp g_prop;

static void vram(const char *tag) {
    size_t free_bytes = 0, total = 0;
    hipError_t ret = hipMemGetInfo(&free_bytes, &total);
    say("    %-34s hipMemGetInfo free=%llu MiB (ret=%d), device counter used=%llu MiB", tag,
        (unsigned long long)free_bytes / MiB, (int)ret, dev_used() / MiB);
}

static void prop_init(int device) {
    memset(&g_prop, 0, sizeof g_prop);
    g_prop.type = MEM_ALLOCATION_TYPE_PINNED;
    g_prop.requestedHandleTypes = MEM_HANDLE_TYPE_NONE;
    g_prop.location.type = MEM_LOCATION_DEVICE;
    g_prop.location.id = device;
}

static void set_access(void *ptr, size_t size) {
    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof desc);
    desc.location = g_prop.location;
    desc.flags = MEM_ACCESS_PROT_READWRITE;
    hipError_t ret = hipMemSetAccess(ptr, size, &desc, 1);
    if (ret != 0) say("    hipMemSetAccess(%p, %llu MiB) -> ret=%d", ptr,
                      (unsigned long long)size / MiB, (int)ret);
}

/* Reserve the largest of the listed sizes that the driver accepts; returns 0 on success. */
static int reserve_regions(size_t *sizes, size_t count, void **base, size_t *got) {
    for (size_t i = 0; i < count; i++) {
        void *p = NULL;
        hipError_t ret = hipMemAddressReserve(&p, sizes[i], 0, NULL, 0);
        say("    hipMemAddressReserve(%llu GiB) -> ret=%d ptr=%p", (unsigned long long)sizes[i] / GiB,
            (int)ret, p);
        if (ret == 0) {
            *base = p;
            *got = sizes[i];
            return 0;
        }
    }
    return -1;
}

/* --- info -------------------------------------------------------------------------------------- */
static int mode_info(void) {
    int count = 0, device = -1;
    hipGetDeviceCount(&count);
    hipGetDevice(&device);
    size_t free_bytes = 0, total = 0;
    hipMemGetInfo(&free_bytes, &total);
    say("== info");
    say("    devices=%d current=%d  hipMemGetInfo free=%llu MiB total=%llu MiB  device counter=%llu MiB",
        count, device, (unsigned long long)free_bytes / MiB, (unsigned long long)total / MiB,
        dev_used() / MiB);
    if (count <= 0) return 1;

    prop_init(device);
    size_t minimum = 0, recommended = 0;
    hipError_t r_min = hipMemGetAllocationGranularity(&minimum, &g_prop, GRANULARITY_MINIMUM);
    hipError_t r_rec = hipMemGetAllocationGranularity(&recommended, &g_prop, GRANULARITY_RECOMMENDED);
    g_granule = minimum;
    say("    granularity: minimum=%llu B (ret=%d) recommended=%llu B (ret=%d)  [§9's 2 MiB is the"
        " runtime's own pool granularity; the minimum is what bounds our mappings]",
        (unsigned long long)minimum, (int)r_min, (unsigned long long)recommended, (int)r_rec);

    size_t sizes[] = {1 * GiB * 1024, 512 * GiB, 256 * GiB, 128 * GiB, 64 * GiB, 16 * GiB, 4 * GiB};
    void *base = NULL;
    size_t got = 0;
    if (reserve_regions(sizes, sizeof sizes / sizeof sizes[0], &base, &got) != 0) {
        say("    no reservation succeeded");
        return 1;
    }
    say("    reserved %llu GiB at %p (offset mod granularity = %llu, mod 1 GiB = %llu)",
        (unsigned long long)got / GiB, base, (unsigned long long)((uintptr_t)base % g_granule),
        (unsigned long long)((uintptr_t)base % GiB));
    vram("with the reservation held");
    hipMemAddressFree(base, got);
    vram("reservation freed");
    return 0;
}

/* --- multimap ---------------------------------------------------------------------------------- */
static int mode_multimap(void) {
    say("== multimap  (the go/no-go: one handle, many virtual addresses)");
    size_t sizes[] = {64 * GiB, 16 * GiB};
    void *base = NULL;
    size_t reserved = 0;
    if (reserve_regions(sizes, 2, &base, &reserved) != 0) return 1;
    vram("after reserve");

    size_t g = g_granule;
    hipMemGenericAllocationHandle_t data = NULL, pad = NULL;
    hipError_t ret = hipMemCreate(&data, 4 * MiB, &g_prop, 0);
    say("    hipMemCreate(4 MiB) -> ret=%d handle=%p", (int)ret, (void *)data);
    if (ret != 0) return 1;
    ret = hipMemCreate(&pad, g, &g_prop, 0);
    say("    hipMemCreate(%llu B) -> ret=%d handle=%p (the one shared pad; charged 2 MiB physically)",
        (unsigned long long)g, (int)ret, (void *)pad);
    if (ret != 0) return 1;

    struct {
        const char *what;
        void *where;
        size_t size;
        hipMemGenericAllocationHandle_t handle;
    } maps[] = {
        {"data", (char *)base + 0, 4 * MiB, data},
        {"data (second VA)", (char *)base + 16 * MiB, 4 * MiB, data},
        {"pad", (char *)base + 4 * MiB, g, pad},
        {"pad", (char *)base + 8 * MiB, g, pad},
        {"pad", (char *)base + 12 * MiB, g, pad},
        {"pad", (char *)base + 20 * MiB, g, pad},
        {"pad", (char *)base + 24 * MiB, g, pad},
        {"pad", (char *)base + 28 * MiB, g, pad},
        {"pad (4 KiB aligned, not 2 MiB aligned)", (char *)base + 30 * MiB + g, g, pad},
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof maps / sizeof maps[0]; i++) {
        void *p = maps[i].where;
        ret = hipMemMap(p, maps[i].size, 0, maps[i].handle, 0);
        say("    hipMemMap(%p, %llu B, offset 0, %s) -> ret=%d", p,
            (unsigned long long)maps[i].size, maps[i].what, (int)ret);
        if (ret != 0) failed++;
        /* Access is granted per mapped region: one hipMemSetAccess over a range that contains an
         * unmapped gap returns hipErrorInvalidValue, and the next access to the region faults. */
        if (ret == 0) set_access(maps[i].where, maps[i].size);
    }
    vram("after the mappings");

    if (failed == 0) {
        unsigned char host[4096];
        hipMemset((char *)base + 16 * MiB, 0xAB, sizeof host);
        hipDeviceSynchronize();
        memset(host, 0, sizeof host);
        ret = hipMemcpy(host, base, sizeof host, MEMCPY_DEVICE_TO_HOST);
        size_t hits = 0;
        for (size_t i = 0; i < sizeof host; i++) hits += host[i] == 0xAB;
        say("    alias check: wrote 0xAB through the second mapping of the data handle, read %zu/%zu"
            " bytes of 0xAB back through the first (ret=%d)",
            hits, sizeof host, (int)ret);
    }

    /* The point of the pad: a read that starts just before the end of the block and runs into the
     * next granule. This is the copy-engine analogue of the kernel over-read of §6. */
    unsigned char tail[32];
    memset(tail, 0, sizeof tail);
    ret = hipMemcpy(tail, (char *)base + 4 * MiB - 8, sizeof tail, MEMCPY_DEVICE_TO_HOST);
    hipError_t sync = hipDeviceSynchronize();
    say("    read past the block end: hipMemcpy(32 bytes from base+4MiB-8) -> ret=%d sync=%d", (int)ret,
        (int)sync);
    if (ret == 0 && sync == 0) {
        say("      bytes: %02x %02x %02x %02x %02x %02x %02x %02x | %02x %02x %02x %02x %02x %02x %02x %02x",
            tail[0], tail[1], tail[2], tail[3], tail[4], tail[5], tail[6], tail[7], tail[8], tail[9],
            tail[10], tail[11], tail[12], tail[13], tail[14], tail[15]);
    }

    for (size_t i = 0; i < sizeof maps / sizeof maps[0]; i++)
        hipMemUnmap(maps[i].where, maps[i].size);
    hipMemRelease(data);
    hipMemRelease(pad);
    hipMemAddressFree(base, reserved);
    vram("after unmap + release + address free");
    return 0;
}

/* --- cost -------------------------------------------------------------------------------------- */
static int mode_cost(void) {
    say("== cost  (1000 x 2 MiB, and the granule table of amdfq.md §9)");
    vram("start");

    const int n = 1000;
    const size_t size = 2 * MiB;

    /* (a) what the hook does today: every request raised by 16 bytes. */
    void **ptrs = calloc((size_t)n, sizeof *ptrs);
    double t0 = now_s();
    for (int i = 0; i < n; i++) hipMalloc(&ptrs[i], size + 16);
    double t_malloc = now_s() - t0;
    hipDeviceSynchronize();
    size_t free_a = 0, total = 0;
    hipMemGetInfo(&free_a, &total);
    say("    (a) %d hipMalloc(2 MiB + 16): %.1f us/call, hipMemGetInfo free=%llu MiB, device used=%llu MiB",
        n, t_malloc / n * 1e6, (unsigned long long)free_a / MiB, dev_used() / MiB);
    for (int i = 0; i < n; i++) hipFree(ptrs[i]);
    size_t free_a2 = 0;
    hipMemGetInfo(&free_a2, &total);
    say("        after free: hipMemGetInfo free=%llu MiB, device used=%llu MiB",
        (unsigned long long)free_a2 / MiB, dev_used() / MiB);
    free(ptrs);

    /* (b) the VMM route: one handle per block, one shared pad granule mapped behind each. */
    size_t sizes[] = {64 * GiB, 16 * GiB};
    void *base = NULL;
    size_t reserved = 0;
    if (reserve_regions(sizes, 2, &base, &reserved) != 0) return 1;
    hipMemGenericAllocationHandle_t pad = NULL;
    hipMemCreate(&pad, g_granule, &g_prop, 0);
    size_t free_b0 = 0;
    hipMemGetInfo(&free_b0, &total);
    hipMemGenericAllocationHandle_t *handles = calloc((size_t)n, sizeof *handles);
    void **vas = calloc((size_t)n, sizeof *vas);
    t0 = now_s();
    for (int i = 0; i < n; i++) {
        void *va = (char *)base + (size_t)i * (size + g_granule);
        vas[i] = va;
        hipMemCreate(&handles[i], size, &g_prop, 0);
        hipMemMap(va, size, 0, handles[i], 0);
        hipMemMap((char *)va + size, g_granule, 0, pad, 0);
        set_access(va, size + g_granule);
    }
    double t_vmm = now_s() - t0;
    hipDeviceSynchronize();
    size_t free_b = 0;
    hipMemGetInfo(&free_b, &total);
    say("    (b) %d x [hipMemCreate(2 MiB) + 2 hipMemMap + hipMemSetAccess]: %.1f us/call",
        n, t_vmm / n * 1e6);
    say("        hipMemGetInfo free=%llu MiB (delta %llu MiB), device used=%llu MiB",
        (unsigned long long)free_b / MiB, (unsigned long long)(free_b0 - free_b) / MiB,
        dev_used() / MiB);
    say("        blocks mapped=%d (%llu MiB) + pad granules=%d (%llu MiB): the pad is one granule total",
        n, (unsigned long long)n * size / MiB, 1, (unsigned long long)g_granule / MiB);

    t0 = now_s();
    for (int i = 0; i < n; i++) {
        hipMemUnmap(vas[i], size);
        hipMemUnmap((char *)vas[i] + size, g_granule);
        hipMemRelease(handles[i]);
    }
    double t_free = now_s() - t0;
    hipMemRelease(pad);
    size_t free_b2 = 0;
    hipMemGetInfo(&free_b2, &total);
    say("        unmap+release: %.1f us/call, after: hipMemGetInfo free=%llu MiB, device used=%llu MiB",
        t_free / n * 1e6, (unsigned long long)free_b2 / MiB, dev_used() / MiB);
    free(handles);
    free(vas);

    /* The size classes of §9, charged by the driver under each route. */
    say("    granule table (one allocation at a time, VRAM charged by hipMemGetInfo):");
    say("      %10s %14s %14s %14s", "request", "hipMalloc", "hipMalloc+16", "VMM map");
    size_t classes[] = {1 * MiB, 2 * MiB, 4 * MiB, 8 * MiB, 32 * MiB, 44 * MiB};
    for (size_t c = 0; c < sizeof classes / sizeof classes[0]; c++) {
        size_t s = classes[c];
        long long costs[3] = {0, 0, 0};
        for (int variant = 0; variant < 3; variant++) {
            size_t before = 0, after = 0;
            hipMemGetInfo(&before, &total);
            if (variant < 2) {
                void *p = NULL;
                hipMalloc(&p, variant == 0 ? s : s + 16);
                hipMemGetInfo(&after, &total);
                hipFree(p);
            } else {
                size_t map_size = (s + g_granule - 1) / g_granule * g_granule;
                hipMemGenericAllocationHandle_t h = NULL;
                hipMemCreate(&h, map_size, &g_prop, 0);
                hipMemMap(base, map_size, 0, h, 0);
                set_access(base, map_size);
                hipMemGetInfo(&after, &total);
                hipMemUnmap(base, map_size);
                hipMemRelease(h);
            }
            costs[variant] = (long long)before - (long long)after;
        }
        say("      %7llu MiB %11lld MiB %11lld MiB %11lld MiB", (unsigned long long)s / MiB,
            costs[0] / (long long)MiB, costs[1] / (long long)MiB, costs[2] / (long long)MiB);
    }

    hipMemAddressFree(base, reserved);
    vram("end");
    return 0;
}

/* --- isolated modes ---------------------------------------------------------------------------- */
static int mode_access(void) {
    say("== access (isolated: skipped hipMemSetAccess)");
    size_t sizes[] = {16 * GiB, 4 * GiB};
    void *base = NULL;
    size_t reserved = 0;
    if (reserve_regions(sizes, 2, &base, &reserved) != 0) return 1;
    hipMemGenericAllocationHandle_t h = NULL;
    hipMemCreate(&h, 2 * MiB, &g_prop, 0);
    hipError_t ret = hipMemMap(base, 2 * MiB, 0, h, 0);
    say("    hipMemMap without hipMemSetAccess -> ret=%d", (int)ret);
    ret = hipMemset(base, 0x5A, 4096);
    hipError_t sync = hipDeviceSynchronize();
    say("    hipMemset(4096 bytes) -> ret=%d sync=%d  (a ret or sync failure means the access flag is required)",
        (int)ret, (int)sync);
    unsigned char host[16];
    memset(host, 0, sizeof host);
    ret = hipMemcpy(host, base, sizeof host, MEMCPY_DEVICE_TO_HOST);
    sync = hipDeviceSynchronize();
    say("    hipMemcpy(device->host) -> ret=%d sync=%d, first byte=0x%02x", (int)ret, (int)sync, host[0]);

    /* control: the same thing with the access flag set */
    void *control = (char *)base + 4 * MiB;
    hipMemGenericAllocationHandle_t h2 = NULL;
    hipMemCreate(&h2, 2 * MiB, &g_prop, 0);
    hipMemMap(control, 2 * MiB, 0, h2, 0);
    set_access(control, 2 * MiB);
    ret = hipMemset(control, 0x5A, 4096);
    sync = hipDeviceSynchronize();
    say("    control with hipMemSetAccess: hipMemset -> ret=%d sync=%d", (int)ret, (int)sync);
    return 0;
}

static int mode_overread(void) {
    say("== overread (isolated: nothing mapped past the block end, expect a fault)");
    size_t sizes[] = {16 * GiB, 4 * GiB};
    void *base = NULL;
    size_t reserved = 0;
    if (reserve_regions(sizes, 2, &base, &reserved) != 0) return 1;
    hipMemGenericAllocationHandle_t h = NULL;
    hipMemCreate(&h, 4 * MiB, &g_prop, 0);
    hipMemMap(base, 4 * MiB, 0, h, 0);
    set_access(base, 4 * MiB);
    hipMemset(base, 0x11, 4096);
    hipDeviceSynchronize();
    unsigned char tail[32];
    memset(tail, 0, sizeof tail);
    hipError_t ret = hipMemcpy(tail, (char *)base + 4 * MiB - 8, sizeof tail, MEMCPY_DEVICE_TO_HOST);
    hipError_t sync = hipDeviceSynchronize();
    say("    hipMemcpy(32 bytes from base+4MiB-8) -> ret=%d sync=%d  (the pad-0 case of §10.2, at probe"
        " scale)", (int)ret, (int)sync);
    return 0;
}

/* Host access to the mapping. `hipMalloc` returns memory the CPU can read at the same address (§9
 * measures an allocation by reading its mapping out of /proc/self/maps), and torch's `.item()` reads
 * a device pointer straight from host code on this stack — a trainer run under a device-only VMM
 * layout died in at::native::_local_scalar_dense_cuda that way. `host` asks hipMemSetAccess to grant
 * the CPU agent as well, which is the candidate fix. */
static int mode_hostaccess(const char *variant) {
    say("== hostaccess %s", variant);
    if (strcmp(variant, "hipmalloc") == 0) {
        void *p = NULL;
        hipError_t ret = hipMalloc(&p, 2 * MiB);
        unsigned char byte = *(volatile unsigned char *)p;
        say("    hipMalloc(2 MiB) -> %p (ret=%d); host read of byte 0 = 0x%02x", p, (int)ret, byte);
        return 0;
    }

    size_t sizes[] = {16 * GiB, 4 * GiB};
    void *base = NULL;
    size_t reserved = 0;
    if (reserve_regions(sizes, 2, &base, &reserved) != 0) return 1;
    hipMemGenericAllocationHandle_t handle = NULL;
    hipError_t ret = hipMemCreate(&handle, 2 * MiB, &g_prop, 0);
    ret = hipMemMap(base, 2 * MiB, 0, handle, 0);
    say("    hipMemCreate/hipMemMap 2 MiB at %p -> ret=%d", base, (int)ret);

    if (strcmp(variant, "host") == 0) {
        hipMemAccessDesc desc[2];
        memset(desc, 0, sizeof desc);
        desc[0].location = g_prop.location;
        desc[0].flags = MEM_ACCESS_PROT_READWRITE;
        desc[1].location.type = MEM_LOCATION_HOST;
        desc[1].location.id = 0;
        desc[1].flags = MEM_ACCESS_PROT_READWRITE;
        ret = hipMemSetAccess(base, 2 * MiB, desc, 2);
        say("    hipMemSetAccess(device + host) -> ret=%d", (int)ret);
    } else {
        set_access(base, 2 * MiB);
    }

    unsigned char byte = *(volatile unsigned char *)base;
    say("    host read of byte 0 = 0x%02x", byte);

    /* If the host can read it, the value the GPU wrote has to come back too. */
    hipMemset(base, 0x7C, 4096);
    hipDeviceSynchronize();
    unsigned char copy[16];
    memset(copy, 0, sizeof copy);
    ret = hipMemcpy(copy, base, sizeof copy, MEMCPY_DEVICE_TO_HOST);
    say("    hipMemcopy(device->host) ret=%d, first byte 0x%02x, host byte 0x%02x", (int)ret, copy[0],
        *(volatile unsigned char *)base);
    return 0;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "all";

    /* These modes end in a fault, and a GPU fault resets the ring and takes the desktop down with it
     * (2026-09-17). They need an explicit opt-in. */
    if (strcmp(mode, "access") == 0 || strcmp(mode, "overread") == 0 ||
        (strcmp(mode, "hostaccess") == 0 && argc > 2 && strcmp(argv[2], "device") == 0)) {
        const char *allow = getenv("AMDFQ_PROBE_ALLOW_FAULTS");
        if (allow == NULL || allow[0] != '1') {
            say("%s would fault on purpose (device mode: this process; access/overread: the GPU, which"
                " resets the ring here). Set AMDFQ_PROBE_ALLOW_FAULTS=1 to run it.", mode);
            return 3;
        }
    }

    hipInit(0);
    int device = 0;
    hipGetDevice(&device);
    prop_init(device);

    size_t minimum = 0;
    if (hipMemGetAllocationGranularity(&minimum, &g_prop, GRANULARITY_MINIMUM) == 0 && minimum > 0)
        g_granule = minimum;

    if (strcmp(mode, "info") == 0) return mode_info();
    if (strcmp(mode, "multimap") == 0) return mode_multimap();
    if (strcmp(mode, "cost") == 0) return mode_cost();
    if (strcmp(mode, "access") == 0) return mode_access();
    if (strcmp(mode, "overread") == 0) return mode_overread();
    if (strcmp(mode, "hostaccess") == 0) return mode_hostaccess(argc > 2 ? argv[2] : "device");
    if (strcmp(mode, "all") == 0) {
        int failed = mode_info();
        failed |= mode_multimap();
        return failed;
    }
    fprintf(stderr, "usage: %s [info|multimap|cost|access|overread|all]\n", argv[0]);
    return 2;
}

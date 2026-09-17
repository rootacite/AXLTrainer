/*
 * Does a kernel blow up on memory allocated the way the hook allocates it?
 *
 * The hook's VMM route (amdfq-vmm/amdfq_vmm.c) is reproduced here call for call — pinned/device
 * property, the *recommended* granularity, a reserved range, hipMemCreate + hipMemMap at offset 0,
 * hipMemSetAccess for the device *and* the host location, and one shared pad granule mapped behind
 * every block. Then a trivial kernel runs on it, next to the same kernel on a hipMalloc block as the
 * control, and the two are compared for data as well: the host writes a pattern for the GPU to check
 * and the GPU writes one for the host to check, because a mapping that grants the CPU access to VRAM
 * is the one property here that hipMalloc's own blocks also have but whose *coherence* nobody has
 * tested at this size.
 *
 * If only the VMM block misbehaves, the next step is the real kernel from the training run; if both
 * behave the same, the training failure needs the training workload to show itself.
 *
 *   bash amdfq-vmm/vmm_kernel_test.sh [blocks] [MiB per block]
 */
#include <hip/hip_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#define MiB (1024ULL * 1024ULL)
#define GiB (1024ULL * 1024ULL * 1024ULL)

#define HIPCHECK(call)                                                                       \
    do {                                                                                     \
        hipError_t e_ = (call);                                                              \
        if (e_ != hipSuccess) {                                                              \
            fprintf(stderr, "%s:%d  %s -> %s\n", __FILE__, __LINE__, #call,                   \
                    hipGetErrorString(e_));                                                  \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

__global__ void fill_kernel(unsigned char *p, unsigned long long n, unsigned char value) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) p[i] = value;
}

__global__ void check_kernel(const unsigned char *p, unsigned long long n, unsigned char value,
                             unsigned long long *bad) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n && p[i] != value) atomicAdd(bad, 1ULL);
}

/* 8-byte loads and stores: the access width the two kernels that died in the training run used
 * (vectorized_elementwise_kernel<4> on bf16 is 8 bytes, Tensile's GRVWA8/GRVWB8 are 8 bytes). */
__global__ void vec8_kernel(const unsigned long long *src, unsigned long long *dst,
                            unsigned long long n8) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n8) dst[i] = src[i] ^ 0x5a5a5a5a5a5a5a5aULL;
}

/* One pattern generator on both sides of the bus, so a mismatch means the two sides disagree about
 * the bytes and not that the pattern was computed differently. */
static unsigned char pattern(unsigned long long i, unsigned seed) {
    return (unsigned char)((i * 7 + seed * 131) & 0xFF);
}

__global__ void write_pattern(unsigned char *p, unsigned long long n, unsigned seed) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) p[i] = (unsigned char)((i * 7 + seed * 131) & 0xFF);
}

__global__ void atomic_add_many(unsigned long long *counter, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counter, 1ULL);
}

/* One workgroup publishes a token, the rest wait for it: the cross-workgroup handshake every
 * work-stealing and split-K kernel is built on, and the thing a hang would come from. The wait is
 * bounded so a broken handshake reports instead of hanging the card. */
__global__ void spin_for_flag(volatile unsigned *flag, unsigned long long *timeouts, unsigned limit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *flag = 0xC0FFEEu;
        __threadfence();
        return;
    }
    unsigned spins = 0;
    while (*flag != 0xC0FFEEu && spins < limit) spins++;
    if (*flag != 0xC0FFEEu) atomicAdd(timeouts, 1ULL);
}

__global__ void check_pattern(const unsigned char *p, unsigned long long n, unsigned seed,
                              unsigned long long *bad) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n && p[i] != (unsigned char)((i * 7 + seed * 131) & 0xFF)) atomicAdd(bad, 1ULL);
}

static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
}

/* --- the hook's allocation, call for call ------------------------------------------------------ */

struct Prop {
    hipMemAllocationProp prop;
    size_t granule;
};

struct Bump {
    void *base;     /* the one reservation, as the hook makes it */
    size_t reserved;
    size_t used;
};

struct Block {
    void *va;     /* what a caller would get */
    size_t block; /* bytes mapped to the block's own handle */
    void *pad;    /* the shared pad granule behind it, or nullptr */
    hipMemGenericAllocationHandle_t handle;
};

static Prop prop_init(int device) {
    Prop p;
    memset(&p, 0, sizeof p);
    p.prop.type = hipMemAllocationTypePinned;
    p.prop.requestedHandleTypes = hipMemHandleTypeNone;
    p.prop.location.type = hipMemLocationTypeDevice;
    p.prop.location.id = device;
    HIPCHECK(hipMemGetAllocationGranularity(&p.granule, &p.prop, hipMemAllocationGranularityRecommended));
    return p;
}

static void set_access(const Prop *p, void *ptr, size_t size, int with_host) {
    hipMemAccessDesc desc[2];
    memset(desc, 0, sizeof desc);
    desc[0].location = p->prop.location;
    desc[0].flags = hipMemAccessFlagsProtReadWrite;
    desc[1].location.type = hipMemLocationTypeHost;
    desc[1].location.id = 0;
    desc[1].flags = hipMemAccessFlagsProtReadWrite;
    HIPCHECK(hipMemSetAccess(ptr, size, desc, with_host ? 2 : 1));
}

static Block vmm_alloc(const Prop *p, Bump *bump, size_t size, int with_pad,
                       hipMemGenericAllocationHandle_t pad) {
    Block b;
    memset(&b, 0, sizeof b);
    b.block = (size + p->granule - 1) / p->granule * p->granule;
    b.va = (char *)bump->base + bump->used;
    bump->used += b.block + (with_pad ? p->granule : 0);
    HIPCHECK(hipMemCreate(&b.handle, b.block, &p->prop, 0));
    HIPCHECK(hipMemMap(b.va, b.block, 0, b.handle, 0));
    set_access(p, b.va, b.block, 1);
    if (with_pad) {
        b.pad = (char *)b.va + b.block;
        HIPCHECK(hipMemMap(b.pad, p->granule, 0, pad, 0));
        set_access(p, b.pad, p->granule, 1);
    }
    return b;
}

/* --- the checks -------------------------------------------------------------------------------- */

static void report(const char *which, void *p, size_t size, int with_pad, int heavy) {
    say("  %s: base %p, size %zu MiB", which, p, size / MiB);
    if (with_pad) say("    guard/pad granule mapped at %p", (char *)p + size);

    unsigned long long *bad = nullptr;
    HIPCHECK(hipMalloc((void **)&bad, sizeof *bad));
    unsigned int blocks = (unsigned int)((size + 255) / 256);

    /* device writes, device reads */
    HIPCHECK(hipMemset(bad, 0, sizeof *bad));
    fill_kernel<<<blocks, 256>>>((unsigned char *)p, size, 0xA7);
    HIPCHECK(hipDeviceSynchronize());
    check_kernel<<<blocks, 256>>>((const unsigned char *)p, size, 0xA7, bad);
    HIPCHECK(hipDeviceSynchronize());
    unsigned long long wrong = 0;
    HIPCHECK(hipMemcpy(&wrong, bad, sizeof wrong, hipMemcpyDeviceToHost));
    say("    GPU fill, GPU check:                 mismatching bytes %llu", wrong);

    if (!heavy) {
        /* scale runs touch every page once per direction instead of the whole block */
        for (size_t i = 0; i < size; i += 4096) ((unsigned char *)p)[i] = pattern(i, 0x21);
        HIPCHECK(hipMemset(bad, 0, sizeof *bad));
        fill_kernel<<<blocks, 256>>>((unsigned char *)p, size, 0x11);
        HIPCHECK(hipDeviceSynchronize());
        check_kernel<<<blocks, 256>>>((const unsigned char *)p, size, 0x11, bad);
        HIPCHECK(hipDeviceSynchronize());
        unsigned long long spot = 0;
        HIPCHECK(hipMemcpy(&spot, bad, sizeof spot, hipMemcpyDeviceToHost));
        say("    GPU fill, GPU check:                 mismatching bytes %llu", spot);
        HIPCHECK(hipFree(bad));
        return;
    }

    /* host writes the whole block through the mapping, device reads it back */
    for (size_t i = 0; i < size; i++) ((unsigned char *)p)[i] = pattern(i, 0x21);
    HIPCHECK(hipMemset(bad, 0, sizeof *bad));
    check_pattern<<<blocks, 256>>>((const unsigned char *)p, size, 0x21, bad);
    HIPCHECK(hipDeviceSynchronize());
    wrong = 0;
    HIPCHECK(hipMemcpy(&wrong, bad, sizeof wrong, hipMemcpyDeviceToHost));
    say("    host write %zu MiB, GPU check:       mismatching bytes %llu", size / MiB, wrong);

    /* device writes the whole block, host reads it back */
    write_pattern<<<blocks, 256>>>((unsigned char *)p, size, 0x33);
    HIPCHECK(hipDeviceSynchronize());
    wrong = 0;
    const volatile unsigned char *host = (const volatile unsigned char *)p;
    for (size_t i = 0; i < size; i++)
        if (host[i] != pattern(i, 0x33)) wrong++;
    say("    GPU write, host check:               mismatching bytes %llu", wrong);

    /* the two directions again, back to back, to catch a cache that only disagrees sometimes */
    for (int round = 0; round < 3; round++) {
        for (size_t i = 0; i < size; i++) ((unsigned char *)p)[i] = pattern(i, 0x40 + round);
        HIPCHECK(hipMemset(bad, 0, sizeof *bad));
        check_pattern<<<blocks, 256>>>((const unsigned char *)p, size, (unsigned)(0x40 + round), bad);
        HIPCHECK(hipDeviceSynchronize());
        unsigned long long to_gpu = 0;
        HIPCHECK(hipMemcpy(&to_gpu, bad, sizeof to_gpu, hipMemcpyDeviceToHost));
        write_pattern<<<blocks, 256>>>((unsigned char *)p, size, (unsigned)(0x50 + round));
        HIPCHECK(hipDeviceSynchronize());
        unsigned long long to_host = 0;
        for (size_t i = 0; i < size; i++)
            if (host[i] != pattern(i, 0x50 + round)) to_host++;
        say("    round %d: host->GPU %llu bad bytes, GPU->host %llu bad bytes", round, to_gpu, to_host);
    }

    /* 8-byte vector accesses, the failing kernels' width */
    vec8_kernel<<<(unsigned int)(size / 8 / 256), 256>>>((const unsigned long long *)p,
                                                         (unsigned long long *)p, size / 8);
    HIPCHECK(hipDeviceSynchronize());

    /* the runtime's own copy paths, which is how torch moves every tensor in and out */
    std::vector<unsigned char> staging((size_t)size);
    for (size_t i = 0; i < size; i++) staging[i] = pattern(i, 0x61);
    HIPCHECK(hipMemcpy(p, staging.data(), size, hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(bad, 0, sizeof *bad));
    check_pattern<<<blocks, 256>>>((const unsigned char *)p, size, 0x61, bad);
    HIPCHECK(hipDeviceSynchronize());
    unsigned long long to_gpu = 0;
    HIPCHECK(hipMemcpy(&to_gpu, bad, sizeof to_gpu, hipMemcpyDeviceToHost));
    say("    hipMemcpy H2D, GPU check:           mismatching bytes %llu", to_gpu);

    HIPCHECK(hipMemcpyAsync(p, staging.data(), size, hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(bad, 0, sizeof *bad));
    check_pattern<<<blocks, 256>>>((const unsigned char *)p, size, 0x61, bad);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(&to_gpu, bad, sizeof to_gpu, hipMemcpyDeviceToHost));
    say("    hipMemcpyAsync H2D, GPU check:      mismatching bytes %llu", to_gpu);

    memset(staging.data(), 0, staging.size());
    HIPCHECK(hipMemcpyAsync(staging.data(), p, size, hipMemcpyDeviceToHost));
    HIPCHECK(hipDeviceSynchronize());
    unsigned long long from_gpu = 0;
    for (size_t i = 0; i < size; i++)
        if (staging[i] != pattern(i, 0x61)) from_gpu++;
    say("    hipMemcpyAsync D2H, host check:     mismatching bytes %llu", from_gpu);

    /* device-side atomics under contention, counted where the kernels can be checked */
    unsigned long long *counter = (unsigned long long *)p;
    HIPCHECK(hipMemset(counter, 0, sizeof *counter));
    const unsigned long long atomics = 1u << 20;
    atomic_add_many<<<(unsigned int)(atomics / 256), 256>>>(counter, atomics);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(&from_gpu, counter, sizeof from_gpu, hipMemcpyDeviceToHost));
    say("    %llu device atomics:                 counter %llu (%s)", atomics, from_gpu,
        from_gpu == atomics ? "exact" : "WRONG");

    /* cross-workgroup publish/wait */
    volatile unsigned *flag = (volatile unsigned *)((char *)p + 4096);
    unsigned long long *timeouts = (unsigned long long *)((char *)p + 8192);
    HIPCHECK(hipMemset((void *)flag, 0, sizeof(unsigned)));
    HIPCHECK(hipMemset(timeouts, 0, sizeof *timeouts));
    spin_for_flag<<<256, 256>>>(flag, timeouts, 20000000u);
    HIPCHECK(hipDeviceSynchronize());
    unsigned long long waited = 0;
    HIPCHECK(hipMemcpy(&waited, timeouts, sizeof waited, hipMemcpyDeviceToHost));
    say("    cross-workgroup flag handshake:      timeouts %llu (%s)", waited,
        waited == 0 ? "every workgroup saw it" : "SOME NEVER SAW IT");

    HIPCHECK(hipFree(bad));
}

int main(int argc, char **argv) {
    int device = 0;
    HIPCHECK(hipGetDevice(&device));
    hipDeviceProp_t props{};
    HIPCHECK(hipGetDeviceProperties(&props, device));
    say("device %s (%s), %d CUs, page size %ld", props.name, props.gcnArchName,
        props.multiProcessorCount, sysconf(_SC_PAGESIZE));

    Prop p = prop_init(device);
    size_t minimum = 0;
    HIPCHECK(hipMemGetAllocationGranularity(&minimum, &p.prop, hipMemAllocationGranularityMinimum));
    say("granularity: minimum %zu B, recommended %zu B — the hook rounds to and pads with the"
        " recommended one", minimum, p.granule);

    const int n_blocks = argc > 1 ? atoi(argv[1]) : 1;
    const size_t size = (argc > 2 ? (size_t)atoi(argv[2]) : 44) * MiB;

    const int heavy = n_blocks == 1;

    /* control: the same kernels on the runtime's own memory */
    say("== control: hipMalloc %zu MiB", size / MiB);
    void *heap = nullptr;
    HIPCHECK(hipMalloc(&heap, size));
    HIPCHECK(hipMemset(heap, 0, 4096));
    report("hipMalloc", heap, size, 0, heavy);
    say("");

    /* the hook's route */
    hipMemGenericAllocationHandle_t pad = nullptr;
    HIPCHECK(hipMemCreate(&pad, p.granule, &p.prop, 0));
    say("== VMM, the hook's sequence, %d block(s) of %zu MiB, one shared pad granule at %p", n_blocks,
        size / MiB, (void *)pad);

    Bump bump{};
    bump.reserved = 1024 * GiB; /* the hook's default reservation */
    HIPCHECK(hipMemAddressReserve(&bump.base, bump.reserved, 0, nullptr, 0));
    say("  reservation %p (%zu GiB)", bump.base, bump.reserved / GiB);

    std::vector<Block> blocks;
    for (int i = 0; i < n_blocks; i++) {
        Block b = vmm_alloc(&p, &bump, size, 1, pad);
        if (n_blocks <= 4 || i % 50 == 0 || i == n_blocks - 1)
            say("  block %d at %p..%p, pad %p (bump used %zu MiB)", i, b.va,
                (char *)b.va + b.block, b.pad, bump.used / MiB);
        report("VMM block", b.va, size, 1, heavy);
        blocks.push_back(b);
    }

    for (Block &b : blocks) {
        HIPCHECK(hipMemUnmap(b.pad, p.granule));
        HIPCHECK(hipMemUnmap(b.va, b.block));
        HIPCHECK(hipMemRelease(b.handle));
    }
    HIPCHECK(hipMemAddressFree(bump.base, bump.reserved));
    HIPCHECK(hipMemRelease(pad));
    HIPCHECK(hipFree(heap));
    say("all of it completed: the same kernels ran on the VMM blocks without a fault");
    return 0;
}

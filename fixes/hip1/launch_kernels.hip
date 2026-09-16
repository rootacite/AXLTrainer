/*
 * Ask every bf16 Tensile kernel, one at a time, whether it reads past its operands, and how far.
 *
 * Method 2 of the bf16 audit (`kernel_inventory.py` writes the plan, hip1.md §12.14): instead of
 * letting hipBLASLt pick a kernel for a shape, this program loads the library's code object
 * itself, resolves one kernel by name, fills the argument block the code object's own metadata
 * declares, and launches that kernel on a shape built from *its* macro tile.
 *
 * Operands are placed with the VMM API, so a read past an operand lands on an unmapped page and
 * ROCr names the kernel.  `--pad` decides how much *mapped* slack sits after every operand, and
 * that is the measure of how far a kernel overruns: with pad 0 the first byte past the operand
 * faults, with pad 4096 only a read more than a page past it does.  Two passes over the same job
 * list therefore separate "overruns at all" from "overruns far enough to hit a hole in a real
 * allocation", and the padding is zeroed so it cannot change D -- the expected-result check below
 * still means what it says.
 *
 * The fault kills the process that takes it (hip1.md §12.1), so the sweep runs one child per slice
 * of the job list and each child writes `AT <job>` to a marker file, fsync'd, before every launch.
 * When a child dies the driver reads the marker to learn which job it was, records FAULT for it,
 * and restarts the child at the next job: one overrun costs one job, not the run.
 *
 * Every kernel's first shape is a control that fills its macro tiles exactly, checked against the
 * exact D that `kernel_inventory.py` computed (`dcrc` in the job list). A WRONG control means this
 * program's argument block or strides do not match that kernel -- reported as uninstrumentable,
 * never as an overrun. A control that *faults* is not that: the fix3 kernel overruns even on exact
 * tiles (hip1.md §12.14), so a faulting control is an overrun like any other, and `--pad` is how it
 * gets its result crc checked anyway. D is primed with 0xffff before every attempt, a value no
 * exact small-integer accumulation can produce, so a launch that landed nowhere shows up as NOOP --
 * the case that used to look like a kernel writing zeros (hip1.md §12.14).
 *
 * Argument block convention (settled from the ROCm 7.14.1 sources, not by calibration -- see
 * hip1.md §12.14; paths are inside rocm-libraries):
 *   - A solution's layout letters are its index assignments in rank order: the gfx1200 logic for
 *     `Cijk_Ailk_Bjlk_...` carries `IndexAssignmentsA: [0, 3, 2]` (i = Free0, l = Sum0, k = batch)
 *     and `IndexAssignmentsB: [1, 3, 2]`, with `TransposeA: false, TransposeB: true`
 *     (hipblaslt/library/.../Logic/asm_full/gfx1200/GridBased/gfx1200_Cijk_Ailk_Bjlk_*.yaml).
 *     Dimension 0 of every operand is therefore the contiguous one, its stride is 1, and it is
 *     never passed: `singleCallArgs` (hipblaslt/tensilelite/src/ContractionSolution.cpp) appends
 *     the strides of dimensions 1.. as `strideA0`, `strideA1`, ... -- in *elements*, because the
 *     only stride hipBLASLt hands `ContractionProblemGemm` is `row_stride = 1, col_stride = ld`
 *     (hipblaslt/library/.../rocblaslt/src/tensile_host.cpp, the problem constructor).
 *   - So `SizesFree0 = M` is the *first* of the kernel's two free dimensions and the first macro
 *     tile number: for the trainer's shape the kernel is dispatched with M = the plugin's N, since
 *     hipBLASLt's caller presents a row-major problem as the transposed product.
 *   - The launch grid is one flat id: `calculateGrid` multiplies ceil(M/MT0) * ceil(N/MT1) and
 *     `numWorkGroups.x *= y * z` flattens it, and the kernel's `remapWgSerial` takes wg0 as the
 *     fastest index (KernelWriterAssembly.py), reconstructing numWG0/1 from SizesFree and MT.
 *
 * Tiers:
 *   packed   one mapping per operand with a guard page after the padding: the class that kills real
 *            runs (a read past the end of the allocation).
 *   slotted  one mapping per row, the row's last byte + pad at the end of its mapping: any read past
 *            the row faults, including one that a packed layout would silently serve from the next
 *            row. Only shapes whose K keeps the row start aligned are eligible (the job list says).
 *
 * usage (the whole pipeline is fixes/hip1/scan_bf16_kernels.sh):
 *   launch_kernels --jobs FILE --kernel NAME [--shape M N K] [--dcrc 0x..] [--pad N] [--tier packed]
 *   launch_kernels --jobs FILE [--from N] [--count N] [--pad N] [--tier packed|slotted]
 *   launch_kernels --jobs FILE --from N --to N --marker FILE --pad N    (internal; driver calls it)
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

#define HIPCHECK(call)                                                                     \
    do {                                                                                   \
        hipError_t e_ = (call);                                                            \
        if (e_ != hipSuccess) {                                                            \
            fprintf(stderr, "%s:%d  %s -> %s\n", __FILE__, __LINE__, #call,                \
                    hipGetErrorString(e_));                                                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// --------------------------------------------------------------------------------- job list

struct Job {
    std::string library, kernel, layout, args;
    int kernarg = 144, workgroup = 128, group_seg = 0;
    uint32_t info0 = 0, info1 = 0;
    int gx = 1, gy = 1, gz = 1;
    int m = 0, n = 0, k = 0;
    bool slotted = false;                 // eligible for the slotted tier (from the job list)
    std::string label;
    uint32_t dcrc = 0;
    bool has_dcrc = false;
    int dump = 0;
    bool ones = false;      // fill both operands with 1 instead of the formula
    int expect = -1;        // with --fill ones: the value every D element should hold
    int spike_i = -1, spike_l = -1, spike_j = -1;   // one-hot probe coordinates (-1 = off)
};

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at <= s.size()) {
        size_t hit = s.find(sep, at);
        out.push_back(s.substr(at, hit == std::string::npos ? std::string::npos : hit - at));
        if (hit == std::string::npos) break;
        at = hit + 1;
    }
    return out;
}

static std::vector<Job> read_jobs(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(1);
    }
    std::vector<Job> jobs;
    char* line = nullptr;
    size_t cap = 0;
    ssize_t len;
    int lineno = 0;
    while ((len = getline(&line, &cap, f)) > 0) {
        std::string row(line, (size_t)len);
        while (!row.empty() && (row.back() == '\n' || row.back() == '\r')) row.pop_back();
        if (++lineno == 1 || row.empty()) continue;                 // header
        const auto f13 = split(row, '\t');
        if (f13.size() < 16) {
            fprintf(stderr, "%s:%d: expected 16 columns, got %zu\n", path, lineno, f13.size());
            exit(1);
        }
        Job j;
        j.library = f13[0];
        j.kernel = f13[1];
        j.layout = f13[2];
        j.kernarg = atoi(f13[3].c_str());
        j.args = f13[4];
        j.info0 = (uint32_t)strtoul(f13[5].c_str(), nullptr, 16);
        j.info1 = (uint32_t)strtoul(f13[6].c_str(), nullptr, 16);
        j.workgroup = atoi(f13[7].c_str());
        const auto grid = split(f13[8], ',');
        j.gx = atoi(grid[0].c_str());
        j.gy = atoi(grid[1].c_str());
        j.gz = atoi(grid[2].c_str());
        j.m = atoi(f13[9].c_str());
        j.n = atoi(f13[10].c_str());
        j.k = atoi(f13[11].c_str());
        j.slotted = atoi(f13[12].c_str()) != 0;
        j.label = f13[13];
        j.dcrc = (uint32_t)strtoul(f13[14].c_str(), nullptr, 16);
        j.has_dcrc = true;
        j.group_seg = atoi(f13[15].c_str());
        jobs.push_back(j);
    }
    free(line);
    fclose(f);
    return jobs;
}

// --------------------------------------------------------------------------------- operands
//
// `rows` slots of `slot_bytes` each, every slot evenly spaced by (mapped_slot + one guard page) and
// ending at a page boundary, so the last byte of a slot is the last mapped byte of that slot.

struct Slots {
    void* base = nullptr;
    size_t rows = 0, slot_bytes = 0, pad = 0, mapped_slot = 0, stride = 0, total = 0;
    hipMemGenericAllocationHandle_t handle = nullptr;

    // The operand's first byte: `slot_bytes` back from the end of slot 0's mapping, with `pad`
    // bytes of mapped slack deliberately left after it.
    void* operand() const { return (char*)base + (mapped_slot - pad - slot_bytes); }
};

static size_t granularity() {
    static size_t gran = 0;
    if (!gran) {
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = 0;
        HIPCHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    }
    return gran;
}

static hipMemLocation device_location() {
    hipMemLocation loc = {};
    loc.type = hipMemLocationTypeDevice;
    loc.id = 0;
    return loc;
}

static Slots guarded_slots(size_t rows, size_t slot_bytes, size_t pad) {
    if (!rows || !slot_bytes) {
        fprintf(stderr, "guarded_slots: empty operand\n");
        exit(1);
    }
    const size_t gran = granularity();
    Slots out;
    out.rows = rows;
    out.slot_bytes = slot_bytes;
    out.pad = pad;
    out.mapped_slot = ((slot_bytes + pad + gran - 1) / gran) * gran;
    out.stride = out.mapped_slot + gran;                            // + the guard page
    out.total = rows * out.stride;

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location = device_location();
    HIPCHECK(hipMemCreate(&out.handle, out.mapped_slot, &prop, 0));
    HIPCHECK(hipMemAddressReserve(&out.base, out.total, gran, nullptr, 0));
    for (size_t r = 0; r < rows; r++)
        HIPCHECK(hipMemMap((char*)out.base + r * out.stride, out.mapped_slot, 0, out.handle, 0));
    hipMemAccessDesc desc = {};
    desc.location = device_location();
    desc.flags = hipMemAccessFlagsProtReadWrite;
    // per slot: the range in between is the guard page and stays unmapped, so a single call over
    // the whole reservation is rejected
    for (size_t r = 0; r < rows; r++)
        HIPCHECK(hipMemSetAccess((char*)out.base + r * out.stride, out.mapped_slot, &desc, 1));
    return out;
}

static void release(Slots& s) {
    if (!s.base) return;
    for (size_t r = 0; r < s.rows; r++) HIPCHECK(hipMemUnmap((char*)s.base + r * s.stride, s.mapped_slot));
    HIPCHECK(hipMemAddressFree(s.base, s.total));
    HIPCHECK(hipMemRelease(s.handle));
    s.base = nullptr;
}

static void show(const char* name, const Slots& s) {
    printf("%-2s %p .. %p  %zu B + %zu B pad in %zu slot(s), stride %zu B, guard at %p\n", name,
           s.operand(), (char*)s.operand() + s.slot_bytes * s.rows, s.slot_bytes * s.rows, s.pad,
           s.rows, s.stride, (char*)s.base + s.mapped_slot);
}

// --------------------------------------------------------------------------------- crc32

static uint32_t crc32_bytes(const void* data, size_t len) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int b = 0; b < 8; b++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
            table[i] = c;
        }
        ready = true;
    }
    const uint8_t* p = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// --------------------------------------------------------------------------------- the operands' contents

// A[i][l] = (2i + l) % 3 for l < 120, B[l][j] = 1 when (2l + j) % 3 == 0: small integers, exact in
// bf16, and the same closed form kernel_inventory.py uses to compute the expected D crc.
static const int REF_LIMIT = 120;

static uint16_t bf16_of(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    return (uint16_t)(bits >> 16);
}

static const char* SCALE_ARGS[] = {"AddressScaleA", "AddressScaleB", "AddressScaleC",
                                   "AddressScaleD", "AddressScaleAlphaVec", "bias"};

struct Args {
    std::map<std::string, int> offset;
    bool has(const char* name) const { return offset.find(name) != offset.end(); }
    int at(const char* name) const { return offset.at(name); }
};

static Args parse_args(const std::string& spec) {
    Args out;
    for (const auto& item : split(spec, ';')) {
        size_t eq = item.find('=');
        if (eq != std::string::npos) out.offset[item.substr(0, eq)] = atoi(item.substr(eq + 1).c_str());
    }
    return out;
}

static void put(std::vector<uint8_t>& block, int offset, const void* src, size_t size) {
    if ((size_t)offset + size > block.size()) {
        fprintf(stderr, "argument at offset %d does not fit the %zu-byte block\n", offset,
                block.size());
        exit(1);
    }
    memcpy(block.data() + offset, src, size);
}

static void set_u32(std::vector<uint8_t>& b, const Args& a, const char* name, uint32_t v) {
    if (a.has(name)) put(b, a.at(name), &v, 4);
}

static void set_f32(std::vector<uint8_t>& b, const Args& a, const char* name, float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    set_u32(b, a, name, bits);
}

static void set_ptr(std::vector<uint8_t>& b, const Args& a, const char* name, void* v) {
    if (a.has(name)) put(b, a.at(name), &v, 8);
}

// --------------------------------------------------------------------------------- descriptors
//
// One operand's dimensions in the kernel's own rank order, as the layout token gives them ("Ailk"
// = (free, sum, batch), "Bjlk" = (free, sum, batch)). Dimension 0 is the contiguous one and its
// stride of 1 is implicit -- the kernel is told the strides of the dimensions after it.

struct Descriptor {
    size_t size[3] = {1, 1, 1};
    size_t stride[3] = {1, 1, 1};
    int    axis[3] = {-1, -1, -1};      // 0 = this operand's free index, 1 = sum, 2 = batch
    size_t elements = 0;

    bool valid() const { return axis[0] >= 0 && axis[1] >= 0; }
};

static Descriptor descriptor(const std::string& token, int m, int n, int k) {
    Descriptor d;
    for (int i = 0; i < 3; i++) {
        const char c = token[(size_t)i + 1];                  // "Ailk"/"Bjlk": skip the operand letter
        const int axis = c == 'k' ? 2 : (c == 'l' ? 1 : 0);   // i/j = free, l = sum, k = batch
        d.axis[i] = axis;
        d.size[i] = axis == 2 ? 1 : axis == 1 ? (size_t)k : (c == 'i' ? (size_t)m : (size_t)n);
    }
    for (int i = 1; i < 3; i++) d.stride[i] = d.stride[i - 1] * d.size[i - 1];
    d.elements = d.stride[2] * d.size[2];
    return d;
}

// Element offset of the operand entry the kernel addresses as (free, l)
static size_t elem_offset(const Descriptor& d, int free, int l) {
    size_t off = 0;
    for (int i = 0; i < 3; i++) {
        if (d.axis[i] == 0) off += d.stride[i] * (size_t)free;
        else if (d.axis[i] == 1) off += d.stride[i] * (size_t)l;
    }
    return off;
}

// `ESMRuntimeSupported` is not a solution property: the host appends the device's
// hipDeviceAttributeExpertSchedMode on gfx1200/gfx1201 only (ContractionSolution.cpp
// `if(sizeMapping.expertSchedulingMode > 0)`), and every kernel here declares the argument.
static uint32_t esm_supported() {
    hipDeviceProp_t prop = {};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 0;
    const std::string arch(prop.gcnArchName);
    if (arch.rfind("gfx1200", 0) != 0 && arch.rfind("gfx1201", 0) != 0) return 0;
    int value = 0;
    (void)hipDeviceGetAttribute(&value, hipDeviceAttributeExpertSchedMode, 0);
    return (uint32_t)value;
}

// --------------------------------------------------------------------------------- one job

struct Outcome {
    std::string status;                    // CLEAN | WRONG | ERROR
    uint32_t crc = 0;
    size_t bytes = 0;
};

static Outcome run_job(hipFunction_t fn, const Job& job, bool slotted, size_t pad, bool verbose) {
    const int m = job.m, n = job.n, k = job.k;   // the kernel's own Free0, Free1, Sum0
    int mt_m = 1, mt_n = 1;
    if (const size_t at = job.kernel.find("_MT"); at != std::string::npos)
        sscanf(job.kernel.c_str() + at, "_MT%dx%dx", &mt_m, &mt_n);

    const std::vector<std::string> tokens = split(job.layout, '_');
    if (tokens.size() != 2 || tokens[0].size() != 4 || tokens[1].size() != 4) {
        fprintf(stderr, "cannot read the operand layout out of \"%s\"\n", job.layout.c_str());
        exit(1);
    }
    const Descriptor da = descriptor(tokens[0], m, n, k);
    const Descriptor db = descriptor(tokens[1], m, n, k);
    const Descriptor dd = descriptor("Dijk", m, n, k);        // C and D: (i, j, batch)

    if (!da.valid() || !db.valid() || !dd.valid()) {
        fprintf(stderr, "layout \"%s\" is not a (free, sum, batch) assignment\n", job.layout.c_str());
        exit(1);
    }

    // one flat grid, wg0 fastest: the kernel reconstructs numWG0/1 and early-stops tiles past the end
    const uint32_t nwg = (uint32_t)((m + mt_m - 1) / mt_m) * (uint32_t)((n + mt_n - 1) / mt_n);

    const size_t d_bytes = dd.elements * 2;
    // the slotted tier guards one contiguous run per slot; the run is dimension 0
    const size_t a_run = da.size[0], a_rows = da.elements / da.size[0];
    const size_t b_run = db.size[0], b_rows = db.elements / db.size[0];

    Slots a = guarded_slots(slotted ? a_rows : 1, slotted ? a_run * 2 : da.elements * 2, pad);
    Slots b = guarded_slots(slotted ? b_rows : 1, slotted ? b_run * 2 : db.elements * 2, pad);
    Slots c = guarded_slots(1, d_bytes, pad);
    Slots d = guarded_slots(1, d_bytes, pad);

    // Every device operation of one job goes on one stream, so the padding, the operand fills, the
    // launch and the readback are ordered by construction. A launch on its own stream while the
    // fills sit on the null stream races, and that race looks exactly like a kernel that wrote
    // zeros -- the flake that first showed up as 3 % of the controls reported WRONG (hip1.md
    // §12.14).
    hipStream_t stream = nullptr;
    HIPCHECK(hipStreamCreate(&stream));
    for (Slots* s : {&a, &b, &c, &d})
        for (size_t r = 0; r < s->rows; r++)
            HIPCHECK(hipMemsetAsync((char*)s->base + r * s->stride, 0, s->mapped_slot, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    // strides in elements; with one slot per run the slot pitch replaces the dense outer stride
    size_t sa[3], sb[3], sd[3];
    for (int i = 0; i < 3; i++) {
        sa[i] = da.stride[i];
        sb[i] = db.stride[i];
        sd[i] = dd.stride[i];
    }
    if (slotted) {
        sa[1] = a.stride / 2;
        sb[1] = b.stride / 2;
        sa[2] = sa[1] * a_rows;
        sb[2] = sb[1] * b_rows;
    }

    if (verbose) {
        printf("strides           : A(%zu,%zu) B(%zu,%zu) D(%zu,%zu), kernel M=%d N=%d K=%d\n", sa[0],
               sa[1], sb[0], sb[1], sd[0], sd[1], m, n, k);
        printf("descriptors       : A dims (%zu,%zu,%zu) axes (%d,%d,%d) in %zu elements\n",
               da.size[0], da.size[1], da.size[2], da.axis[0], da.axis[1], da.axis[2],
               da.elements);
        printf("                    B dims (%zu,%zu,%zu) axes (%d,%d,%d) in %zu elements\n",
               db.size[0], db.size[1], db.size[2], db.axis[0], db.axis[1], db.axis[2],
               db.elements);
        show("A", a);
        show("B", b);
        show("C", c);
        show("D", d);
    }

    // The operand contents, built on the host at each element's descriptor offset: A(i, l) =
    // (2i+l) % 3 below REF_LIMIT, B(l, j) = 1 when (2l+j) % 3 == 0 -- small integers, exact in
    // bf16, and the same closed form kernel_inventory.py builds the expected D crc from.
    std::vector<uint16_t> host_a(da.elements), host_b(db.elements);
    for (int i = 0; i < m; i++)
        for (int l = 0; l < k; l++) {
            float av;
            if (job.spike_i >= 0)
                av = (i == job.spike_i && l == job.spike_l) ? 1.0f : 0.0f;
            else
                av = job.ones ? 1.0f : (float)(l < REF_LIMIT ? (2 * i + l) % 3 : 0);
            host_a[elem_offset(da, i, l)] = bf16_of(av);
        }
    for (int l = 0; l < k; l++)
        for (int j = 0; j < n; j++) {
            float bv;
            if (job.spike_j >= 0)
                bv = (l == job.spike_l && j == job.spike_j) ? 1.0f : 0.0f;
            else
                bv = (job.ones || (2 * l + j) % 3 == 0) ? 1.0f : 0.0f;
            host_b[elem_offset(db, j, l)] = bf16_of(bv);
        }
    // the packed tier is one contiguous operand; the slotted tier has a guard page between runs
    auto upload = [&](const Slots& s, const std::vector<uint16_t>& host, size_t run, size_t rows) {
        if (!slotted) {
            HIPCHECK(hipMemcpyAsync(s.operand(), host.data(), host.size() * 2,
                                    hipMemcpyHostToDevice, stream));
            return;
        }
        for (size_t r = 0; r < rows; r++)
            HIPCHECK(hipMemcpyAsync((char*)s.operand() + r * s.stride, host.data() + r * run,
                                    run * 2, hipMemcpyHostToDevice, stream));
    };
    // read a mapping back: a fill that did not land would otherwise be indistinguishable from a
    // kernel that read zeros, and the control's crc check would blame the kernel for it
    auto landed = [&](const Slots& s, const std::vector<uint16_t>& host, size_t run, size_t rows) {
        std::vector<uint16_t> back(host.size());
        if (!slotted) {
            HIPCHECK(hipMemcpy(back.data(), s.operand(), host.size() * 2, hipMemcpyDeviceToHost));
        } else {
            for (size_t r = 0; r < rows; r++)
                HIPCHECK(hipMemcpy(back.data() + r * run, (char*)s.operand() + r * s.stride,
                                   run * 2, hipMemcpyDeviceToHost));
        }
        return back == host;
    };
    // Scale vectors and bias: identity values, so a kernel that does read them still computes the
    // expected D, and one that reads past them faults on a guard page like anything else.
    const Args args = parse_args(job.args);
    std::vector<float> identity((size_t)std::max(m, n), 1.0f);
    std::vector<float> zeros(identity.size(), 0.0f);
    std::vector<Slots> extras;
    std::map<std::string, void*> extra_ptr;
    for (const char* name : SCALE_ARGS) {
        if (!args.has(name)) continue;
        Slots s = guarded_slots(1, identity.size() * sizeof(float), 0);
        const bool zero = std::string(name) == "bias";
        HIPCHECK(hipMemcpyAsync(s.operand(), zero ? zeros.data() : identity.data(),
                                identity.size() * sizeof(float), hipMemcpyHostToDevice, stream));
        extra_ptr[name] = s.operand();
        extras.push_back(s);
    }

    std::vector<uint8_t> block((size_t)job.kernarg, 0);
    set_u32(block, args, "Gemm info", 0x00000001u);      // argument type 0 (inline), 1 GEMM
    set_u32(block, args, "kernel info0", job.info0);
    set_u32(block, args, "kernel info1", job.info1);
    set_u32(block, args, "numWG", nwg);
    set_u32(block, args, "SizesFree0", (uint32_t)m);
    set_u32(block, args, "SizesFree1", (uint32_t)n);
    set_u32(block, args, "SizesFree2", 1);
    set_u32(block, args, "SizesSum0", (uint32_t)k);
    set_ptr(block, args, "A", a.operand());
    set_ptr(block, args, "B", b.operand());
    set_ptr(block, args, "C", c.operand());
    set_ptr(block, args, "D", d.operand());
    // strideX0 takes the descriptor's *second* dimension (hip1.md §12.14): dimension 0 is the
    // contiguous one and the kernel hard-codes its stride of 1
    auto set_strides = [&](const char* prefix, const size_t* st) {
        for (int i = 1; i < 3; i++) {
            char name[16];
            snprintf(name, sizeof name, "%s%d", prefix, i - 1);
            set_u32(block, args, name, (uint32_t)st[i]);
        }
    };
    set_strides("strideA", sa);
    set_strides("strideB", sb);
    set_strides("strideC", sd);
    set_strides("strideD", sd);
    set_f32(block, args, "alpha", 1.0f);
    set_f32(block, args, "beta", 0.0f);
    set_u32(block, args, "ESMRuntimeSupported", esm_supported());
    for (const auto& kv : extra_ptr) set_ptr(block, args, kv.first.c_str(), kv.second);
    set_u32(block, args, "biasType", 0);
    set_u32(block, args, "StrideBias", 0);
    set_u32(block, args, "activationType", 0);
    set_f32(block, args, "activationAlpha", 0.0f);
    set_f32(block, args, "activationBeta", 0.0f);

    size_t block_size = block.size();
    void* launch_extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, block.data(),
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &block_size, HIP_LAUNCH_PARAM_END};
    if (verbose)
        printf("launch            : flat grid %u blocks, block %d, kernarg %zu B\n", nwg,
               job.workgroup, block.size());

    // D is primed with a sentinel that no exact small-integer accumulation can produce, so a launch
    // that never landed is a result this program can see instead of a silent zero. What landed is
    // checked before launching, so a fill the driver dropped is not blamed on the kernel.
    static const uint16_t kSentinel = 0xffff;
    std::vector<uint16_t> host_d(dd.elements, 0);
    const std::vector<uint16_t> sentinel(dd.elements, kSentinel);
    auto d_primed = [&]() {
        std::vector<uint16_t> back(dd.elements);
        HIPCHECK(hipMemcpy(back.data(), d.operand(), d_bytes, hipMemcpyDeviceToHost));
        return back == sentinel;
    };
    auto readback = [&]() {
        HIPCHECK(hipMemcpy(host_d.data(), d.operand(), d_bytes, hipMemcpyDeviceToHost));
    };
    auto unwritten = [&]() {
        for (uint16_t v : host_d)
            if (v != kSentinel) return false;
        return true;
    };

    Outcome out;
    bool primed = false;
    for (int t = 0; t < 3 && !primed; t++) {
        upload(a, host_a, a_run, a_rows);
        upload(b, host_b, b_run, b_rows);
        HIPCHECK(hipMemcpyAsync(d.operand(), sentinel.data(), d_bytes, hipMemcpyHostToDevice,
                                stream));
        HIPCHECK(hipStreamSynchronize(stream));
        primed = d_primed() && landed(a, host_a, a_run, a_rows) && landed(b, host_b, b_run, b_rows);
    }
    if (!primed) {
        out.status = "BADCOPY";                            // fills did not reach the device
        return out;
    }
    auto launch = [&]() -> hipError_t {
        HIPCHECK(hipModuleLaunchKernel(fn, nwg, 1, 1, (unsigned)job.workgroup, 1, 1, 0, stream,
                                       nullptr, launch_extra));
        return hipStreamSynchronize(stream);               // a fault aborts the process here
    };
    hipError_t err = launch();
    if (err == hipSuccess) {
        readback();
        if (unwritten()) {                                 // retry once, then say so
            HIPCHECK(hipMemcpyAsync(d.operand(), sentinel.data(), d_bytes, hipMemcpyHostToDevice,
                                    stream));
            err = launch();
            if (err == hipSuccess) readback();
        }
    }
    if (err != hipSuccess) {
        out.status = "ERROR";
        printf("sync              : %s (%d)\n", hipGetErrorString(err), (int)err);
        return out;
    }
    if (unwritten()) {
        out.status = "NOOP";                               // D untouched: the launch did nothing
        return out;
    }
    auto d_at = [&](int i, int j) -> float {
        const uint32_t bits = (uint32_t)host_d[(size_t)i + (size_t)j * (size_t)m] << 16;
        float value;
        memcpy(&value, &bits, 4);
        return value;
    };
    const int total = m * n;
    if (job.expect >= 0) {
        int wrong = 0, first = -1, last = -1;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                if ((int)d_at(i, j) != job.expect) {
                    if (!wrong) first = (int)((size_t)i + (size_t)j * (size_t)m);
                    last = (int)((size_t)i + (size_t)j * (size_t)m);
                    wrong++;
                }
        printf("D mismatch        : %d of %d elements differ from %d", wrong, total, job.expect);
        if (wrong) printf("  first at (i=%d, j=%d), last at (i=%d, j=%d)", first % m, first / m,
                          last % m, last / m);
        printf("\n");
    }
    if (job.dump) {
        printf("D[0..%d]          :", job.dump);
        for (int j = 0; j < n && j * m < job.dump; j++)
            for (int i = 0; i < m && j * m + i < job.dump; i++)
                printf(" %.0f", d_at(i, j));
        printf("\n");
    }
    out.crc = crc32_bytes(host_d.data(), d_bytes);
    out.bytes = d_bytes;
    out.status = (job.has_dcrc && out.crc != job.dcrc) ? "WRONG" : "CLEAN";

    HIPCHECK(hipStreamDestroy(stream));
    for (Slots& s : extras) release(s);
    release(a);
    release(b);
    release(c);
    release(d);
    return out;
}

// --------------------------------------------------------------------------------- plumbing

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) out += (c == '\'') ? "'\\''" : std::string(1, c);
    return out + "'";
}

static std::string default_libdir() {
    const char* env = getenv("AXL_HIPBLASLT_LIBDIR");
    return env ? env : "";
}

static std::string library_path(const std::string& libdir, const std::string& library) {
    if (libdir.empty()) {
        fprintf(stderr, "no --libdir (or $AXL_HIPBLASLT_LIBDIR): cannot find %s\n", library.c_str());
        exit(1);
    }
    return libdir + "/" + library;
}

static hipModule_t load_library(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        perror(path.c_str());
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> image((size_t)size);
    if (fread(image.data(), 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
    hipModule_t module = nullptr;
    HIPCHECK(hipModuleLoadData(&module, image.data()));
    return module;
}

static void usage(const char* self) {
    fprintf(stderr,
            "usage: %s --jobs FILE --kernel NAME [--shape M N K] [--dcrc 0x..] [--tier packed]\n"
            "       %s --jobs FILE [--library LIB] [--from N] [--count N] [--tier packed|slotted]\n"
            "       %s --jobs FILE --from N --to N --marker FILE   (child; the driver calls this)\n"
            "\n"
            "  --shape M N K  the kernel's own free and sum sizes (Free0, Free1, Sum0), not the\n"
            "                 plugin's: for `Cijk_Ailk_Bjlk` the kernel's M is the plugin's N\n"
            "  --pad N        mapped slack after each operand: 0 = fault on the first byte past it,\n"
            "                 4096 = only a read more than a page past it faults\n"
            "  --libdir DIR   hipBLASLt gfx1201 library (or $AXL_HIPBLASLT_LIBDIR)\n",
            self, self, self);
}

// --------------------------------------------------------------------------------- child

static int run_child(const std::vector<Job>& jobs, const std::string& libdir, int from, int to,
                     const std::string& marker, bool slotted, size_t pad) {
    if (from >= (int)jobs.size()) return 0;
    if (to < 0 || to > (int)jobs.size()) to = (int)jobs.size();

    const std::string library = jobs[from].library;      // a slice never crosses libraries
    hipModule_t module = load_library(library_path(libdir, library));
    FILE* mark = nullptr;
    if (!marker.empty()) {
        mark = fopen(marker.c_str(), "w");
        if (!mark) {
            perror(marker.c_str());
            return 1;
        }
    }

    for (int i = from; i < to; i++) {
        const Job& job = jobs[i];
        if (job.library != library) break;
        if (slotted && !job.slotted) {
            printf("JOB %d SKIP not-slotted-eligible\n", i);
            fflush(stdout);
            continue;
        }
        if (mark) {                                      // fsync'd before the launch that may die
            fseek(mark, 0, SEEK_SET);
            fprintf(mark, "%d %zu\n", i, pad);
            fflush(mark);
            fsync(fileno(mark));
        }
        hipFunction_t fn = nullptr;
        hipError_t err = hipModuleGetFunction(&fn, module, job.kernel.c_str());
        if (err != hipSuccess) {
            printf("JOB %d ERROR %s\n", i, hipGetErrorString(err));
            fflush(stdout);
            continue;
        }
        const Outcome out = run_job(fn, job, slotted, pad, false);
        printf("JOB %d %s crc %08x expected %08x bytes %zu\n", i,
               out.status.c_str(), out.crc, job.dcrc, out.bytes);
        fflush(stdout);
    }
    if (mark) fclose(mark);
    return 0;
}

// --------------------------------------------------------------------------------- driver

static int drive(const std::vector<Job>& jobs, const std::string& libdir, const std::string& jobs_path,
                 const std::string& marker, const std::string& tier, int from, int limit,
                 size_t pad, const char* self) {
    const bool slotted = tier == "slotted";
    const std::string slice_marker = marker.empty() ? "/tmp/axl-hip1-kernel-marker" : marker;
    const int end = limit > 0 ? std::min((int)jobs.size(), from + limit) : (int)jobs.size();
    int start = from, faults = 0, wrong = 0, errors = 0, clean = 0, skipped = 0, noops = 0;
    int badcopies = 0;

    while (start < end) {
        int stop = start;
        while (stop < end && jobs[stop].library == jobs[start].library) stop++;

        std::vector<std::string> argv = {self, "--jobs", jobs_path, "--from", std::to_string(start),
                                        "--to", std::to_string(stop), "--marker", slice_marker,
                                        "--tier", tier, "--pad", std::to_string(pad),
                                        "--libdir", libdir};
        std::string cmd;
        for (const auto& a : argv) cmd += shell_quote(a) + " ";
        printf("-- child [%d,%d) %s\n", start, stop, jobs[start].library.c_str());
        fflush(stdout);

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            perror("popen");
            return 1;
        }
        char* line = nullptr;
        size_t cap = 0;
        ssize_t len;
        while ((len = getline(&line, &cap, pipe)) > 0) {
            const std::string row(line, (size_t)len);
            const auto f = split(row, ' ');
            if (f.size() >= 3 && f[0] == "JOB") {
                if (f[2] == "CLEAN") clean++;
                else if (f[2] == "WRONG") wrong++;
                else if (f[2] == "SKIP") skipped++;
                else if (f[2] == "NOOP") noops++;
                else if (f[2] == "BADCOPY") badcopies++;
                else errors++;
            }
            if (f.size() >= 2 && (f[0] == "JOB" || f[0] == "AT")) {
                if (f[2] != "CLEAN") printf("%s", row.c_str());
            }
        }
        free(line);
        const int rc = pclose(pipe);
        const int signal = WIFEXITED(rc) ? 0 : WTERMSIG(rc);

        int in_flight = -1;
        FILE* mark = fopen(slice_marker.c_str(), "r");
        if (mark) {
            if (fscanf(mark, "%d", &in_flight) != 1) in_flight = -1;
            fclose(mark);
        }
        if (signal != 0 && in_flight >= start && in_flight < stop) {
            const Job& j = jobs[in_flight];
            faults++;
            printf("FAULT %d signal %d  %s\n", in_flight, signal, j.kernel.c_str());
            printf("      %s  %dx%dx%d  %s  %s\n", j.label.c_str(), j.m, j.n, j.k,
                   j.slotted ? "slotted-eligible" : "packed-only", j.library.c_str());
            fflush(stdout);
            start = in_flight + 1;
        } else {
            if (signal != 0)
                printf("FAULT-UNKNOWN signal %d over [%d,%d) (marker %d)\n", signal, start, stop,
                       in_flight);
            start = stop;
        }
        fflush(stdout);
    }
    printf("\n== summary (tier %s) == clean %d  wrong %d  noop %d  badcopy %d  error %d  fault %d"
           "  skipped %d\n",
           tier.c_str(), clean, wrong, noops, badcopies, errors, faults, skipped);
    (void)slotted;
    return 0;
}

// --------------------------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::string jobs_path, kernel, library, marker, tier = "packed", libdir = default_libdir();
    int shape[3] = {0, 0, 0};
    bool have_shape = false, child = false;
    uint32_t dcrc = 0;
    bool have_dcrc = false;
    int dump = 0;
    int from = 0, to = -1, limit = 0;
    size_t pad = 0;
    bool ones = false;
    int expect = -1;
    int spike_i = -1, spike_l = -1, spike_j = -1;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (arg == "--jobs") jobs_path = next();
        else if (arg == "--kernel") kernel = next();
        else if (arg == "--library") { (void)next(); }         // informational; slices follow the list
        else if (arg == "--marker") { marker = next(); child = true; }
        else if (arg == "--tier") tier = next();
        else if (arg == "--libdir") libdir = next();
        else if (arg == "--from") from = atoi(next().c_str());
        else if (arg == "--to") to = atoi(next().c_str());
        else if (arg == "--count") limit = atoi(next().c_str());
        else if (arg == "--pad") pad = (size_t)strtoul(next().c_str(), nullptr, 10);
        else if (arg == "--dcrc") { dcrc = (uint32_t)strtoul(next().c_str(), nullptr, 16); have_dcrc = true; }
        else if (arg == "--dump") dump = atoi(next().c_str());
        else if (arg == "--fill") ones = next() == "ones";
        else if (arg == "--expect-const") expect = atoi(next().c_str());
        else if (arg == "--spike-a") { sscanf(next().c_str(), "%d,%d", &spike_i, &spike_l); ones = true; }
        else if (arg == "--spike-b") { sscanf(next().c_str(), "%d,%d", &spike_l, &spike_j); ones = true; }
        else if (arg == "--shape") {
            shape[0] = atoi(next().c_str());
            shape[1] = atoi(next().c_str());
            shape[2] = atoi(next().c_str());
            have_shape = true;
        } else if (arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (jobs_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    const std::vector<Job> jobs = read_jobs(jobs_path.c_str());

    if (child) return run_child(jobs, libdir, from, to, marker, tier == "slotted", pad);

    if (!kernel.empty()) {                                     // one kernel, one shape (verbose)
        std::vector<Job> picked;
        for (const Job& j : jobs)
            if (j.kernel == kernel) picked.push_back(j);
        if (picked.empty()) {
            fprintf(stderr, "no job for kernel %s\n", kernel.c_str());
            return 2;
        }
        Job job = picked.front();                              // the control shape, unless overridden
        for (const Job& j : picked)
            if (have_shape && j.m == shape[0] && j.n == shape[1] && j.k == shape[2]) job = j;
        if (!have_shape && picked.size() > 1)
            printf("note              : %zu shapes for this kernel, running %s; "
                   "--shape overrides it\n", picked.size(), job.label.c_str());
        if (have_shape) {
            int mt_m = 1, mt_n = 1;
            const size_t at = job.kernel.find("_MT");
            if (at != std::string::npos)
                sscanf(job.kernel.c_str() + at, "_MT%dx%dx", &mt_m, &mt_n);
            job.m = shape[0];                                  // the kernel's Free0, Free1, Sum0
            job.n = shape[1];
            job.k = shape[2];
            job.gx = (job.m + mt_m - 1) / mt_m;
            job.gy = (job.n + mt_n - 1) / mt_n;
            job.gz = 1;
            job.label = "manual";
            job.slotted = (job.k * 2) % 32 == 0;
            job.has_dcrc = have_dcrc;
            job.dcrc = dcrc;
            job.dump = dump;
            job.ones = ones;
            job.expect = expect;
            job.spike_i = spike_i;
            job.spike_l = spike_l;
            job.spike_j = spike_j;
        }
        printf("kernel            : %s\n", job.kernel.c_str());
        printf("library           : %s\n", job.library.c_str());
        printf("shape             : M=%d N=%d K=%d (the kernel's Free0, Free1, Sum0), block %d, "
               "tier %s\n", job.m, job.n, job.k, job.workgroup, tier.c_str());
        if (!have_dcrc && !job.has_dcrc)
            printf("note              : no expected D crc given, result not checked\n");
        hipModule_t module = load_library(library_path(libdir, job.library));
        hipFunction_t fn = nullptr;
        HIPCHECK(hipModuleGetFunction(&fn, module, job.kernel.c_str()));
        if (tier == "slotted" && !job.slotted) {
            fprintf(stderr, "this shape is not slotted-eligible (K*2 is not a multiple of 32)\n");
            return 2;
        }
        const Outcome out = run_job(fn, job, tier == "slotted", pad, true);
        printf("result            : %s   crc32 %08x (expected %08x) over %zu bytes\n",
               out.status.c_str(), out.crc, job.has_dcrc ? job.dcrc : 0u, out.bytes);
        return out.status == "CLEAN" ? 0 : 1;
    }

    return drive(jobs, libdir, jobs_path, marker, tier, from, limit, pad, argv[0]);
}

/*
 * Reproduce the fixes/fix3 fault on hipBLASLt's gfx1201 Tensile kernel
 *   Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_…_WS32_WG64_2_1
 * with nothing but hipModule: no torch, no hipBLASLt, no selection logic.
 *
 * The kernel reads past the end of its operands at an edge tile (K = 308 = 19x16 + 4
 * is a partial tile, M = 48 < 64 is another).  Whether that raises a fault depends
 * entirely on what is mapped after the buffer, and `hipMalloc` gives no control over
 * that: its four buffers land in one 2 MiB arena of adjacent, fully mapped device
 * memory, so the overrun silently reads a neighbour (measured — hip1.md §12.9).  So
 * each operand is placed by the VMM API instead, end-aligned in its own mapping with
 * an unmapped page right after it: the first byte past the operand is unmapped, and
 * the read has nowhere to go but a `Page not present` fault.
 *
 * Build and run in the environment's stack (HIP.md §3), or one command for all of it
 * (fixes/hip1/repro_standalone.sh):
 *     conda activate axl_rocm_7_14
 *     python fixes/hip1/extract_kernel_codeobject.py       # writes the code object
 *     hipcc --offload-arch=gfx1201 -O2 fixes/hip1/launch_fault_kernel.hip -o /tmp/launch_fault_kernel
 *     /tmp/launch_fault_kernel
 *
 * Expect ROCr's "Memory access fault by GPU node-1 … Reason: Page not present or
 * supervisor privilege" and an aborted process.  The fault address it prints, next to
 * the addresses this program prints, says which operand was overrun and how far; a
 * fault that reports RW=1 instead is a store (see hip1.md §12.9).
 *
 * It does fault, and it did on 2026-09-16: past the end of A — the M edge of the Ailk
 * operand (M = 48 only fills the first 48 rows of a 64-row macro tile), not its 24-byte K
 * tail — with ROCr naming A's guard page or the page 4096 bytes past it.  Both are
 * unmapped and which of the two it reports varies between runs of this program, so read
 * the fault address against the operands it prints rather than as a fixed offset.
 * Transcript and reading: hip1.md §12.9; the six repetitions behind that sentence:
 * hip1.md §12.12.
 *
 * The fp16 sibling of this kernel was measured with an earlier revision of this file and
 * overruns the same way; that variant has been withdrawn again so the program stays
 * single-purpose, and the record is hip1.md §12.12.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// The code object written by fixes/hip1/extract_kernel_codeobject.py (29 MB).
#define OBJECT_PATH                                                                            \
    "/tmp/axl-hip1/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_"   \
    "Cijk_Dijk_gfx1201.co.codeobject.elf"

// The faulting kernel, verbatim from the code object's `.kd` symbol and from ROCr's
// fault message (fixes/fix3/resources/live-run-fault.txt).
static const char* KERNEL =
    "Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_"
    "AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA1_DTVB1_DTVMXSA0_DTVMXSB0_"
    "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_"
    "LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_"
    "MIAV1_MIWT1_8_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_"
    "NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB8_ONLL0_PAP0_PGL0_PGR1_"
    "PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_"
    "TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS0_TLDSM1_ULSGRO0_"
    "USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG64_2_1";

// The failing dispatch: grad_A of a text_encoder_2 LoRA projection (hip1.md §12.1).
#define M 48
#define N 1280
#define K 308

// One workgroup per 64x128 macro tile of the output, 128 threads each.
#define GRID 10
#define BLOCK 128

// The four packed scalars of the argument block, and the values the kernel's prologue
// and the solution's own metadata imply (hip1.md §12.9 has the bit fields and the
// instruction that reads each one): arg type 0 = arguments inline, GSU 1, staggerU 32
// / stride shift 3 / mapping 1, WGM 8 | WGMXCC 1 | WGMXCCG -1.
#define GEMM_INFO 0x00000001u
#define KERNEL_INFO0 0x23200001u
#define KERNEL_INFO1 0xffc10008u

// The kernel's argument block: 144 bytes, laid out exactly as the code object's own
// NT_AMDGPU_METADATA note declares (.kernarg_segment_size = 144, and one .args entry
// per field below).  Only offsets 0..107 are ever read; the bias/activation tail pads
// the block to the size the runtime expects.
#pragma pack(push, 1)
struct Args {
    uint32_t gemm_info;   // 0
    uint32_t kernel_info0;
    uint32_t kernel_info1;
    uint32_t num_wg;               // 12
    uint32_t sizes_free[3];        // 16 M, 20 N, 24 batch
    uint32_t sizes_sum0;           // 28 K
    void* d;                       // 32
    void* c;
    void* a;
    void* b;                       // 56
    uint32_t stride_d0, stride_d1, stride_c0, stride_c1;      // 64
    uint32_t stride_a0, stride_a1, stride_b0, stride_b1;      // 80
    float alpha;                   // 96
    float beta;                    // 100
    uint32_t esm_runtime_supported;  // 104   (last field the kernel reads)
    void* address_scale_alpha_vec;   // 108 }
    void* bias;                      // 116 } unread
    uint32_t bias_type;              // 124 }
    uint32_t stride_bias;
    float activation_alpha, activation_beta;
    uint32_t activation_type;        // 140 }
};
#pragma pack(pop)

static_assert(sizeof(Args) == 144, "the metadata says kernarg_segment_size is 144");

#define HIPCHECK(call)                                                                     \
    do {                                                                                   \
        hipError_t e_ = (call);                                                            \
        if (e_ != hipSuccess) {                                                            \
            fprintf(stderr, "%s:%d  %s -> %s\n", __FILE__, __LINE__, #call,                \
                    hipGetErrorString(e_));                                                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// Reserve `size` bytes so that the last one is the last byte of a mapping: the next
// page is reserved but never mapped, so the first byte past the operand faults.  The
// padding in front is (4096 - size % 4096) % 4096 bytes, a multiple of 16 because
// every operand size is, which keeps the 16-byte vector loads GRVWA8/GRVWB8 aligned.
static void* guarded(const char* name, size_t size) {
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
    HIPCHECK(hipMemAddressReserve(&base, mapped + gran, gran, nullptr, 0));  // + one guard page
    HIPCHECK(hipMemMap(base, mapped, 0, handle, 0));

    hipMemAccessDesc desc = {};
    desc.location = prop.location;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIPCHECK(hipMemSetAccess(base, mapped, &desc, 1));

    void* p = (char*)base + (mapped - size);
    printf("%-2s %p .. %p  %zu bytes, %zu bytes of lead-in, guard page at %p\n", name, p,
           (char*)p + size, size, mapped - size, (char*)base + mapped);
    return p;
}

int main() {
    int device = 0;
    HIPCHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIPCHECK(hipGetDeviceProperties(&prop, device));
    printf("device            : %s (%s, %d CUs)\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);

    FILE* f = fopen(OBJECT_PATH, "rb");
    if (!f) {
        perror(OBJECT_PATH);
        fprintf(stderr, "run: python fixes/hip1/extract_kernel_codeobject.py\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    const long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> image((size_t)fsize);
    if (fread(image.data(), 1, (size_t)fsize, f) != (size_t)fsize) {
        fprintf(stderr, "short read on %s\n", OBJECT_PATH);
        return 1;
    }
    fclose(f);

    hipModule_t module = nullptr;
    HIPCHECK(hipModuleLoadData(&module, image.data()));
    hipFunction_t fn = nullptr;
    HIPCHECK(hipModuleGetFunction(&fn, module, KERNEL));
    printf("module            : %p, function %p\n", (void*)module, (void*)fn);

    // A[M,K] row-major, B[K,N] row-major (the `transB` of Bjlk), C and D [M,N] row-major.
    void* a = guarded("A", (size_t)M * K * 2);
    void* b = guarded("B", (size_t)K * N * 2);
    void* c = guarded("C", (size_t)M * N * 2);
    void* d = guarded("D", (size_t)M * N * 2);

    Args args{};
    args.gemm_info = GEMM_INFO;
    args.kernel_info0 = KERNEL_INFO0;
    args.kernel_info1 = KERNEL_INFO1;
    args.num_wg = GRID;
    args.sizes_free[0] = M;
    args.sizes_free[1] = N;
    args.sizes_free[2] = 1;
    args.sizes_sum0 = K;
    args.d = d;
    args.c = c;
    args.a = a;
    args.b = b;
    args.stride_d0 = N, args.stride_d1 = 1;
    args.stride_c0 = N, args.stride_c1 = 1;
    args.stride_a0 = K, args.stride_a1 = 1;
    args.stride_b0 = 1, args.stride_b1 = N;
    args.alpha = 1.0f;
    args.beta = 0.0f;

    printf("problem           : A[%d,%d] x B[%d,%d] -> D[%d,%d], bf16, grid %d, block %d\n", M, K, K, N, M, N,
           GRID, BLOCK);
    fflush(stdout);

    hipStream_t stream = nullptr;
    HIPCHECK(hipStreamCreate(&stream));
    size_t args_size = sizeof(args);
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                     HIP_LAUNCH_PARAM_END};
    HIPCHECK(hipModuleLaunchKernel(fn, GRID, 1, 1, BLOCK, 1, 1, /*sharedMemBytes=*/0, stream,
                                   /*kernelParams=*/nullptr, extra));

    // The fault lands here; ROCr prints its own line and aborts before this returns.
    hipError_t err = hipStreamSynchronize(stream);
    printf("result            : %s\n", err == hipSuccess ? "completed without a fault"
                                                          : hipGetErrorString(err));
    return err == hipSuccess ? 0 : 1;
}

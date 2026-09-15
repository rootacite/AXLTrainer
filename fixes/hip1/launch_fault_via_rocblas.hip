/*
 * Second route to the fixes/fix3 fault: let rocBLAS pick the kernel instead of
 * loading the Tensile code object by hand (that is launch_fault_kernel.hip).
 *
 * Same operands and the same guard-page placement as that program; the only thing
 * this one chooses is the rocBLAS call.
 *
 * rocBLAS on this stack hands bf16 gemm_ex to hipBLASLt — under `strace -e openat` the
 * run opens `hipblaslt/library/gfx1201/…` and nothing from rocBLAS's own
 * `rocblas/library/`.  hipBLASLt then picks a Tensile contraction library from the two
 * transpose flags, and the mapping is not the obvious one.  Measured on gfx1201 /
 * 7.14.60850, one row per run, with legal leading dimensions:
 *
 *     transA=none      transB=none      -> …_Contraction_l_Ailk_Bljk_Cijk_Dijk
 *     transA=none      transB=transpose -> …_Contraction_l_Ailk_Bjlk_Cijk_Dijk   <- the fault
 *     transA=transpose transB=none      -> …_Contraction_l_Alik_Bljk_Cijk_Dijk
 *     transA=transpose transB=transpose -> …_Contraction_l_Alik_Bjlk_Cijk_Dijk
 *
 * The A letter follows transA (none -> Ailk) and the B letter follows transB
 * (transpose -> Bjlk), so `_Ailk_Bjlk_` — the library the faulting kernel lives in —
 * means none/transpose.  That is the row-major case in a column-major API, and it is
 * the pair the failing dispatch must have used.
 *
 * Getting the same *solution* needs the problem handed over the way the row-major
 * caller does it, with the two operands exchanged: hipBLASLt's m is the row-major
 * tensor's last dimension (1280), its n is the network_dim (48), and the K stays 308.
 * Passing m=48, n=1280 instead reaches the right library but picks a different
 * solution — `…_MT32x32x64_…_WS32_WG32_4_1`, whose name comes from a run with A's
 * mapping shrunk to a single page so that any read past it has to name the kernel —
 * and with the guard pages in place that one completes without faulting.  With the
 * swap it selects `…_MT64x128x16_…_WS32_WG64_2_1`, and the dispatch then matches the
 * recorded fault on every field the runtime prints:
 *
 *     grid=[2560,1,1]  workgroup=[128,1,1]  private_seg_size=0  group_seg_size=1638
 *     kernel_obj=<load base> + 0x282380
 *
 * `0x282380` is the descriptor's vaddr inside the code object — the offset every
 * recorded `kernel_obj` shares, and what marks these as the same descriptor rather
 * than merely the same kernel name (hip1.md §12.6).
 *
 * Build and run in the environment's stack (HIP.md §3), or one command for all of it
 * (fixes/hip1/repro_rocblas.sh):
 *     conda activate axl_rocm_7_14
 *     hipcc --offload-arch=gfx1201 -O2 fixes/hip1/launch_fault_via_rocblas.hip \
 *         -o /tmp/launch_fault_via_rocblas -lrocblas
 *     /tmp/launch_fault_via_rocblas
 *
 * It faults, and it did on 2026-09-16: at the first byte of B's guard page, B being
 * the 29,568-byte [48,308] operand.  The 128-wide N tile covers only 48 of its
 * `n = 48` rows, so the edge tile reads 79 bf16 values (158 bytes) past the operand —
 * less than one page, hence a fault at the guard page's own address rather than the
 * page after it.  Transcript and reading: hip1.md.
 *
 * An earlier revision of this file also took fp16 and f32 operands, and an fp32 call
 * completes cleanly where the two 16-bit ones fault.  Those variants have been withdrawn
 * again so the program stays single-purpose; the record is hip1.md §12.12.
 *
 * The rocBLAS API is declared below instead of included: no ROCm wheel ships rocBLAS
 * headers, the only copy on this machine is /opt/rocm's (part of the HIP 7.15 stack),
 * and adding `-I /opt/rocm/core/include` would shadow the environment's HIP 7.14
 * headers with 7.15 ones — the pitfall HIP.md §6 lays out.  Values taken from
 * /opt/rocm/core/include/rocblas/internal/rocblas-types.h.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

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

// The failing dispatch as hipBLASLt saw it: the row-major 48x1280x308 matmul of
// hip1.md §12.1 with its operands exchanged, so m is the tensor's last dimension and n
// is network_dim.  k is the batch-times-sequence 308 unchanged.
#define M 1280
#define N 48
#define K 308

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
// every operand size is, which keeps the 16-byte vector loads the kernels use aligned.
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
    printf("device            : %s (%s, %d CUs)\n", prop.name, prop.gcnArchName,
           prop.multiProcessorCount);

    // Column-major, each the byte size its operand has in the failing dispatch: A is
    // m x k with lda = m, B is n x k because transB is transpose (ldb = n), D is m x n
    // (ldd = m).  C and D are the same buffer, as in a beta = 0 GEMM they may be.
    void* a = guarded("A", (size_t)M * K * 2);
    void* b = guarded("B", (size_t)N * K * 2);
    void* d = guarded("D", (size_t)M * N * 2);

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("problem           : A[%d,%d] x B[%d,%d] -> D[%d,%d], bf16, transA none, transB transpose,"
           " lda %d, ldb %d, ldd %d\n",
           M, K, N, K, M, N, M, N, M);
    fflush(stdout);

    rocblas_handle handle;
    if (rocblas_create_handle(&handle) != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle failed\n");
        return 1;
    }
    rocblas_status st = rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_transpose, M, N, K, &alpha, a,
        rocblas_datatype_bf16_r, M, b, rocblas_datatype_bf16_r, N, &beta, d,
        rocblas_datatype_bf16_r, M, d, rocblas_datatype_bf16_r, M, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    if (st != rocblas_status_success) {
        fprintf(stderr, "rocblas_gemm_ex -> %d\n", (int)st);
        return 1;
    }

    // The fault lands here; ROCr prints its own line, naming the kernel, and aborts.
    hipError_t err = hipDeviceSynchronize();
    printf("result            : %s\n", err == hipSuccess ? "completed without a fault"
                                                          : hipGetErrorString(err));
    rocblas_destroy_handle(handle);
    return err == hipSuccess ? 0 : 1;
}

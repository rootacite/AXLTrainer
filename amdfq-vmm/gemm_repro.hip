/*
 * The third step: run the GEMM the VMM runs died in, from our own program.
 *
 * The two VMM runs of 2026-09-17 (amdfq.md §10.4) both aborted inside a hipBLASLt Tensile bf16
 * kernel, and run B named it: an `MT128x128x16 … _WG16_8_1_WGMXCC1` solution of the `BB_BB`,
 * `l_Ailk_Bljk` library that ships with this ROCm. The TensorBoard-free way to get *that* kernel
 * onto our memory is to ask hipBLASLt for the same GEMM the trainer asked for and let its own
 * solution selection pick: the log of the dispatch (amdfq_hooks_launch.c) then says whether the
 * kernel is the one that died, which is what `AMDFQ_LAUNCH_LOG=1` under amdfq-vmm/hook.sh shows.
 *
 * The shapes come from the dispatch record in the traced VMM run — M 960, N 320, K 60, batch 2,
 * `d` and `c` the same pointer — and the layouts follow the library's name: `Ailk_Bljk` is
 * A[M,K] row-major times B[N,K] row-major, the `linear` layout a LoRA projection has (weight is
 * [out, in]).  hipBLASLt describes matrices column-major, so the row-major problem is posed as its
 * transpose with the operands swapped (`--config 1`), and `--config 2` poses it the other way to
 * check whether the two descriptions select the same kernel.
 *
 * Whichever config runs, the operands are ordinary hipMalloc allocations: under amdfq-vmm/hook.sh those
 * are the VMM blocks, which is the whole point — the same kernel, the same shapes, our memory.
 *
 *   bash amdfq-vmm/gemm_repro.sh [M] [N] [K] [batch] [--config 1|2] [--reps N] [--algo N] [--no-check]
 */
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define MiB (1024ULL * 1024ULL)

#define HIPCHECK(call)                                                                       \
    do {                                                                                     \
        hipError_t e_ = (call);                                                              \
        if (e_ != hipSuccess) {                                                              \
            fprintf(stderr, "%s:%d  %s -> %s\n", __FILE__, __LINE__, #call,                   \
                    hipGetErrorString(e_));                                                  \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

#define LTCHECK(call)                                                                        \
    do {                                                                                     \
        hipblasStatus_t s_ = (call);                                                          \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                                   \
            fprintf(stderr, "%s:%d  %s -> status %d\n", __FILE__, __LINE__, #call, (int)s_);  \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

/* bf16 through its bit pattern, so the reference and the device see the same values. */
static float bf16_to_float(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof value);
    return value;
}

static uint16_t float_to_bf16(float value) {
    uint32_t wide;
    memcpy(&wide, &value, sizeof wide);
    return (uint16_t)((wide + 0x8000u) >> 16);
}

/* A deterministic spread in [-1, 1): any pattern works, it only has to be the same on both sides. */
static uint16_t pattern(unsigned long long index) {
    unsigned long long x = index * 6364136223846793005ULL + 1442695040888963407ULL;
    float value = (float)((x >> 40) & 0xffff) / 32768.0f - 1.0f;
    return float_to_bf16(value);
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void set_batch(hipblasLtMatrixLayout_t layout, int batch, long long stride) {
    if (batch <= 1) return;
    int32_t count = batch;
    int64_t space = stride;
    LTCHECK(hipblasLtMatrixLayoutSetAttribute(layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count,
                                              sizeof count));
    LTCHECK(hipblasLtMatrixLayoutSetAttribute(layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &space, sizeof space));
}

int main(int argc, char **argv) {
    int M = 960, N = 320, K = 60, batch = 2;
    int config = 1, reps = 1, algo_index = 0;
    double workspace_hint_mib = 25.0; /* the trainer's hipBLASLt workspace, from the VMM run's log */
    bool check = true, in_place = true;
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) config = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--algo" && i + 1 < argc) algo_index = atoi(argv[++i]);
        else if (a == "--workspace" && i + 1 < argc) workspace_hint_mib = atof(argv[++i]);
        else if (a == "--no-check") check = false;
        else if (a == "--separate-d") in_place = false;
        else if (a.rfind("--", 0) == 0) { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
        else {
            switch (positional++) {
                case 0: M = atoi(argv[i]); break;
                case 1: N = atoi(argv[i]); break;
                case 2: K = atoi(argv[i]); break;
                case 3: batch = atoi(argv[i]); break;
                default: fprintf(stderr, "too many positional arguments\n"); return 2;
            }
        }
    }
    if (config != 1 && config != 2) { fprintf(stderr, "--config is 1 or 2\n"); return 2; }
    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0 || reps <= 0) {
        fprintf(stderr, "shapes and reps must be positive\n");
        return 2;
    }

    int device = 0;
    HIPCHECK(hipGetDevice(&device));
    hipDeviceProp_t props{};
    HIPCHECK(hipGetDeviceProperties(&props, device));

    const size_t a_elems = (size_t)M * K * batch;
    const size_t b_elems = (size_t)N * K * batch;
    const size_t c_elems = (size_t)M * N * batch;
    printf("device      : %s (%s)\n", props.name, props.gcnArchName);
    printf("problem     : C[%d,%d] = A[%d,%d] x B[%d,%d]%s, bf16, batch %d, %s\n", M, N, M, K,
           config == 1 ? N : K, config == 1 ? K : N, config == 1 ? "^T" : "", batch,
           in_place ? "C and D the same buffer" : "C and D separate");

    std::vector<uint16_t> A_h(a_elems), B_h(b_elems);
    for (size_t i = 0; i < a_elems; i++) A_h[i] = pattern(i);
    for (size_t i = 0; i < b_elems; i++) B_h[i] = pattern(i + a_elems);

    void *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
    HIPCHECK(hipMalloc(&A, a_elems * 2));
    HIPCHECK(hipMalloc(&B, b_elems * 2));
    HIPCHECK(hipMalloc(&C, c_elems * 2));
    HIPCHECK(hipMemset(C, 0, c_elems * 2));
    if (in_place) {
        D = C;
    } else {
        HIPCHECK(hipMalloc(&D, c_elems * 2));
        HIPCHECK(hipMemset(D, 0, c_elems * 2));
    }
    HIPCHECK(hipMemcpy(A, A_h.data(), a_elems * 2, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(B, B_h.data(), b_elems * 2, hipMemcpyHostToDevice));

    hipblasLtHandle_t handle = nullptr;
    LTCHECK(hipblasLtCreate(&handle));
    hipblasLtMatmulDesc_t desc = nullptr;
    LTCHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    int32_t opA = config == 1 ? HIPBLAS_OP_T : HIPBLAS_OP_N; /* op(A) is the [N,K] operand either way */
    int32_t opB = HIPBLAS_OP_N;                              /* op(B) is A[M,K] read as (K,M) */
    LTCHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof opA));
    LTCHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof opB));

    /* Layout descriptors describe the *stored* matrix, and hipBLASLt stores column-major: with
     * opA/opB = T the shape is (k, m) / (n, k), with N it is (m, k) / (k, n).  The leading
     * dimensions are the physical row lengths of the row-major buffers, which is what makes the
     * transposed description exact. */
    hipblasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    if (config == 1) {
        LTCHECK(hipblasLtMatrixLayoutCreate(&Ad, HIP_R_16BF, K, N, K)); /* holds B[N,K] row-major */
        LTCHECK(hipblasLtMatrixLayoutCreate(&Bd, HIP_R_16BF, K, M, K)); /* holds A[M,K] row-major */
    } else {
        LTCHECK(hipblasLtMatrixLayoutCreate(&Ad, HIP_R_16BF, N, K, N)); /* holds B[K,N] row-major */
        LTCHECK(hipblasLtMatrixLayoutCreate(&Bd, HIP_R_16BF, K, M, K)); /* holds A[M,K] row-major */
    }
    LTCHECK(hipblasLtMatrixLayoutCreate(&Cd, HIP_R_16BF, N, M, N));
    LTCHECK(hipblasLtMatrixLayoutCreate(&Dd, HIP_R_16BF, N, M, N));
    const long long stride_a = config == 1 ? (long long)K * N : (long long)N * K;
    const long long stride_b = config == 1 ? (long long)K * M : (long long)M * K;
    const long long stride_c = (long long)N * M;
    set_batch(Ad, batch, stride_a);
    set_batch(Bd, batch, stride_b);
    set_batch(Cd, batch, stride_c);
    set_batch(Dd, batch, stride_c);

    hipblasLtMatmulPreference_t pref = nullptr;
    LTCHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t workspace_hint = (size_t)(workspace_hint_mib * (double)MiB);
    LTCHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_hint, sizeof workspace_hint));

    enum { MAX_ALGOS = 16 };
    hipblasLtMatmulHeuristicResult_t results[MAX_ALGOS];
    int found = 0;
    LTCHECK(hipblasLtMatmulAlgoGetHeuristic(handle, desc, Ad, Bd, Cd, Dd, pref, MAX_ALGOS, results,
                                            &found));
    printf("workspace   : hint %zu MiB, heuristic returned %d algo(s)\n", (size_t)workspace_hint_mib,
           found);
    if (found == 0) { fprintf(stderr, "no algo for this problem\n"); return 1; }
    for (int i = 0; i < found; i++)
        printf("  algo %d     : workspace %zu B, waves %.1f, state %d\n", i, results[i].workspaceSize,
               (double)results[i].wavesCount, (int)results[i].state);
    if (algo_index >= found) { fprintf(stderr, "--algo %d of %d\n", algo_index, found); return 2; }
    const hipblasLtMatmulAlgo_t *algo = &results[algo_index].algo;

    void *workspace = nullptr;
    const size_t workspace_size = results[algo_index].workspaceSize;
    if (workspace_size > 0) HIPCHECK(hipMalloc(&workspace, workspace_size));

    float alpha = 1.0f, beta = in_place ? 1.0f : 0.0f;
    const int m = N, n = M, k = K; /* the column-major view of the row-major problem */
    double started = now_seconds();
    for (int rep = 0; rep < reps; rep++)
        LTCHECK(hipblasLtMatmul(handle, desc, &alpha, B, Ad, A, Bd, &beta, C, Cd, D, Dd, algo,
                                workspace, workspace_size, nullptr));
    HIPCHECK(hipDeviceSynchronize());
    double elapsed = now_seconds() - started;
    printf("matmul      : m=%d n=%d k=%d batch=%d, %d call(s) in %.3f s (%.1f ms each)\n", m, n, k,
           batch, reps, elapsed, elapsed * 1e3 / reps);

    int failures = 0;
    std::vector<uint16_t> D_h(c_elems);
    HIPCHECK(hipMemcpy(D_h.data(), D, c_elems * 2, hipMemcpyDeviceToHost));
    if (check) {
        double worst = 0.0;
        for (int b = 0; b < batch; b++) {
            const uint16_t *a = A_h.data() + (size_t)b * M * K;
            const uint16_t *bm = B_h.data() + (size_t)b * N * K;
            const uint16_t *d = D_h.data() + (size_t)b * M * N;
            for (int row = 0; row < M; row++) {
                for (int col = 0; col < N; col++) {
                    double want = 0.0;
                    for (int kk = 0; kk < K; kk++) {
                        double av = bf16_to_float(a[(size_t)row * K + kk]);
                        double bv = config == 1 ? bf16_to_float(bm[(size_t)col * K + kk])
                                                : bf16_to_float(bm[(size_t)kk * N + col]);
                        want += av * bv;
                    }
                    double got = bf16_to_float(d[(size_t)row * N + col]);
                    double diff = got - want;
                    if (diff < 0) diff = -diff;
                    double relative = diff / (want < 0 ? -want : want) ;
                    if (relative > 0.05 && diff > 0.05) failures++;
                    if (diff > worst) worst = diff;
                }
            }
        }
        printf("check       : %s (worst absolute error %.4f, %d of %zu elements outside 5%%)\n",
               failures == 0 ? "C = A x B holds" : "MISMATCH", worst, failures, c_elems);
        if (failures != 0) {
            /* Which cell of D holds which sum: enough to read the layout off the numbers when the
             * description and the library disagree. */
            const uint16_t *a = A_h.data(), *bm = B_h.data(), *d = D_h.data();
            double ref00 = 0.0, ref01 = 0.0, ref10 = 0.0;
            for (int kk = 0; kk < K; kk++) {
                double a0 = bf16_to_float(a[(size_t)0 * K + kk]);
                double a1 = bf16_to_float(a[(size_t)1 * K + kk]);
                double b0 = config == 1 ? bf16_to_float(bm[(size_t)0 * K + kk])
                                        : bf16_to_float(bm[(size_t)kk * N + 0]);
                double b1 = config == 1 ? bf16_to_float(bm[(size_t)1 * K + kk])
                                        : bf16_to_float(bm[(size_t)kk * N + 1]);
                ref00 += a0 * b0;
                ref01 += a0 * b1;
                ref10 += a1 * b0;
            }
            printf("first cells : D[0,0]=%.4f want %.4f | D[0,1]=%.4f want %.4f | D[1,0]=%.4f want"
                   " %.4f\n", bf16_to_float(d[0]), ref00, bf16_to_float(d[1]), ref01,
                   bf16_to_float(d[N]), ref10);
        }
    }

    if (workspace != nullptr) HIPCHECK(hipFree(workspace));
    if (!in_place && D != nullptr) HIPCHECK(hipFree(D));
    LTCHECK(hipblasLtMatrixLayoutDestroy(Ad));
    LTCHECK(hipblasLtMatrixLayoutDestroy(Bd));
    LTCHECK(hipblasLtMatrixLayoutDestroy(Cd));
    LTCHECK(hipblasLtMatrixLayoutDestroy(Dd));
    LTCHECK(hipblasLtMatmulDescDestroy(desc));
    LTCHECK(hipblasLtMatmulPreferenceDestroy(pref));
    LTCHECK(hipblasLtDestroy(handle));
    HIPCHECK(hipFree(A));
    HIPCHECK(hipFree(B));
    HIPCHECK(hipFree(C));
    printf("result      : completed, %s\n", failures == 0 ? "nothing threw" : "but the numbers differ");
    return 0;
}

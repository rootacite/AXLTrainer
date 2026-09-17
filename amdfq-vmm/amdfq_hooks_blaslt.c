/*
 * The hipBLASLt problem log: what the *library* was asked for, next to the dispatch it causes in
 * amdfq_hooks_launch.c's log. The two lines have to be read together, because which Tensile kernel a
 * GEMM gets is decided by exactly these fields — transposes, the four stored shapes, the leading
 * dimensions, the batch stride, the epilogue and the workspace — and the kernel that died in the
 * VMM runs (amdfq.md §10.4) only comes back if the same problem is posed again.
 *
 * Off unless AMDFQ_LAUNCH_LOG=1, like the dispatch log it pairs with: one line per matmul.
 */
#define _GNU_SOURCE

#include "amdfq_log.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int hipblasStatus_t;
typedef void *hipblasLtHandle_t;
typedef void *hipblasLtMatmulDesc_t;
typedef void *hipblasLtMatrixLayout_t;
typedef void *hipblasLtMatmulAlgo_t;

hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
                                const void *alpha, const void *A, hipblasLtMatrixLayout_t A_desc,
                                const void *B, hipblasLtMatrixLayout_t B_desc, const void *beta,
                                const void *C, hipblasLtMatrixLayout_t C_desc, void *D,
                                hipblasLtMatrixLayout_t D_desc, const hipblasLtMatmulAlgo_t *algo,
                                void *workspace, size_t workspace_size, void *stream);

/* hipblasLtMatrixLayoutAttribute_t and hipblasLtMatmulDescAttributes_t, as hipblaslt.h numbers them. */
enum {
    AMDFQ_LT_BATCH_COUNT = 0,
    AMDFQ_LT_STRIDED_BATCH_OFFSET = 1,
    AMDFQ_LT_TYPE = 2,
    AMDFQ_LT_ROWS = 4,
    AMDFQ_LT_COLS = 5,
    AMDFQ_LT_LD = 6,
};
enum {
    AMDFQ_LT_TRANSA = 0,
    AMDFQ_LT_TRANSB = 1,
    AMDFQ_LT_EPILOGUE = 2,
    AMDFQ_LT_BIAS_POINTER = 3,
};

hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(hipblasLtMatrixLayout_t layout, int attr, void *buf,
                                                  size_t size, size_t *written);
hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t desc, int attr, void *buf,
                                                size_t size, size_t *written);

static int enabled(void);

static void layout_of(const char *label, hipblasLtMatrixLayout_t layout, char *out, size_t out_size) {
    if (layout == NULL) {
        snprintf(out, out_size, "%s=null", label);
        return;
    }
    uint64_t rows = 0, cols = 0, ld = 0;
    int type = -1;
    int32_t batch = 0;
    int64_t stride = 0;
    size_t written = 0;
    hipblasLtMatrixLayoutGetAttribute(layout, AMDFQ_LT_ROWS, &rows, sizeof rows, &written);
    hipblasLtMatrixLayoutGetAttribute(layout, AMDFQ_LT_COLS, &cols, sizeof cols, &written);
    hipblasLtMatrixLayoutGetAttribute(layout, AMDFQ_LT_LD, &ld, sizeof ld, &written);
    hipblasLtMatrixLayoutGetAttribute(layout, AMDFQ_LT_TYPE, &type, sizeof type, &written);
    hipblasLtMatrixLayoutGetAttribute(layout, AMDFQ_LT_BATCH_COUNT, &batch, sizeof batch, &written);
    hipblasLtMatrixLayoutGetAttribute(layout, AMDFQ_LT_STRIDED_BATCH_OFFSET, &stride, sizeof stride,
                                      &written);
    snprintf(out, out_size, "%s=[type=%d rows=%llu cols=%llu ld=%llu batch=%d stride=%lld]", label,
             type, (unsigned long long)rows, (unsigned long long)cols, (unsigned long long)ld,
             (int)batch, (long long)stride);
}

static int enabled(void) {
    static int state = -1;
    if (state < 0) {
        const char *env = getenv("AMDFQ_LAUNCH_LOG");
        state = env != NULL && env[0] == '1';
    }
    return state;
}

hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
                                const void *alpha, const void *A, hipblasLtMatrixLayout_t A_desc,
                                const void *B, hipblasLtMatrixLayout_t B_desc, const void *beta,
                                const void *C, hipblasLtMatrixLayout_t C_desc, void *D,
                                hipblasLtMatrixLayout_t D_desc, const hipblasLtMatmulAlgo_t *algo,
                                void *workspace, size_t workspace_size, void *stream) {
    AMDFQ_REAL(hipblasLtMatmul);
    if (real == NULL) return 1; /* HIPBLAS_STATUS_NOT_INITIALIZED */
    if (enabled()) {
        char a[200], b[200], c[200], d[200];
        layout_of("A", A_desc, a, sizeof a);
        layout_of("B", B_desc, b, sizeof b);
        layout_of("C", C_desc, c, sizeof c);
        layout_of("D", D_desc, d, sizeof d);
        int32_t transA = -1, transB = -1;
        uint32_t epilogue = 0;
        void *bias = NULL;
        size_t written = 0;
        if (desc != NULL) {
            hipblasLtMatmulDescGetAttribute(desc, AMDFQ_LT_TRANSA, &transA, sizeof transA, &written);
            hipblasLtMatmulDescGetAttribute(desc, AMDFQ_LT_TRANSB, &transB, sizeof transB, &written);
            hipblasLtMatmulDescGetAttribute(desc, AMDFQ_LT_EPILOGUE, &epilogue, sizeof epilogue, &written);
            hipblasLtMatmulDescGetAttribute(desc, AMDFQ_LT_BIAS_POINTER, &bias, sizeof bias, &written);
        }
        /* alpha/beta are whatever the scale type says; the trainer and this hook only ever see 32F. */
        float alpha_f = alpha != NULL ? *(const float *)alpha : 0.0f;
        float beta_f = beta != NULL ? *(const float *)beta : 0.0f;
        amdfq_logf("blaslt matmul transA=%d transB=%d epilogue=%u alpha=%g beta=%g bias=%p ws=%zu"
                   " A=%p B=%p C=%p D=%p %s %s %s %s caller=%s",
                   (int)transA, (int)transB, (unsigned)epilogue, (double)alpha_f, (double)beta_f,
                   bias, workspace_size, A, B, C, D, a, b, c, d, AMDFQ_CALLER());
    }
    return real(handle, desc, alpha, A, A_desc, B, B_desc, beta, C, C_desc, D, D_desc, algo, workspace,
                workspace_size, stream);
}

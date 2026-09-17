// hip_smoke.hip — toolchain smoke test: device properties, HIP versions, one kernel.
//
//   conda activate axl
//   hipcc -O3 tools/hip/hip_smoke.hip -o /tmp/hip_smoke && /tmp/hip_smoke
//
// This is what tools/hip/setup_rocm_dev.sh --verify builds. It exits non-zero when
// a HIP call fails or the kernel result is wrong, so it can gate a toolchain.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                   \
    do {                                                                  \
        hipError_t _e = (expr);                                           \
        if (_e != hipSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, \
                         #expr, hipGetErrorString(_e));                   \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

__global__ void scale(float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

int main() {
    const int n = 1024;

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    int runtime = 0;
    HIP_CHECK(hipRuntimeGetVersion(&runtime));

    std::printf("device        : %s\n", prop.name);
    std::printf("arch          : %s (%d.%d, %d CUs, warp %d)\n", prop.gcnArchName,
                prop.major, prop.minor, prop.multiProcessorCount, prop.warpSize);
    std::printf("HIP header    : %d.%d.%d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR,
                HIP_VERSION_PATCH);
    std::printf("HIP runtime   : %d.%d.%d\n", runtime / 10000000,
                (runtime / 100000) % 100, runtime % 100000);

    float* dev = nullptr;
    HIP_CHECK(hipMalloc(&dev, n * sizeof(float)));

    float host[n];
    for (int i = 0; i < n; i++) host[i] = 1.0f;
    HIP_CHECK(hipMemcpy(dev, host, sizeof(host), hipMemcpyHostToDevice));

    scale<<<(n + 255) / 256, 256>>>(dev, 3.0f, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dev));

    int wrong = 0;
    for (int i = 0; i < n; i++) {
        if (host[i] != 3.0f) wrong++;
    }
    std::printf("kernel        : %d/%d wrong\n", wrong, n);
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

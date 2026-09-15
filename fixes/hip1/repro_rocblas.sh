#!/usr/bin/env bash
# repro_rocblas.sh — route 2 to the fixes/fix3 fault (hip1.md §12.10).
#
#   bash fixes/hip1/repro_rocblas.sh
#
# Compiles launch_fault_via_rocblas.hip and runs it: one rocblas_gemm_ex carrying the
# trainer's tensors, with the same VMM guard-page placement as route 1, so hipBLASLt
# does the solution selection and the edge-tile overrun still has an unmapped page to
# run into.  Expect the same ROCr fault line as route 1, with the same kernel name.
#
# No code object to extract: the library loads its own container.
#
# Activation is the caller's job (HIP.md §3) and no environment is pinned: the fault belongs
# to whichever stack runs the call, so any hipcc on PATH is accepted and the resolved
# toolchain and libraries are printed instead.  Build a comparison out of those lines.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
BIN="/tmp/launch_fault_via_rocblas"

die() {
    echo "repro_rocblas.sh: $*" >&2
    exit 1
}

# rocBLAS is not beside libamdhip64 in every layout: the ROCm wheels keep it in
# `_rocm_sdk_libraries/lib`, a system install in `<root>/lib`.  An activated environment puts
# its library directories on LD_LIBRARY_PATH — the same list the loader will use at run time —
# so that is searched first, then the directories HIP_PATH/ROCM_PATH name, then the system
# default.  No rocBLAS header is involved: the program declares the three entry points it
# uses, because the only copy of the headers on this machine is /opt/rocm's, and putting that
# on the include path would shadow the environment's HIP headers with the other stack's
# (HIP.md §6).
find_rocblas_dir() {
    local dir dirs="${LD_LIBRARY_PATH:-}"
    dirs="${dirs//:/ }"
    for dir in $dirs ${HIP_PATH:+$HIP_PATH/lib} ${ROCM_PATH:+$ROCM_PATH/lib} \
        /opt/rocm/lib /opt/rocm/core/lib; do
        if [ -f "$dir/librocblas.so" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

report_toolchain() {
    command -v hipcc >/dev/null 2>&1 || die "hipcc not on PATH
    Activate a ROCm environment (HIP.md §3) or put the system hipcc on PATH."
    echo "hipcc     : $(command -v hipcc)"
    echo "HIP_PATH  : ${HIP_PATH:-unset}"
    echo "ROCM_PATH : ${ROCM_PATH:-unset}"
}

report_toolchain
cd "$REPO_ROOT"

LIBS="$(find_rocblas_dir)" || die "no librocblas.so in LD_LIBRARY_PATH, \$HIP_PATH/lib,
    \$ROCM_PATH/lib, /opt/rocm/lib or /opt/rocm/core/lib.
    In the ROCm wheel layout, bash tools/hip/setup_rocm_dev.sh creates the unversioned
    links hipcc needs to link against it (HIP.md §2)."
echo "rocBLAS   : $LIBS"

echo
echo "==> compiling $HERE/launch_fault_via_rocblas.hip"
hipcc --offload-arch=gfx1201 -O2 "$HERE/launch_fault_via_rocblas.hip" -o "$BIN" -L"$LIBS" -lrocblas

echo
# What this run will actually load, which need not be what HIP_PATH says (HIP.md §3).
echo "==> HIP and rocBLAS this binary loads"
ldd "$BIN" | grep -E 'libamdhip64|librocblas' || echo "    none — ldd resolved neither"

echo
echo "==> running $BIN"
status=0
"$BIN" || status=$?

echo
if [ "$status" -eq 0 ]; then
    echo "$BIN completed without faulting, so no overrun reached a guard page and this run"
    echo "did not reproduce (hip1.md §12.10)."
    exit 1
fi
echo "$BIN exited with $status — the fault killing the process, which is the expected"
echo "outcome (hip1.md §12.10)."

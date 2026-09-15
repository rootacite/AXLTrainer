#!/usr/bin/env bash
# repro_standalone.sh — route 1 to the fixes/fix3 fault (hip1.md §12.9).
#
#   bash fixes/hip1/repro_standalone.sh
#
# Extracts the hipModule-loadable code object out of the CCOB container that sits next
# to this script, compiles launch_fault_kernel.hip against it, and runs it: the program
# loads that one object, places A/B/C/D with VMM guard pages, and fires the 48x1280x308
# bf16 GEMM on the faulting kernel.  Expect ROCr's "Memory access fault ... Reason: Page
# not present or supervisor privilege", a non-zero exit, and the kernel named on the
# fault line.
#
# Activation is the caller's job (HIP.md §3) and no environment is pinned: the fault belongs
# to whichever stack runs the object, so any hipcc on PATH is accepted and the resolved
# toolchain and runtime library are printed instead.  Build a comparison out of those lines.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OBJECT_ELF="/tmp/axl-hip1/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co.codeobject.elf"
BIN="/tmp/launch_fault_kernel"

die() {
    echo "repro_standalone.sh: $*" >&2
    exit 1
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

# The launcher reads the object from a fixed path, so extract it there.
echo "==> extracting the code object (fixes/hip1/hip1.md §12.3)"
python3 "$HERE/extract_kernel_codeobject.py"
[ -f "$OBJECT_ELF" ] || die "the extractor did not write $OBJECT_ELF"

echo
echo "==> compiling $HERE/launch_fault_kernel.hip"
hipcc --offload-arch=gfx1201 -O2 "$HERE/launch_fault_kernel.hip" -o "$BIN"

echo
# What this run will actually load, which need not be what HIP_PATH says (HIP.md §3).
echo "==> HIP runtime this binary loads"
ldd "$BIN" | grep libamdhip64 || echo "    none — ldd resolved no libamdhip64"

echo
echo "==> running $BIN"
status=0
"$BIN" || status=$?

echo
if [ "$status" -eq 0 ]; then
    echo "$BIN completed without faulting, so no overrun reached a guard page and this run"
    echo "did not reproduce (hip1.md §12.9)."
    exit 1
fi
echo "$BIN exited with $status — the fault killing the process, which is the expected"
echo "outcome (hip1.md §12.9)."

#!/usr/bin/env bash
#
# Builds and runs amdfq-vmm/gemm_repro.hip: the bf16 GEMM the VMM runs died in, dispatched through
# hipBLASLt so the library's own solution selection picks the kernel.
#
# Plain, the operands are ordinary hipMalloc allocations. Under amdfq-vmm/hook.sh they are the VMM
# blocks, and the dispatch log names the kernel:
#
#   bash amdfq-vmm/gemm_repro.sh                                    # the shape the dispatch record had
#   AMDFQ_LAUNCH_LOG=1 bash amdfq-vmm/hook.sh bash amdfq-vmm/gemm_repro.sh               # on VMM memory
#
# Extra arguments go to the program (see its header): --config, --reps, --algo, --no-check.
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=/tmp/amdfq-gemm
py=${AXL_PYTHON:-python3}
rocm=$("$py" -c 'import os, _rocm_sdk_core; print(os.path.dirname(_rocm_sdk_core.__file__))' 2>/dev/null)
libs=$("$py" -c 'import os, _rocm_sdk_libraries; print(os.path.dirname(_rocm_sdk_libraries.__file__))' 2>/dev/null)
if [[ -z $rocm || ! -x $rocm/bin/hipcc ]]; then
    echo "amdfq: no hipcc found via $py ($rocm) — run this in the env environment.yml names, or set AXL_PYTHON" >&2
    exit 2
fi

mkdir -p "$work/lib"
# Link through a plain name: hipcc passes `-l:` oddly, and the SONAME is what the loader resolves.
ln -sf "$libs/lib/libhipblaslt.so.1" "$work/lib/libhipblaslt.so"
"$rocm/bin/hipcc" --offload-arch=gfx1201 -O2 -w -I/opt/rocm/core/include "$here/gemm_repro.hip" \
    -o "$work/gemm_repro" -L"$work/lib" -lhipblaslt || exit 2

LD_LIBRARY_PATH="$work/lib:$libs/lib:$rocm/lib" exec "$work/gemm_repro" "$@"

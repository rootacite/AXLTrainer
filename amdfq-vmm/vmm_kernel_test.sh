#!/usr/bin/env bash
#
# Builds and runs amdfq-vmm/vmm_kernel_test.hip: the hook's VMM allocation, then trivial kernels on it,
# next to the same kernels on a hipMalloc block.
#
#   bash amdfq-vmm/vmm_kernel_test.sh [blocks] [MiB per block]
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=/tmp/amdfq-vmm
py=${AXL_PYTHON:-python3}
rocm=$("$py" -c 'import os, _rocm_sdk_core; print(os.path.dirname(_rocm_sdk_core.__file__))' 2>/dev/null)
if [[ -z $rocm || ! -x $rocm/bin/hipcc ]]; then
    echo "amdfq: no hipcc found via $py ($rocm) — run this in the env environment.yml names, or set AXL_PYTHON" >&2
    exit 2
fi

mkdir -p "$work"
"$rocm/bin/hipcc" --offload-arch=gfx1201 -O2 -Wall "$here/vmm_kernel_test.hip" -o "$work/vmm_kernel_test" || exit 2
"$work/vmm_kernel_test" "$@"

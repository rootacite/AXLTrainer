#!/usr/bin/env bash
#
# Builds and runs amdfq-vmm/vmm_diff.c: hipMalloc's block against the VMM sequence the hook uses, as the
# runtime describes them (pointer attributes, address range, granted access, host mapping).
#
#   bash amdfq-vmm/vmm_diff.sh
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=/tmp/amdfq-vmm
py=${AXL_PYTHON:-python3}
lib=$("$py" -c 'import os, _rocm_sdk_core; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null)
if [[ -z $lib || ! -d $lib ]]; then
    echo "amdfq: no ROCm runtime found via $py — run this in the env environment.yml names, or set AXL_PYTHON" >&2
    exit 2
fi

mkdir -p "$work"
cc -O2 -Wall -Wextra -o "$work/vmm_diff" "$here/vmm_diff.c" \
    -L"$lib" -l:libamdhip64.so.7 -Wl,-rpath,"$lib" || exit 2
"$work/vmm_diff"

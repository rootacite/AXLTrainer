#!/usr/bin/env bash
#
# Builds and runs the VMM go/no-go probe (amdfq.md §10.3, route 1) against the ROCm runtime of the
# active python environment. The modes that can fault the GPU (`access`, `overread`) get their own
# invocation, so a dead one costs a line of report instead of the whole probe.
#
#   bash amdfq-vmm/vmm_probe.sh
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
echo "amdfq: rocm lib $lib" >&2
cc -O2 -Wall -Wextra -o "$work/vmm_probe" "$here/vmm_probe.c" \
    -L"$lib" -l:libamdhip64.so.7 -Wl,-rpath,"$lib" || exit 2

rc=0
modes=(info multimap cost "hostaccess hipmalloc" "hostaccess host")
# `hostaccess device` segfaults this process and `access` / `overread` fault the GPU, which resets the
# ring and takes the display with it on this box — so they are opt-in, not part of a normal run.
if [[ ${AMDFQ_PROBE_ALLOW_FAULTS:-0} = 1 ]]; then
    modes+=("hostaccess device" access overread)
else
    echo "amdfq: skipping the faulting modes (hostaccess device, access, overread);" \
         "AMDFQ_PROBE_ALLOW_FAULTS=1 runs them" >&2
fi
for mode in "${modes[@]}"; do
    printf '\n===== %s\n' "$mode"
    # shellcheck disable=SC2086
    "$work/vmm_probe" $mode
    status=$?
    if [[ $status -ne 0 ]]; then
        if [[ $status -gt 128 ]]; then
            echo "probe: mode $mode died on signal $((status - 128))"
        else
            echo "probe: mode $mode exited $status"
        fi
        rc=1
    fi
done
exit $rc

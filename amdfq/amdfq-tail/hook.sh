#!/usr/bin/env bash
#
# Loads the hook into a command and prints the log it produced (amdfq/doc/amdfq.md §9).
#
#   bash amdfq/amdfq-tail/hook.sh                       # default: one torch call in the env's interpreter
#   bash amdfq/amdfq-tail/hook.sh bash start_train.sh   # a real run
#   bash amdfq/amdfq-tail/hook.sh ./some/hip/program    # anything that calls hipMalloc
#
# The shared object is built on the first call if it is missing. The log is written by the hook
# itself, so a command that dies still leaves one; for a long run, `tail -f` it from another shell.
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
so=$here/cmake-build-debug/libamdfq.so
log=${AMDFQ_LOG:-/tmp/amdfq-hook.$$.log}

if [[ ! -f $so ]]; then
    echo "amdfq: building $so" >&2
    cmake -S "$here" -B "$here/cmake-build-debug" >/dev/null || exit 2
    cmake --build "$here/cmake-build-debug" >/dev/null || exit 2
fi

if [[ $# -eq 0 ]]; then
    set -- "${AXL_PYTHON:-python3}" -c 'import torch; torch.zeros(8, device="cuda"); torch.cuda.synchronize()'
fi

echo "amdfq: LD_PRELOAD=$so AMDFQ_LOG=$log" >&2
rc=0
AMDFQ_LOG=$log LD_PRELOAD=$so "$@" || rc=$?
[[ $rc -ne 0 ]] && echo "amdfq: the command exited $rc" >&2

printf '\n== %s\n' "$log"
if [[ -f $log ]]; then cat -- "$log"; else echo "amdfq: no log was written"; fi
exit $rc

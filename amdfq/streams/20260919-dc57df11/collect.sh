#!/usr/bin/env bash
#
# Collects one pad-0 allocation stream from a real training run under the C hook, and lets the run
# die of its own fault: AMDFQ_TAIL=0 turns the tail guard off, which is the one configuration in
# which the gfx1201 over-read still aborts the trainer (see amdfq/doc/amdfq.md §11/§12). The window
# therefore ends where the process does, not at a freeze point.
#
#   bash amdfq/streams/<tag>/collect.sh
#
# Writes into this directory:
#   frozen.log   the hook's own log (it writes it itself, line by line, unbuffered)
#   train.log    the run's stderr
#   sample.log   4 Hz vis/gtt over the whole window (sample-host.sh)
#   crash.txt    exit code, state.json, dmesg tail, wall clock, .so sha256, exact command
#
# Preconditions, all checked below: the hook is built, the interpreter exists, and no other process
# holds renderD128 or /dev/kfd (one GPU client at a time is the rule every amdfq measurement used).
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)   # amdfq/streams/<tag>/ -> repo root
so=$repo/amdfq/amdfq-tail/cmake-build-release/libamdfq.so
py=${AXL_PYTHON:-/home/acite/miniconda3/envs/axl/bin/python}

log=$here/frozen.log
trainlog=$here/train.log
samplog=$here/sample.log
crash=$here/crash.txt
sample_seconds=${SAMPLE_SECONDS:-300}

[ -x "$so" ] || { echo "collect.sh: $so is missing; build amdfq/amdfq-tail first" >&2; exit 2; }
[ -x "$py" ] || { echo "collect.sh: interpreter $py not executable (set AXL_PYTHON)" >&2; exit 2; }
[ -s "$log" ] && { echo "collect.sh: $log already has content; move it aside first" >&2; exit 2; }

holders=$(
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
        ls -l "/proc/$p/fd" 2>/dev/null | grep -qE 'renderD128|/dev/kfd' && printf '%s ' "$p"
    done
)
[ -n "$holders" ] && { echo "collect.sh: another GPU client holds the device: $holders" >&2; exit 2; }

baseline() {
    for d in /sys/class/drm/card*/device; do
        [ -r "$d/mem_info_vram_used" ] || continue
        printf '%s vis=%s gtt=%s\n' "$d" "$(cat "$d/mem_info_vram_used")" "$(cat "$d/mem_info_gtt_used")"
    done
}

runtime=${AXL_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/axltrainer}

{
    echo "== collect.sh $(date -Is)"
    echo "repo      $repo @ $(git -C "$repo" rev-parse --short HEAD)"
    echo "hook      $so"
    echo "sha256    $(sha256sum "$so" | cut -d' ' -f1)"
    echo "python    $py ($("$py" -V 2>&1))"
    echo "runtime   $runtime"
    echo "baseline (before):"
    baseline | sed 's/^/  /'
} | tee "$here/baseline.txt"

bash "$here/sample-host.sh" "$samplog" "$sample_seconds" &
sampler=$!

start=$(date +%s.%N)
(
    ulimit -c 0
    cd "$repo" || exit 2
    PATH="$(dirname "$py"):$PATH" \
        AMDFQ_TAIL=0 \
        AMDFQ_LOG="$log" \
        LD_PRELOAD="$so" \
        PYTHONUNBUFFERED=1 \
        bash start_train.sh
) > /dev/null 2> >(tee "$trainlog" >&2)
rc=$?
end=$(date +%s.%N)

kill "$sampler" 2>/dev/null
wait "$sampler" 2>/dev/null
sleep 1

{
    echo "== crash.txt $(date -Is)"
    echo "exit code      $rc"
    echo "wall seconds   $(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b - a}')"
    echo "command        AMDFQ_TAIL=0 AMDFQ_LOG=$log LD_PRELOAD=$so bash start_train.sh (cwd $repo)"
    echo
    echo "-- state.json ($runtime/state.json)"
    cat "$runtime/state.json" 2>&1
    echo
    echo "-- hook log tail (last 5 lines)"
    tail -5 "$log" 2>&1
    echo
    echo "-- hook log size / line count"
    wc -c -l "$log" 2>&1
    echo
    echo "-- baseline (after)"
    baseline | sed 's/^/  /'
    echo
    echo "-- dmesg tail (amdgpu/amdkfd)"
    (dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null || echo "dmesg not readable") | grep -iE 'amdgpu|amdkfd|kfd|fault|ring' | tail -15
    echo
    echo "-- stderr tail ($trainlog)"
    tail -15 "$trainlog" 2>&1
} | tee "$crash"

exit 0

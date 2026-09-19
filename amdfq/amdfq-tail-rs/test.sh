#!/usr/bin/env bash
#
# The self-check for this interposer: builds it, checks the object against the source and the runtime,
# runs a command with it preloaded and judges what it logged.
#
#   bash amdfq/amdfq-tail-rs/test.sh                        # default: one torch allocation
#   bash amdfq/amdfq-tail-rs/test.sh bash start_train.sh    # a real run
#   bash amdfq/amdfq-tail-rs/test.sh ./any/hip/program      # anything that calls hipMalloc
#
# The hook logs through the `log` crate to stderr, so the command's stderr is collected into
# $AMDFQ_LOG_FILE (default /tmp/amdfq-tail-rs.<pid>.log); AMDFQ_LOG_LEVEL=warn keeps only the
# anomalies. AMDFQ_TAIL=0 disables the guard (DESIGN.md). Check results are printed after the run and
# the script exits non-zero if one of them failed. Starting a real workload under the hook — with the
# profile of your choice — is run.sh's job.
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest=$here/Cargo.toml
so=$here/target/release/libamdfq_tail_rs.so
log=${AMDFQ_LOG_FILE:-/tmp/amdfq-tail-rs.$$.log}
runtime=${AMDFQ_HIP_RUNTIME:-/opt/rocm/core/lib/libamdhip64.so.7}
level=${AMDFQ_LOG_LEVEL:-info}
failed=0

report() { # report <label> <ok|FAIL> [detail]
    if [[ $2 == ok ]]; then printf 'ok    %s%s\n' "$1" "${3:+ — $3}"
    else printf 'FAIL  %s%s\n' "$1" "${3:+ — $3}"; failed=1; fi
}

if [[ ! -f $so ]]; then
    echo "amdfq-tail-rs: building $so" >&2
    cargo build --release --offline --manifest-path "$manifest" >/dev/null 2>&1 ||
        cargo build --release --manifest-path "$manifest" >/dev/null || exit 2
fi

# The names this object declares, taken from the source that writes them — no second list here.
declared=$(grep -oP 'pub unsafe extern "C" fn \K\w+' "$here/src/hooks.rs" | sort)
defined=$(nm -D --defined-only "$so" | awk '{print $NF}' | sort)

echo "== interposer $so"
if [[ $declared == "$defined" ]]; then
    report "exports == declared" ok "$(echo "$declared" | tr '\n' ' ')"
else
    report "exports == declared" FAIL "declared: $(echo "$declared" | tr '\n' ' ') / defined: $(echo "$defined" | tr '\n' ' ')"
fi

if [[ -f $runtime ]]; then
    for name in $declared; do
        nm -D --defined-only "$runtime" | grep -q " $name@" ||
            report "$name exists in $(basename "$runtime")" FAIL
    done
    report "declared names exist in $(basename "$runtime")" ok
else
    report "runtime $runtime present" FAIL
fi

# Style: the hooks hold no lock (D3), and only real.rs names a symbol or calls dlsym (D8).
locks=$(grep -nE 'RwLock|Mutex|\.lock\(|\.read\(|\.write\(' "$here/src/hooks.rs" || true)
[[ -z $locks ]] && report "src/hooks.rs holds no lock" ok || report "src/hooks.rs holds no lock" FAIL "$locks"
resolvers=$(grep -rl dlsym "$here/src" | sort | tr '\n' ' ')
[[ $resolvers == "$here/src/real.rs " ]] && report "dlsym only in src/real.rs" ok || report "dlsym only in src/real.rs" FAIL "$resolvers"

if [[ $# -eq 0 ]]; then
    set -- "${AXL_PYTHON:-python3}" -c \
        'import torch; x = torch.zeros(8, device="cuda"); torch.cuda.synchronize(); del x; torch.cuda.empty_cache()'
fi

echo "== run AMDFQ_LOG_LEVEL=$level LD_PRELOAD=$so" >&2
rc=0
LD_PRELOAD=$so "$@" 2>"$log" || rc=$?
[[ $rc -ne 0 ]] && echo "amdfq-tail-rs: the command exited $rc" >&2

lines=$(grep -c 'amdfq_tail_rs::hooks' "$log" || true)
mallocs=$(grep -c 'amdfq_tail_rs::hooks hipMalloc(' "$log" || true)
frees=$(grep -c 'amdfq_tail_rs::hooks hipFree(' "$log" || true)
guarded=$(grep -cE 'hipMalloc\(.*\) -> ret=0 .* block=[0-9]+ guarded$' "$log" || true)
backed=$(grep -cE 'hipMalloc\(.*\) -> ret=0 .* block=[0-9]+ backed$' "$log" || true)
padded=$(grep -cE 'hipMalloc\(.*\) -> ret=0 .* padded from=' "$log" || true)
released=$(grep -cE 'hipFree\(.* -> ret=.* released page=' "$log" || true)
untracked=$(grep -c 'untracked' "$log" || true)
duplicates=$(grep -c 'duplicate ' "$log" || true)
threads=$(grep -oP 'T=\K[0-9]+' "$log" | sort -u | wc -l)

echo
echo "== $log"
grep 'amdfq_tail_rs::hooks' "$log" | head -5
echo "== coverage"
printf 'hook lines  %s\n' "$lines"
printf 'hipMalloc   %s (guarded %s, backed %s, padded %s)\n' "$mallocs" "$guarded" "$backed" "$padded"
printf 'hipFree     %s (released %s, untracked %s)\n' "$frees" "$released" "$untracked"
printf 'duplicate   %s\n' "$duplicates"
printf 'threads     %s\n' "$threads"

if [[ $level == off || $level == error || $level == warn ]]; then
    echo "note  AMDFQ_LOG_LEVEL=$level filters the per-call lines; the checks below need 'info'"
elif [[ $lines -gt 0 ]]; then
    report "preload reached the gates" ok
else
    report "preload reached the gates" FAIL "no line with target amdfq_tail_rs::hooks"
fi

if [[ ${lines:-0} -gt 0 ]]; then
    [[ $mallocs -eq $frees && $frees -gt 0 ]] &&
        report "malloc and free counts match" ok ||
        report "malloc and free counts match" FAIL "hipMalloc $mallocs, hipFree $frees"
    [[ $untracked -eq 0 && $duplicates -eq 0 ]] &&
        report "no duplicate and no untracked" ok ||
        report "no duplicate and no untracked" FAIL "duplicate $duplicates, untracked $untracked"
fi

[[ $failed -eq 0 ]] && exit $rc || exit 1

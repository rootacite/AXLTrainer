#!/usr/bin/env bash
#
# The self-check for this interposer: builds it, checks the object against the source and the runtime,
# runs a command with it preloaded and judges what it logged.
#
#   bash amdfq/amdfq-vmm-rs/test.sh                        # default: one torch allocation
#   bash amdfq/amdfq-vmm-rs/test.sh bash start_train.sh    # a real run
#   bash amdfq/amdfq-vmm-rs/test.sh ./any/hip/program      # anything that calls hipMalloc
#
# The hook logs through the `log` crate to stderr, so the command's stderr is collected into
# $AMDFQ_LOG_FILE (default /tmp/amdfq-rs.<pid>.log); AMDFQ_LOG_LEVEL=warn keeps only the anomalies.
# Check results are printed after the run and the script exits non-zero if one of them failed.
# Starting a real workload under the hook — with the profile of your choice — is run.sh's job.
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest=$here/Cargo.toml
so=$here/target/release/libamdfq_vmm_rs.so
log=${AMDFQ_LOG_FILE:-/tmp/amdfq-rs.$$.log}
runtime=${AMDFQ_HIP_RUNTIME:-/opt/rocm/core/lib/libamdhip64.so.7}
level=${AMDFQ_LOG_LEVEL:-info}
failed=0

report() { # report <label> <ok|FAIL> [detail]
    if [[ $2 == ok ]]; then printf 'ok    %s%s\n' "$1" "${3:+ — $3}"
    else printf 'FAIL  %s%s\n' "$1" "${3:+ — $3}"; failed=1; fi
}

if [[ ! -f $so ]]; then
    echo "amdfq-rs: building $so" >&2
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
    # Two allocations with a free between them, so the VA a free gave back (reuse mode) — or kept
    # (never-reuse mode) — shows up as two served lines and one or no "gave ... VA back" line.
    set -- "${AXL_PYTHON:-python3}" -c \
        'import torch
x = torch.zeros(8, device="cuda"); torch.cuda.synchronize(); del x; torch.cuda.empty_cache()
y = torch.zeros(8, device="cuda"); torch.cuda.synchronize(); del y; torch.cuda.empty_cache()'
    modes=(0 1)
else
    # A command was given: run it once, in the mode the environment asks for.
    modes=("${AMDFQ_VA_NEVER_REUSE:-0}")
fi

for mode in "${modes[@]}"; do
export AMDFQ_VA_NEVER_REUSE=$mode
run_log=$log
if [[ ${#modes[@]} -gt 1 ]]; then
    run_log="$log.$mode"
    echo "== mode AMDFQ_VA_NEVER_REUSE=$mode" >&2
fi
echo "== run AMDFQ_LOG_LEVEL=$level LD_PRELOAD=$so" >&2
rc=0
LD_PRELOAD=$so "$@" 2>"$run_log" || rc=$?
[[ $rc -ne 0 ]] && echo "amdfq-rs: the command exited $rc" >&2

lines=$(grep -c 'amdfq_vmm_rs::hooks' "$run_log" || true)
mallocs=$(grep -c 'amdfq_vmm_rs::hooks hipMalloc(' "$run_log" || true)
frees=$(grep -c 'amdfq_vmm_rs::hooks hipFree(' "$run_log" || true)
served=$(grep -c 'hipMalloc(.*) -> ret=0 served va=' "$run_log" || true)
forwarded=$(grep -c 'hipMalloc(.*) -> ret=0 forwarded va=' "$run_log" || true)
matched=$(( $(grep -cE 'hipFree\(.* -> released size=' "$run_log" || true) + $(grep -c 'hipFree(.*forwarded size=' "$run_log" || true) ))
untracked=$(grep -c 'untracked' "$run_log" || true)
duplicates=$(grep -c 'duplicate ' "$run_log" || true)
threads=$(grep -oP 'T=\K[0-9]+' "$run_log" | sort -u | wc -l)
given_back=$(grep -c 'gave .* bytes of VA back' "$run_log" || true)
torn_down=$(grep -cE 'hipFree\(.* -> released size=' "$run_log" || true)

echo
echo "== $run_log"
grep 'amdfq_vmm_rs::' "$run_log" | head -6
echo "== coverage"
printf 'hook lines  %s\n' "$lines"
printf 'hipMalloc   %s (served %s, forwarded %s)\n' "$mallocs" "$served" "$forwarded"
printf 'hipFree     %s (matched %s, untracked %s)\n' "$frees" "$matched" "$untracked"
printf 'duplicate   %s\n' "$duplicates"
printf 'threads     %s\n' "$threads"
printf 'VA given back %s (mode %s)\n' "$given_back" "$mode"

if [[ $level == off || $level == error || $level == warn ]]; then
    echo "note  AMDFQ_LOG_LEVEL=$level filters the per-call lines; the checks below need 'info'"
elif [[ $lines -gt 0 ]]; then
    report "preload reached the gates" ok
else
    report "preload reached the gates" FAIL "no line with target amdfq_vmm_rs::hooks"
fi

if [[ ${lines:-0} -gt 0 ]]; then
    [[ $mallocs -eq $frees && $frees -gt 0 && $matched -eq $frees ]] &&
        report "every free matched a registration" ok ||
        report "every free matched a registration" FAIL "hipMalloc $mallocs, hipFree $frees, matched $matched"
    [[ $untracked -eq 0 && $duplicates -eq 0 ]] &&
        report "no duplicate and no untracked" ok ||
        report "no duplicate and no untracked" FAIL "duplicate $duplicates, untracked $untracked"
    reused=$(grep -oP 'served va=\K0x[0-9a-f]+' "$run_log" | sort | uniq -d | tr '\n' ' ')
    if [[ $mode == 0 ]]; then
        # Reuse mode: every extent teardown hands its span back, so the next allocation may land on
        # the same VA again.
        [[ $torn_down -gt 0 && $given_back -eq $torn_down ]] &&
            report "every teardown gave its VA back" ok "$reused" ||
            report "every teardown gave its VA back" FAIL "torn down $torn_down, given back $given_back"
    else
        [[ $given_back -eq 0 ]] &&
            report "no free gave its VA back (never-reuse)" ok ||
            report "no free gave its VA back (never-reuse)" FAIL "given back $given_back"
        [[ -z $reused ]] &&
            report "served VA never reused" ok ||
            report "served VA never reused" FAIL "$reused"
    fi
fi
done

[[ $failed -eq 0 ]] && exit $rc || exit 1

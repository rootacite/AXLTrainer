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
#
# A second, built-in pass runs the same tiny torch workload with AMDFQ_POOL_SIZE set (default
# 16 MiB, AMDFQ_POOL_CHECK=0 to skip), because the pool's own path — carve, give back, release the
# pool when its last block goes — needs the knob on to be covered at all.
#
# A third pass covers the load-time warm-up (early.rs, the one thing here that is not an allocation
# gate): it names the runtime it touched, and measures that no thread of a settled process burns a
# core. AMDFQ_EARLY_HIP=0 skips it.
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

thread_ticks() { # thread_ticks <pid> — one "tid utime+stime" line per thread, for the deltas below
    for task in /proc/$1/task/*; do
        awk -v tid="${task##*/}" '{print tid, $14+$15}' "$task/stat" 2>/dev/null
    done
}

busiest_ticks() { # busiest_ticks <run_log> [environment assignments...] — ticks a thread used in 5s
    local run_log=$1; shift
    env "$@" LD_PRELOAD=$so "${AXL_PYTHON:-python3}" -c 'import torch, time
torch.cuda.init()
x = torch.ones(8, 8, device="cuda"); torch.cuda.synchronize()
print("ready", flush=True); time.sleep(14)' >"$run_log" 2>&1 &
    local pid=$!
    sleep 6
    thread_ticks "$pid" >"$run_log.before" 2>/dev/null
    sleep 5
    thread_ticks "$pid" >"$run_log.after" 2>/dev/null
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    join <(sort -n "$run_log.before") <(sort -n "$run_log.after") |
        awk '{delta = $3 - $2; if (delta > worst) worst = delta} END {print worst + 0}'
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
pooled=$(grep -c 'hipMalloc(.*) -> ret=0 pooled va=' "$run_log" || true)
forwarded=$(grep -c 'hipMalloc(.*) -> ret=0 forwarded va=' "$run_log" || true)
matched=$(( $(grep -cE 'hipFree\(.* -> released size=' "$run_log" || true) + $(grep -c 'hipFree(.*forwarded size=' "$run_log" || true) ))
untracked=$(grep -c 'untracked' "$run_log" || true)
duplicates=$(grep -c 'duplicate ' "$run_log" || true)
threads=$(grep -oP 'T=\K[0-9]+' "$run_log" | sort -u | wc -l)
given_back=$(grep -c 'gave .* bytes of VA back' "$run_log" || true)
# A pooled free logs the same `released size=` shape plus ` pool=…`, so the two kinds of teardown
# are told apart here: only a teardown that owns its span gives a VA back on the spot, while a pool
# gives its span back when its last block goes — which the pool's own line reports.
pool_teardowns=$(grep -c 'amdfq_vmm_rs::pool pool released' "$run_log" || true)
teardowns=$(( $(grep -E 'hipFree\(.* -> released size=' "$run_log" | grep -vc ' pool=' || true) + pool_teardowns ))
pool_lines=$(grep -c 'amdfq_vmm_rs::pool' "$run_log" || true)
# pool.rs also says what the knob came to, once, whether or not it is on; only the lines that report
# something the route *did* are evidence of a pool being used.
pool_activity=$(grep -cE 'amdfq_vmm_rs::pool pool (created|released|failed|refused)' "$run_log" || true)

echo
echo "== $run_log"
grep 'amdfq_vmm_rs::' "$run_log" | head -6
echo "== coverage"
printf 'hook lines  %s\n' "$lines"
printf 'hipMalloc   %s (served %s, pooled %s, forwarded %s)\n' "$mallocs" "$served" "$pooled" "$forwarded"
printf 'hipFree     %s (matched %s, untracked %s, of the matched %s pooled)\n' \
    "$frees" "$matched" "$untracked" "$(grep -cE 'hipFree\(.* -> released size=.* pool=' "$run_log" || true)"
printf 'duplicate   %s\n' "$duplicates"
printf 'threads     %s\n' "$threads"
printf 'VA given back %s (mode %s)  teardowns %s (of them %s pool teardowns)\n' \
    "$given_back" "$mode" "$teardowns" "$pool_teardowns"
printf 'pool lines  %s (%s of them report something the route did)\n' "$pool_lines" "$pool_activity"

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
    [[ ${AMDFQ_POOL_SIZE:-0} == 0 && $pool_activity -eq 0 ]] &&
        report "no pool without the knob" ok ||
        { [[ ${AMDFQ_POOL_SIZE:-0} != 0 ]] &&
            report "no pool without the knob" ok "the knob is on for this run" ||
            report "no pool without the knob" FAIL "$pool_activity pool line(s)"; }
    reused=$(grep -oP 'served va=\K0x[0-9a-f]+' "$run_log" | sort | uniq -d | tr '\n' ' ')
    if [[ $mode == 0 ]]; then
        # Reuse mode: every extent teardown hands its span back, and every pool teardown hands the
        # pool's span back, so the next allocation may land on the same VA again.
        [[ $teardowns -gt 0 && $given_back -eq $teardowns ]] &&
            report "every teardown gave its VA back" ok "$reused" ||
            report "every teardown gave its VA back" FAIL "teardowns $teardowns, given back $given_back"
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

# == pool pass
# The pool path only exists with the knob on, so it gets its own run of the same tiny workload: one
# allocation small enough to be carved (2 MiB here) and one that is not, each followed by a free and
# an empty_cache, which is what makes the pool empty and hand its object back.
pool_size=${AMDFQ_POOL_CHECK:-16777216}
if [[ $pool_size == 0 ]]; then
    echo
    echo "== pool pass skipped (AMDFQ_POOL_CHECK=0)"
else
    pool_log="$log.pool"
    echo
    echo "== pool pass AMDFQ_POOL_SIZE=$pool_size, log $pool_log" >&2
    set -- "${AXL_PYTHON:-python3}" -c \
        'import torch
x = torch.zeros(8, device="cuda"); y = torch.zeros(1024, device="cuda")
torch.cuda.synchronize(); del x, y; torch.cuda.empty_cache()'
    pool_rc=0
    AMDFQ_POOL_SIZE=$pool_size AMDFQ_VA_NEVER_REUSE=0 LD_PRELOAD=$so "$@" 2>"$pool_log" || pool_rc=$?
    created=$(grep -c 'amdfq_vmm_rs::pool pool created' "$pool_log" || true)
    released=$(grep -c 'amdfq_vmm_rs::pool pool released' "$pool_log" || true)
    carved=$(grep -c 'hipMalloc(.*) -> ret=0 pooled va=' "$pool_log" || true)
    given_back_pool=$(grep -c 'gave .* bytes of VA back' "$pool_log" || true)
    pool_teardowns=$(grep -c 'amdfq_vmm_rs::pool pool released' "$pool_log" || true)
    pool_errors=$(( $(grep -c 'amdfq_vmm_rs::pool pool failed' "$pool_log" || true) + $(grep -c 'amdfq_vmm_rs::pool pool refused' "$pool_log" || true) ))
    pool_untracked=$(grep -c 'untracked' "$pool_log" || true)
    pool_duplicates=$(grep -c 'duplicate ' "$pool_log" || true)

    grep 'amdfq_vmm_rs::pool' "$pool_log" | head -4
    echo "== pool coverage"
    printf 'pools       %s created, %s released, %s alive at exit\n' \
        "$created" "$released" "$((created - released))"
    printf 'carved      %s (pool failures/refusals %s)\n' "$carved" "$pool_errors"

    [[ $carved -gt 0 ]] &&
        report "an allocation was carved out of a pool" ok ||
        report "an allocation was carved out of a pool" FAIL "no pooled line in $pool_log"
    [[ $created -gt 0 && $released -gt 0 && $created -ge $released ]] &&
        report "a pool was released once its blocks were gone" ok "$created created, $released released" ||
        report "a pool was released once its blocks were gone" FAIL \
            "$created created, $released released"
    [[ $given_back_pool -ge $pool_teardowns ]] &&
        report "every pool teardown gave its VA back" ok ||
        report "every pool teardown gave its VA back" FAIL \
            "pool teardowns $pool_teardowns, given back $given_back_pool"
    [[ $pool_errors -eq 0 ]] &&
        report "no pool fell back to the solo route" ok ||
        report "no pool fell back to the solo route" FAIL "$pool_errors pool failure/refusal line(s)"
    [[ $pool_untracked -eq 0 && $pool_duplicates -eq 0 ]] &&
        report "pool pass: no duplicate and no untracked" ok ||
        report "pool pass: no duplicate and no untracked" FAIL \
            "duplicate $pool_duplicates, untracked $pool_untracked"
    [[ $pool_rc -eq 0 ]] ||
        echo "amdfq-rs: the pool pass exited $pool_rc" >&2
fi

# == early touch pass
# On this box's ROCR the runtime's AsyncEventsLoop spins a whole CPU core from the first GPU op on,
# for the life of the process, and the crate's load-time warm-up (early.rs) is what stops it. Two
# things are worth checking and neither is the allocation path: that what it touched is the
# interpreter's own runtime and not the system one under /opt/rocm (touching that one kills
# `import torch`), and that a settled process has no thread burning a core. Measured the way the
# spin was found: the busiest thread of the process, 5s window, in ticks (100/s).
if [[ ${AMDFQ_EARLY_HIP:-} == 0 ]]; then
    echo
    echo "== early touch pass skipped (AMDFQ_EARLY_HIP=0 turns the warm-up off)"
else
    echo
    echo "== early touch pass"
    early_logs="$log.early"
    warmed=$(busiest_ticks "$early_logs")
    plain=$(busiest_ticks "$early_logs.off" AMDFQ_EARLY_HIP=0)
    printf 'busiest thread  %s ticks/5s with the warm-up, %s with AMDFQ_EARLY_HIP=0\n' "$warmed" "$plain"
    [[ $plain -ge 100 ]] ||
        echo "note  AMDFQ_EARLY_HIP=0 shows $plain ticks/5s: this ROCR does not spin anyway, so the number above proves nothing either way"

    early_line=$(grep -m1 'amdfq_vmm_rs::early early touch' "$early_logs" || true)
    if [[ $early_line == *" via "* && $early_line != *"/opt/rocm/core/lib/"* ]]; then
        report "the warm-up touched the interpreter's own runtime" ok "${early_line##* via }"
    else
        report "the warm-up touched the interpreter's own runtime" FAIL \
            "${early_line:-no early touch line in $early_logs}"
    fi
    [[ $warmed -lt 100 ]] &&
        report "no thread burns a core under the hook" ok "$warmed ticks/5s" ||
        report "no thread burns a core under the hook" FAIL "$warmed ticks/5s (one core is 500)"
fi

[[ $failed -eq 0 ]] && exit $rc || exit 1

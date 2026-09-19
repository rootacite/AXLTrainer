#!/usr/bin/env bash
#
# Builds the interposer in the profile you pick and starts a real workload under it.
#
#   bash amdfq/amdfq-vmm-rs/run.sh test                 # test/torch-test.py under the hook
#   bash amdfq/amdfq-vmm-rs/run.sh train                # bash start_train.sh under the hook
#   bash amdfq/amdfq-vmm-rs/run.sh --debug test         # hook built into target/debug instead of release
#   bash amdfq/amdfq-vmm-rs/run.sh test --time-scale 4  # arguments after the target go to the target
#
# The hook logs to stderr, so the target's stderr stays on the terminal and is copied into
# $AMDFQ_LOG_FILE (default /tmp/amdfq-rs-<target>.<pid>.log); AMDFQ_LOG_LEVEL=warn keeps only the
# anomalies. test.sh is the other entry point: it builds the release object and checks it (export
# face, style, gate coverage) instead of starting a workload.
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)

usage() {
    cat >&2 <<'EOF'
usage: run.sh [--debug|--release] <train|test> [arguments...]

  train    starts bash start_train.sh (cwd = repo root; config.toml decides the run, no arguments)
  test     runs test/torch-test.py (arguments are passed through)
  --debug  build the hook into target/debug, --release into target/release (default)

environment: AXL_PYTHON (interpreter for test, default python3; for train its directory is prepended
             to PATH, because start_train.sh calls a bare python), AMDFQ_LOG_FILE, AMDFQ_LOG_LEVEL
EOF
}

profile=release
target=
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug) profile=debug; shift ;;
        --release) profile=release; shift ;;
        -h|--help) usage; exit 0 ;;
        train|test) target=$1; shift; break ;;
        *) echo "run.sh: unknown argument '$1'" >&2; usage; exit 2 ;;
    esac
done

if [[ -z $target ]]; then
    usage
    exit 2
fi

extra=("$@")
if [[ $target == train && ${#extra[@]} -gt 0 ]]; then
    echo "run.sh: train takes no arguments (config.toml decides the run)" >&2
    exit 2
fi

cargo_args=(build --manifest-path "$here/Cargo.toml")
case $profile in
    debug) so=$here/target/debug/libamdfq_vmm_rs.so ;;
    release)
        cargo_args+=(--release)
        so=$here/target/release/libamdfq_vmm_rs.so
        ;;
esac

echo "run.sh: cargo ${cargo_args[*]}" >&2
cargo "${cargo_args[@]}" --offline >/dev/null 2>&1 || cargo "${cargo_args[@]}" >/dev/null || exit 2
if [[ ! -f $so ]]; then
    echo "run.sh: $so was not produced" >&2
    exit 2
fi

log=${AMDFQ_LOG_FILE:-/tmp/amdfq-rs-$target.$$.log}
echo "run.sh: target=$target profile=$profile level=${AMDFQ_LOG_LEVEL:-info}" >&2
echo "run.sh: LD_PRELOAD=$so" >&2
echo "run.sh: stderr -> $log" >&2

case $target in
    train)
        # start_train.sh calls a bare `python`; honour AXL_PYTHON the way the rest of the repo does.
        if [[ -n ${AXL_PYTHON:-} ]]; then
            PATH="$(dirname -- "$AXL_PYTHON"):$PATH"
            export PATH
        fi
        set -- bash start_train.sh
        ;;
    test) set -- "${AXL_PYTHON:-python3}" test/torch-test.py "${extra[@]}" ;;
esac

cd "$repo" || exit 2
rc=0
LD_PRELOAD=$so "$@" 2> >(tee "$log" >&2) || rc=$?

# What the hook saw, counted apart from the target's own output.
if [[ -f $log ]]; then
    printf 'run.sh: hook lines %s (hipMalloc %s, hipFree %s, untracked %s, duplicate %s, threads %s)\n' \
        "$(grep -c 'amdfq_vmm_rs::hooks' "$log" || true)" \
        "$(grep -c 'hipMalloc(' "$log" || true)" \
        "$(grep -c 'hipFree(' "$log" || true)" \
        "$(grep -c 'untracked' "$log" || true)" \
        "$(grep -c 'duplicate ' "$log" || true)" \
        "$(grep -oP 'T=\K[0-9]+' "$log" | sort -u | wc -l)" >&2
fi
echo "run.sh: exit $rc, log $log" >&2
exit $rc

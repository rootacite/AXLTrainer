#!/bin/bash
set -euo pipefail

# Starts an arbitrary program under the allocation patch this repo's config.toml selects, i.e. the
# same LD_PRELOAD / AMDFQ_VA_STATUS / AMDFQ_VRAM_RESERVE / AMDFQ_VA_NEVER_REUSE start_train.sh would
# set:
#
#   ./start_hook.sh --workd "/opt/X" Y              # run Y with cwd /opt/X
#   ./start_hook.sh python -u test/torch-test.py    # cwd defaults to the repo root
#   ./start_hook.sh -- bash start_train.sh          # a path relative to the work directory
#
# The choice and its parameters come from [environment].amdfq (none | tail | vmm),
# amdfq_vram_reserve_gib and amdfq_va_never_reuse, through trainer/amdfq_patch.py — the same source
# start_train.sh reads.

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat >&2 <<EOF
usage: start_hook.sh [--workd DIR] [--] PROGRAM [ARGUMENTS...]

  --workd DIR, --workdir DIR   run PROGRAM in DIR (default: this repo's root)
  -h, --help                   this text

The patch [environment].amdfq selects is preloaded into PROGRAM: tail and vmm put their release
.so in LD_PRELOAD (vmm also exports AMDFQ_VA_STATUS, AMDFQ_VRAM_RESERVE from
amdfq_vram_reserve_gib and AMDFQ_VA_NEVER_REUSE from amdfq_va_never_reuse), none runs PROGRAM
untouched. A missing .so fails the start instead of running unpatched. AMDFQ_LOG_LEVEL and
AMDFQ_LOG_FILE pass through from the environment.

PROGRAM is looked up after the directory change, so a relative path means "inside DIR"; write an
absolute path to pin it. Example:

  $0 --workd "/opt/X" Y
EOF
}

workdir=$repo
while [[ $# -gt 0 ]]; do
    case $1 in
        --workd|--workdir)
            if [[ $# -lt 2 ]]; then
                echo "start_hook.sh: $1 needs a directory" >&2
                usage
                exit 2
            fi
            workdir=$2
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "start_hook.sh: unknown option '$1'" >&2; usage; exit 2 ;;
        *) break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "start_hook.sh: no program given" >&2
    usage
    exit 2
fi
if [[ ! -d $workdir ]]; then
    echo "start_hook.sh: work directory does not exist: $workdir" >&2
    exit 2
fi

# Resolved from the repo root, because the lookup imports the trainer package from there.
if ! amdfq_line=$(cd "$repo" && "${AXL_PYTHON:-python3}" -u \
        -c "from trainer.amdfq_patch import launch_env_line; print(launch_env_line())"); then
    echo "start_hook.sh: could not read [environment].amdfq from $repo/config.toml" >&2
    exit 1
fi
IFS='|' read -r amdfq_choice amdfq_so amdfq_va_status amdfq_vram_reserve amdfq_va_never_reuse <<< "$amdfq_line"

# Never guess: an unparsable line would otherwise read as "none" and run the target unpatched.
case $amdfq_choice in
    none|tail|vmm) ;;
    *)
        echo "start_hook.sh: unexpected amdfq value in $repo/config.toml: '$amdfq_choice'" >&2
        exit 1
        ;;
esac

if [[ $amdfq_choice == tail || $amdfq_choice == vmm ]]; then
    if [[ -z $amdfq_so || ! -f $amdfq_so ]]; then
        echo "start_hook.sh: $amdfq_choice patch library missing${amdfq_so:+: $amdfq_so}" >&2
        exit 1
    fi
    if [[ -n ${LD_PRELOAD:-} ]]; then
        export LD_PRELOAD="$amdfq_so:$LD_PRELOAD"
    else
        export LD_PRELOAD="$amdfq_so"
    fi
    if [[ $amdfq_choice == vmm && -n $amdfq_va_status ]]; then
        export AMDFQ_VA_STATUS="$amdfq_va_status"
    fi
    if [[ $amdfq_choice == vmm && -n $amdfq_vram_reserve ]]; then
        export AMDFQ_VRAM_RESERVE="$amdfq_vram_reserve"
    fi
    if [[ $amdfq_choice == vmm && -n $amdfq_va_never_reuse ]]; then
        export AMDFQ_VA_NEVER_REUSE="$amdfq_va_never_reuse"
    fi
fi

cd "$workdir"
echo "start_hook.sh: amdfq=$amdfq_choice workdir=$PWD${amdfq_so:+ LD_PRELOAD=$amdfq_so}${AMDFQ_VRAM_RESERVE:+ AMDFQ_VRAM_RESERVE=$AMDFQ_VRAM_RESERVE}${AMDFQ_VA_NEVER_REUSE:+ AMDFQ_VA_NEVER_REUSE=$AMDFQ_VA_NEVER_REUSE}" >&2

exec "$@"

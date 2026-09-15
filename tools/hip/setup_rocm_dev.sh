#!/usr/bin/env bash
# setup_rocm_dev.sh — make the ROCm stack inside a conda environment usable for HIP builds.
#
#   bash tools/hip/setup_rocm_dev.sh              apply to $CONDA_PREFIX
#   bash tools/hip/setup_rocm_dev.sh /path/to/env apply to another environment
#   bash tools/hip/setup_rocm_dev.sh --check      report what is missing, change nothing
#   bash tools/hip/setup_rocm_dev.sh --verify     apply, then build and run hip_smoke.hip
#   -h                                           usage
#
# The ROCm wheels (`rocm-sdk-core`, `rocm-sdk-libraries`) ship the runtime and the
# compiler but none of the development glue: no unversioned `.so` links for the
# linker, no `amdgcn` device-library path under ROCM_PATH, and no cmake configs.
# Left alone, `hipcc` takes its clang from the system /opt/rocm, and a binary built
# here loads the system ROCm at run time. Every step below is idempotent; the
# reasoning and the compile commands are in HIP.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_SRC="$REPO_ROOT/tools/hip/hip_smoke.hip"
HOOK_NAME="axl_rocm_hip_toolchain.sh"

usage() {
    cat <<'EOF'
Usage: bash tools/hip/setup_rocm_dev.sh [ENV_PREFIX] [--check] [--verify]

  ENV_PREFIX   conda environment to fix up. Defaults to $CONDA_PREFIX, and
               otherwise to the environment environment.yml names.
  --check      Report which links and hooks are missing; write nothing.
  --verify     After applying, compile and run tools/hip/hip_smoke.hip.
  -h           Show this help.

Idempotent: re-running refreshes whatever is missing and reports the rest as
already in place.
EOF
}

die() {
    echo "setup_rocm_dev.sh: $*" >&2
    exit 1
}

# --- environment discovery ----------------------------------------------------

# The environment `environment.yml` names, so the script also works from a shell
# that never activated it.
env_name_from_yml() {
    local yml="$REPO_ROOT/environment.yml" name
    [ -f "$yml" ] || return 1
    name="$(sed -n 's/^name:[[:space:]]*//p' "$yml" | head -n 1)"
    [ -n "$name" ] || return 1
    echo "$name"
}

conda_base() {
    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE:-}" ]; then
        dirname "$(dirname "$CONDA_EXE")"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null
        return 0
    fi
    local base
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        if [ -x "$base/bin/conda" ]; then
            echo "$base"
            return 0
        fi
    done
    return 1
}

# --- link helpers -------------------------------------------------------------

created=0
present=0

# 0 = linked now (or would be, under --check), 1 = already correct,
# 2 = a real file occupies the path and is left alone.
ensure_link() {
    local link="$1" target="$2"
    if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
        return 1
    fi
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        return 2
    fi
    if [ "$check_only" = no ]; then
        mkdir -p "$(dirname "$link")"
        ln -sfn "$target" "$link"
    fi
    return 0
}

count_link() {
    local rc=0
    if ensure_link "$1" "$2"; then
        created=$((created + 1))
        [ "$check_only" = yes ] && echo "    missing $1 -> $2"
    else
        rc=$?
        if [ "$rc" = 2 ]; then
            echo "    skipped $1 (a real file is in the way)" >&2
        else
            present=$((present + 1))
        fi
    fi
    return 0
}

report_step() {
    printf '  %-14s: %d added, %d already in place\n' "$1" "$created" "$present"
    created=0
    present=0
}

# --- steps --------------------------------------------------------------------

# The linker looks for `libfoo.so`; the wheels only ship `libfoo.so.N`.
link_dev_libs() {
    local dir file base
    for dir in "$core/lib" "$libs/lib"; do
        for file in "$dir"/*.so.*; do
            [ -f "$file" ] || continue
            base="${file%%.so.*}"
            count_link "$base.so" "$(basename "$file")"
        done
    done
    report_step "lib links"
}

# hipcc derives the device bitcode path from ROCM_PATH, which in the system
# layout is <root>/amdgcn -> lib/llvm/amdgcn.
link_device_libs() {
    count_link "$core/amdgcn" "lib/llvm/amdgcn"
    report_step "amdgcn"
}

# These three are header-only and are not part of any ROCm wheel, but torch's
# headers include <thrust/complex.h>, so extension builds need them.
link_host_headers() {
    local sys_inc="" dir
    for dir in /opt/rocm/core/include /opt/rocm/include; do
        if [ -f "$dir/thrust/complex.h" ]; then
            sys_inc="$dir"
            break
        fi
    done
    if [ -z "$sys_inc" ]; then
        echo "  host headers  : skipped, no rocthrust headers found under /opt/rocm" >&2
        echo "                  (HIP builds are unaffected; torch extension builds are not)" >&2
        return 0
    fi
    for dir in thrust rocprim hipcub; do
        [ -d "$sys_inc/$dir" ] || continue
        count_link "$core/include/$dir" "$sys_inc/$dir"
    done
    report_step "host headers"
}

# Activating the environment then points HIP_PATH/ROCM_PATH/HIP_CLANG_PATH at it
# and puts its lib directories in front of the system ROCm in LD_LIBRARY_PATH;
# deactivating restores whatever was there before.
write_conda_hooks() {
    local activate="$env_prefix/etc/conda/activate.d/$HOOK_NAME"
    local deactivate="$env_prefix/etc/conda/deactivate.d/$HOOK_NAME"
    local missing=0 hook
    if [ "$check_only" = yes ]; then
        for hook in "$activate" "$deactivate"; do
            if [ -f "$hook" ]; then
                continue
            fi
            echo "    missing $hook"
            missing=$((missing + 1))
        done
        if [ "$missing" = 0 ]; then
            echo "  conda hooks   : activate + deactivate in place"
        else
            echo "  conda hooks   : $missing missing"
        fi
        return 0
    fi
    mkdir -p "$(dirname "$activate")" "$(dirname "$deactivate")"
    cat >"$activate" <<'HOOK'
# Generated by tools/hip/setup_rocm_dev.sh — re-run it to refresh.
#
# The ROCm wheels here ship runtime + compiler only (the `rocm[devel]` package is
# not installed), so hipcc resolves its clang from HIP_CLANG_PATH/PATH, the device
# bitcode from ROCM_PATH, and a compiled binary resolves libamdhip64 at run time
# through /etc/ld.so.conf.d/rocm-bin.conf. Left alone, all three land on the system
# /opt/rocm (HIP 7.15) while this environment is 7.14.

_axl_rocm_core=""
for _axl_d in "$CONDA_PREFIX"/lib/python*/site-packages/_rocm_sdk_core; do
    [ -d "$_axl_d" ] && _axl_rocm_core="$_axl_d" && break
done

if [ -n "$_axl_rocm_core" ]; then
    _axl_rocm_libs="${_axl_rocm_core%/_rocm_sdk_core}/_rocm_sdk_libraries"

    export _AXL_OLD_HIP_PATH="${HIP_PATH-}"
    export _AXL_OLD_ROCM_PATH="${ROCM_PATH-}"
    export _AXL_OLD_HIP_CLANG_PATH="${HIP_CLANG_PATH-}"
    export _AXL_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

    export HIP_PATH="$_axl_rocm_core"
    export ROCM_PATH="$_axl_rocm_core"
    export HIP_CLANG_PATH="$_axl_rocm_core/lib/llvm/bin"
    export LD_LIBRARY_PATH="$_axl_rocm_core/lib:$_axl_rocm_libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

unset _axl_rocm_core _axl_rocm_libs _axl_d
HOOK
        cat >"$deactivate" <<'HOOK'
# Generated by tools/hip/setup_rocm_dev.sh — undo axl_rocm_hip_toolchain.sh.

if [ -n "${_AXL_OLD_HIP_PATH}" ]; then export HIP_PATH="${_AXL_OLD_HIP_PATH}"; else unset HIP_PATH; fi
if [ -n "${_AXL_OLD_ROCM_PATH}" ]; then export ROCM_PATH="${_AXL_OLD_ROCM_PATH}"; else unset ROCM_PATH; fi
if [ -n "${_AXL_OLD_HIP_CLANG_PATH}" ]; then export HIP_CLANG_PATH="${_AXL_OLD_HIP_CLANG_PATH}"; else unset HIP_CLANG_PATH; fi
if [ -n "${_AXL_OLD_LD_LIBRARY_PATH}" ]; then export LD_LIBRARY_PATH="${_AXL_OLD_LD_LIBRARY_PATH}"; else unset LD_LIBRARY_PATH; fi

unset _AXL_OLD_HIP_PATH _AXL_OLD_ROCM_PATH _AXL_OLD_HIP_CLANG_PATH _AXL_OLD_LD_LIBRARY_PATH
HOOK
    echo "  conda hooks   : activate + deactivate written"
}

# The env variables are set explicitly here so the check also works from a shell
# where the environment was never activated.
verify() {
    [ -f "$SMOKE_SRC" ] || die "smoke program not found at $SMOKE_SRC"
    local tmp_dir bin rc
    tmp_dir="$(mktemp -d)"
    bin="$tmp_dir/hip_smoke"

    echo
    echo "Building $SMOKE_SRC"
    if ! HIP_PATH="$core" ROCM_PATH="$core" HIP_CLANG_PATH="$core/lib/llvm/bin" \
        "$env_prefix/bin/hipcc" -O3 "$SMOKE_SRC" -o "$bin"; then
        die "hipcc failed"
    fi
    echo "Running $bin"
    rc=0
    LD_LIBRARY_PATH="$core/lib:$libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$bin" || rc=$?
    rm -f "$bin"
    rmdir "$tmp_dir" 2>/dev/null || true
    [ "$rc" = 0 ] || die "hip_smoke reported a failure (exit $rc)"
    echo "setup_rocm_dev.sh: hip_smoke passed"
}

# --- main ---------------------------------------------------------------------

env_prefix=""
check_only=no
run_verify=no

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --check) check_only=yes ;;
        --verify) run_verify=yes ;;
        -*) die "unknown option '$arg' (try -h)" ;;
        *)
            [ -z "$env_prefix" ] || die "only one environment prefix can be given"
            env_prefix="$arg"
            ;;
    esac
done

if [ -z "$env_prefix" ]; then
    env_prefix="${CONDA_PREFIX:-}"
fi
if [ -z "$env_prefix" ]; then
    env_name="$(env_name_from_yml || true)"
    base="$(conda_base || true)"
    if [ -n "$env_name" ] && [ -n "$base" ] && [ -d "$base/envs/$env_name" ]; then
        env_prefix="$base/envs/$env_name"
    fi
fi
[ -n "$env_prefix" ] || die "no environment given: activate it, pass a prefix, or create the one environment.yml names"
[ -d "$env_prefix" ] || die "no such directory: $env_prefix"

core=""
for dir in "$env_prefix"/lib/python*/site-packages/_rocm_sdk_core; do
    [ -d "$dir" ] && core="$dir" && break
done
[ -n "$core" ] || die "no _rocm_sdk_core under $env_prefix/lib/python*/site-packages — is this a ROCm environment?"
libs="${core%/_rocm_sdk_core}/_rocm_sdk_libraries"
[ -d "$libs" ] || die "no _rocm_sdk_libraries next to $core"

echo "setup_rocm_dev.sh: $env_prefix"
echo "  rocm core     : $core"
echo "  rocm libs     : $libs"
[ "$check_only" = yes ] && echo "  mode          : check only, nothing is written"

link_dev_libs
link_device_libs
link_host_headers
write_conda_hooks

if [ "$run_verify" = yes ]; then
    verify
elif [ "$check_only" = yes ]; then
    echo
    echo "Re-run without --check to apply, or with --verify to apply and build hip_smoke.hip."
else
    echo
    echo "Done. HIP.md documents the compile commands and what each variable does."
fi

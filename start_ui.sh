#!/bin/bash
set -euo pipefail

# Launch the AxlRanko desktop UI with the trainer's conda env on PATH.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_SH="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}"
if [ -z "${CONDA_SH:-}" ] || [ ! -f "$CONDA_SH" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    else
        for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
            if [ -f "$base/etc/profile.d/conda.sh" ]; then
                CONDA_SH="$base/etc/profile.d/conda.sh"
                break
            fi
        done
    fi
fi

if [ ! -f "${CONDA_SH:-}" ]; then
    echo "start_ui.sh: could not locate conda.sh; install conda or activate the 'axl' env manually." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate axl

# Ranko spawns api.py itself (and api.py spawns the trainer). Point it at this
# env's interpreter instead of whatever `python3` resolves to on PATH.
export AXL_PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT/ranko"
exec ./gradlew :desktopApp:run

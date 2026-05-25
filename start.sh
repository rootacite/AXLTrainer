#!/bin/bash
set -euo pipefail

export AMD_LOG_LEVEL=0
export CK_LOG_LEVEL=0

export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
export MIOPEN_LOG_LEVEL=1
export MIOPEN_LOG_BUFFER_SIZE=0

export MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS=0
export MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR=0
export MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK=0

export MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen"
export MIOPEN_USER_DB_PATH="$HOME/.config/miopen"

exec python -u train.py \
    1> >(grep -Ev "grid_desc|CandidateSelectionModel|metadata" >> /dev/null)

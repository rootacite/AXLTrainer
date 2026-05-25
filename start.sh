#!/bin/bash
set -euo pipefail

export AMD_LOG_LEVEL=0
export CK_LOG_LEVEL=0

export MIOPEN_LOG_LEVEL=1
unset MIOPEN_ENABLE_LOGGING
unset MIOPEN_ENABLE_LOGGING_CMD

export MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen"
export MIOPEN_USER_DB_PATH="$HOME/.config/miopen"

unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE

python train.py
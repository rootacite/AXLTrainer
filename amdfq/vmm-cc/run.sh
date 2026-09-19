#!/usr/bin/env bash
# Build and run the VMM reuse/integrity probe. Extra args go to the binary.
#
#   bash amdfq/vmm-cc/run.sh
#   bash amdfq/vmm-cc/run.sh --iters 800 --check-every 1
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hipcc --offload-arch=gfx1201 -O2 -Wall -std=c++17 "$here/vmm_cc.hip" -o "$here/vmm_cc"
exec "$here/vmm_cc" "$@"

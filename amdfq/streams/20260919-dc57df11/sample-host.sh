#!/usr/bin/env bash
#
# 4 Hz device-counter sampler: "<epoch>.<ns> <cardX vram_used> [<cardY vram_used> ...] <gtt_used ...>".
#
# This exists because the run this directory collects is meant to die: a faulted process leaves no
# /proc to read afterwards, so the only memory-account trace over the window is an external sampler.
# It reads sysfs only — no HIP context of its own, so it does not perturb what it measures.
#
#   bash sample-host.sh OUT.log [SECONDS]
set -u

out=${1:?sample-host.sh: need an output path}
seconds=${2:-180}

cards=()
for d in /sys/class/drm/card*/device; do
    [ -r "$d/mem_info_vram_used" ] || continue
    cards+=("$d")
done
if [ ${#cards[@]} -eq 0 ]; then
    echo "sample-host.sh: no /sys/class/drm/card*/device/mem_info_vram_used" >&2
    exit 2
fi

printf '## columns: time %s %s\n' \
    "$(printf 'vram_used(%s) ' "${cards[@]##*/device}")" \
    "$(printf 'gtt_used(%s) ' "${cards[@]##*/device}")" >>"$out"

end=$(( $(date +%s) + seconds ))
while [ "$(date +%s)" -lt "$end" ]; do
    line=$(date +%s.%N)
    for d in "${cards[@]}"; do line+=" $(cat "$d/mem_info_vram_used")"; done
    for d in "${cards[@]}"; do line+=" $(cat "$d/mem_info_gtt_used")"; done
    printf '%s\n' "$line" >>"$out"
    sleep 0.25
done

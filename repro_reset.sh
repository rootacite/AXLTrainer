#!/usr/bin/env bash
#
# Reproduce the machine's intermittent hard reset under load.
#
# What it replays: a short SDXL LoRA training run over a throwaway 40-image copy of the dataset on
# tmpfs (cold latent cache, so the run encodes during the window), with a load fired at the moment the
# validation sampling pass starts — the state the machine was in when it hard-reset twice on
# 2026-09-23. Around the run it records 1 Hz hardware state (temps, GPU power/util/VRAM, NVMe and
# network throughput, CPU) and every kernel line, fsynced per record, so a reset leaves the last
# second on disk.
#
#   ./repro_reset.sh                       # the crash-2 sequence: decrypt the archive bundles,
#                                          #   re-seal them (tar|zstd -19 -T0|gpg) and rsync them
#   ./repro_reset.sh --mode seal1          # the crash-1 sequence: re-seal + rsync only
#   ./repro_reset.sh --mode full           # compression + rsync in one pass
#   ./repro_reset.sh --mode none           # control: the sampling pass with no extra load
#   ./repro_reset.sh --mode rsync|zstd|gpg|localwrite   # one factor at a time
#   ./repro_reset.sh --repeats 10          # rate test: the same round ten times
#   AXL_PYTHON=$CONDA_PREFIX/bin/python ./repro_reset.sh    # interpreter with torch/accelerate
#
# Everything it writes goes to --out (default ~/axl-reset-repro/<UTC timestamp>): per round a
# timeline.log, recorder.csv (1 Hz hardware), kernel.log, train.out (the trainer's stdout, including
# the allocation patch's log), mirror/ (the run's own config.toml) and summary.txt.
#
# What it does NOT do: edit the repository's config.toml, write anything under output_dir that the
# repository's own runs use beyond one throwaway run directory per round, or touch archive/ — the
# bundles are only read, and the re-sealed copies go to the round directory.
#
# Two deliberate limitations, both noted in doc/hardware-reset-investigation.md:
#   * the allocation patch (LD_PRELOAD, knobs) is resolved from the repository's config.toml, because
#     trainer/amdfq_patch.py resolves the repo root from its own file, not from the working directory;
#     the mirror only overrides the trainer's own config. Set [environment].amdfq there if you want a
#     different patch setting.
#   * the load sizes are fixed (the sealed material, ~125 MB) so that rounds are comparable; adjust
#     --load-src and the tar path lists below if you want another shape.
set -uo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=seal2
repeats=1
out=""
load_src=""
witness_remote=""
subset_prefix="repro-subset"

usage() {
    sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) mode=${2:?}; shift 2 ;;
        --repeats) repeats=${2:?}; shift 2 ;;
        --out) out=${2:?}; shift 2 ;;
        --load-src) load_src=${2:?}; shift 2 ;;
        --witness-remote) witness_remote=${2:?}; shift 2 ;;
        --subset-prefix) subset_prefix=${2:?}; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "repro_reset.sh: unknown argument '$1'" >&2; usage 2 ;;
    esac
done

case $mode in
    none|rsync|zstd|gpg|localwrite|full|seal1|seal2) ;;
    *) echo "repro_reset.sh: unknown mode '$mode'" >&2; usage 2 ;;
esac

py=${AXL_PYTHON:-python3}
resolved=$(command -v "$py" || true)
[[ -n $resolved && -x $resolved ]] || { echo "repro_reset.sh: interpreter not found: $py" >&2; exit 2; }
py=$resolved
pybin=$(dirname "$py")
# The trainer runs through `start_train.sh`, which calls a bare `python`: the interpreter's directory
# is what goes on PATH below. It must be the environment with the training dependencies.
if ! "$py" -c 'import accelerate, torch, tensorboard' 2>/dev/null; then
    echo "repro_reset.sh: $py cannot import accelerate/torch/tensorboard." >&2
    echo "  Point AXL_PYTHON at the training environment's interpreter, e.g." >&2
    echo "  AXL_PYTHON=\$CONDA_PREFIX/bin/python bash repro_reset.sh ..." >&2
    exit 2
fi

out=${out:-$HOME/axl-reset-repro/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out" || exit 2
log=$out/repro.log
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$log"; }

state=/run/user/1000/axltrainer/state.json
train_data_dir=$("$py" - "$repo/config.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    print(tomllib.load(fh)["environment"]["train_data_dir"])
PY
) || { echo "repro_reset.sh: cannot read train_data_dir from config.toml" >&2; exit 2; }

subset=${TMPDIR:-/tmp}/$subset_prefix
say "repo=$repo kernel=$(uname -r) mode=$mode repeats=$repeats"
say "out=$out"
say "dataset source=$train_data_dir -> tmpfs subset=$subset (40 images, cold cache)"
say "patch env per repo config: $(cd "$repo" && "$py" -c 'from trainer.amdfq_patch import launch_env_line; print(launch_env_line())')"

# The 1 Hz recorder: hardware state plus the kernel log, fsynced per record.
cat > "$out/recorder.py" <<'PY'
import os, subprocess, sys, threading, time

out = sys.argv[1]
remote = sys.argv[2] if len(sys.argv) > 2 else ""
HW = {
    "pkg": "/sys/class/hwmon/hwmon6/temp1_input",
    "edge": "/sys/class/hwmon/hwmon2/temp1_input",
    "junction": "/sys/class/hwmon/hwmon2/temp2_input",
    "gpu_power_uw": "/sys/class/hwmon/hwmon2/power1_average",
    "nvme": "/sys/class/hwmon/hwmon1/temp1_input",
}
GPU = "/sys/class/drm/card1/device"


def read(path):
    try:
        with open(path) as fh:
            return float(fh.read().strip())
    except Exception:
        return 0.0


def cpu_busy_reader():
    def total_idle():
        with open("/proc/stat") as fh:
            v = [int(x) for x in fh.readline().split()[1:]]
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)
    prev = total_idle()

    def busy():
        nonlocal prev
        cur = total_idle()
        dt = max(1, cur[0] - prev[0])
        b = 100.0 * (1 - (cur[1] - prev[1]) / dt)
        prev = cur
        return b
    return busy


def counts():
    out = {}
    for e in os.listdir("/proc"):
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/comm") as fh:
                c = fh.read().strip()
        except Exception:
            continue
        if c.startswith(("zstd", "gpg", "tar", "rsync")):
            out[c] = out.get(c, 0) + 1
    return out


def trainer_pid():
    try:
        import json
        with open("/run/user/1000/axltrainer/state.json") as fh:
            pid = json.load(fh)["pid"]
        os.kill(pid, 0)
        return pid
    except Exception:
        return ""


busy = cpu_busy_reader()
sinks = []
for path in [os.path.join(out, "recorder.csv")] + ([os.path.join(remote, "recorder.csv")] if remote else []):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sinks.append(open(path, "a", buffering=1))
    except OSError:
        pass
if sinks:
    sinks[0].write("iso,epoch,load1,cpu_busy_pct,memavail_gb,pkg_c,edge_c,junction_c,gpu_power_w,nvme_c,"
                   "gpu_busy_pct,vram_used_mb,gtt_used_mb,nvme_write_mb,net_tx_mb,zstd,gpg,tar,rsync,trainer_pid\n")


def emit(line):
    for fh in sinks:
        try:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        except OSError:
            pass


def hardware():
    while True:
        mem = ""
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                mem = f"{int(line.split()[1]) / 1048576:.1f}"
        try:
            load1 = open("/proc/loadavg").read().split()[0]
        except OSError:
            load1 = ""
        c = counts()
        row = [
            time.strftime("%H:%M:%S"), f"{time.time():.0f}", load1, f"{busy():.1f}", mem,
            f"{read(HW['pkg']) / 1000:.1f}", f"{read(HW['edge']) / 1000:.1f}",
            f"{read(HW['junction']) / 1000:.1f}", f"{read(HW['gpu_power_uw']) / 1e6:.1f}",
            f"{read(HW['nvme']) / 1000:.1f}",
            f"{read(GPU + '/gpu_busy_percent'):.0f}",
            f"{read(GPU + '/mem_info_vram_used') / 1048576:.0f}",
            f"{read(GPU + '/mem_info_gtt_used') / 1048576:.0f}",
            "0.0", "0.0",
            c.get("zstd", 0), c.get("gpg", 0), c.get("tar", 0), c.get("rsync", 0), trainer_pid(),
        ]
        emit(",".join(str(x) for x in row))
        time.sleep(1.0)


threading.Thread(target=hardware, daemon=True).start()
proc = subprocess.Popen(["journalctl", "-k", "-f", "-o", "short-precise", "--no-pager"],
                        stdout=subprocess.PIPE, text=True, bufsize=1)
while True:
    line = proc.stdout.readline()
    if line:
        klog = os.path.join(out, "kernel.log")
        with open(klog, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line.rstrip()}\n")
            fh.flush()
            os.fsync(fh.fileno())
    else:
        time.sleep(0.05)
PY

build_subset() {
    local n=0
    mkdir -p "$subset"
    while IFS= read -r img; do
        ln -sf "$img" "$subset/$(basename "$img")"
        local stem=${img%.*}
        [[ -f $stem.txt ]] && ln -sf "$stem.txt" "$subset/$(basename "$stem").txt"
        [[ -f $stem.mask.png ]] && ln -sf "$stem.mask.png" "$subset/$(basename "$stem").mask.png"
        n=$((n + 1))
    done < <(find "$train_data_dir" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) \
        ! -iname '*.mask.png' | sort | head -40)
    say "subset: $n images + captions at $subset"
    rm -rf "$subset/.latents_cache"
}

make_mirror() {
    local dir=$1
    mkdir -p "$dir"
    ln -sfn "$repo/trainer" "$dir/trainer"
    ln -sfn "$repo/text_processing.py" "$dir/text_processing.py"
    ln -sfn "$repo/start_train.sh" "$dir/start_train.sh"
    "$py" - "$repo/config.toml" "$dir/config.toml" "$subset" <<'PY'
import sys
src, dst, subset = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()
for old, new in (('train_data_dir = "', 'train_data_dir = "X'), ):
    pass
import re
text = re.sub(r'train_data_dir = "[^"]*"', f'train_data_dir = "{subset}"', text, count=1)
text = re.sub(r'^epoch = \d+', 'epoch = 1', text, count=1, flags=re.M)
text = re.sub(r'^save_every_n_steps = \d+', 'save_every_n_steps = 10', text, count=1, flags=re.M)
open(dst, "w").write(text)
PY
}

load_round() {
    local dir=$1 kind=$2
    case $kind in
        none) sleep 40 ;;
        rsync)
            mkdir -p "$dir/nfs"
            ( cd "$load_src" && rsync -aR --checksum . "$dir/nfs"/ ) >"$dir/load.log" 2>&1 ;;
        zstd)
            tar -C "$load_src" -cf - . | zstd -19 -T0 >/dev/null 2>"$dir/load.log" ;;
        gpg)
            tar -C "$load_src" -cf - . | zstd -1 | gpg --batch --yes --symmetric --cipher-algo AES256 \
                --passphrase-file "$keyfile" -o /dev/null 2>"$dir/load.log" ;;
        localwrite)
            tar -C "$load_src" -cf - . | zstd -19 -T0 >"$dir/local.zst" 2>"$dir/load.log" ;;
        full)
            tar -C "$load_src" -cf - . | zstd -19 -T0 | gpg --batch --yes --symmetric --cipher-algo AES256 \
                --passphrase-file "$keyfile" -o "$dir/full.gpg" 2>"$dir/load.log"
            mkdir -p "$dir/nfs"; ( cd "$load_src" && rsync -aR --checksum . "$dir/nfs"/ ) >>"$dir/load.log" 2>&1 ;;
        seal1|seal2) seal_round "$dir" "$kind" ;;
    esac
}

seal_round() {
    local dir=$1 kind=$2 stage=$1/stage
    if [[ $kind == seal2 ]]; then
        mkdir -p "$stage"
        for b in amdfq-pocs-2026-09 gfx1201-disclosure-2026-09 gfx1201-research-2026-09; do
            gpg --batch --quiet --decrypt --passphrase-file "$keyfile" "$repo/archive/$b.tar.zst.gpg" \
                | zstd -d | tar -x -C "$stage" 2>>"$dir/load.log"
        done
    else
        stage=$load_src
    fi
    local pocs="amdfq/amdfq-tail amdfq/vmm-cc amdfq/vmm-ru amdfq/vmm-ru.oneshot-wip amdfq/fk-vmm amdfq/replay amdfq/streams amdfq/eva-2.md"
    local disclosure="amdfq/final amdfq/doc/psirt"
    local research="--exclude=amdfq/doc/psirt amdfq/doc conclusions fixes tools/hip HIP.md"
    local name paths
    for name in pocs disclosure research; do
        eval "paths=\$$name"
        # shellcheck disable=SC2086
        tar -C "$stage" -cf - $paths | zstd -19 -T0 \
            | gpg --batch --yes --symmetric --cipher-algo AES256 --passphrase-file "$keyfile" \
                  -o "$dir/sealed-$name.gpg" 2>>"$dir/load.log"
    done
    mkdir -p "$dir/nfs"
    ( cd "$stage" && rsync -aR --checksum . "$dir/nfs"/ ) >>"$dir/load.log" 2>&1
}

keyfile=$HOME/.axl-archive-key
[[ -f $keyfile ]] || { echo "repro_reset.sh: passphrase file $keyfile missing (needed by seal1/seal2/gpg)" >&2; exit 2; }

build_subset

for round in $(seq 1 "$repeats"); do
    dir=$out/round-$round
    mkdir -p "$dir"
    say "== round $round/$repeats ($mode) in $dir"
    for _ in $(seq 1 20); do pgrep -f 'trainer/main.py' >/dev/null || break; sleep 2; done
    if pgrep -f 'trainer/main.py' >/dev/null; then say "ABORT: a trainer is already running"; exit 1; fi

    if [[ -z $load_src && $mode != none ]]; then
        load_src=$dir/load-src
        if [[ ! -d $load_src/amdfq ]]; then
            mkdir -p "$load_src"
            for b in amdfq-pocs-2026-09 gfx1201-disclosure-2026-09 gfx1201-research-2026-09; do
                gpg --batch --quiet --decrypt --passphrase-file "$keyfile" "$repo/archive/$b.tar.zst.gpg" \
                    | zstd -d | tar -x -C "$load_src" || say "  (decrypt of $b failed)"
            done
        fi
        say "load source: $load_src ($(du -sh "$load_src" | cut -f1))"
    fi

    make_mirror "$dir/mirror"
    setsid "$py" "$out/recorder.py" "$dir" "$witness_remote" >"$dir/recorder.err" 2>&1 &
    recorder_pid=$!
    echo "$recorder_pid" > "$dir/recorder.pid"
    sleep 2

    # The runtime state file is shared with the dashboard and may still describe the previous run, so
    # a sample only counts once it names a run other than the one already on disk.
    prev_run=$("$py" - "$state" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("run_id") or "")
except Exception:
    print("")
PY
)
    ( cd "$dir/mirror" && setsid env PATH="$pybin:$PATH" AMDFQ_LOG_LEVEL=info bash start_train.sh \
        >"$dir/train.out" 2>&1 & )
    say "training run started (previous run_id on disk: ${prev_run:-none})"

    active=False
    status=pending
    for _ in $(seq 1 90); do
        sleep 2
        read -r run_id status step active <<<"$("$py" - "$state" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1])); s = d.get("sampling") or {}
    print(d.get("run_id") or "-", d["status"], (d.get("training") or {}).get("step"), s.get("active"))
except Exception:
    print("- none 0 False")
PY
)"
        if [[ $run_id == "$prev_run" ]]; then
            say "  (state still names the previous run; waiting)"
            continue
        fi
        say "  run: status=$status step=$step sampling_active=$active"
        [[ $active == True || $status == finished || $status == error ]] && break
    done
    [[ $status == pending ]] && say "no state from the new run within 180 s"

    if [[ ${active:-False} == True ]]; then
        say "SAMPLING ONSET -> firing load: $mode"
        t0=$(date +%s)
        load_round "$dir" "$mode"
        say "load finished in $(( $(date +%s) - t0 ))s"
    else
        say "no sampling pass seen (status=$status); no load fired"
    fi

    for _ in $(seq 1 120); do
        sleep 3
        st=$("$py" - "$state" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["status"])
except Exception:
    print("gone")
PY
)
        [[ $st == finished || $st == error || $st == gone ]] && { say "run ended: $st"; break; }
    done

    kill "$(cat "$dir/recorder.pid" 2>/dev/null)" 2>/dev/null
    "$py" - "$dir/recorder.csv" "$dir/summary.txt" "$mode" <<'PY'
import csv, sys
src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(src)))
peak = lambda c: max((float(r[c]) for r in rows if r.get(c) not in ("", None)), default=0.0)
with open(dst, "w") as fh:
    fh.write(f"mode={mode} rows={len(rows)}\n")
    fh.write(f"peak_gpu_busy_pct={peak('gpu_busy_pct'):.0f}\n")
    fh.write(f"peak_gpu_power_w={peak('gpu_power_w'):.0f}\n")
    fh.write(f"peak_vram_mb={peak('vram_used_mb'):.0f}\n")
    fh.write(f"peak_junction_c={peak('junction_c'):.0f}\n")
    fh.write(f"peak_pkg_c={peak('pkg_c'):.0f}\n")
    fh.write(f"peak_cpu_busy_pct={peak('cpu_busy_pct'):.1f}\n")
    fh.write(f"last_row={rows[-1]['iso'] if rows else 'none'}\n")
print(open(dst).read())
PY
done

say "done: $repeats round(s), out=$out"
say "read a round's evidence: timeline is repro.log + round-N/recorder.csv (1 Hz) + round-N/kernel.log"
say "a hard reset looks like: recorder.csv stops mid-second, kernel.log has no fault line, and the"
say "next boot reports 'recovering journal' (unclean shutdown) in journalctl -b 0"

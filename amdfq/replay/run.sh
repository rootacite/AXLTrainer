#!/usr/bin/env bash
#
# Builds the replay program, runs it against one collected stream in the foreground, and checks the
# replay's own accounting against the hook log line by line.
#
#   bash amdfq/replay/run.sh [STREAM_DIR] [extra replay args...]
#
# STREAM_DIR defaults to the newest amdfq/streams/*/ that has an ops.tsv.
#
# WHAT REACHES THE TERMINAL.  The replay writes its per-op output (thousands of lines) to
# <STREAM_DIR>/replay.out and its status to stderr; this script lets the status through and nothing
# else, so the terminal shows the account heartbeat — `ledger free vram`, `driver free vram` and
# their difference — and a few lines of its own.  Nothing else.
#
# WHERE IT STOPS.  When the stream is done the replay keeps its allocations and waits.  It runs in
# the foreground here, so the terminal's Ctrl+C goes straight to it: the replay releases everything,
# prints one line, exits 0, and only then does this script print the check.  There is no process for
# this script to leave behind, and none for it to kill by hand — the signal path is the terminal's,
# not a shell trap's.
set -u

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)

stream=${1:-}
if [ -n "$stream" ]; then shift; else
    stream=$(ls -dt "$repo"/amdfq/streams/*/ 2>/dev/null | while read -r d; do
        [ -f "$d/ops.tsv" ] && { echo "$d"; break; }
    done)
fi
[ -n "$stream" ] && [ -d "$stream" ] || { echo "run.sh: no stream directory with an ops.tsv found" >&2; exit 2; }
stream=${stream%/}
out=$stream/replay.out
err=$stream/replay.err

echo "run.sh    stream $stream"
hipcc --offload-arch=gfx1201 -O2 -Wall -std=c++17 "$here/replay.cpp" -o "$here/replay" || exit 2

# One replay holds ~11.7 GB, so a second one started while the first holds cannot allocate (its
# heartbeats show `ledger free vram 0 B`, and the two would write into the same replay.out).
stale=$(pgrep -x replay | tr '\n' ' ')
if [ -n "$stale" ]; then
    echo "run.sh: a replay is already running ($stale); end it first (kill -INT <pid>)" >&2
    exit 2
fi

echo "run.sh    running the replay; its status is on stderr, its op log in $out"
echo "run.sh    the run holds its allocations at the end — press Ctrl+C to end it"

# The terminal's Ctrl+C belongs to the replay (same foreground process group); this script rides it
# out and prints the check afterwards.
trap '' INT
"$here/replay" "$stream/ops.tsv" ${args[@]+"${args[@]}"} > "$out" 2> >(tee "$err" >&2)
rc=$?
trap - INT

echo "run.sh    replay exit $rc, $(wc -l < "$out") lines in $out"
left=$(pgrep -x replay | tr '\n' ' ')
if [ -n "$left" ]; then
    echo "run.sh: WARNING a replay survived the run ($left) and still holds its allocations" >&2
fi

python3 - "$stream/frozen.log" "$out" "$err" <<'PY'
import re, sys

PREFIX = re.compile(r"^\d+\s+[\d.]+\s+T=\d+\s+")
FIELDS = re.compile(r"live=(\d+) live_bytes=(\d+)")

def stream(path):
    out = []
    for line in open(path, errors="replace"):
        m = PREFIX.match(line)
        if m is None:
            continue
        f = FIELDS.search(line)
        if f is None:
            continue
        out.append((int(f.group(1)), int(f.group(2))))
    return out

hook, replay = stream(sys.argv[1]), stream(sys.argv[2])
status = open(sys.argv[3], errors="replace").read() if len(sys.argv) > 3 else ""
print("check     hook %d ops, replay %d ops" % (len(hook), len(replay)))
bad_prefix = [i for i, (h, r) in enumerate(zip(hook, replay)) if h != r]
if "interrupted by signal" in status and len(replay) < len(hook):
    print("check     Ctrl+C at op %d of %d; the prefix %s" %
          (len(replay), len(hook),
           "agrees op for op" if not bad_prefix
           else "does NOT agree (first at op %d)" % (bad_prefix[0] + 1)))
    sys.exit(0 if not bad_prefix else 1)
if not replay:
    print("check     skipped: replay.out has no per-op lines -- --quiet suppresses them, and this\n"
          "          op-for-op check reads exactly those lines.  Drop --quiet to run the check.")
    sys.exit(3)
bad = [i for i, (h, r) in enumerate(zip(hook, replay)) if h != r]
print("check     mismatch %d%s" % (len(bad), "" if not bad else " (first at op %d: hook %s, replay %s)"
                                   % (bad[0] + 1, hook[bad[0]], replay[bad[0]])))
if len(hook) != len(replay):
    print("check     LENGTH DIFFERS by %d ops" % (len(hook) - len(replay)))
print("check     %s" % ("accounting agrees op for op" if not bad and len(hook) == len(replay)
                        else "DOES NOT AGREE"))
PY
check_rc=$?
if [ "$rc" -eq 0 ]; then rc=$check_rc; fi
exit $rc

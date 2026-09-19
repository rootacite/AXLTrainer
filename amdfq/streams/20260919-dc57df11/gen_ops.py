#!/usr/bin/env python3
"""Turn one amdfq hook log into the op table the replay program reads.

    python3 gen_ops.py --log frozen.log --out ops.tsv [--tids A,B] [--cut N] [--json]

The log line shapes are the ones amdfq/amdfq-tail writes (amdfq_hooks_hip.c for the three gates,
amdfq_log.c for the "<seq> <elapsed_s> T=<tid> " prefix).  Comments carry the prefix too, so the
prefix is what decides a line's shape:

    000001     0.000012 T=405921 # amdfq preload, pid 405921 ppid 404874, log /tmp/frozen.log
    002231    12.345678 T=406787 hipMalloc(size=2097152) -> ptr=0x7f49d7c00000 ret=0 live=1217 live_bytes=9657384960 caller=libc10_hip.so+0x41efa
    002232    12.345690 T=406787 hipFree(ptr=0x7f49d7c00000) -> ret=0 live=1216 live_bytes=9655287808 caller=libc10_hip.so+0x41efa
    002233    17.02  T=406641 hipHostMalloc(size=4096 flags=0x0) -> ptr=0x7f49d7800000 ret=0 caller=libc10_hip.so+0x41efa

ONE LOG, SEVERAL PROCESSES, SEVERAL THREADS.  Every process that inherits LD_PRELOAD writes to the
same file: the trainer, the orphan reaper, the DataLoader forkserver, and any helper it spawns.  A
process's threads share the hook's atomic counter, so their seq values interleave and never collide;
a process that started fresh (execve) numbers its own lines from 1 and collides; a process that
forked from the trainer continues its own copy of the counter and collides too.  The stream worth
replaying is one process's — all of its threads, none of the helper processes — so the tids are
grouped by that rule: the group seeded from the tid with the most hipMalloc calls, extended with
every other tid whose seq values are disjoint from the group's (i.e. shares the same counter).

Table format (one op per line, tab separated):

    seq  kind  size  slot  flags  ret  t

`t` is the hook's own clock for that op (seconds since the run's first log line), so the replay can
put its calls on the trainer's timeline instead of running them all at once.  `kind` is M (hipMalloc),
H (hipHostMalloc) or F (hipFree).
For M/H, `slot` is the id the op itself gets; for F it is the slot the freed pointer belongs to, and
`size` is that block's request size.
`flags` is only meaningful for H, `ret` is the return code the hook logged.  Nothing in the table is
a raw address: the replay builds its own pointer for each slot, so the table stays portable and can
be replayed with a pad the original run did not have.

The script also does the accounting the stream has to pass before it is worth replaying: parse
residuals, the live table at the cut against the hook's own last live=/live_bytes=, whether the hook
ran any allocation of its own outside the three gates, and how the collection says the run ended.
What it prints is meant to be pasted into summary.md.
"""

import argparse
import hashlib
import json
import os
import re
import sys

PREFIX = re.compile(r"^(?P<seq>\d+)\s+(?P<t>\d+\.\d+)\s+T=(?P<tid>\d+)\s+(?P<body>.*)$")
COMMENT = re.compile(r"^#")
BANNER = re.compile(r"^# amdfq preload, pid (?P<pid>\d+) ppid (?P<ppid>\d+)")
MALLOC = re.compile(
    r"^hipMalloc\(size=(?P<size>\d+)\) -> ptr=(?P<ptr>0x[0-9a-f]+|\(nil\)) ret=(?P<ret>-?\d+) "
    r"live=(?P<live>\d+) live_bytes=(?P<live_bytes>\d+) caller=(?P<caller>.*)$"
)
FREE = re.compile(
    r"^hipFree\(ptr=(?P<ptr>0x[0-9a-f]+|\(nil\))\) -> ret=(?P<ret>-?\d+) "
    r"live=(?P<live>\d+) live_bytes=(?P<live_bytes>\d+) caller=(?P<caller>.*)$"
)
HOST = re.compile(
    r"^hipHostMalloc\(size=(?P<size>\d+) flags=(?P<flags>0x[0-9a-f]+)\) -> "
    r"ptr=(?P<ptr>0x[0-9a-f]+|\(nil\)) ret=(?P<ret>-?\d+) caller=(?P<caller>.*)$"
)


def classify(body):
    """(kind, fields) for one line body, or (None, None) when nothing matches."""
    if COMMENT.match(body):
        return "comment", body
    if (m := MALLOC.match(body)) is not None:
        return "M", m
    if (m := FREE.match(body)) is not None:
        return "F", m
    if (m := HOST.match(body)) is not None:
        return "H", m
    return None, None


def scan(path):
    """Pass one: every prefixed line, grouped per tid."""
    rows = []            # (tid, seq, t, kind, fields, line)
    per_tid = {}         # tid -> stats
    unprefixed = []
    for raw in open(path, "r", errors="replace"):
        line = raw.rstrip("\n")
        if not line:
            continue
        m = PREFIX.match(line)
        if m is None:
            unprefixed.append(line)
            continue
        tid, seq, t = int(m.group("tid")), int(m.group("seq")), float(m.group("t"))
        kind, fields = classify(m.group("body"))
        entry = per_tid.setdefault(tid, {
            "lines": 0, "M": 0, "F": 0, "H": 0, "comment": 0, "other": 0,
            "seq": set(), "first_t": t, "last_t": t, "first_seq": seq, "last_seq": seq,
            "banner_pid": None,
        })
        entry["lines"] += 1
        entry["last_t"], entry["last_seq"] = t, seq
        entry["seq"].add(seq)
        entry["comment" if kind == "comment" else kind if kind else "other"] += 1
        if kind == "comment" and (b := BANNER.match(fields)) is not None:
            entry["banner_pid"] = int(b.group("pid"))
        rows.append((tid, seq, t, kind, fields, line))
    return rows, per_tid, unprefixed


def pick_stream(per_tid, override=None):
    """The tids of one process: the seeded tid plus every tid that shares its counter.

    Threads share the hook's counter, so their seq values are disjoint.  A helper process numbers
    from 1 (fresh execve) or continues its own copy of the counter (fork), and in both cases its
    numbers collide with the seed's — which is what keeps it out.
    """
    if override:
        return sorted(override), {tid: "named on the command line" for tid in override}

    def hip(tid):
        e = per_tid[tid]
        return e["M"] + e["F"] + e["H"]

    ranked = sorted((tid for tid in per_tid if hip(tid) > 0), key=lambda tid: -hip(tid))
    if not ranked:
        return [], {}
    seed = ranked[0]
    chosen = [seed]
    taken = set(per_tid[seed]["seq"])
    excluded = {}
    for tid in ranked[1:]:
        clash = taken & per_tid[tid]["seq"]
        if clash:
            excluded[tid] = ("shares no counter with T=%d (seq %d collides: its own numbering or a "
                             "fork copy)" % (seed, min(clash)))
        else:
            chosen.append(tid)
            taken |= per_tid[tid]["seq"]
    for tid in per_tid:
        if tid not in chosen and tid not in excluded:
            excluded[tid] = "no hipMalloc/hipFree/hipHostMalloc in this process"
    return sorted(chosen), excluded


def build(rows, tids):
    """Pass two: the chosen tids' lines -> ops, with the hook's own accounting mirrored."""
    ops = []
    live = {}        # hipMalloc pointer -> (slot, size); what live=/live_bytes= counts
    allocs = {}      # hipFree lookup: M and H pointers -> (slot, size)
    slots = {}
    state = {
        "tids": sorted(tids),
        "ops": 0,
        "counts": {"M": 0, "F": 0, "H": 0},
        "per_tid_ops": {},
        "comments": [],
        "unmatched_frees": 0,
        "duplicate_pointers": 0,
        "malloc_failures": 0,
        "free_failures": 0,
        "last_live": None,
        "last_live_bytes": None,
        "first_t": None,
        "last_t": None,
        "first_seq": None,
        "last_seq": None,
        "stray_lines": [],
    }

    for tid, seq, t, kind, fields, line in rows:
        if tid not in tids:
            continue
        if state["first_t"] is None:
            state["first_t"], state["first_seq"] = t, seq
        state["last_t"], state["last_seq"] = t, seq

        if kind == "comment":
            state["comments"].append(line.split(" T=%d " % tid, 1)[-1][:160])
            continue
        if kind is None:
            state["stray_lines"].append(line[:160])
            continue

        if kind == "F":
            ret = int(fields.group("ret"))
            state["last_live"] = int(fields.group("live"))
            state["last_live_bytes"] = int(fields.group("live_bytes"))
            state["counts"]["F"] += 1
            state["ops"] += 1
            state["per_tid_ops"][tid] = state["per_tid_ops"].get(tid, 0) + 1
            ptr = fields.group("ptr")
            target = allocs.pop(ptr, None)
            if target is None:
                state["unmatched_frees"] += 1
                ops.append({"seq": seq, "kind": "F", "size": 0, "slot": -1, "flags": 0,
                            "ret": ret, "t": t})
            else:
                if ret != 0:
                    state["free_failures"] += 1
                target_slot, target_size = target
                live.pop(ptr, None)
                ops.append({"seq": seq, "kind": "F", "size": target_size, "slot": target_slot,
                            "flags": 0, "ret": ret, "t": t})
            continue

        size = int(fields.group("size"))
        ret = int(fields.group("ret"))
        state["counts"][kind] += 1
        state["ops"] += 1
        state["per_tid_ops"][tid] = state["per_tid_ops"].get(tid, 0) + 1
        if kind == "M":
            state["last_live"] = int(fields.group("live"))
            state["last_live_bytes"] = int(fields.group("live_bytes"))
        slot = len(slots)
        slots[slot] = size
        ptr = fields.group("ptr")
        if ret == 0 and ptr != "(nil)":
            if ptr in allocs:
                state["duplicate_pointers"] += 1
            allocs[ptr] = (slot, size)
            if kind == "M":
                live[ptr] = (slot, size)
        else:
            state["malloc_failures"] += 1
        flags = int(fields.group("flags"), 16) if kind == "H" else 0
        ops.append({"seq": seq, "kind": kind, "size": size, "slot": slot, "flags": flags,
                    "ret": ret, "t": t})

    state["live_blocks"] = len(live)
    state["live_bytes"] = sum(size for _, size in live.values())
    state["live_sizes_sha1"] = size_multiset_sha1([size for _, size in live.values()])
    return ops, state


def size_multiset_sha1(sizes):
    """sha1 over the live set's sizes as sorted "<size> x<count>" lines.

    The recipe is this script's own: the old experiment recorded a hash (amdfq/doc/live-data.md
    §9.3 B, 6d7c7d7804352d0e) without its recipe, so that value can only be compared on counts and
    byte totals, not recomputed.  This one is comparable to future runs of this script.
    """
    counts = {}
    for size in sizes:
        counts[size] = counts.get(size, 0) + 1
    blob = "\n".join("%d x%d" % (size, counts[size]) for size in sorted(counts))
    return hashlib.sha1(blob.encode()).hexdigest()


def state_after(ops, count):
    """Replays the first `count` ops: counts, live table and multiset hash at that point."""
    live, allocs = {}, {}
    for op in ops[:count]:
        if op["kind"] in "MH":
            if op["ret"] == 0:
                allocs[op["slot"]] = op["size"]
                if op["kind"] == "M":
                    live[op["slot"]] = op["size"]
        else:
            allocs.pop(op["slot"], None)
            live.pop(op["slot"], None)
    return {
        "ops": min(count, len(ops)),
        "counts": {kind: sum(1 for op in ops[:count] if op["kind"] == kind) for kind in "MFH"},
        "live_blocks": len(live),
        "live_bytes": sum(live.values()),
        "live_sizes_sha1": size_multiset_sha1(live.values()),
        "t": ops[min(count, len(ops)) - 1]["t"] if count else 0.0,
    }


def index_of_malloc(ops, nth):
    seen = 0
    for i, op in enumerate(ops):
        if op["kind"] == "M":
            seen += 1
            if seen == nth:
                return i + 1
    return None


def index_at_time(ops, seconds):
    count = 0
    for op in ops:
        if op["t"] > seconds:
            break
        count += 1
    return count


def run_signature(directory):
    """What the collection recorded about how the run ended (crash.txt next to the log)."""
    path = os.path.join(directory, "crash.txt")
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    fields = {}
    for key, pattern in (("exit_code", r"exit code\s+(-?\d+)"),
                         ("wall_seconds", r"wall seconds\s+([\d.]+)"),
                         ("status", r'"status":\s*"([^"]+)"'),
                         ("step", r'"step":\s*(\d+)'),
                         ("memory_fault", r"(HSA_STATUS_ERROR_MEMORY_FAULT)"),
                         ("live_snapshot_taken", r"(frozen-snap/)"),
                         ("killed_by_hand", r"kill -KILL")):
        if (m := re.search(pattern, text)) is not None:
            fields[key] = m.group(1)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", default="frozen.log", help="the hook log to read")
    parser.add_argument("--out", default="ops.tsv", help="where to write the op table")
    parser.add_argument("--tids", default=None,
                        help="comma separated tids to take the stream from (default: auto)")
    parser.add_argument("--cut", type=int, default=None,
                        help="also report the state after this many ops (old stream: 4853)")
    parser.add_argument("--cut-malloc", type=int, default=None,
                        help="also report the state right after the Nth hipMalloc (old stream: 2994)")
    parser.add_argument("--cut-time", type=float, default=None,
                        help="also report the state at the last op with t <= this (old stream: 19.859)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    rows, per_tid, unprefixed = scan(args.log)
    directory = os.path.dirname(os.path.abspath(args.log))
    override = [int(part) for part in args.tids.split(",")] if args.tids else None
    tids, excluded = pick_stream(per_tid, override)
    if not tids:
        print("gen_ops: no hipMalloc/hipFree/hipHostMalloc lines in %s" % args.log, file=sys.stderr)
        return 2
    ops, state = build(rows, set(tids))

    with open(args.out, "w") as handle:
        handle.write("# amdfq op table (gen_ops.py) from %s, stream of T=%s\n" %
                     (args.log, ",".join(str(t) for t in tids)))
        handle.write("# seq<TAB>kind<TAB>size<TAB>slot<TAB>flags<TAB>ret<TAB>t\n")
        handle.write("# kind M = hipMalloc, H = hipHostMalloc, F = hipFree(slot)\n")
        handle.write("# for M/H slot is the op's own id; for F it is the slot that was freed\n")
        handle.write("# t = the hook's clock for that op, in seconds since the run's first log line\n")
        for op in ops:
            handle.write("%d\t%s\t%d\t%d\t0x%x\t%d\t%.6f\n" %
                         (op["seq"], op["kind"], op["size"], op["slot"], op["flags"], op["ret"],
                          op["t"]))

    cut_checks = []
    for label, count in (("ops=%s" % args.cut, args.cut),
                         ("malloc=#%s" % args.cut_malloc,
                          index_of_malloc(ops, args.cut_malloc) if args.cut_malloc else None),
                         ("t<=%ss" % args.cut_time,
                          index_at_time(ops, args.cut_time) if args.cut_time is not None else None)):
        if count:
            cut = state_after(ops, count)
            cut["label"] = label
            cut_checks.append(cut)
    report = {
        "log": args.log,
        "stream_tids": tids,
        "per_tid": {tid: {k: v for k, v in e.items() if k != "seq"} for tid, e in per_tid.items()},
        "excluded_tids": excluded,
        "unprefixed_lines": unprefixed[:5],
        "unprefixed_count": len(unprefixed),
        "ops": state["ops"],
        "op_counts": state["counts"],
        "per_tid_ops": state["per_tid_ops"],
        "window": {"first_seq": state["first_seq"], "last_seq": state["last_seq"],
                   "first_t": state["first_t"], "last_t": state["last_t"]},
        "comments": state["comments"],
        "stray_lines": state["stray_lines"][:5],
        "live_blocks": state["live_blocks"],
        "live_bytes": state["live_bytes"],
        "live_sizes_sha1": state["live_sizes_sha1"],
        "last_live": state["last_live"],
        "last_live_bytes": state["last_live_bytes"],
        "residuals": {"unmatched_frees": state["unmatched_frees"],
                      "duplicate_pointers": state["duplicate_pointers"],
                      "malloc_failures": state["malloc_failures"],
                      "free_failures": state["free_failures"]},
        "run": run_signature(directory),
        "cut_checks": cut_checks,
    }
    report["verdict"] = {
        "parses_clean": not state["stray_lines"] and not unprefixed,
        "no_unmatched_frees": state["unmatched_frees"] == 0,
        "no_duplicate_pointers": state["duplicate_pointers"] == 0,
        "no_hook_own_ops": not any("falling back to pad" in c for c in state["comments"]),
        "live_matches_last_line": state["live_blocks"] == state["last_live"],
        "live_bytes_matches_last_line": state["live_bytes"] == state["last_live_bytes"],
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if all(v is not False for v in report["verdict"].values()) else 1

    print("log            %s (%d lines, %d without a prefix)" %
          (args.log, len(rows) + len(unprefixed), len(unprefixed)))
    print("tids in log    ")
    for other in sorted(per_tid, key=lambda tid: -per_tid[tid]["lines"]):
        e = per_tid[other]
        mark = " <- stream" if other in tids else " (%s)" % excluded.get(other, "")
        print("  T=%-8d %6d lines  hipMalloc %5d, hipFree %5d, hipHostMalloc %3d, comments %2d"
              "  t=%.3f..%.3f%s" %
              (other, e["lines"], e["M"], e["F"], e["H"], e["comment"], e["first_t"], e["last_t"],
               mark))
    print("stream         T=%s, %d ops (hipMalloc %d, hipFree %d, hipHostMalloc %d), "
          "seq %s..%s, t=%.6f..%.6f s" %
          (",".join(str(t) for t in tids), state["ops"], state["counts"]["M"], state["counts"]["F"],
           state["counts"]["H"], state["first_seq"], state["last_seq"], state["first_t"],
           state["last_t"]))
    print("ops per tid    %s" % ", ".join("T=%d: %d" % (tid, n)
                                          for tid, n in sorted(state["per_tid_ops"].items())))
    print("live at end    %d blocks / %d B (hook's last line: %s blocks / %s B)" %
          (state["live_blocks"], state["live_bytes"], state["last_live"],
           state["last_live_bytes"]))
    print("live sha1      %s (our recipe, see docstring)" % state["live_sizes_sha1"])
    print("residuals      unmatched frees %d, duplicate pointers %d, malloc failures %d, "
          "free failures %d" % (state["unmatched_frees"], state["duplicate_pointers"],
                                state["malloc_failures"], state["free_failures"]))
    for comment in state["comments"]:
        print("comment        %s" % comment)
    for line in state["stray_lines"][:5]:
        print("  stray:       %s" % line)
    for cut in report["cut_checks"]:
        print("cut %-12s %5d ops  hipMalloc %d, hipFree %d, hipHostMalloc %d; live %d blocks / %d B"
              " (t=%.3f s)" %
              (cut["label"], cut["ops"], cut["counts"]["M"], cut["counts"]["F"], cut["counts"]["H"],
               cut["live_blocks"], cut["live_bytes"], cut["t"]))
    if report["run"]:
        print("run end        %s" % ", ".join("%s=%s" % (k, v)
                                              for k, v in sorted(report["run"].items())))
    print("verdict        %s" % ", ".join("%s=%s" % (k, v)
                                          for k, v in sorted(report["verdict"].items())))
    return 0 if all(v is not False for v in report["verdict"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

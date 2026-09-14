#!/usr/bin/env python3
"""Sweep `repro_unet.py` across the batch/seq combinations and print a failing-vs-ok table.

Each configuration runs in its own subprocess: the fault aborts the process (SIGABRT), so it
cannot be caught in-process.

    python repro_unet_sweep.py            # the grid below
    python repro_unet_sweep.py --detach 1 # detached prompt embeds (fix2's control case)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAULT_MARKERS = ("HSA_STATUS_ERROR_MEMORY_FAULT", "Memory access fault by GPU node",
                 "GCVM_L2_PROTECTION_FAULT", "Page not present or supervisor privilege")
GRID = [(1, 1), (2, 1), (2, 2), (3, 1), (2, 3), (3, 2), (3, 3)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--detach", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default=str(HERE / "resources" / "unet-sweep.json"))
    args = parser.parse_args()

    print(f"== real SDXL UNet + LoRA, synthetic cross-attention input, detach={args.detach}")
    print("== M = batch * 77 * chunks; the trainer hit M=462 (batch 3 x 2 chunks / batch 2 x 3 chunks)\n")
    rows = []
    for batch, chunks in GRID:
        command = [sys.executable, str(HERE / "repro_unet.py"), "--batch", str(batch),
                   "--chunks", str(chunks), "--iters", str(args.iters), "--detach", str(args.detach)]
        started = time.time()
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            stdout, stderr, code = done.stdout, done.stderr, done.returncode
        except subprocess.TimeoutExpired as expired:
            stdout = (expired.stdout or "").decode() if isinstance(expired.stdout, bytes) else (expired.stdout or "")
            stderr = (expired.stderr or "").decode() if isinstance(expired.stderr, bytes) else (expired.stderr or "")
            code = None
        blob = f"{stdout}\n{stderr}"
        fault = any(marker in blob for marker in FAULT_MARKERS)
        result_line = next((line for line in stdout.splitlines() if line.startswith("RESULT ")), None)
        verdict = ("timeout" if code is None else
                   "FAULT" if (fault and code != 0) else
                   "ok" if (code == 0 and result_line) else "error")
        entry = {"batch": batch, "chunks": chunks, "seq": 77 * chunks, "M": batch * 77 * chunks,
                 "detach": args.detach, "verdict": verdict, "exit_code": code,
                 "seconds": round(time.time() - started, 1),
                 "tail": "\n".join((blob.strip().splitlines() or [""])[-2:])}
        rows.append(entry)
        print(f"batch={batch} chunks={chunks} M={entry['M']:<4} -> {verdict:<7} "
              f"exit={code} ({entry['seconds']}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n| batch | CLIP chunks | seq | M | detach | result |")
    print("|---|---|---|---|---|---|")
    for entry in rows:
        print(f"| {entry['batch']} | {entry['chunks']} | {entry['seq']} | {entry['M']} | "
              f"{entry['detach']} | {entry['verdict']} |")
    faults = [entry for entry in rows if entry["verdict"] == "FAULT"]
    print(f"\n== {len(faults)}/{len(rows)} configs faulted; json: {args.out}")
    print("DONE" if faults else "NO-FAULT-REPRODUCED")
    return 0 if faults else 1


if __name__ == "__main__":
    raise SystemExit(main())

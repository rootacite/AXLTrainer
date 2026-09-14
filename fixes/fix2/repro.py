#!/usr/bin/env python3
"""Minimal, model-free reproduction of the gfx1201 page fault (fixes/fix2.txt, fixes/fix2-ex.md).

Every configuration runs in its own subprocess, because the fault aborts the process with
SIGABRT and cannot be caught in-process. The sweep walks the parameters that separate the
failing training runs from the surviving ones:

  rows (M)   train_batch_size * encoder_seq_len: 154 / 231 / 308 / 462 / 693. The real trainer
             produces these as batch 2 x seq 77 (154), batch 3 x seq 77 (231), batch 2 x seq 154
             (308), batch 2 x seq 231 (462, what fixes/fix2.txt derives) and batch 3 x seq 231
  rank (K)   network_dim: 36 (faults in fix2) vs 32 / 64 / 16 / 24
  dtype      bfloat16 (Tensile BF16 kernel) vs float32
  stress     keep odd-sized allocations alive to shift the caching allocator's packing
  mode       PEFT LoRA on a pair of projections, or the same GEMMs hand-written

No model weights are involved. Usage:

    python repro.py                       # default grid
    python repro.py --quick               # one dtype, one mode
    python repro.py --rows 231,462,693 --ranks 36,32,64
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAULT_MARKERS = ("HSA_STATUS_ERROR_MEMORY_FAULT", "Memory access fault by GPU node",
                 "GCVM_L2_PROTECTION_FAULT", "Page not present or supervisor privilege")


def run_trial(args: argparse.Namespace, rows: int, rank: int, dtype: str, stress: int, mode: str,
              layers: int, lived_gb: float, alternate: int) -> dict:
    command = [
        sys.executable, str(HERE / "trial.py"),
        "--rows", str(rows), "--rank", str(rank), "--width", str(args.width),
        "--dtype", dtype, "--mode", mode, "--stress", str(stress), "--iters", str(args.iters),
        "--layers", str(layers), "--lived-gb", str(lived_gb),
    ]
    if alternate and len(args.rows.split(",")) > 1:
        command += ["--rows2", args.rows.split(",")[1]]
    started = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as expired:
        stdout = (expired.stdout or b"").decode("utf-8", "replace") if isinstance(expired.stdout, bytes) else (expired.stdout or "")
        stderr = (expired.stderr or b"").decode("utf-8", "replace") if isinstance(expired.stderr, bytes) else (expired.stderr or "")
        code = None
    elapsed = time.time() - started

    blob = f"{stdout}\n{stderr}"
    fault = any(marker in blob for marker in FAULT_MARKERS)
    result_line = next((line for line in stdout.splitlines() if line.startswith("RESULT ")), None)
    if code is None:
        verdict = "timeout"
    elif fault and code != 0:
        verdict = "FAULT"
    elif code == 0 and result_line:
        verdict = "ok"
    else:
        verdict = "error"
    return {
        "rows": rows, "rank": rank, "dtype": dtype, "stress": stress, "mode": mode,
        "layers": layers, "lived_gb": lived_gb, "alternate": alternate,
        "verdict": verdict, "exit_code": code, "seconds": round(elapsed, 2),
        "signal": (-code) if (code is not None and code < 0) else None,
        "result": json.loads(result_line[len("RESULT "):]) if result_line else None,
        "tail": "\n".join((blob.strip().splitlines() or [""])[-4:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", default="154,231,308,462,693",
                        help="M values: batch 1 / 2 / 3 at encoder seq len 231")
    parser.add_argument("--ranks", default="36",
                        help="K values (network_dim); 36 is the one fix2 pins the fault on")
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument("--stress", default="0,1", help="allocator-layout perturbations")
    parser.add_argument("--modes", default="peft", help="peft and/or manual")
    parser.add_argument("--layers", default="1",
                        help="LoRA projection pairs per step; an SDXL UNet repeats this dozens of times")
    parser.add_argument("--lived-gb", default="0",
                        help="resident GB kept alive during the GEMMs, mimicking the loaded model")
    parser.add_argument("--alternate", default="0",
                        help="1 = alternate M between the first two --rows values every iteration, "
                             "as the trainer did when the caption's CLIP chunk count changed")
    parser.add_argument("--width", type=int, default=2048, help="N (SDXL attention width)")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", default=str(HERE / "resources" / "minimal-results.json"))
    parser.add_argument("--quick", action="store_true", help="bfloat16 + peft only")
    args = parser.parse_args()

    rows_values = [int(v) for v in args.rows.split(",")]
    rank_values = [int(v) for v in args.ranks.split(",")]
    dtypes = ["bfloat16"] if args.quick else args.dtypes.split(",")
    stresses = [0] if args.quick else [int(v) for v in args.stress.split(",")]
    modes = ["peft"] if args.quick else args.modes.split(",")
    layers_values = [int(v) for v in args.layers.split(",")]
    lived_values = [float(v) for v in args.lived_gb.split(",")]
    alternate_values = [int(v) for v in args.alternate.split(",")]

    trials = list(itertools.product(rows_values, rank_values, dtypes, stresses, modes,
                                    layers_values, lived_values, alternate_values))
    print(f"== {len(trials)} trials, one process each (the fault aborts the process, it cannot be caught)")
    print(f"== python {sys.version.split()[0]}, grid rows={rows_values} ranks={rank_values} "
          f"dtypes={dtypes} stress={stresses} modes={modes} layers={layers_values} "
          f"lived_gb={lived_values} alternate={alternate_values} width={args.width}\n")

    results = []
    for index, (rows, rank, dtype, stress, mode, layers, lived, alternate) in enumerate(trials, 1):
        entry = run_trial(args, rows, rank, dtype, stress, mode, layers, lived, alternate)
        results.append(entry)
        print(f"[{index:>2}/{len(trials)}] M={rows:<4}{'/' + args.rows.split(',')[1] if alternate else '':<5} "
              f"K={rank:<3} {dtype:<9} stress={stress} {mode:<7} layers={layers:<3} "
              f"lived={lived:<4} -> {entry['verdict']:<7} exit={entry['exit_code']} "
              f"({entry['seconds']}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    faults = [e for e in results if e["verdict"] == "FAULT"]
    print("\n== table (failing vs surviving configurations)")
    print(f"| {'M':>7} | {'K':>3} | {'dtype':<9} | {'stress':>6} | {'layers':>6} | {'lived GB':>8} | "
          f"{'alt':>3} | result |")
    print("|" + "---|" * 8)
    for entry in results:
        span = f"{entry['rows']}" + (f"/{args.rows.split(',')[1]}" if entry["alternate"] else "")
        print(f"| {span:>7} | {entry['rank']:>3} | {entry['dtype']:<9} | {entry['stress']:>6} | "
              f"{entry['layers']:>6} | {entry['lived_gb']:>8} | {entry['alternate']:>3} | {entry['verdict']} |")
    print(f"\n== {len(faults)}/{len(results)} trials hit the fault; results written to {args.out}")
    print("DONE" if faults else "NO-FAULT-REPRODUCED")
    return 0 if faults else 1


if __name__ == "__main__":
    raise SystemExit(main())

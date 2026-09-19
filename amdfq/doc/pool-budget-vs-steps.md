# The deferred-free pool's budget decides how long a hooked run survives

**What this answers.** With the peralloc hook (`amdfq/amdfq-vmm-rs/`) preloaded, a training run on this box
dies before it finishes step 0. With the hook's deferred-free pool turned on at its default budget
(`DESIGN.md` D12, `AMDFQ_POOL` = 2.5 GiB) the same run gets a few steps further, which is the
maintainer's observation of 2026-09-19 and the reason the pool exists. This campaign asks the next
question: **does the budget decide how far the run gets?** It does, and the relation is a threshold,
not a slope: below 2.5 GiB no run of the 25 below it got past step 3; every one of the 14 runs between
2.5 and 4 GiB reached 4 to 6 steps; past that the configuration runs out of VRAM.

**Where the numbers come from.** 47 runs under the hook, 2026-09-19 03:15–04:25, one card (RX 9070 XT
16 GB), desktop and sddm down so the card is otherwise idle (VRAM at rest 72–86 MiB, checked before and
after every arm). Only `AMDFQ_POOL` varies between arms: same `config.toml`
(md5 `bcd1b2362168643638768a4e9bef98e3`, unchanged before and after), same seed (1145141919, so every
run replays the same data order), same hook object (md5 `3bb36a16de8a9b22e93f101dce87ccaa`). Figure:
[`hipmalloc_pool_vs_steps.png`](hipmalloc_pool_vs_steps.png). Raw rows, harness and analysis:
`/tmp/poolsweep/` (`arm.sh`, `sweep.sh`, `rounds.sh`, `stats.py`, `plot.py`, `results.tsv`, one
directory per label) — `/tmp` is volatile, so this file carries the numbers.

## 1. The measurement

One row per run, `steps` from the trainer's own `step=` line in its stderr. `park ms` is the pool's own
eviction line (`pool: released va=… parked_ms=…`): how long a block actually stayed alive after its
`hipFree`, median and worst over the run.

| `AMDFQ_POOL` | runs | steps reached (each run) | median | ≥4 steps | ends in | park ms median (worst) |
| --- | --- | --- | --- | --- | --- | --- |
| no hook at all | 4 | 10, 10, 10, 10 | **10** | 4/4 | HSA memory fault | — |
| 0 (every free tears down) | 5 | 0, 0, 0, 0, 1 | **0** | 0/5 | fault ×4, nan ×1 | — |
| 64 MiB | 5 | 0, 0, 1, 1, 1 | 1 | 0/5 | nan ×3, **stall ×2** | 1 (5.3 s) |
| 256 MiB | 5 | 0, 1, 1, 1, 1 | 1 | 0/5 | fault ×3, **stall ×2** | 32 (5.3 s) |
| 768 MiB | 5 | 1, 1, 1, 2, 2 | 1 | 0/5 | nan ×2, fault ×2, **stall ×1** | 84 (5.6 s) |
| 1.5 GiB | 5 | 1, 1, 1, 3, 3 | 1 | 0/5 | nan ×2, fault ×1, **stall ×2** | 261 (5.7 s) |
| 2.5 GiB (default) | 5 | 4, 4, 4, 5, 6 | **4** | 5/5 | fault ×4, nan ×1 | 354 (5.8 s) |
| 3 GiB | 5 | 4, 6, 6, 6, 6 | **6** | 5/5 | nan ×4, fault ×1 | 452 (6.0 s) |
| 4 GiB | 4 | 4, 5, 6, 6 | **5.5** | 4/4 | nan ×2, fault ×2 | 564 (6.0 s) |
| 5 GiB | 2 | 2, 4 | 3 | 1/2 | **stall ×1**, nan ×1 | 736 (5.9 s) |
| 6 GiB | 2 | 1, 1 | 1 | 0/2 | **out of memory ×2** | 1439 (6.0 s) |

Reading it:

- **The threshold is between 1.5 GiB and 2.5 GiB.** 0 of 25 runs below 2.5 GiB reached step 4; 14 of 14
  between 2.5 and 4 GiB did. The two lowest columns are not graded — 64 MiB to 1.5 GiB all sit at a
  median of 1 step, and 0 (pool off) at 0 — so what the small budgets buy is not visible in steps at
  all.
- **What scales with the budget is the deferral, not the step count.** The median park age climbs
  with the budget by construction (1 ms → 564 ms), and the worst case sits at 5.2–6.0 s in every
  pooled configuration from 64 MiB up. So the budget is doing what it says; the run's survival saturates
  at 4–6 steps anyway.
- **The plateau is below the configuration's own ceiling.** With no interposer at all the run reaches
  step 10 every time (4/4) and dies of the bf16 over-read (`HSA_STATUS_ERROR_MEMORY_FAULT`), which is
  the defect `bf16-kernel-overrun.md` establishes — not of anything this hook did. Even at 4 GiB the
  hooked run stops at 4–6, so the pool does not remove the failure; it moves it, and something else caps
  it before the run reaches what the raw configuration can do.
- **6 GiB is past the ceiling of this configuration.** Both runs end in torch's own
  `OutOfMemoryError` ("of which 40.00 MiB is free", 8.75 GiB already allocated by PyTorch): the pool is
  holding a quarter of the card that the training run itself needs. The useful range on this box is
  therefore 2.5–4 GiB.
- **The obvious confound is closed by the design.** A larger pool means *less* free VRAM, so "more
  steps with a larger budget" cannot be a headroom effect — the memory side of the ledger is strictly
  worse at 4 GiB than at 64 MiB. Whatever the budget buys, it buys by holding freed memory back, not by
  freeing memory up.

## 2. A third failure mode: the run stalls

8 of the 47 runs did not die at all — they stopped making progress while the process stayed alive, and
the harness had to kill them after 150 s of no new step (`exit=137`). Where they land matters:
**7 of the 20 runs in 64 MiB–1.5 GiB**, 1 of 2 at 5 GiB, and **none** of the 14 runs at 2.5–4 GiB, none
of the 5 with the pool off, none of the 4 with no hook. Their stderr ends on ordinary
`hipFree(…) -> pooled` lines with no fault string of any kind.

One of them (1.5 GiB, 04:21:30) was caught with the GPU and thread state recorded:

```
gpu=100%  main_state=S  main_wchan=__futex_wait  threads=156  step=1
threads: 154 state=S, 2 state=R
wchan:   152 __futex_wait, 2 spinning (wchan=0), 1 poll_schedule_timeout, 1 kfd_wait_on_events
```

So the card is pinned at 100 % while the process waits — one thread blocked in
`kfd_wait_on_events`, waiting for a GPU event that never arrives, everything else queued behind it.
This is a run-level stall, not the machine-wide wedge of
[`gfx1201-fault-response-wedge.md`](gfx1201-fault-response-wedge.md): the arms *after* a stalled one ran
normally, and the canary (`fixes/hip1/repro_standalone.sh`, one deliberate fault per round, 6 rounds)
came back healthy every time, including the rounds that contained a stall.

That the stalls concentrate exactly where the pool churns hardest is worth recording as an observation,
not a conclusion: the eight stalled arms evicted 153–611 blocks each (a small budget has to evict to
make room), against 1604–1709 in the 2.5 GiB arms, which park instead — the mechanism was not identified. `pending: the stall is a kernel
that never completes on a mapping something else has already changed; that reading is inference.`

## 3. What is not established

- **The budget and the teardown rate are the same knob in this implementation.** A larger pool defers
  more *and* runs fewer unhooks/unmaps in the hot path, so "holding freed memory back" and "tearing
  memory down less often" are not separated by this campaign. Separating them needs a second knob —
  e.g. hold exactly the last N frees regardless of size — which does not exist yet.
- **What sets the 4–6 step plateau.** The campaign shows where the failure stops moving, not what
  happens at step 4–6. It is not the over-read (that one costs 10 steps and is visible with no hook).
- **What makes a run stall rather than die.** One stall has a state dump; the other seven have only
  "alive, GPU 100 %, no progress". Whether all eight are one phenomenon is unknown.
- **Scope.** One card, one workload, one config, one seed, one stack
  (`torch 2.13.0+rocm10.0.0` / HIP `7.15.26333`). The step counts are properties of *this* training
  configuration's memory behaviour; the shape (a threshold, then a plateau) is the part worth
  generalising, and only as a hypothesis.
- **Repeated labels lose their raw logs.** Each label reuses `/tmp/poolsweep/<label>/`, so only the
  last run of a multi-run label still has its stderr on disk. The per-run numbers in §1 were extracted
  when the run ended and live in `results.tsv`; the raw logs of the earlier runs of a label are gone.

## 4. How the campaign was run, and what it cost in mistakes

Rules the harness enforces, because a wrecked environment is a wrecked measurement:

- the trainer gets a private `AXL_RUNTIME_DIR` and it is emptied before each arm, so a stale pid can
  never be read as this run's (this was a bug in the first round of the campaign — see below);
- the pid is taken from a `state.json` written *after* that launch;
- every process holding `/dev/kfd` is gone before the launch and gone again before the next arm: a
  run's 20 DataLoader workers outlive the trainer by seconds, and the repo's own reaper
  (`trainer/orphans.py`) is not on this script's clock;
- `cap = 45` steps sits below `save_every_n_steps = 50`, so no arm writes a checkpoint or a sample —
  only TensorBoard scalars and an empty run directory; `hang = 150` s, `wall = 900` s;
- a canary runs before each round; the arm is marked `foreign` and dropped if another trainer appears
  on the card.

Mistakes that cost runs, all of which are why some columns have n = 4 rather than 5:

- **Stale pid + lingering workers** (rounds of 03:22–03:35): 14 rows were recorded against a dead pid
  from the previous arm of the same label, so their `steps` column was wrong and their environments
  were polluted by the previous arm's workers. All 14 were dropped and re-run with the fixed harness.
  The harness's `kfd_holders` line now records what was on the card at each arm's start; the ladder's
  `p4G` and `p6G` rows were dropped for the same reason (they started with 21 of a previous arm's
  workers still on the card) — 2 further runs.
- **Four runs whose `AMDFQ_POOL` was never set** (03:19–03:21): a zsh word-splitting bug in the round
  launcher passed a whole spec list as one argument, the hook could not parse it and fell back to
  `DEFAULT_POOL_LIMIT`. Those runs *are* valid 2.5 GiB runs and are kept, relabelled `p2G5`; the note is
  in `/tmp/poolsweep/README-p2G5_from_bug.txt`.
- **One arm lost to a mid-run edit** (`p4G`, rep 3): the harness file was rewritten while a bash was
  reading it, which is a syntax error; that is why 4 GiB has n = 4. Editing a file a running sweep
  executes must be an atomic replace.

## 5. Re-running it

```bash
# one arm: label, budget in bytes (or `none` for no interposer), cap, hang timeout, wall limit, rep
bash /tmp/poolsweep/arm.sh p2G5 2684354560 45 150 900 1

# the whole ladder / the rounds, with the canary before each round
CAP=45 HANG=150 WALL=900 bash /tmp/poolsweep/sweep.sh \
  nohook:none p0:0 p64M:67108864 p256M:268435456 p768M:805306368 p1G5:1610612736 \
  p2G5:2684354560 p3G:3221225472 p4G:4294967296 p5G:5368709120 p6G:6442450944
bash /tmp/poolsweep/rounds.sh

python3 /tmp/poolsweep/stats.py     # the table above
python3 /tmp/poolsweep/plot.py      # hipmalloc_pool_vs_steps.png next to this file
```

The pool itself is `amdfq/amdfq-vmm-rs/DESIGN.md` D12 (`peralloc.rs`: `park` / `teardown`, budget from
`AMDFQ_POOL`, oldest-first eviction); `AMDFQ_POOL=0` turns it off and is the `p0` column above.

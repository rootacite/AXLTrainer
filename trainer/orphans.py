"""Reap what a trainer killed by a signal leaves behind.

`start_train.sh` `exec`s the trainer, so the launcher's PID and *session* are the trainer's. When the
trainer dies from a signal — the gfx1201 Tensile over-read aborts it inside the HIP runtime
(see doc/troubleshooting.md) — Python's `atexit` never runs: the DataLoader forkserver
outlives the trainer, keeps the workers it forked, and each of them holds `/dev/kfd` and ~0.5 GB of
RSS. Nothing inside the trainer can clean that up, so the launcher starts this module in its own
session first and `exec`s the trainer second.

Torch-free and cheap on purpose: one of these runs beside every training run.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

PROC = Path("/proc")
# The literal CPython's forkserver is spawned with. Matching the shorter "multiprocessing.forkserver"
# would also match any shell whose *text* mentions it, including the one that runs this module.
FORKSERVER_CMD = "from multiprocessing.forkserver import main"


def _stat_tail(pid: int) -> list[str] | None:
    """Fields of `/proc/<pid>/stat` from the state field on, or None once the process is gone."""
    try:
        raw = (PROC / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # `comm` may contain spaces and parentheses, so split after its closing parenthesis:
    # tail[0] is field 3 (state), so field N is tail[N - 3].
    return raw.rsplit(")", 1)[-1].split()


def is_running(pid: int) -> bool:
    """A process that is still running.

    A dead child the parent has not waited on is a *zombie*: its `/proc` entry stays, its state
    field reads `Z`, and it holds nothing — no descriptors, no memory. `api.py` never `wait()`s the
    trainer it spawns, so the trainer sits in exactly that state until api.py exits, and treating it
    as alive would keep this module waiting for a trainer that is already gone.
    """
    tail = _stat_tail(pid)
    return bool(tail) and tail[0] != "Z"


def start_ticks(pid: int) -> int | None:
    """Field 22: the process's start time. Equal PIDs with different values are different processes."""
    tail = _stat_tail(pid)
    return int(tail[19]) if tail and len(tail) > 19 else None


def session_of(pid: int) -> int | None:
    """Field 6: the session id. Everything a trainer spawns shares it."""
    tail = _stat_tail(pid)
    return int(tail[3]) if tail and len(tail) > 3 else None


def is_alive(pid: int, born: int | None = None) -> bool:
    """True while `pid` is running *and* still the process that `born` was read from."""
    if not is_running(pid):
        return False
    return born is None or start_ticks(pid) == born


def pids() -> list[int]:
    return [int(entry.name) for entry in PROC.iterdir() if entry.name.isdigit()]


def session_members(sid: int) -> list[int]:
    """Every running process in session `sid`, this one excluded. Zombies are not members."""
    mine = os.getpid()
    return [pid for pid in pids() if pid != mine and is_running(pid) and session_of(pid) == sid]


def matches_forkserver(cmdline: str, marker: str) -> bool:
    """A DataLoader forkserver started for `marker` — the run's own `trainer/main.py`.

    Workers are forked *by* that server and inherit its command line, so one match covers them all.
    """
    return FORKSERVER_CMD in cmdline and marker in cmdline


def _ancestors() -> set[int]:
    """This process and everything above it: signalling one of those would take the reaper with it."""
    chain: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in chain:
        chain.add(pid)
        try:
            tail = (PROC / str(pid) / "stat").read_text().rsplit(")", 1)[-1].split()
            pid = int(tail[1])
        except (OSError, IndexError, ValueError):
            break
    return chain


def _cmdline(pid: int) -> str | None:
    try:
        return (PROC / str(pid) / "cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return None


def forkserver_members(marker: str) -> list[int]:
    """Forkserver processes (and their workers) belonging to `marker`, this reaper's tree excluded."""
    skip = _ancestors()
    found = []
    for pid in pids():
        if pid in skip or not is_running(pid):
            continue
        cmdline = _cmdline(pid)
        if cmdline and matches_forkserver(cmdline, marker):
            found.append(pid)
    return found


def reap(pids_to_reap: list[int], *, term_timeout: float = 2.0, poll: float = 0.05) -> list[int]:
    """SIGTERM the given processes, then SIGKILL whatever ignored it. Returns what was signalled."""
    skip = _ancestors()
    signalled: list[int] = []
    stubborn: list[int] = []
    for pid in sorted(set(pids_to_reap) - skip):
        if not is_running(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        signalled.append(pid)
        stubborn.append(pid)
    deadline = time.monotonic() + term_timeout
    while stubborn and time.monotonic() < deadline:
        stubborn = [pid for pid in stubborn if is_running(pid)]
        if stubborn:
            time.sleep(poll)
    for pid in stubborn:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return signalled


def watch(
    pid: int,
    *,
    marker: str | None = None,
    grace: float = 3.0,
    poll: float = 1.0,
    session: int | None = None,
    born: int | None = None,
    log=None,
) -> list[int]:
    """Wait for `pid` to die, then reap the session (or the forkservers) it left behind.

    The session and start time are read here rather than passed in, so a shell that asks for this
    cannot hand over a mis-parsed field. Only the trainer's **own** session is ever torn down — that
    is, when the launcher's `setsid` made it the session leader; anything else would be the session
    of whoever started the run, and killing that would take their shell with it.

    `grace` lets a *clean* shutdown finish on its own before anything is signalled; a run that ends
    normally has nothing left to kill, so the wait costs nothing but a timer.
    """
    if born is None:
        born = start_ticks(pid)
    if session is None:
        session = session_of(pid)
    if session != pid or session == os.getsid(0):
        session = None  # not ours to reap; fall back to the forkserver command line
    while is_alive(pid, born):
        time.sleep(poll)
    if grace > 0:
        time.sleep(grace)
    if session is not None:
        targets = session_members(session)
    elif marker is not None:
        targets = forkserver_members(marker)
    else:
        targets = []
    if not targets:
        return []
    if log is not None:
        log(f"trainer {pid} is gone; reaping {len(targets)} leftover process(es): {targets}")
    return reap(targets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reap the processes a dead trainer leaves behind.")
    parser.add_argument("--watch", type=int, required=True, metavar="PID",
                        help="the trainer's PID (the launcher's own, before it execs the trainer)")
    parser.add_argument("--script", default=None, metavar="PATH",
                        help="trainer/main.py of this run, to match forkservers by command line")
    parser.add_argument("--grace", type=float, default=3.0, help="seconds to let a clean shutdown finish")
    parser.add_argument("--poll", type=float, default=1.0, help="seconds between liveness checks")
    args = parser.parse_args(argv)

    watch(
        args.watch,
        marker=args.script,
        grace=args.grace,
        poll=args.poll,
        log=lambda line: print(line, file=sys.stderr, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

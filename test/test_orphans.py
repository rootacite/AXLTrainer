"""Reaping the processes a signal-killed trainer leaves behind.

No GPU and no torch: the module under test only reads `/proc` and sends signals.
"""

import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

# `python test/test_orphans.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.orphans import (
    forkserver_members,
    is_alive,
    matches_forkserver,
    reap,
    session_members,
    session_of,
    start_ticks,
    watch,
)


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.05)
    return not is_alive(pid)


def _state(pid: int) -> str:
    """Field 3 of `/proc/<pid>/stat`: `Z` for a child nobody has waited on."""
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    return raw.rsplit(")", 1)[-1].split()[0]


class StartTimeTest(unittest.TestCase):
    def test_liveness_is_bound_to_the_start_time(self):
        me = os.getpid()
        born = start_ticks(me)
        self.assertIsNotNone(born)
        self.assertTrue(is_alive(me, born))
        self.assertFalse(is_alive(me, born + 1), "a reused PID must not pass for the trainer")

    def test_dead_pid_is_not_alive(self):
        proc = subprocess.Popen(["true"])
        proc.wait()
        self.assertFalse(is_alive(proc.pid))
        self.assertIsNone(start_ticks(proc.pid))

    def test_an_unreaped_child_counts_as_gone(self):
        """`api.py` never waits on the trainer it spawns, so the trainer is a zombie when it dies.

        A zombie keeps its `/proc` entry, which is why liveness cannot be "the file exists": waiting
        for a zombie to disappear would never end.
        """
        proc = subprocess.Popen(["true"])
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and _state(proc.pid) != "Z":
                time.sleep(0.02)
            self.assertEqual(_state(proc.pid), "Z", "expected an unreaped child")
            self.assertIsNotNone(start_ticks(proc.pid), "a zombie still has a /proc entry")
            self.assertFalse(is_alive(proc.pid), "a zombie is gone for our purposes")
        finally:
            proc.wait()

    def test_session_of_self(self):
        self.assertEqual(session_of(os.getpid()), os.getsid(0))


class SessionReapTest(unittest.TestCase):
    """A trainer's whole tree shares one session, and that is what the reaper tears down."""

    def setUp(self):
        # A leader plus a background `sleep` in the same session: the shape of a forkserver with a
        # worker, without a forkserver or a dataset.
        self.leader = subprocess.Popen(["sh", "-c", "sleep 300 & wait"], start_new_session=True)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        reap(session_members(self.leader.pid))
        if self.leader.poll() is None:
            self.leader.kill()
        self.leader.wait()

    def _members(self):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            found = session_members(self.leader.pid)
            if len(found) >= 2:
                return found
            time.sleep(0.05)
        return session_members(self.leader.pid)

    def test_session_members_covers_the_leader_and_its_children(self):
        members = self._members()
        self.assertIn(self.leader.pid, members)
        self.assertEqual(session_of(self.leader.pid), self.leader.pid)

    def test_reap_terminates_a_whole_session(self):
        members = self._members()
        self.assertGreaterEqual(len(members), 2)
        signalled = reap(members)
        self.assertEqual(sorted(signalled), sorted(members))
        for pid in members:
            self.assertTrue(_wait_gone(pid), f"{pid} survived the reap")

    def test_watch_reaps_once_the_leader_is_gone(self):
        members = self._members()
        done = []

        def run():
            # Session and start time are derived from /proc inside `watch`.
            done.extend(watch(self.leader.pid, grace=0.0, poll=0.05))

        watcher = threading.Thread(target=run)
        watcher.start()
        self.assertIsNone(self.leader.poll(), "watch must wait for the leader")
        self.leader.kill()
        watcher.join(timeout=10)
        self.assertFalse(watcher.is_alive(), "watch did not return after the leader died")
        for pid in members:
            self.assertTrue(_wait_gone(pid), f"{pid} survived the watch")
        # The leader is a zombie by now — nothing has waited on it — so the reaper reports only the
        # processes that were still running when it looked.
        expected = sorted(set(members) - {self.leader.pid})
        self.assertEqual(sorted(done), expected)

    def test_watch_never_tears_down_a_session_that_is_not_the_trainers_own(self):
        """A run started from a terminal shares its session with the shell that started it.

        Reaping that session would kill the shell, so only a session the trainer leads itself counts.
        """
        gone = subprocess.Popen(["true"])
        gone.wait()
        self.assertEqual(watch(gone.pid, session=os.getsid(0), grace=0.0, poll=0.05), [])
        self.assertEqual(watch(gone.pid, session=os.getsid(0) + 1, grace=0.0, poll=0.05), [])
        self.assertTrue(is_alive(os.getpid()))


class ForkserverMatchTest(unittest.TestCase):
    """The command-line fallback, for a trainer that is not its own session leader."""

    def test_predicate(self):
        real = ("/usr/bin/python\x00-c\x00from multiprocessing.forkserver import main\x00"
                "main(15, 16, ['__main__'], main_path='/repo/trainer/main.py')\x00")
        self.assertTrue(matches_forkserver(real, "/repo/trainer/main.py"))
        self.assertFalse(matches_forkserver(real, "/other/trainer/main.py"))
        self.assertFalse(matches_forkserver("python -u trainer/main.py", "/repo/trainer/main.py"))
        # A shell that merely mentions the module — `pgrep -f multiprocessing.forkserver` — is not a
        # forkserver, and matching it would have this reaper signal the process that started it.
        mention = "bash -c pgrep -f multiprocessing.forkserver /repo/trainer/main.py"
        self.assertFalse(matches_forkserver(mention, "/repo/trainer/main.py"))

    def test_scan_finds_only_what_carries_both_tokens(self):
        marker = "/tmp/axl-orphans-marker"
        # A real forkserver's command line carries the import line, so any process carrying it is
        # what the fallback is looking for.
        fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)",
                                 "from multiprocessing.forkserver import main", marker])
        self.addCleanup(fake.wait)
        self.addCleanup(fake.kill)
        plain = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)",
                                  "multiprocessing.forkserver", marker])
        self.addCleanup(plain.wait)
        self.addCleanup(plain.kill)
        deadline = time.monotonic() + 5.0
        found = []
        while time.monotonic() < deadline:
            found = forkserver_members(marker)
            if fake.pid in found:
                break
            time.sleep(0.05)
        self.assertIn(fake.pid, found)
        self.assertNotIn(plain.pid, found, "a process without the forkserver import must not match")
        self.assertNotIn(os.getpid(), found)
        self.assertNotIn(os.getppid(), found, "the reaper must never target its own parent")


if __name__ == "__main__":
    unittest.main()

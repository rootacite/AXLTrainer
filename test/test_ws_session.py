"""One client at a time: the helper's session (`api.ClientSession` and the connection handler).

api.py does no locking of its own, so a second client must not be served interleaved with the first
— it is refused with a reason it can show the user, and the door is only left open for the owner's
own connections (its lanes) until it has been gone for the grace period.
"""
import functools
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import api


class FakeClock:
    """The session's clock, so a grace period costs no test time."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeConnection:
    """Just enough of a connection for the handler: queued frames in, replies out."""

    remote_address = ("127.0.0.1", 4444)

    def __init__(self, frames=()) -> None:
        self._frames = list(frames)
        self.replies: list[dict] = []
        self.closed = False

    def __iter__(self):
        return iter(self._frames)

    def send(self, payload: str) -> None:
        self.replies.append(json.loads(payload))

    def close(self) -> None:
        self.closed = True


def request(req_id: int, method: str, params: dict | None = None) -> str:
    return json.dumps({"id": req_id, "method": method, "params": params or {}})


class ClientSessionRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.session = api.ClientSession(grace=5.0, clock=self.clock)

    def test_the_first_client_owns_the_helper(self):
        ok, payload = self.session.hello("axlranko-desktop", "a-1")
        self.assertTrue(ok)
        self.assertTrue(payload["owner"])
        self.assertEqual("a-1", payload["instance"])
        self.assertEqual("axlranko-desktop", payload["client"])
        self.assertEqual(1, self.session.connections)

    def test_the_same_instance_may_open_many_connections(self):
        for expected in (1, 2, 3, 4):
            ok, payload = self.session.hello("axlranko-desktop", "a-1")
            self.assertTrue(ok, payload)
            self.assertEqual(expected, payload["connections"])
        self.assertEqual(4, self.session.connections)

    def test_a_second_client_is_refused_with_a_readable_reason(self):
        self.session.hello("axlranko-desktop", "a-1")
        self.clock.advance(30)
        ok, payload = self.session.hello("axlranko-web", "b-1")
        self.assertFalse(ok)
        self.assertEqual("CLIENT_BUSY", payload["code"])
        self.assertIn("axlranko-desktop", payload["error"])
        self.assertIn("close it and retry", payload["error"])
        self.assertEqual("a-1", payload["holder"]["instance"])
        # The refused client did not become a second owner, and took no connection slot.
        self.assertEqual("a-1", self.session.instance)
        self.assertEqual(1, self.session.connections)

    def test_asking_politely_does_not_take_the_helper_over(self):
        """No takeover path: extra parameters are ignored, however they are spelled."""
        self.session.hello("axlranko-desktop", "a-1")
        for extra in ({"force": True}, {"takeover": True}, {"instance": "b-1", "force": "yes"}):
            with self.subTest(extra=extra):
                ok, payload = self.session.hello("axlranko-web", str(extra.get("instance", "b-1")))
                self.assertFalse(ok)
                self.assertEqual("CLIENT_BUSY", payload["code"])

    def test_a_second_client_is_refused_inside_the_grace_period(self):
        self.session.hello("axlranko-desktop", "a-1")
        self.session.released()
        self.clock.advance(4.0)
        ok, _ = self.session.hello("axlranko-web", "b-1")
        self.assertFalse(ok)

    def test_the_owner_is_replaced_once_its_connections_are_gone_and_the_grace_has_passed(self):
        self.session.hello("axlranko-desktop", "a-1")
        self.session.released()
        self.clock.advance(api._OWNER_GRACE_SECONDS + 0.5)
        ok, payload = self.session.hello("axlranko-web", "b-1")
        self.assertTrue(ok, payload)
        self.assertEqual("b-1", self.session.instance)
        self.assertEqual(1, self.session.connections)

    def test_the_grace_is_five_seconds(self):
        self.assertEqual(5.0, api._OWNER_GRACE_SECONDS)

    def test_too_many_connections_are_refused(self):
        for _ in range(api._MAX_SESSION_CONNECTIONS):
            ok, _ = self.session.hello("axlranko-desktop", "a-1")
            self.assertTrue(ok)
        ok, payload = self.session.hello("axlranko-desktop", "a-1")
        self.assertFalse(ok)
        self.assertEqual("CLIENT_BUSY", payload["code"])
        self.assertIn("connections", payload["error"])

    def test_an_unnamed_client_shares_the_anonymous_session(self):
        """A script's second connection is not a second client; a named one still is."""
        self.assertTrue(self.session.hello("", "")[0])
        self.assertTrue(self.session.hello("", "")[0])
        ok, payload = self.session.hello("axlranko-desktop", "a-1")
        self.assertFalse(ok)
        self.assertEqual("CLIENT_BUSY", payload["code"])
        self.assertIn("unnamed client", payload["error"])

    def test_releasing_counts_down_and_info_follows(self):
        self.session.hello("axlranko-desktop", "a-1")
        self.session.hello("axlranko-desktop", "a-1")
        self.session.released()
        info = self.session.info()
        self.assertEqual(1, info["connections"])
        self.assertTrue(info["live"])
        self.session.released()
        self.assertTrue(self.session.info()["live"], "still inside the grace period")
        self.clock.advance(api._OWNER_GRACE_SECONDS + 0.5)
        self.assertFalse(self.session.info()["live"])


class WsHandlerAdmissionTest(unittest.TestCase):
    """The handler's side: which connection is admitted, and what a probe costs."""

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.session = api.ClientSession(grace=5.0, clock=self.clock)

    def run_handler(self, frames=()) -> FakeConnection:
        conn = FakeConnection(frames)
        api._ws_handler(conn, networks=[], session=self.session)
        return conn

    def hello(self, req_id: int, client: str, instance: str, **extra) -> str:
        params = {"client": client, "instance": instance, "lane": "control", **extra}
        return request(req_id, "hello", params)

    def test_a_probe_that_sends_nothing_never_claims_the_helper(self):
        # `helperListening` on the client side opens a connection, waits for the handshake and
        # closes it without a request: the helper must still be free afterwards.
        self.run_handler([])
        self.assertEqual("", self.session.instance)
        self.assertEqual(0, self.session.connections)
        conn = self.run_handler([self.hello(1, "axlranko-desktop", "a-1")])
        self.assertTrue(conn.replies[0]["ok"], conn.replies)

    def test_a_first_request_without_hello_claims_a_free_helper(self):
        conn = self.run_handler([request(1, "ping")])
        self.assertTrue(conn.replies[0]["ok"], conn.replies)
        self.assertEqual(0, self.session.connections, "the connection ended, so it is released")

    def test_a_request_before_an_owner_is_refused_and_closed(self):
        self.session.hello("axlranko-desktop", "a-1")
        conn = self.run_handler([request(1, "ping")])
        self.assertFalse(conn.replies[0]["ok"])
        self.assertEqual("CLIENT_BUSY", conn.replies[0]["code"])
        self.assertEqual(1, len(conn.replies), "the refused connection serves nothing else")

    def test_a_refused_hello_answers_with_the_holder_and_takes_no_slot(self):
        self.run_handler([self.hello(1, "axlranko-desktop", "a-1")])
        conn = self.run_handler([self.hello(1, "axlranko-web", "b-1", force=True)])
        reply = conn.replies[0]
        self.assertFalse(reply["ok"])
        self.assertEqual("CLIENT_BUSY", reply["code"])
        self.assertIn("axlranko-desktop", reply["error"])
        self.assertEqual("a-1", reply["holder"]["instance"])
        self.assertEqual(0, self.session.connections, "both connections ended")

    def test_hello_on_an_open_connection_answers_with_the_session(self):
        conn = self.run_handler([
            self.hello(1, "axlranko-desktop", "a-1"),
            self.hello(2, "axlranko-desktop", "a-1", lane="poll"),
        ])
        self.assertTrue(conn.replies[1]["ok"])
        self.assertEqual("a-1", conn.replies[1]["result"]["instance"])
        self.assertEqual(1, conn.replies[1]["result"]["connections"], "a second hello is not a lane")

    def test_the_connection_count_falls_when_a_connection_ends(self):
        self.run_handler([self.hello(1, "axlranko-desktop", "a-1")])
        self.assertEqual(0, self.session.connections)
        self.session.hello("axlranko-desktop", "a-1")
        self.assertEqual(1, self.session.connections)


class RealSocketSessionTest(unittest.TestCase):
    """The same rules over a real socket, so the refusal is exercised end to end."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_runtime = os.environ.get("AXL_RUNTIME_DIR")
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        self.session = api.ClientSession(grace=0.3)
        from websockets.sync.server import serve

        api._quiet_websockets_log()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.server = serve(
            functools.partial(api._ws_handler, networks=[], session=self.session),
            sock=self.sock,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        try:
            self.sock.close()
        except OSError:
            pass
        self.tmp.cleanup()
        if self._orig_runtime is None:
            os.environ.pop("AXL_RUNTIME_DIR", None)
        else:
            os.environ["AXL_RUNTIME_DIR"] = self._orig_runtime

    def connect(self):
        from websockets.sync.client import connect

        return connect(f"ws://127.0.0.1:{self.port}", proxy=None, open_timeout=5)

    def hello(self, ws, req_id: int, client: str, instance: str, lane: str = "control") -> dict:
        ws.send(request(req_id, "hello", {"client": client, "instance": instance, "lane": lane}))
        return json.loads(ws.recv())

    def test_the_owner_keeps_its_lanes_and_a_second_client_is_turned_away(self):
        # A probe connection that never sends a request must not claim anything.
        with self.connect():
            pass
        a1 = self.connect()
        a2 = self.connect()
        try:
            first = self.hello(a1, 1, "axlranko-desktop", "a-1")
            self.assertTrue(first["ok"], first)
            self.assertTrue(first["result"]["owner"])
            self.assertTrue(self.hello(a2, 1, "axlranko-desktop", "a-1", lane="poll")["ok"])
            self.assertEqual(2, self.session.connections, "both of the owner's lanes are open")
            with self.connect() as b:
                refusal = self.hello(b, 1, "axlranko-web", "b-1")
                self.assertFalse(refusal["ok"])
                self.assertEqual("CLIENT_BUSY", refusal["code"])
                self.assertIn("axlranko-desktop", refusal["error"])
                # The helper closes a refused connection instead of serving it.
                from websockets.exceptions import ConnectionClosed

                with self.assertRaises(ConnectionClosed):
                    b.recv()
        finally:
            a1.close()
            a2.close()
        time.sleep(0.5)
        with self.connect() as b2:
            takeover = self.hello(b2, 1, "axlranko-web", "b-1")
            self.assertTrue(takeover["ok"], takeover)
            self.assertEqual("b-1", self.session.instance)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""The Automation page's server side: ComfyUI client, discovery, workflow pre-check, jobs.

The ComfyUI is a stub (`http.server`) on a random loopback port, so these tests run
without a GPU and without the real instance — except for the two real-instance checks
that skip themselves when nothing answers on 127.0.0.1:8188.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import api
from trainer import automation, comfy

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "trainer" / "run_automation.py"
REAL_COMFY = "127.0.0.1:8188"


def _png_bytes(size=(8, 8), color=(200, 100, 50)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def stub_workflow() -> dict:
    """A miniature API-format workflow: one sampler, one prompt node, one SaveImage."""
    return {
        "1": {"class_type": "DiffusersLoader", "inputs": {"model_path": "waillu_170"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality", "clip": ["1", 1]}},
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "batch_size": 1,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "model": ["1", 0],
            },
        },
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "stub"}},
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
    }


STUB_OBJECT_INFO = {
    "DiffusersLoader": {"input": {"required": {"model_path": [["waillu_170"]]}}},
    "UpscaleModelLoader": {
        "input": {"required": {"model_name": ["COMBO", {"multiselect": False, "options": ["4x-AnimeSharp.pth"]}]}}
    },
    "UltralyticsDetectorProvider": {
        "input": {"required": {"model_name": [[("bbox/hand_yolov8s.pt")]]}}
    },
}


class StubComfy:
    """A ComfyUI-shaped HTTP server: enough to queue, poll and download."""

    def __init__(
        self,
        *,
        version: str = "9.9.9",
        object_info: dict | None = None,
        fail_calls: set[int] | None = None,
        hang_calls: set[int] | None = None,
        image: bytes | None = None,
        delay: float = 0.0,
    ) -> None:
        self.version = version
        self.object_info_payload = object_info if object_info is not None else STUB_OBJECT_INFO
        self.fail_calls = set(fail_calls or ())
        self.hang_calls = set(hang_calls or ())
        self.image = image or _png_bytes()
        self.delay = delay
        self.queued: list[dict] = []
        self.history: dict[str, dict] = {}
        self.calls = 0

        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # keep the test output clean
                pass

            def _send(self, payload: bytes, status: int = 200, content_type: str = "application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):  # noqa: N802 - http.server's naming
                path = self.path.split("?")[0]
                if path == "/system_stats":
                    system = {"comfyui_version": stub.version} if stub.version else {}
                    self._send(json.dumps({"system": system}).encode())
                    return
                if path == "/object_info":
                    self._send(json.dumps(stub.object_info_payload).encode())
                    return
                if path == "/queue":
                    self._send(json.dumps({"queue_running": [], "queue_pending": []}).encode())
                    return
                if path.startswith("/history/"):
                    prompt_id = path[len("/history/"):]
                    entry = stub.history.get(prompt_id)
                    self._send(json.dumps({prompt_id: entry} if entry else {}).encode())
                    return
                if path == "/view":
                    self._send(stub.image, content_type="image/png")
                    return
                self._send(b"{}", status=404)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                if self.path != "/prompt":
                    self._send(b"{}", status=404)
                    return
                payload = json.loads(body)
                workflow = payload.get("prompt") or {}
                stub.calls += 1
                call = stub.calls
                stub.queued.append(workflow)
                if stub.delay:
                    time.sleep(stub.delay)
                if call in stub.hang_calls:
                    self._send(json.dumps({"prompt_id": f"hang-{call}"}).encode())
                    return  # no history entry: the client keeps polling
                prompt_id = f"stub-{call}-{uuid.uuid4().hex[:6]}"
                status = "error" if call in stub.fail_calls else "success"
                batch = 1
                for node in workflow.values():
                    if isinstance(node, dict):
                        value = (node.get("inputs") or {}).get("batch_size")
                        if isinstance(value, int) and not isinstance(value, bool):
                            batch = max(1, value)
                            break
                stub.history[prompt_id] = {
                    "status": {
                        "completed": True,
                        "status_str": status,
                        "messages": [] if status == "success" else [["execution_error", {"node": "4"}]],
                    },
                    "outputs": (
                        {
                            "5": {
                                "images": [
                                    {
                                        "filename": f"stub_{call:05d}_{index:02d}_.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                    for index in range(batch)
                                ]
                            }
                        }
                        if status == "success"
                        else {}
                    ),
                }
                self._send(json.dumps({"prompt_id": prompt_id}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def positive_texts(self) -> list[str]:
        """The text the client put into node 2 of every queued workflow, in order."""
        return [str(w.get("2", {}).get("inputs", {}).get("text", "")) for w in self.queued]


class ComfyClientTest(unittest.TestCase):
    def setUp(self):
        self.stub = StubComfy()
        self.addCleanup(self.stub.close)

    def test_the_client_bypasses_the_http_proxy(self):
        # The maintainer's session exports these; loopback must not go through them.
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9"}):
            client = comfy.ComfyClient(self.stub.url)
            self.assertEqual("9.9.9", client.version())

    def test_the_client_queues_polls_and_downloads(self):
        client = comfy.ComfyClient(self.stub.url)
        prompt_id = client.queue_prompt(stub_workflow())
        entry = client.wait_for_prompt(prompt_id, poll_interval=0.05)
        images = entry["outputs"]["5"]["images"]
        self.assertEqual(1, len(images))
        self.assertEqual(b"\x89PNG", client.get_image(images[0]["filename"])[:4])

    def test_a_failed_prompt_raises_with_the_comfy_message(self):
        self.stub.fail_calls = {1}
        client = comfy.ComfyClient(self.stub.url)
        prompt_id = client.queue_prompt(stub_workflow())
        with self.assertRaises(comfy.ComfyError):
            client.wait_for_prompt(prompt_id, poll_interval=0.05)

    def test_waiting_can_be_stopped(self):
        self.stub.hang_calls = {1}
        client = comfy.ComfyClient(self.stub.url)
        prompt_id = client.queue_prompt(stub_workflow())
        with self.assertRaises(comfy.ComfyCancelled):
            client.wait_for_prompt(prompt_id, poll_interval=0.05, should_stop=lambda: True)

    def test_a_listener_that_is_not_comfyui_is_not_accepted(self):
        other = StubComfy(version="")
        self.addCleanup(other.close)
        found = comfy.discover(server=other.url)
        self.assertFalse(found["found"])
        self.assertIn("comfyui_version", found["checked"][0]["reason"])

    def test_discovery_finds_the_stub_through_the_port_scan(self):
        ports = comfy.listening_ports()
        self.assertIn(self.stub.port, ports)
        found = comfy.discover(ports=[self.stub.port], budget=2.0)
        self.assertTrue(found["found"])
        self.assertEqual(self.stub.url, found["url"])
        self.assertEqual("9.9.9", found["version"])

    def test_a_listener_that_speaks_another_protocol_is_reported_not_raised(self):
        # sshd and friends answer an HTTP probe with a banner; discovery must survive that.
        import socket

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        port = server.getsockname()[1]

        def serve():
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                with conn:
                    conn.recv(1024)
                    conn.sendall(b"SSH-2.0-OpenSSH_10.5\r\n")

        threading.Thread(target=serve, daemon=True).start()
        self.addCleanup(server.close)
        found = comfy.discover(server=f"http://127.0.0.1:{port}", budget=2.0)
        self.assertFalse(found["found"])
        self.assertEqual(1, len(found["checked"]))
        self.assertFalse(found["checked"][0]["ok"])
        self.assertTrue(found["checked"][0]["reason"])
        # The client itself names the real reason; discovery only reports "not a ComfyUI".
        client = comfy.ComfyClient(f"http://127.0.0.1:{port}")
        with self.assertRaises(comfy.ComfyError) as ctx:
            client.system_stats()
        self.assertIn("did not answer HTTP", str(ctx.exception))

    def test_the_environment_url_wins_over_scanning(self):
        with mock.patch.dict(os.environ, {"AXL_COMFY_URL": self.stub.url}):
            found = comfy.discover(ports=[])
        self.assertTrue(found["found"])
        self.assertEqual(self.stub.url, found["url"])


class WorkflowValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._env = mock.patch.dict(os.environ, {"AXL_AUTOMATION_DIR": str(self.root / "automation")})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_a_valid_api_workflow_reports_its_nodes(self):
        report = automation.validate_workflow(stub_workflow())
        self.assertTrue(report["valid"], report)
        self.assertEqual(["5"], report["save_image_nodes"])
        self.assertEqual(["4"], report["batch_size_nodes"])
        self.assertEqual("2", report["positive_node"])
        self.assertTrue(report["positive_node_guessed"])
        self.assertEqual(2, len(report["text_nodes"]))
        self.assertEqual("placeholder", report["text_nodes"][0]["text"])

    def test_the_editor_format_is_refused(self):
        report = automation.validate_workflow({"nodes": [], "links": []})
        self.assertFalse(report["valid"])
        self.assertIn("API format", report["error"])

    def test_a_workflow_without_save_image_or_prompt_is_refused(self):
        no_save = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}
        self.assertIn("SaveImage", automation.validate_workflow(no_save)["error"])
        no_text = {"5": {"class_type": "SaveImage", "inputs": {}}}
        self.assertIn("CLIPTextEncode", automation.validate_workflow(no_text)["error"])

    def test_the_positive_node_is_validated_when_given(self):
        report = automation.validate_workflow(stub_workflow(), positive_node="3")
        self.assertTrue(report["valid"])
        self.assertEqual("3", report["positive_node"])
        self.assertFalse(report["positive_node_guessed"])
        wrong = automation.validate_workflow(stub_workflow(), positive_node="5")
        self.assertFalse(wrong["valid"])
        self.assertIn("not a CLIPTextEncode", wrong["error"])
        missing = automation.validate_workflow(stub_workflow(), positive_node="99")
        self.assertIn("does not exist", missing["error"])

    def test_the_model_precheck_reads_both_combo_schemas(self):
        workflow = stub_workflow()
        workflow["1"]["inputs"]["model_path"] = "waillu_170"
        report = automation.validate_workflow(workflow, object_info=STUB_OBJECT_INFO)
        self.assertTrue(report["checked_models"])
        self.assertEqual([], report["missing_models"])

        workflow["1"]["inputs"]["model_path"] = "not-installed"
        workflow["7"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "missing.pth"}}
        broken = automation.validate_workflow(workflow, object_info=STUB_OBJECT_INFO)
        values = sorted(item["value"] for item in broken["missing_models"])
        self.assertEqual(["missing.pth", "not-installed"], values)

    def test_a_linked_input_is_not_checked_as_a_model(self):
        workflow = stub_workflow()
        workflow["2"]["inputs"]["clip"] = ["1", 1]
        report = automation.validate_workflow(workflow, object_info=STUB_OBJECT_INFO)
        self.assertEqual([], report["missing_models"])

    def test_workflow_save_refuses_a_broken_file_and_round_trips_a_good_one(self):
        with self.assertRaises(ValueError):
            automation.workflow_save("broken", '{"nodes": []}')
        saved = automation.workflow_save("stub", json.dumps(stub_workflow()))
        self.assertEqual("stub", saved["name"])
        self.assertTrue(Path(saved["path"]).is_file())
        listed = automation.workflow_list()
        self.assertEqual(["stub"], [item["name"] for item in listed])
        self.assertEqual("placeholder", automation.load_workflow(saved["path"])["2"]["inputs"]["text"])
        automation.workflow_delete("stub")
        self.assertEqual([], automation.workflow_list())


class AutomationStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._env = mock.patch.dict(os.environ, {"AXL_AUTOMATION_DIR": str(self.root / "automation")})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_settings_defaults_round_trip_and_validation(self):
        self.assertEqual(str(automation.default_output_dir()), automation.load_settings()["output_dir"])
        saved = automation.save_settings({"server": "127.0.0.1:8188", "count": 2, "poll": 0.25, "workflow": "stub"})
        self.assertEqual("http://127.0.0.1:8188", saved["server"])
        self.assertEqual(2, saved["count"])
        self.assertEqual(saved, automation.load_settings())
        with self.assertRaises(ValueError):
            automation.save_settings({"count": 0})
        with self.assertRaises(ValueError):
            automation.save_settings({"poll": 99})
        # A hand-edited bad value must not lock the page out of its own settings.
        automation.settings_path().write_text('{"count": 999}', encoding="utf-8")
        self.assertEqual(automation.default_settings()["count"], automation.load_settings()["count"])

    def test_a_relative_output_dir_resolves_against_the_repo(self):
        saved = automation.normalize_settings({"output_dir": "automation/jobs"})
        self.assertTrue(Path(saved["output_dir"]).is_absolute())

    def test_prompt_sets_round_trip_and_normalize(self):
        saved = automation.prompt_set_save("batch", "  a, b  \n\n c \n")
        self.assertEqual(2, saved["count"])
        self.assertEqual("a, b\nc\n", automation.prompt_set_get("batch")["text"])
        self.assertEqual(2, automation.prompt_set_list()[0]["count"])
        with self.assertRaises(ValueError):
            automation.prompt_set_save("batch", "\n\n")
        automation.prompt_set_delete("batch")
        self.assertEqual([], automation.prompt_set_list())

    def test_prompt_normalization_bounds(self):
        self.assertEqual(["one"], automation.normalize_prompts(" one "))
        self.assertEqual(["a", "b"], automation.normalize_prompts(["a", "b"]))
        with self.assertRaises(ValueError):
            automation.normalize_prompts("")

    def test_the_entry_holding_an_image_is_found_by_name(self):
        prompts = [
            {"index": 0, "images": ["p0001_01.png", "p0001_02.png"]},
            {"index": 1, "images": ["p0002_01.png"]},
            {"index": 2, "images": []},
        ]
        self.assertEqual(0, automation.prompt_entry_index(prompts, "p0001_02.png"))
        self.assertEqual(1, automation.prompt_entry_index(prompts, "p0002_01.png"))
        self.assertIsNone(automation.prompt_entry_index(prompts, "p0009_01.png"))
        self.assertIsNone(automation.prompt_entry_index("nonsense", "p0001_01.png"))
        self.assertEqual(prompts[1], automation.prompt_entry_at(prompts, 1))
        self.assertIsNone(automation.prompt_entry_at(prompts, 9))
        self.assertIsNone(automation.prompt_entry_at(prompts, -1))

    def test_the_next_image_number_is_one_past_the_highest(self):
        names = ["p0001_01.png", "p0001_02.png", "p0002_07.png", "nonsense.png"]
        self.assertEqual(3, automation.next_image_number(names, 0))
        self.assertEqual(8, automation.next_image_number(names, 1))
        self.assertEqual(1, automation.next_image_number([], 0))
        # A name for another prompt never moves this prompt's counter.
        self.assertEqual(1, automation.next_image_number(names, 4))

    def test_dropping_the_last_image_of_a_prompt_drops_the_entry(self):
        prompts = [
            {"index": 0, "text": "one", "images": ["p0001_01.png", "p0001_02.png"], "image_seeds": [1, 2]},
            {"index": 1, "text": "two", "images": ["p0002_01.png"], "image_seeds": [3]},
            {"index": 2, "text": "three", "images": ["p0003_01.png"]},
        ]
        kept = automation.drop_image(prompts, "p0002_01.png")
        self.assertEqual([0, 1], [entry["index"] for entry in kept], "the entry is gone and the rest renumber")
        self.assertEqual(["one", "three"], [entry["text"] for entry in kept])
        self.assertEqual(["p0001_01.png", "p0001_02.png"], kept[0]["images"])
        self.assertEqual(["p0003_01.png"], kept[1]["images"])

    def test_dropping_one_of_several_images_keeps_the_entry(self):
        prompts = [
            {"index": 0, "text": "one", "images": ["p0001_01.png", "p0001_02.png"], "image_seeds": [11, 12]},
        ]
        kept = automation.drop_image(prompts, "p0001_01.png")
        self.assertEqual([0], [entry["index"] for entry in kept])
        self.assertEqual(["p0001_02.png"], kept[0]["images"], "the seed that went with it goes too")
        self.assertEqual([12], kept[0]["image_seeds"])
        self.assertEqual("one", kept[0]["text"])

    def test_dropping_from_a_record_without_seeds_leaves_it_without_them(self):
        kept = automation.drop_image([{"index": 0, "images": ["p0004_01.png", "p0004_02.png"]}], "p0004_01.png")
        self.assertEqual(["p0004_02.png"], kept[0]["images"])
        self.assertNotIn("image_seeds", kept[0], "an older record does not grow the field")

    def test_dropping_a_name_no_entry_holds_changes_nothing(self):
        prompts = [{"index": 0, "text": "one", "images": ["p0001_01.png"], "image_seeds": [1]}]
        self.assertEqual(prompts, automation.drop_image(prompts, "p0009_01.png"))
        with self.assertRaises(ValueError):
            automation.normalize_prompts([f"p{i}" for i in range(automation.MAX_PROMPTS + 1)])
        with self.assertRaises(ValueError):
            automation.normalize_prompts("x" * (automation.MAX_PROMPT_CHARS + 1))
        with self.assertRaises(ValueError):
            automation.normalize_prompts(7)

    def test_job_names_and_records(self):
        job_id = automation.new_job_id("stub", now=1_700_000_000.0)
        self.assertRegex(job_id, r"^stub_\d{8}_\d{6}$")
        self.assertEqual("automation", automation.safe_name("", "automation"))
        self.assertNotIn("/", automation.new_job_id("a/b"))
        job = {"id": job_id, "state": automation.STATE_RUNNING, "pid": None, "prompts": []}
        automation.write_job(job, automation.default_output_dir())
        automation.update_job(job_id, automation.default_output_dir(), state=automation.STATE_DONE)
        self.assertEqual(automation.STATE_DONE, automation.read_job(automation.job_path(job_id))["state"])
        listed = automation.list_jobs()
        self.assertEqual([job_id], [item["id"] for item in listed])

    def test_a_running_job_whose_process_is_gone_becomes_an_error(self):
        os.makedirs(automation.default_output_dir(), exist_ok=True)
        job = {
            "id": "ghost_20260101_000000",
            "state": automation.STATE_RUNNING,
            "pid": _dead_pid(),
            "prompts": [{"index": 0, "text": "a", "state": automation.PROMPT_STATE_RUNNING, "images": []}],
        }
        automation.write_job(job, automation.default_output_dir())
        reconciled = automation.reconcile_jobs(automation.default_output_dir())
        self.assertEqual(automation.STATE_ERROR, reconciled[0]["state"])
        self.assertIn("log.txt", reconciled[0]["error"])
        self.assertEqual(automation.STATE_ERROR, automation.read_job(automation.job_path(job["id"]))["state"])

    def test_the_summary_counts_prompts_and_images(self):
        job = {
            "id": "x_20260101_000000",
            "state": automation.STATE_DONE,
            "output_dir": str(self.root / "jobs"),
            "prompts": [
                {"index": 0, "state": automation.PROMPT_STATE_DONE, "images": ["p0001_01.png", "p0001_02.png"]},
                {"index": 1, "state": automation.PROMPT_STATE_ERROR, "images": []},
            ],
        }
        summary = automation.job_summary(job)
        self.assertEqual(2, summary["total"])
        self.assertEqual(1, summary["done"])
        self.assertEqual(1, summary["failed"])
        self.assertEqual(2, summary["images"])
        self.assertEqual(2, len(summary["preview_paths"]))
        self.assertTrue(summary["preview_paths"][0].endswith("p0001_01.png"))
        self.assertEqual(summary["preview_paths"], summary["recent_paths"])


def _dead_pid() -> int:
    """A PID that is certainly gone: spawn `true` and wait for it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class RunnerTest(unittest.TestCase):
    """The real `trainer/run_automation.py` against the stub, as api.py spawns it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.automation_dir = self.root / "automation"
        self.env = {**os.environ, "AXL_AUTOMATION_DIR": str(self.automation_dir), "PYTHONUNBUFFERED": "1"}
        self.stub = StubComfy()
        self.addCleanup(self.stub.close)
        self.workflow_path = self.root / "stub.json"
        self.workflow_path.write_text(json.dumps(stub_workflow()), encoding="utf-8")
        self.output_dir = self.root / "jobs"

    def _job(self, prompts, count=1, poll=0.1):
        job_id = automation.new_job_id("stub")
        job = {
            "id": job_id,
            "state": automation.STATE_RUNNING,
            "created_at": time.time(),
            "started_at": None,
            "pid": None,
            "server": self.stub.url,
            "workflow_path": str(self.workflow_path),
            "positive_node": "2",
            "count": count,
            "poll": poll,
            "output_dir": str(self.output_dir),
            "prompts": [
                {"index": index, "text": text, "state": automation.PROMPT_STATE_PENDING, "images": []}
                for index, text in enumerate(prompts)
            ],
        }
        automation.write_job(job, self.output_dir)
        return job_id

    def _run(self, job_id, *extra):
        spec = automation.job_path(job_id, self.output_dir)
        proc = subprocess.run(
            [sys.executable, "-u", str(RUNNER), "--spec", str(spec), *extra],
            env=self.env,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=120,
        )
        job = automation.read_job(automation.job_path(job_id, self.output_dir))
        return proc, job

    def test_two_prompts_produce_images_and_sidecars(self):
        job_id = self._job(["alpha prompt", "beta prompt"], count=2)
        proc, job = self._run(job_id)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual(automation.STATE_DONE, job["state"], job.get("error"))
        self.assertEqual([], [p for p in job["prompts"] if p["state"] != automation.PROMPT_STATE_DONE])
        images = sorted(automation.images_dir(job_id, self.output_dir).glob("p*.png"))
        self.assertEqual(4, len(images), [p.name for p in images])
        self.assertEqual(b"\x89PNG", images[0].read_bytes()[:4])
        sidecar = images[0].with_suffix(".txt").read_text(encoding="utf-8")
        self.assertIn("seed: ", sidecar)
        self.assertIn("prompt_id: stub-", sidecar)
        self.assertIn("prompt: ", sidecar)
        # ComfyUI got the prompt in node 2, the batch size, and its own seed each time.
        self.assertEqual(["alpha prompt", "beta prompt"], self.stub.positive_texts())
        for workflow in self.stub.queued:
            self.assertEqual(2, workflow["4"]["inputs"]["batch_size"])
            self.assertIsInstance(workflow["4"]["inputs"]["seed"], int)
            self.assertEqual("stub", workflow["5"]["inputs"]["filename_prefix"])
        seeds = {p["seed"] for p in job["prompts"]}
        self.assertEqual(2, len(seeds))

    def test_a_single_seed_drives_every_seed_input(self):
        workflow = stub_workflow()
        workflow["9"] = {"class_type": "SeedNode", "inputs": {"seed": 5}}
        workflow["4"]["inputs"]["seed"] = ["9", 0]
        targets = [node_id for node_id, _input in run_module_seed_targets(workflow)]
        self.assertEqual(["9"], sorted(targets))
        job_id = self._job(["only"])
        self.workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
        proc, _job = self._run(job_id)
        self.assertEqual(0, proc.returncode, proc.stderr)
        queued = self.stub.queued[-1]
        self.assertNotIsInstance(queued["4"]["inputs"]["seed"], int)
        self.assertIsInstance(queued["9"]["inputs"]["seed"], int)

    def test_a_failing_prompt_is_recorded_and_the_batch_continues(self):
        self.stub.fail_calls = {1}
        job_id = self._job(["first prompt", "second prompt"])
        proc, job = self._run(job_id)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual(automation.PROMPT_STATE_ERROR, job["prompts"][0]["state"])
        self.assertIn("ComfyUI execution failed", job["prompts"][0]["error"])
        self.assertEqual(automation.PROMPT_STATE_DONE, job["prompts"][1]["state"])
        self.assertEqual(automation.STATE_DONE, job["state"])
        self.assertEqual(1, len(list(automation.images_dir(job_id, self.output_dir).glob("p*.png"))))

    def test_consecutive_failures_stop_the_batch(self):
        self.stub.fail_calls = {1, 2, 3}
        job_id = self._job(["a", "b", "c", "d"])
        proc, job = self._run(job_id)
        self.assertEqual(1, proc.returncode)
        self.assertEqual(automation.STATE_ERROR, job["state"])
        self.assertIn("failed in a row", job["error"])
        self.assertEqual(3, len(self.stub.queued), "the fourth prompt must not be queued")

    def test_only_failed_reruns_the_prompts_without_images(self):
        self.stub.fail_calls = {1}
        job_id = self._job(["first prompt", "second prompt"])
        self._run(job_id)
        queued_before = len(self.stub.queued)
        self.stub.fail_calls = set()
        proc, job = self._run(job_id, "--only-failed")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual(queued_before + 1, len(self.stub.queued), "the finished prompt must not be re-queued")
        self.assertEqual([automation.PROMPT_STATE_DONE] * 2, [p["state"] for p in job["prompts"]])
        self.assertEqual(2, len(list(automation.images_dir(job_id, self.output_dir).glob("p*.png"))))

    def test_regenerating_one_image_overwrites_it_with_a_new_seed(self):
        job_id = self._job(["only prompt"])
        _proc, job = self._run(job_id)
        name = job["prompts"][0]["images"][0]
        self.assertEqual([job["prompts"][0]["seed"]], job["prompts"][0]["image_seeds"])
        target = automation.images_dir(job_id, self.output_dir) / name
        before_bytes = target.read_bytes()
        before_seed = job["prompts"][0]["seed"]
        sidecar = target.with_suffix(".txt")
        self.assertIn(f"seed: {before_seed}", sidecar.read_text(encoding="utf-8"))

        self.stub.image = b"\x89PNG-other-bytes"
        proc, job = self._run(job_id, "--image", name)

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual([name], job["prompts"][0]["images"], "a redraw replaces the image, it adds none")
        self.assertEqual(1, len(list(automation.images_dir(job_id, self.output_dir).glob("p*.png"))))
        after_seed = job["prompts"][0]["image_seeds"][0]
        self.assertNotEqual(before_seed, after_seed)
        self.assertEqual(after_seed, job["prompts"][0]["seed"])
        self.assertEqual(self.stub.image, target.read_bytes())
        self.assertIn(f"seed: {after_seed}", sidecar.read_text(encoding="utf-8"))
        self.assertEqual(automation.PROMPT_STATE_DONE, job["prompts"][0]["state"])
        self.assertEqual(1, self.stub.queued[-1]["4"]["inputs"]["batch_size"], "one image per redraw")

    def test_append_adds_images_each_with_its_own_seed(self):
        job_id = self._job(["only prompt"])
        _proc, job = self._run(job_id)
        first_seed = job["prompts"][0]["image_seeds"][0]

        proc, job = self._run(job_id, "--append", "0", "--images", "3")

        self.assertEqual(0, proc.returncode, proc.stderr)
        images = automation.images_dir(job_id, self.output_dir)
        self.assertEqual(
            ["p0001_01.png", "p0001_02.png", "p0001_03.png", "p0001_04.png"],
            sorted(path.name for path in images.glob("p*.png")),
        )
        self.assertEqual(["p0001_01.png", "p0001_02.png", "p0001_03.png", "p0001_04.png"], job["prompts"][0]["images"])
        seeds = job["prompts"][0]["image_seeds"]
        self.assertEqual(4, len(seeds))
        self.assertEqual(first_seed, seeds[0])
        self.assertEqual(3, len(set(seeds[1:])), "each appended image drew its own random seed")
        for name, seed in zip(job["prompts"][0]["images"], seeds):
            sidecar = (images / name).with_suffix(".txt").read_text(encoding="utf-8")
            self.assertIn(f"seed: {seed}", sidecar)
        self.assertEqual(automation.PROMPT_STATE_DONE, job["prompts"][0]["state"])
        self.assertEqual([1, 1, 1], [w["4"]["inputs"]["batch_size"] for w in self.stub.queued[-3:]])

    def test_append_all_adds_images_to_every_prompt(self):
        job_id = self._job(["first", "second"])
        _proc, job = self._run(job_id)
        before = [list(entry["images"]) for entry in job["prompts"]]
        seeds_before = [list(entry["image_seeds"]) for entry in job["prompts"]]

        proc, job = self._run(job_id, "--append-all", "--images", "2")

        self.assertEqual(0, proc.returncode, proc.stderr)
        images = automation.images_dir(job_id, self.output_dir)
        self.assertEqual(
            ["p0001_01.png", "p0001_02.png", "p0001_03.png", "p0002_01.png", "p0002_02.png", "p0002_03.png"],
            sorted(path.name for path in images.glob("p*.png")),
        )
        for entry, had, had_seeds in zip(job["prompts"], before, seeds_before):
            # Each entry continues its own numbering, and keeps the images it already had.
            self.assertEqual(had + [f"p{entry['index'] + 1:04d}_{n:02d}.png" for n in (2, 3)], entry["images"])
            self.assertEqual(had_seeds, entry["image_seeds"][: len(had_seeds)])
            self.assertEqual(3, len(set(entry["image_seeds"])), "each appended image drew its own random seed")
            self.assertEqual(automation.PROMPT_STATE_DONE, entry["state"])
        queued_texts = [call["2"]["inputs"]["text"] for call in self.stub.queued[-4:]]
        self.assertEqual(["first", "first", "second", "second"], queued_texts)
        self.assertEqual([1, 1, 1, 1], [call["4"]["inputs"]["batch_size"] for call in self.stub.queued[-4:]])

    def test_append_all_reports_its_progress_over_every_prompt(self):
        job_id = self._job(["first", "second"])
        _proc, _job = self._run(job_id)

        seen = self._run_and_watch(job_id, ["--append-all", "--images", "1"], expect_done=1)

        self.assertEqual("append_all", seen["mode"])
        self.assertEqual(2, seen["total_images"], "the pass covers one image per prompt")
        self.assertGreaterEqual(seen["images_done"], 1)
        self.assertIsNone(automation.read_job(automation.job_path(job_id, self.output_dir)).get("pass"))

    def test_a_targeted_pass_leaves_the_other_prompts_alone(self):
        job_id = self._job(["first", "second"], count=1)
        _proc, job = self._run(job_id)
        second = job["prompts"][1]
        queued_before = len(self.stub.queued)

        proc, job = self._run(job_id, "--image", job["prompts"][0]["images"][0])

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual(queued_before + 1, len(self.stub.queued))
        self.assertEqual(second, job["prompts"][1], "the untouched prompt keeps its seed and images")

    def test_a_pass_that_names_nothing_is_refused(self):
        job_id = self._job(["only prompt"])
        self._run(job_id)
        queued_before = len(self.stub.queued)
        for extra in (["--image", "p0009_01.png"], ["--append", "7", "--images", "2"]):
            proc, job = self._run(job_id, *extra)
            self.assertEqual(1, proc.returncode, extra)
            self.assertEqual(automation.STATE_ERROR, job["state"], extra)
            self.assertIn("ComfyError", job["error"], extra)
        self.assertEqual(queued_before, len(self.stub.queued), "nothing may be queued for a pass that cannot run")

    def test_image_and_append_are_two_different_passes(self):
        job_id = self._job(["only prompt"])
        for extra in (
            ["--image", "p0001_01.png", "--append", "0"],
            ["--append", "0", "--append-all"],
            ["--image", "p0001_01.png", "--append-all"],
        ):
            proc, _job = self._run(job_id, *extra)
            self.assertEqual(2, proc.returncode, extra)
            self.assertIn("two different passes", proc.stderr, extra)

    def test_a_targeted_pass_reports_its_progress_while_it_runs(self):
        job_id = self._job(["only prompt"])
        _proc, job = self._run(job_id)
        name = job["prompts"][0]["images"][0]

        # A redraw keeps the job's own counters still, so the record has to say what it is doing.
        seen = self._run_and_watch(job_id, ["--image", name])
        self.assertEqual("image", seen["mode"])
        self.assertEqual(0, seen["prompt_index"])
        self.assertEqual(1, seen["total_images"])
        self.assertIsNone(automation.read_job(automation.job_path(job_id, self.output_dir)).get("pass"))

        # An append walks its own images; the closest the record gets to a finished count is what
        # a poll can catch, and the last one written clears the line again.
        seen = self._run_and_watch(job_id, ["--append", "0", "--images", "3"], expect_done=1)
        self.assertEqual("append", seen["mode"])
        self.assertEqual(3, seen["total_images"])
        self.assertGreaterEqual(seen["images_done"], 1)
        self.assertIsNotNone(seen["image"])
        self.assertIsNone(automation.read_job(automation.job_path(job_id, self.output_dir)).get("pass"))

    def _run_and_watch(self, job_id, extra, expect_done=0, timeout=60.0):
        """Run one pass, and return its `pass` block as a poll caught it (the last one seen)."""
        self.stub.delay = 1.5
        spec = automation.job_path(job_id, self.output_dir)
        proc = subprocess.Popen(
            [sys.executable, "-u", str(RUNNER), "--spec", str(spec), *extra],
            env=self.env,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        seen = None
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() is None:
            job = automation.read_job(automation.job_path(job_id, self.output_dir))
            block = job.get("pass") if job else None
            if isinstance(block, dict):
                seen = block
                if block.get("images_done", 0) >= expect_done and expect_done > 0:
                    break
            time.sleep(0.05)
        proc.wait(timeout=timeout)
        self.stub.delay = 0
        self.assertIsNotNone(seen, f"the pass of {extra} was never recorded")
        return seen

    def test_cancel_stops_between_prompts(self):
        self.stub.hang_calls = {1}
        job_id = self._job(["hang me", "never"])
        spec = automation.job_path(job_id, self.output_dir)
        proc = subprocess.Popen(
            [sys.executable, "-u", str(RUNNER), "--spec", str(spec)],
            env=self.env,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        # Wait until the runner has queued the first prompt, then cancel like api.py does.
        deadline = time.time() + 20
        while time.time() < deadline and not self.stub.queued:
            time.sleep(0.05)
        self.assertTrue(self.stub.queued, "the runner never queued anything")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
        job = automation.read_job(automation.job_path(job_id, self.output_dir))
        self.assertEqual(automation.STATE_CANCELLED, job["state"], job.get("error"))
        self.assertEqual(1, len(self.stub.queued), "the second prompt must not be queued")


def run_module_seed_targets(workflow):
    """`run_automation.seed_targets`, imported the way the runner imports it."""
    sys.path.insert(0, str(REPO / "trainer"))
    try:
        import run_automation  # noqa: PLC0415 - imported on purpose here

        return run_automation.seed_targets(workflow)
    finally:
        sys.path.remove(str(REPO / "trainer"))


class AutomationApiTest(unittest.TestCase):
    """The IPC handlers, through `api.dispatch` exactly like Ranko calls them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.automation_dir = self.root / "automation"
        self.env = mock.patch.dict(os.environ, {"AXL_AUTOMATION_DIR": str(self.automation_dir)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.stub = StubComfy()
        self.addCleanup(self.stub.close)
        self.workflow_path = self.root / "stub.json"
        self.workflow_path.write_text(json.dumps(stub_workflow()), encoding="utf-8")
        self.output_dir = self.root / "jobs"
        # The dataset and training output live elsewhere, so only the automation root can
        # be the reason an image is servable.
        self.out = self.root / "out"
        self.data = self.root / "data"
        self.out.mkdir()
        self.data.mkdir()
        self._orig = api._train_config_dict
        api._train_config_dict = lambda: {
            "train_data_dir": str(self.data),
            "output_dir": str(self.out),
            "output_name": "rein",
        }
        self.addCleanup(lambda: setattr(api, "_train_config_dict", self._orig))

    def tearDown(self):
        # api.py spawns the runner detached, so no runner may outlive the test.
        for job in automation.list_jobs(self.output_dir):
            pid = job.get("pid")
            if not pid:
                continue
            for _ in range(200):
                try:
                    os.kill(int(pid), 0)
                except (ProcessLookupError, PermissionError, OSError, TypeError, ValueError):
                    break
                time.sleep(0.05)

    def _settings(self, **overrides):
        payload = {
            "server": self.stub.url,
            "workflow": str(self.workflow_path),
            "positive_node": "2",
            "count": 1,
            "poll": 0.1,
            "output_dir": str(self.output_dir),
        }
        payload.update(overrides)
        return api.dispatch("automation_config_save", {"settings": payload})["settings"]

    def test_config_get_and_save(self):
        got = api.dispatch("automation_config_get", {})
        self.assertIn("settings", got)
        self.assertEqual(str(automation.default_output_dir()), got["default_output_dir"])
        self.assertIn("root", got["paths"])
        saved = self._settings(count=3)
        self.assertEqual(3, api.dispatch("automation_config_get", {})["settings"]["count"])
        self.assertEqual(3, saved["count"])
        with self.assertRaises(ValueError):
            api.dispatch("automation_config_save", {"settings": {"count": -1}})

    def test_discover_reports_the_stub(self):
        found = api.dispatch("automation_discover", {"server": self.stub.url})
        self.assertTrue(found["found"])
        self.assertEqual("9.9.9", found["version"])
        missing = api.dispatch("automation_discover", {"server": "127.0.0.1:9"})
        self.assertFalse(missing["found"])

    def test_workflow_handlers_report_and_refuse(self):
        # Point the settings at the stub first: the model pre-check must never depend on whatever
        # else happens to be listening on this machine.
        self._settings()
        listed = api.dispatch("automation_workflow_list", {})
        self.assertEqual([], listed["workflows"])
        saved = api.dispatch(
            "automation_workflow_save", {"name": "stub", "text": json.dumps(stub_workflow())}
        )
        self.assertTrue(saved["valid"], saved)
        self.assertEqual("2", saved["positive_node"])
        listed = api.dispatch("automation_workflow_list", {})
        self.assertEqual(["stub"], [item["name"] for item in listed["workflows"]])
        self.assertTrue(listed["model_check"], "the stub answers /object_info")
        self.assertEqual([], listed["workflows"][0]["missing_models"])
        with self.assertRaises(ValueError):
            api.dispatch("automation_workflow_save", {"name": "editor", "text": '{"nodes": []}'})
        report = api.dispatch("automation_workflow_validate", {"path": str(self.workflow_path)})
        self.assertTrue(report["valid"])
        api.dispatch("automation_workflow_delete", {"name": "stub"})
        self.assertEqual([], api.dispatch("automation_workflow_list", {})["workflows"])

    def test_prompt_set_handlers(self):
        api.dispatch("automation_prompt_save", {"name": "batch", "text": "a\nb\n"})
        listed = api.dispatch("automation_prompt_list", {})["prompts"]
        self.assertEqual(["batch"], [item["name"] for item in listed])
        self.assertEqual(2, listed[0]["count"])
        self.assertEqual("a\nb\n", api.dispatch("automation_prompt_get", {"name": "batch"})["text"])
        api.dispatch("automation_prompt_delete", {"name": "batch"})
        self.assertEqual([], api.dispatch("automation_prompt_list", {})["prompts"])

    def test_a_job_runs_end_to_end_and_its_images_are_servable(self):
        self._settings()
        started = api.dispatch(
            "automation_job_start", {"prompts": ["api prompt one", "api prompt two"]}
        )
        job_id = started["job"]["id"]
        self.assertTrue(Path(started["log_path"]).name == "log.txt")
        job = self._wait_for(job_id)
        self.assertEqual(automation.STATE_DONE, job["state"], job.get("error"))
        self.assertEqual(2, job["summary"]["done"])
        self.assertEqual(2, job["summary"]["images"])
        self.assertIn("api prompt one", self.stub.positive_texts())

        listed = api.dispatch("automation_job_list", {})["jobs"]
        self.assertEqual([job_id], [item["id"] for item in listed])
        self.assertEqual(2, listed[0]["done"])

        # The images live under the automation output dir, which blob_* must now allow.
        image = job["summary"]["preview_paths"][0]
        resolved = api.dispatch("blob_stat", {"paths": [image], "max_edge": 64})
        self.assertNotIn("error", resolved["items"][0], resolved["items"][0])
        self.assertEqual(8, resolved["items"][0]["width"])

        deleted = api.dispatch("automation_job_delete", {"id": job_id})
        self.assertEqual(job_id, deleted["id"])
        self.assertEqual([], api.dispatch("automation_job_list", {})["jobs"])

    def test_start_refuses_without_prompts_or_a_workflow(self):
        self._settings()
        with self.assertRaises(ValueError):
            api.dispatch("automation_job_start", {"prompts": []})
        api.dispatch("automation_config_save", {"settings": {**self._settings(), "workflow": ""}})
        with self.assertRaises(ValueError):
            api.dispatch("automation_job_start", {"prompts": ["x"]})

    def test_start_takes_a_prompt_set_and_pins_the_positive_node(self):
        self._settings()
        api.dispatch("automation_prompt_save", {"name": "batch", "text": "from the set\n"})
        started = api.dispatch("automation_job_start", {"prompt_set": "batch"})
        job = self._wait_for(started["job"]["id"])
        self.assertEqual(automation.STATE_DONE, job["state"], job.get("error"))
        self.assertEqual(["from the set"], self.stub.positive_texts())
        self.assertEqual("2", job["positive_node"])

    def test_a_running_job_blocks_a_second_start_and_can_be_cancelled(self):
        self._settings()
        self.stub.hang_calls = {1}
        started = api.dispatch("automation_job_start", {"prompts": ["hang", "later"]})
        job_id = started["job"]["id"]
        self._wait_until(lambda: self.stub.queued)
        with self.assertRaises(ValueError) as ctx:
            api.dispatch("automation_job_start", {"prompts": ["another"]})
        self.assertIn("already running", str(ctx.exception))
        cancelled = api.dispatch("automation_job_cancel", {"id": job_id})
        self.assertEqual(automation.STATE_CANCELLED, cancelled["job"]["state"])
        self.assertIn("cancel", (cancelled["job"]["error"] or "").lower())
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_retry_failed_only_reruns_the_failures(self):
        self._settings()
        self.stub.fail_calls = {1}
        started = api.dispatch("automation_job_start", {"prompts": ["one", "two"]})
        job_id = started["job"]["id"]
        job = self._wait_for(job_id)
        self.assertEqual(1, job["summary"]["failed"])
        self.stub.fail_calls = set()
        queued_before = len(self.stub.queued)
        api.dispatch("automation_job_retry_failed", {"id": job_id})
        job = self._wait_for(job_id)
        self.assertEqual(0, job["summary"]["failed"], job)
        self.assertEqual(queued_before + 1, len(self.stub.queued))
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_regenerating_an_image_swaps_its_seed_and_uses_the_edited_text(self):
        self._settings()
        job_id = api.dispatch("automation_job_start", {"prompts": ["original text"]})["job"]["id"]
        job = self._wait_for(job_id)
        name = job["prompts"][0]["images"][0]
        before = job["prompts"][0]["image_seeds"][0]

        api.dispatch("automation_job_prompt_edit", {"id": job_id, "prompt_index": 0, "text": "edited text"})
        edited = api.dispatch("automation_job_get", {"id": job_id})
        self.assertEqual("edited text", edited["prompts"][0]["text"])

        api.dispatch("automation_image_regenerate", {"id": job_id, "image": name})
        job = self._wait_for(job_id)

        self.assertEqual([name], job["prompts"][0]["images"], "a redraw replaces the image in place")
        self.assertEqual(1, job["summary"]["images"])
        self.assertNotEqual(before, job["prompts"][0]["image_seeds"][0])
        self.assertEqual("edited text", self.stub.positive_texts()[-1], "the redraw sends the new text")
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_extending_a_prompt_appends_images_and_seeds(self):
        self._settings()
        job_id = api.dispatch("automation_job_start", {"prompts": ["only prompt"]})["job"]["id"]
        job = self._wait_for(job_id)
        first = job["prompts"][0]["images"][0]

        api.dispatch("automation_prompt_extend", {"id": job_id, "prompt_index": 0, "count": 3})
        job = self._wait_for(job_id)

        images = automation.images_dir(job_id, self.output_dir)
        self.assertEqual(4, job["summary"]["images"])
        self.assertEqual(["p0001_01.png", "p0001_02.png", "p0001_03.png", "p0001_04.png"], job["prompts"][0]["images"])
        self.assertEqual([first] + sorted(job["prompts"][0]["images"][1:]), job["prompts"][0]["images"])
        self.assertEqual(4, len(set(job["prompts"][0]["image_seeds"])))
        self.assertEqual(4, len(list(images.glob("p*.png"))))
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_extending_every_prompt_appends_to_each(self):
        self._settings()
        job_id = api.dispatch("automation_job_start", {"prompts": ["first", "second"]})["job"]["id"]
        job = self._wait_for(job_id)
        first = [entry["images"][0] for entry in job["prompts"]]

        api.dispatch("automation_prompt_extend_all", {"id": job_id, "count": 2})
        job = self._wait_for(job_id)

        images = automation.images_dir(job_id, self.output_dir)
        self.assertEqual(6, job["summary"]["images"])
        self.assertEqual(["p0001_01.png", "p0001_02.png", "p0001_03.png"], job["prompts"][0]["images"])
        self.assertEqual(["p0002_01.png", "p0002_02.png", "p0002_03.png"], job["prompts"][1]["images"])
        self.assertEqual(first, [entry["images"][0] for entry in job["prompts"]], "nothing is overwritten")
        self.assertEqual(6, len(list(images.glob("p*.png"))))
        for entry in job["prompts"]:
            self.assertEqual(3, len(set(entry["image_seeds"])), "every added image drew its own seed")
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_deleting_the_last_image_drops_the_prompt_entry(self):
        self._settings(count=2)
        job_id = api.dispatch("automation_job_start", {"prompts": ["solo", "second"]})["job"]["id"]
        job = self._wait_for(job_id)
        images = automation.images_dir(job_id, self.output_dir)
        first, second = job["prompts"][0]["images"]

        # One of the prompt's two images: the entry survives with the other one.
        api.dispatch("automation_image_delete", {"id": job_id, "image": first})
        job = api.dispatch("automation_job_get", {"id": job_id})
        self.assertEqual([second], job["prompts"][0]["images"])
        self.assertFalse((images / first).exists())
        self.assertFalse((images / first).with_suffix(".txt").exists(), "the sidecar goes with the image")
        self.assertEqual(1, len(job["prompts"][0]["image_seeds"]), "its seed went with it")
        self.assertTrue((images / second).exists())

        # Its last image: the entry is gone and the remaining entries renumber.
        api.dispatch("automation_image_delete", {"id": job_id, "image": second})
        job = api.dispatch("automation_job_get", {"id": job_id})
        self.assertEqual(["second"], [entry["text"] for entry in job["prompts"]])
        self.assertEqual([0], [entry["index"] for entry in job["prompts"]])
        self.assertEqual(2, len(job["prompts"][0]["images"]), "the other prompt is untouched")
        self.assertEqual(2, len(list(images.glob("p*.png"))))
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_the_image_handlers_refuse_what_they_cannot_do(self):
        self._settings()
        job_id = api.dispatch("automation_job_start", {"prompts": ["only prompt"]})["job"]["id"]
        job = self._wait_for(job_id)
        name = job["prompts"][0]["images"][0]

        for params in (
            {"id": job_id, "image": "p0009_01.png"},   # no prompt holds that name
            {"id": "nope", "image": name},             # no such job
            {"id": job_id, "image": ""},
        ):
            with self.assertRaises(ValueError):
                api.dispatch("automation_image_delete", params)
            with self.assertRaises(ValueError):
                api.dispatch("automation_image_regenerate", params)
        for params in (
            {"id": job_id, "prompt_index": 7, "count": 1},
            {"id": job_id, "prompt_index": 0, "count": 0},
            {"id": job_id, "prompt_index": 0, "count": 17},
            {"id": job_id, "prompt_index": 0, "count": "many"},
            {"id": job_id, "prompt_index": "first", "count": 1},
        ):
            with self.assertRaises(ValueError):
                api.dispatch("automation_prompt_extend", params)
        for params in (
            {"id": job_id, "count": 0},
            {"id": job_id, "count": 17},
            {"id": job_id, "count": "many"},
            {"id": "nope", "count": 1},
        ):
            with self.assertRaises(ValueError):
                api.dispatch("automation_prompt_extend_all", params)
        for params in (
            {"id": job_id, "prompt_index": 0, "text": "   "},
            {"id": job_id, "prompt_index": 7, "text": "x"},
            {"id": job_id, "prompt_index": 0, "text": "x" * 4001},
        ):
            with self.assertRaises(ValueError):
                api.dispatch("automation_job_prompt_edit", params)
        # The record is untouched by all of that.
        job = api.dispatch("automation_job_get", {"id": job_id})
        self.assertEqual([name], job["prompts"][0]["images"])
        self.assertEqual("only prompt", job["prompts"][0]["text"])
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_image_delete_refuses_a_name_that_leaves_the_job_directory(self):
        self._settings()
        job_id = api.dispatch("automation_job_start", {"prompts": ["only prompt"]})["job"]["id"]
        job = self._wait_for(job_id)
        outside = self.output_dir / "outside.png"
        outside.write_bytes(b"\x89PNG-outside")
        # A hand-edited record must not be able to name a path outside the job's images.
        prompts = job["prompts"]
        prompts[0]["images"] = ["../../outside.png"]
        automation.update_job(job_id, self.output_dir, prompts=prompts)
        with self.assertRaises(ValueError) as ctx:
            api.dispatch("automation_image_delete", {"id": job_id, "image": "../../outside.png"})
        self.assertIn("not an image name", str(ctx.exception))
        self.assertTrue(outside.exists())
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_the_image_actions_refuse_while_the_job_runs(self):
        self._settings()
        self.stub.hang_calls = {1}
        job_id = api.dispatch("automation_job_start", {"prompts": ["hang", "later"]})["job"]["id"]
        self._wait_until(lambda: self.stub.queued)
        for method, params in (
            ("automation_image_delete", {"id": job_id, "image": "p0001_01.png"}),
            ("automation_image_regenerate", {"id": job_id, "image": "p0001_01.png"}),
            ("automation_prompt_extend", {"id": job_id, "prompt_index": 0, "count": 1}),
            ("automation_prompt_extend_all", {"id": job_id, "count": 1}),
        ):
            with self.assertRaises(ValueError) as ctx:
                api.dispatch(method, params)
            self.assertIn("still running", str(ctx.exception))
        # Editing the text is a record-only change, so it is allowed while the job runs.
        edited = api.dispatch("automation_job_prompt_edit", {"id": job_id, "prompt_index": 0, "text": "fixed"})
        self.assertEqual("fixed", edited["prompts"][0]["text"])
        api.dispatch("automation_job_cancel", {"id": job_id})
        api.dispatch("automation_job_delete", {"id": job_id})

    def test_a_redrawn_image_changes_the_hash_the_client_revalidates_against(self):
        # The Gallery's thumbnails are revalidated by comparing the server's blob hash, which the
        # server derives from the file's mtime and size — so a redraw has to end in a different
        # hash for the new pixels to replace the cached ones on the client.
        self._settings()
        job_id = api.dispatch("automation_job_start", {"prompts": ["only prompt"]})["job"]["id"]
        job = self._wait_for(job_id)
        name = job["prompts"][0]["images"][0]
        image = automation.images_dir(job_id, self.output_dir) / name
        params = {"paths": [str(image)], "max_edge": 128, "quality": 80, "format": "png"}
        before = api.dispatch("blob_stat", params)["items"][0]["hash"]

        self.stub.image = _png_bytes(color=(200, 10, 10))
        api.dispatch("automation_image_regenerate", {"id": job_id, "image": name})
        self._wait_for(job_id)

        after = api.dispatch("blob_stat", params)["items"][0]["hash"]
        self.assertNotEqual(before, after, "the same hash would keep the client on the old picture")
        api.dispatch("automation_job_delete", {"id": job_id})

    def _wait_for(self, job_id, timeout=60.0):
        settings = api.dispatch("automation_config_get", {})["settings"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = automation.read_job(automation.job_path(job_id, settings["output_dir"]))
            if job and job.get("state") != automation.STATE_RUNNING:
                return api.dispatch("automation_job_get", {"id": job_id})
            time.sleep(0.1)
        self.fail(f"job {job_id} never finished")

    def _wait_until(self, predicate, timeout=20.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        self.fail("the condition never came true")


class RealComfyReadOnlyTest(unittest.TestCase):
    """Read-only checks against the machine's own ComfyUI; silent when it is not running."""

    def setUp(self):
        found = comfy.discover(server=REAL_COMFY, budget=1.0)
        if not found["found"]:
            self.skipTest(f"no ComfyUI on {REAL_COMFY}")
        self.found = found
        self.client = comfy.ComfyClient(found["url"])

    def test_discovery_reports_the_version_and_queue(self):
        self.assertTrue(self.found["version"])
        self.assertGreaterEqual(self.found["queue_running"], 0)
        self.assertGreaterEqual(self.found["queue_pending"], 0)

    def test_the_object_info_pre_check_answers_for_every_combo_shape(self):
        info = self.client.object_info()
        shapes = {"new": 0, "old": 0, "other": 0}
        for meta in info.values():
            if not isinstance(meta, dict):
                continue
            declared = meta.get("input") if isinstance(meta.get("input"), dict) else {}
            for bucket in ("required", "optional"):
                for spec in (declared.get(bucket) or {}).values():
                    if isinstance(spec, list) and len(spec) == 2 and spec[0] == "COMBO":
                        shapes["new"] += 1
                    elif isinstance(spec, list) and spec and isinstance(spec[0], list):
                        shapes["old"] += 1
                    else:
                        shapes["other"] += 1
        self.assertGreater(shapes["new"], 0, "0.35 reports at least one new-style combo")
        self.assertGreater(shapes["old"], 0, "…and at least one old-style combo")
        self.assertEqual(["a"], automation.combo_options(["COMBO", {"options": ["a"]}]))
        self.assertEqual(["a"], automation.combo_options([["a"]]))
        self.assertIsNone(automation.combo_options(["INT", {"default": 1}]))


if __name__ == "__main__":
    unittest.main()

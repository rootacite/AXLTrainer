"""Talk to a local ComfyUI, and find the one that is listening.

Torch-free on purpose: `api.py` imports this to answer `automation_*` calls, and
`run_automation.py` runs it in its own process. Two things here are not obvious:

* **Every request bypasses the HTTP proxy.** A desktop session here exports
  `http_proxy`, and urllib then sends a loopback request to the proxy, which
  answers `502` (measured: `curl http://127.0.0.1:8188/system_stats` → 502 through
  the proxy, 200 with `--noproxy '*'`). `ProxyHandler({})` is what makes the
  client work at all in that environment.
* **Discovery probes ports, it does not guess.** The listening ports come from
  `/proc/net/tcp{,6}`; a port counts as ComfyUI only when `/system_stats` answers
  with a `system.comfyui_version` field.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable, Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
# Tried before the rest of the machine's listeners; ComfyUI's own default port.
PREFERRED_PORTS = (8188,)
PROBE_TIMEOUT = 0.5
DISCOVER_BUDGET = 3.0
ENV_URL = "AXL_COMFY_URL"


class ComfyError(RuntimeError):
    """A ComfyUI request failed, with a message Ranko can show as-is."""


class ComfyCancelled(RuntimeError):
    """Waiting for a prompt was interrupted by a stop request."""


def normalize_server(server: str) -> str:
    text = (server or "").strip().rstrip("/")
    if not text:
        return ""
    if not text.startswith(("http://", "https://")):
        text = "http://" + text
    return text


def proxy_free_opener() -> urllib.request.OpenerDirector:
    """An opener that never consults `http_proxy`/`https_proxy`."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _loopback_ips() -> set[int]:
    """Hex-encoded little-endian IPv4 addresses that count as the local machine."""
    return {
        int("0100007F", 16),  # 127.0.0.1
        int("00000000", 16),  # 0.0.0.0: reachable on loopback
    }


def _parse_ports(text: str, wildcard: str) -> set[int]:
    ports: set[int] = set()
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 4 or fields[3] != "0A":  # 0A = LISTEN
            continue
        address, _, port_hex = fields[1].partition(":")
        if not port_hex:
            continue
        try:
            port = int(port_hex, 16)
        except ValueError:
            continue
        host = address.upper()
        if host in ("0100007F", wildcard) or host.endswith("0100007F"):
            ports.add(port)
    return ports


def listening_ports(
    tcp_path: str = "/proc/net/tcp",
    tcp6_path: str = "/proc/net/tcp6",
) -> list[int]:
    """Ports this machine listens on, IPv4 and IPv6, loopback and wildcard bindings.

    A `0.0.0.0` / `::` listener is reachable on loopback, so it is a candidate too.
    """
    ports: set[int] = set()
    for path, wildcard in ((tcp_path, "00000000"), (tcp6_path, "0" * 32)):
        try:
            with open(path, encoding="utf-8") as handle:
                ports |= _parse_ports(handle.read(), wildcard)
        except OSError:
            continue
    return sorted(ports)


def candidate_servers(server: Optional[str] = None, ports: Optional[Iterable[int]] = None) -> list[str]:
    """The URLs discovery probes, in order: an explicit/inherited server first."""
    explicit = normalize_server(server or os.environ.get(ENV_URL) or "")
    if explicit:
        return [explicit]
    known = list(ports) if ports is not None else listening_ports()
    ordered = [port for port in PREFERRED_PORTS if port in known]
    ordered.extend(port for port in sorted(known) if port not in ordered)
    if not ordered:
        ordered = list(PREFERRED_PORTS)
    return [f"http://{DEFAULT_HOST}:{port}" for port in ordered]


class ComfyClient:
    def __init__(
        self,
        server: str,
        timeout: float = 30.0,
        opener: Optional[urllib.request.OpenerDirector] = None,
        client_id: str = "axlranko",
    ) -> None:
        self.server = normalize_server(server)
        if not self.server:
            raise ComfyError("no ComfyUI server address")
        self.timeout = timeout
        self.client_id = client_id
        self._opener = opener or proxy_free_opener()

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[bytes] = None,
        content_type: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(self.server + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout if timeout is None else timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ComfyError(f"HTTP {exc.code} from {path}: {body[:400]}") from exc
        except urllib.error.URLError as exc:
            raise ComfyError(f"Cannot connect to ComfyUI at {self.server}: {exc}") from exc
        except http.client.HTTPException as exc:
            # A loopback listener that is not an HTTP server at all (sshd answers with a banner,
            # for instance): report it as "not a ComfyUI" instead of letting it escape.
            raise ComfyError(f"{self.server} did not answer HTTP: {type(exc).__name__}: {exc}") from exc
        except TimeoutError as exc:
            raise ComfyError(f"ComfyUI at {self.server} did not answer {path}") from exc

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        raw = self._request(method, path, **kwargs)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ComfyError(f"ComfyUI returned invalid JSON for {path}") from exc
        if not isinstance(payload, dict):
            raise ComfyError(f"ComfyUI returned an unexpected payload for {path}")
        return payload

    def system_stats(self) -> dict[str, Any]:
        return self._json("GET", "/system_stats")

    def object_info(self, node: Optional[str] = None) -> dict[str, Any]:
        path = "/object_info" + (f"/{urllib.parse.quote(node, safe='')}" if node else "")
        return self._json("GET", path)

    def queue(self) -> dict[str, Any]:
        return self._json("GET", "/queue")

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": workflow, "client_id": self.client_id},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        result = self._json("POST", "/prompt", data=payload, content_type="application/json")
        node_errors = result.get("node_errors")
        if node_errors:
            raise ComfyError(
                "ComfyUI rejected the workflow: "
                + json.dumps(node_errors, ensure_ascii=False)[:800]
            )
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise ComfyError("ComfyUI did not return a prompt_id")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any]:
        return self._json("GET", f"/history/{urllib.parse.quote(prompt_id, safe='')}")

    def wait_for_prompt(
        self,
        prompt_id: str,
        poll_interval: float = 0.5,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """Poll until the prompt finishes; raises [ComfyCancelled] when asked to stop."""
        while True:
            if should_stop is not None and should_stop():
                raise ComfyCancelled("cancelled while waiting for ComfyUI")
            entry = self.history(prompt_id).get(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("completed"):
                    if status.get("status_str") != "success":
                        messages = status.get("messages")
                        raise ComfyError(
                            "ComfyUI execution failed: " + json.dumps(messages, ensure_ascii=False)[:800]
                        )
                    return entry
            time.sleep(poll_interval)

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        query = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        return self._request("GET", f"/view?{query}", timeout=max(self.timeout, 120.0))

    def version(self) -> Optional[str]:
        try:
            stats = self.system_stats()
        except ComfyError:
            return None
        system = stats.get("system")
        if isinstance(system, dict):
            value = system.get("comfyui_version")
            if isinstance(value, str) and value:
                return value
        return None


def discover(
    server: Optional[str] = None,
    timeout: float = PROBE_TIMEOUT,
    budget: float = DISCOVER_BUDGET,
    ports: Optional[Iterable[int]] = None,
    opener: Optional[urllib.request.OpenerDirector] = None,
) -> dict[str, Any]:
    """Find a listening ComfyUI.

    `server` (or `$AXL_COMFY_URL`) is trusted as-is once it answers; otherwise every
    loopback listener is probed and only a `system.comfyui_version` answer counts.
    """
    explicit = normalize_server(server or os.environ.get(ENV_URL) or "")
    checked: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(0.5, budget)
    candidates = candidate_servers(server or None, ports=ports)
    probed_all = True
    for index, url in enumerate(candidates):
        if index and time.monotonic() > deadline:
            probed_all = False
            break
        client = ComfyClient(url, timeout=timeout, opener=opener)
        try:
            version = client.version()
        except ComfyError as exc:
            checked.append({"url": url, "ok": False, "reason": str(exc)[:160]})
            continue
        except Exception as exc:  # noqa: BLE001 - one odd listener must not break discovery
            checked.append({"url": url, "ok": False, "reason": f"{type(exc).__name__}: {exc}"[:160]})
            continue
        if version:
            queue = {}
            try:
                queue = client.queue()
            except ComfyError:
                queue = {}
            running = queue.get("queue_running")
            pending = queue.get("queue_pending")
            return {
                "found": True,
                "url": url,
                "version": version,
                "queue_running": len(running) if isinstance(running, list) else 0,
                "queue_pending": len(pending) if isinstance(pending, list) else 0,
                "checked": checked,
                "probed_all": probed_all,
            }
        checked.append({"url": url, "ok": False, "reason": "not a ComfyUI (no comfyui_version)"})
    return {
        "found": False,
        "url": "",
        "version": "",
        "queue_running": 0,
        "queue_pending": 0,
        "checked": checked,
        "probed_all": probed_all,
    }

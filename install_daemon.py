#!/usr/bin/env python3
"""Install the wasm Ranko site and one user systemd unit that serves it with api.py.

The static site listens on 0.0.0.0:18766 and has no allowlist of its own: it is only a client.
api.py listens on 0.0.0.0:18765 and admits 192.168.0.0/16 (loopback is always admitted on top).
One unit supervises both processes.

    python install_daemon.py             build, install, enable --user
    python install_daemon.py --uninstall disable the unit and remove what this script installed
    python install_daemon.py --serve     what the unit's ExecStart runs

Refuses to run as root. Does not enable linger, does not stop a helper already listening, and
does not start training.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

UNIT_NAME = "axlranko-daemon.service"
WEB_PORT = 18766
API_PORT = 18765
ALLOW_CIDR = "192.168.0.0/16"
ENV_NAME = "axl"
_CONDA_HOMES = ("miniconda3", "anaconda3", "miniforge3")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def web_dest() -> Path:
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "axlranko" / "web"


def unit_path() -> Path:
    config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config) / "systemd" / "user" / UNIT_NAME


def refuse_root() -> None:
    if os.geteuid() == 0:
        raise SystemExit(
            "install_daemon.py: refusing to install for root; run it as the user who will log in"
        )


def wasm_dist_dir(ranko_dir: Path) -> Path:
    return ranko_dir / "webApp" / "build" / "dist" / "wasmJs" / "productionExecutable"


def child_commands(python: str, repo: Path, web_root: Path) -> list[list[str]]:
    """The two processes the unit supervises. `--serve` and the unit text both come from here."""
    api = [
        python,
        "-u",
        str(repo / "api.py"),
        "--host",
        "0.0.0.0",
        "--port",
        str(API_PORT),
        "--allow-ip",
        ALLOW_CIDR,
    ]
    web = [
        python,
        "-m",
        "http.server",
        str(WEB_PORT),
        "--bind",
        "0.0.0.0",
        "--directory",
        str(web_root),
    ]
    return [api, web]


def command_line(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def render_unit(python: str, repo: Path, web_root: Path) -> str:
    """A user unit. The child command lines are comments generated from `child_commands`."""
    api_cmd, web_cmd = child_commands(python, repo, web_root)
    serve = command_line([python, "-u", str(repo / "install_daemon.py"), "--serve"])
    return (
        "[Unit]\n"
        "Description=AxlRanko wasm site and api.py helper\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={repo}\n"
        f"Environment=AXL_PYTHON={python}\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        f"ExecStart={serve}\n"
        "Restart=on-failure\n"
        "RestartSec=2\n"
        "\n"
        "# Children started by --serve. The static site has no allowlist of its own.\n"
        f"# api: {command_line(api_cmd)}\n"
        f"# web: {command_line(web_cmd)}\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def install_web_tree(src: Path, dest: Path) -> None:
    """Replace `dest` with a copy of the wasm distribution."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _conda_python(base: Path) -> Optional[Path]:
    candidate = base / "envs" / ENV_NAME / "bin" / "python"
    return candidate if candidate.is_file() else None


def resolve_python(
    environ: Optional[Mapping[str, str]] = None,
    *,
    home: Optional[Path] = None,
    opt_conda: Path = Path("/opt/conda"),
) -> str:
    """The axl interpreter the unit will pin.

    `AXL_PYTHON` wins. Otherwise the conda env named `axl` (from `CONDA_EXE`, then `conda info
    --base`, then the usual install prefixes). If the current process is already that env, its
    own interpreter is used. Anything else is a hard stop — the unit must not guess.
    """
    env = os.environ if environ is None else environ
    home_dir = Path.home() if home is None else home
    override = (env.get("AXL_PYTHON") or "").strip()
    if override:
        if not Path(override).is_file():
            raise SystemExit(f"install_daemon.py: AXL_PYTHON is not a file: {override}")
        return override

    bases: list[Path] = []
    conda_exe = (env.get("CONDA_EXE") or "").strip()
    if conda_exe and Path(conda_exe).is_file():
        bases.append(Path(conda_exe).resolve().parent.parent)
    # An explicit empty PATH must not fall through to this process's own PATH.
    conda_bin = shutil.which("conda", path=env.get("PATH", ""))
    if conda_bin:
        try:
            probed = subprocess.run(
                [conda_bin, "info", "--base"],
                check=False,
                capture_output=True,
                text=True,
                env=dict(env),
            )
        except OSError:
            probed = None
        if probed is not None and probed.returncode == 0:
            text = (probed.stdout or "").strip()
            if text:
                bases.append(Path(text))
    for name in _CONDA_HOMES:
        bases.append(home_dir / name)
    bases.append(opt_conda)

    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        found = _conda_python(base)
        if found is not None:
            return str(found)

    if (env.get("CONDA_DEFAULT_ENV") or "").strip() == ENV_NAME:
        return sys.executable
    raise SystemExit(
        "install_daemon.py: could not find the axl interpreter; set AXL_PYTHON to it"
    )


def build_wasm(ranko_dir: Path) -> Path:
    gradle = ranko_dir / "gradlew"
    if not gradle.is_file():
        raise SystemExit(f"install_daemon.py: missing gradle wrapper: {gradle}")
    subprocess.run(
        [str(gradle), ":webApp:wasmJsBrowserDistribution", "--no-daemon"],
        cwd=str(ranko_dir),
        check=True,
    )
    dist = wasm_dist_dir(ranko_dir)
    if not dist.is_dir():
        raise SystemExit(f"install_daemon.py: wasm distribution missing: {dist}")
    return dist


def _local_urls() -> list[str]:
    lines = [f"http://0.0.0.0:{WEB_PORT}", f"ws://0.0.0.0:{API_PORT}"]
    hosts: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ip not in hosts and not ip.startswith("127.") and ip != "::1":
                hosts.append(ip)
    except OSError:
        hosts = []
    for ip in hosts:
        lines.append(f"http://{ip}:{WEB_PORT}")
    return lines


def install() -> None:
    refuse_root()
    repo = repo_root()
    python = resolve_python()
    dist = build_wasm(repo / "ranko")
    dest = web_dest()
    install_web_tree(dist, dest)
    text = render_unit(python, repo, dest)
    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", UNIT_NAME], check=True)
    print(f"installed {dest}")
    print(f"installed {path}")
    print("The static site has no allowlist. api.py admits " + ALLOW_CIDR + " and loopback.")
    for line in _local_urls():
        print(line)


def uninstall() -> None:
    refuse_root()
    subprocess.run(["systemctl", "--user", "disable", "--now", UNIT_NAME], check=False)
    path = unit_path()
    path.unlink(missing_ok=True)
    dest = web_dest()
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    print(f"removed {path}")
    print(f"removed {dest}")


def serve() -> int:
    """Run both children. Exit 0 on SIGTERM; exit 1 if either child stops on its own."""
    refuse_root()
    repo = repo_root()
    python = os.environ.get("AXL_PYTHON") or resolve_python()
    web_root = web_dest()
    if not web_root.is_dir():
        raise SystemExit(f"install_daemon.py: wasm site is not installed at {web_root}")
    commands = child_commands(python, repo, web_root)
    procs: list[subprocess.Popen[bytes]] = []
    signaled = False

    def _stop(signum: int, _frame: object) -> None:
        nonlocal signaled
        signaled = True
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    for cmd in commands:
        procs.append(subprocess.Popen(cmd, cwd=str(repo)))

    child_died = False
    while not signaled and not child_died:
        for proc in procs:
            if proc.poll() is not None:
                child_died = True
                break
        else:
            time.sleep(0.25)
            continue
        break

    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 5
    for proc in procs:
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if signaled and not child_died:
        return 0
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Install the wasm Ranko site and its user service")
    parser.add_argument("--uninstall", action="store_true", help="remove the unit and the copied site")
    parser.add_argument("--serve", action="store_true", help="run api.py and the static site (the unit)")
    args = parser.parse_args(argv)
    if args.serve and args.uninstall:
        raise SystemExit("install_daemon.py: --serve and --uninstall cannot be combined")
    if args.serve:
        return serve()
    if args.uninstall:
        uninstall()
        return 0
    install()
    return 0


if __name__ == "__main__":
    sys.exit(main())

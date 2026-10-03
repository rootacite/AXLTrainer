"""Run one Automation job: push prompts through a ComfyUI workflow, keep the images.

    python -u trainer/run_automation.py --spec <job.json> [--only-failed]
                                        [--image p0003_01.png
                                         | --append 2 --images 4
                                         | --append-all --images 4]

`api.py` writes the job record (see `trainer/automation.py`) and spawns this detached,
so a long batch survives Chromatrix closing. Progress, seeds and ComfyUI's own file names go
back into the job file; the images land in `<job>/images/` under our own names
(`p0003_01.png`) so the order never depends on ComfyUI's `filename_prefix`.

`--only-failed` re-runs the prompts that produced nothing yet. The targeted forms narrow the
pass down, which is what the Gallery's actions ask for: `--image` redraws that one image in
place (new random seed, its sidecar rewritten), `--append <index> --images N` adds N images
to that one entry, and `--append-all --images N` adds N to every entry — each image from its
own random seed, appended after the images the entry already has.

A SIGTERM (the Cancel button) stops between prompts and inside a history poll, then the
job is marked `cancelled` — the trainer's own rule that a half-finished run keeps what
it already produced.
"""

from __future__ import annotations

import argparse
import copy
import os
import secrets
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

# `python trainer/run_automation.py` puts trainer/ on sys.path, `import api` does not;
# support both (see AGENT.md "Import dualism").
try:
    import automation
    import comfy
except ImportError:
    from trainer import automation
    from trainer import comfy

MAX_CONSECUTIVE_FAILURES = 3


class Stop:
    """The runner's stop flag: set by SIGTERM/SIGINT, read between prompts and polls."""

    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def install(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self.requested = True
            self.reason = signal.Signals(signum).name

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


def _log(message: str) -> None:
    print(f"[automation] {message}", flush=True)


def set_positive_prompt(workflow: dict[str, Any], node_id: str, prompt: str) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        raise comfy.ComfyError(f"positive CLIP node {node_id!r} does not exist")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise comfy.ComfyError(f"node {node_id!r} has no inputs object")
    if "text" not in inputs:
        raise comfy.ComfyError(f"node {node_id!r} has no 'text' input")
    inputs["text"] = prompt


def set_batch_size(workflow: dict[str, Any], batch_size: int) -> list[str]:
    changed: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        value = inputs.get("batch_size")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            inputs["batch_size"] = batch_size
            changed.append(str(node_id))
    return changed


def seed_targets(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    """Every numeric `seed` input, including one hop through a linked seed node."""
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or "seed" not in inputs:
            continue
        value = inputs["seed"]
        if isinstance(value, int) and not isinstance(value, bool):
            key = (str(node_id), "seed")
            if key not in seen:
                seen.add(key)
                targets.append(key)
            continue
        if isinstance(value, list) and value and isinstance(value[0], str):
            upstream = workflow.get(value[0])
            if not isinstance(upstream, dict):
                continue
            upstream_inputs = upstream.get("inputs")
            if isinstance(upstream_inputs, dict) and isinstance(upstream_inputs.get("seed"), int):
                key = (value[0], "seed")
                if key not in seen:
                    seen.add(key)
                    targets.append(key)
    return targets


def set_seed(workflow: dict[str, Any], seed: int) -> list[str]:
    targets = seed_targets(workflow)
    if not targets:
        raise comfy.ComfyError("the workflow has no numeric 'seed' input to vary")
    changed: list[str] = []
    for node_id, input_name in targets:
        workflow[node_id]["inputs"][input_name] = seed
        changed.append(node_id)
    return changed


def saved_images(history_entry: dict[str, Any], save_nodes: set[str]) -> list[dict[str, str]]:
    """Only SaveImage outputs; the workflow's PreviewImage nodes are ignored."""
    outputs = history_entry.get("outputs") or {}
    found: list[dict[str, str]] = []
    if not isinstance(outputs, dict):
        return found
    for node_id, node_output in outputs.items():
        if str(node_id) not in save_nodes or not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict) or not image.get("filename"):
                continue
            found.append(
                {
                    "filename": str(image["filename"]),
                    "subfolder": str(image.get("subfolder") or ""),
                    "type": str(image.get("type") or "output"),
                }
            )
    return found


def write_sidecar(path: Path, prompt: str, seed: int, prompt_id: str, comfy_name: str) -> None:
    path.write_text(
        f"seed: {seed}\n"
        f"prompt_id: {prompt_id}\n"
        f"prompt: {prompt}\n"
        f"comfy_filename: {comfy_name}\n",
        encoding="utf-8",
    )


def aligned_seeds(entry: Mapping[str, Any], names: list[str]) -> list[Any]:
    """One seed per name. A record from before the Gallery kept per-image seeds — or one whose
    list does not line up with its names — falls back to the entry's single `seed`, which is the
    number the Gallery showed for those images anyway."""
    seeds = entry.get("image_seeds")
    if isinstance(seeds, list) and len(seeds) == len(names):
        return list(seeds)
    row = entry.get("seed")
    return [row if isinstance(row, int) else None] * len(names)


def output_name(
    plan: Mapping[str, Any],
    index: int,
    image_index: int,
    names: list[str],
    images_dir: Path,
) -> str:
    """Where one rendered image lands. A redraw writes over the name it was asked to replace, a
    normal pass numbers the batch (`p0003_01.png`…), and an append takes the next free number so
    a name that was deleted cannot come back."""
    fixed = plan.get("fixed")
    if isinstance(fixed, list) and image_index <= len(fixed):
        return str(fixed[image_index - 1])
    if plan.get("redraw") and not fixed:
        on_disk = [path.name for path in images_dir.glob(f"p{index + 1:04d}_*.png")]
        return automation.image_name(index, automation.next_image_number(sorted(set(names) | set(on_disk)), index))
    return automation.image_name(index, image_index)


def run_job(spec_path: Path, only_failed: bool = False, target: Optional[Mapping[str, Any]] = None) -> int:
    spec = automation.read_job(spec_path)
    if spec is None:
        print(f"[automation] unreadable spec: {spec_path}", file=sys.stderr)
        return 2
    job_id = str(spec.get("id") or spec_path.stem)
    output_dir = spec.get("output_dir") or spec_path.parent.parent
    images = automation.images_dir(job_id, output_dir)
    images.mkdir(parents=True, exist_ok=True)

    stop = Stop()
    stop.install()

    prompts = spec.get("prompts") if isinstance(spec.get("prompts"), list) else []
    if not prompts:
        automation.update_job(
            job_id, output_dir, state=automation.STATE_ERROR, error="the job has no prompts", finished_at=time.time()
        )
        return 1

    started = time.time()
    automation.update_job(
        job_id,
        output_dir,
        state=automation.STATE_RUNNING,
        pid=os.getpid(),
        started_at=spec.get("started_at") or started,
        finished_at=None,
        error=None,
    )

    try:
        workflow_path = spec.get("workflow_path") or ""
        base_workflow = automation.load_workflow(workflow_path)
        positive = str(spec.get("positive_node") or "")
        count = int(spec.get("count") or 1)
        poll = float(spec.get("poll") or 0.5)
        report = automation.validate_workflow(base_workflow, positive_node=positive)
        if not report["valid"]:
            raise comfy.ComfyError(f"workflow is not usable: {report['error']}")
        if count > 1 and not report["batch_size_nodes"]:
            raise comfy.ComfyError("the workflow has no numeric batch_size input, so 'images per prompt' cannot apply")
        save_nodes = set(report["save_image_nodes"])
        positive = report["positive_node"] or positive

        url = str(spec.get("server") or "")
        if not url:
            found = comfy.discover()
            if not found["found"]:
                raise comfy.ComfyError("no ComfyUI found listening on this machine")
            url = found["url"]
        client = comfy.ComfyClient(url)
        automation.update_job(job_id, output_dir, comfy_url=client.server)

        # What this pass runs. A normal pass walks every entry and queues each once with the
        # configured images-per-prompt; a targeted pass (the Gallery's actions) narrows that down —
        # `image` redraws one image in place, `append` adds `images` more to one entry, and
        # `append_all` adds `images` more to every entry, each image from its own random seed.
        mode = str((target or {}).get("mode") or "")
        queues = max(1, int((target or {}).get("images") or 1))
        plans: list[dict[str, Any]] = []
        if not mode:
            for position, prompt in enumerate(prompts):
                if not isinstance(prompt, dict):
                    continue
                if only_failed and prompt.get("state") == automation.PROMPT_STATE_DONE:
                    continue
                plans.append({"index": position, "queues": 1, "batch": count, "redraw": False, "fixed": None})
        elif mode == "image":
            wanted = str((target or {}).get("name") or "")
            position = automation.prompt_entry_index(prompts, wanted)
            if position is None:
                raise comfy.ComfyError(f"no prompt holds the image {wanted}")
            plans.append({"index": position, "queues": 1, "batch": 1, "redraw": True, "fixed": [wanted]})
        elif mode == "append_all":
            plans = [
                {"index": position, "queues": queues, "batch": 1, "redraw": True, "fixed": None}
                for position, prompt in enumerate(prompts)
                if isinstance(prompt, dict)
            ]
            if not plans:
                raise comfy.ComfyError("this job has no prompt to append to")
        else:
            position = int((target or {}).get("index") or 0)
            if automation.prompt_entry_at(prompts, position) is None:
                raise comfy.ComfyError(f"no prompt at index {position}")
            plans.append(
                {
                    "index": position,
                    "queues": queues,
                    "batch": 1,
                    "redraw": True,
                    "fixed": None,
                }
            )
        if mode:
            _log(f"{mode} pass: {len(plans)} entry(s), {queues} image(s) each")

        planned_images = sum(int(plan["queues"]) for plan in plans)
        written_images = 0
        if mode:
            automation.set_pass(
                job_id,
                output_dir,
                {
                    "mode": mode,
                    "prompt_index": plans[0]["index"],
                    "images_done": 0,
                    "total_images": planned_images,
                    "image": None,
                },
            )

        consecutive_failures = 0
        completed_any = False
        counter = 0
        for plan in plans:
            if stop.requested:
                break
            index = plan["index"]
            prompt_entry = automation.prompt_entry_at(prompts, index) or {}
            prompt_text = str(prompt_entry.get("text") or "")
            names = [str(name) for name in prompt_entry.get("images") or []] if plan["redraw"] else []
            seeds: list[Any] = aligned_seeds(prompt_entry, names) if plan["redraw"] else []
            if plan["redraw"]:
                # A redraw keeps the images the entry already has; only their seeds are rewritten.
                _record_prompt(job_id, output_dir, index, state=automation.PROMPT_STATE_RUNNING, error=None)
            else:
                _record_prompt(
                    job_id,
                    output_dir,
                    index,
                    state=automation.PROMPT_STATE_RUNNING,
                    error=None,
                    seed=None,
                    prompt_id=None,
                    images=[],
                )
            try:
                for _ in range(plan["queues"]):
                    if stop.requested:
                        break
                    counter += 1
                    seed = secrets.randbits(63)
                    workflow = copy.deepcopy(base_workflow)
                    set_positive_prompt(workflow, positive, prompt_text)
                    set_batch_size(workflow, plan["batch"])
                    changed_seeds = set_seed(workflow, seed)
                    _log(f"[{counter}] seed={seed} count={plan['batch']} seed nodes={changed_seeds}")
                    prompt_id = client.queue_prompt(workflow)
                    _record_prompt(job_id, output_dir, index, prompt_id=prompt_id, seed=seed)
                    _log(f"[{counter}] queued {prompt_id}")
                    entry = client.wait_for_prompt(prompt_id, poll_interval=poll, should_stop=lambda: stop.requested)
                    found = saved_images(entry, save_nodes)
                    if not found:
                        raise comfy.ComfyError("the prompt finished but SaveImage returned no image")
                    for image_index, image in enumerate(found, start=1):
                        data = client.get_image(image["filename"], image["subfolder"], image["type"])
                        name = output_name(plan, index, image_index, names, images)
                        (images / name).write_bytes(data)
                        write_sidecar(images / name.replace(".png", ".txt"), prompt_text, seed, prompt_id, image["filename"])
                        if name in names:
                            seeds[names.index(name)] = seed
                        else:
                            names.append(name)
                            seeds.append(seed)
                        _record_prompt(job_id, output_dir, index, images=names, image_seeds=seeds)
                        _log(f"[{counter}] [{image_index}/{len(found)}] {name} ({len(data)} bytes)")
                        if mode:
                            written_images += 1
                            automation.set_pass(
                                job_id,
                                output_dir,
                                {
                                    "mode": mode,
                                    "prompt_index": index,
                                    "images_done": written_images,
                                    "total_images": planned_images,
                                    "image": name,
                                },
                            )
                _record_prompt(
                    job_id,
                    output_dir,
                    index,
                    state=automation.PROMPT_STATE_DONE,
                    images=names,
                    image_seeds=seeds,
                    finished_at=time.time(),
                )
                completed_any = True
                consecutive_failures = 0
            except comfy.ComfyCancelled:
                break
            except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill the batch
                traceback.print_exc()
                _record_prompt(
                    job_id,
                    output_dir,
                    index,
                    state=automation.PROMPT_STATE_ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                    images=names,
                    image_seeds=seeds,
                    finished_at=time.time(),
                )
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise comfy.ComfyError(
                        f"{consecutive_failures} prompts failed in a row; stopping"
                    ) from exc
    except Exception as exc:  # noqa: BLE001 - the job file is what the client reads
        traceback.print_exc()
        automation.set_pass(job_id, output_dir, None)
        automation.update_job(
            job_id,
            output_dir,
            state=automation.STATE_CANCELLED if stop.requested else automation.STATE_ERROR,
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )
        return 1

    if mode:
        # The pass is over (done, stopped or failed): the Gallery's progress line goes with it.
        automation.set_pass(job_id, output_dir, None)

    if stop.requested:
        _log(f"stopped on {stop.reason}")
        automation.update_job(
            job_id,
            output_dir,
            state=automation.STATE_CANCELLED,
            error=f"cancelled ({stop.reason})" if stop.reason else "cancelled",
            finished_at=time.time(),
        )
        return 0

    job_now = automation.read_job(automation.job_path(job_id, output_dir)) or {}
    job_prompts = job_now.get("prompts") if isinstance(job_now.get("prompts"), list) else []
    failed = sum(1 for p in job_prompts if isinstance(p, dict) and p.get("state") == automation.PROMPT_STATE_ERROR)
    done = sum(1 for p in job_prompts if isinstance(p, dict) and p.get("state") == automation.PROMPT_STATE_DONE)
    if job_prompts and not done and failed:
        automation.update_job(
            job_id,
            output_dir,
            state=automation.STATE_ERROR,
            error=f"every prompt failed ({failed})",
            finished_at=time.time(),
        )
        return 1
    automation.update_job(
        job_id,
        output_dir,
        state=automation.STATE_DONE,
        error=None,
        finished_at=time.time(),
    )
    _log(f"done: {done} prompt(s), {failed} failed, {time.time() - started:.1f}s")
    return 0


def _record_prompt(job_id: str, output_dir: Any, index: int, **fields: Any) -> None:
    """Update one prompt inside the job record; api.py never loses fields it wrote."""
    path = automation.job_path(job_id, output_dir)
    payload = automation.read_job(path) or {"id": job_id}
    prompts = payload.get("prompts")
    if not isinstance(prompts, list):
        prompts = []
    while len(prompts) <= index:
        prompts.append({"index": len(prompts), "text": "", "state": automation.PROMPT_STATE_PENDING, "images": []})
    entry = prompts[index]
    if not isinstance(entry, dict):
        entry = {"index": index, "text": "", "state": automation.PROMPT_STATE_PENDING, "images": []}
        prompts[index] = entry
    entry.update(fields)
    payload["prompts"] = prompts
    payload["updated_at"] = time.time()
    automation.atomic_write_json(path, payload)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Automation job against ComfyUI")
    parser.add_argument("--spec", required=True, help="job JSON written by api.py")
    parser.add_argument("--only-failed", action="store_true", help="skip the prompts that already produced images")
    parser.add_argument("--image", help="redraw this one image in place, with a new random seed")
    parser.add_argument("--append", type=int, help="append images to this prompt index")
    parser.add_argument("--append-all", action="store_true", help="append images to every prompt")
    parser.add_argument("--images", type=int, default=1, help="how many images an append pass adds")
    args = parser.parse_args(argv)

    chosen = [
        flag
        for flag, given in (
            ("--image", bool(args.image)),
            ("--append", args.append is not None),
            ("--append-all", bool(args.append_all)),
        )
        if given
    ]
    if len(chosen) > 1:
        print(f"[automation] {chosen[0]} and {chosen[1]} are two different passes", file=sys.stderr)
        return 2
    target: Optional[dict[str, Any]] = None
    if args.image:
        target = {"mode": "image", "name": str(args.image)}
    elif args.append is not None:
        target = {"mode": "append", "index": int(args.append), "images": max(1, int(args.images))}
    elif args.append_all:
        target = {"mode": "append_all", "images": max(1, int(args.images))}
    return run_job(Path(args.spec), only_failed=args.only_failed, target=target)


if __name__ == "__main__":
    sys.exit(main())

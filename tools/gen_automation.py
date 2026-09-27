#!/usr/bin/env python3

import argparse
import copy
import json
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def normalize_server(server: str) -> str:
    server = server.strip().rstrip("/")
    if not server.startswith(("http://", "https://")):
        server = "http://" + server
    return server


class ComfyUIClient:
    def __init__(self, server: str, timeout: float = 30.0):
        self.server = normalize_server(server)
        self.client_id = str(uuid.uuid4())
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        data: bytes | None = None,
        content_type: str | None = None,
        timeout: float | None = None,
    ) -> bytes:
        headers = {}

        if content_type is not None:
            headers["Content-Type"] = content_type

        request = urllib.request.Request(
            self.server + path,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout if timeout is None else timeout,
            ) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code} from {path}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot connect to ComfyUI at {self.server}: {exc}"
            ) from exc

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        response = self._request(
            "POST",
            "/prompt",
            data=data,
            content_type="application/json",
        )

        result = json.loads(response)

        if "node_errors" in result and result["node_errors"]:
            raise RuntimeError(
                "ComfyUI rejected the workflow:\n"
                + json.dumps(
                    result["node_errors"],
                    ensure_ascii=False,
                    indent=2,
                )
            )

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(
                "ComfyUI /prompt response does not contain prompt_id:\n"
                + json.dumps(result, ensure_ascii=False, indent=2)
            )

        return prompt_id

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"/history/{urllib.parse.quote(prompt_id, safe='')}",
        )
        return json.loads(response)

    def wait_for_prompt(
        self,
        prompt_id: str,
        poll_interval: float = 0.5,
    ) -> dict[str, Any]:
        while True:
            history = self.get_history(prompt_id)
            entry = history.get(prompt_id)

            if entry is not None:
                status = entry.get("status", {})
                completed = status.get("completed", False)
                status_str = status.get("status_str")

                if completed:
                    if status_str != "success":
                        messages = status.get("messages", [])
                        raise RuntimeError(
                            "ComfyUI execution failed:\n"
                            + json.dumps(
                                messages,
                                ensure_ascii=False,
                                indent=2,
                            )
                        )

                    return entry

            time.sleep(poll_interval)

    def get_image(
        self,
        filename: str,
        subfolder: str,
        folder_type: str,
    ) -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            }
        )

        return self._request(
            "GET",
            f"/view?{query}",
            timeout=120.0,
        )


def set_positive_prompt(
    workflow: dict[str, Any],
    node_id: str,
    prompt: str,
) -> None:
    node = workflow.get(node_id)

    if node is None:
        raise ValueError(
            f"Positive CLIP node {node_id!r} does not exist."
        )

    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(
            f"Node {node_id!r} has no valid inputs object."
        )

    if "text" not in inputs:
        raise ValueError(
            f"Node {node_id!r} has no 'text' input."
        )

    inputs["text"] = prompt


def set_batch_size(
    workflow: dict[str, Any],
    batch_size: int,
) -> list[str]:
    changed = []

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue

        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        if "batch_size" in inputs and isinstance(
            inputs["batch_size"],
            (int, float),
        ):
            inputs["batch_size"] = batch_size
            changed.append(node_id)

    return changed


def collect_seed_targets(
    workflow: dict[str, Any],
) -> list[tuple[str, str]]:
    """
    Find seed-bearing nodes.

    Supports both:

        "seed": 123456

    and:

        "seed": ["224", 0]

    In the latter case, the upstream node is inspected and its own
    'seed' input is modified.

    Returns:
        [(node_id, input_name), ...]
    """
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue

        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        seed_input = inputs.get("seed")
        if seed_input is None:
            continue

        # Direct integer seed.
        if isinstance(seed_input, int):
            key = (node_id, "seed")
            if key not in seen:
                seen.add(key)
                targets.append(key)
            continue

        # Seed linked from another node, e.g. ["224", 0].
        if (
            isinstance(seed_input, list)
            and len(seed_input) >= 1
            and isinstance(seed_input[0], str)
        ):
            upstream_id = seed_input[0]
            upstream = workflow.get(upstream_id)

            if not isinstance(upstream, dict):
                continue

            upstream_inputs = upstream.get("inputs")
            if not isinstance(upstream_inputs, dict):
                continue

            if isinstance(upstream_inputs.get("seed"), int):
                key = (upstream_id, "seed")
                if key not in seen:
                    seen.add(key)
                    targets.append(key)

    return targets


def set_seed(
    workflow: dict[str, Any],
    seed: int,
) -> list[str]:
    targets = collect_seed_targets(workflow)

    if not targets:
        raise ValueError(
            "Could not find any numeric seed input in the workflow."
        )

    changed = []

    for node_id, input_name in targets:
        workflow[node_id]["inputs"][input_name] = seed
        changed.append(node_id)

    return changed


def find_save_image_nodes(
    workflow: dict[str, Any],
) -> set[str]:
    return {
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict)
        and node.get("class_type") == "SaveImage"
    }


def collect_saved_images(
    history_entry: dict[str, Any],
    save_nodes: set[str],
) -> list[tuple[str, str, str, str]]:
    """
    Returns:

        (save_node_id, filename, subfolder, type)
    """
    outputs = history_entry.get("outputs", {})
    result = []

    for node_id, node_output in outputs.items():
        if node_id not in save_nodes:
            continue

        if not isinstance(node_output, dict):
            continue

        images = node_output.get("images", [])
        if not isinstance(images, list):
            continue

        for image in images:
            if not isinstance(image, dict):
                continue

            filename = image.get("filename")
            subfolder = image.get("subfolder", "")
            folder_type = image.get("type", "output")

            if not filename:
                continue

            result.append(
                (
                    node_id,
                    filename,
                    subfolder,
                    folder_type,
                )
            )

    return result


def write_metadata(
    path: Path,
    prompt: str,
    seed: int,
    prompt_id: str,
) -> None:
    text = (
        f"seed: {seed}\n"
        f"prompt_id: {prompt_id}\n"
        f"prompt: {prompt}\n"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read prompts from stdin and execute a ComfyUI "
            "API-format workflow once per line."
        )
    )

    parser.add_argument(
        "--server",
        default="127.0.0.1:8188",
        help="ComfyUI server address, default: 127.0.0.1:8188",
    )

    parser.add_argument(
        "--workflow",
        required=True,
        type=Path,
        help="ComfyUI workflow exported as API format JSON",
    )

    parser.add_argument(
        "--positive-node",
        required=True,
        help="Node ID of the positive CLIPTextEncode",
    )

    parser.add_argument(
        "--count",
        required=True,
        type=int,
        help="Number of images to generate for each prompt",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Local directory where downloaded images and txt files are stored",
    )

    parser.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="History polling interval in seconds, default: 0.5",
    )

    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be greater than zero")

    if args.poll <= 0:
        parser.error("--poll must be greater than zero")

    if not args.workflow.is_file():
        parser.error(
            f"Workflow file does not exist: {args.workflow}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        base_workflow = json.loads(
            args.workflow.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Invalid workflow JSON: {exc}"
        )

    if not isinstance(base_workflow, dict):
        raise SystemExit(
            "Workflow JSON root must be an object."
        )

    save_nodes = find_save_image_nodes(base_workflow)

    if not save_nodes:
        raise SystemExit(
            "Workflow contains no SaveImage node."
        )

    # Validate the positive node before consuming stdin.
    positive_node = base_workflow.get(args.positive_node)
    if not isinstance(positive_node, dict):
        raise SystemExit(
            f"Positive node {args.positive_node!r} does not exist."
        )

    client = ComfyUIClient(args.server)

    print(
        f"ComfyUI: {client.server}",
        file=sys.stderr,
    )
    print(
        f"positive node: {args.positive_node}",
        file=sys.stderr,
    )
    print(
        f"images per prompt: {args.count}",
        file=sys.stderr,
    )
    print(
        f"SaveImage nodes: {', '.join(sorted(save_nodes))}",
        file=sys.stderr,
    )
    print(
        "Reading prompts from stdin...",
        file=sys.stderr,
    )

    prompt_index = 0

    try:
        for raw_line in sys.stdin:
            # Remove only the line ending.
            prompt = raw_line.rstrip("\r\n")

            # Ignore completely empty lines.
            if not prompt.strip():
                continue

            prompt_index += 1

            workflow = copy.deepcopy(base_workflow)

            set_positive_prompt(
                workflow,
                args.positive_node,
                prompt,
            )

            changed_batch_nodes = set_batch_size(
                workflow,
                args.count,
            )

            seed = secrets.randbits(63)

            changed_seed_nodes = set_seed(
                workflow,
                seed,
            )

            print(
                f"[{prompt_index}] "
                f"seed={seed} "
                f"count={args.count}",
                file=sys.stderr,
            )
            print(
                f"[{prompt_index}] prompt={prompt}",
                file=sys.stderr,
            )

            if changed_batch_nodes:
                print(
                    f"[{prompt_index}] "
                    f"batch_size -> {', '.join(changed_batch_nodes)}",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    "No node with numeric 'batch_size' input was found. "
                    "The workflow cannot honor --count."
                )

            print(
                f"[{prompt_index}] "
                f"seed updated in: {', '.join(changed_seed_nodes)}",
                file=sys.stderr,
            )

            prompt_id = client.queue_prompt(workflow)

            print(
                f"[{prompt_index}] queued {prompt_id}",
                file=sys.stderr,
            )

            history_entry = client.wait_for_prompt(
                prompt_id,
                poll_interval=args.poll,
            )

            saved_images = collect_saved_images(
                history_entry,
                save_nodes,
            )

            if not saved_images:
                raise RuntimeError(
                    f"Prompt {prompt_id} completed successfully, "
                    "but no images were returned by SaveImage."
                )

            print(
                f"[{prompt_index}] "
                f"received {len(saved_images)} image(s)",
                file=sys.stderr,
            )

            for image_no, (
                save_node_id,
                filename,
                subfolder,
                folder_type,
            ) in enumerate(saved_images, start=1):
                image_data = client.get_image(
                    filename,
                    subfolder,
                    folder_type,
                )

                source_name = Path(filename).name
                output_path = args.output / source_name

                output_path.write_bytes(image_data)

                txt_path = output_path.with_suffix(".txt")

                write_metadata(
                    txt_path,
                    prompt,
                    seed,
                    prompt_id,
                )

                print(
                    f"[{prompt_index}] "
                    f"[{image_no}/{len(saved_images)}] "
                    f"SaveImage {save_node_id}: "
                    f"{output_path}",
                    file=sys.stderr,
                )

    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        return 130

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

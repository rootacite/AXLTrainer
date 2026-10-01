"""The bookkeeping behind a checkpoint's "Evaluate" action: what images it has, what has to be
rendered to reach a requested depth, and what the tagger's labels say about them.

Torch-free and GPU-free on purpose (like `runs.py` and `genjob.py`), so the whole calculation is
unit-testable and api.py can prepare an evaluation before spawning anything:

* `collect_images` — the checkpoint's sample images, from the run's own `{name}_samples/` and from
  the generations already recorded for it, each with the prompt it was rendered from.
* `expansion_plan` — what to render so the count reaches `depth`, as a whole number of the config's
  sample passes, listing only the `(set, repeat)` slots that are missing.
* `score_images` — two scoreboards over tagged images: per image (micro) and per prompt (union),
  optionally narrowed to the tags the caller picked out of `prompt_tag_counts`.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

try:
    from genjob import STATE_RUNNING, read_job
except ImportError:
    from trainer.genjob import STATE_RUNNING, read_job

# `{name}_{step:06d}_p{set}_{repeat}.png`, and the older two-number form, which api.scan_samples
# also reads as set 0.
RUN_SAMPLE_NAME = re.compile(r"_(\d+)_p(\d+)_(\d+)\.png$")
LEGACY_SAMPLE_NAME = re.compile(r"_(\d+)_(\d+)\.png$")
# `{job_id}_p{set}_{repeat}.png`: what a `sets` pass (and an evaluation's own top-up) writes.
GENERATED_SET_NAME = re.compile(r"_p(\d+)_(\d+)\.png$")

# Where an image came from: the trainer's own sample point, or a job under `generated/`.
SOURCE_RUN = "run"
SOURCE_GENERATED = "generated"

MIN_DEPTH = 1
MAX_DEPTH = 512
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0
# One evaluation renders at most this many images. A depth costs a whole pass at a time, so the
# overshoot is bounded by the config's own size rather than by the depth alone.
MAX_RENDER = 512
# How many offenders the summary lists; the counts behind them are complete.
TOP_TAGS = 20


def normalize_depth(value: Any) -> int:
    """The requested Depth as an int in range. The message is what the Dashboard shows as-is."""
    if isinstance(value, bool):
        raise ValueError("depth must be a whole number")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("depth must be a whole number")
    try:
        depth = int(value)
    except (TypeError, ValueError):
        raise ValueError("depth must be a whole number") from None
    if not (MIN_DEPTH <= depth <= MAX_DEPTH):
        raise ValueError(f"depth must be between {MIN_DEPTH} and {MAX_DEPTH}")
    return depth


def normalize_threshold(value: Any) -> float:
    """The tagger floor as a float in range. Checked here so a bad one costs no GPU time."""
    if isinstance(value, bool):
        raise ValueError("threshold must be a number")
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError("threshold must be a number") from None
    if not (MIN_THRESHOLD <= threshold <= MAX_THRESHOLD):
        raise ValueError(f"threshold must be between {MIN_THRESHOLD:g} and {MAX_THRESHOLD:g}")
    return threshold


# `(anal:1.2)` after its parentheses are stripped: a weight, not part of the tag. The head before
# the colon may be empty (a stray `:1.2`), which normalizes to nothing and is dropped.
_WEIGHT_SUFFIX = re.compile(r"^(.*?)\s*:\s*[-+]?(?:\d+\.?\d*|\.\d+)$")
_WRAPPERS = {"(": ")", "[": "]", "{": "}"}


def normalize_tag(raw: Any) -> str:
    """One tag in the form the tagger's own output takes, so the two sides compare.

    `(anal:1.2)` -> `anal`, `[long hair]` -> `long hair`, `Masking_Tape_(Medium)` ->
    `masking tape (medium)`: the same underscore-to-space, lowercase form `tagger2.tag_text`
    produces. Wrapper brackets are stripped before underscores become spaces, so the
    `_(medium)` suffix a Danbooru tag carries survives as its own tag's tail.
    """
    text = str(raw or "").strip()
    while len(text) >= 2 and _WRAPPERS.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    match = _WEIGHT_SUFFIX.match(text)
    if match:
        text = match.group(1).strip()
    text = text.replace("_", " ").strip().lower()
    return " ".join(text.split())


def prompt_tags(prompt: Any) -> list[str]:
    """The tags a prompt asks for, in first-seen order and de-duplicated."""
    tags: list[str] = []
    seen: set[str] = set()
    for part in str(prompt or "").split(","):
        tag = normalize_tag(part)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def selected_tags(tags: Any) -> set[str]:
    """The tags an evaluation was asked to score, normalized; empty means "every tag asked for".

    A string is accepted as a comma-separated list, the way the CLI and a hand-written request
    write it. Anything that normalizes to nothing is dropped.
    """
    if tags is None:
        return set()
    items = str(tags).split(",") if isinstance(tags, str) else list(tags)
    chosen: set[str] = set()
    for item in items:
        tag = normalize_tag(item)
        if tag:
            chosen.add(tag)
    return chosen


def prompt_tag_counts(sets: Sequence[Any]) -> list[dict[str, Any]]:
    """How often each tag appears across the config's prompt sets, most frequent first.

    The picker the Dashboard shows before an evaluation is started: one row per tag that appears in
    at least one set's prompt, with `count` = the number of sets asking for it and `frequency` =
    `count / sets × 100` (`0.0` when there are no sets). Ties break on the tag name, so the list is
    stable between polls.
    """
    total = len(sets)
    counts: Counter[str] = Counter()
    for entry in sets:
        for tag in set(prompt_tags(sample_set_prompt(entry))):
            counts[tag] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "tag": tag,
            "count": int(count),
            "frequency": round(count / total * 100, 4) if total else 0.0,
        }
        for tag, count in ordered
    ]


def sample_name_parts(name: str) -> Optional[tuple[int, int, int]]:
    """`(step, set_index, repeat_idx)` of one of the run's own samples, or None.

    Mirrors `api.scan_samples`: `{name}_{step:06d}_p{set}_{repeat}.png`, and the older
    `_{step}_{repeat}.png` as set 0.
    """
    match = RUN_SAMPLE_NAME.search(str(name))
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    legacy = LEGACY_SAMPLE_NAME.search(str(name))
    if legacy:
        return int(legacy.group(1)), 0, int(legacy.group(2))
    return None


def generated_set_parts(name: str) -> Optional[tuple[int, int]]:
    """`(set_index, repeat_idx)` of a generated sample, or None for a file without that shape."""
    match = GENERATED_SET_NAME.search(str(name))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def sample_set_repeat(entry: Any) -> int:
    """A `SampleSet`'s (or a recorded `sample_sets` entry's) repeat, 0 when it carries none."""
    raw = entry.get("repeat") if isinstance(entry, Mapping) else getattr(entry, "repeat", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def sample_set_prompt(entry: Any) -> str:
    """A `SampleSet`'s (or a recorded entry's) prompt, `""` when it carries none."""
    if isinstance(entry, Mapping):
        return str(entry.get("prompt") or "")
    return str(getattr(entry, "prompt", "") or "")


@dataclass
class ImageRef:
    """One sample image of a checkpoint, the prompt it was rendered from, and what the tagger saw.

    `set_index` / `repeat_idx` are -1 for an image that came from no `[[validation.samples]]` set (a
    `single` ad-hoc generation): it still counts towards the depth and is still scored, against the
    prompt recorded on its own job.
    """

    path: str
    name: str = ""
    set_index: int = -1
    repeat_idx: int = -1
    source: str = SOURCE_RUN
    prompt: str = ""
    tags: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = Path(self.path).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "path": str(self.path),
            "set_index": int(self.set_index),
            "repeat_idx": int(self.repeat_idx),
            "source": str(self.source),
            "prompt": str(self.prompt),
            "tags": [str(tag) for tag in self.tags],
            "error": None if self.error is None else str(self.error),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImageRef":
        """Read one image back from a job record; a hand-written or partial one reads leniently."""
        path = str(payload.get("path") or "")
        tags = payload.get("tags")
        error = payload.get("error")
        return cls(
            path=path,
            name=str(payload.get("name") or Path(path).name),
            set_index=_int_or(payload.get("set_index"), -1),
            repeat_idx=_int_or(payload.get("repeat_idx"), -1),
            source=str(payload.get("source") or SOURCE_RUN),
            prompt=str(payload.get("prompt") or ""),
            tags=[str(tag) for tag in tags] if isinstance(tags, list) else [],
            error=None if error is None else str(error),
        )


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def images_from_payload(payload: Any) -> list[ImageRef]:
    """Every image a record's `images` list holds; anything not a table is skipped."""
    if not isinstance(payload, list):
        return []
    return [ImageRef.from_dict(item) for item in payload if isinstance(item, Mapping)]


def _set_prompt(sets: Optional[Sequence[Any]], index: int) -> str:
    if sets is not None and 0 <= index < len(sets):
        return sample_set_prompt(sets[index])
    return ""


def _job_image_paths(job: Mapping[str, Any]) -> list[str]:
    """A job's images: its `files` list, plus the `image_path` a `single` job records instead."""
    paths = [str(item) for item in (job.get("files") or []) if str(item)]
    single = str(job.get("image_path") or "")
    if single and single not in paths:
        paths.append(single)
    return paths


def collect_images(
    *,
    samples_dir: Union[str, Path, None],
    generated_dir: Union[str, Path, None],
    checkpoint: Union[str, Path],
    step: Optional[int],
    sets: Sequence[Any],
) -> list[ImageRef]:
    """Every sample image this checkpoint has, with the prompt each was rendered from.

    Two sources: the run's own sample points at the checkpoint's step (each image's set index names
    the config entry that drew it), and the generations recorded for this checkpoint under
    `generated/` — those carry their own `sample_sets`, which is what keeps an image drawn under a
    different config scored against the prompt it actually used. A running job is skipped (its
    files are still being written, and it holds the GPU anyway); a cancelled or failed one keeps the
    images it wrote, exactly as its card shows them.
    """
    images: list[ImageRef] = []
    reference = os.path.normpath(str(checkpoint))

    if step is not None and samples_dir is not None:
        directory = Path(samples_dir)
        if directory.is_dir():
            for path in sorted(directory.glob("*.png")):
                if path.name.lower().endswith(".mask.png"):
                    continue
                parts = sample_name_parts(path.name)
                if parts is None or parts[0] != int(step):
                    continue
                images.append(
                    ImageRef(
                        path=str(path),
                        name=path.name,
                        set_index=parts[1],
                        repeat_idx=parts[2],
                        source=SOURCE_RUN,
                        prompt=_set_prompt(sets, parts[1]),
                    )
                )

    if generated_dir is not None:
        directory = Path(generated_dir)
        if directory.is_dir():
            for spec in sorted(directory.glob("*.json")):
                job = read_job(spec)
                if job is None or job.get("state") == STATE_RUNNING:
                    continue
                if not job.get("checkpoint"):
                    continue
                if os.path.normpath(str(job.get("checkpoint"))) != reference:
                    continue
                recorded = job.get("sample_sets")
                job_sets: Optional[Sequence[Any]] = recorded if isinstance(recorded, list) and recorded else None
                for raw in _job_image_paths(job):
                    name = Path(raw).name
                    parts = generated_set_parts(name)
                    if parts is None:
                        # No `(set, repeat)` in the name: a `single` generation, scored against the
                        # prompt its own job recorded.
                        images.append(
                            ImageRef(
                                path=raw,
                                name=name,
                                set_index=-1,
                                repeat_idx=-1,
                                source=SOURCE_GENERATED,
                                prompt=str(job.get("prompt") or ""),
                            )
                        )
                        continue
                    images.append(
                        ImageRef(
                            path=raw,
                            name=name,
                            set_index=parts[0],
                            repeat_idx=parts[1],
                            source=SOURCE_GENERATED,
                            prompt=_set_prompt(job_sets if job_sets is not None else sets, parts[0]),
                        )
                    )

    # The run's own samples first, then the generations, each by slot: images that came from no set
    # (a `single` generation) come last, where their unknown slot cannot be mistaken for set -1.
    images.sort(
        key=lambda image: (
            image.source != SOURCE_RUN,
            image.set_index < 0,
            image.set_index,
            image.repeat_idx,
            image.name,
        )
    )
    return images


def expansion_plan(sets: Sequence[Any], depth: int, images: Sequence[ImageRef]) -> dict[str, Any]:
    """What to render so this checkpoint's sample count reaches `depth`.

    `depth` is a floor, and the shortfall is covered in the config's own unit: a whole pass of its
    sample sets, repeated `ceil((depth - M) / N)` times — `M` being the images the checkpoint
    already has, whatever produced them (the trainer's sample point, an earlier `sets` pass, an
    earlier evaluation), and `N = Σ repeat` the images one pass renders. `M >= depth` renders
    nothing; the answer never renders a partial pass, so each set keeps the config's own share and
    the total overshoots by less than one pass.

    Every rendered slot continues the set's own repeat numbering after the highest index already in
    use, so a new pass of an already sampled set never writes over an image that exists (they are
    duplicates of the same prompt, which is the point: more images per prompt).
    """
    depth = normalize_depth(depth)
    repeats = [sample_set_repeat(entry) for entry in sets]
    per_pass = sum(repeats)
    if per_pass <= 0:
        raise ValueError("the config's sample sets render no images")

    existing: dict[int, list[int]] = defaultdict(list)
    for image in images:
        existing[int(image.set_index)].append(int(image.repeat_idx))

    shortfall = depth - len(images)
    passes = -(-shortfall // per_pass) if shortfall > 0 else 0
    if passes * per_pass > MAX_RENDER:
        raise ValueError(
            f"depth {depth} needs {passes} more pass(es) of {per_pass} images to top "
            f"{len(images)} existing image(s) up, {passes * per_pass} images in all, more than the "
            f"{MAX_RENDER} one evaluation renders; pick a depth within {MAX_RENDER} of the images "
            f"this checkpoint already has"
        )

    planned: list[dict[str, Any]] = []
    render_total = 0
    for index, repeat in enumerate(repeats):
        used = sorted(slot for slot in existing.get(index, []) if slot >= 0)
        start = used[-1] + 1 if used else 0
        slots = list(range(start, start + passes * repeat))
        render_total += len(slots)
        planned.append(
            {
                "set_index": index,
                "repeat": repeat,
                "existing": used,
                "render": slots,
            }
        )

    return {
        "needed": passes > 0,
        "depth": depth,
        "passes": passes,
        "per_pass": per_pass,
        "existing_images": len(images),
        "render_total": render_total,
        "sets": planned,
    }


def render_slots(plan: Mapping[str, Any]) -> list[tuple[int, int]]:
    """The `(set_index, repeat_idx)` pairs a plan lists, in render order."""
    slots: list[tuple[int, int]] = []
    for entry in plan.get("sets") or []:
        if not isinstance(entry, Mapping):
            continue
        set_index = _int_or(entry.get("set_index"), -1)
        for slot in entry.get("render") or []:
            slots.append((set_index, _int_or(slot, -1)))
    return slots


def rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision / recall / F1 over the counts; an empty denominator scores 0, never NaN."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _top_tags(counter: Counter[str]) -> list[dict[str, Any]]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [{"tag": tag, "count": int(count)} for tag, count in ordered[:TOP_TAGS]]


def score_images(images: Sequence[ImageRef], tags: Any = None) -> dict[str, Any]:
    """Both scoreboards over already-tagged images.

    The prompt an image was rendered from is the ground truth of what was asked for, and the
    tagger's labels are the positive predictions:

    * **per image (micro)** — a requested tag found on the image is a true positive; one the tagger
      did not report is a false negative; a tag the tagger reported that was not requested is a
      false positive. Counts are summed over every image, so a request that shows up in 1 of 20
      images is 1 TP and 19 FN.
    * **per prompt (union)** — the images of one prompt are pooled: a requested tag counts as found
      when any of them shows it, and an unrequested tag that appeared in any of them is one FP.

    `tags` narrows the whole comparison to the tags the caller cares about: a prompt tag outside that
    set is not a miss and a label outside it is not an extra, so the scores answer "how well are
    *these* tags drawn" instead of "how well does the caption match the prompt". Empty or None scores
    every tag the prompt asks for, which is the behaviour without a selection.

    An image whose tagging failed, or whose prompt asks for no tag at all (none of the selected ones,
    when a selection is given), is counted (`images_failed` / `images_skipped`) but contributes
    nothing.
    """
    chosen = selected_tags(tags) if tags else set()
    restricted = bool(chosen)
    tp = fp = fn = 0
    union_tp = union_fp = union_fn = 0
    false_positives: Counter[str] = Counter()
    false_negatives: Counter[str] = Counter()
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    scored = failed = skipped = 0

    for image in images:
        expected = set(prompt_tags(image.prompt))
        if restricted:
            expected &= chosen
        if image.error:
            failed += 1
            continue
        if not expected:
            skipped += 1
            continue
        observed = {normalize_tag(tag) for tag in image.tags}
        observed.discard("")
        if restricted:
            observed &= chosen
        scored += 1

        hits = expected & observed
        misses = expected - observed
        extras = observed - expected
        tp += len(hits)
        fn += len(misses)
        fp += len(extras)
        false_positives.update(extras)
        false_negatives.update(misses)

        key = tuple(sorted(expected))
        group = groups.get(key)
        if group is None:
            group = {
                "prompt": str(image.prompt),
                "images": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "expected": expected,
                "observed": set(),
            }
            groups[key] = group
        group["images"] += 1
        group["tp"] += len(hits)
        group["fn"] += len(misses)
        group["fp"] += len(extras)
        group["observed"] |= observed

    group_rows: list[dict[str, Any]] = []
    for group in groups.values():
        expected = group.pop("expected")
        observed_union = group.pop("observed")
        group_tp = len(expected & observed_union)
        group_fn = len(expected - observed_union)
        group_fp = len(observed_union - expected)
        union_tp += group_tp
        union_fn += group_fn
        union_fp += group_fp
        per_image = rates(group["tp"], group["fp"], group["fn"])
        per_prompt = rates(group_tp, group_fp, group_fn)
        group_rows.append(
            {
                "prompt": group["prompt"],
                "images": int(group["images"]),
                "tp": int(group["tp"]),
                "fp": int(group["fp"]),
                "fn": int(group["fn"]),
                **per_image,
                "union_tp": group_tp,
                "union_fp": group_fp,
                "union_fn": group_fn,
                "union_precision": per_prompt["precision"],
                "union_recall": per_prompt["recall"],
                "union_f1": per_prompt["f1"],
            }
        )
    group_rows.sort(key=lambda row: (-row["images"], row["prompt"]))

    micro = rates(tp, fp, fn)
    macro = rates(union_tp, union_fp, union_fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": micro["precision"],
        "recall": micro["recall"],
        "f1": micro["f1"],
        "union_tp": int(union_tp),
        "union_fp": int(union_fp),
        "union_fn": int(union_fn),
        "union_precision": macro["precision"],
        "union_recall": macro["recall"],
        "union_f1": macro["f1"],
        "images_scored": int(scored),
        "images_failed": int(failed),
        "images_skipped": int(skipped),
        "tags": sorted(chosen),
        "groups": group_rows,
        "top_false_positives": _top_tags(false_positives),
        "top_false_negatives": _top_tags(false_negatives),
    }

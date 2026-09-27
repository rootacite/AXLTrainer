#!/usr/bin/env python3
"""Illustrious XL prompt sampler over input_matrix.txt.

CLI picks the wizard language; character, mode, exposure and preferences are
collected on a curses checklist. Quality and rating tags are never appended.
"""

from __future__ import annotations

import argparse
import curses
import locale
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO_ROOT / "input_matrix.txt"

PREFER_WEIGHT = 5
COUNT_MIN = 1
COUNT_MAX = 200
COUNT_DEFAULT = 10
VAGINAL_RATIO_DEFAULT = 0.5

MODES = ("sfw", "nsfw", "sex")
EXPOSURE_LEVELS = ("covered", "casual", "revealing", "open", "nude")
HIGH_EXPOSURE = frozenset({"revealing", "open", "nude"})
CLOTHING_GROUPS = ("covered", "casual", "revealing")
CHANNELS = ("both", "anal", "vaginal")

MODE_EXPOSURE_DEFAULTS = {
    "sfw": ("covered", "casual"),
    "nsfw": ("revealing", "open", "nude"),
    "sex": ("open",),
}

SECTION_NAMES = {"POSES", "CLOTHING", "SCENE", "SUFFIX", "SFW_POSES"}
GROUP_RE = re.compile(r"^\[(covered|casual|revealing)\]$")
OPEN_MARK = "(open clothes)"
CHANNEL_RE = re.compile(
    r"^(?P<tags>.+?)\s*:\s*(?P<channel>both|anal only|vaginal only)\s*$"
)

RATING_TAGS = {
    "sfw",
    "nsfw",
    "explicit",
    "sensitive",
    "questionable",
    "rating:general",
    "rating:sensitive",
    "rating:questionable",
    "rating:explicit",
}
QUALITY_TAGS = {
    "masterpiece",
    "best quality",
    "amazing quality",
    "newest",
    "absurdres",
    "highres",
    "ultra detailed",
    "extremely detailed",
    "8k",
}
YEAR_RE = re.compile(r"^year 20\d\d$")
SCORE_RE = re.compile(r"^score_\d")

POSE_FAMILY_IDS = (
    "face",
    "behind",
    "stand_behind",
    "side_behind",
    "girl_on_top",
    "hold",
)
CHEST_LEVELS = ("covered", "cleavage", "breasts_out", "nipples", "auto")
BELLY_LEVELS = ("covered", "midriff", "navel", "auto")
CHEST_DEFAULT = {
    "sfw": "covered",
    "nsfw": "auto",
    "sex": "auto",
}
BELLY_DEFAULT = {
    "sfw": "covered",
    "nsfw": "auto",
    "sex": "auto",
}
ASS_MARKERS = (
    "from behind",
    "doggystyle",
    "prone bone",
    "reverse cowgirl",
    "reverse suspended",
    "top-down bottom-up",
    "full nelson",
    "spooning",
)
SPREAD_MARKERS = (
    "spread legs",
    "legs up",
    "folded",
    "knees to chest",
    "legs over head",
    "leg lift",
    "squatting",
)
TOP_MARKERS = ("girl on top", "cowgirl", "sitting on lap")
LIE_MARKERS = (
    "lying",
    "on back",
    "on stomach",
    "on side",
    "sleeping",
    "missionary",
    "mating press",
    "anvil position",
    "spooning",
    "prone bone",
    "seventh posture",
)

NUDE_SENTINEL = "__nude__"

_ALL_MODES = frozenset(MODES)
_NSFW_MODES = frozenset({"nsfw", "sex"})
_SEX_MODES = frozenset({"sex"})


@dataclass(frozen=True)
class FaceOption:
    tag: str
    modes: frozenset[str]
    zh: str
    en: str


EXPRESSIONS: tuple[FaceOption, ...] = (
    FaceOption("smile", _ALL_MODES, "微笑 — smile", "smile"),
    FaceOption("grin", _ALL_MODES, "咧嘴笑 — grin", "grin"),
    FaceOption("smirk", _ALL_MODES, "坏笑 — smirk", "smirk"),
    FaceOption("blush", _ALL_MODES, "脸红 — blush", "blush"),
    FaceOption("shy", _ALL_MODES, "害羞 — shy", "shy"),
    FaceOption("embarrassed", _ALL_MODES, "尴尬 — embarrassed", "embarrassed"),
    FaceOption("open mouth", _ALL_MODES, "张嘴 — open mouth", "open mouth"),
    FaceOption("closed mouth", _ALL_MODES, "闭嘴 — closed mouth", "closed mouth"),
    FaceOption("tongue out", _NSFW_MODES, "吐舌 — tongue out", "tongue out"),
    FaceOption("tears", _ALL_MODES, "含泪 — tears", "tears"),
    FaceOption("crying", _ALL_MODES, "哭泣 — crying", "crying"),
    FaceOption("angry", _ALL_MODES, "生气 — angry", "angry"),
    FaceOption("surprised", _ALL_MODES, "惊讶 — surprised", "surprised"),
    FaceOption("scared", _ALL_MODES, "害怕 — scared", "scared"),
    FaceOption("naughty face", _NSFW_MODES, "坏脸 — naughty face", "naughty face"),
    FaceOption("drooling", _NSFW_MODES, "流口水 — drooling", "drooling"),
    FaceOption("heavy breathing", _NSFW_MODES, "喘息 — heavy breathing", "heavy breathing"),
    FaceOption("pain", _NSFW_MODES, "痛苦 — pain", "pain"),
    FaceOption("clenched teeth", _NSFW_MODES, "咬牙 — clenched teeth", "clenched teeth"),
    FaceOption("ahegao", _NSFW_MODES, "阿嘿颜 — ahegao", "ahegao"),
    FaceOption("orgasm", _NSFW_MODES, "高潮 — orgasm", "orgasm"),
)

EYES: tuple[FaceOption, ...] = (
    FaceOption("looking at viewer", _ALL_MODES, "看镜头 — looking at viewer", "looking at viewer"),
    FaceOption("looking away", _ALL_MODES, "视线移开 — looking away", "looking away"),
    FaceOption("closed eyes", _ALL_MODES, "闭眼 — closed eyes", "closed eyes"),
    FaceOption("half-closed eyes", _ALL_MODES, "半闭眼 — half-closed eyes", "half-closed eyes"),
    FaceOption("empty eyes", _ALL_MODES, "空洞眼 — empty eyes", "empty eyes"),
    FaceOption("sparkling eyes", _ALL_MODES, "亮晶晶 — sparkling eyes", "sparkling eyes"),
    FaceOption("wide-eyed", _ALL_MODES, "睁大眼 — wide-eyed", "wide-eyed"),
    FaceOption("one eye closed", _ALL_MODES, "闭一只眼 — one eye closed", "one eye closed"),
    FaceOption("squinting", _ALL_MODES, "眯眼 — squinting", "squinting"),
    FaceOption("rolling eyes", _NSFW_MODES, "翻白眼 — rolling eyes", "rolling eyes"),
    FaceOption("heart-shaped pupils", _NSFW_MODES, "爱心瞳 — heart-shaped pupils", "heart-shaped pupils"),
    FaceOption("spiral eyes", _SEX_MODES, "螺旋眼 — spiral eyes", "spiral eyes"),
)

SEX_FACE_TAGS = frozenset(
    {
        "ahegao",
        "orgasm",
        "pain",
        "clenched teeth",
        "rolling eyes",
        "spiral eyes",
        "naughty face",
        "drooling",
        "heavy breathing",
        "tongue out",
        "heart-shaped pupils",
    }
)


class MatrixError(ValueError):
    pass


class WizardCancelled(Exception):
    pass


class GoBack(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    tags: tuple[str, ...]
    comment: str = ""
    channel: str | None = None
    open_clothes: bool = False
    group: str | None = None

    @property
    def blob(self) -> str:
        return ", ".join(self.tags)

    @property
    def key(self) -> tuple[str, ...]:
        return self.tags


@dataclass
class Matrix:
    poses: list[Entry]
    clothing: list[Entry]
    scenes: list[Entry]
    suffixes: list[Entry]
    sfw_poses: list[Entry]


@dataclass
class PromptSpec:
    character: str
    mode: str
    exposure: tuple[str, ...]
    clothing_any: bool = True
    clothing_keys: frozenset[tuple[str, ...]] = field(default_factory=frozenset)
    scene_any: bool = True
    scene_keys: frozenset[tuple[str, ...]] = field(default_factory=frozenset)
    pose_any: bool = True
    pose_keys: frozenset[tuple[str, ...]] = field(default_factory=frozenset)
    family_any: bool = True
    families: frozenset[str] = field(default_factory=frozenset)
    vaginal_ratio: float = VAGINAL_RATIO_DEFAULT
    chest: str = "auto"
    belly: str = "auto"
    expression_any: bool = True
    expression_none: bool = False
    expression_keys: frozenset[str] = field(default_factory=frozenset)
    eye_any: bool = True
    eye_none: bool = False
    eye_keys: frozenset[str] = field(default_factory=frozenset)
    count: int = COUNT_DEFAULT


def split_tags(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_channel(raw: str) -> str:
    if raw == "both":
        return "both"
    if raw == "anal only":
        return "anal"
    if raw == "vaginal only":
        return "vaginal"
    raise MatrixError(f"unknown channel: {raw}")


def parse_matrix(text: str) -> Matrix:
    poses: list[Entry] = []
    clothing: list[Entry] = []
    scenes: list[Entry] = []
    suffixes: list[Entry] = []
    sfw_poses: list[Entry] = []
    buckets = {
        "POSES": poses,
        "CLOTHING": clothing,
        "SCENE": scenes,
        "SUFFIX": suffixes,
        "SFW_POSES": sfw_poses,
    }

    section: str | None = None
    group: str | None = None
    pending: Entry | None = None

    def commit() -> None:
        nonlocal pending
        if pending is None:
            return
        if section is None:
            raise MatrixError("entry before a section header")
        buckets[section].append(pending)
        pending = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":") and line[:-1] in SECTION_NAMES:
            commit()
            section = line[:-1]
            group = None
            continue
        if section == "CLOTHING" and GROUP_RE.match(line):
            commit()
            group = GROUP_RE.match(line).group(1)
            continue
        if line.startswith("#"):
            if pending is None:
                continue
            pending = Entry(
                tags=pending.tags,
                comment=line[1:].strip(),
                channel=pending.channel,
                open_clothes=pending.open_clothes,
                group=pending.group,
            )
            continue
        commit()
        if section is None:
            raise MatrixError(f"line {lineno}: tags before a section header")
        if section == "POSES":
            match = CHANNEL_RE.match(line)
            if match is None:
                raise MatrixError(
                    f"line {lineno}: pose must end with ': both|anal only|vaginal only'"
                )
            pending = Entry(
                tags=tuple(split_tags(match.group("tags"))),
                channel=parse_channel(match.group("channel")),
            )
            continue
        open_clothes = False
        body = line
        if body.endswith(OPEN_MARK):
            open_clothes = True
            body = body[: -len(OPEN_MARK)].rstrip()
        if section == "CLOTHING":
            if group is None:
                raise MatrixError(f"line {lineno}: clothing row before a [group] header")
            pending = Entry(
                tags=tuple(split_tags(body)),
                open_clothes=open_clothes,
                group=group,
            )
            continue
        pending = Entry(tags=tuple(split_tags(body)), open_clothes=open_clothes)

    commit()
    if not poses or not clothing or not scenes or not suffixes or not sfw_poses:
        raise MatrixError("matrix is missing a required section")
    if any(item.group is None for item in clothing):
        raise MatrixError("clothing entry without an exposure group")
    return Matrix(
        poses=poses,
        clothing=clothing,
        scenes=scenes,
        suffixes=suffixes,
        sfw_poses=sfw_poses,
    )


def load_matrix(path: Path) -> Matrix:
    return parse_matrix(path.read_text(encoding="utf-8"))


def is_forbidden_tag(tag: str) -> bool:
    lowered = tag.lower().strip()
    if lowered in RATING_TAGS or lowered in QUALITY_TAGS:
        return True
    return bool(YEAR_RE.match(lowered) or SCORE_RE.match(lowered))


def contains_marker(blob: str, markers: Iterable[str]) -> bool:
    return any(marker in blob for marker in markers)


def needs_sfw_exposure_warning(mode: str, exposure: Sequence[str]) -> bool:
    return mode == "sfw" and bool(HIGH_EXPOSURE.intersection(exposure))


def default_exposure(mode: str) -> tuple[str, ...]:
    return MODE_EXPOSURE_DEFAULTS[mode]


def pose_pool(matrix: Matrix, mode: str) -> list[Entry]:
    if mode == "sex":
        return list(matrix.poses)
    return list(matrix.sfw_poses)


def parse_vaginal_ratio(raw: str, default: float = VAGINAL_RATIO_DEFAULT) -> float:
    text = str(raw).strip()
    if not text:
        return default
    value = float(text)
    if value > 1.0:
        value /= 100.0
    return min(1.0, max(0.0, value))


def pose_families(entry: Entry) -> frozenset[str]:
    blob = entry.blob
    tags = set(entry.tags)
    found: set[str] = set()
    if (
        "cowgirl" in blob
        or "girl on top" in blob
        or "upright straddle" in blob
        or "sitting on lap" in blob
    ):
        found.add("girl_on_top")
    if (
        "suspended congress" in blob
        or "held up" in tags
        or "full nelson" in blob
        or "sitting on lap" in blob
        or "upright straddle" in blob
    ):
        found.add("hold")
    if "standing doggystyle" in blob:
        found.add("stand_behind")
    elif "doggystyle" in blob or "prone bone" in blob:
        found.add("behind")
    if "reverse suspended" in blob or "full nelson" in blob:
        found.add("stand_behind")
    if "spooning" in blob or "seventh posture" in blob:
        found.add("side_behind")
    elif "on side" in tags and "lying" in tags:
        found.add("side_behind")
    if "reverse cowgirl" in blob:
        found.add("behind")
    if (
        "mating press" in blob
        or "anvil position" in blob
        or "standing missionary" in blob
        or ("missionary" in blob and "doggystyle" not in blob)
    ):
        found.add("face")
    if "suspended congress" in blob and "reverse" not in blob:
        found.add("face")
    return frozenset(found)


def chest_view(entry: Entry) -> str:
    """How visible the chest is: front, side, back, or optional (kneeling/prone)."""
    blob = entry.blob
    tags = set(entry.tags)
    if "prone bone" in blob or "on stomach" in tags:
        return "optional"
    if "doggystyle" in blob and "standing doggystyle" not in blob:
        return "optional"
    if (
        "standing doggystyle" in blob
        or "reverse cowgirl" in blob
        or "reverse suspended" in blob
    ):
        return "back"
    if "spooning" in blob or "seventh posture" in blob or "on side" in tags:
        return "side"
    if "full nelson" in blob:
        return "side"
    families = pose_families(entry)
    if "face" in families or "girl_on_top" in families:
        return "front"
    if "hold" in families and "stand_behind" not in families:
        return "front"
    if "looking at viewer" in blob and "from behind" not in blob:
        return "front"
    if "from behind" in blob or "looking back" in blob:
        return "back"
    return "front"


def torso_tags(
    spec: PromptSpec,
    pose: Entry,
    clothing: Entry | None,
    add_open: bool,
    rng: random.Random,
) -> list[str]:
    """Chest/belly tags. Front + open clothes in nsfw/sex always pulls breasts out."""
    nude = clothing is None
    clothes_allow_out = nude or add_open
    view = chest_view(pose)
    chest_pref = spec.chest if spec.chest in CHEST_LEVELS else "auto"
    belly_pref = spec.belly if spec.belly in BELLY_LEVELS else "auto"
    if spec.mode == "sfw":
        if chest_pref == "auto":
            chest_pref = "covered"
        if belly_pref == "auto":
            belly_pref = "covered"

    tags: list[str] = []
    if nude:
        tags.extend(["breasts", "nipples"])
    else:
        force_front_out = (
            spec.mode in ("nsfw", "sex") and add_open and view == "front"
        )
        optional_view = view in ("optional", "side", "back")
        show_out = False
        show_nipples = False
        show_cleavage = False
        if force_front_out:
            show_out = True
            show_nipples = True
        elif chest_pref == "cleavage":
            show_cleavage = not add_open
        elif chest_pref == "breasts_out":
            show_out = add_open
        elif chest_pref == "nipples":
            show_out = add_open
            show_nipples = add_open
        elif chest_pref == "auto" and spec.mode in ("nsfw", "sex"):
            if add_open and optional_view:
                show_out = rng.random() < 0.5
                show_nipples = show_out
            elif not add_open and view == "front" and spec.mode == "nsfw":
                show_cleavage = True
        if show_out:
            tags.extend(["breasts", "breasts out"])
            if show_nipples:
                tags.append("nipples")
        elif show_nipples:
            tags.extend(["breasts", "nipples"])
        elif show_cleavage:
            tags.append("cleavage")

    if belly_pref == "midriff":
        tags.append("midriff")
    elif belly_pref == "navel":
        tags.extend(["midriff", "navel"])
    elif belly_pref == "auto" and spec.mode in ("nsfw", "sex"):
        if (nude or add_open) and view == "front":
            tags.append("navel")

    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def face_label(item: FaceOption, lang: str) -> str:
    return item.zh if lang == "chinese" else item.en


def catalog_for_mode(catalog: Sequence[FaceOption], mode: str) -> list[FaceOption]:
    return [item for item in catalog if mode in item.modes]


def resolve_face_pool(
    catalog: Sequence[FaceOption],
    mode: str,
    any_flag: bool,
    none: bool,
    keys: frozenset[str],
) -> list[str]:
    if none:
        return []
    allowed = [item.tag for item in catalog_for_mode(catalog, mode)]
    if any_flag or not keys:
        return allowed
    ordered = [tag for tag in allowed if tag in keys]
    for tag in keys:
        if tag not in ordered:
            ordered.append(tag)
    return ordered


def _blocked_eyes(pose: Entry) -> set[str]:
    tags = set(pose.tags)
    blocked: set[str] = set()
    if "looking at viewer" in tags:
        blocked.update({"closed eyes", "looking away"})
    if "closed eyes" in tags:
        blocked.update(
            {
                "looking at viewer",
                "sparkling eyes",
                "wide-eyed",
                "heart-shaped pupils",
                "spiral eyes",
                "empty eyes",
                "rolling eyes",
            }
        )
    return blocked


def _pick_face_tag(
    pool: Sequence[str],
    pose: Entry,
    any_flag: bool,
    rng: random.Random,
    blocked: set[str],
    skip_if_any_present: bool,
) -> str | None:
    if not pool:
        return None
    present = set(pose.tags)
    if any_flag:
        if skip_if_any_present and any(tag in present for tag in pool):
            return None
        candidates = [tag for tag in pool if tag not in present and tag not in blocked]
        if not candidates:
            return None
        return rng.choice(candidates)
    return rng.choice(list(pool))


def face_tags(spec: PromptSpec, pose: Entry, rng: random.Random) -> list[str]:
    """One expression and one eye tag, or none if the user skipped that axis."""
    tags: list[str] = []
    expr_pool = resolve_face_pool(
        EXPRESSIONS,
        spec.mode,
        spec.expression_any,
        spec.expression_none,
        spec.expression_keys,
    )
    expr = _pick_face_tag(
        expr_pool, pose, spec.expression_any, rng, set(), skip_if_any_present=True
    )
    if expr:
        tags.append(expr)
    eye_pool = resolve_face_pool(
        EYES,
        spec.mode,
        spec.eye_any,
        spec.eye_none,
        spec.eye_keys,
    )
    blocked = _blocked_eyes(pose) if spec.eye_any else set()
    eye = _pick_face_tag(
        eye_pool, pose, spec.eye_any, rng, blocked, skip_if_any_present=False
    )
    if eye:
        tags.append(eye)
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def needs_sfw_face_warning(mode: str, keys: Iterable[str]) -> bool:
    return mode == "sfw" and bool(set(keys) & SEX_FACE_TAGS)


def _parse_face_choice(
    flags: Sequence[bool], tags: Sequence[str]
) -> tuple[bool, bool, frozenset[str]]:
    """Map checklist flags (index 0 = none) to (any, none, keys)."""
    if not any(flags):
        return True, False, frozenset()
    none_on = bool(flags[0])
    chosen = frozenset(tag for tag, on in zip(tags, flags[1:]) if on)
    if none_on and not chosen:
        return False, True, frozenset()
    return False, False, chosen


def _face_summary(any_flag: bool, none: bool, keys: frozenset[str]) -> str:
    if none:
        return "none"
    if any_flag or not keys:
        return "any"
    return ", ".join(sorted(keys))


def family_pool(poses: Sequence[Entry], spec: PromptSpec) -> list[Entry]:
    if spec.family_any or not spec.families:
        return list(poses)
    chosen = [pose for pose in poses if pose_families(pose) & spec.families]
    return chosen or list(poses)


def scene_flags(tags: Sequence[str]) -> set[str]:
    present = set(tags)
    flags: set[str] = set()
    if "on bed" in present:
        flags.add("on_bed")
    if "bed" in present or "on bed" in present:
        flags.add("bed")
    if "desk" in present:
        flags.add("desk")
    if "window" in present:
        flags.add("window")
    if "indoors" in present:
        flags.add("indoor")
    if "outdoors" in present:
        flags.add("outdoor")
    if "grass" in present or "sand" in present:
        flags.add("ground")
    if "sofa" in present or "bench" in present:
        flags.add("seat")
    if "ocean" in present or "pool" in present or "onsen" in present:
        flags.add("water")
    if "train interior" in present:
        flags.add("cramped")
    if any(
        name in present
        for name in ("hallway", "classroom", "bedroom", "cafe", "living room", "street")
    ):
        flags.add("wall")
    return flags


def pose_locus(entry: Entry) -> str:
    tags = set(entry.tags)
    blob = entry.blob
    if "sitting on stairs" in blob:
        return "sit_stairs"
    if "sitting" in tags and "desk" in tags:
        return "sit_desk"
    if "looking outside" in tags and "window" in tags:
        return "window"
    if "walking" in tags:
        return "walk"
    if "holding bouquet" in tags or "twirl" in tags:
        return "twirl"
    if (
        "held up" in tags
        or "suspended congress" in blob
        or "reverse suspended congress" in blob
        or "full nelson" in blob
    ):
        return "lift"
    if (
        "standing doggystyle" in blob
        or "standing missionary" in blob
        or "standing sex" in blob
    ):
        return "sex_stand"
    if "doggystyle" in blob:
        return "sex_kneel"
    if (
        "cowgirl" in blob
        or "girl on top" in blob
        or "sitting on lap" in blob
        or "upright straddle" in blob
    ):
        return "cowgirl"
    if "against wall" in tags:
        return "wall"
    if contains_marker(blob, LIE_MARKERS):
        return "lie"
    if "standing" in tags:
        return "stand"
    if "sitting" in tags:
        return "sit"
    return "other"


def scene_compatible(pose: Entry, scene: Entry) -> bool:
    locus = pose_locus(pose)
    tags = set(scene.tags)
    flags = scene_flags(scene.tags)

    def has(*names: str) -> bool:
        return any(name in tags for name in names)

    if locus == "lie":
        if has("hallway", "rooftop", "train interior", "street", "cafe"):
            return False
        return bool({"on_bed", "bed", "ground", "seat"} & flags)
    if locus == "sit_desk":
        if "on bed" in tags or has(
            "rooftop", "onsen", "hallway", "street", "grass", "sand", "forest", "pool"
        ):
            return False
        return "desk" in tags or ("window" in tags and "indoors" in tags)
    if locus == "sit_stairs":
        if has(
            "on bed",
            "beach",
            "train interior",
            "cafe",
            "poolside",
            "forest",
            "classroom",
        ):
            return False
        return has("hallway", "shrine", "street", "school courtyard")
    if locus == "window":
        return "window" in tags and "outdoors" not in tags
    if locus == "wall":
        if has("grass", "beach", "forest", "poolside", "rooftop", "onsen"):
            return False
        return "indoors" in tags or "street" in tags
    if locus == "lift":
        if "on bed" in tags or has("train interior", "cafe", "desk"):
            return False
        return True
    if locus == "walk":
        if "on bed" in tags or has("train interior", "cafe", "onsen", "desk"):
            return False
        return has(
            "hallway",
            "street",
            "park",
            "school courtyard",
            "beach",
            "shrine",
            "rooftop",
        )
    if locus == "twirl":
        if "on bed" in tags or has("train interior", "desk"):
            return False
        return has(
            "shrine",
            "school courtyard",
            "park",
            "rooftop",
            "street",
            "beach",
        )
    if locus == "stand":
        return "on bed" not in tags
    if locus == "cowgirl":
        if has("hallway", "street", "rooftop"):
            return False
        return bool({"bed", "on_bed", "seat", "ground"} & flags)
    if locus == "sex_kneel":
        if has("rooftop", "street", "train interior", "cafe"):
            return False
        return bool({"bed", "on_bed", "ground"} & flags) or has(
            "bedroom", "living room", "classroom"
        )
    if locus == "sex_stand":
        return "on bed" not in tags and "train interior" not in tags
    return True


def anatomy_tags(pose: Entry, channel: str) -> list[str]:
    blob = pose.blob
    out = ["penis"]
    if channel == "vaginal":
        out.append("pussy")
    elif channel == "anal":
        out.append("anus")
    if contains_marker(blob, ASS_MARKERS) and "ass" not in out:
        out.append("ass")
    if contains_marker(blob, SPREAD_MARKERS):
        if "pussy" not in out:
            out.append("pussy")
        if channel == "anal" and "ass" not in out:
            out.append("ass")
    if contains_marker(blob, TOP_MARKERS):
        if channel == "vaginal" and "pussy" not in out:
            out.append("pussy")
        if channel == "anal" and "ass" not in out:
            out.append("ass")
    return out


def pick_channel(pose: Entry, spec: PromptSpec, rng: random.Random) -> str:
    if pose.channel == "anal":
        return "anal"
    if pose.channel == "vaginal":
        return "vaginal"
    if rng.random() < spec.vaginal_ratio:
        return "vaginal"
    return "anal"


def _weight_for(entry: Entry, any_selected: bool, selected: frozenset[tuple[str, ...]]) -> int:
    if any_selected or not selected:
        return 1
    return PREFER_WEIGHT if entry.key in selected else 1


def preferred_pool(
    items: Sequence[Entry], any_selected: bool, selected: frozenset[tuple[str, ...]]
) -> list[Entry]:
    if any_selected or not selected:
        return list(items)
    chosen = [item for item in items if item.key in selected]
    return chosen or list(items)


def pose_weight(entry: Entry, spec: PromptSpec) -> int:
    weight = _weight_for(entry, spec.pose_any, spec.pose_keys)
    if spec.mode != "sex":
        return weight
    matched = pose_families(entry)
    if not spec.family_any and spec.families:
        hit = matched & spec.families
        if not hit:
            return 0
        weight *= max(1, len(hit))
    vaginal = min(1000, max(0, round(spec.vaginal_ratio * 1000)))
    anal = 1000 - vaginal
    if entry.channel == "vaginal":
        weight *= vaginal
    elif entry.channel == "anal":
        weight *= anal
    else:
        weight *= 1000
    return weight


def weighted_choice(rng: random.Random, items: Sequence[Entry], weights: Sequence[int]) -> Entry:
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(items))
    pick = rng.randrange(total)
    acc = 0
    for item, weight in zip(items, weights):
        acc += weight
        if pick < acc:
            return item
    return items[-1]


def clothing_buckets(spec: PromptSpec) -> list[str]:
    buckets = [level for level in EXPOSURE_LEVELS if level in spec.exposure]
    return buckets or list(default_exposure(spec.mode))


def clothing_pool_for_bucket(matrix: Matrix, bucket: str) -> list[Entry]:
    if bucket == "nude":
        return []
    if bucket == "open":
        return [item for item in matrix.clothing if item.open_clothes]
    return [item for item in matrix.clothing if item.group == bucket]


def pick_clothing(
    matrix: Matrix, spec: PromptSpec, rng: random.Random
) -> tuple[Entry | None, bool]:
    buckets = clothing_buckets(spec)
    if not buckets:
        buckets = list(default_exposure(spec.mode))
    bucket = rng.choice(buckets)
    if bucket == "nude":
        return None, False
    pool = preferred_pool(
        clothing_pool_for_bucket(matrix, bucket),
        spec.clothing_any,
        spec.clothing_keys,
    )
    if not pool:
        pool = list(matrix.clothing)
    weights = [_weight_for(item, spec.clothing_any, spec.clothing_keys) for item in pool]
    chosen = weighted_choice(rng, pool, weights)
    add_open = bucket == "open"
    return chosen, add_open


def pick_scene(
    matrix: Matrix, spec: PromptSpec, pose: Entry, rng: random.Random, warnings: list[str]
) -> Entry:
    pool = preferred_pool(matrix.scenes, spec.scene_any, spec.scene_keys)
    compatible = [scene for scene in pool if scene_compatible(pose, scene)]
    if not compatible:
        warnings.append(f"no compatible scene for pose {pose.blob!r}; using full list")
        compatible = list(pool) or list(matrix.scenes)
    weights = [_weight_for(scene, spec.scene_any, spec.scene_keys) for scene in compatible]
    return weighted_choice(rng, compatible, weights)


def assemble(
    spec: PromptSpec,
    pose: Entry,
    clothing: Entry | None,
    add_open: bool,
    scene: Entry,
    suffix: Entry,
    channel: str | None,
    torso: Sequence[str] = (),
    face: Sequence[str] = (),
) -> str:
    prefix = split_tags(spec.character)
    extra: list[str] = []
    if spec.mode == "sex":
        extra.extend(["1boy", "hetero"])
    else:
        extra.append("solo")
    if clothing is None:
        extra.append("nude")
    else:
        extra.extend(clothing.tags)
        if add_open:
            extra.append("open clothes")
    extra.extend(torso)
    extra.extend(face)
    extra.extend(pose.tags)
    if spec.mode == "sex":
        extra.append("sex")
        if channel:
            extra.append(channel)
        extra.extend(anatomy_tags(pose, channel or "vaginal"))
    extra.extend(scene.tags)
    extra.extend(suffix.tags)

    seen = {tag.lower() for tag in prefix}
    out = list(prefix)
    for tag in extra:
        if is_forbidden_tag(tag):
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return ", ".join(out)


def generate(
    spec: PromptSpec,
    matrix: Matrix,
    seed: int | None = None,
    warnings: list[str] | None = None,
) -> list[str]:
    if spec.mode not in MODES:
        raise ValueError(f"unknown mode: {spec.mode}")
    if spec.count < COUNT_MIN or spec.count > COUNT_MAX:
        raise ValueError(f"count must be {COUNT_MIN}..{COUNT_MAX}")
    rng = random.Random(seed)
    notes = warnings if warnings is not None else []
    poses = pose_pool(matrix, spec.mode)
    if spec.mode == "sex":
        poses = family_pool(poses, spec)
    poses = preferred_pool(poses, spec.pose_any, spec.pose_keys)
    if not poses:
        raise ValueError("empty pose pool")
    results: list[str] = []
    used: set[tuple] = set()
    attempts = 0
    limit = max(spec.count * 40, 80)
    while len(results) < spec.count and attempts < limit:
        attempts += 1
        pose_weights = [pose_weight(pose, spec) for pose in poses]
        pose = weighted_choice(rng, poses, pose_weights)
        clothing, add_open = pick_clothing(matrix, spec, rng)
        if spec.mode != "sex" and clothing is None:
            add_open = False
        scene = pick_scene(matrix, spec, pose, rng, notes)
        suffix = weighted_choice(rng, matrix.suffixes, [1] * len(matrix.suffixes))
        channel = pick_channel(pose, spec, rng) if spec.mode == "sex" else None
        torso = torso_tags(spec, pose, clothing, add_open, rng)
        face = face_tags(spec, pose, rng)
        combo = (
            pose.key,
            clothing.key if clothing is not None else (NUDE_SENTINEL,),
            add_open,
            tuple(torso),
            tuple(face),
            scene.key,
            suffix.key,
            channel,
        )
        if combo in used:
            continue
        used.add(combo)
        results.append(
            assemble(spec, pose, clothing, add_open, scene, suffix, channel, torso, face)
        )
    while len(results) < spec.count:
        pose = weighted_choice(rng, poses, [pose_weight(p, spec) for p in poses])
        clothing, add_open = pick_clothing(matrix, spec, rng)
        scene = pick_scene(matrix, spec, pose, rng, notes)
        suffix = rng.choice(matrix.suffixes)
        channel = pick_channel(pose, spec, rng) if spec.mode == "sex" else None
        torso = torso_tags(spec, pose, clothing, add_open, rng)
        face = face_tags(spec, pose, rng)
        results.append(
            assemble(spec, pose, clothing, add_open, scene, suffix, channel, torso, face)
        )
    return results


# --- wizard (language + UI protocol; curses is one backend) ---

STRINGS = {
    "chinese": {
        "character_title": "角色词（置于最前，生成时不会改写）",
        "character_hint": "例如 (sena_character:1.1), 1girl",
        "mode_title": "模式",
        "mode_sfw": "SFW — 日常姿态，不写性交",
        "mode_nsfw": "NSFW — 日常姿态，默认可高暴露",
        "mode_sex": "SEX — 只从性交体位里抽",
        "exposure_title": "衣物暴露程度（可多选）",
        "exp_covered": "遮住 — 校服、正装、婚纱等",
        "exp_casual": "日常 — 连衣裙、睡衣、便装",
        "exp_revealing": "暴露 — 泳装、睡裙",
        "exp_open": "敞开 — 给可敞开的套装加上 open clothes",
        "exp_nude": "裸体 — 不抽服装，写 nude",
        "sfw_warn": "SFW 搭配了高暴露（泳装 / 敞开 / 裸体）。仍可生成，提示词里不会写入 sfw/nsfw。",
        "warn_continue": "继续",
        "warn_back": "返回修改",
        "clothing_title": "服装偏好（可多选；选「不限」则该维均匀抽）",
        "chest_title": "胸部裸露",
        "chest_covered": "遮住 — 不写胸部标签",
        "chest_cleavage": "乳沟 — cleavage",
        "chest_out": "掏出 — breasts out（需敞开或裸体）",
        "chest_nipples": "乳头 — breasts out + nipples",
        "chest_auto": "按姿势 — 正面敞开必写 breasts out/nipples，跪趴可加可不加",
        "belly_title": "腹部裸露",
        "belly_covered": "遮住 — 不写腹部标签",
        "belly_midriff": "露腰 — midriff",
        "belly_navel": "肚脐 — midriff, navel",
        "belly_auto": "按姿势 — 正面敞开或裸体时写 navel",
        "sfw_body_warn": "SFW 勾了胸部掏出/乳头。仍可生成，提示词里不会写入 sfw/nsfw。",
        "expression_title": "表情（可多选；选「不限」则从当前模式的池子抽一条）",
        "eye_title": "眼睛状态（可多选；选「不限」则从当前模式的池子抽一条）",
        "face_none": "不写 — 不追加表情/眼睛标签",
        "sfw_face_warn": "SFW 勾了偏性交的表情或眼睛（阿嘿颜 / 高潮 / 痛苦等）。仍可生成，提示词里不会写入 sfw/nsfw。",
        "scene_title": "场景偏好（可多选）",
        "family_title": "体位大类（可多选）",
        "fam_face": "面对面位 — 传教士、交配压、铁砧、站立对面",
        "fam_behind": "背后位 — 跪趴/平趴后入、反骑",
        "fam_stand_behind": "站立背后位 — 站立弯腰后入、背对抱插、全尼尔森",
        "fam_side_behind": "侧背后位 — 汤匙、第七式",
        "fam_girl_on_top": "女上 — 正骑、蹲骑、反骑、坐抱",
        "fam_hold": "抱位 — 对面/背对抱插、全尼尔森、坐抱",
        "ratio_title": "阴道交占比（0–1；肛交权值 = 1 − 该值）",
        "ratio_hint": "0 = 全肛交，1 = 全阴道，0.7 = 七成阴道。也可输入 0–100。",
        "pose_title": "姿态（可多选）",
        "count_title": "生成条数",
        "any": "不限",
        "confirm_title": "确认并生成",
        "confirm_go": "生成",
        "footer": "↑↓ 移动  空格 勾选  a 全选  Enter 下一步  b 返回  q 放弃",
        "footer_text": "输入文字  Enter 下一步  b 返回  q 放弃",
        "empty_character": "角色词不能为空",
        "need_tty": "需要在终端里运行向导（stdin 不是 TTY）。",
        "need_exposure": "至少勾选一档暴露程度。",
    },
    "english": {
        "character_title": "Character prefix (immutable, always first)",
        "character_hint": "e.g. (sena_character:1.1), 1girl",
        "mode_title": "Mode",
        "mode_sfw": "SFW — everyday poses, no sex tags",
        "mode_nsfw": "NSFW — everyday poses, high exposure by default",
        "mode_sex": "SEX — poses only from the sex list",
        "exposure_title": "Clothing exposure (multi-select)",
        "exp_covered": "covered — uniforms, formal, wedding",
        "exp_casual": "casual — sundress, pajamas, daywear",
        "exp_revealing": "revealing — swimwear, nightgown",
        "exp_open": "open — add open clothes on marked outfits",
        "exp_nude": "nude — skip clothing, write nude",
        "sfw_warn": "SFW with high exposure (revealing / open / nude). Generation continues; sfw/nsfw are not written into the prompt.",
        "warn_continue": "Continue",
        "warn_back": "Go back",
        "clothing_title": "Clothing preference (multi-select; Any = uniform in the pool)",
        "chest_title": "Chest exposure",
        "chest_covered": "covered — no chest tags",
        "chest_cleavage": "cleavage",
        "chest_out": "breasts out (needs open clothes or nude)",
        "chest_nipples": "breasts out + nipples",
        "chest_auto": "by pose — front+open always breasts out/nipples; kneeling/prone optional",
        "belly_title": "Belly exposure",
        "belly_covered": "covered — no belly tags",
        "belly_midriff": "midriff",
        "belly_navel": "midriff + navel",
        "belly_auto": "by pose — navel when front and open/nude",
        "sfw_body_warn": "SFW with breasts out / nipples. Generation continues; sfw/nsfw are not written into the prompt.",
        "expression_title": "Expression (multi-select; Any = one tag from the mode pool)",
        "eye_title": "Eyes (multi-select; Any = one tag from the mode pool)",
        "face_none": "None — do not add expression/eye tags",
        "sfw_face_warn": "SFW with sex-leaning face tags (ahegao / orgasm / pain, …). Generation continues; sfw/nsfw are not written into the prompt.",
        "scene_title": "Scene preference (multi-select)",
        "family_title": "Pose families (multi-select)",
        "fam_face": "face to face — missionary, mating press, anvil, standing missionary",
        "fam_behind": "from behind — kneeling/prone doggy, reverse cowgirl",
        "fam_stand_behind": "standing from behind — standing doggy, reverse suspended, full nelson",
        "fam_side_behind": "side from behind — spooning, seventh posture",
        "fam_girl_on_top": "girl on top — cowgirl, squat, reverse, lap sit",
        "fam_hold": "held / carried — suspended congress, full nelson, lap sit",
        "ratio_title": "Vaginal share (0–1; anal weight is 1 minus this)",
        "ratio_hint": "0 = all anal, 1 = all vaginal, 0.7 = 70% vaginal. 0–100 is also accepted.",
        "pose_title": "Poses (multi-select)",
        "count_title": "How many prompts",
        "any": "Any",
        "confirm_title": "Confirm and generate",
        "confirm_go": "Generate",
        "footer": "↑↓ move  Space toggle  a all  Enter next  b back  q quit",
        "footer_text": "type  Enter next  b back  q quit",
        "empty_character": "Character prefix cannot be empty",
        "need_tty": "The wizard needs a terminal (stdin is not a TTY).",
        "need_exposure": "Select at least one exposure level.",
    },
}


def t(lang: str, key: str) -> str:
    table = STRINGS.get(lang, STRINGS["english"])
    return table[key]


def option_label(entry: Entry, lang: str) -> str:
    if lang == "chinese" and entry.comment:
        extra = ""
        if entry.channel:
            extra = f"  [{entry.channel}]"
        return f"{entry.comment}{extra}"
    extra = ""
    if entry.channel:
        extra = f"  [{entry.channel}]"
    return f"{entry.blob}{extra}"


class WizardUI:
    def text_input(self, title: str, hint: str, footer: str) -> str:
        raise NotImplementedError

    def radio(self, title: str, options: list[str], footer: str, selected: int = 0) -> int:
        raise NotImplementedError

    def checklist(
        self,
        title: str,
        options: list[str],
        footer: str,
        preselected: list[bool] | None = None,
        allow_any: bool = True,
        any_label: str = "Any",
        any_on: bool = False,
    ) -> tuple[bool, list[bool]]:
        raise NotImplementedError

    def number(self, title: str, footer: str, default: int, lo: int, hi: int) -> int:
        raise NotImplementedError

    def ratio(self, title: str, hint: str, footer: str, default: float) -> float:
        raise NotImplementedError

    def confirm(self, title: str, lines: list[str], footer: str, ok_label: str) -> bool:
        raise NotImplementedError

    def warn_continue(self, message: str, continue_label: str, back_label: str, footer: str) -> bool:
        raise NotImplementedError


class ScriptedUI(WizardUI):
    """Feeds canned answers in order. Used by tests."""

    def __init__(self, answers: list[object]):
        self.answers = list(answers)
        self.i = 0

    def _next(self) -> object:
        if self.i >= len(self.answers):
            raise WizardCancelled("scripted UI ran out of answers")
        value = self.answers[self.i]
        self.i += 1
        if value == "__back__":
            raise GoBack
        if value == "__quit__":
            raise WizardCancelled
        return value

    def text_input(self, title: str, hint: str, footer: str) -> str:
        return str(self._next())

    def radio(self, title: str, options: list[str], footer: str, selected: int = 0) -> int:
        return int(self._next())

    def checklist(
        self,
        title: str,
        options: list[str],
        footer: str,
        preselected: list[bool] | None = None,
        allow_any: bool = True,
        any_label: str = "Any",
        any_on: bool = False,
    ) -> tuple[bool, list[bool]]:
        value = self._next()
        if value == "any":
            return True, [False] * len(options)
        flags = list(value)
        return False, [bool(x) for x in flags]

    def number(self, title: str, footer: str, default: int, lo: int, hi: int) -> int:
        return int(self._next())

    def ratio(self, title: str, hint: str, footer: str, default: float) -> float:
        value = self._next()
        if isinstance(value, str):
            return parse_vaginal_ratio(value, default)
        return parse_vaginal_ratio(str(value), default)

    def confirm(self, title: str, lines: list[str], footer: str, ok_label: str) -> bool:
        return bool(self._next())

    def warn_continue(self, message: str, continue_label: str, back_label: str, footer: str) -> bool:
        return bool(self._next())


def _allowed_clothing(matrix: Matrix, exposure: Sequence[str]) -> list[Entry]:
    seen: set[tuple[str, ...]] = set()
    out: list[Entry] = []
    for bucket in exposure:
        if bucket == "nude":
            continue
        for item in clothing_pool_for_bucket(matrix, bucket):
            if item.key in seen:
                continue
            seen.add(item.key)
            out.append(item)
    return out


def run_wizard(matrix: Matrix, lang: str, ui: WizardUI) -> PromptSpec:
    character = ""
    mode = "sfw"
    exposure: list[str] = list(default_exposure("sfw"))
    clothing_any = True
    clothing_keys: frozenset[tuple[str, ...]] = frozenset()
    scene_any = True
    scene_keys: frozenset[tuple[str, ...]] = frozenset()
    pose_any = True
    pose_keys: frozenset[tuple[str, ...]] = frozenset()
    family_any = True
    families: frozenset[str] = frozenset()
    vaginal_ratio = VAGINAL_RATIO_DEFAULT
    chest = CHEST_DEFAULT["sfw"]
    belly = BELLY_DEFAULT["sfw"]
    expression_any = True
    expression_none = False
    expression_keys: frozenset[str] = frozenset()
    eye_any = True
    eye_none = False
    eye_keys: frozenset[str] = frozenset()
    count = COUNT_DEFAULT

    step = 0
    while True:
        try:
            if step == 0:
                character = ui.text_input(
                    t(lang, "character_title"),
                    t(lang, "character_hint"),
                    t(lang, "footer_text"),
                ).strip()
                if not character:
                    continue
                step = 1
                continue
            if step == 1:
                idx = ui.radio(
                    t(lang, "mode_title"),
                    [t(lang, "mode_sfw"), t(lang, "mode_nsfw"), t(lang, "mode_sex")],
                    t(lang, "footer"),
                    selected=MODES.index(mode) if mode in MODES else 0,
                )
                mode = MODES[idx]
                exposure = list(default_exposure(mode))
                chest = CHEST_DEFAULT[mode]
                belly = BELLY_DEFAULT[mode]
                step = 2
                continue
            if step == 2:
                pre = [level in exposure for level in EXPOSURE_LEVELS]
                any_on, flags = ui.checklist(
                    t(lang, "exposure_title"),
                    [
                        t(lang, "exp_covered"),
                        t(lang, "exp_casual"),
                        t(lang, "exp_revealing"),
                        t(lang, "exp_open"),
                        t(lang, "exp_nude"),
                    ],
                    t(lang, "footer"),
                    preselected=pre,
                    allow_any=False,
                )
                chosen = [level for level, on in zip(EXPOSURE_LEVELS, flags) if on]
                if not chosen:
                    chosen = list(default_exposure(mode))
                exposure = chosen
                if needs_sfw_exposure_warning(mode, exposure):
                    if not ui.warn_continue(
                        t(lang, "sfw_warn"),
                        t(lang, "warn_continue"),
                        t(lang, "warn_back"),
                        t(lang, "footer"),
                    ):
                        continue
                step = 3
                continue
            if step == 3:
                if exposure == ["nude"] or set(exposure) == {"nude"}:
                    clothing_any = True
                    clothing_keys = frozenset()
                    step = 4
                    continue
                items = _allowed_clothing(matrix, exposure)
                labels = [option_label(item, lang) for item in items]
                any_on, flags = ui.checklist(
                    t(lang, "clothing_title"),
                    labels,
                    t(lang, "footer"),
                    allow_any=True,
                    any_label=t(lang, "any"),
                    any_on=True,
                )
                clothing_any = any_on or not any(flags)
                clothing_keys = frozenset(
                    items[i].key for i, on in enumerate(flags) if on
                )
                step = 4
                continue
            if step == 4:
                idx = ui.radio(
                    t(lang, "chest_title"),
                    [
                        t(lang, "chest_covered"),
                        t(lang, "chest_cleavage"),
                        t(lang, "chest_out"),
                        t(lang, "chest_nipples"),
                        t(lang, "chest_auto"),
                    ],
                    t(lang, "footer"),
                    selected=CHEST_LEVELS.index(chest) if chest in CHEST_LEVELS else 0,
                )
                chest = CHEST_LEVELS[idx]
                if mode == "sfw" and chest in ("breasts_out", "nipples"):
                    if not ui.warn_continue(
                        t(lang, "sfw_body_warn"),
                        t(lang, "warn_continue"),
                        t(lang, "warn_back"),
                        t(lang, "footer"),
                    ):
                        continue
                step = 5
                continue
            if step == 5:
                idx = ui.radio(
                    t(lang, "belly_title"),
                    [
                        t(lang, "belly_covered"),
                        t(lang, "belly_midriff"),
                        t(lang, "belly_navel"),
                        t(lang, "belly_auto"),
                    ],
                    t(lang, "footer"),
                    selected=BELLY_LEVELS.index(belly) if belly in BELLY_LEVELS else 0,
                )
                belly = BELLY_LEVELS[idx]
                step = 6
                continue
            if step == 6:
                items = list(EXPRESSIONS)
                labels = [t(lang, "face_none")] + [face_label(item, lang) for item in items]
                any_on, flags = ui.checklist(
                    t(lang, "expression_title"),
                    labels,
                    t(lang, "footer"),
                    allow_any=True,
                    any_label=t(lang, "any"),
                    any_on=True,
                )
                if any_on:
                    expression_any, expression_none, expression_keys = True, False, frozenset()
                else:
                    expression_any, expression_none, expression_keys = _parse_face_choice(
                        flags, [item.tag for item in items]
                    )
                if needs_sfw_face_warning(mode, expression_keys):
                    if not ui.warn_continue(
                        t(lang, "sfw_face_warn"),
                        t(lang, "warn_continue"),
                        t(lang, "warn_back"),
                        t(lang, "footer"),
                    ):
                        continue
                step = 7
                continue
            if step == 7:
                items = list(EYES)
                labels = [t(lang, "face_none")] + [face_label(item, lang) for item in items]
                any_on, flags = ui.checklist(
                    t(lang, "eye_title"),
                    labels,
                    t(lang, "footer"),
                    allow_any=True,
                    any_label=t(lang, "any"),
                    any_on=True,
                )
                if any_on:
                    eye_any, eye_none, eye_keys = True, False, frozenset()
                else:
                    eye_any, eye_none, eye_keys = _parse_face_choice(
                        flags, [item.tag for item in items]
                    )
                if needs_sfw_face_warning(mode, eye_keys):
                    if not ui.warn_continue(
                        t(lang, "sfw_face_warn"),
                        t(lang, "warn_continue"),
                        t(lang, "warn_back"),
                        t(lang, "footer"),
                    ):
                        continue
                step = 8
                continue
            if step == 8:
                labels = [option_label(item, lang) for item in matrix.scenes]
                any_on, flags = ui.checklist(
                    t(lang, "scene_title"),
                    labels,
                    t(lang, "footer"),
                    allow_any=True,
                    any_label=t(lang, "any"),
                    any_on=True,
                )
                scene_any = any_on or not any(flags)
                scene_keys = frozenset(
                    matrix.scenes[i].key for i, on in enumerate(flags) if on
                )
                step = 9 if mode == "sex" else 11
                continue
            if step == 9:
                any_on, flags = ui.checklist(
                    t(lang, "family_title"),
                    [
                        t(lang, "fam_face"),
                        t(lang, "fam_behind"),
                        t(lang, "fam_stand_behind"),
                        t(lang, "fam_side_behind"),
                        t(lang, "fam_girl_on_top"),
                        t(lang, "fam_hold"),
                    ],
                    t(lang, "footer"),
                    allow_any=True,
                    any_label=t(lang, "any"),
                    any_on=True,
                )
                family_any = any_on or not any(flags)
                families = frozenset(
                    fam for fam, on in zip(POSE_FAMILY_IDS, flags) if on
                )
                step = 10
                continue
            if step == 10:
                vaginal_ratio = ui.ratio(
                    t(lang, "ratio_title"),
                    t(lang, "ratio_hint"),
                    t(lang, "footer_text"),
                    VAGINAL_RATIO_DEFAULT,
                )
                step = 11
                continue
            if step == 11:
                pool = pose_pool(matrix, mode)
                if mode == "sex":
                    pool = family_pool(
                        pool,
                        PromptSpec(
                            character=character,
                            mode=mode,
                            exposure=tuple(exposure),
                            family_any=family_any,
                            families=families,
                        ),
                    )
                labels = [option_label(item, lang) for item in pool]
                any_on, flags = ui.checklist(
                    t(lang, "pose_title"),
                    labels,
                    t(lang, "footer"),
                    allow_any=True,
                    any_label=t(lang, "any"),
                    any_on=True,
                )
                pose_any = any_on or not any(flags)
                pose_keys = frozenset(pool[i].key for i, on in enumerate(flags) if on)
                step = 12
                continue
            if step == 12:
                count = ui.number(
                    t(lang, "count_title"),
                    t(lang, "footer_text"),
                    COUNT_DEFAULT,
                    COUNT_MIN,
                    COUNT_MAX,
                )
                step = 13
                continue
            if step == 13:
                lines = [
                    f"character: {character}",
                    f"mode: {mode}",
                    f"exposure: {', '.join(exposure)}",
                    f"chest: {chest}",
                    f"belly: {belly}",
                    f"expression: {_face_summary(expression_any, expression_none, expression_keys)}",
                    f"eyes: {_face_summary(eye_any, eye_none, eye_keys)}",
                    f"count: {count}",
                ]
                if mode == "sex":
                    fams = "any" if family_any else ", ".join(sorted(families))
                    lines.append(f"families: {fams}")
                    lines.append(f"vaginal_ratio: {vaginal_ratio:g} (anal {1 - vaginal_ratio:g})")
                if not ui.confirm(
                    t(lang, "confirm_title"),
                    lines,
                    t(lang, "footer"),
                    t(lang, "confirm_go"),
                ):
                    raise GoBack
                return PromptSpec(
                    character=character,
                    mode=mode,
                    exposure=tuple(exposure),
                    clothing_any=clothing_any,
                    clothing_keys=clothing_keys,
                    scene_any=scene_any,
                    scene_keys=scene_keys,
                    pose_any=pose_any,
                    pose_keys=pose_keys,
                    family_any=family_any,
                    families=families,
                    vaginal_ratio=vaginal_ratio,
                    chest=chest,
                    belly=belly,
                    expression_any=expression_any,
                    expression_none=expression_none,
                    expression_keys=expression_keys,
                    eye_any=eye_any,
                    eye_none=eye_none,
                    eye_keys=eye_keys,
                    count=count,
                )
        except GoBack:
            if step == 11 and mode != "sex":
                step = 8
            elif step == 4 and (exposure == ["nude"] or set(exposure) == {"nude"}):
                step = 2
            elif step == 3 and (exposure == ["nude"] or set(exposure) == {"nude"}):
                step = 2
            else:
                step = max(0, step - 1)


# --- curses backend ---

def _add_wrapped(win, y: int, x: int, text: str, width: int, attr: int = 0) -> int:
    if width < 8:
        width = 8
    remaining = text
    row = y
    max_y, _ = win.getmaxyx()
    while remaining and row < max_y - 2:
        chunk = remaining
        if len(chunk) > width:
            chunk = remaining[:width]
        try:
            win.addstr(row, x, chunk, attr)
        except curses.error:
            break
        remaining = remaining[len(chunk) :]
        row += 1
    return row


def _curses_page_header(stdscr, title: str, footer: str) -> tuple[int, int, int]:
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()
    try:
        stdscr.addstr(0, 1, title[: max_x - 2], curses.A_BOLD)
        stdscr.addstr(max_y - 1, 1, footer[: max_x - 2], curses.A_DIM)
    except curses.error:
        pass
    return max_y, max_x, 2


def _key_is(key, *chars: str) -> bool:
    if isinstance(key, str):
        return key in chars
    if isinstance(key, int):
        return any(len(ch) == 1 and key == ord(ch) for ch in chars)
    return False


def _key_enter(key) -> bool:
    return key in ("\n", "\r") or (isinstance(key, int) and key in (curses.KEY_ENTER, 10, 13))


def _key_backspace(key) -> bool:
    return key in ("\x7f", "\b") or (isinstance(key, int) and key in (curses.KEY_BACKSPACE, 127, 8))


def _key_up(key) -> bool:
    return _key_is(key, "k") or (isinstance(key, int) and key == curses.KEY_UP)


def _key_down(key) -> bool:
    return _key_is(key, "j") or (isinstance(key, int) and key == curses.KEY_DOWN)


class CursesUI(WizardUI):
    def __init__(self, stdscr):
        self.stdscr = stdscr

    def _read_key(self):
        try:
            return self.stdscr.get_wch()
        except curses.error:
            return -1

    def text_input(self, title: str, hint: str, footer: str) -> str:
        buf: list[str] = []
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            _add_wrapped(self.stdscr, y, 1, hint, max_x - 2, curses.A_DIM)
            shown = "".join(buf)
            try:
                self.stdscr.addstr(y + 2, 1, (shown + "█")[: max_x - 2])
            except curses.error:
                pass
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_backspace(key):
                if buf:
                    buf.pop()
                continue
            if _key_is(key, "b", "B") and not buf:
                raise GoBack
            if _key_enter(key):
                text = "".join(buf).strip()
                if not text:
                    continue
                return text
            if isinstance(key, str) and key.isprintable():
                buf.append(key)

    def radio(self, title: str, options: list[str], footer: str, selected: int = 0) -> int:
        cursor = selected
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            for i, label in enumerate(options):
                mark = "(•)" if i == cursor else "( )"
                attr = curses.A_REVERSE if i == cursor else 0
                _add_wrapped(self.stdscr, y + i, 1, f"{mark} {label}", max_x - 2, attr)
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B") or _key_backspace(key):
                raise GoBack
            if _key_up(key):
                cursor = (cursor - 1) % len(options)
            elif _key_down(key):
                cursor = (cursor + 1) % len(options)
            elif _key_enter(key) or _key_is(key, " "):
                return cursor

    def checklist(
        self,
        title: str,
        options: list[str],
        footer: str,
        preselected: list[bool] | None = None,
        allow_any: bool = True,
        any_label: str = "Any",
        any_on: bool = False,
    ) -> tuple[bool, list[bool]]:
        flags = list(preselected) if preselected is not None else [False] * len(options)
        if len(flags) != len(options):
            flags = [False] * len(options)
        any_flag = bool(any_on and allow_any)
        if any_flag:
            flags = [False] * len(options)
        cursor = 0
        offset = 1 if allow_any else 0
        n = len(options) + offset
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            view_h = max(1, max_y - y - 2)
            start = 0
            if cursor >= view_h:
                start = cursor - view_h + 1
            rows: list[tuple[str, int]] = []
            if allow_any:
                mark = "[x]" if any_flag else "[ ]"
                rows.append((f"{mark} {any_label}", 0 if cursor == 0 else curses.A_DIM))
            for i, label in enumerate(options):
                mark = "[x]" if flags[i] and not any_flag else "[ ]"
                attr = curses.A_REVERSE if cursor == i + offset else 0
                rows.append((f"{mark} {label}", attr))
            for row_i, (text, attr) in enumerate(rows[start : start + view_h]):
                _add_wrapped(self.stdscr, y + row_i, 1, text, max_x - 2, attr)
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B") or _key_backspace(key):
                raise GoBack
            if _key_up(key):
                cursor = (cursor - 1) % n
            elif _key_down(key):
                cursor = (cursor + 1) % n
            elif _key_is(key, "a", "A"):
                any_flag = False
                flags = [True] * len(options)
            elif _key_is(key, " "):
                if allow_any and cursor == 0:
                    any_flag = not any_flag
                    if any_flag:
                        flags = [False] * len(options)
                else:
                    idx = cursor - offset
                    flags[idx] = not flags[idx]
                    if flags[idx]:
                        any_flag = False
            elif _key_enter(key):
                if allow_any and any_flag:
                    return True, [False] * len(options)
                if not allow_any and not any(flags):
                    continue
                return False, flags

    def number(self, title: str, footer: str, default: int, lo: int, hi: int) -> int:
        buf = list(str(default))
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            shown = "".join(buf) or str(default)
            try:
                self.stdscr.addstr(y, 1, f"{lo}..{hi}")
                self.stdscr.addstr(y + 2, 1, shown + "█")
            except curses.error:
                pass
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B") and "".join(buf) in ("", str(default)):
                raise GoBack
            if _key_backspace(key):
                if buf:
                    buf.pop()
                continue
            if _key_enter(key):
                raw = "".join(buf).strip() or str(default)
                try:
                    value = int(raw)
                except ValueError:
                    continue
                if lo <= value <= hi:
                    return value
            if isinstance(key, str) and key.isdigit():
                buf.append(key)
            elif isinstance(key, int) and ord("0") <= key <= ord("9"):
                buf.append(chr(key))

    def ratio(self, title: str, hint: str, footer: str, default: float) -> float:
        buf = list(f"{default:g}")
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            shown = "".join(buf) or f"{default:g}"
            try:
                _add_wrapped(self.stdscr, y, 1, hint, max_x - 2, curses.A_DIM)
                self.stdscr.addstr(y + 2, 1, shown + "█")
            except curses.error:
                pass
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B") and "".join(buf) in ("", f"{default:g}"):
                raise GoBack
            if _key_backspace(key):
                if buf:
                    buf.pop()
                continue
            if _key_enter(key):
                raw = "".join(buf).strip() or str(default)
                try:
                    return parse_vaginal_ratio(raw, default)
                except ValueError:
                    continue
            if isinstance(key, str) and (key.isdigit() or key == "."):
                if key == "." and "." in buf:
                    continue
                buf.append(key)
            elif isinstance(key, int) and ord("0") <= key <= ord("9"):
                buf.append(chr(key))

    def confirm(self, title: str, lines: list[str], footer: str, ok_label: str) -> bool:
        cursor = 0
        choices = [ok_label, "back"]
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            row = y
            for line in lines:
                row = _add_wrapped(self.stdscr, row, 1, line, max_x - 2) + 1
            for i, label in enumerate(choices):
                attr = curses.A_REVERSE if i == cursor else 0
                mark = "(•)" if i == cursor else "( )"
                _add_wrapped(self.stdscr, row + i, 1, f"{mark} {label}", max_x - 2, attr)
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B") or _key_backspace(key):
                return False
            if _key_up(key) or _key_down(key):
                cursor = 1 - cursor
            elif _key_enter(key) or _key_is(key, " "):
                return cursor == 0

    def warn_continue(self, message: str, continue_label: str, back_label: str, footer: str) -> bool:
        cursor = 0
        choices = [continue_label, back_label]
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, "!", footer)
            row = _add_wrapped(self.stdscr, y, 1, message, max_x - 2)
            for i, label in enumerate(choices):
                attr = curses.A_REVERSE if i == cursor else 0
                mark = "(•)" if i == cursor else "( )"
                _add_wrapped(self.stdscr, row + 2 + i, 1, f"{mark} {label}", max_x - 2, attr)
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B"):
                return False
            if _key_up(key) or _key_down(key):
                cursor = 1 - cursor
            elif _key_enter(key) or _key_is(key, " "):
                return cursor == 0


def _curses_main(stdscr, matrix: Matrix, lang: str) -> PromptSpec:
    curses.curs_set(0)
    stdscr.keypad(True)
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    return run_wizard(matrix, lang, CursesUI(stdscr))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Illustrious XL prompts from input_matrix.txt via a TUI wizard."
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=("chinese", "english"),
        default="chinese",
        help="Wizard language (default: chinese).",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help="Path to the input matrix (default: repo-root input_matrix.txt).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Also write prompts to this file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        matrix = load_matrix(args.matrix)
    except (OSError, MatrixError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(t(args.language, "need_tty"), file=sys.stderr)
        return 2
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass
    try:
        spec = curses.wrapper(lambda stdscr: _curses_main(stdscr, matrix, args.language))
    except WizardCancelled:
        return 1
    prompts = generate(spec, matrix)
    try:
        from rich.console import Console

        console = Console()
        for line in prompts:
            console.print(line)
    except Exception:
        for line in prompts:
            print(line)
    if args.output is not None:
        args.output.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

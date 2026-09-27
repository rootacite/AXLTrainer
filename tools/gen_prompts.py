#!/usr/bin/env python3
"""Illustrious XL prompt sampler over input_matrix.txt.

CLI picks the wizard language; character, mode, exposure and preferences are
collected on a curses checklist. Quality and rating tags are never appended.

The wizard's confirm page can save the finished configuration as a named
profile, and the next run offers those profiles up front so a saved one
generates straight away without walking the checklist again.
"""

from __future__ import annotations

import argparse
import curses
import json
import locale
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

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
CHANNELS = ("both", "anal", "vaginal", "none")
HOLE_CHANNELS = frozenset({"anal", "vaginal"})
SEX_STAGES = (
    "pose",
    "before",
    "during",
    "object_insertion",
    "fingering",
    "ejaculation",
    "after",
    "done",
)
STAGES_WITH_PARTNER = frozenset({"before", "during", "ejaculation", "after"})
STAGES_PENETRATING = frozenset({"during", "ejaculation"})
STAGES_OBJECT = frozenset({"object_insertion", "fingering"})


def default_stage_weights() -> dict[str, float]:
    return {stage: (1.0 if stage == "during" else 0.0) for stage in SEX_STAGES}

MODE_EXPOSURE_DEFAULTS = {
    "sfw": ("covered", "casual"),
    "nsfw": ("revealing", "open", "nude"),
    "sex": ("open",),
}

SECTION_NAMES = {"POSES", "CLOTHING", "SCENE", "SUFFIX", "SFW_POSES"}
GROUP_RE = re.compile(r"^\[(covered|casual|revealing)\]$")
OPEN_MARK = "(open clothes)"
CHANNEL_RE = re.compile(
    r"^(?P<tags>.+?)\s*:\s*(?P<channel>both|anal only|vaginal only|none)\s*$"
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


FACE_ANY = "any"
FACE_NONE = "none"


@dataclass(frozen=True)
class FaceGroup:
    """One mutually exclusive axis of the face page: at most one tag per prompt."""

    id: str
    zh: str
    en: str
    options: tuple[FaceOption, ...]
    default: str = FACE_ANY


FACE_GROUPS: tuple[FaceGroup, ...] = (
    FaceGroup(
        "expression",
        "总表情",
        "Expression",
        (
            FaceOption("smile", _ALL_MODES, "微笑 — smile", "smile"),
            FaceOption("grin", _ALL_MODES, "咧嘴笑 — grin", "grin"),
            FaceOption("smirk", _ALL_MODES, "坏笑 — smirk", "smirk"),
            FaceOption("shy", _ALL_MODES, "害羞 — shy", "shy"),
            FaceOption("embarrassed", _ALL_MODES, "尴尬 — embarrassed", "embarrassed"),
            FaceOption("angry", _ALL_MODES, "生气 — angry", "angry"),
            FaceOption("surprised", _ALL_MODES, "惊讶 — surprised", "surprised"),
            FaceOption("scared", _ALL_MODES, "害怕 — scared", "scared"),
            FaceOption("naughty face", _NSFW_MODES, "坏脸 — naughty face", "naughty face"),
            FaceOption("pain", _NSFW_MODES, "痛苦 — pain", "pain"),
            FaceOption("ahegao", _NSFW_MODES, "阿嘿颜 — ahegao", "ahegao"),
            FaceOption("orgasm", _NSFW_MODES, "高潮 — orgasm", "orgasm"),
        ),
    ),
    FaceGroup(
        "gaze",
        "视线",
        "Gaze",
        (
            FaceOption("looking at viewer", _ALL_MODES, "看镜头 — looking at viewer", "looking at viewer"),
            FaceOption("looking away", _ALL_MODES, "视线移开 — looking away", "looking away"),
        ),
        FACE_NONE,
    ),
    FaceGroup(
        "eyes",
        "眼睛状态",
        "Eye state",
        (
            FaceOption("open eyes", _ALL_MODES, "睁眼 — open eyes", "open eyes"),
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
        ),
    ),
    FaceGroup(
        "mouth",
        "嘴状态",
        "Mouth",
        (
            FaceOption("open mouth", _ALL_MODES, "张嘴 — open mouth", "open mouth"),
            FaceOption("closed mouth", _ALL_MODES, "闭嘴 — closed mouth", "closed mouth"),
            FaceOption("tongue out", _NSFW_MODES, "吐舌 — tongue out", "tongue out"),
            FaceOption("clenched teeth", _NSFW_MODES, "咬牙 — clenched teeth", "clenched teeth"),
            FaceOption("drooling", _NSFW_MODES, "流口水 — drooling", "drooling"),
            FaceOption("heavy breathing", _NSFW_MODES, "喘息 — heavy breathing", "heavy breathing"),
        ),
        FACE_NONE,
    ),
    FaceGroup(
        "blush",
        "脸红",
        "Blush",
        (FaceOption("blush", _ALL_MODES, "脸红 — blush", "blush"),),
        FACE_NONE,
    ),
    FaceGroup(
        "tears",
        "眼泪",
        "Tears",
        (
            FaceOption("tears", _ALL_MODES, "含泪 — tears", "tears"),
            FaceOption("crying", _ALL_MODES, "哭泣 — crying", "crying"),
        ),
        FACE_NONE,
    ),
)

FACE_GROUPS_BY_ID = {group.id: group for group in FACE_GROUPS}
FACE_TAG_GROUP: dict[str, str] = {
    option.tag: group.id for group in FACE_GROUPS for option in group.options
}
# Cross-group contradictions; pairs inside one group are already exclusive there.
FACE_CONFLICTS: dict[str, frozenset[str]] = {
    "closed eyes": frozenset({"looking at viewer", "looking away"}),
    "looking at viewer": frozenset({"closed eyes"}),
    "looking away": frozenset({"closed eyes"}),
}


def default_face() -> dict[str, str]:
    return {group.id: group.default for group in FACE_GROUPS}


def group_tags(group: FaceGroup) -> tuple[str, ...]:
    return tuple(option.tag for option in group.options)


def group_pool(group: FaceGroup, mode: str) -> list[str]:
    return [option.tag for option in group.options if mode in option.modes]

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
    stage_weights: dict[str, float] = field(default_factory=default_stage_weights)
    chest: str = "auto"
    belly: str = "auto"
    # Group id -> FACE_ANY, FACE_NONE, or a tuple of tags; a tuple means "pick one of these
    # per prompt", so a group still contributes at most one tag to a line.
    face: dict[str, str | tuple[str, ...]] = field(default_factory=default_face)
    count: int = COUNT_DEFAULT


FacePick = str | tuple[str, ...]


def face_choice_of(face: dict[str, FacePick], group: FaceGroup) -> FacePick:
    return face.get(group.id, group.default)


def face_choice(spec: PromptSpec, group: FaceGroup) -> FacePick:
    return face_choice_of(spec.face, group)


def face_tags_of(value: FacePick) -> tuple[str, ...]:
    """The explicitly chosen tags; empty for the 'any' and 'none' sentinels."""
    return () if isinstance(value, str) else tuple(value)


def selected_face_tags(face: dict[str, FacePick]) -> frozenset[str]:
    return frozenset(tag for value in face.values() for tag in face_tags_of(value))


def face_conflict(tag: str, other: Iterable[str]) -> bool:
    return bool(FACE_CONFLICTS.get(tag, frozenset()) & set(other))


def split_tags(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_channel(raw: str) -> str:
    if raw == "both":
        return "both"
    if raw == "anal only":
        return "anal"
    if raw == "vaginal only":
        return "vaginal"
    if raw == "none":
        return "none"
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
                    f"line {lineno}: pose must end with ': both|anal only|vaginal only|none'"
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


def parse_stage_weight(raw: str, default: float = 0.0) -> float:
    return parse_vaginal_ratio(raw, default)


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


def face_tags(spec: PromptSpec, pose: Entry, rng: random.Random) -> list[str]:
    """At most one tag per group, none of them contradicting the pose or an earlier group."""
    pose_tags = set(pose.tags)
    out: list[str] = []
    for group in FACE_GROUPS:
        if pose_tags.intersection(group_tags(group)):
            continue  # the pose already fills this axis
        choice = face_choice(spec, group)
        if choice == FACE_NONE:
            continue
        taken = pose_tags | set(out)
        pool = (
            group_pool(group, spec.mode)
            if choice == FACE_ANY
            else [tag for tag in face_tags_of(choice)]
        )
        candidates = [
            tag for tag in pool if tag not in taken and not face_conflict(tag, taken)
        ]
        if not candidates:
            continue
        out.append(rng.choice(candidates))
    return out


def face_selection(group: FaceGroup, value: FacePick) -> tuple[bool, list[bool]]:
    """(any?, per-tag flags) for the wizard page, from a stored group value."""
    flags = [False] * len(group.options)
    if value == FACE_ANY:
        return True, flags
    tags = group_tags(group)
    for tag in face_tags_of(value):
        if tag in tags:
            flags[tags.index(tag)] = True
    return False, flags


def face_value(group: FaceGroup, any_on: bool, flags: Sequence[bool]) -> FacePick:
    """The stored group value for a page answer: any > tags > nothing."""
    if any_on:
        return FACE_ANY
    tags = tuple(
        tag for tag, on in zip(group_tags(group), flags) if on
    )
    return tags or FACE_NONE


def needs_sfw_face_warning(mode: str, keys: Iterable[str]) -> bool:
    return mode == "sfw" and bool(set(keys) & SEX_FACE_TAGS)


def face_value_label(value: FacePick, lang: str) -> str:
    if value == FACE_ANY:
        return t(lang, "any")
    if value == FACE_NONE:
        return t(lang, "face_skip")
    return "/".join(face_tags_of(value))


def face_summary(face: dict[str, FacePick], lang: str) -> str:
    parts: list[str] = []
    for group in FACE_GROUPS:
        value = face.get(group.id, group.default)
        name = group.zh if lang == "chinese" else group.en
        parts.append(f"{name}={face_value_label(value, lang)}")
    return ", ".join(parts)


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


def pose_allows_hole(pose: Entry) -> bool:
    return pose.channel in HOLE_CHANNELS or pose.channel == "both"


def anatomy_tags(pose: Entry, channel: str, include_penis: bool = True) -> list[str]:
    blob = pose.blob
    out: list[str] = []
    if include_penis:
        out.append("penis")
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


def stage_tags(stage: str, channel: str) -> list[str]:
    if stage == "object_insertion":
        if channel == "anal":
            return ["anal object insertion"]
        if channel == "vaginal":
            return ["vaginal object insertion"]
        return []
    if stage == "fingering":
        if channel == "anal":
            return ["anal fingering"]
        if channel == "vaginal":
            return ["fingering"]
        return []
    if channel not in HOLE_CHANNELS:
        if stage == "pose":
            return ["presenting"]
        if stage == "before":
            return []
        if stage == "during":
            return []
        if stage == "ejaculation":
            return ["ejaculation"]
        return ["after sex", "cumdrip"]
    if stage == "pose":
        return ["presenting"]
    if stage == "before":
        if channel == "anal":
            return ["imminent anal", "penis on ass"]
        return ["imminent vaginal", "penis on pussy"]
    if stage == "during":
        return []
    if stage == "ejaculation":
        if channel == "anal":
            return ["ejaculation", "cum in ass", "cum overflow"]
        return ["ejaculation", "cum in pussy", "cum overflow"]
    extra = ["after sex", "cumdrip", "gaping"]
    if channel == "anal":
        extra.extend(["after anal", "cum in ass"])
    else:
        extra.extend(["after vaginal", "cum in pussy"])
    return extra


def stage_weight_value(spec: PromptSpec, stage: str) -> int:
    value = spec.stage_weights.get(stage, 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(1000, max(0, round(number * 1000)))


def pick_stage(spec: PromptSpec, rng: random.Random) -> str:
    weights = [stage_weight_value(spec, stage) for stage in SEX_STAGES]
    total = sum(weights)
    if total <= 0:
        return "during"
    pick = rng.randrange(total)
    acc = 0
    for stage, weight in zip(SEX_STAGES, weights):
        acc += weight
        if pick < acc:
            return stage
    return "during"


def bind_stage_to_poses(
    poses: Sequence[Entry], stage: str | None
) -> tuple[str | None, list[Entry]]:
    """Object-insertion / fingering only ride poses that can take a hole."""
    pool = list(poses)
    if stage not in STAGES_OBJECT:
        return stage, pool
    hole_poses = [pose for pose in pool if pose_allows_hole(pose)]
    if hole_poses:
        return stage, hole_poses
    return "during", pool


def pick_channel(pose: Entry, spec: PromptSpec, rng: random.Random) -> str:
    if pose.channel == "anal":
        return "anal"
    if pose.channel == "vaginal":
        return "vaginal"
    if pose.channel == "none":
        return "none"
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
    stage: str | None = None,
) -> str:
    prefix = split_tags(spec.character)
    extra: list[str] = []
    if spec.mode == "sex":
        stage = stage if stage in SEX_STAGES else "during"
        if stage in STAGES_WITH_PARTNER:
            extra.extend(["1boy", "hetero"])
        else:
            extra.append("solo")
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
        hole = channel or "vaginal"
        if stage in STAGES_PENETRATING and hole in HOLE_CHANNELS:
            extra.append("sex")
            extra.append(hole)
        extra.extend(anatomy_tags(pose, hole, include_penis=stage in STAGES_WITH_PARTNER))
        extra.extend(stage_tags(stage, hole))
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
        stage = pick_stage(spec, rng) if spec.mode == "sex" else None
        stage, stage_poses = bind_stage_to_poses(poses, stage)
        pose_weights = [pose_weight(pose, spec) for pose in stage_poses]
        pose = weighted_choice(rng, stage_poses, pose_weights)
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
            stage,
        )
        if combo in used:
            continue
        used.add(combo)
        results.append(
            assemble(
                spec, pose, clothing, add_open, scene, suffix, channel, torso, face, stage
            )
        )
    while len(results) < spec.count:
        stage = pick_stage(spec, rng) if spec.mode == "sex" else None
        stage, stage_poses = bind_stage_to_poses(poses, stage)
        pose = weighted_choice(rng, stage_poses, [pose_weight(p, spec) for p in stage_poses])
        clothing, add_open = pick_clothing(matrix, spec, rng)
        scene = pick_scene(matrix, spec, pose, rng, notes)
        suffix = rng.choice(matrix.suffixes)
        channel = pick_channel(pose, spec, rng) if spec.mode == "sex" else None
        torso = torso_tags(spec, pose, clothing, add_open, rng)
        face = face_tags(spec, pose, rng)
        results.append(
            assemble(
                spec, pose, clothing, add_open, scene, suffix, channel, torso, face, stage
            )
        )
    return results


# --- profiles (named wizard configurations) ---

PROFILE_VERSION = 3
PROFILE_SUFFIX = ".json"

# Which new groups each v1 face axis covered, so a v1 profile's "locked" axis can be
# migrated without inventing tags the user never asked for.
_V1_FACE_AXES = (
    (
        ("expression_any", "expression_none", "expression_keys", "expression"),
        ("expression", "mouth", "blush", "tears"),
    ),
    (
        ("eye_any", "eye_none", "eye_keys", "eyes"),
        ("gaze", "eyes"),
    ),
)
_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class ProfileError(ValueError):
    pass


def default_profiles_dir() -> Path:
    return REPO_ROOT / "prompt_profiles"


def safe_profile_name(name: str) -> str:
    """Filesystem-safe stem for a profile name (unicode is kept)."""
    return _UNSAFE_NAME_RE.sub("_", name).strip().strip(".")


def profile_file(profiles_dir: Path, name: str) -> Path:
    safe = safe_profile_name(name)
    if not safe:
        raise ProfileError(f"invalid profile name: {name!r}")
    return profiles_dir / f"{safe}{PROFILE_SUFFIX}"


def spec_to_dict(spec: PromptSpec) -> dict:
    return {
        "character": spec.character,
        "mode": spec.mode,
        "exposure": list(spec.exposure),
        "clothing_any": spec.clothing_any,
        "clothing_keys": [list(key) for key in sorted(spec.clothing_keys)],
        "scene_any": spec.scene_any,
        "scene_keys": [list(key) for key in sorted(spec.scene_keys)],
        "pose_any": spec.pose_any,
        "pose_keys": [list(key) for key in sorted(spec.pose_keys)],
        "family_any": spec.family_any,
        "families": sorted(spec.families),
        "vaginal_ratio": spec.vaginal_ratio,
        "stage_weights": {stage: spec.stage_weights.get(stage, 0.0) for stage in SEX_STAGES},
        "chest": spec.chest,
        "belly": spec.belly,
        "face": {group.id: _face_to_json(face_choice(spec, group)) for group in FACE_GROUPS},
        "count": spec.count,
    }


def _profile_bool(data: dict, key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ProfileError(f"{key} must be true or false")
    return value


def _profile_key_set(data: dict, key: str) -> frozenset[tuple[str, ...]]:
    raw = data.get(key, [])
    if not isinstance(raw, list):
        raise ProfileError(f"{key} must be a list")
    keys: set[tuple[str, ...]] = set()
    for item in raw:
        if isinstance(item, str) and item:
            keys.add((item,))
        elif isinstance(item, list) and item and all(isinstance(x, str) and x for x in item):
            keys.add(tuple(item))
        else:
            raise ProfileError(f"{key} entries must be tag strings or lists of tag strings")
    return frozenset(keys)


def _profile_str_set(
    data: dict, key: str, allowed: Sequence[str] | None = None
) -> frozenset[str]:
    raw = data.get(key, [])
    if not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw):
        raise ProfileError(f"{key} must be a list of strings")
    values = frozenset(raw)
    if allowed is not None:
        unknown = values - set(allowed)
        if unknown:
            raise ProfileError(f"unknown {key}: {', '.join(sorted(unknown))}")
    return values


def _profile_stage_weights(data: dict) -> dict[str, float]:
    raw = data.get("stage_weights")
    if raw is None:
        return default_stage_weights()
    if not isinstance(raw, dict):
        raise ProfileError("stage_weights must be a JSON object")
    weights = {stage: 0.0 for stage in SEX_STAGES}
    for key, value in raw.items():
        if key not in SEX_STAGES:
            raise ProfileError(f"unknown stage: {key}")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ProfileError(f"stage_weights.{key} must be a number in 0..1")
        weights[key] = float(value)
    return weights


def _face_to_json(value: FacePick) -> str | list[str]:
    return value if isinstance(value, str) else list(value)


def _profile_face(data: dict) -> dict[str, FacePick]:
    raw = data.get("face", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ProfileError("face must be a JSON object")
    face: dict[str, FacePick] = {}
    for key, value in raw.items():
        group = FACE_GROUPS_BY_ID.get(key)
        if group is None:
            raise ProfileError(f"unknown face group: {key!r}")
        if isinstance(value, str):
            if value not in (FACE_ANY, FACE_NONE):
                raise ProfileError(f"face.{key} must be 'any', 'none' or a list of tags")
            face[key] = value
            continue
        if not isinstance(value, list) or not value:
            raise ProfileError(f"face.{key} must be 'any', 'none' or a non-empty list")
        for tag in value:
            if not isinstance(tag, str) or tag not in group_tags(group):
                raise ProfileError(f"{tag!r} is not an option of face group {key!r}")
        face[key] = tuple(dict.fromkeys(value))
    return face


def spec_from_dict(data: dict) -> PromptSpec:
    if not isinstance(data, dict):
        raise ProfileError("profile spec must be a JSON object")
    character = data.get("character")
    if not isinstance(character, str) or not character.strip():
        raise ProfileError("character must be a non-empty string")
    mode = data.get("mode")
    if mode not in MODES:
        raise ProfileError(f"mode must be one of {', '.join(MODES)}")
    raw_exposure = data.get("exposure")
    if not isinstance(raw_exposure, list) or not raw_exposure:
        raise ProfileError("exposure must be a non-empty list")
    unknown = [level for level in raw_exposure if level not in EXPOSURE_LEVELS]
    if unknown:
        raise ProfileError(f"unknown exposure level: {', '.join(map(str, unknown))}")
    ratio = data.get("vaginal_ratio", VAGINAL_RATIO_DEFAULT)
    if (
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not 0.0 <= float(ratio) <= 1.0
    ):
        raise ProfileError("vaginal_ratio must be a number in 0..1")
    stage_weights = _profile_stage_weights(data)
    chest = data.get("chest", "auto")
    if chest not in CHEST_LEVELS:
        raise ProfileError(f"chest must be one of {', '.join(CHEST_LEVELS)}")
    belly = data.get("belly", "auto")
    if belly not in BELLY_LEVELS:
        raise ProfileError(f"belly must be one of {', '.join(BELLY_LEVELS)}")
    count = data.get("count", COUNT_DEFAULT)
    if not isinstance(count, int) or isinstance(count, bool):
        raise ProfileError("count must be an integer")
    if not COUNT_MIN <= count <= COUNT_MAX:
        raise ProfileError(f"count must be in {COUNT_MIN}..{COUNT_MAX}")
    return PromptSpec(
        character=character,
        mode=mode,
        exposure=tuple(raw_exposure),
        clothing_any=_profile_bool(data, "clothing_any", True),
        clothing_keys=_profile_key_set(data, "clothing_keys"),
        scene_any=_profile_bool(data, "scene_any", True),
        scene_keys=_profile_key_set(data, "scene_keys"),
        pose_any=_profile_bool(data, "pose_any", True),
        pose_keys=_profile_key_set(data, "pose_keys"),
        family_any=_profile_bool(data, "family_any", True),
        families=_profile_str_set(data, "families", POSE_FAMILY_IDS),
        vaginal_ratio=float(ratio),
        stage_weights=stage_weights,
        chest=chest,
        belly=belly,
        face=_profile_face(data),
        count=count,
    )


def save_profile(profiles_dir: Path, name: str, spec: PromptSpec) -> Path:
    path = profile_file(profiles_dir, name)
    payload = {
        "version": PROFILE_VERSION,
        "name": name.strip(),
        "spec": spec_to_dict(spec),
    }
    profiles_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
    return path


def _upgrade_profile_v1(spec_data: dict, warn: bool = True) -> dict:
    """Map a v1 spec's expression/eye fields onto the grouped, multi-select face model.

    Each locked tag goes to whichever group now holds it, so v1's multi-select becomes a
    candidate list per group. An axis left 'any' or 'none' keeps that meaning for every
    group it covered.
    """
    upgraded = dict(spec_data)
    face: dict[str, str | list[str]] = {}
    buckets: dict[str, list[str]] = {}
    for (any_key, none_key, keys_key, primary), scope in _V1_FACE_AXES:
        any_value = upgraded.pop(any_key, True)
        none_value = upgraded.pop(none_key, False)
        any_value = any_value if isinstance(any_value, bool) else True
        none_value = none_value if isinstance(none_value, bool) else False
        for group_id in scope:
            face[group_id] = FACE_NONE
        if any_value and not none_value:
            # v1 "any" rolled from one pool; only the axis's own group keeps that here.
            face[primary] = FACE_ANY
        raw_keys = upgraded.pop(keys_key, [])
        if isinstance(raw_keys, list):
            for tag in raw_keys:
                group_id = FACE_TAG_GROUP.get(tag) if isinstance(tag, str) else None
                if group_id is None:
                    if warn:
                        print(
                            f"warning: profile v1 face tag {tag!r} is unknown; dropped",
                            file=sys.stderr,
                        )
                    continue
                buckets.setdefault(group_id, []).append(tag)
    for group_id, tags in buckets.items():
        face[group_id] = list(dict.fromkeys(tags))
    upgraded["face"] = face
    return upgraded


def _upgrade_profile_v2(spec_data: dict) -> dict:
    """v2 stored one tag per group; v3 stores a candidate list, so wrap bare tags."""
    upgraded = dict(spec_data)
    raw = upgraded.get("face")
    if isinstance(raw, dict):
        upgraded["face"] = {
            key: [value] if isinstance(value, str) and value not in (FACE_ANY, FACE_NONE) else value
            for key, value in raw.items()
        }
    return upgraded


def _read_profile(path: Path, warn: bool = True) -> tuple[str, PromptSpec]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProfileError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path.name}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ProfileError(f"{path.name}: profile must be a JSON object")
    if "spec" not in data:
        raise ProfileError(f"{path.name}: missing spec")
    version = data.get("version", PROFILE_VERSION)
    raw_spec = data["spec"]
    if version == 1:
        if isinstance(raw_spec, dict):
            raw_spec = _upgrade_profile_v1(raw_spec, warn=warn)
    elif version == 2:
        if isinstance(raw_spec, dict):
            raw_spec = _upgrade_profile_v2(raw_spec)
    elif version != PROFILE_VERSION:
        raise ProfileError(f"{path.name}: unsupported profile version {version!r}")
    name = data.get("name")
    name = name.strip() if isinstance(name, str) and name.strip() else path.stem
    return name, spec_from_dict(raw_spec)


def load_profile(path: Path) -> PromptSpec:
    return _read_profile(path)[1]


def list_profiles(profiles_dir: Path) -> list[tuple[str, Path]]:
    """(display name, path) for every profile, sorted by name; unreadable files keep their stem."""
    if not profiles_dir.is_dir():
        return []
    entries: list[tuple[str, Path]] = []
    for path in profiles_dir.glob(f"*{PROFILE_SUFFIX}"):
        try:
            name = _read_profile(path, warn=False)[0]
        except ProfileError:
            name = path.stem
        entries.append((name, path))
    entries.sort(key=lambda item: (item[0].lower(), item[0]))
    return entries


def resolve_profile(profiles_dir: Path, name: str) -> Path:
    wanted = name.strip().lower()
    entries = list_profiles(profiles_dir)
    for stored, path in entries:
        if stored.lower() == wanted or path.stem.lower() == wanted:
            return path
    available = ", ".join(stored for stored, _ in entries) or "none"
    raise ProfileError(f"profile {name!r} not found (available: {available})")


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
        "face_title": "表情（组内可多选：每条从勾选项里抽一个；一条不勾 = 不写；「不限」= 该组整池随机）",
        "face_skip": "不写",
        "footer_face": "↑↓ 移动  n/p 上下一组  空格 勾选  Enter 下一步  b 返回  q 放弃",
        "sfw_face_warn": "SFW 选了偏性交的表情（阿嘿颜 / 高潮 / 痛苦等）。仍可生成，提示词里不会写入 sfw/nsfw。",
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
        "stage_title": "性爱阶段占比（相对权重，0 表示不抽）",
        "stage_hint": "↑↓ 改哪一行；数字同阴道占比。默认只抽性爱中。",
        "stage_pose": "摆姿势 — 性交体位、无阴茎",
        "stage_before": "性爱前 — 对准、未插入",
        "stage_during": "性爱中 — 已插入",
        "stage_object_insertion": "物体插入 — 玩具/物体，无阴茎（随体位走肛或阴道）",
        "stage_fingering": "手指 — 无阴茎（后穴 anal fingering，阴道 fingering）",
        "stage_ejaculation": "射精 — 中出（仍插入）",
        "stage_after": "性爱后 — 已拔出、精液、扩开（有阴茎）",
        "stage_done": "完成 — 同性爱后，无阴茎",
        "pose_title": "姿态（可多选）",
        "count_title": "生成条数",
        "any": "不限",
        "confirm_title": "确认并生成",
        "confirm_go": "生成",
        "confirm_save": "保存为 Profile",
        "profile_title": "选择 Profile（或新建一个）",
        "profile_new": "＋ 新建（运行向导）",
        "profile_save_title": "Profile 名称",
        "profile_save_hint": "保存后下次启动可直接选它生成；同名会覆盖",
        "profile_saved": "已保存 Profile",
        "profile_save_cancelled": "未保存（已取消覆盖）",
        "profile_save_failed": "保存失败",
        "profile_overwrite_title": "同名 Profile 已存在，要覆盖吗？",
        "profile_overwrite_yes": "覆盖",
        "manifest_title": "配置总清单（Enter 编辑选中项）",
        "manifest_go": "生成",
        "manifest_save": "保存为 Profile",
        "manifest_back": "返回选择 Profile",
        "manifest_selected": "已选 {n} 项",
        "manifest_stage_fallback": "全为 0 → during",
        "footer_manifest": "↑↓ 移动  Enter 编辑/执行  b 返回  q 放弃",
        "item_character": "角色词",
        "item_mode": "模式",
        "item_exposure": "暴露程度",
        "item_clothing": "服装",
        "item_chest": "胸部",
        "item_belly": "腹部",
        "item_face": "表情",
        "item_scene": "场景",
        "item_family": "体位大类",
        "item_ratio": "阴道占比",
        "item_stages": "阶段权重",
        "item_pose": "姿态",
        "item_count": "条数",
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
        "face_title": "Face (multi-select per group: one of the ticked tags per prompt; none ticked = off; Any = the whole pool)",
        "face_skip": "off",
        "footer_face": "↑↓ move  n/p group  Space toggle  Enter next  b back  q quit",
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
        "stage_title": "Sex stage share (relative weights; 0 skips that stage)",
        "stage_hint": "↑↓ picks the row. Numbers work like the vaginal share. Default is during only.",
        "stage_pose": "posing — sex pose, no penis",
        "stage_before": "before — aimed, not inserted",
        "stage_during": "during — inserted",
        "stage_object_insertion": "object insertion — toy/object, no penis (anal or vaginal from the pose)",
        "stage_fingering": "fingering — no penis (anal fingering on anal poses, fingering on vaginal)",
        "stage_ejaculation": "ejaculation — creampie while still in",
        "stage_after": "after — pulled out, drip, gape, penis still in frame",
        "stage_done": "done — same as after, no penis",
        "pose_title": "Poses (multi-select)",
        "count_title": "How many prompts",
        "any": "Any",
        "confirm_title": "Confirm and generate",
        "confirm_go": "Generate",
        "confirm_save": "Save as profile",
        "profile_title": "Choose a profile (or start a new one)",
        "profile_new": "+ New (run the wizard)",
        "profile_save_title": "Profile name",
        "profile_save_hint": "Saved profiles can generate straight away next run; a same-name profile is overwritten",
        "profile_saved": "Saved profile",
        "profile_save_cancelled": "Not saved (overwrite declined)",
        "profile_save_failed": "Save failed",
        "profile_overwrite_title": "A profile with that name exists; overwrite it?",
        "profile_overwrite_yes": "Overwrite",
        "manifest_title": "Configuration (Enter edits the highlighted row)",
        "manifest_go": "Generate",
        "manifest_save": "Save as profile",
        "manifest_back": "Back to profiles",
        "manifest_selected": "{n} selected",
        "manifest_stage_fallback": "all zero → during",
        "footer_manifest": "↑↓ move  Enter edit/run  b back  q quit",
        "item_character": "Character",
        "item_mode": "Mode",
        "item_exposure": "Exposure",
        "item_clothing": "Clothing",
        "item_chest": "Chest",
        "item_belly": "Belly",
        "item_face": "Face",
        "item_scene": "Scene",
        "item_family": "Pose families",
        "item_ratio": "Vaginal share",
        "item_stages": "Stage weights",
        "item_pose": "Poses",
        "item_count": "Count",
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
    def text_input(self, title: str, hint: str, footer: str, initial: str = "") -> str:
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

    def face_groups(
        self,
        title: str,
        groups: list[tuple[str, list[str]]],
        footer: str,
        preselected: list[tuple[bool, list[bool]]],
    ) -> list[tuple[bool, list[bool]]]:
        """Per group (any?, flags per tag). Labels are [any, *tags]; nothing ticked = skip."""
        raise NotImplementedError

    def number(self, title: str, footer: str, default: int, lo: int, hi: int) -> int:
        raise NotImplementedError

    def ratio(self, title: str, hint: str, footer: str, default: float) -> float:
        raise NotImplementedError

    def weights(
        self,
        title: str,
        items: list[tuple[str, str]],
        hint: str,
        footer: str,
        defaults: dict[str, float],
    ) -> dict[str, float]:
        raise NotImplementedError

    def manifest(
        self,
        title: str,
        items: list[tuple[str, str]],
        actions: list[str],
        footer: str,
        note: str = "",
        initial: int = 0,
    ) -> int:
        """Index into items, or len(items) + k for action k; `initial` preselects a row."""
        raise NotImplementedError

    def confirm(self, title: str, lines: list[str], footer: str, ok_label: str) -> bool:
        raise NotImplementedError

    def confirm_with_save(
        self,
        title: str,
        lines: list[str],
        footer: str,
        ok_label: str,
        save_label: str,
    ) -> str:
        """Return 'go', 'save' or 'back'; backends without a save choice return 'go'/'back'."""
        return "go" if self.confirm(title, lines, footer, ok_label) else "back"

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

    def text_input(self, title: str, hint: str, footer: str, initial: str = "") -> str:
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

    def face_groups(
        self,
        title: str,
        groups: list[tuple[str, list[str]]],
        footer: str,
        preselected: list[tuple[bool, list[bool]]],
    ) -> list[tuple[bool, list[bool]]]:
        """Answers per group: 'default', 'any', 'none', or a list of tag indices."""
        answers = self._next()
        if answers == "default":
            return [(bool(any_on), list(flags)) for any_on, flags in preselected]
        out: list[tuple[bool, list[bool]]] = []
        for index, (_, labels) in enumerate(groups):
            count = max(0, len(labels) - 1)
            answer = answers[index]
            if answer == "default":
                any_on, flags = preselected[index]
                out.append((bool(any_on), list(flags)))
            elif answer == "any":
                out.append((True, [False] * count))
            elif answer in ("none", "off") or answer == []:
                out.append((False, [False] * count))
            else:
                flags = [False] * count
                for tag_index in answer:
                    flags[int(tag_index)] = True
                out.append((False, flags))
        return out

    def number(self, title: str, footer: str, default: int, lo: int, hi: int) -> int:
        return int(self._next())

    def ratio(self, title: str, hint: str, footer: str, default: float) -> float:
        value = self._next()
        if isinstance(value, str):
            return parse_vaginal_ratio(value, default)
        return parse_vaginal_ratio(str(value), default)

    def weights(
        self,
        title: str,
        items: list[tuple[str, str]],
        hint: str,
        footer: str,
        defaults: dict[str, float],
    ) -> dict[str, float]:
        value = self._next()
        if value == "default":
            return {key: float(defaults.get(key, 0.0)) for key, _ in items}
        if isinstance(value, dict):
            out = {key: float(defaults.get(key, 0.0)) for key, _ in items}
            for key, weight in value.items():
                if key in out:
                    out[key] = parse_stage_weight(str(weight), out[key])
            return out
        flags = list(value)
        out: dict[str, float] = {}
        for i, (key, _) in enumerate(items):
            fallback = float(defaults.get(key, 0.0))
            if i < len(flags):
                out[key] = parse_stage_weight(str(flags[i]), fallback)
            else:
                out[key] = fallback
        return out

    def manifest(
        self,
        title: str,
        items: list[tuple[str, str]],
        actions: list[str],
        footer: str,
        note: str = "",
        initial: int = 0,
    ) -> int:
        value = self._next()
        if isinstance(value, bool):
            return len(items) if value else len(items) + len(actions) - 1
        if isinstance(value, int):
            return value
        return len(items) + ("go", "save", "back").index(str(value))

    def confirm(self, title: str, lines: list[str], footer: str, ok_label: str) -> bool:
        return bool(self._next())

    def confirm_with_save(
        self,
        title: str,
        lines: list[str],
        footer: str,
        ok_label: str,
        save_label: str,
    ) -> str:
        value = self._next()
        if value == "save":
            return "save"
        return "go" if value else "back"

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


def save_profile_interactive(
    ui: WizardUI, lang: str, profiles_dir: Path | None, spec: PromptSpec
) -> str:
    """Ask for a name and write the profile; returns a status line for the confirm page."""
    target_dir = profiles_dir if profiles_dir is not None else default_profiles_dir()
    try:
        name = ui.text_input(
            t(lang, "profile_save_title"),
            t(lang, "profile_save_hint"),
            t(lang, "footer_text"),
        )
    except GoBack:
        return ""
    try:
        path = profile_file(target_dir, name)
    except ProfileError as exc:
        return f"{t(lang, 'profile_save_failed')}: {exc}"
    if path.exists() and not ui.confirm(
        t(lang, "profile_overwrite_title"),
        [name],
        t(lang, "footer"),
        t(lang, "profile_overwrite_yes"),
    ):
        return t(lang, "profile_save_cancelled")
    try:
        save_profile(target_dir, name, spec)
    except (OSError, ProfileError) as exc:
        return f"{t(lang, 'profile_save_failed')}: {exc}"
    return f"{t(lang, 'profile_saved')}: {name}"


def _page_character(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    while True:
        text = ui.text_input(
            t(lang, "character_title"),
            t(lang, "character_hint"),
            t(lang, "footer_text"),
            initial=spec.character,
        ).strip()
        if text:
            spec.character = text
            return


def _page_mode(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    idx = ui.radio(
        t(lang, "mode_title"),
        [t(lang, "mode_sfw"), t(lang, "mode_nsfw"), t(lang, "mode_sex")],
        t(lang, "footer"),
        selected=MODES.index(spec.mode) if spec.mode in MODES else 0,
    )
    mode = MODES[idx]
    if mode == spec.mode:
        return
    spec.mode = mode
    spec.exposure = default_exposure(mode)
    spec.chest = CHEST_DEFAULT[mode]
    spec.belly = BELLY_DEFAULT[mode]
    spec.face = default_face()


def _page_exposure(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    while True:
        _, flags = ui.checklist(
            t(lang, "exposure_title"),
            [
                t(lang, "exp_covered"),
                t(lang, "exp_casual"),
                t(lang, "exp_revealing"),
                t(lang, "exp_open"),
                t(lang, "exp_nude"),
            ],
            t(lang, "footer"),
            preselected=[level in spec.exposure for level in EXPOSURE_LEVELS],
            allow_any=False,
        )
        chosen = [level for level, on in zip(EXPOSURE_LEVELS, flags) if on]
        if not chosen:
            chosen = list(default_exposure(spec.mode))
        spec.exposure = tuple(chosen)
        if set(spec.exposure) == {"nude"}:
            spec.clothing_any = True
            spec.clothing_keys = frozenset()
        if not needs_sfw_exposure_warning(spec.mode, spec.exposure):
            return
        if ui.warn_continue(
            t(lang, "sfw_warn"),
            t(lang, "warn_continue"),
            t(lang, "warn_back"),
            t(lang, "footer"),
        ):
            return


def _page_clothing(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    items = _allowed_clothing(matrix, spec.exposure)
    labels = [option_label(item, lang) for item in items]
    any_on, flags = ui.checklist(
        t(lang, "clothing_title"),
        labels,
        t(lang, "footer"),
        preselected=[item.key in spec.clothing_keys for item in items],
        allow_any=True,
        any_label=t(lang, "any"),
        any_on=spec.clothing_any,
    )
    spec.clothing_any = any_on or not any(flags)
    spec.clothing_keys = frozenset(items[i].key for i, on in enumerate(flags) if on)


def _page_chest(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    while True:
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
            selected=CHEST_LEVELS.index(spec.chest) if spec.chest in CHEST_LEVELS else 0,
        )
        spec.chest = CHEST_LEVELS[idx]
        if spec.mode != "sfw" or spec.chest not in ("breasts_out", "nipples"):
            return
        if ui.warn_continue(
            t(lang, "sfw_body_warn"),
            t(lang, "warn_continue"),
            t(lang, "warn_back"),
            t(lang, "footer"),
        ):
            return


def _page_belly(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    idx = ui.radio(
        t(lang, "belly_title"),
        [
            t(lang, "belly_covered"),
            t(lang, "belly_midriff"),
            t(lang, "belly_navel"),
            t(lang, "belly_auto"),
        ],
        t(lang, "footer"),
        selected=BELLY_LEVELS.index(spec.belly) if spec.belly in BELLY_LEVELS else 0,
    )
    spec.belly = BELLY_LEVELS[idx]


def _page_face(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    while True:
        view = [
            (
                group.zh if lang == "chinese" else group.en,
                [t(lang, "any")] + [face_label(item, lang) for item in group.options],
            )
            for group in FACE_GROUPS
        ]
        preselected = [
            face_selection(group, face_choice_of(spec.face, group)) for group in FACE_GROUPS
        ]
        chosen = ui.face_groups(
            t(lang, "face_title"), view, t(lang, "footer_face"), preselected
        )
        spec.face = {
            group.id: face_value(group, chosen[i][0], chosen[i][1])
            for i, group in enumerate(FACE_GROUPS)
        }
        if not needs_sfw_face_warning(spec.mode, selected_face_tags(spec.face)):
            return
        if ui.warn_continue(
            t(lang, "sfw_face_warn"),
            t(lang, "warn_continue"),
            t(lang, "warn_back"),
            t(lang, "footer"),
        ):
            return


def _page_scene(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    labels = [option_label(item, lang) for item in matrix.scenes]
    any_on, flags = ui.checklist(
        t(lang, "scene_title"),
        labels,
        t(lang, "footer"),
        preselected=[item.key in spec.scene_keys for item in matrix.scenes],
        allow_any=True,
        any_label=t(lang, "any"),
        any_on=spec.scene_any,
    )
    spec.scene_any = any_on or not any(flags)
    spec.scene_keys = frozenset(matrix.scenes[i].key for i, on in enumerate(flags) if on)


def _page_family(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
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
        preselected=[fam in spec.families for fam in POSE_FAMILY_IDS],
        allow_any=True,
        any_label=t(lang, "any"),
        any_on=spec.family_any,
    )
    spec.family_any = any_on or not any(flags)
    spec.families = frozenset(fam for fam, on in zip(POSE_FAMILY_IDS, flags) if on)


def _page_ratio(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    spec.vaginal_ratio = ui.ratio(
        t(lang, "ratio_title"),
        t(lang, "ratio_hint"),
        t(lang, "footer_text"),
        spec.vaginal_ratio,
    )


def _page_stages(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    spec.stage_weights = ui.weights(
        t(lang, "stage_title"),
        [(stage, t(lang, f"stage_{stage}")) for stage in SEX_STAGES],
        t(lang, "stage_hint"),
        t(lang, "footer_text"),
        dict(spec.stage_weights),
    )


def _page_pose(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    pool = pose_pool(matrix, spec.mode)
    if spec.mode == "sex":
        pool = family_pool(pool, spec)
    labels = [option_label(item, lang) for item in pool]
    any_on, flags = ui.checklist(
        t(lang, "pose_title"),
        labels,
        t(lang, "footer"),
        preselected=[item.key in spec.pose_keys for item in pool],
        allow_any=True,
        any_label=t(lang, "any"),
        any_on=spec.pose_any,
    )
    spec.pose_any = any_on or not any(flags)
    spec.pose_keys = frozenset(pool[i].key for i, on in enumerate(flags) if on)


def _page_count(spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI) -> None:
    spec.count = ui.number(
        t(lang, "count_title"),
        t(lang, "footer_text"),
        spec.count,
        COUNT_MIN,
        COUNT_MAX,
    )


def _page_confirm(
    spec: PromptSpec, lang: str, ui: WizardUI, profiles_dir: Path | None
) -> PromptSpec:
    lines = [
        f"character: {spec.character}",
        f"mode: {spec.mode}",
        f"exposure: {', '.join(spec.exposure)}",
        f"chest: {spec.chest}",
        f"belly: {spec.belly}",
        f"face: {face_summary(spec.face, lang)}",
        f"count: {spec.count}",
    ]
    if spec.mode == "sex":
        fams = "any" if spec.family_any else ", ".join(sorted(spec.families))
        lines.append(f"families: {fams}")
        lines.append(
            f"vaginal_ratio: {spec.vaginal_ratio:g} (anal {1 - spec.vaginal_ratio:g})"
        )
        stages = ", ".join(
            f"{stage} {spec.stage_weights.get(stage, 0.0):g}" for stage in SEX_STAGES
        )
        lines.append(f"stages: {stages}")
    note = ""
    while True:
        choice = ui.confirm_with_save(
            t(lang, "confirm_title"),
            lines + ([note] if note else []),
            t(lang, "footer"),
            t(lang, "confirm_go"),
            t(lang, "confirm_save"),
        )
        if choice == "go":
            return spec
        if choice == "back":
            raise GoBack
        note = save_profile_interactive(ui, lang, profiles_dir, spec)


def _always(spec: PromptSpec) -> bool:
    return True


def _sex_only(spec: PromptSpec) -> bool:
    return spec.mode == "sex"


def _nude_only(spec: PromptSpec) -> bool:
    return set(spec.exposure) == {"nude"}


def _not_nude(spec: PromptSpec) -> bool:
    return not _nude_only(spec)


@dataclass(frozen=True)
class WizardPage:
    key: str
    run: Callable[[PromptSpec, Matrix, str, WizardUI], None]
    applies: Callable[[PromptSpec], bool]


WIZARD_PAGES: tuple[WizardPage, ...] = (
    WizardPage("character", _page_character, _always),
    WizardPage("mode", _page_mode, _always),
    WizardPage("exposure", _page_exposure, _always),
    WizardPage("clothing", _page_clothing, _not_nude),
    WizardPage("chest", _page_chest, _always),
    WizardPage("belly", _page_belly, _always),
    WizardPage("face", _page_face, _always),
    WizardPage("scene", _page_scene, _always),
    WizardPage("family", _page_family, _sex_only),
    WizardPage("ratio", _page_ratio, _sex_only),
    WizardPage("stages", _page_stages, _sex_only),
    WizardPage("pose", _page_pose, _always),
    WizardPage("count", _page_count, _always),
)

PAGE_BY_KEY = {page.key: page for page in WIZARD_PAGES}


def default_spec() -> PromptSpec:
    """What a fresh wizard starts from; PromptSpec's own chest/belly defaults are per-mode."""
    return PromptSpec(
        character="",
        mode="sfw",
        exposure=default_exposure("sfw"),
        chest=CHEST_DEFAULT["sfw"],
        belly=BELLY_DEFAULT["sfw"],
    )


def _previous_page(index: int, spec: PromptSpec) -> int:
    """Where a GoBack lands: the nearest earlier page that still applies to this spec."""
    for candidate in range(index - 1, -1, -1):
        if WIZARD_PAGES[candidate].applies(spec):
            return candidate
    return 0


def run_wizard(
    matrix: Matrix, lang: str, ui: WizardUI, profiles_dir: Path | None = None
) -> PromptSpec:
    spec = default_spec()
    index = 0
    while True:
        if index >= len(WIZARD_PAGES):
            try:
                return _page_confirm(spec, lang, ui, profiles_dir)
            except GoBack:
                index = len(WIZARD_PAGES) - 1
                continue
        page = WIZARD_PAGES[index]
        if not page.applies(spec):
            index += 1
            continue
        try:
            page.run(spec, matrix, lang, ui)
        except GoBack:
            index = _previous_page(index, spec)
            continue
        index += 1


def manifest_face_value(spec: PromptSpec, lang: str) -> str:
    """Face groups that write something, e.g. '总表情=不限, 视线=looking at viewer'."""
    parts: list[str] = []
    for group in FACE_GROUPS:
        value = face_choice(spec, group)
        if value == FACE_NONE:
            continue
        name = group.zh if lang == "chinese" else group.en
        parts.append(f"{name}={face_value_label(value, lang)}")
    return ", ".join(parts) or t(lang, "face_skip")


def manifest_stage_value(spec: PromptSpec, lang: str) -> str:
    parts = [
        f"{stage} {float(spec.stage_weights.get(stage, 0.0)):g}"
        for stage in SEX_STAGES
        if stage_weight_value(spec, stage) > 0
    ]
    return ", ".join(parts) or t(lang, "manifest_stage_fallback")


def manifest_items(spec: PromptSpec, lang: str) -> list[tuple[str, str, str]]:
    """(page key, label, current value) for every adjustable item, in wizard order."""

    def label(key: str) -> str:
        return t(lang, f"item_{key}")

    def picked(any_flag: bool, count: int) -> str:
        return t(lang, "any") if any_flag else t(lang, "manifest_selected").format(n=count)

    items: list[tuple[str, str, str]] = [
        ("character", label("character"), spec.character),
        ("mode", label("mode"), spec.mode),
        ("exposure", label("exposure"), ", ".join(spec.exposure)),
    ]
    if not _nude_only(spec):
        items.append(
            ("clothing", label("clothing"), picked(spec.clothing_any, len(spec.clothing_keys)))
        )
    items.extend(
        [
            ("chest", label("chest"), spec.chest),
            ("belly", label("belly"), spec.belly),
            ("face", label("face"), manifest_face_value(spec, lang)),
            ("scene", label("scene"), picked(spec.scene_any, len(spec.scene_keys))),
        ]
    )
    if spec.mode == "sex":
        items.extend(
            [
                ("family", label("family"), picked(spec.family_any, len(spec.families))),
                ("ratio", label("ratio"), f"{spec.vaginal_ratio:g}"),
                ("stages", label("stages"), manifest_stage_value(spec, lang)),
            ]
        )
    items.extend(
        [
            ("pose", label("pose"), picked(spec.pose_any, len(spec.pose_keys))),
            ("count", label("count"), str(spec.count)),
        ]
    )
    return items


def run_manifest(
    spec: PromptSpec, matrix: Matrix, lang: str, ui: WizardUI, profiles_dir: Path | None
) -> tuple[PromptSpec, str]:
    """Review a profile's whole configuration, edit single items, then generate or save."""
    actions = ("go", "save", "back")
    action_labels = [
        t(lang, "manifest_go"),
        t(lang, "manifest_save"),
        t(lang, "manifest_back"),
    ]
    note = ""
    focus = 0
    while True:
        items = manifest_items(spec, lang)
        try:
            index = ui.manifest(
                t(lang, "manifest_title"),
                [(label, value) for _, label, value in items],
                action_labels,
                t(lang, "footer_manifest"),
                note,
                focus,
            )
        except GoBack:
            return spec, "back"
        focus = index
        if index < len(items):
            note = ""
            try:
                PAGE_BY_KEY[items[index][0]].run(spec, matrix, lang, ui)
            except GoBack:
                pass
            continue
        action = actions[index - len(items)]
        if action == "save":
            note = save_profile_interactive(ui, lang, profiles_dir, spec)
            continue
        return spec, action


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


def _char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _clip(text: str, width: int) -> str:
    """Truncate to a terminal width so one list row stays one row (CJK is two columns)."""
    if width <= 0:
        return ""
    if sum(_char_width(ch) for ch in text) <= width:
        return text
    out: list[str] = []
    used = 0
    for ch in text:
        step = _char_width(ch)
        if used + step > width - 1:
            break
        out.append(ch)
        used += step
    return "".join(out) + "…"


class CursesUI(WizardUI):
    def __init__(self, stdscr):
        self.stdscr = stdscr

    def _read_key(self):
        try:
            return self.stdscr.get_wch()
        except curses.error:
            return -1

    def text_input(self, title: str, hint: str, footer: str, initial: str = "") -> str:
        buf: list[str] = list(initial)
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
            if _key_is(key, "b", "B") and "".join(buf) in ("", initial):
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
            view_h = max(1, max_y - y - 2)
            start = 0
            if cursor >= view_h:
                start = cursor - view_h + 1
            visible = range(start, min(len(options), start + view_h))
            for row_i, i in enumerate(visible):
                mark = "(•)" if i == cursor else "( )"
                attr = curses.A_REVERSE if i == cursor else 0
                _add_wrapped(self.stdscr, y + row_i, 1, f"{mark} {options[i]}", max_x - 2, attr)
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

    def face_groups(
        self,
        title: str,
        groups: list[tuple[str, list[str]]],
        footer: str,
        preselected: list[tuple[bool, list[bool]]],
    ) -> list[tuple[bool, list[bool]]]:
        any_on = [bool(entry[0]) for entry in preselected]
        flags = [
            [bool(flag) for flag in entry[1]] + [False] * (len(labels) - 1 - len(entry[1]))
            for (_, labels), entry in zip(groups, preselected)
        ]
        rows: list[tuple[str, int, int, str]] = []
        for group_i, (header, labels) in enumerate(groups):
            rows.append(("head", group_i, -1, header))
            for option_i, label in enumerate(labels):
                rows.append(("opt", group_i, option_i, label))
        option_rows = [i for i, row in enumerate(rows) if row[0] == "opt"]
        group_start = [
            next(k for k, row_i in enumerate(option_rows) if rows[row_i][1] == group_i)
            for group_i in range(len(groups))
        ]
        cursor = 0
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            view_h = max(1, max_y - y - 2)
            focus = option_rows[cursor]
            start = focus - view_h + 1 if focus >= view_h else 0
            start = min(start, max(0, len(rows) - view_h))
            for row_i, (kind, group_i, option_i, text) in enumerate(
                rows[start : start + view_h]
            ):
                if kind == "head":
                    _add_wrapped(self.stdscr, y + row_i, 1, text, max_x - 2, curses.A_BOLD)
                    continue
                ticked = any_on[group_i] if option_i == 0 else flags[group_i][option_i - 1]
                mark = "[x]" if ticked else "[ ]"
                attr = curses.A_REVERSE if start + row_i == focus else 0
                _add_wrapped(self.stdscr, y + row_i, 1, f"{mark} {text}", max_x - 2, attr)
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B") or _key_backspace(key):
                raise GoBack
            if _key_up(key):
                cursor = (cursor - 1) % len(option_rows)
            elif _key_down(key):
                cursor = (cursor + 1) % len(option_rows)
            elif _key_is(key, "n", "N"):
                cursor = group_start[(rows[focus][1] + 1) % len(groups)]
            elif _key_is(key, "p", "P"):
                cursor = group_start[(rows[focus][1] - 1) % len(groups)]
            elif _key_is(key, " "):
                group_i = rows[focus][1]
                option_i = rows[focus][2]
                if option_i == 0:
                    any_on[group_i] = not any_on[group_i]
                    if any_on[group_i]:
                        flags[group_i] = [False] * len(flags[group_i])
                else:
                    flags[group_i][option_i - 1] = not flags[group_i][option_i - 1]
                    if flags[group_i][option_i - 1]:
                        any_on[group_i] = False
            elif _key_enter(key):
                return [
                    (any_on[group_i], flags[group_i]) for group_i in range(len(groups))
                ]

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

    def weights(
        self,
        title: str,
        items: list[tuple[str, str]],
        hint: str,
        footer: str,
        defaults: dict[str, float],
    ) -> dict[str, float]:
        keys = [item[0] for item in items]
        labels = [item[1] for item in items]
        buffers = [list(f"{float(defaults.get(key, 0.0)):g}") for key in keys]
        cursor = 0
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            row = _add_wrapped(self.stdscr, y, 1, hint, max_x - 2, curses.A_DIM) + 1
            for i, label in enumerate(labels):
                shown = "".join(buffers[i]) or "0"
                mark = "█" if i == cursor else ""
                attr = curses.A_REVERSE if i == cursor else 0
                _add_wrapped(
                    self.stdscr,
                    row + i,
                    1,
                    f"{shown}{mark}  {label}",
                    max_x - 2,
                    attr,
                )
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B"):
                raise GoBack
            if _key_up(key):
                cursor = (cursor - 1) % len(items)
            elif _key_down(key):
                cursor = (cursor + 1) % len(items)
            elif _key_backspace(key):
                if buffers[cursor]:
                    buffers[cursor].pop()
            elif _key_enter(key):
                out: dict[str, float] = {}
                ok = True
                for i, stage in enumerate(keys):
                    raw = "".join(buffers[i]).strip()
                    fallback = float(defaults.get(stage, 0.0))
                    try:
                        out[stage] = parse_stage_weight(raw, fallback)
                    except ValueError:
                        ok = False
                        break
                if ok:
                    return out
            elif isinstance(key, str) and (key.isdigit() or key == "."):
                if key == "." and "." in buffers[cursor]:
                    continue
                buffers[cursor].append(key)
            elif isinstance(key, int) and ord("0") <= key <= ord("9"):
                buffers[cursor].append(chr(key))

    def manifest(
        self,
        title: str,
        items: list[tuple[str, str]],
        actions: list[str],
        footer: str,
        note: str = "",
        initial: int = 0,
    ) -> int:
        pad = max((len(label) for label, _ in items), default=0)
        rows: list[tuple[bool, str, int]] = [
            (True, f"{label.ljust(pad)}  {value}", index)
            for index, (label, value) in enumerate(items)
        ]
        rows.append((False, "", -1))  # separator; the actions sit under it
        rows.extend(
            (True, action, len(items) + action_i)
            for action_i, action in enumerate(actions)
        )
        selectable = [i for i, row in enumerate(rows) if row[0]]
        cursor = initial if 0 <= initial < len(selectable) else 0
        while True:
            max_y, max_x, y = _curses_page_header(self.stdscr, title, footer)
            view_h = max(1, max_y - y - 2)
            width = max(8, max_x - 2)
            focus = selectable[cursor]
            start = focus - view_h + 1 if focus >= view_h else 0
            start = min(start, max(0, len(rows) - view_h))
            for row_i, (is_row, text, _) in enumerate(rows[start : start + view_h]):
                attr = curses.A_DIM
                if is_row:
                    attr = curses.A_REVERSE if start + row_i == focus else 0
                try:
                    self.stdscr.addstr(
                        y + row_i, 1, _clip(text if is_row else "─" * width, width), attr
                    )
                except curses.error:
                    pass
            if note:
                try:
                    self.stdscr.addstr(max_y - 2, 1, _clip(note, width), curses.A_DIM)
                except curses.error:
                    pass
            self.stdscr.refresh()
            key = self._read_key()
            if _key_is(key, "q", "Q"):
                raise WizardCancelled
            if _key_is(key, "b", "B") or _key_backspace(key):
                raise GoBack
            if _key_up(key):
                cursor = (cursor - 1) % len(selectable)
            elif _key_down(key):
                cursor = (cursor + 1) % len(selectable)
            elif _key_enter(key) or _key_is(key, " "):
                return rows[focus][2]

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

    def confirm_with_save(
        self,
        title: str,
        lines: list[str],
        footer: str,
        ok_label: str,
        save_label: str,
    ) -> str:
        results = ("go", "save", "back")
        choices = (ok_label, save_label, "back")
        cursor = 0
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
                return "back"
            if _key_up(key):
                cursor = (cursor - 1) % len(choices)
            elif _key_down(key):
                cursor = (cursor + 1) % len(choices)
            elif _key_enter(key) or _key_is(key, " "):
                return results[cursor]

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


def _curses_main(stdscr, matrix: Matrix, lang: str, profiles_dir: Path) -> PromptSpec:
    curses.curs_set(0)
    stdscr.keypad(True)
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    ui = CursesUI(stdscr)
    while True:
        profiles = list_profiles(profiles_dir)
        if not profiles:
            return run_wizard(matrix, lang, ui, profiles_dir)
        labels = [name for name, _ in profiles] + [t(lang, "profile_new")]
        try:
            idx = ui.radio(t(lang, "profile_title"), labels, t(lang, "footer"), selected=0)
        except GoBack:
            idx = len(profiles)
        if idx >= len(profiles):
            return run_wizard(matrix, lang, ui, profiles_dir)
        spec, action = run_manifest(
            load_profile(profiles[idx][1]), matrix, lang, ui, profiles_dir
        )
        if action == "go":
            return spec


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
    parser.add_argument(
        "--profile",
        default=None,
        help="Skip the wizard: generate from this saved profile by name.",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="Directory holding saved profiles (default: repo-root prompt_profiles).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        matrix = load_matrix(args.matrix)
    except (OSError, MatrixError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    profiles_dir = args.profiles_dir if args.profiles_dir is not None else default_profiles_dir()
    if args.profile:
        try:
            spec = load_profile(resolve_profile(profiles_dir, args.profile))
        except ProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print(t(args.language, "need_tty"), file=sys.stderr)
            return 2
        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            pass
        try:
            spec = curses.wrapper(
                lambda stdscr: _curses_main(stdscr, matrix, args.language, profiles_dir)
            )
        except WizardCancelled:
            return 1
        except ProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        prompts = generate(spec, matrix)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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

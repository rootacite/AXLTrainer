#!/usr/bin/env python3
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from tools.gen_prompts import (
    FACE_ANY,
    FACE_GROUPS,
    FACE_GROUPS_BY_ID,
    FACE_NONE,
    FACE_TAG_GROUP,
    SEX_STAGES,
    STAGES_OBJECT,
    Entry,
    MatrixError,
    ProfileError,
    PromptSpec,
    ScriptedUI,
    anatomy_tags,
    assemble,
    bind_stage_to_poses,
    chest_view,
    default_face,
    default_spec,
    default_stage_weights,
    face_selection,
    face_tags,
    face_tags_of,
    face_value,
    generate,
    group_pool,
    group_tags,
    is_forbidden_tag,
    list_profiles,
    load_matrix,
    load_profile,
    main,
    manifest_face_value,
    manifest_items,
    manifest_stage_value,
    needs_sfw_exposure_warning,
    needs_sfw_face_warning,
    parse_matrix,
    parse_vaginal_ratio,
    pick_stage,
    pose_allows_hole,
    pose_families,
    pose_locus,
    pose_pool,
    pose_weight,
    resolve_profile,
    run_manifest,
    run_wizard,
    needs_sfw_exposure_warning,
    needs_sfw_face_warning,
    parse_matrix,
    parse_vaginal_ratio,
    pick_stage,
    pose_families,
    pose_locus,
    pose_pool,
    pose_weight,
    resolve_profile,
    run_wizard,
    safe_profile_name,
    save_profile,
    scene_compatible,
    selected_face_tags,
    split_tags,
    stage_tags,
    t,
    torso_tags,
)

REPO = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO / "input_matrix.txt"

MINI = """\
POSES:
mating press, legs up, folded, knees to chest : both
# mating press comment
doggystyle, sex from behind, top-down bottom-up : both
# doggy
full nelson : anal only
# nelson
standing missionary, standing sex, against wall : both
# stand sex
cowgirl position, girl on top, sitting, straddling, looking at viewer : both
# cowgirl
CLOTHING:
[covered]
serafuku, white thighhighs (open clothes)
# sailor
wedding dress, veil, white gloves
# wedding
[casual]
white sundress, white thighhighs (open clothes)
# sundress
[revealing]
bikini, side-tie bikini bottom (open clothes)
# bikini
SCENE:
bedroom, indoors, bed
# bedroom
bedroom, indoors, on bed, sheets
# on bed
hallway, indoors, school
# hallway
beach, outdoors, ocean, sand
# beach
cafe, indoors, window
# cafe
classroom, indoors, desk
# desk
SUFFIX:
soft lighting, warm light
# warm
SFW_POSES:
sitting, looking at viewer, smile
# sit smile
sitting, desk, chin rest
# desk sit
walking, looking back, smile
# walk
lying, on back, looking at viewer
# lie
window, sitting, looking outside
# window sit
standing, looking at viewer, arms behind back
# stand
"""


def _spec(**kwargs) -> PromptSpec:
    base = dict(
        character="(sena_character:1.1), 1girl",
        mode="sfw",
        exposure=("covered", "casual"),
        count=8,
    )
    base.update(kwargs)
    return PromptSpec(**base)


def _face(**overrides) -> dict[str, object]:
    """Group values from the defaults; a bare tag means a one-tag candidate list."""
    face: dict[str, object] = dict(default_face())
    for key, value in overrides.items():
        if isinstance(value, str) and value not in (FACE_ANY, FACE_NONE):
            value = (value,)
        face[key] = value
    return face


def _groups_hit(line: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tag in split_tags(line):
        group = FACE_TAG_GROUP.get(tag)
        if group:
            counts[group] = counts.get(group, 0) + 1
    return counts


def _find(entries, needle: str) -> Entry:
    for entry in entries:
        if needle in entry.blob:
            return entry
    raise AssertionError(f"no entry containing {needle!r}")


class ParseTest(unittest.TestCase):
    def test_groups_and_comments_on_real_matrix(self):
        matrix = load_matrix(MATRIX_PATH)
        self.assertTrue(matrix.poses)
        self.assertTrue(matrix.sfw_poses)
        groups = {item.group for item in matrix.clothing}
        self.assertEqual(groups, {"covered", "casual", "revealing"})
        sailor = _find(matrix.clothing, "serafuku, white thighhighs")
        self.assertTrue(sailor.open_clothes)
        self.assertEqual(sailor.group, "covered")
        self.assertIn("水手服", sailor.comment)
        wedding = _find(matrix.clothing, "wedding dress, veil, white gloves")
        self.assertFalse(wedding.open_clothes)
        press = _find(matrix.poses, "mating press")
        self.assertEqual(press.channel, "both")
        nelson = _find(matrix.poses, "full nelson")
        self.assertEqual(nelson.channel, "anal")
        oral = _find(matrix.poses, "oral, fellatio")
        self.assertEqual(oral.channel, "none")
        paizuri = _find(matrix.poses, "paizuri")
        self.assertEqual(paizuri.channel, "none")
        nursing = _find(matrix.poses, "nursing handjob")
        self.assertEqual(nursing.channel, "none")

    def test_pose_without_channel_raises(self):
        raw = MINI.replace(
            "full nelson : anal only",
            "full nelson",
        )
        with self.assertRaises(MatrixError):
            parse_matrix(raw)

    def test_clothing_without_group_raises(self):
        raw = MINI.replace("[covered]\n", "")
        with self.assertRaises(MatrixError):
            parse_matrix(raw)


class ModePoolTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)

    def test_sex_does_not_draw_sfw_poses(self):
        spec = _spec(mode="sex", exposure=("open",), count=20)
        prompts = generate(spec, self.matrix, seed=1)
        for line in prompts:
            tags = set(split_tags(line))
            self.assertNotIn("chin rest", tags)
            self.assertNotIn("peace sign", tags)
            self.assertIn("sex", tags)
            self.assertIn("penis", tags)

    def test_sfw_does_not_draw_sex_poses(self):
        spec = _spec(mode="sfw", count=20)
        prompts = generate(spec, self.matrix, seed=2)
        blob = "\n".join(prompts)
        self.assertNotIn("doggystyle", blob)
        self.assertNotIn("mating press", blob)
        self.assertNotIn("penis", blob)
        self.assertNotIn("pussy", blob)
        self.assertNotIn("anus", blob)
        self.assertNotIn("sex", blob)

    def test_nsfw_same_pose_pool_as_sfw(self):
        spec = _spec(mode="nsfw", exposure=("nude",), count=12)
        prompts = generate(spec, self.matrix, seed=3)
        blob = "\n".join(prompts)
        self.assertNotIn("doggystyle", blob)
        self.assertIn("nude", blob)
        self.assertNotIn("penis", blob)


class PrefixAndForbiddenTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)

    def test_prefix_is_first_and_untouched(self):
        spec = _spec(count=5)
        for line in generate(spec, self.matrix, seed=4):
            self.assertTrue(line.startswith("(sena_character:1.1), 1girl"))

    def test_no_rating_or_quality_tags(self):
        spec = _spec(mode="sex", exposure=("open",), count=15)
        for line in generate(spec, self.matrix, seed=5):
            tags = {tag.lower() for tag in split_tags(line)}
            self.assertNotIn("nsfw", tags)
            self.assertNotIn("sfw", tags)
            self.assertNotIn("explicit", tags)
            self.assertNotIn("masterpiece", tags)
            self.assertNotIn("best quality", tags)
            self.assertNotIn("newest", tags)

    def test_forbidden_helper(self):
        self.assertTrue(is_forbidden_tag("nsfw"))
        self.assertTrue(is_forbidden_tag("masterpiece"))
        self.assertTrue(is_forbidden_tag("year 2024"))
        self.assertTrue(is_forbidden_tag("score_9"))
        self.assertFalse(is_forbidden_tag("1girl"))


class ChannelAndAnatomyTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)

    def test_anal_only_never_writes_vaginal(self):
        nelson = _find(self.matrix.poses, "full nelson")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({nelson.key}),
            count=8,
        )
        for line in generate(spec, self.matrix, seed=6):
            tags = set(split_tags(line))
            self.assertIn("anal", tags)
            self.assertNotIn("vaginal", tags)
            self.assertIn("penis", tags)
            self.assertIn("anus", tags)
            self.assertIn("ass", tags)

    def test_doggystyle_has_ass(self):
        dog = _find(self.matrix.poses, "doggystyle")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({dog.key}),
            vaginal_ratio=0.0,
            count=5,
        )
        for line in generate(spec, self.matrix, seed=7):
            tags = set(split_tags(line))
            self.assertIn("ass", tags)
            self.assertIn("penis", tags)

    def test_mating_press_vaginal_has_pussy(self):
        press = _find(self.matrix.poses, "mating press")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({press.key}),
            vaginal_ratio=1.0,
            count=5,
        )
        for line in generate(spec, self.matrix, seed=8):
            tags = set(split_tags(line))
            self.assertIn("pussy", tags)
            self.assertIn("penis", tags)
            self.assertNotIn("ass", tags)

    def test_standing_missionary_has_no_ass(self):
        pose = _find(self.matrix.poses, "standing missionary")
        tags = anatomy_tags(pose, "vaginal")
        self.assertIn("penis", tags)
        self.assertIn("pussy", tags)
        self.assertNotIn("ass", tags)
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({pose.key}),
            vaginal_ratio=1.0,
            count=6,
        )
        for line in generate(spec, self.matrix, seed=9):
            self.assertNotIn("ass", set(split_tags(line)))


class CompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)

    def test_lie_rejects_hallway(self):
        pose = _find(self.matrix.sfw_poses, "lying, on back")
        hallway = _find(self.matrix.scenes, "hallway")
        bed = _find(self.matrix.scenes, "on bed")
        self.assertFalse(scene_compatible(pose, hallway))
        self.assertTrue(scene_compatible(pose, bed))
        self.assertEqual(pose_locus(pose), "lie")

    def test_walk_rejects_on_bed(self):
        pose = _find(self.matrix.sfw_poses, "walking")
        on_bed = _find(self.matrix.scenes, "on bed")
        beach = _find(self.matrix.scenes, "beach")
        self.assertFalse(scene_compatible(pose, on_bed))
        self.assertTrue(scene_compatible(pose, beach))

    def test_window_rejects_beach(self):
        pose = _find(self.matrix.sfw_poses, "looking outside")
        beach = _find(self.matrix.scenes, "beach")
        cafe = _find(self.matrix.scenes, "cafe")
        self.assertFalse(scene_compatible(pose, beach))
        self.assertTrue(scene_compatible(pose, cafe))

    def test_generated_walk_never_on_bed(self):
        pose = _find(self.matrix.sfw_poses, "walking")
        spec = _spec(pose_any=False, pose_keys=frozenset({pose.key}), count=12)
        for line in generate(spec, self.matrix, seed=10):
            self.assertNotIn("on bed", line)


class WarningAndSeedTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)

    def test_sfw_high_exposure_warning_flag(self):
        self.assertTrue(needs_sfw_exposure_warning("sfw", ("nude",)))
        self.assertTrue(needs_sfw_exposure_warning("sfw", ("covered", "open")))
        self.assertFalse(needs_sfw_exposure_warning("sfw", ("covered", "casual")))
        self.assertFalse(needs_sfw_exposure_warning("nsfw", ("nude",)))

    def test_seed_reproducible(self):
        spec = _spec(mode="sex", exposure=("open",), count=6)
        a = generate(spec, self.matrix, seed=0)
        b = generate(spec, self.matrix, seed=0)
        self.assertEqual(a, b)

    def test_open_bucket_writes_open_clothes(self):
        spec = _spec(mode="sex", exposure=("open",), count=6)
        for line in generate(spec, self.matrix, seed=11):
            self.assertIn("open clothes", split_tags(line))

    def test_covered_bucket_skips_open_clothes(self):
        spec = _spec(mode="sfw", exposure=("covered",), count=8)
        for line in generate(spec, self.matrix, seed=12):
            self.assertNotIn("open clothes", split_tags(line))


class WizardScriptTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)

    def test_any_does_not_prefer_keys(self):
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                0,
                [True, True, False, False, False],
                "any",
                0,
                0,
                "default",
                "any",
                "any",
                4,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertEqual(spec.mode, "sfw")
        self.assertTrue(spec.clothing_any)
        self.assertTrue(spec.scene_any)
        self.assertTrue(spec.pose_any)
        self.assertEqual(spec.count, 4)
        self.assertEqual(spec.face, default_face())
        pose = self.matrix.sfw_poses[0]
        self.assertEqual(pose_weight(pose, spec), 1)

    def test_sfw_nude_warns_then_continues(self):
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                0,
                [False, False, False, False, True],
                True,
                0,
                0,
                "default",
                "any",
                "any",
                3,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "chinese", ui)
        self.assertEqual(spec.exposure, ("nude",))
        self.assertTrue(needs_sfw_exposure_warning(spec.mode, spec.exposure))

    def test_sex_family_and_ratio_pages(self):
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                2,
                [False, False, False, True, False],
                "any",
                4,
                3,
                "default",
                "any",
                [False, True, False, False, False, False],
                0.3,
                "default",
                "any",
                5,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertEqual(spec.mode, "sex")
        self.assertFalse(spec.family_any)
        self.assertEqual(spec.families, frozenset({"behind"}))
        self.assertAlmostEqual(spec.vaginal_ratio, 0.3)
        self.assertEqual(spec.stage_weights, default_stage_weights())
        self.assertEqual(spec.chest, "auto")
        self.assertEqual(spec.belly, "auto")
        self.assertEqual(spec.face, default_face())

    def test_sex_stage_weights_page(self):
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                2,
                [False, False, False, True, False],
                "any",
                4,
                3,
                "default",
                "any",
                "any",
                0.5,
                {"pose": 0.2, "during": 0.8},
                "any",
                5,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertAlmostEqual(spec.stage_weights["pose"], 0.2)
        self.assertAlmostEqual(spec.stage_weights["during"], 0.8)
        self.assertAlmostEqual(spec.stage_weights["before"], 0.0)

    def test_default_spec_matches_the_sfw_page_defaults(self):
        spec = default_spec()
        self.assertEqual(spec.mode, "sfw")
        self.assertEqual(spec.exposure, ("covered", "casual"))
        self.assertEqual(spec.chest, "covered")
        self.assertEqual(spec.belly, "covered")
        self.assertEqual(spec.face, default_face())
        self.assertEqual(spec.stage_weights, default_stage_weights())

    def test_back_from_pose_lands_on_scene_outside_sex(self):
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                0,
                [True, True, False, False, False],
                "any",
                0,
                0,
                "default",
                "any",  # scene
                "__back__",  # pose -> back
                "any",  # scene again
                "any",  # pose again
                4,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertEqual(spec.count, 4)

    def test_back_from_pose_lands_on_stages_in_sex_mode(self):
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                2,
                [False, False, False, True, False],
                "any",
                4,
                3,
                "default",
                "any",
                [False, True, False, False, False, False],
                0.3,
                "default",  # stages
                "__back__",  # pose -> back
                "default",  # stages again
                "any",  # pose again
                5,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertEqual(spec.count, 5)
        self.assertEqual(spec.families, frozenset({"behind"}))

    def test_back_from_chest_lands_on_exposure_when_nude(self):
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                0,
                [False, False, False, False, True],
                True,  # exposure warning
                "__back__",  # chest -> back
                [False, False, False, False, True],
                True,  # exposure warning again
                0,  # chest
                0,  # belly
                "default",
                "any",
                "any",
                3,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertEqual(spec.exposure, ("nude",))
        self.assertEqual(spec.count, 3)


class ManifestTest(unittest.TestCase):
    """A profile's whole configuration on one page, with per-row editing."""

    def setUp(self):
        self.matrix = parse_matrix(MINI)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _profile_spec(self, **kwargs) -> PromptSpec:
        base = dict(
            character="(sena_character:1.1), 1girl",
            mode="sex",
            exposure=("revealing", "open"),
            clothing_any=False,
            clothing_keys=frozenset({("bikini", "side-tie bikini bottom")}),
            scene_any=False,
            scene_keys=frozenset({("bedroom", "indoors", "bed")}),
            family_any=False,
            families=frozenset({"behind"}),
            vaginal_ratio=0.3,
            chest="nipples",
            belly="navel",
            face=_face(expression="smile", gaze=FACE_NONE, eyes=FACE_NONE),
            count=9,
            stage_weights={**default_stage_weights(), "during": 0.0, "pose": 1.0},
        )
        base.update(kwargs)
        return PromptSpec(**base)

    def _keys(self, spec: PromptSpec, lang: str = "english") -> list[str]:
        return [key for key, _, _ in manifest_items(spec, lang)]

    def test_sfw_manifest_drops_the_sex_only_rows(self):
        keys = self._keys(self._profile_spec(mode="sfw", exposure=("covered",)))
        self.assertIn("clothing", keys)
        for key in ("family", "ratio", "stages"):
            self.assertNotIn(key, keys)

    def test_sex_manifest_carries_the_sex_only_rows(self):
        keys = self._keys(self._profile_spec())
        self.assertEqual(keys[:3], ["character", "mode", "exposure"])
        for key in ("family", "ratio", "stages", "pose", "count"):
            self.assertIn(key, keys)

    def test_nude_manifest_drops_the_clothing_row(self):
        keys = self._keys(self._profile_spec(mode="nsfw", exposure=("nude",)))
        self.assertNotIn("clothing", keys)

    def test_rows_show_the_profile_values(self):
        values = {
            key: value for key, _, value in manifest_items(self._profile_spec(), "english")
        }
        self.assertEqual(values["mode"], "sex")
        self.assertEqual(values["count"], "9")
        self.assertEqual(values["ratio"], "0.3")
        self.assertEqual(values["clothing"], "1 selected")
        self.assertEqual(values["face"], "Expression=smile")
        self.assertEqual(values["stages"], "pose 1")

    def test_off_groups_are_hidden_and_all_zero_stages_fall_back(self):
        spec = self._profile_spec(
            face={group.id: FACE_NONE for group in FACE_GROUPS},
            stage_weights={stage: 0.0 for stage in SEX_STAGES},
        )
        self.assertEqual(manifest_face_value(spec, "english"), "off")
        self.assertEqual(manifest_stage_value(spec, "english"), "all zero → during")

    def test_editing_one_row_keeps_everything_else(self):
        spec = self._profile_spec()
        index = self._keys(spec).index("count")
        ui = ScriptedUI([index, 7, "go"])
        out, action = run_manifest(spec, self.matrix, "english", ui, self.dir)
        self.assertEqual(action, "go")
        self.assertEqual(out.count, 7)
        self.assertEqual(out.mode, "sex")
        self.assertEqual(out.families, frozenset({"behind"}))
        self.assertAlmostEqual(out.vaginal_ratio, 0.3)
        self.assertEqual(out.chest, "nipples")
        self.assertEqual(len(out.clothing_keys), 1)
        self.assertEqual(out.face["expression"], ("smile",))
        self.assertAlmostEqual(out.stage_weights["pose"], 1.0)

    def test_leaving_an_item_page_keeps_the_old_value(self):
        spec = self._profile_spec()
        index = self._keys(spec).index("count")
        ui = ScriptedUI([index, "__back__", "go"])
        out, action = run_manifest(spec, self.matrix, "english", ui, self.dir)
        self.assertEqual(action, "go")
        self.assertEqual(out.count, 9)

    def test_save_action_writes_a_profile_then_generates(self):
        spec = self._profile_spec()
        ui = ScriptedUI(["save", "from list", True, "go"])
        out, action = run_manifest(spec, self.matrix, "english", ui, self.dir)
        self.assertEqual(action, "go")
        self.assertEqual(out, spec)
        self.assertEqual([name for name, _ in list_profiles(self.dir)], ["from list"])
        self.assertEqual(load_profile(self.dir / "from list.json"), spec)

    def test_back_action_leaves_the_manifest(self):
        ui = ScriptedUI(["back"])
        _, action = run_manifest(self._profile_spec(), self.matrix, "english", ui, self.dir)
        self.assertEqual(action, "back")

    def test_face_row_opens_with_the_current_selection(self):
        spec = self._profile_spec(face=_face(expression=("smile", "shy"), eyes=("open eyes",)))
        index = self._keys(spec).index("face")
        ui = ScriptedUI([index, "default", "go"])
        out, action = run_manifest(spec, self.matrix, "english", ui, self.dir)
        self.assertEqual(action, "go")
        self.assertEqual(out.face["expression"], ("smile", "shy"))
        self.assertEqual(out.face["eyes"], ("open eyes",))


class FamilyAndRatioTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)
        self.real = load_matrix(MATRIX_PATH)

    def test_parse_ratio(self):
        self.assertEqual(parse_vaginal_ratio(""), 0.5)
        self.assertEqual(parse_vaginal_ratio("0"), 0.0)
        self.assertEqual(parse_vaginal_ratio("1"), 1.0)
        self.assertEqual(parse_vaginal_ratio("0.7"), 0.7)
        self.assertEqual(parse_vaginal_ratio("70"), 0.7)

    def test_mini_family_map(self):
        self.assertEqual(pose_families(_find(self.matrix.poses, "mating press")), frozenset({"face"}))
        self.assertEqual(pose_families(_find(self.matrix.poses, "doggystyle")), frozenset({"behind"}))
        self.assertEqual(
            pose_families(_find(self.matrix.poses, "full nelson")),
            frozenset({"hold", "stand_behind"}),
        )
        self.assertEqual(
            pose_families(_find(self.matrix.poses, "standing missionary")),
            frozenset({"face"}),
        )
        self.assertEqual(
            pose_families(_find(self.matrix.poses, "cowgirl")),
            frozenset({"girl_on_top"}),
        )

    def test_real_matrix_families(self):
        expected = {
            "mating press": {"face"},
            "missionary, spread legs": {"face"},
            "missionary, on back": {"face"},
            "anvil position": {"face"},
            "leaning back": {"girl_on_top"},
            "looking at viewer : both": None,  # skip
            "squatting cowgirl": {"girl_on_top"},
            "reverse cowgirl": {"girl_on_top", "behind"},
            "top-down bottom-up": {"behind"},
            "all fours": {"behind"},
            "standing doggystyle": {"stand_behind"},
            "spooning": {"side_behind"},
            "prone bone": {"behind"},
            "reverse suspended": {"hold", "stand_behind"},
            "suspended congress, held up": {"hold", "face"},
            "full nelson": {"hold", "stand_behind"},
            "standing missionary": {"face"},
            "upright straddle": {"girl_on_top", "hold"},
            "seventh posture": {"side_behind"},
        }
        for needle, families in expected.items():
            if families is None:
                continue
            pose = _find(self.real.poses, needle)
            self.assertEqual(set(pose_families(pose)), families, needle)
        viewer = _find(self.real.poses, "straddling, looking at viewer")
        self.assertEqual(pose_families(viewer), frozenset({"girl_on_top"}))

    def test_camera_variants_keep_families(self):
        expected = {
            "mating press": {"face"},
            "anvil position": {"face"},
            "standing missionary": {"face"},
            "squatting cowgirl": {"girl_on_top"},
            "reverse cowgirl": {"girl_on_top", "behind"},
            "standing doggystyle": {"stand_behind"},
            "spooning": {"side_behind"},
            "prone bone": {"behind"},
            "full nelson": {"hold", "stand_behind"},
            "upright straddle": {"girl_on_top", "hold"},
            "seventh posture": {"side_behind"},
        }
        for pose in self.real.poses:
            blob = pose.blob
            if "reverse suspended" in blob:
                self.assertEqual(set(pose_families(pose)), {"hold", "stand_behind"}, blob)
                continue
            if "suspended congress" in blob:
                self.assertEqual(set(pose_families(pose)), {"hold", "face"}, blob)
                continue
            if "standing doggystyle" in blob:
                self.assertEqual(set(pose_families(pose)), {"stand_behind"}, blob)
                continue
            if "doggystyle" in blob:
                self.assertEqual(set(pose_families(pose)), {"behind"}, blob)
                continue
            if "reverse cowgirl" in blob:
                self.assertEqual(set(pose_families(pose)), {"girl_on_top", "behind"}, blob)
                continue
            if "squatting cowgirl" in blob:
                self.assertEqual(set(pose_families(pose)), {"girl_on_top"}, blob)
                continue
            if "cowgirl" in blob:
                self.assertEqual(set(pose_families(pose)), {"girl_on_top"}, blob)
                continue
            if "missionary" in blob and "standing missionary" not in blob:
                self.assertEqual(set(pose_families(pose)), {"face"}, blob)
                continue
            for needle, families in expected.items():
                if needle in blob:
                    self.assertEqual(set(pose_families(pose)), families, blob)
                    break

    def test_camera_variants_exist(self):
        cameras = {
            "from above",
            "from below",
            "from behind",
            "from front",
            "from side",
            "looking at viewer",
            "looking back",
        }
        tagged = [
            pose
            for pose in self.real.poses
            if cameras.intersection(pose.tags)
        ]
        self.assertGreaterEqual(len(tagged), 30)
        self.assertTrue(any("from above" in p.tags for p in self.real.poses))
        self.assertTrue(any("from behind" in p.tags for p in self.real.poses))
        self.assertTrue(any("looking back" in p.tags for p in self.real.poses))
        self.assertTrue(
            any(
                "looking back" in p.tags and "looking at viewer" in p.tags
                for p in self.real.poses
            )
        )

    def test_camera_variants_keep_channel(self):
        for pose in self.real.poses:
            blob = pose.blob
            if "reverse cowgirl" in blob:
                self.assertEqual(pose.channel, "vaginal", blob)
            if "full nelson" in blob:
                self.assertEqual(pose.channel, "anal", blob)
            if "squatting cowgirl" in blob:
                self.assertEqual(pose.channel, "vaginal", blob)
            if "leaning back" in blob:
                self.assertEqual(pose.channel, "anal", blob)
            if "oral" in pose.tags or "paizuri" in pose.tags or "nursing handjob" in pose.tags:
                self.assertEqual(pose.channel, "none", blob)

    def test_girl_on_top_pool_excludes_doggystyle(self):
        spec = _spec(
            mode="sex",
            exposure=("open",),
            family_any=False,
            families=frozenset({"girl_on_top"}),
            count=12,
        )
        blob = "\n".join(generate(spec, self.matrix, seed=21))
        self.assertIn("cowgirl", blob)
        self.assertNotIn("doggystyle", blob)
        self.assertNotIn("mating press", blob)

    def test_ratio_one_is_always_vaginal_on_both(self):
        dog = _find(self.matrix.poses, "doggystyle")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({dog.key}),
            vaginal_ratio=1.0,
            count=10,
        )
        for line in generate(spec, self.matrix, seed=22):
            tags = set(split_tags(line))
            self.assertIn("vaginal", tags)
            self.assertNotIn("anal", tags)

    def test_ratio_zero_is_always_anal_on_both(self):
        dog = _find(self.matrix.poses, "doggystyle")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({dog.key}),
            vaginal_ratio=0.0,
            count=10,
        )
        for line in generate(spec, self.matrix, seed=23):
            tags = set(split_tags(line))
            self.assertIn("anal", tags)
            self.assertNotIn("vaginal", tags)

    def test_ratio_one_skips_anal_only_poses(self):
        spec = _spec(
            mode="sex",
            exposure=("open",),
            vaginal_ratio=1.0,
            count=20,
        )
        blob = "\n".join(generate(spec, self.matrix, seed=24))
        self.assertNotIn("full nelson", blob)


class TorsoTagTest(unittest.TestCase):
    def setUp(self):
        self.clothes = Entry(
            tags=("serafuku", "white thighhighs"),
            open_clothes=True,
            group="covered",
        )
        self.rng = random.Random(0)

    def _spec(self, **kwargs) -> PromptSpec:
        base = dict(
            character="(sena_character:1.1), 1girl",
            mode="sex",
            exposure=("open",),
            chest="auto",
            belly="covered",
        )
        base.update(kwargs)
        return PromptSpec(**base)

    def test_chest_view_front_and_optional(self):
        self.assertEqual(
            chest_view(Entry(tags=("missionary", "on back", "from front"), channel="both")),
            "front",
        )
        self.assertEqual(
            chest_view(Entry(tags=("suspended congress", "held up"), channel="both")),
            "front",
        )
        self.assertEqual(
            chest_view(
                Entry(
                    tags=("cowgirl position", "girl on top", "looking at viewer"),
                    channel="both",
                )
            ),
            "front",
        )
        self.assertEqual(
            chest_view(Entry(tags=("doggystyle", "all fours", "from behind"), channel="both")),
            "optional",
        )
        self.assertEqual(
            chest_view(Entry(tags=("prone bone", "on stomach"), channel="both")),
            "optional",
        )

    def test_front_open_always_breasts_out(self):
        pose = Entry(tags=("missionary", "on back", "from front"), channel="both")
        tags = torso_tags(self._spec(), pose, self.clothes, True, self.rng)
        self.assertIn("breasts", tags)
        self.assertIn("breasts out", tags)
        self.assertIn("nipples", tags)
        hold = Entry(tags=("suspended congress", "held up"), channel="both")
        tags = torso_tags(self._spec(), hold, self.clothes, True, self.rng)
        self.assertIn("breasts out", tags)
        cowgirl = Entry(
            tags=("cowgirl position", "girl on top", "looking at viewer"),
            channel="both",
        )
        tags = torso_tags(self._spec(), cowgirl, self.clothes, True, self.rng)
        self.assertIn("breasts out", tags)
        self.assertIn("nipples", tags)

    def test_doggy_covered_pref_skips_chest(self):
        pose = Entry(tags=("doggystyle", "all fours", "from behind"), channel="both")
        tags = torso_tags(
            self._spec(chest="covered"), pose, self.clothes, True, self.rng
        )
        self.assertNotIn("breasts out", tags)
        self.assertNotIn("nipples", tags)

    def test_doggy_nipples_pref_adds_out(self):
        pose = Entry(tags=("doggystyle", "all fours", "from behind"), channel="both")
        tags = torso_tags(
            self._spec(chest="nipples"), pose, self.clothes, True, self.rng
        )
        self.assertIn("breasts out", tags)
        self.assertIn("nipples", tags)

    def test_nude_has_breasts_not_out(self):
        pose = Entry(tags=("sitting", "looking at viewer"))
        tags = torso_tags(
            self._spec(mode="nsfw", exposure=("nude",), chest="auto"),
            pose,
            None,
            False,
            self.rng,
        )
        self.assertIn("breasts", tags)
        self.assertIn("nipples", tags)
        self.assertNotIn("breasts out", tags)

    def test_sfw_no_chest_tags(self):
        pose = Entry(tags=("sitting", "looking at viewer"))
        tags = torso_tags(
            self._spec(mode="sfw", exposure=("covered",), chest="auto", belly="auto"),
            pose,
            self.clothes,
            False,
            self.rng,
        )
        self.assertEqual(tags, [])

    def test_belly_navel(self):
        pose = Entry(tags=("missionary", "from front"), channel="both")
        tags = torso_tags(
            self._spec(chest="covered", belly="navel"),
            pose,
            self.clothes,
            True,
            self.rng,
        )
        # front+open still forces chest out
        self.assertIn("navel", tags)
        self.assertIn("midriff", tags)

    def test_assemble_puts_torso_before_pose(self):
        spec = self._spec()
        pose = Entry(tags=("missionary", "from front"), channel="both")
        clothes = self.clothes
        scene = Entry(tags=("bedroom", "indoors", "bed"))
        suffix = Entry(tags=("soft lighting",))
        line = assemble(
            spec,
            pose,
            clothes,
            True,
            scene,
            suffix,
            "vaginal",
            ["breasts", "breasts out", "nipples"],
        )
        tags = split_tags(line)
        self.assertLess(tags.index("open clothes"), tags.index("breasts out"))
        self.assertLess(tags.index("breasts out"), tags.index("missionary"))

    def test_assemble_puts_face_between_torso_and_pose(self):
        spec = self._spec()
        pose = Entry(tags=("missionary", "from front"), channel="both")
        scene = Entry(tags=("bedroom", "indoors", "bed"))
        suffix = Entry(tags=("soft lighting",))
        line = assemble(
            spec,
            pose,
            self.clothes,
            True,
            scene,
            suffix,
            "vaginal",
            ["breasts", "breasts out"],
            ["smile", "half-closed eyes"],
        )
        tags = split_tags(line)
        self.assertLess(tags.index("breasts out"), tags.index("smile"))
        self.assertLess(tags.index("half-closed eyes"), tags.index("missionary"))


class FaceTagTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)
        self.rng = random.Random(0)

    def test_sfw_pools_omit_sex_tags(self):
        expression = group_pool(FACE_GROUPS_BY_ID["expression"], "sfw")
        for tag in ("smile", "shy", "grin"):
            self.assertIn(tag, expression)
        for tag in ("orgasm", "ahegao", "pain"):
            self.assertNotIn(tag, expression)
        eyes = group_pool(FACE_GROUPS_BY_ID["eyes"], "sfw")
        self.assertIn("closed eyes", eyes)
        self.assertNotIn("rolling eyes", eyes)
        self.assertNotIn("spiral eyes", eyes)
        mouth = group_pool(FACE_GROUPS_BY_ID["mouth"], "sfw")
        self.assertIn("open mouth", mouth)
        self.assertIn("closed mouth", mouth)
        self.assertNotIn("tongue out", mouth)

    def test_sex_pool_includes_orgasm_and_pain(self):
        tags = group_pool(FACE_GROUPS_BY_ID["expression"], "sex")
        for tag in ("orgasm", "pain", "ahegao", "smile"):
            self.assertIn(tag, tags)

    def test_eyes_group_offers_open_eyes(self):
        tags = group_tags(FACE_GROUPS_BY_ID["eyes"])
        self.assertIn("open eyes", tags)

    def test_locked_smile_is_the_only_face_tag(self):
        pose = Entry(tags=("sitting", "desk", "chin rest"))
        spec = _spec(face=_face(expression="smile", gaze=FACE_NONE, eyes=FACE_NONE))
        for _ in range(8):
            self.assertEqual(face_tags(spec, pose, self.rng), ["smile"])

    def test_several_ticked_tags_still_yield_one_per_prompt(self):
        pose = Entry(tags=("sitting", "desk", "chin rest"))
        face = _face(
            expression=("smile", "shy", "grin"), gaze=FACE_NONE, eyes=FACE_NONE
        )
        seen = set()
        for seed in range(60):
            tags = face_tags(_spec(face=face), pose, random.Random(seed))
            self.assertEqual(len(tags), 1, tags)
            seen.add(tags[0])
        self.assertEqual(seen, {"smile", "shy", "grin"})

    def test_multi_tag_group_never_writes_two_of_its_tags(self):
        spec = _spec(
            mode="nsfw",
            exposure=("open",),
            face=_face(
                expression=("smile", "shy"),
                gaze=FACE_NONE,
                eyes=("open eyes", "closed eyes"),
                mouth=FACE_NONE,
            ),
            count=30,
        )
        for seed in range(20):
            for line in generate(spec, self.matrix, seed=seed):
                hits = _groups_hit(line)
                self.assertLessEqual(hits.get("expression", 0), 1, line)
                self.assertLessEqual(hits.get("eyes", 0), 1, line)

    def test_face_selection_and_value_round_trip(self):
        group = FACE_GROUPS_BY_ID["expression"]
        for value in (FACE_ANY, FACE_NONE, ("smile",), ("smile", "shy")):
            any_on, flags = face_selection(group, value)
            self.assertEqual(face_value(group, any_on, flags), value)
            self.assertEqual(face_tags_of(face_value(group, any_on, flags)), face_tags_of(value))

    def test_skip_writes_no_face_tags(self):
        pose = Entry(tags=("sitting", "desk", "chin rest"))
        spec = _spec(face={group.id: FACE_NONE for group in FACE_GROUPS})
        self.assertEqual(face_tags(spec, pose, self.rng), [])

    def test_any_yields_to_a_pose_that_fills_the_group(self):
        pose = Entry(tags=("sitting", "looking at viewer", "smile"))
        face = {group.id: FACE_NONE for group in FACE_GROUPS}
        face.update(expression=FACE_ANY, gaze=FACE_ANY)
        self.assertEqual(face_tags(_spec(face=face), pose, self.rng), [])

    def test_any_never_picks_closed_eyes_with_looking_at_viewer(self):
        pose = Entry(tags=("sitting", "looking at viewer"))
        face = _face(expression=FACE_NONE, gaze=FACE_NONE, eyes=FACE_ANY)
        for seed in range(20):
            tags = face_tags(_spec(face=face), pose, random.Random(seed))
            self.assertNotIn("closed eyes", tags)
            self.assertNotIn("looking away", tags)

    def test_locked_eyes_also_respect_a_pose_gaze(self):
        pose = Entry(tags=("sitting", "looking at viewer"))
        face = _face(expression=FACE_NONE, gaze=FACE_NONE, eyes="closed eyes")
        for seed in range(8):
            self.assertNotIn("closed eyes", face_tags(_spec(face=face), pose, random.Random(seed)))

    def test_locked_expression_yields_to_a_pose_expression(self):
        pose = Entry(tags=("covering own mouth", "shy", "blush"))
        face = _face(expression="smile", gaze=FACE_NONE, eyes=FACE_NONE, blush=FACE_NONE)
        for seed in range(8):
            self.assertEqual(face_tags(_spec(face=face), pose, random.Random(seed)), [])

    def test_at_most_one_tag_per_group_over_many_prompts(self):
        spec = _spec(
            mode="nsfw",
            exposure=("open",),
            face={group.id: FACE_ANY for group in FACE_GROUPS},
            count=40,
        )
        for seed in range(30):
            for line in generate(spec, self.matrix, seed=seed):
                for group, count in _groups_hit(line).items():
                    self.assertLessEqual(count, 1, f"{group} twice in: {line}")

    def test_generate_locked_orgasm(self):
        spec = _spec(
            mode="sex",
            exposure=("open",),
            face=_face(expression="orgasm", gaze=FACE_NONE, eyes=FACE_NONE),
            count=8,
        )
        for line in generate(spec, self.matrix, seed=31):
            self.assertIn("orgasm", split_tags(line))

    def test_sfw_auto_generate_skips_sex_face(self):
        spec = _spec(mode="sfw", count=20)
        blob = "\n".join(generate(spec, self.matrix, seed=32))
        for tag in ("orgasm", "ahegao", "pain", "rolling eyes", "spiral eyes"):
            self.assertNotIn(tag, blob)

    def test_wizard_keeps_every_ticked_tag_in_a_group(self):
        # expression: smile + shy ticked; every other group left alone / switched off.
        answer = [
            [0, 3],  # expression: smile, shy
            "none",  # gaze
            "none",  # eyes
            "none",  # mouth
            "none",  # blush
            "none",  # tears
        ]
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                0,
                [True, True, False, False, False],
                "any",
                0,
                0,
                answer,
                "any",
                "any",
                4,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertEqual(spec.face["expression"], ("smile", "shy"))
        self.assertEqual(spec.face["gaze"], FACE_NONE)
        self.assertEqual(spec.face["eyes"], FACE_NONE)
        self.assertEqual(spec.count, 4)

    def test_sfw_face_warning_flag(self):
        self.assertTrue(needs_sfw_face_warning("sfw", ["orgasm"]))
        self.assertFalse(needs_sfw_face_warning("sfw", ["smile"]))
        self.assertFalse(needs_sfw_face_warning("sex", ["orgasm"]))

    def test_selected_face_tags_ignores_any_and_off(self):
        face = _face(expression="orgasm", gaze=FACE_ANY, eyes=FACE_NONE, blush="blush")
        self.assertEqual(selected_face_tags(face), frozenset({"orgasm", "blush"}))


class AssembleOrderTest(unittest.TestCase):
    def test_sex_slot_order(self):
        spec = _spec(mode="sex", exposure=("open",))
        pose = Entry(
            tags=("doggystyle", "sex from behind"),
            channel="both",
        )
        clothes = Entry(tags=("serafuku", "white thighhighs"), open_clothes=True, group="covered")
        scene = Entry(tags=("bedroom", "indoors", "bed"))
        suffix = Entry(tags=("soft lighting",))
        line = assemble(spec, pose, clothes, True, scene, suffix, "anal")
        tags = split_tags(line)
        self.assertEqual(tags[:2], ["(sena_character:1.1)", "1girl"])
        self.assertLess(tags.index("1boy"), tags.index("serafuku"))
        self.assertLess(tags.index("open clothes"), tags.index("doggystyle"))
        self.assertLess(tags.index("sex"), tags.index("penis"))
        self.assertLess(tags.index("anal"), tags.index("bedroom"))


def _assemble_line(stage: str, channel: str = "vaginal") -> str:
    spec = _spec(mode="sex", exposure=("open",))
    pose = Entry(tags=("doggystyle", "sex from behind"), channel="both")
    clothes = Entry(tags=("serafuku", "white thighhighs"), open_clothes=True, group="covered")
    scene = Entry(tags=("bedroom", "indoors", "bed"))
    suffix = Entry(tags=("soft lighting",))
    return assemble(spec, pose, clothes, True, scene, suffix, channel, stage=stage)


class SexStageTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(MINI)

    def test_default_weights_are_during_only(self):
        weights = default_stage_weights()
        self.assertEqual(tuple(weights), SEX_STAGES)
        self.assertEqual(weights["during"], 1.0)
        self.assertTrue(all(weights[stage] == 0.0 for stage in SEX_STAGES if stage != "during"))
        for lang in ("chinese", "english"):
            for stage in SEX_STAGES:
                self.assertTrue(t(lang, f"stage_{stage}"))

    def test_pick_stage_falls_back_to_during(self):
        spec = _spec(mode="sex", stage_weights={stage: 0.0 for stage in SEX_STAGES})
        rng = random.Random(0)
        self.assertEqual({pick_stage(spec, rng) for _ in range(20)}, {"during"})

    def test_pick_stage_respects_a_single_weight(self):
        spec = _spec(mode="sex", stage_weights={**default_stage_weights(), "during": 0.0, "pose": 1.0})
        rng = random.Random(1)
        self.assertEqual({pick_stage(spec, rng) for _ in range(20)}, {"pose"})

    def test_pose_stage_has_no_penis_or_sex(self):
        tags = set(split_tags(_assemble_line("pose", "vaginal")))
        self.assertIn("solo", tags)
        self.assertIn("presenting", tags)
        self.assertIn("pussy", tags)
        self.assertNotIn("penis", tags)
        self.assertNotIn("1boy", tags)
        self.assertNotIn("sex", tags)
        self.assertNotIn("vaginal", tags)

    def test_before_stage_aims_without_sex(self):
        tags = set(split_tags(_assemble_line("before", "vaginal")))
        self.assertIn("1boy", tags)
        self.assertIn("penis", tags)
        self.assertIn("imminent vaginal", tags)
        self.assertIn("penis on pussy", tags)
        self.assertNotIn("sex", tags)
        self.assertNotIn("vaginal", tags)
        anal = set(split_tags(_assemble_line("before", "anal")))
        self.assertIn("imminent anal", anal)
        self.assertIn("penis on ass", anal)
        self.assertNotIn("sex", anal)

    def test_during_stage_matches_old_sex_line(self):
        tags = set(split_tags(_assemble_line("during", "anal")))
        self.assertIn("sex", tags)
        self.assertIn("anal", tags)
        self.assertIn("penis", tags)
        self.assertIn("1boy", tags)
        self.assertNotIn("ejaculation", tags)
        self.assertNotIn("after sex", tags)

    def test_ejaculation_uses_overflow_not_drip(self):
        tags = set(split_tags(_assemble_line("ejaculation", "vaginal")))
        self.assertIn("sex", tags)
        self.assertIn("vaginal", tags)
        self.assertIn("ejaculation", tags)
        self.assertIn("cum in pussy", tags)
        self.assertIn("cum overflow", tags)
        self.assertNotIn("cumdrip", tags)
        anal = set(split_tags(_assemble_line("ejaculation", "anal")))
        self.assertIn("cum in ass", anal)

    def test_after_keeps_penis_and_drops_sex(self):
        tags = set(split_tags(_assemble_line("after", "vaginal")))
        self.assertIn("1boy", tags)
        self.assertIn("penis", tags)
        self.assertIn("after sex", tags)
        self.assertIn("after vaginal", tags)
        self.assertIn("cum in pussy", tags)
        self.assertIn("cumdrip", tags)
        self.assertIn("gaping", tags)
        self.assertNotIn("sex", tags)
        self.assertNotIn("vaginal", tags)

    def test_done_is_after_without_penis(self):
        tags = set(split_tags(_assemble_line("done", "anal")))
        self.assertIn("solo", tags)
        self.assertIn("after sex", tags)
        self.assertIn("after anal", tags)
        self.assertIn("cumdrip", tags)
        self.assertIn("gaping", tags)
        self.assertNotIn("penis", tags)
        self.assertNotIn("1boy", tags)
        self.assertNotIn("sex", tags)
        self.assertNotIn("anal", tags)

    def test_generate_pose_only_omits_penis(self):
        spec = _spec(
            mode="sex",
            exposure=("open",),
            vaginal_ratio=1.0,
            stage_weights={**default_stage_weights(), "during": 0.0, "pose": 1.0},
            count=6,
        )
        for line in generate(spec, self.matrix, seed=11):
            tags = set(split_tags(line))
            self.assertIn("solo", tags)
            self.assertNotIn("penis", tags)
            self.assertNotIn("sex", tags)
            self.assertNotIn("1boy", tags)

    def test_generate_ejaculation_vaginal_writes_cum_in_pussy(self):
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({_find(self.matrix.poses, "mating press").key}),
            vaginal_ratio=1.0,
            stage_weights={**default_stage_weights(), "during": 0.0, "ejaculation": 1.0},
            count=5,
        )
        for line in generate(spec, self.matrix, seed=12):
            tags = set(split_tags(line))
            self.assertIn("ejaculation", tags)
            self.assertIn("cum in pussy", tags)
            self.assertIn("sex", tags)
            self.assertIn("penis", tags)

    def test_anatomy_can_omit_penis(self):
        pose = _find(self.matrix.poses, "standing missionary")
        self.assertIn("penis", anatomy_tags(pose, "vaginal"))
        self.assertNotIn("penis", anatomy_tags(pose, "vaginal", include_penis=False))

    def test_stage_tags_channel_split(self):
        self.assertEqual(stage_tags("during", "vaginal"), [])
        self.assertIn("imminent vaginal", stage_tags("before", "vaginal"))
        self.assertIn("after anal", stage_tags("after", "anal"))
        self.assertEqual(stage_tags("object_insertion", "anal"), ["anal object insertion"])
        self.assertEqual(stage_tags("object_insertion", "vaginal"), ["vaginal object insertion"])
        self.assertEqual(stage_tags("fingering", "anal"), ["anal fingering"])
        self.assertEqual(stage_tags("fingering", "vaginal"), ["fingering"])
        self.assertEqual(stage_tags("object_insertion", "none"), [])
        self.assertEqual(stage_tags("ejaculation", "none"), ["ejaculation"])
        self.assertNotIn("gaping", stage_tags("after", "none"))
        self.assertNotIn("cum in pussy", stage_tags("after", "none"))


NONE_MINI = MINI.replace(
    "full nelson : anal only",
    "full nelson : anal only\noral, fellatio : none\n# oral\npaizuri : none\n# paizuri\nnursing handjob : none\n# nursing",
)


class NoneChannelAndObjectStageTest(unittest.TestCase):
    def setUp(self):
        self.matrix = parse_matrix(NONE_MINI)
        self.real = load_matrix(MATRIX_PATH)

    def test_none_channel_parse(self):
        oral = _find(self.matrix.poses, "oral, fellatio")
        self.assertEqual(oral.channel, "none")
        self.assertFalse(pose_allows_hole(oral))
        self.assertTrue(pose_allows_hole(_find(self.matrix.poses, "doggystyle")))
        self.assertTrue(pose_allows_hole(_find(self.matrix.poses, "full nelson")))

    def test_bind_object_stage_drops_none_channel_poses(self):
        stage, pool = bind_stage_to_poses(self.matrix.poses, "object_insertion")
        self.assertEqual(stage, "object_insertion")
        blobs = [pose.blob for pose in pool]
        self.assertTrue(any("doggystyle" in blob for blob in blobs))
        self.assertFalse(any("oral" in blob for blob in blobs))
        self.assertFalse(any("paizuri" in blob for blob in blobs))

    def test_bind_falls_back_when_every_pose_is_none_channel(self):
        oral = _find(self.matrix.poses, "oral, fellatio")
        stage, pool = bind_stage_to_poses([oral], "fingering")
        self.assertEqual(stage, "during")
        self.assertEqual(pool, [oral])

    def test_oral_during_writes_penis_not_sex_or_hole(self):
        oral = _find(self.matrix.poses, "oral, fellatio")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({oral.key}),
            count=6,
        )
        for line in generate(spec, self.matrix, seed=40):
            tags = set(split_tags(line))
            self.assertIn("oral", tags)
            self.assertIn("fellatio", tags)
            self.assertIn("penis", tags)
            self.assertIn("1boy", tags)
            self.assertNotIn("sex", tags)
            self.assertNotIn("vaginal", tags)
            self.assertNotIn("anal", tags)
            self.assertNotIn("pussy", tags)
            self.assertNotIn("anus", tags)
            self.assertNotIn("imminent vaginal", tags)

    def test_paizuri_and_nursing_are_none_channel(self):
        for needle in ("paizuri", "nursing handjob"):
            pose = _find(self.matrix.poses, needle)
            spec = _spec(
                mode="sex",
                exposure=("open",),
                pose_any=False,
                pose_keys=frozenset({pose.key}),
                count=4,
            )
            for line in generate(spec, self.matrix, seed=41):
                tags = set(split_tags(line))
                self.assertIn(needle, tags)
                self.assertIn("penis", tags)
                self.assertNotIn("sex", tags)
                self.assertNotIn("vaginal", tags)
                self.assertNotIn("anal", tags)

    def test_real_matrix_none_channel_rows(self):
        for needle in ("oral, fellatio", "paizuri", "nursing handjob"):
            pose = _find(self.real.poses, needle)
            self.assertEqual(pose.channel, "none", needle)
            self.assertEqual(pose_families(pose), frozenset(), needle)

    def test_paizuri_oral_branch_on_the_real_matrix(self):
        combo = _find(self.real.poses, "paizuri, oral, fellatio")
        self.assertEqual(combo.channel, "none")
        self.assertEqual(set(combo.tags), {"paizuri", "oral", "fellatio"})
        cameras = [
            pose
            for pose in self.real.poses
            if "paizuri" in pose.tags and "fellatio" in pose.tags
        ]
        self.assertGreaterEqual(len(cameras), 4)
        for pose in cameras:
            self.assertEqual(pose.channel, "none", pose.blob)
            self.assertIn("oral", pose.tags)

    def test_anal_object_insertion_skips_penis_and_other_stages(self):
        nelson = _find(self.matrix.poses, "full nelson")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({nelson.key}),
            stage_weights={**default_stage_weights(), "during": 0.0, "object_insertion": 1.0},
            count=6,
        )
        for line in generate(spec, self.matrix, seed=42):
            tags = set(split_tags(line))
            self.assertIn("anal object insertion", tags)
            self.assertIn("full nelson", tags)
            self.assertIn("anus", tags)
            self.assertIn("solo", tags)
            self.assertNotIn("penis", tags)
            self.assertNotIn("1boy", tags)
            self.assertNotIn("sex", tags)
            self.assertNotIn("anal", tags)
            self.assertNotIn("ejaculation", tags)
            self.assertNotIn("presenting", tags)
            self.assertNotIn("imminent anal", tags)

    def test_vaginal_fingering_on_a_vaginal_pose(self):
        press = _find(self.matrix.poses, "mating press")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({press.key}),
            vaginal_ratio=1.0,
            stage_weights={**default_stage_weights(), "during": 0.0, "fingering": 1.0},
            count=6,
        )
        for line in generate(spec, self.matrix, seed=43):
            tags = set(split_tags(line))
            self.assertIn("fingering", tags)
            self.assertNotIn("anal fingering", tags)
            self.assertNotIn("penis", tags)
            self.assertNotIn("sex", tags)
            self.assertNotIn("vaginal", tags)
            self.assertIn("pussy", tags)

    def test_object_insertion_never_lands_on_oral(self):
        spec = _spec(
            mode="sex",
            exposure=("open",),
            stage_weights={**default_stage_weights(), "during": 0.0, "object_insertion": 1.0},
            count=20,
        )
        blob = "\n".join(generate(spec, self.matrix, seed=44))
        self.assertNotIn("oral", blob)
        self.assertNotIn("paizuri", blob)
        self.assertNotIn("nursing handjob", blob)
        self.assertIn("object insertion", blob)

    def test_anal_fingering_follows_anal_only_pose(self):
        nelson = _find(self.matrix.poses, "full nelson")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({nelson.key}),
            stage_weights={**default_stage_weights(), "during": 0.0, "fingering": 1.0},
            count=5,
        )
        for line in generate(spec, self.matrix, seed=45):
            tags = set(split_tags(line))
            self.assertIn("anal fingering", tags)
            self.assertNotIn("penis", tags)

    def test_both_pose_object_insertion_follows_ratio(self):
        dog = _find(self.matrix.poses, "doggystyle")
        spec = _spec(
            mode="sex",
            exposure=("open",),
            pose_any=False,
            pose_keys=frozenset({dog.key}),
            vaginal_ratio=1.0,
            stage_weights={**default_stage_weights(), "during": 0.0, "object_insertion": 1.0},
            count=6,
        )
        for line in generate(spec, self.matrix, seed=46):
            tags = set(split_tags(line))
            self.assertIn("vaginal object insertion", tags)
            self.assertNotIn("anal object insertion", tags)

    def test_default_object_stages_are_zero(self):
        weights = default_stage_weights()
        for stage in STAGES_OBJECT:
            self.assertEqual(weights[stage], 0.0)

    def test_assemble_object_insertion_vaginal(self):
        spec = _spec(mode="sex", exposure=("open",))
        pose = Entry(tags=("doggystyle", "sex from behind"), channel="both")
        clothes = Entry(tags=("serafuku",), open_clothes=True, group="covered")
        scene = Entry(tags=("bedroom",))
        suffix = Entry(tags=("soft lighting",))
        line = assemble(
            spec, pose, clothes, True, scene, suffix, "vaginal", stage="object_insertion"
        )
        tags = set(split_tags(line))
        self.assertIn("vaginal object insertion", tags)
        self.assertNotIn("penis", tags)
        self.assertNotIn("sex", tags)
        self.assertIn("solo", tags)
        self.assertIn("pussy", tags)


class ProfileTest(unittest.TestCase):
    """Named wizard configurations: JSON round trip, the confirm-page save, and the CLI."""

    def setUp(self):
        self.matrix = parse_matrix(MINI)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _full_spec(self) -> PromptSpec:
        return _spec(
            mode="sex",
            exposure=("revealing", "open"),
            clothing_any=False,
            clothing_keys=frozenset({("bikini", "side-tie bikini bottom")}),
            scene_any=False,
            scene_keys=frozenset({("bedroom", "indoors", "bed")}),
            pose_any=False,
            pose_keys=frozenset(
                {("cowgirl position", "girl on top", "sitting", "straddling", "looking at viewer")}
            ),
            family_any=False,
            families=frozenset({"girl_on_top"}),
            vaginal_ratio=0.3,
            stage_weights={
                **default_stage_weights(),
                "pose": 0.1,
                "before": 0.1,
                "during": 0.4,
                "ejaculation": 0.2,
                "after": 0.1,
                "done": 0.1,
            },
            chest="nipples",
            belly="navel",
            face=_face(expression="orgasm", gaze=FACE_NONE, eyes=FACE_NONE),
            count=7,
        )

    def test_round_trip_preserves_spec(self):
        spec = self._full_spec()
        path = save_profile(self.dir, "sena test", spec)
        self.assertTrue(path.exists())
        self.assertEqual(load_profile(path), spec)

    def test_reloaded_profile_generates_the_same_prompts(self):
        spec = self._full_spec()
        save_profile(self.dir, "sena test", spec)
        loaded = load_profile(resolve_profile(self.dir, "sena test"))
        self.assertEqual(
            generate(spec, self.matrix, seed=5), generate(loaded, self.matrix, seed=5)
        )

    def test_list_and_resolve_by_name(self):
        save_profile(self.dir, "Bravo", self._full_spec())
        save_profile(self.dir, "alpha", self._full_spec())
        self.assertEqual([name for name, _ in list_profiles(self.dir)], ["alpha", "Bravo"])
        self.assertEqual(load_profile(resolve_profile(self.dir, "BRAVO")), self._full_spec())
        with self.assertRaises(ProfileError):
            resolve_profile(self.dir, "missing")

    def test_unsafe_names_stay_inside_the_dir(self):
        path = save_profile(self.dir, "sena/v2", self._full_spec())
        self.assertEqual(path.parent, self.dir)
        self.assertEqual(path.name, "sena_v2.json")
        self.assertEqual(safe_profile_name(".."), "")
        with self.assertRaises(ProfileError):
            save_profile(self.dir, "..", self._full_spec())

    def test_invalid_spec_rejected(self):
        bad = self.dir / "bad.json"
        bad.write_text(
            json.dumps({"version": 2, "name": "bad", "spec": {"character": "", "mode": "sfw"}}),
            encoding="utf-8",
        )
        with self.assertRaises(ProfileError):
            load_profile(bad)

    def test_missing_stage_weights_default_to_during(self):
        path = save_profile(self.dir, "old", self._full_spec())
        data = json.loads(path.read_text(encoding="utf-8"))
        data["spec"].pop("stage_weights", None)
        path.write_text(json.dumps(data), encoding="utf-8")
        spec = load_profile(path)
        self.assertEqual(spec.stage_weights, default_stage_weights())

    def test_unknown_stage_rejected(self):
        path = save_profile(self.dir, "badstage", self._full_spec())
        data = json.loads(path.read_text(encoding="utf-8"))
        data["spec"]["stage_weights"] = {"pose": 1.0, "foreplay": 0.5}
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ProfileError):
            load_profile(path)

    def test_old_six_stage_weights_zero_the_new_stages(self):
        path = save_profile(self.dir, "oldstages", self._full_spec())
        data = json.loads(path.read_text(encoding="utf-8"))
        data["spec"]["stage_weights"] = {
            "pose": 0.0,
            "before": 0.0,
            "during": 1.0,
            "ejaculation": 0.0,
            "after": 0.0,
            "done": 0.0,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        spec = load_profile(path)
        self.assertEqual(spec.stage_weights["during"], 1.0)
        self.assertEqual(spec.stage_weights["object_insertion"], 0.0)
        self.assertEqual(spec.stage_weights["fingering"], 0.0)

    def test_unknown_face_option_rejected(self):
        bad = self.dir / "bad_face.json"
        bad.write_text(
            json.dumps(
                {
                    "version": 2,
                    "name": "bad face",
                    "spec": {
                        "character": "c",
                        "mode": "sfw",
                        "exposure": ["covered"],
                        "face": {"expression": "not a tag"},
                    },
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ProfileError):
            load_profile(bad)

    def test_face_option_from_another_group_rejected(self):
        for face in ({"mouth": "smile"}, {"mouth": ["smile"]}):
            bad = self.dir / "misgrouped.json"
            bad.write_text(
                json.dumps(
                    {
                        "version": 3,
                        "name": "misgrouped",
                        "spec": {
                            "character": "c",
                            "mode": "sfw",
                            "exposure": ["covered"],
                            "face": face,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ProfileError):
                load_profile(bad)

    def test_face_list_round_trips_through_a_profile(self):
        spec = self._full_spec()
        spec.face["expression"] = ("smile", "shy")
        path = save_profile(self.dir, "multi", spec)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 3)
        self.assertEqual(load_profile(path).face["expression"], ("smile", "shy"))

    def _write_v1(self, name: str, spec: dict) -> Path:
        path = self.dir / f"{name}.json"
        path.write_text(
            json.dumps({"version": 1, "name": name, "spec": spec}), encoding="utf-8"
        )
        return path

    def _v1_spec(self, **overrides) -> dict:
        spec = {
            "character": "(sena_character:1.1), 1girl",
            "mode": "nsfw",
            "exposure": ["revealing"],
            "count": 4,
            "expression_any": True,
            "expression_none": False,
            "expression_keys": [],
            "eye_any": True,
            "eye_none": False,
            "eye_keys": [],
        }
        spec.update(overrides)
        return spec

    def test_v1_profile_upgrades_any_to_the_new_defaults(self):
        path = self._write_v1("v1any", self._v1_spec())
        spec = load_profile(path)
        self.assertEqual(spec.face["expression"], FACE_ANY)
        self.assertEqual(spec.face["eyes"], FACE_ANY)
        self.assertEqual(spec.face["gaze"], FACE_NONE)
        self.assertEqual(spec.face["mouth"], FACE_NONE)
        self.assertEqual(spec.face["blush"], FACE_NONE)
        self.assertEqual(spec.face["tears"], FACE_NONE)
        self.assertEqual(spec.face, default_face())

    def test_v1_profile_routes_locked_tags_into_their_groups(self):
        path = self._write_v1(
            "v1route",
            self._v1_spec(
                expression_any=False,
                expression_keys=["blush"],
                eye_any=False,
                eye_keys=["looking at viewer"],
            ),
        )
        spec = load_profile(path)
        self.assertEqual(spec.face["blush"], ("blush",))
        self.assertEqual(spec.face["gaze"], ("looking at viewer",))
        # Groups the v1 axis covered but the user pinned to nothing stay off.
        self.assertEqual(spec.face["expression"], FACE_NONE)
        self.assertEqual(spec.face["mouth"], FACE_NONE)
        self.assertEqual(spec.face["tears"], FACE_NONE)
        self.assertEqual(spec.face["eyes"], FACE_NONE)

    def test_v1_same_group_multi_select_becomes_a_candidate_list(self):
        path = self._write_v1(
            "v1multi",
            self._v1_spec(
                expression_any=False,
                expression_keys=["smile", "shy"],
                eye_any=False,
                eye_none=True,
                eye_keys=[],
            ),
        )
        spec = load_profile(path)
        self.assertEqual(spec.face["expression"], ("smile", "shy"))

    def test_v2_tag_becomes_a_candidate_list(self):
        path = save_profile(self.dir, "v2tag", self._full_spec())
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 2
        data["spec"]["face"] = {"expression": "smile", "gaze": "none", "eyes": "any"}
        path.write_text(json.dumps(data), encoding="utf-8")
        spec = load_profile(path)
        self.assertEqual(spec.face["expression"], ("smile",))
        self.assertEqual(spec.face["gaze"], FACE_NONE)
        self.assertEqual(spec.face["eyes"], FACE_ANY)

    def test_v1_migration_warning_only_on_load_not_on_listing(self):
        self._write_v1(
            "v1quiet", self._v1_spec(expression_any=False, expression_keys=["not a tag"])
        )
        with contextlib.redirect_stderr(io.StringIO()) as err:
            list_profiles(self.dir)
        self.assertEqual(err.getvalue(), "")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            load_profile(self.dir / "v1quiet.json")
        self.assertIn("unknown", err.getvalue())

    def test_v1_profile_none_axis_stays_off(self):
        path = self._write_v1(
            "v1none",
            self._v1_spec(
                expression_any=False,
                expression_none=True,
                eye_any=False,
                eye_none=True,
            ),
        )
        spec = load_profile(path)
        for group in ("expression", "mouth", "blush", "tears", "gaze", "eyes"):
            self.assertEqual(spec.face[group], FACE_NONE)

    def test_unsupported_version_rejected(self):
        path = save_profile(self.dir, "p", self._full_spec())
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ProfileError):
            load_profile(path)

    def _sfw_wizard_answers(self, tail: list) -> list:
        return [
            "(sena_character:1.1), 1girl",
            0,  # mode: sfw
            [True, True, False, False, False],  # exposure: covered + casual
            "any",  # clothing
            0,  # chest: covered
            0,  # belly: covered
            "default",  # face: keep the per-group defaults
            "any",  # scene
            "any",  # poses
            4,  # count
        ] + tail

    def test_wizard_saves_then_generates(self):
        ui = ScriptedUI(self._sfw_wizard_answers(["save", "saved one", True]))
        spec = run_wizard(self.matrix, "english", ui, self.dir)
        entries = list_profiles(self.dir)
        self.assertEqual([name for name, _ in entries], ["saved one"])
        self.assertEqual(load_profile(entries[0][1]), spec)
        self.assertEqual(spec.count, 4)

    def test_wizard_save_can_be_skipped(self):
        ui = ScriptedUI(self._sfw_wizard_answers([True]))
        spec = run_wizard(self.matrix, "english", ui, self.dir)
        self.assertEqual(list_profiles(self.dir), [])
        self.assertEqual(spec.count, 4)

    def test_wizard_declining_overwrite_keeps_old_profile(self):
        save_profile(self.dir, "dup", self._full_spec())
        ui = ScriptedUI(self._sfw_wizard_answers(["save", "dup", False, True]))
        spec = run_wizard(self.matrix, "english", ui, self.dir)
        self.assertEqual(spec.count, 4)
        self.assertEqual(load_profile(self.dir / "dup.json"), self._full_spec())

    def test_main_profile_flag_generates_without_a_wizard(self):
        save_profile(self.dir, "cli", _spec(count=3))
        out = self.dir / "out.txt"
        rc = main(
            [
                "--matrix",
                str(MATRIX_PATH),
                "--language",
                "english",
                "--profiles-dir",
                str(self.dir),
                "--profile",
                "cli",
                "-o",
                str(out),
            ]
        )
        self.assertEqual(rc, 0)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertTrue(line.startswith("(sena_character:1.1), 1girl"))


if __name__ == "__main__":
    unittest.main()

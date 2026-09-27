#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from tools.gen_prompts import (
    EXPRESSIONS,
    EYES,
    Entry,
    MatrixError,
    PromptSpec,
    ScriptedUI,
    anatomy_tags,
    assemble,
    chest_view,
    face_tags,
    generate,
    is_forbidden_tag,
    load_matrix,
    needs_sfw_exposure_warning,
    needs_sfw_face_warning,
    parse_matrix,
    parse_vaginal_ratio,
    pose_families,
    pose_locus,
    pose_pool,
    pose_weight,
    resolve_face_pool,
    run_wizard,
    scene_compatible,
    split_tags,
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
                "any",
                "any",
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
                "any",
                "any",
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
                "any",
                "any",
                "any",
                [False, True, False, False, False, False],
                0.3,
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
        self.assertEqual(spec.chest, "auto")
        self.assertEqual(spec.belly, "auto")
        self.assertTrue(spec.expression_any)
        self.assertTrue(spec.eye_any)


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

    def test_sfw_pool_omits_sex_expressions(self):
        tags = resolve_face_pool(EXPRESSIONS, "sfw", True, False, frozenset())
        self.assertIn("smile", tags)
        self.assertIn("blush", tags)
        self.assertIn("shy", tags)
        self.assertNotIn("orgasm", tags)
        self.assertNotIn("ahegao", tags)
        self.assertNotIn("pain", tags)
        self.assertNotIn("tongue out", tags)
        eyes = resolve_face_pool(EYES, "sfw", True, False, frozenset())
        self.assertIn("closed eyes", eyes)
        self.assertNotIn("rolling eyes", eyes)
        self.assertNotIn("spiral eyes", eyes)

    def test_sex_pool_includes_orgasm_and_pain(self):
        tags = resolve_face_pool(EXPRESSIONS, "sex", True, False, frozenset())
        self.assertIn("orgasm", tags)
        self.assertIn("pain", tags)
        self.assertIn("ahegao", tags)
        self.assertIn("smile", tags)

    def test_locked_smile_always_written(self):
        pose = Entry(tags=("sitting", "desk", "chin rest"))
        spec = _spec(
            expression_any=False,
            expression_keys=frozenset({"smile"}),
            eye_none=True,
            eye_any=False,
        )
        for _ in range(8):
            tags = face_tags(spec, pose, self.rng)
            self.assertEqual(tags, ["smile"])

    def test_none_skips_catalog_tags(self):
        pose = Entry(tags=("sitting", "desk", "chin rest"))
        spec = _spec(
            expression_none=True,
            expression_any=False,
            eye_none=True,
            eye_any=False,
        )
        self.assertEqual(face_tags(spec, pose, self.rng), [])

    def test_auto_skips_when_pose_already_has_expression(self):
        pose = Entry(tags=("sitting", "looking at viewer", "smile"))
        spec = _spec(expression_any=True, eye_none=True, eye_any=False)
        self.assertEqual(face_tags(spec, pose, self.rng), [])

    def test_auto_skips_closed_eyes_when_looking_at_viewer(self):
        pose = Entry(tags=("sitting", "looking at viewer"))
        spec = _spec(
            expression_none=True,
            expression_any=False,
            eye_any=True,
        )
        for seed in range(20):
            tags = face_tags(spec, pose, random.Random(seed))
            self.assertNotIn("closed eyes", tags)
            self.assertNotIn("looking away", tags)

    def test_generate_locked_orgasm(self):
        spec = _spec(
            mode="sex",
            exposure=("open",),
            expression_any=False,
            expression_keys=frozenset({"orgasm"}),
            eye_none=True,
            eye_any=False,
            count=8,
        )
        for line in generate(spec, self.matrix, seed=31):
            self.assertIn("orgasm", split_tags(line))

    def test_sfw_auto_generate_skips_sex_face(self):
        spec = _spec(mode="sfw", count=20)
        blob = "\n".join(generate(spec, self.matrix, seed=32))
        for tag in ("orgasm", "ahegao", "pain", "rolling eyes", "spiral eyes"):
            self.assertNotIn(tag, blob)

    def test_wizard_locks_expression_and_none_eyes(self):
        n_expr = 1 + len(EXPRESSIONS)
        expr_flags = [False] * n_expr
        expr_flags[1] = True
        n_eye = 1 + len(EYES)
        eye_flags = [False] * n_eye
        eye_flags[0] = True
        ui = ScriptedUI(
            [
                "(sena_character:1.1), 1girl",
                0,
                [True, True, False, False, False],
                "any",
                0,
                0,
                expr_flags,
                eye_flags,
                "any",
                "any",
                4,
                True,
            ]
        )
        spec = run_wizard(self.matrix, "english", ui)
        self.assertFalse(spec.expression_any)
        self.assertEqual(spec.expression_keys, frozenset({"smile"}))
        self.assertTrue(spec.eye_none)
        self.assertFalse(spec.eye_any)

    def test_sfw_face_warning_flag(self):
        self.assertTrue(needs_sfw_face_warning("sfw", ["orgasm"]))
        self.assertFalse(needs_sfw_face_warning("sfw", ["smile"]))
        self.assertFalse(needs_sfw_face_warning("sex", ["orgasm"]))


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


if __name__ == "__main__":
    unittest.main()

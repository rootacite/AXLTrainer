package com.acite.axlranko.prompt

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The port against the repo's own `input_matrix.txt` and the named profiles under
 * `prompt_profiles/`, mirroring the Python suite's `test_real_matrix_*` checks. That folder is
 * gitignored, so a checkout without local profiles only asserts that an empty store reads cleanly.
 */
class RealMatrixPromptTest {

    private val root: File = repoRoot()
    private val matrix: PromptMatrix by lazy {
        parseMatrix(File(root, "input_matrix.txt").readText(Charsets.UTF_8))
    }

    private fun repoRoot(): File {
        var dir: File? = File(System.getProperty("user.dir").orEmpty()).absoluteFile
        while (dir != null) {
            if (File(dir, "input_matrix.txt").isFile && File(dir, "trainer").isDirectory) return dir
            dir = dir.parentFile
        }
        throw AssertionError("could not find the repo root from ${System.getProperty("user.dir")}")
    }

    @Test
    fun theRealMatrixParsesEverySection() {
        assertTrue(matrix.poses.size >= 40, "poses: ${matrix.poses.size}")
        assertTrue(matrix.clothing.size >= 20)
        assertTrue(matrix.scenes.size >= 10)
        assertTrue(matrix.suffixes.isNotEmpty())
        assertTrue(matrix.sfwPoses.size >= 10)
        assertTrue(matrix.questionablePoses.size >= 100, "questionable: ${matrix.questionablePoses.size}")
        assertTrue(matrix.clothing.all { it.group != null })
        assertTrue(matrix.questionablePoses.all { it.selfStated })
        assertFalse(matrix.sfwPoses.any { it.selfStated })
    }

    @Test
    fun theFillOnlyAddsWhatARowLeavesOpen() {
        val chestFill = listOf("topless", "nipples", "breasts hanging", "sideboob")
        val rows = matrix.questionablePoses
        var untouched = 0
        rows.forEach { row ->
            val fill = PromptGenerator.stateFill(row)
            if (POSE_CHEST_WORDS.any { row.blob.contains(it) }) {
                assertTrue(fill.none { it in chestFill }, "${row.blob} -> $fill")
            }
            if (POSE_BOTTOM_WORDS.any { row.blob.contains(it) }) {
                assertFalse("bottomless" in fill, "${row.blob} -> $fill")
            }
            if (PromptGenerator.chestView(row) !in setOf("front", "side")) {
                assertTrue(fill.none { it in chestFill }, "chest not in frame: ${row.blob} -> $fill")
            }
            if (fill.isEmpty()) untouched++
        }
        assertTrue(untouched >= 15, "rows stating both halves: $untouched")
        assertTrue(rows.any { "bottomless" in PromptGenerator.stateFill(it) })
        assertTrue(rows.any { "topless" in PromptGenerator.stateFill(it) })
        assertTrue(rows.any { "breasts hanging" in PromptGenerator.stateFill(it) })
        // No shipped row is a side view with an unstated chest, so the `sideboob` fill is only
        // exercised by the fixture in PoseStateTest.
    }

    @Test
    fun theThreePoolsDoNotOverlap() {
        assertEquals(matrix.poses, PromptGenerator.posePool(matrix, PromptMode.Sex))
        assertEquals(matrix.sfwPoses, PromptGenerator.posePool(matrix, PromptMode.Sfw))
        assertEquals(
            matrix.sfwPoses + matrix.questionablePoses,
            PromptGenerator.posePool(matrix, PromptMode.Nsfw),
        )
        val sfw = PromptGenerator.generate(PromptSpec(mode = PromptMode.Sfw, count = 200), matrix, 31)
        val sex = PromptGenerator.generate(
            PromptSpec(mode = PromptMode.Sex, exposure = listOf("open"), count = 200),
            matrix,
            32,
        )
        val nsfw = PromptGenerator.generate(PromptSpec(mode = PromptMode.Nsfw, count = 200), matrix, 33)
        assertNoQuestionableWords(sfw.joinToString("\n"))
        assertNoQuestionableWords(sex.joinToString("\n"))
        val nsfwBlob = nsfw.joinToString("\n")
        assertTrue(nsfwBlob.contains("panties"), "nsfw draws none of the questionable rows")
        listOf("mating press", "doggystyle", "spooning", "prone bone", "sex,", "penis", "pussy", "anus")
            .forEach { assertFalse(nsfwBlob.contains(it), "$it in nsfw") }
    }

    @Test
    fun aSelfStatedRowIsNeverContradicted() {
        val spec = PromptSpec(mode = PromptMode.Nsfw, count = 200)
        val warnings = mutableListOf<String>()
        val lines = (1..5).flatMap { PromptGenerator.generate(spec, matrix, seed = 800L + it, warnings) }
        assertEquals(1000, lines.size)
        assertTrue(warnings.isEmpty(), warnings.toString())
        lines.forEach { line ->
            val tags = splitTags(line).toSet()
            if ("covering breasts" in tags) assertFalse("nipples" in tags, line)
            if ("covering crotch" in tags) assertFalse("bottomless" in tags, line)
            if ("panties" in tags || "skirt" in tags || "ass" in tags) assertFalse("bottomless" in tags, line)
            if ("bra" in tags || "towel" in tags || "open shirt" in tags) assertFalse("nude" in tags, line)
            if ("topless" in tags) assertFalse("open clothes" in tags, line)
        }
    }

    @Test
    fun theBarePlaceWordsOnTheNewRowsHold() {
        val railing = find(matrix.questionablePoses, "bent over, railing")
        assertTrue(PromptGenerator.sceneCompatible(railing, find(matrix.scenes, "rooftop, outdoors, school")))
        assertFalse(PromptGenerator.sceneCompatible(railing, find(matrix.scenes, "bedroom, indoors, bed")))

        val desk = find(matrix.questionablePoses, "bent over, desk")
        assertTrue(PromptGenerator.sceneCompatible(desk, find(matrix.scenes, "classroom, indoors, desk")))
        assertTrue(PromptGenerator.sceneCompatible(desk, find(matrix.scenes, "cafe, indoors, window")))
        assertFalse(PromptGenerator.sceneCompatible(desk, find(matrix.scenes, "onsen, indoors, steam")))

        val steam = find(matrix.questionablePoses, "steam, towel, sideboob")
        assertTrue(PromptGenerator.sceneCompatible(steam, find(matrix.scenes, "onsen, indoors, steam")))
        assertFalse(PromptGenerator.sceneCompatible(steam, find(matrix.scenes, "bedroom, indoors, bed")))

        // A skeleton window pose: the bare `window` word, not only `looking outside`, pins the scene.
        val atWindow = find(matrix.sfwPoses, "sitting at window")
        assertTrue(PromptGenerator.sceneCompatible(atWindow, find(matrix.scenes, "cafe, indoors, window")))
        assertFalse(PromptGenerator.sceneCompatible(atWindow, find(matrix.scenes, "outdoors, street, city")))

        // The one older row the bare `desk` word narrows: a desk or a cafe table, no classroom window.
        val chinRest = find(matrix.sfwPoses, "sitting, desk, chin rest")
        assertTrue(PromptGenerator.sceneCompatible(chinRest, find(matrix.scenes, "classroom, indoors, desk")))
        assertFalse(PromptGenerator.sceneCompatible(chinRest, find(matrix.scenes, "classroom, indoors, window")))
    }

    @Test
    fun realPosesKeepTheirFamilies() {
        val expected = mapOf(
            "mating press" to setOf(PoseFamily.Face),
            "missionary, spread legs" to setOf(PoseFamily.Face),
            "missionary, on back" to setOf(PoseFamily.Face),
            "anvil position" to setOf(PoseFamily.Face),
            "leaning back" to setOf(PoseFamily.GirlOnTop),
            "squatting cowgirl" to setOf(PoseFamily.GirlOnTop),
            "reverse cowgirl" to setOf(PoseFamily.GirlOnTop, PoseFamily.Behind),
            "top-down bottom-up" to setOf(PoseFamily.Behind),
            "all fours" to setOf(PoseFamily.Behind),
            "standing doggystyle" to setOf(PoseFamily.StandBehind),
            "spooning" to setOf(PoseFamily.SideBehind),
            "prone bone" to setOf(PoseFamily.Behind),
            "reverse suspended" to setOf(PoseFamily.Hold, PoseFamily.StandBehind),
            "suspended congress, held up" to setOf(PoseFamily.Hold, PoseFamily.Face),
            "full nelson" to setOf(PoseFamily.Hold, PoseFamily.StandBehind),
            "standing missionary" to setOf(PoseFamily.Face),
            "upright straddle" to setOf(PoseFamily.GirlOnTop, PoseFamily.Hold),
            "seventh posture" to setOf(PoseFamily.SideBehind),
        )
        expected.forEach { (needle, families) ->
            assertEquals(families, PromptGenerator.poseFamilies(find(matrix.poses, needle)), needle)
        }
        assertEquals(
            setOf(PoseFamily.GirlOnTop),
            PromptGenerator.poseFamilies(find(matrix.poses, "straddling, looking at viewer")),
        )
    }

    @Test
    fun cameraVariantsKeepTheirChannel() {
        matrix.poses.forEach { pose ->
            val blob = pose.blob
            if (blob.contains("reverse cowgirl")) assertEquals(PromptChannel.Vaginal, pose.channel, blob)
            if (blob.contains("full nelson")) assertEquals(PromptChannel.Anal, pose.channel, blob)
            if (blob.contains("squatting cowgirl")) assertEquals(PromptChannel.Vaginal, pose.channel, blob)
            if (blob.contains("leaning back")) assertEquals(PromptChannel.Anal, pose.channel, blob)
            if (pose.tags.any { it in setOf("oral", "paizuri", "nursing handjob") }) {
                assertEquals(PromptChannel.None, pose.channel, blob)
            }
        }
    }

    @Test
    fun theNoneChannelRowsCarryNoFamily() {
        listOf("oral, fellatio", "paizuri", "nursing handjob").forEach { needle ->
            val pose = find(matrix.poses, needle)
            assertEquals(PromptChannel.None, pose.channel, needle)
            assertTrue(PromptGenerator.poseFamilies(pose).isEmpty(), needle)
        }
    }

    @Test
    fun thePaizuriOralBranchKeepsEveryCameraVariantNoChannel() {
        val combo = find(matrix.poses, "paizuri, oral, fellatio")
        assertEquals(PromptChannel.None, combo.channel)
        assertEquals(setOf("paizuri", "oral", "fellatio"), combo.tags.toSet())
        val cameras = matrix.poses.filter { it.tags.contains("paizuri") && it.tags.contains("fellatio") }
        assertTrue(cameras.size >= 4, "camera variants: ${cameras.size}")
        cameras.forEach { pose ->
            assertEquals(PromptChannel.None, pose.channel, pose.blob)
            assertTrue(pose.tags.contains("oral"), pose.blob)
        }
    }

    @Test
    fun aRealProfileGeneratesThePromptsItDescribes() {
        val profiles = File(root, "prompt_profiles")
        val files = profiles.listFiles { file -> file.isFile && file.extension.equals("json", true) }
            ?.sortedBy { it.name.lowercase() }
            .orEmpty()
        if (files.isEmpty()) {
            assertTrue(files.isEmpty())
            return
        }
        files.forEach { file ->
            val loaded = ProfileCodec.loadProfile(file.readText(Charsets.UTF_8), file.nameWithoutExtension)
            assertTrue(loaded.spec.character.isNotBlank(), file.name)
            if (file.nameWithoutExtension == "General2") {
                assertTrue(
                    loaded.notes.any { it.contains("upgraded from v2") },
                    "General2 is a v2 profile: ${loaded.notes}",
                )
            }
            val prompts = PromptGenerator.generate(loaded.spec, matrix, seed = 4242)
            assertEquals(loaded.spec.count, prompts.size, file.name)
            prompts.forEach { line ->
                assertTrue(line.startsWith(splitTags(loaded.spec.character).joinToString(", ")), line)
            }
        }
    }
    @Test
    fun onlyTheFoldedPosesCountAsHoldingTheirLegs() {
        // What a `fingering` stage has no arm for: the poses that fold or pin the legs. The check is
        // on the repo's own rows, because the data decides — 9 poses hold their legs, and
        // `spread legs` / `squatting` / `leg lift` stay out of it even though SPREAD_MARKERS (the
        // legs' shape, not who holds them) lists them.
        val held = matrix.poses.filter { PromptGenerator.poseHoldsLegs(it) }.map { it.blob }
        assertEquals(9, held.size, held.toString())
        held.forEach {
            assertTrue(it.contains("mating press") || it.contains("anvil") || it.contains("full nelson"), it)
        }
        listOf(
            "missionary, spread legs, from above",
            "squatting cowgirl position, squatting",
            "seventh posture, on side, leg lift",
            "doggystyle, all fours, from behind",
        ).forEach { pose ->
            assertFalse(PromptGenerator.poseHoldsLegs(find(matrix.poses, pose)), "$pose holds no legs")
        }
    }

    @Test
    fun aFingeringStageOnAHeldLegsPoseAsksForAPartner() {
        // The three-handed draw this rule exists for, on the real matrix: `full nelson` with
        // `anal fingering` used to be written as `solo`.
        val nelson = find(matrix.poses, "full nelson")
        val spec = defaultSpec().apply {
            mode = PromptMode.Sex
            exposure = listOf("open")
            poseAny = false
            poseKeys = setOf(nelson.key)
            stageWeights = defaultStageWeights() + mapOf(SexStage.During to 0.0, SexStage.Fingering to 1.0)
            count = 5
        }
        PromptGenerator.generate(spec, matrix, seed = 9).forEach { line ->
            val tags = splitTags(line).toSet()
            assertTrue(tags.contains("full nelson"), line)
            assertTrue(tags.contains("anal fingering"), line)
            assertTrue(tags.contains("1boy") && tags.contains("hetero"), line)
            assertFalse(tags.contains("solo"), line)
        }
    }

    @Test
    fun everyPoseThatNamesAPlaceHasASceneForIt() {
        // The place layer only narrows; a row whose place no scene carries would warn on every draw
        // and fall back to the whole list, so the data has to keep at least one scene per place.
        val named = (matrix.poses + matrix.sfwPoses).filter { PromptGenerator.posePlaces(it).isNotEmpty() }
        assertTrue(named.size >= 40, "rows naming a place: ${named.size}")
        named.forEach { pose ->
            val reachable = matrix.scenes.filter { PromptGenerator.sceneCompatible(pose, it) }
            assertTrue(reachable.isNotEmpty(), "${pose.blob} -> ${PromptGenerator.posePlaces(pose).map { it.marker }}")
        }
    }

    @Test
    fun everyPoseHasACompatibleScene() {
        (matrix.poses + matrix.sfwPoses).forEach { pose ->
            assertTrue(
                matrix.scenes.any { PromptGenerator.sceneCompatible(pose, it) },
                "no scene fits: ${pose.blob} (${PromptGenerator.poseLocus(pose)})",
            )
        }
    }

    @Test
    fun theHandEditedRowsPairWithTheirOwnPlace() {
        val desk = find(matrix.sfwPoses, "resting head on desk")
        assertTrue(PromptGenerator.sceneCompatible(desk, find(matrix.scenes, "classroom, indoors, desk")))
        assertFalse(PromptGenerator.sceneCompatible(desk, find(matrix.scenes, "living room, indoors, sofa")))
        assertFalse(PromptGenerator.sceneCompatible(desk, find(matrix.scenes, "beach, outdoors, ocean, sand")))

        val sofa = find(matrix.sfwPoses, "sitting on sofa, legs tucked")
        assertTrue(PromptGenerator.sceneCompatible(sofa, find(matrix.scenes, "living room, indoors, sofa")))
        assertFalse(PromptGenerator.sceneCompatible(sofa, find(matrix.scenes, "outdoors, street, city")))

        val praying = find(matrix.sfwPoses, "standing, praying, hands together")
        assertTrue(PromptGenerator.sceneCompatible(praying, find(matrix.scenes, "shrine, outdoors, torii")))
        assertFalse(PromptGenerator.sceneCompatible(praying, find(matrix.scenes, "outdoors, park, grass")))

        val railing = find(matrix.sfwPoses, "standing, leaning on railing, looking at viewer")
        assertTrue(PromptGenerator.sceneCompatible(railing, find(matrix.scenes, "rooftop, outdoors, school")))
        assertFalse(PromptGenerator.sceneCompatible(railing, find(matrix.scenes, "bedroom, indoors, bed")))

        val outside = find(matrix.sfwPoses, "profile, sitting, looking outside")
        assertTrue(PromptGenerator.sceneCompatible(outside, find(matrix.scenes, "cafe, indoors, window")))
        assertFalse(PromptGenerator.sceneCompatible(outside, find(matrix.scenes, "onsen, indoors, steam")))
    }

    @Test
    fun generatedSfwPromptsKeepEveryPlaceWithItsScene() {
        val spec = PromptSpec(mode = PromptMode.Sfw, count = 200)
        val warnings = mutableListOf<String>()
        val lines = (1..5).flatMap { PromptGenerator.generate(spec, matrix, seed = 700L + it, warnings) }
        assertEquals(1000, lines.size)
        assertTrue(warnings.isEmpty(), warnings.toString())
        lines.forEach { line ->
            // A row may name two places (`on railing, looking outside`); either one may be the scene.
            val named = POSE_PLACES.filter { line.contains(it.marker) }
            if (named.isNotEmpty()) {
                assertTrue(
                    named.any { place -> place.scenes.any { line.contains(it) } },
                    "${named.map { it.marker }}: $line",
                )
            }
        }
    }
}

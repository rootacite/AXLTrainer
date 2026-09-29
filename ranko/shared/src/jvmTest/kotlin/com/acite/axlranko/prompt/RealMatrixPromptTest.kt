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
        assertTrue(matrix.clothing.all { it.group != null })
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
}

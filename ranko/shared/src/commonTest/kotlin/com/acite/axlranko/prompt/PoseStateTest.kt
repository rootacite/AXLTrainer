package com.acite.axlranko.prompt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * A `QUESTIONABLE_POSES` row states its own clothing and exposure, so the sampler adds no outfit,
 * no `nude` and no `open clothes`, and fills only the half the row leaves open — aggressively
 * ([PromptGenerator.stateFill]). Rows from the other sections keep the old clothes/torso layers.
 */
class PoseStateTest {

    private val matrix = parseMatrix(
        """
        SCENE:
        bedroom, indoors, bed
        outdoors, park, grass
        classroom, indoors, desk
        cafe, indoors, window
        CLOTHING:
        [covered]
        serafuku, white thighhighs (open clothes)
        [revealing]
        bikini, side-tie bikini bottom (open clothes)
        SUFFIX:
        soft lighting
        POSES:
        doggystyle, sex from behind : both
        SFW_POSES:
        sitting, looking at viewer, smile
        lying, on back, on bed, looking at viewer
        QUESTIONABLE_POSES:
        sitting, panties, looking at viewer
        standing, topless, nipples, looking at viewer
        all fours, looking at viewer, panties
        standing, from behind, panties, ass
        standing, from behind, no panties, ass, skirt
        from side, sitting, bra, looking at viewer
        FIGURE:
        petite
        PUSSY_SHAPE:
        labia
        PUSSY_HAIR:
        shaved pussy
        """.trimIndent(),
    )

    private fun pose(needle: String): MatrixEntry = find(matrix.questionablePoses, needle)

    private fun draw(needle: String, count: Int = 6, exposure: List<String> = listOf("revealing", "open", "nude")): String {
        val spec = testSpec(
            mode = PromptMode.Nsfw,
            exposure = exposure,
            poseAny = false,
            poseKeys = setOf(pose(needle).key),
            count = count,
        )
        return generatePrompts(spec, matrix, 7).joinToString("\n")
    }

    @Test
    fun everyRowFromTheSectionIsSelfStated() {
        assertTrue(matrix.questionablePoses.all { it.selfStated })
        assertFalse(matrix.sfwPoses.any { it.selfStated })
    }

    @Test
    fun aRowBringsItsOwnClothes() {
        // The exposure page cannot add an outfit, `nude` or `open clothes` to these rows.
        listOf("revealing", "open", "nude", "covered").forEach { bucket ->
            val blob = draw("sitting, panties", exposure = listOf(bucket))
            assertFalse(blob.contains("serafuku"), "$bucket: $blob")
            assertFalse(blob.contains("bikini"), "$bucket: $blob")
            assertFalse(blob.contains("open clothes"), "$bucket: $blob")
            assertFalse(blob.contains("nude"), "$bucket: $blob")
            assertTrue(blob.contains("panties"), "$bucket: $blob")
        }
    }

    @Test
    fun anUnstatedChestGoesBare() {
        val blob = draw("sitting, panties")
        assertTrue(blob.contains("topless"), blob)
        assertTrue(blob.contains("nipples"), blob)
        assertFalse(blob.contains("bottomless"), blob)
    }

    @Test
    fun aHangingChestIsNamedByItsShape() {
        val blob = draw("all fours, looking at viewer, panties")
        assertTrue(blob.contains("breasts hanging"), blob)
        assertTrue(blob.contains("topless"), blob)
    }

    @Test
    fun aChestThatIsNotInFrameGetsNothing() {
        val blob = draw("standing, from behind, panties, ass")
        assertFalse(blob.contains("topless"), blob)
        assertFalse(blob.contains("nipples"), blob)
        assertFalse(blob.contains("breasts"), blob)
        assertTrue(blob.contains("panties"), blob)
    }

    @Test
    fun aStatedChestIsLeftAlone() {
        val blob = draw("standing, topless, nipples, looking at viewer")
        assertTrue(blob.contains("topless"), blob)
        assertTrue(blob.contains("nipples"), blob)
        assertTrue(blob.contains("bottomless"), blob)
    }

    @Test
    fun aStatedBottomIsNotOverwritten() {
        val blob = draw("standing, from behind, no panties, ass, skirt")
        assertFalse(blob.contains("bottomless"), blob)
        assertTrue(blob.contains("skirt"), blob)
    }

    @Test
    fun aSideViewGetsTheSideOfTheChest() {
        val blob = draw("from side, sitting, bra, looking at viewer")
        assertTrue(blob.contains("bra"), blob)
        assertTrue(blob.contains("bottomless"), blob)
        assertFalse(blob.contains("topless"), blob)
    }

    @Test
    fun theSfwPosesKeepTheOldClothesAndTorsoLayers() {
        val spec = testSpec(
            mode = PromptMode.Nsfw,
            exposure = listOf("revealing"),
            poseAny = false,
            poseKeys = setOf(find(matrix.sfwPoses, "sitting, looking at viewer").key),
            count = 6,
        )
        val blob = generatePrompts(spec, matrix, 7).joinToString("\n")
        assertTrue(blob.contains("bikini"), blob)
        assertFalse(blob.contains("bottomless"), blob)
        assertFalse(blob.contains("topless"), blob)
    }
}

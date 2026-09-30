package com.acite.axlranko.prompt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * A pose that names its own place decides the scene, ahead of [PromptGenerator.poseLocus]. The
 * fixture is local because `MINI_MATRIX` mirrors the Python CLI's, and the CLI knows no such rule.
 */
class PosePlaceSceneTest {

    private val matrix = parseMatrix(
        """
        SCENE:
        bedroom, indoors, bed
        living room, indoors, sofa
        outdoors, park, grass
        outdoors, park, bench
        beach, outdoors, ocean, sand
        classroom, indoors, desk
        cafe, indoors, window
        rooftop, outdoors, school
        shrine, outdoors, torii
        onsen, indoors, steam
        CLOTHING:
        [covered]
        serafuku (open clothes)
        SUFFIX:
        soft lighting
        POSES:
        doggystyle, sex from behind : both
        SFW_POSES:
        sitting on sand, looking at viewer
        sitting, resting head on desk, sleeping
        standing, leaning on railing, looking outside
        against tree, sitting, looking at viewer
        lying, on back, looking at viewer
        standing, praying, hands together
        sitting, looking at viewer
        QUESTIONABLE_POSES:
        bent over, railing, from behind, panties, ass
        bent over, desk, skirt lift, panties, from behind
        FIGURE:
        petite
        PUSSY_SHAPE:
        labia
        PUSSY_HAIR:
        shaved pussy
        """.trimIndent(),
    )

    private fun pose(needle: String): MatrixEntry = find(matrix.sfwPoses, needle)

    private fun questionable(needle: String): MatrixEntry = find(matrix.questionablePoses, needle)

    private fun scene(needle: String): MatrixEntry = find(matrix.scenes, needle)

    @Test
    fun aNamedSurfaceDecidesTheScene() {
        val sand = pose("sitting on sand")
        assertTrue(PromptGenerator.sceneCompatible(sand, scene("beach")))
        listOf("bedroom", "sofa", "grass", "bench", "classroom", "cafe", "rooftop", "shrine", "onsen")
            .forEach { assertFalse(PromptGenerator.sceneCompatible(sand, scene(it)), it) }
    }

    @Test
    fun aNamedSurfaceBeatsTheLocus() {
        val desk = pose("resting head on desk")
        // The locus misfires on `sleeping`; the row's own word is what decides.
        assertEquals("lie", PromptGenerator.poseLocus(desk))
        assertTrue(PromptGenerator.sceneCompatible(desk, scene("classroom")))
        assertTrue(PromptGenerator.sceneCompatible(desk, scene("cafe")))
        listOf("bedroom", "sofa", "beach", "bench", "grass", "rooftop").forEach {
            assertFalse(PromptGenerator.sceneCompatible(desk, scene(it)), it)
        }
    }

    @Test
    fun twoNamedPlacesTakeEither() {
        val railing = pose("on railing")
        assertTrue(PromptGenerator.sceneCompatible(railing, scene("rooftop")))
        assertTrue(PromptGenerator.sceneCompatible(railing, scene("cafe")))
        listOf("bedroom", "sofa", "beach", "grass", "onsen").forEach {
            assertFalse(PromptGenerator.sceneCompatible(railing, scene(it)), it)
        }
    }

    @Test
    fun aTreeRowTakesAnyParkOrForestScene() {
        val tree = pose("against tree")
        assertTrue(PromptGenerator.sceneCompatible(tree, scene("outdoors, park, bench")))
        assertTrue(PromptGenerator.sceneCompatible(tree, scene("outdoors, park, grass")))
        assertFalse(PromptGenerator.sceneCompatible(tree, scene("rooftop")))
        assertFalse(PromptGenerator.sceneCompatible(tree, scene("bedroom")))
    }

    @Test
    fun aShrineRowTakesOnlyTheShrine() {
        val praying = pose("standing, praying")
        assertTrue(PromptGenerator.sceneCompatible(praying, scene("shrine")))
        listOf("rooftop", "cafe", "bedroom", "beach", "park", "onsen").forEach {
            assertFalse(PromptGenerator.sceneCompatible(praying, scene(it)), it)
        }
    }

    @Test
    fun aPoseWithNoPlaceStillFollowsItsLocus() {
        val lying = pose("lying, on back, looking at viewer")
        assertTrue(PromptGenerator.posePlaces(lying).isEmpty())
        assertEquals("lie", PromptGenerator.poseLocus(lying))
        assertFalse(PromptGenerator.sceneCompatible(lying, scene("cafe")))
        assertFalse(PromptGenerator.sceneCompatible(lying, scene("rooftop")))
        assertTrue(PromptGenerator.sceneCompatible(lying, scene("bedroom")))
        assertTrue(PromptGenerator.sceneCompatible(lying, scene("bench")))
    }

    @Test
    fun aPlaceNoSceneCarriesWarnsAndFallsBack() {
        val scenes = parseMatrix(
            """
            SCENE:
            bedroom, indoors, bed
            CLOTHING:
            [covered]
            serafuku (open clothes)
            SUFFIX:
            soft lighting
            POSES:
            doggystyle, sex from behind : both
            SFW_POSES:
            sitting on sand, looking at viewer
            QUESTIONABLE_POSES:
            sitting, panties, looking at viewer
            FIGURE:
            petite
            PUSSY_SHAPE:
            labia
            PUSSY_HAIR:
            shaved pussy
            """.trimIndent(),
        )
        val spec = testSpec(poseAny = false, poseKeys = setOf(pose("sitting on sand").key), count = 4)
        val warnings = mutableListOf<String>()
        val lines = PromptGenerator.generate(spec, scenes, 21, warnings)
        assertEquals(4, lines.size)
        assertTrue(warnings.any { it.contains("no compatible scene") }, warnings.toString())
        lines.forEach { assertTrue(it.contains("bedroom"), it) }
    }

    @Test
    fun aBarePlaceWordConstrainsTheSceneToo() {
        // `bent over, railing…` names no `on railing`, and `bent over, desk…` no `on desk`.
        val railing = questionable("railing")
        assertTrue(PromptGenerator.sceneCompatible(railing, scene("rooftop")))
        listOf("bedroom", "sofa", "grass", "beach", "classroom", "cafe", "onsen").forEach {
            assertFalse(PromptGenerator.sceneCompatible(railing, scene(it)), it)
        }
        val desk = questionable("desk")
        assertTrue(PromptGenerator.sceneCompatible(desk, scene("classroom")))
        assertTrue(PromptGenerator.sceneCompatible(desk, scene("cafe")))
        listOf("bedroom", "rooftop", "beach", "onsen").forEach {
            assertFalse(PromptGenerator.sceneCompatible(desk, scene(it)), it)
        }
    }

    @Test
    fun aGeneratedPlanKeepsEveryPlaceWithItsScene() {
        val spec = testSpec(count = 200)
        val warnings = mutableListOf<String>()
        val lines = (1..5).flatMap { PromptGenerator.generate(spec, matrix, 300L + it, warnings) }
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

package com.acite.axlranko.prompt

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** The same small matrix the Python suite uses, so the ported expectations stay comparable. */
internal const val MINI_MATRIX = """POSES:
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
QUESTIONABLE_POSES:
standing, topless, nipples, looking at viewer
# topless
sitting, panties, looking at viewer
# panties sit
FIGURE:
petite
# petite
narrow waist, wide hips, thick thighs
# hourglass
PUSSY_SHAPE:
cleft of venus
# cleft
labia, long labia
# labia
PUSSY_HAIR:
pubic hair
# hair
shaved pussy
# shaved
"""

internal fun miniMatrix(): PromptMatrix = parseMatrix(MINI_MATRIX)

/** The whole-config helper the Python suite calls `_spec`. */
internal fun testSpec(
    character: String = "(sena_character:1.1), 1girl",
    mode: PromptMode = PromptMode.Sfw,
    exposure: List<String> = listOf("covered", "casual"),
    clothingAny: Boolean = true,
    clothingKeys: Set<List<String>> = emptySet(),
    sceneAny: Boolean = true,
    sceneKeys: Set<List<String>> = emptySet(),
    poseAny: Boolean = true,
    poseKeys: Set<List<String>> = emptySet(),
    familyAny: Boolean = true,
    families: Set<PoseFamily> = emptySet(),
    vaginalRatio: Double = PromptLimits.VAGINAL_RATIO_DEFAULT,
    stageWeights: Map<SexStage, Double> = defaultStageWeights(),
    chest: String = "auto",
    belly: String = "auto",
    figure: List<String> = emptyList(),
    pussyShape: List<String> = emptyList(),
    pussyHair: List<String> = emptyList(),
    face: Map<String, FacePick> = defaultFace(),
    count: Int = 8,
): PromptSpec = PromptSpec(
    character = character,
    mode = mode,
    exposure = exposure,
    clothingAny = clothingAny,
    clothingKeys = clothingKeys,
    sceneAny = sceneAny,
    sceneKeys = sceneKeys,
    poseAny = poseAny,
    poseKeys = poseKeys,
    familyAny = familyAny,
    families = families,
    vaginalRatio = vaginalRatio,
    stageWeights = stageWeights,
    chest = chest,
    belly = belly,
    figure = figure,
    pussyShape = pussyShape,
    pussyHair = pussyHair,
    face = face,
    count = count,
)

/** Group values from the defaults; a bare name means a one-tag candidate list. */
internal fun testFace(vararg overrides: Pair<String, FacePick>): Map<String, FacePick> {
    val face = LinkedHashMap(defaultFace())
    overrides.forEach { (key, value) -> face[key] = value }
    return face
}

internal fun tagsOf(line: String): Set<String> = splitTags(line).toSet()

/**
 * The anal channel word is only ever written weighted (`(anal:1.2)`), so "no anal here" means
 * neither form is in the line — the bare token is what the weight replaced.
 */
internal fun assertNoAnalChannel(tags: Set<String>, message: String) {
    assertFalse(tags.contains("anal"), message)
    assertFalse(tags.contains(PromptLimits.ANAL_CHANNEL_TAG), message)
}

/** Words only a `QUESTIONABLE_POSES` row, or the fill it gets, can put into a prompt. */
internal fun assertNoQuestionableWords(blob: String) {
    listOf("panties", "topless", "bottomless", "sideboob", "skirt lift", "undressing").forEach { word ->
        assertFalse(blob.contains(word), "$word in $blob")
    }
}

internal fun find(entries: List<MatrixEntry>, needle: String): MatrixEntry =
    entries.firstOrNull { it.blob.contains(needle) }
        ?: throw AssertionError("no entry containing '$needle'")

internal fun generatePrompts(
    spec: PromptSpec,
    matrix: PromptMatrix,
    seed: Long? = null,
): List<String> = PromptGenerator.generate(spec, matrix, seed)

private fun groupsHit(line: String): Map<String, Int> {
    val counts = mutableMapOf<String, Int>()
    splitTags(line).forEach { tag ->
        FACE_TAG_GROUP[tag]?.let { group -> counts[group] = (counts[group] ?: 0) + 1 }
    }
    return counts
}

class ParseMatrixTest {
    private val matrix = miniMatrix()

    @Test
    fun everySectionGroupAndCommentIsRead() {
        assertEquals(5, matrix.poses.size)
        assertEquals(4, matrix.clothing.size)
        assertEquals(6, matrix.scenes.size)
        assertEquals(1, matrix.suffixes.size)
        assertEquals(6, matrix.sfwPoses.size)
        assertEquals(2, matrix.questionablePoses.size)
        assertEquals("mating press comment", matrix.poses[0].comment)
        assertEquals(PromptChannel.Both, matrix.poses[0].channel)
        assertEquals(PromptChannel.Anal, find(matrix.poses, "full nelson").channel)
        assertEquals("covered", find(matrix.clothing, "serafuku").group)
        assertTrue(find(matrix.clothing, "serafuku").openClothes)
        assertFalse(find(matrix.clothing, "wedding dress").openClothes)
        assertEquals(listOf("wedding dress", "veil", "white gloves"), find(matrix.clothing, "wedding dress").tags)
    }

    @Test
    fun onlyTheQuestionableRowsStateTheirOwnBody() {
        assertEquals("topless", matrix.questionablePoses[0].comment)
        assertTrue(matrix.questionablePoses.all { it.selfStated })
        assertFalse(matrix.sfwPoses.any { it.selfStated })
        assertFalse(matrix.poses.any { it.selfStated })
        assertFalse(matrix.clothing.any { it.selfStated })
    }

    @Test
    fun anUnknownSectionIsRejected() {
        // An unrecognized header used to become a tag row of the section above it, taking every
        // row under it along into that pool.
        val broken = MINI_MATRIX + "PROPS:\nwall clock\n"
        val error = assertFailsWith<MatrixException> { parseMatrix(broken) }
        assertTrue(error.message!!.contains("unknown section PROPS"), error.message!!)
    }

    @Test
    fun poseWithoutAChannelIsRejected() {
        val broken = MINI_MATRIX.replace("full nelson : anal only", "full nelson")
        val error = assertFailsWith<MatrixException> { parseMatrix(broken) }
        assertTrue(error.message!!.contains("both|anal only|vaginal only|none"), error.message!!)
    }

    @Test
    fun clothingRowBeforeAGroupIsRejected() {
        val broken = MINI_MATRIX.replace("[covered]\n", "")
        assertFailsWith<MatrixException> { parseMatrix(broken) }
    }

    @Test
    fun aMissingSectionIsRejected() {
        val broken = MINI_MATRIX.substringBefore("SUFFIX:") + "SUFFIX:\n"
        assertFailsWith<MatrixException> { parseMatrix(broken) }
    }
}

class ModePoolTest {
    private val matrix = miniMatrix()

    @Test
    fun sexNeverDrawsSfwPoses() {
        val prompts = generatePrompts(testSpec(mode = PromptMode.Sex, exposure = listOf("open"), count = 20), matrix, 1)
        prompts.forEach { line ->
            val tags = tagsOf(line)
            assertFalse(tags.contains("chin rest"), line)
            assertFalse(tags.contains("peace sign"), line)
            assertTrue(tags.contains("sex"), line)
            assertTrue(tags.contains("penis"), line)
        }
        assertNoQuestionableWords(prompts.joinToString("\n"))
    }

    @Test
    fun sfwNeverDrawsSexPoses() {
        val blob = generatePrompts(testSpec(mode = PromptMode.Sfw, count = 20), matrix, 2).joinToString("\n")
        listOf("doggystyle", "mating press", "penis", "pussy", "anus", "sex").forEach { needle ->
            assertFalse(blob.contains(needle), "$needle in $blob")
        }
        assertNoQuestionableWords(blob)
    }

    @Test
    fun nsfwDrawsTheSfwPosesAndTheQuestionableOnesOnly() {
        val pool = PromptGenerator.posePool(matrix, PromptMode.Nsfw)
        assertEquals(8, pool.size)
        assertEquals(matrix.sfwPoses + matrix.questionablePoses, pool)
        val blob = generatePrompts(testSpec(mode = PromptMode.Nsfw, exposure = listOf("nude"), count = 40), matrix, 3)
            .joinToString("\n")
        assertTrue(blob.contains("panties") || blob.contains("topless"), blob)
        listOf("doggystyle", "mating press", "penis", "pussy", "anus", "sex").forEach { needle ->
            assertFalse(blob.contains(needle), "$needle in $blob")
        }
    }
}

class PrefixAndForbiddenTagTest {
    private val matrix = miniMatrix()

    @Test
    fun prefixIsFirstAndUntouched() {
        generatePrompts(testSpec(count = 5), matrix, 4).forEach { line ->
            assertTrue(line.startsWith("(sena_character:1.1), 1girl"), line)
        }
    }

    @Test
    fun ratingAndQualityTagsNeverAppear() {
        val prompts = generatePrompts(
            testSpec(mode = PromptMode.Sex, exposure = listOf("open"), count = 15),
            matrix,
            5,
        )
        prompts.forEach { line ->
            val tags = tagsOf(line)
            listOf("nsfw", "sfw", "explicit", "masterpiece", "best quality", "newest").forEach { banned ->
                assertFalse(tags.contains(banned), "$banned in $line")
            }
        }
    }

    @Test
    fun forbiddenHelperKnowsTheFamilies() {
        assertTrue(isForbiddenTag("nsfw"))
        assertTrue(isForbiddenTag("masterpiece"))
        assertTrue(isForbiddenTag("year 2024"))
        assertTrue(isForbiddenTag("score_9"))
        assertFalse(isForbiddenTag("1girl"))
    }
}

class ChannelAndAnatomyTest {
    private val matrix = miniMatrix()

    @Test
    fun analOnlyNeverWritesVaginal() {
        val nelson = find(matrix.poses, "full nelson")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(nelson.key),
            count = 8,
        )
        generatePrompts(spec, matrix, 6).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains(PromptLimits.ANAL_CHANNEL_TAG), line)
            assertFalse(tags.contains("anal"), line)
            assertFalse(tags.contains("vaginal"), line)
            assertTrue(tags.contains("penis"), line)
            assertTrue(tags.contains("anus"), line)
            assertTrue(tags.contains("ass"), line)
        }
    }

    @Test
    fun aHandStageOnAHeldLegsPoseGetsAPartner() {
        // `anal fingering` beside `full nelson` asks for a hand the pose has no arm for, and the
        // model answers with a third one: the draw gets a partner instead of claiming she is alone.
        val nelson = find(matrix.poses, "full nelson")
        assertTrue(PromptGenerator.poseHoldsLegs(nelson))
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(nelson.key),
            stageWeights = defaultStageWeights() + mapOf(
                SexStage.During to 0.0,
                SexStage.Fingering to 1.0,
            ),
            count = 6,
        )
        generatePrompts(spec, matrix, 51).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("anal fingering"), line)
            assertTrue(tags.contains("1boy"), line)
            assertTrue(tags.contains("hetero"), line)
            assertFalse(tags.contains("solo"), line)
            // The stage still means no penis: a partner holding the legs, not a penetration.
            assertFalse(tags.contains("penis"), line)
            assertFalse(tags.contains("sex"), line)
        }
    }

    @Test
    fun aHandStageOnAFreeLegsPoseStaysSolo() {
        // `mating press` holds the legs, `doggystyle` does not: the rule must not spread to the
        // poses whose hands are free, or every fingering draw would grow a partner.
        val dog = find(matrix.poses, "doggystyle")
        assertFalse(PromptGenerator.poseHoldsLegs(dog))
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(dog.key),
            stageWeights = defaultStageWeights() + mapOf(
                SexStage.During to 0.0,
                SexStage.Fingering to 1.0,
            ),
            count = 6,
        )
        generatePrompts(spec, matrix, 52).forEach { line ->
            val tags = tagsOf(line)
            // The channel is drawn per prompt, so this is either word; neither may bring a partner.
            assertTrue(tags.contains("fingering") || tags.contains("anal fingering"), line)
            assertTrue(tags.contains("solo"), line)
            assertFalse(tags.contains("1boy"), line)
            assertFalse(tags.contains("hetero"), line)
        }
    }

    @Test
    fun theHeldLegsMarkersPickTheFoldedPosesOnly() {
        assertTrue(PromptGenerator.poseHoldsLegs(find(matrix.poses, "mating press")))
        assertTrue(PromptGenerator.poseHoldsLegs(find(matrix.poses, "full nelson")))
        assertFalse(PromptGenerator.poseHoldsLegs(find(matrix.poses, "cowgirl")))
    }

    @Test
    fun theHeldLegsRuleLeavesTheOtherStagesAlone() {
        // Object insertion is the same shape of problem but not what was asked for; it still draws
        // solo, so this test fails loudly if someone extends HAND_ONLY_STAGES without meaning to.
        val nelson = find(matrix.poses, "full nelson")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(nelson.key),
            stageWeights = defaultStageWeights() + mapOf(
                SexStage.During to 0.0,
                SexStage.ObjectInsertion to 1.0,
            ),
            count = 4,
        )
        generatePrompts(spec, matrix, 53).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("anal object insertion"), line)
            assertTrue(tags.contains("solo"), line)
            assertFalse(tags.contains("1boy"), line)
        }
    }

    @Test
    fun theAnalChannelIsWrittenWeighted() {
        assertEquals("(anal:1.2)", PromptGenerator.channelTag(PromptChannel.Anal))
        assertEquals("vaginal", PromptGenerator.channelTag(PromptChannel.Vaginal))
        // The weight is what keeps the element from being dropped, so it may not be filtered out.
        assertFalse(isForbiddenTag(PromptLimits.ANAL_CHANNEL_TAG))

        val anal = tagsOf(assembleLine(SexStage.During, PromptChannel.Anal))
        assertTrue(anal.contains(PromptLimits.ANAL_CHANNEL_TAG), anal.toString())
        assertFalse(anal.contains("anal"), anal.toString())

        val vaginal = tagsOf(assembleLine(SexStage.During, PromptChannel.Vaginal))
        assertTrue(vaginal.contains("vaginal"), vaginal.toString())
        assertFalse(vaginal.any { it.contains("anal") }, vaginal.toString())
    }

    @Test
    fun doggystyleGetsAss() {
        val dog = find(matrix.poses, "doggystyle")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(dog.key),
            vaginalRatio = 0.0,
            count = 5,
        )
        generatePrompts(spec, matrix, 7).forEach { line ->
            assertTrue(tagsOf(line).contains("ass"), line)
            assertTrue(tagsOf(line).contains("penis"), line)
        }
    }

    @Test
    fun matingPressVaginalHasPussy() {
        val press = find(matrix.poses, "mating press")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(press.key),
            vaginalRatio = 1.0,
            count = 5,
        )
        generatePrompts(spec, matrix, 8).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("pussy"), line)
            assertTrue(tags.contains("penis"), line)
            assertFalse(tags.contains("ass"), line)
        }
    }

    @Test
    fun standingMissionaryHasNoAss() {
        val pose = find(matrix.poses, "standing missionary")
        val tags = PromptGenerator.anatomyTags(pose, PromptChannel.Vaginal)
        assertTrue(tags.contains("penis"))
        assertTrue(tags.contains("pussy"))
        assertFalse(tags.contains("ass"))
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(pose.key),
            vaginalRatio = 1.0,
            count = 6,
        )
        generatePrompts(spec, matrix, 9).forEach { line ->
            assertFalse(tagsOf(line).contains("ass"), line)
        }
    }

    @Test
    fun anatomyCanOmitPenis() {
        val pose = find(matrix.poses, "standing missionary")
        assertTrue(PromptGenerator.anatomyTags(pose, PromptChannel.Vaginal).contains("penis"))
        assertFalse(
            PromptGenerator.anatomyTags(pose, PromptChannel.Vaginal, includePenis = false).contains("penis"),
        )
    }
}

/**
 * The three single-pick matrix groups: one row each, or none. A figure is part of the body in any
 * mode; the two pussy groups are SEX only and ride a prompt that already says `pussy`.
 */
class SinglePickGroupTest {
    private val matrix = miniMatrix()

    private val petite = find(matrix.figure, "petite")
    private val hourglass = find(matrix.figure, "narrow waist")
    private val labia = find(matrix.pussyShape, "labia")
    private val shaved = find(matrix.pussyHair, "shaved")

    @Test
    fun theThreeSectionsParseIntoTheirOwnPools() {
        assertEquals(listOf("petite", "narrow waist, wide hips, thick thighs"), matrix.figure.map { it.blob })
        assertEquals(listOf("cleft of venus", "labia, long labia"), matrix.pussyShape.map { it.blob })
        assertEquals(listOf("pubic hair", "shaved pussy"), matrix.pussyHair.map { it.blob })
    }

    @Test
    fun aFigureRidesEveryMode() {
        // The figure page is not mode-gated and neither is the word: SFW, NSFW and SEX all write it.
        listOf(PromptMode.Sfw, PromptMode.Nsfw, PromptMode.Sex).forEach { mode ->
            val spec = testSpec(
                mode = mode,
                exposure = defaultExposure(mode),
                figure = petite.key,
                count = 12,
            )
            val lines = generatePrompts(spec, matrix, 61)
            assertEquals(12, lines.size)
            lines.forEach { line -> assertTrue(tagsOf(line).contains("petite"), "$mode: $line") }
        }
    }

    @Test
    fun aRowBringsAllOfItsTags() {
        val spec = testSpec(figure = hourglass.key, count = 6)
        generatePrompts(spec, matrix, 63).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.containsAll(hourglass.tags), line)
            assertFalse(tags.contains("petite"), line)
        }
    }

    @Test
    fun everyGroupOffWritesNoWordOfIt() {
        // The state every profile written before these sections loads in: no row of any of the three
        // may reach a prompt by itself.
        val rows = matrix.figure + matrix.pussyShape + matrix.pussyHair
        listOf(PromptMode.Sfw, PromptMode.Nsfw, PromptMode.Sex).forEach { mode ->
            val blob = generatePrompts(
                testSpec(mode = mode, exposure = defaultExposure(mode), count = 20),
                matrix,
                62,
            ).joinToString("\n")
            rows.forEach { row ->
                row.tags.forEach { tag -> assertFalse(blob.contains(tag), "$mode $tag") }
            }
        }
    }

    @Test
    fun aShapeWordRidesOnlyAPussyDraw() {
        // The vaginal channel writes `pussy`; an anal draw writes `anus` instead (unless the pose
        // spreads, which puts `pussy` back). The word follows that, nothing else.
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            pussyShape = labia.key,
            count = 24,
        )
        val lines = generatePrompts(spec, matrix, 64)
        assertTrue(lines.any { tagsOf(it).contains("pussy") }, lines.toString())
        lines.forEach { line ->
            val tags = tagsOf(line)
            assertEquals(tags.contains("pussy"), tags.contains("labia"), line)
        }
    }

    @Test
    fun anAnalDrawNeverGetsThePussyWords() {
        val nelson = find(matrix.poses, "full nelson")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(nelson.key),
            pussyShape = labia.key,
            pussyHair = shaved.key,
            count = 8,
        )
        generatePrompts(spec, matrix, 65).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("anus"), line)
            assertFalse(tags.contains("labia"), line)
            assertFalse(tags.contains("shaved pussy"), line)
        }
    }

    @Test
    fun aNonSexDrawNeverGetsThem() {
        // A profile can carry the picks and be switched to another mode; the generator is what keeps
        // the SEX-only words out of those prompts.
        listOf(PromptMode.Sfw, PromptMode.Nsfw).forEach { mode ->
            val blob = generatePrompts(
                testSpec(
                    mode = mode,
                    exposure = defaultExposure(mode),
                    pussyShape = labia.key,
                    pussyHair = find(matrix.pussyHair, "pubic hair").key,
                    count = 16,
                ),
                matrix,
                66,
            ).joinToString("\n")
            assertFalse(blob.contains("labia"), "$mode: $blob")
            assertFalse(blob.contains("pubic hair"), "$mode: $blob")
        }
    }

    @Test
    fun switchingModeClearsTheSexOnlyPicks() {
        val spec = testSpec(mode = PromptMode.Sex, pussyShape = labia.key, pussyHair = shaved.key, figure = petite.key)
        WizardModel.applyModeChange(spec, PromptMode.Nsfw)
        assertTrue(spec.pussyShape.isEmpty())
        assertTrue(spec.pussyHair.isEmpty())
        // The figure is not mode-specific, so the reset leaves it alone.
        assertEquals(petite.key, spec.figure)
    }
}

class CompatibilityTest {
    private val matrix = miniMatrix()

    @Test
    fun lieRejectsHallwayAndAcceptsBed() {
        val pose = find(matrix.sfwPoses, "lying, on back")
        assertFalse(PromptGenerator.sceneCompatible(pose, find(matrix.scenes, "hallway")))
        assertTrue(PromptGenerator.sceneCompatible(pose, find(matrix.scenes, "on bed")))
        assertEquals("lie", PromptGenerator.poseLocus(pose))
    }

    @Test
    fun walkRejectsOnBedAndAcceptsBeach() {
        val pose = find(matrix.sfwPoses, "walking")
        assertFalse(PromptGenerator.sceneCompatible(pose, find(matrix.scenes, "on bed")))
        assertTrue(PromptGenerator.sceneCompatible(pose, find(matrix.scenes, "beach")))
    }

    @Test
    fun windowRejectsBeach() {
        val pose = find(matrix.sfwPoses, "looking outside")
        assertFalse(PromptGenerator.sceneCompatible(pose, find(matrix.scenes, "beach")))
        assertTrue(PromptGenerator.sceneCompatible(pose, find(matrix.scenes, "cafe")))
    }

    @Test
    fun generatedWalkIsNeverOnBed() {
        val pose = find(matrix.sfwPoses, "walking")
        val spec = testSpec(poseAny = false, poseKeys = setOf(pose.key), count = 12)
        generatePrompts(spec, matrix, 10).forEach { line ->
            assertFalse(line.contains("on bed"), line)
        }
    }
}

class WarningAndSeedTest {
    private val matrix = miniMatrix()

    @Test
    fun sfwHighExposureIsFlagged() {
        assertTrue(WizardModel.needsSfwExposureWarning(testSpec(mode = PromptMode.Sfw, exposure = listOf("nude"))))
        assertTrue(
            WizardModel.needsSfwExposureWarning(testSpec(mode = PromptMode.Sfw, exposure = listOf("covered", "open"))),
        )
        assertFalse(
            WizardModel.needsSfwExposureWarning(testSpec(mode = PromptMode.Sfw, exposure = listOf("covered", "casual"))),
        )
        assertFalse(WizardModel.needsSfwExposureWarning(testSpec(mode = PromptMode.Nsfw, exposure = listOf("nude"))))
    }

    @Test
    fun theSameSeedGivesTheSamePrompts() {
        val spec = testSpec(mode = PromptMode.Sex, exposure = listOf("open"), count = 6)
        assertEquals(generatePrompts(spec, matrix, 0), generatePrompts(spec, matrix, 0))
    }

    @Test
    fun theOpenBucketWritesOpenClothes() {
        val spec = testSpec(mode = PromptMode.Sex, exposure = listOf("open"), count = 6)
        generatePrompts(spec, matrix, 11).forEach { line ->
            assertTrue(tagsOf(line).contains("open clothes"), line)
        }
    }

    @Test
    fun theCoveredBucketSkipsOpenClothes() {
        val spec = testSpec(mode = PromptMode.Sfw, exposure = listOf("covered"), count = 8)
        generatePrompts(spec, matrix, 12).forEach { line ->
            assertFalse(tagsOf(line).contains("open clothes"), line)
        }
    }

    @Test
    fun countOutsideTheRangeIsRefused() {
        assertFailsWith<IllegalArgumentException> {
            generatePrompts(testSpec(count = 0), matrix, 1)
        }
        assertFailsWith<IllegalArgumentException> {
            generatePrompts(testSpec(count = PromptLimits.COUNT_MAX + 1), matrix, 1)
        }
    }
}

class TorsoTagTest {
    private val clothes = MatrixEntry(
        tags = listOf("serafuku", "white thighhighs"),
        openClothes = true,
        group = "covered",
    )
    private val rng = Random(0)

    private fun spec(
        mode: PromptMode = PromptMode.Sex,
        exposure: List<String> = listOf("open"),
        chest: String = "auto",
        belly: String = "covered",
    ) = testSpec(mode = mode, exposure = exposure, chest = chest, belly = belly)

    @Test
    fun chestViewFrontOptionalAndBack() {
        assertEquals(
            "front",
            PromptGenerator.chestView(MatrixEntry(listOf("missionary", "on back", "from front"), channel = PromptChannel.Both)),
        )
        assertEquals(
            "front",
            PromptGenerator.chestView(MatrixEntry(listOf("suspended congress", "held up"), channel = PromptChannel.Both)),
        )
        assertEquals(
            "front",
            PromptGenerator.chestView(
                MatrixEntry(listOf("cowgirl position", "girl on top", "looking at viewer"), channel = PromptChannel.Both),
            ),
        )
        assertEquals(
            "optional",
            PromptGenerator.chestView(MatrixEntry(listOf("doggystyle", "all fours", "from behind"), channel = PromptChannel.Both)),
        )
        assertEquals(
            "optional",
            PromptGenerator.chestView(MatrixEntry(listOf("prone bone", "on stomach"), channel = PromptChannel.Both)),
        )
    }

    @Test
    fun frontWithOpenClothesAlwaysPullsBreastsOut() {
        val pose = MatrixEntry(listOf("missionary", "on back", "from front"), channel = PromptChannel.Both)
        val tags = PromptGenerator.torsoTags(spec(), pose, clothes, true, rng)
        assertTrue(tags.contains("breasts"))
        assertTrue(tags.contains("breasts out"))
        assertTrue(tags.contains("nipples"))
        val cowgirl = MatrixEntry(
            listOf("cowgirl position", "girl on top", "looking at viewer"),
            channel = PromptChannel.Both,
        )
        val cowgirlTags = PromptGenerator.torsoTags(spec(), cowgirl, clothes, true, rng)
        assertTrue(cowgirlTags.contains("breasts out"))
        assertTrue(cowgirlTags.contains("nipples"))
    }

    @Test
    fun doggystyleWithCoveredPreferenceSkipsTheChest() {
        val pose = MatrixEntry(listOf("doggystyle", "all fours", "from behind"), channel = PromptChannel.Both)
        val tags = PromptGenerator.torsoTags(spec(chest = "covered"), pose, clothes, true, rng)
        assertFalse(tags.contains("breasts out"))
        assertFalse(tags.contains("nipples"))
    }

    @Test
    fun doggystyleWithNipplePreferenceAddsIt() {
        val pose = MatrixEntry(listOf("doggystyle", "all fours", "from behind"), channel = PromptChannel.Both)
        val tags = PromptGenerator.torsoTags(spec(chest = "nipples"), pose, clothes, true, rng)
        assertTrue(tags.contains("breasts out"))
        assertTrue(tags.contains("nipples"))
    }

    @Test
    fun nudityWritesBreastsWithoutBreastsOut() {
        val pose = MatrixEntry(listOf("sitting", "looking at viewer"))
        val tags = PromptGenerator.torsoTags(
            spec(mode = PromptMode.Nsfw, exposure = listOf("nude"), chest = "auto"),
            pose,
            null,
            false,
            rng,
        )
        assertTrue(tags.contains("breasts"))
        assertTrue(tags.contains("nipples"))
        assertFalse(tags.contains("breasts out"))
    }

    @Test
    fun sfwWritesNoChestTags() {
        val pose = MatrixEntry(listOf("sitting", "looking at viewer"))
        val tags = PromptGenerator.torsoTags(
            spec(mode = PromptMode.Sfw, exposure = listOf("covered"), chest = "auto", belly = "auto"),
            pose,
            clothes,
            false,
            rng,
        )
        assertEquals(emptyList(), tags)
    }

    @Test
    fun theNavelPreferenceAddsMidriff() {
        val pose = MatrixEntry(listOf("missionary", "from front"), channel = PromptChannel.Both)
        val tags = PromptGenerator.torsoTags(spec(chest = "covered", belly = "navel"), pose, clothes, true, rng)
        assertTrue(tags.contains("navel"))
        assertTrue(tags.contains("midriff"))
    }

    @Test
    fun torsoComesBeforeThePoseAndFaceBetweenThem() {
        val pose = MatrixEntry(listOf("missionary", "from front"), channel = PromptChannel.Both)
        val scene = MatrixEntry(listOf("bedroom", "indoors", "bed"))
        val suffix = MatrixEntry(listOf("soft lighting"))
        val line = PromptGenerator.assemble(
            spec(), pose, clothes, true, scene, suffix, PromptChannel.Vaginal,
            torso = listOf("breasts", "breasts out"),
            face = listOf("smile", "half-closed eyes"),
        )
        val tags = splitTags(line)
        assertTrue(tags.indexOf("open clothes") < tags.indexOf("breasts out"), line)
        assertTrue(tags.indexOf("breasts out") < tags.indexOf("smile"), line)
        assertTrue(tags.indexOf("half-closed eyes") < tags.indexOf("missionary"), line)
    }
}

/** The slot order of a full sex line, with the stage picked explicitly. */
private fun assembleLine(stage: SexStage, channel: PromptChannel = PromptChannel.Vaginal): String {
    val spec = testSpec(mode = PromptMode.Sex, exposure = listOf("open"))
    val pose = MatrixEntry(listOf("doggystyle", "sex from behind"), channel = PromptChannel.Both)
    val clothes = MatrixEntry(listOf("serafuku", "white thighhighs"), openClothes = true, group = "covered")
    val scene = MatrixEntry(listOf("bedroom", "indoors", "bed"))
    val suffix = MatrixEntry(listOf("soft lighting"))
    return PromptGenerator.assemble(spec, pose, clothes, true, scene, suffix, channel, stage = stage)
}

class SexStageTest {
    private val matrix = miniMatrix()

    @Test
    fun defaultWeightsAreDuringOnly() {
        val weights = defaultStageWeights()
        assertEquals(SEX_STAGES, weights.keys.toList())
        assertEquals(1.0, weights.getValue(SexStage.During))
        assertTrue(SEX_STAGES.filter { it != SexStage.During }.all { weights.getValue(it) == 0.0 })
        listOf(PromptLang.Chinese, PromptLang.English).forEach { lang ->
            SEX_STAGES.forEach { stage ->
                assertTrue(t(lang, stage.stringKey).isNotEmpty())
            }
        }
    }

    @Test
    fun pickStageFallsBackToDuring() {
        val spec = testSpec(mode = PromptMode.Sex, stageWeights = SEX_STAGES.associateWith { 0.0 })
        val rng = Random(0)
        val picked = (1..20).map { PromptGenerator.pickStage(spec, rng) }.toSet()
        assertEquals(setOf(SexStage.During), picked)
    }

    @Test
    fun pickStageRespectsASingleWeight() {
        val spec = testSpec(
            mode = PromptMode.Sex,
            stageWeights = defaultStageWeights() + mapOf(SexStage.During to 0.0, SexStage.Pose to 1.0),
        )
        val rng = Random(1)
        assertEquals(setOf(SexStage.Pose), (1..20).map { PromptGenerator.pickStage(spec, rng) }.toSet())
    }

    @Test
    fun thePoseStageHasNoPenisOrSex() {
        val tags = tagsOf(assembleLine(SexStage.Pose))
        assertTrue(tags.contains("solo"))
        assertTrue(tags.contains("presenting"))
        assertTrue(tags.contains("pussy"))
        assertFalse(tags.contains("penis"))
        assertFalse(tags.contains("1boy"))
        assertFalse(tags.contains("sex"))
        assertFalse(tags.contains("vaginal"))
    }

    @Test
    fun theBeforeStageAimsWithoutSex() {
        val tags = tagsOf(assembleLine(SexStage.Before))
        assertTrue(tags.contains("1boy"))
        assertTrue(tags.contains("penis"))
        assertTrue(tags.contains("imminent vaginal"))
        assertTrue(tags.contains("penis on pussy"))
        assertFalse(tags.contains("sex"))
        assertFalse(tags.contains("vaginal"))
        val anal = tagsOf(assembleLine(SexStage.Before, PromptChannel.Anal))
        assertTrue(anal.contains("imminent anal"))
        assertTrue(anal.contains("penis on ass"))
        assertFalse(anal.contains("sex"))
    }

    @Test
    fun theDuringStageMatchesTheOldSexLine() {
        val tags = tagsOf(assembleLine(SexStage.During, PromptChannel.Anal))
        assertTrue(tags.contains("sex"))
        assertTrue(tags.contains(PromptLimits.ANAL_CHANNEL_TAG))
        assertTrue(tags.contains("penis"))
        assertTrue(tags.contains("1boy"))
        assertFalse(tags.contains("ejaculation"))
        assertFalse(tags.contains("after sex"))
    }

    @Test
    fun ejaculationUsesOverflowNotDrip() {
        val tags = tagsOf(assembleLine(SexStage.Ejaculation))
        assertTrue(tags.contains("sex"))
        assertTrue(tags.contains("vaginal"))
        assertTrue(tags.contains("ejaculation"))
        assertTrue(tags.contains("cum in pussy"))
        assertTrue(tags.contains("cum overflow"))
        assertFalse(tags.contains("cumdrip"))
        assertTrue(tagsOf(assembleLine(SexStage.Ejaculation, PromptChannel.Anal)).contains("cum in ass"))
    }

    @Test
    fun afterKeepsThePenisAndDropsSex() {
        val tags = tagsOf(assembleLine(SexStage.After))
        assertTrue(tags.contains("1boy"))
        assertTrue(tags.contains("penis"))
        assertTrue(tags.contains("after sex"))
        assertTrue(tags.contains("after vaginal"))
        assertTrue(tags.contains("cum in pussy"))
        assertTrue(tags.contains("cumdrip"))
        assertTrue(tags.contains("gaping"))
        assertFalse(tags.contains("sex"))
        assertFalse(tags.contains("vaginal"))
    }

    @Test
    fun doneIsAfterWithoutPenis() {
        val tags = tagsOf(assembleLine(SexStage.Done, PromptChannel.Anal))
        assertTrue(tags.contains("solo"))
        assertTrue(tags.contains("after sex"))
        assertTrue(tags.contains("after anal"))
        assertTrue(tags.contains("cumdrip"))
        assertTrue(tags.contains("gaping"))
        assertFalse(tags.contains("penis"))
        assertFalse(tags.contains("1boy"))
        assertFalse(tags.contains("sex"))
        assertNoAnalChannel(tags, tags.toString())
    }

    @Test
    fun generateWithThePoseStageOmitsThePenis() {
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            vaginalRatio = 1.0,
            stageWeights = defaultStageWeights() + mapOf(SexStage.During to 0.0, SexStage.Pose to 1.0),
            count = 6,
        )
        generatePrompts(spec, matrix, 11).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("solo"), line)
            assertFalse(tags.contains("penis"), line)
            assertFalse(tags.contains("sex"), line)
            assertFalse(tags.contains("1boy"), line)
        }
    }

    @Test
    fun generateEjaculationVaginalWritesCumInPussy() {
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(find(matrix.poses, "mating press").key),
            vaginalRatio = 1.0,
            stageWeights = defaultStageWeights() + mapOf(SexStage.During to 0.0, SexStage.Ejaculation to 1.0),
            count = 5,
        )
        generatePrompts(spec, matrix, 12).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("ejaculation"), line)
            assertTrue(tags.contains("cum in pussy"), line)
            assertTrue(tags.contains("sex"), line)
            assertTrue(tags.contains("penis"), line)
        }
    }

    @Test
    fun stageTagsSplitByChannel() {
        assertEquals(emptyList(), PromptGenerator.stageTags(SexStage.During, PromptChannel.Vaginal))
        assertTrue(PromptGenerator.stageTags(SexStage.Before, PromptChannel.Vaginal).contains("imminent vaginal"))
        assertTrue(PromptGenerator.stageTags(SexStage.After, PromptChannel.Anal).contains("after anal"))
        assertEquals(listOf("anal object insertion"), PromptGenerator.stageTags(SexStage.ObjectInsertion, PromptChannel.Anal))
        assertEquals(
            listOf("vaginal object insertion"),
            PromptGenerator.stageTags(SexStage.ObjectInsertion, PromptChannel.Vaginal),
        )
        assertEquals(listOf("anal fingering"), PromptGenerator.stageTags(SexStage.Fingering, PromptChannel.Anal))
        assertEquals(listOf("fingering"), PromptGenerator.stageTags(SexStage.Fingering, PromptChannel.Vaginal))
        assertEquals(emptyList(), PromptGenerator.stageTags(SexStage.ObjectInsertion, PromptChannel.None))
        assertEquals(listOf("ejaculation"), PromptGenerator.stageTags(SexStage.Ejaculation, PromptChannel.None))
        assertFalse(PromptGenerator.stageTags(SexStage.After, PromptChannel.None).contains("gaping"))
        assertFalse(PromptGenerator.stageTags(SexStage.After, PromptChannel.None).contains("cum in pussy"))
    }
}

/** The Python suite's `NONE_MINI`: the same matrix plus the no-channel rows. */
internal val NONE_MINI_MATRIX: String = MINI_MATRIX.replace(
    "full nelson : anal only",
    "full nelson : anal only\noral, fellatio : none\n# oral\npaizuri : none\n# paizuri\nnursing handjob : none\n# nursing",
)

class NoneChannelAndObjectStageTest {
    private val matrix = parseMatrix(NONE_MINI_MATRIX)

    @Test
    fun theNoneChannelIsParsedAndAllowsNoHole() {
        val oral = find(matrix.poses, "oral, fellatio")
        assertEquals(PromptChannel.None, oral.channel)
        assertFalse(PromptGenerator.poseAllowsHole(oral))
        assertTrue(PromptGenerator.poseAllowsHole(find(matrix.poses, "doggystyle")))
        assertTrue(PromptGenerator.poseAllowsHole(find(matrix.poses, "full nelson")))
    }

    @Test
    fun anObjectStageDropsNoneChannelPoses() {
        val (stage, pool) = PromptGenerator.bindStageToPoses(matrix.poses, SexStage.ObjectInsertion)
        assertEquals(SexStage.ObjectInsertion, stage)
        val blobs = pool.map { it.blob }
        assertTrue(blobs.any { it.contains("doggystyle") })
        assertFalse(blobs.any { it.contains("oral") })
        assertFalse(blobs.any { it.contains("paizuri") })
    }

    @Test
    fun anObjectStageFallsBackWhenEveryPoseHasNoChannel() {
        val oral = find(matrix.poses, "oral, fellatio")
        val (stage, pool) = PromptGenerator.bindStageToPoses(listOf(oral), SexStage.Fingering)
        assertEquals(SexStage.During, stage)
        assertEquals(listOf(oral), pool)
    }

    @Test
    fun oralDuringWritesPenisNotSexOrAHole() {
        val oral = find(matrix.poses, "oral, fellatio")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(oral.key),
            count = 6,
        )
        generatePrompts(spec, matrix, 40).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("oral"), line)
            assertTrue(tags.contains("fellatio"), line)
            assertTrue(tags.contains("penis"), line)
            assertTrue(tags.contains("1boy"), line)
            assertFalse(tags.contains("sex"), line)
            assertFalse(tags.contains("vaginal"), line)
            assertNoAnalChannel(tags, line)
            assertFalse(tags.contains("pussy"), line)
            assertFalse(tags.contains("anus"), line)
            assertFalse(tags.contains("imminent vaginal"), line)
        }
    }

    @Test
    fun aNoneChannelDrawGetsNoPussyWords() {
        // The rule reads the line, not the mode: no `pussy` among the anatomy tags means neither the
        // shape nor the hair word rides this draw.
        val oral = find(matrix.poses, "oral, fellatio")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(oral.key),
            pussyShape = find(matrix.pussyShape, "labia").key,
            pussyHair = find(matrix.pussyHair, "shaved pussy").key,
            count = 6,
        )
        generatePrompts(spec, matrix, 67).forEach { line ->
            val tags = tagsOf(line)
            assertFalse(tags.contains("pussy"), line)
            assertFalse(tags.contains("labia"), line)
            assertFalse(tags.contains("shaved pussy"), line)
        }
    }

    @Test
    fun paizuriAndNursingAreNoneChannel() {
        listOf("paizuri", "nursing handjob").forEach { needle ->
            val pose = find(matrix.poses, needle)
            val spec = testSpec(
                mode = PromptMode.Sex,
                exposure = listOf("open"),
                poseAny = false,
                poseKeys = setOf(pose.key),
                count = 4,
            )
            generatePrompts(spec, matrix, 41).forEach { line ->
                val tags = tagsOf(line)
                assertTrue(tags.contains(needle), line)
                assertTrue(tags.contains("penis"), line)
                assertFalse(tags.contains("sex"), line)
                assertFalse(tags.contains("vaginal"), line)
                assertNoAnalChannel(tags, line)
            }
        }
    }

    @Test
    fun analObjectInsertionSkipsPenisAndOtherStages() {
        val nelson = find(matrix.poses, "full nelson")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(nelson.key),
            stageWeights = defaultStageWeights() + mapOf(SexStage.During to 0.0, SexStage.ObjectInsertion to 1.0),
            count = 6,
        )
        generatePrompts(spec, matrix, 42).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("anal object insertion"), line)
            assertTrue(tags.contains("full nelson"), line)
            assertTrue(tags.contains("anus"), line)
            assertTrue(tags.contains("solo"), line)
            assertFalse(tags.contains("penis"), line)
            assertFalse(tags.contains("1boy"), line)
            assertFalse(tags.contains("sex"), line)
            assertNoAnalChannel(tags, line)
            assertFalse(tags.contains("ejaculation"), line)
            assertFalse(tags.contains("presenting"), line)
            assertFalse(tags.contains("imminent anal"), line)
        }
    }

    @Test
    fun objectInsertionNeverLandsOnANoneChannelPose() {
        val oral = find(matrix.poses, "oral, fellatio")
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            poseAny = false,
            poseKeys = setOf(oral.key),
            stageWeights = defaultStageWeights() + mapOf(
                SexStage.During to 0.0,
                SexStage.ObjectInsertion to 1.0,
            ),
            count = 6,
        )
        val warnings = mutableListOf<String>()
        PromptGenerator.generate(spec, matrix, 43, warnings).forEach { line ->
            val tags = tagsOf(line)
            assertTrue(tags.contains("oral"), line)
            assertFalse(tags.contains("object insertion"), line)
            assertFalse(tags.contains("anal object insertion"), line)
            assertFalse(tags.contains("vaginal object insertion"), line)
        }
    }
}

class FaceTagTest {
    private val matrix = miniMatrix()
    private val rng = Random(0)

    @Test
    fun sfwPoolsOmitSexTags() {
        val expression = groupPool(FACE_GROUPS_BY_ID.getValue("expression"), PromptMode.Sfw)
        listOf("smile", "shy", "grin").forEach { assertTrue(expression.contains(it)) }
        listOf("orgasm", "ahegao", "pain").forEach { assertFalse(expression.contains(it)) }
        val eyes = groupPool(FACE_GROUPS_BY_ID.getValue("eyes"), PromptMode.Sfw)
        assertTrue(eyes.contains("closed eyes"))
        assertFalse(eyes.contains("rolling eyes"))
        assertFalse(eyes.contains("spiral eyes"))
        val mouth = groupPool(FACE_GROUPS_BY_ID.getValue("mouth"), PromptMode.Sfw)
        assertTrue(mouth.contains("open mouth"))
        assertTrue(mouth.contains("closed mouth"))
        assertFalse(mouth.contains("tongue out"))
    }

    @Test
    fun theSexPoolIncludesOrgasmAndPain() {
        val tags = groupPool(FACE_GROUPS_BY_ID.getValue("expression"), PromptMode.Sex)
        listOf("orgasm", "pain", "ahegao", "smile").forEach { assertTrue(tags.contains(it)) }
        assertTrue(groupTags(FACE_GROUPS_BY_ID.getValue("eyes")).contains("open eyes"))
    }

    @Test
    fun aLockedSmileIsTheOnlyFaceTag() {
        val pose = MatrixEntry(listOf("sitting", "desk", "chin rest"))
        val spec = testSpec(
            face = testFace(
                "expression" to FacePick.Tags(listOf("smile")),
                "gaze" to FacePick.None,
                "eyes" to FacePick.None,
            ),
        )
        repeat(8) { assertEquals(listOf("smile"), PromptGenerator.faceTags(spec, pose, rng)) }
    }

    @Test
    fun severalTickedTagsStillYieldOnePerPrompt() {
        val pose = MatrixEntry(listOf("sitting", "desk", "chin rest"))
        val face = testFace(
            "expression" to FacePick.Tags(listOf("smile", "shy", "grin")),
            "gaze" to FacePick.None,
            "eyes" to FacePick.None,
        )
        val spec = testSpec(face = face)
        val seen = mutableSetOf<String>()
        repeat(60) { seed ->
            val tags = PromptGenerator.faceTags(spec, pose, Random(seed.toLong()))
            assertEquals(1, tags.size, tags.toString())
            seen.add(tags[0])
        }
        assertEquals(setOf("smile", "shy", "grin"), seen)
    }

    @Test
    fun aMultiTagGroupNeverWritesTwoOfItsTags() {
        val spec = testSpec(
            mode = PromptMode.Nsfw,
            exposure = listOf("open"),
            face = testFace(
                "expression" to FacePick.Tags(listOf("smile", "shy")),
                "gaze" to FacePick.None,
                "eyes" to FacePick.Tags(listOf("open eyes", "closed eyes")),
                "mouth" to FacePick.None,
            ),
            count = 30,
        )
        repeat(20) { seed ->
            generatePrompts(spec, matrix, seed.toLong()).forEach { line ->
                val hits = groupsHit(line)
                assertTrue((hits["expression"] ?: 0) <= 1, line)
                assertTrue((hits["eyes"] ?: 0) <= 1, line)
            }
        }
    }

    @Test
    fun selectionAndValueRoundTrip() {
        val group = FACE_GROUPS_BY_ID.getValue("expression")
        listOf(
            FacePick.ANY,
            FacePick.NONE,
            FacePick.Tags(listOf("smile")),
            FacePick.Tags(listOf("smile", "shy")),
        ).forEach { value ->
            val (anyOn, flags) = faceSelection(group, value)
            assertEquals(value, faceValue(group, anyOn, flags))
        }
    }

    @Test
    fun switchingEveryGroupOffWritesNoFaceTags() {
        val pose = MatrixEntry(listOf("sitting", "desk", "chin rest"))
        val spec = testSpec(face = FACE_GROUPS.associate { it.id to FacePick.NONE })
        assertEquals(emptyList(), PromptGenerator.faceTags(spec, pose, rng))
    }

    @Test
    fun anyYieldsToAPoseThatFillsTheGroup() {
        val pose = MatrixEntry(listOf("sitting", "looking at viewer", "smile"))
        val face = LinkedHashMap(FACE_GROUPS.associate { it.id to FacePick.NONE })
        face["expression"] = FacePick.ANY
        face["gaze"] = FacePick.ANY
        assertEquals(emptyList(), PromptGenerator.faceTags(testSpec(face = face), pose, rng))
    }

    @Test
    fun anyNeverPicksClosedEyesWithLookingAtViewer() {
        val pose = MatrixEntry(listOf("sitting", "looking at viewer"))
        val spec = testSpec(
            face = testFace("expression" to FacePick.None, "gaze" to FacePick.None, "eyes" to FacePick.ANY),
        )
        repeat(20) { seed ->
            val tags = PromptGenerator.faceTags(spec, pose, Random(seed.toLong()))
            assertFalse(tags.contains("closed eyes"))
            assertFalse(tags.contains("looking away"))
        }
    }

    @Test
    fun lockedEyesRespectAPoseGaze() {
        val pose = MatrixEntry(listOf("sitting", "looking at viewer"))
        val spec = testSpec(
            face = testFace(
                "expression" to FacePick.None,
                "gaze" to FacePick.None,
                "eyes" to FacePick.Tags(listOf("closed eyes")),
            ),
        )
        repeat(8) { seed ->
            assertFalse(PromptGenerator.faceTags(spec, pose, Random(seed.toLong())).contains("closed eyes"))
        }
    }

    @Test
    fun aLockedExpressionYieldsToAPoseExpression() {
        val pose = MatrixEntry(listOf("covering own mouth", "shy", "blush"))
        val spec = testSpec(
            face = testFace(
                "expression" to FacePick.Tags(listOf("smile")),
                "gaze" to FacePick.None,
                "eyes" to FacePick.None,
                "blush" to FacePick.None,
            ),
        )
        repeat(8) { seed ->
            assertEquals(emptyList(), PromptGenerator.faceTags(spec, pose, Random(seed.toLong())))
        }
    }

    @Test
    fun atMostOneTagPerGroupOverManyPrompts() {
        val spec = testSpec(
            mode = PromptMode.Nsfw,
            exposure = listOf("open"),
            face = FACE_GROUPS.associate { it.id to FacePick.ANY },
            count = 40,
        )
        repeat(30) { seed ->
            generatePrompts(spec, matrix, seed.toLong()).forEach { line ->
                groupsHit(line).forEach { (group, count) ->
                    assertTrue(count <= 1, "$group twice in: $line")
                }
            }
        }
    }

    @Test
    fun generateWithALockedOrgasm() {
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            face = testFace(
                "expression" to FacePick.Tags(listOf("orgasm")),
                "gaze" to FacePick.None,
                "eyes" to FacePick.None,
            ),
            count = 8,
        )
        generatePrompts(spec, matrix, 31).forEach { line ->
            assertTrue(tagsOf(line).contains("orgasm"), line)
        }
    }

    @Test
    fun sfwAutoGenerateSkipsSexFaceTags() {
        val blob = generatePrompts(testSpec(mode = PromptMode.Sfw, count = 20), matrix, 32).joinToString("\n")
        listOf("orgasm", "ahegao", "pain", "rolling eyes", "spiral eyes").forEach { needle ->
            assertFalse(blob.contains(needle), needle)
        }
    }

    @Test
    fun theSfwFaceWarningSeesOnlySexTags() {
        assertTrue(needsSfwFaceWarning(PromptMode.Sfw, listOf("orgasm")))
        assertFalse(needsSfwFaceWarning(PromptMode.Sfw, listOf("smile")))
        assertFalse(needsSfwFaceWarning(PromptMode.Sex, listOf("orgasm")))
    }

    @Test
    fun selectedFaceTagsIgnoreAnyAndOff() {
        val face = testFace(
            "expression" to FacePick.Tags(listOf("orgasm")),
            "gaze" to FacePick.ANY,
            "eyes" to FacePick.None,
            "blush" to FacePick.Tags(listOf("blush")),
        )
        assertEquals(setOf("orgasm", "blush"), selectedFaceTags(face))
    }
}

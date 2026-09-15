package com.acite.axlranko.pages.components

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.SampleItem
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ChartPickTest {

    private fun checkpoint(
        dir: String,
        step: Int?,
        runId: String = "run_20260101_000000",
        final: Boolean = false,
    ) = CheckpointItem(
        path = "/out/$runId/$dir/$dir.safetensors",
        runId = runId,
        dir = dir,
        filename = "$dir.safetensors",
        step = step,
        final = final,
    )

    private fun sample(step: Int, index: Int = 0) = SampleItem(
        filename = "run_${step.toString().padStart(6, '0')}_$index.png",
        repeatIdx = index,
        path = "/out/run_samples/run_${step.toString().padStart(6, '0')}_$index.png",
    )

    private fun samplesOf(vararg steps: Int): Map<String, List<SampleItem>> =
        steps.associate { it.toString() to listOf(sample(it)) }

    @Test
    fun mapsPlotXToDataStep() {
        assertEquals(50f, stepAtPlotX(PLOT_LEFT_PADDING + 50f, 100f, 0f, 100f)!!, 1e-3f)
        assertEquals(0f, stepAtPlotX(PLOT_LEFT_PADDING, 100f, 0f, 1000f)!!, 1e-3f)
        assertEquals(1000f, stepAtPlotX(PLOT_LEFT_PADDING + 100f, 100f, 0f, 1000f)!!, 1e-3f)
        // A zoomed/panned viewport is honoured: xMin/xRange come from the live viewport.
        assertEquals(250f, stepAtPlotX(PLOT_LEFT_PADDING + 50f, 100f, 200f, 100f)!!, 1e-3f)
    }

    @Test
    fun rejectsTheAxisGutterAndDegeneratePlots() {
        assertNull(stepAtPlotX(PLOT_LEFT_PADDING - 1f, 100f, 0f, 100f))
        assertNull(stepAtPlotX(0f, 100f, 0f, 100f))
        assertNull(stepAtPlotX(PLOT_LEFT_PADDING + 10f, 0f, 0f, 100f))
        assertNull(stepAtPlotX(PLOT_LEFT_PADDING + 10f, -5f, 0f, 100f))
    }

    @Test
    fun keepsCheckpointsOfTheResolvedRun() {
        val items = listOf(
            checkpoint("old_s000100", 100, runId = "run_old"),
            checkpoint("new_s000100", 100, runId = "run_new"),
        )
        assertEquals(items, checkpointsForRun(items, null))
        assertEquals(items, checkpointsForRun(items, "  "))
        assertEquals(listOf(items[1]), checkpointsForRun(items, "run_new"))
        assertTrue(checkpointsForRun(items, "run_missing").isEmpty())
    }

    @Test
    fun picksTheClosestCheckpointByStep() {
        val items = listOf(
            checkpoint("run_s000050", 50),
            checkpoint("run_s000100", 100),
            checkpoint("run_s000150", 150),
        )
        assertEquals("run_s000100", nearestCheckpoint(items, 118f)?.dir)
        assertEquals("run_s000150", nearestCheckpoint(items, 149f)?.dir)
        assertEquals("run_s000050", nearestCheckpoint(items, 0f)?.dir)
        assertEquals("run_s000150", nearestCheckpoint(items, 9999f)?.dir)
    }

    @Test
    fun tiesPreferTheLowerStepThenNonFinalThenDirectoryName() {
        val lower = checkpoint("run_s000100", 100)
        val higher = checkpoint("run_s000140", 140)
        assertEquals("run_s000100", nearestCheckpoint(listOf(higher, lower), 120f)?.dir)

        val final = checkpoint("run_final", 300, final = true)
        val step = checkpoint("run_s000300", 300)
        assertEquals("run_s000300", nearestCheckpoint(listOf(final, step), 300f)?.dir)

        val a = checkpoint("run_s000301", 301)
        val b = checkpoint("run_s000299", 299)
        // Same distance from 300: the lower step wins rather than the list order.
        assertEquals("run_s000299", nearestCheckpoint(listOf(a, b), 300f)?.dir)
    }

    @Test
    fun ignoresCheckpointsWithoutAStep() {
        val noStep = checkpoint("run_final", null)
        assertNull(nearestCheckpoint(listOf(noStep), 500f))
        assertEquals(
            "run_s000500",
            nearestCheckpoint(listOf(noStep, checkpoint("run_s000500", 500)), 400f)?.dir,
        )
        assertNull(nearestCheckpoint(emptyList(), 10f))
    }

    @Test
    fun nearestMetricPointFollowsTheSameRule() {
        val points = listOf(
            MetricPoint(step = 10, value = 1f),
            MetricPoint(step = 20, value = 2f),
            MetricPoint(step = 30, value = 3f),
        )
        assertNull(nearestMetricPoint(emptyList(), 15f))
        assertEquals(20, nearestMetricPoint(points, 19f)?.step)
        assertEquals(2f, nearestMetricPoint(points, 19f)?.value)
        assertEquals(10, nearestMetricPoint(points, 15f)?.step)
        assertEquals(30, nearestMetricPoint(points, 999f)?.step)
    }

    @Test
    fun samplesAreLookedUpByExactStep() {
        val samples = samplesOf(50, 100)
        assertEquals(sample(100), samplesForStep(samples, 100).first())
        assertTrue(samplesForStep(samples, 75).isEmpty())
        assertTrue(samplesForStep(samples, null).isEmpty())
    }

    @Test
    fun nearestSampledStepSkipsTheUnmatchedBucket() {
        val samples = samplesOf(50, 100, 150) + mapOf("-1" to listOf(sample(0, index = 7)))
        assertEquals(100, nearestSampledStep(samples, 96))
        assertEquals(100, nearestSampledStep(samples, 125))
        assertEquals(50, nearestSampledStep(samples, 10))
        assertEquals(150, nearestSampledStep(samples, 500))
        assertNull(nearestSampledStep(emptyMap(), 100))
        assertNull(nearestSampledStep(samples, null))
        // Only the unmatched bucket exists: nothing to fall back to.
        assertNull(nearestSampledStep(mapOf("-1" to listOf(sample(0, index = 7))), 100))
    }

    @Test
    fun stepsAreReadFromTheSampleKeysDefensively() {
        val samples = mapOf("abc" to listOf(sample(0)), "120" to listOf(sample(120)))
        assertEquals(120, nearestSampledStep(samples, 100))
    }

    @Test
    fun threeSamplesFitThePanelWithoutScrolling() {
        val thumb = sampleThumbWidth(PANEL_MAX_WIDTH, SAMPLES_PER_ROW)
        val panel = checkpointPanelWidth(PANEL_MAX_WIDTH, thumb, SAMPLES_PER_ROW)

        assertEquals(SAMPLE_THUMB_TARGET_WIDTH, thumb)
        assertTrue(panel <= PANEL_MAX_WIDTH, "panel $panel exceeded the cap")
        assertTrue(
            thumb * SAMPLES_PER_ROW + SAMPLE_SLOT_SPACING * (SAMPLES_PER_ROW - 1) + PANEL_CARD_PADDING <= panel,
            "three slots of $thumb do not fit the $panel panel",
        )
    }

    @Test
    fun narrowWindowsShrinkTheSlotsInsteadOfClipping() {
        val max = 900.dp
        val thumb = sampleThumbWidth(max, SAMPLES_PER_ROW)
        val panel = checkpointPanelWidth(max, thumb, SAMPLES_PER_ROW)

        assertTrue(thumb < SAMPLE_THUMB_TARGET_WIDTH, "slots should shrink on a narrow window, got $thumb")
        assertTrue(thumb >= SAMPLE_THUMB_MIN_WIDTH)
        assertTrue(thumb * SAMPLES_PER_ROW + SAMPLE_SLOT_SPACING * 2 + PANEL_CARD_PADDING <= panel)
        assertTrue(panel <= max)
    }

    @Test
    fun fewerSamplesMakeANarrowerPanel() {
        val three = checkpointPanelWidth(PANEL_MAX_WIDTH, sampleThumbWidth(PANEL_MAX_WIDTH, 3), 3)
        val one = checkpointPanelWidth(PANEL_MAX_WIDTH, sampleThumbWidth(PANEL_MAX_WIDTH, 1), 1)
        val none = checkpointPanelWidth(PANEL_MAX_WIDTH, sampleThumbWidth(PANEL_MAX_WIDTH, 0), 0)

        assertEquals(PANEL_TEXT_MIN_WIDTH, none)
        assertEquals(PANEL_TEXT_MIN_WIDTH, one)
        assertTrue(three > one, "three samples should widen the panel ($three vs $one)")
    }

    @Test
    fun aWindowTooNarrowForThreeSlotsFallsBackToScrolling() {
        val max = 200.dp
        val thumb = sampleThumbWidth(max, SAMPLES_PER_ROW)
        val panel = checkpointPanelWidth(max, thumb, SAMPLES_PER_ROW)

        assertEquals(SAMPLE_THUMB_MIN_WIDTH, thumb)
        assertEquals(max, panel)
    }

    @Test
    fun theDefaultPanelIsTwiceTheEarlierContentSize() {
        // 3 x 400 dp slots + 2 gaps + card padding; the previous default was half of this.
        val width = checkpointPanelWidth(PANEL_MAX_WIDTH, sampleThumbWidth(PANEL_MAX_WIDTH, 3), 3)
        assertEquals(1268.dp, width)
        assertEquals(400.dp, sampleThumbWidth(PANEL_MAX_WIDTH, SAMPLES_PER_ROW))
    }

    @Test
    fun draggedSizesAreClampedToTheWindowAndTheMinimum() {
        assertEquals(
            DpSize(820.dp, 640.dp),
            clampPanelSize(820.dp, 640.dp, maxWidth = 1200.dp, maxHeight = 900.dp),
        )
        assertEquals(
            DpSize(PANEL_MIN_WIDTH, PANEL_MIN_HEIGHT),
            clampPanelSize(10.dp, 10.dp, maxWidth = 1200.dp, maxHeight = 900.dp),
        )
        assertEquals(
            DpSize(1200.dp, 900.dp),
            clampPanelSize(5000.dp, 5000.dp, maxWidth = 1200.dp, maxHeight = 900.dp),
        )
        // A window smaller than the minimum still wins, so the panel can never exceed the window.
        assertEquals(
            DpSize(240.dp, 180.dp),
            clampPanelSize(900.dp, 900.dp, maxWidth = 240.dp, maxHeight = 180.dp),
        )
    }

    @Test
    fun slotsFillTheDraggedPanelWidth() {
        // Dragging wider must grow the pictures past the 400 dp default, not add empty card space.
        val wide = sampleSlotWidth(1440.dp, SAMPLES_PER_ROW)
        val default = sampleSlotWidth(1268.dp, SAMPLES_PER_ROW)
        val narrow = sampleSlotWidth(700.dp, SAMPLES_PER_ROW)

        assertTrue(wide > default, "$wide should exceed the default $default")
        assertTrue(narrow < default, "$narrow should be below the default $default")
        assertTrue(
            wide * SAMPLES_PER_ROW + SAMPLE_SLOT_SPACING * 2 + PANEL_CARD_PADDING <= 1440.dp,
            "the slots overflow the panel",
        )
        assertEquals(SAMPLE_THUMB_MAX_WIDTH, sampleSlotWidth(4000.dp, 1))
        assertEquals(SAMPLE_THUMB_MIN_WIDTH, sampleSlotWidth(200.dp, SAMPLES_PER_ROW))
    }

    @Test
    fun columnsFollowTheSampleCount() {
        assertEquals(1, sampleColumns(0))
        assertEquals(1, sampleColumns(1))
        assertEquals(2, sampleColumns(2))
        assertEquals(3, sampleColumns(3))
        assertEquals(3, sampleColumns(7))
    }

    @Test
    fun thePanelOpensRightAndBelowWhenThereIsRoom() {
        val origin = placePanelOrigin(
            anchor = Offset(200f, 300f),
            panelWidth = 1000f,
            minVisibleHeight = 200f,
            bounds = Size(2400f, 1300f),
            gap = 16f,
            margin = 8f,
        )
        assertEquals(Offset(216f, 316f), origin)
    }

    @Test
    fun aPanelWithoutRoomToTheRightFlipsLeft() {
        val origin = placePanelOrigin(
            anchor = Offset(1800f, 300f),
            panelWidth = 1000f,
            minVisibleHeight = 200f,
            bounds = Size(2400f, 1300f),
            gap = 16f,
            margin = 8f,
        )
        assertEquals(784f, origin.x)
        assertEquals(316f, origin.y)
    }

    @Test
    fun aClickNearTheBottomOpensThePanelAbove() {
        val origin = placePanelOrigin(
            anchor = Offset(200f, 1250f),
            panelWidth = 1000f,
            minVisibleHeight = 200f,
            bounds = Size(2400f, 1300f),
            gap = 16f,
            margin = 8f,
        )
        assertEquals(1250f - 16f - 200f, origin.y)
    }

    @Test
    fun resizingSlidesThePanelBackInsideInsteadOfFlippingIt() {
        val anchor = Offset(1800f, 700f)
        val bounds = Size(2400f, 1300f)
        val origin = placePanelOrigin(anchor, 1000f, 200f, bounds, gap = 16f, margin = 8f)
        assertEquals(Offset(784f, 716f), origin)

        val small = clampPanelOrigin(origin, Size(600f, 400f), bounds, margin = 8f)
        assertEquals(origin, small, "a smaller panel must not move")

        // Grown past the bottom edge: the panel slides up by the least amount instead of jumping to
        // the flip position (anchor.y - gap - height = -216, which would clamp to the 8 px margin).
        val grown = clampPanelOrigin(origin, Size(2000f, 900f), bounds, margin = 8f)
        assertEquals(2400f - 8f - 2000f, grown.x)
        assertEquals(1300f - 8f - 900f, grown.y)
        assertTrue(grown.y > 8f, "the panel slid into view, it did not flip above the click")
    }

    @Test
    fun clampingKeepsEveryCornerOnScreen() {
        val bounds = Size(1000f, 800f)
        assertEquals(Offset(8f, 8f), clampPanelOrigin(Offset(-50f, -50f), Size(200f, 100f), bounds, 8f))
        assertEquals(
            Offset(792f, 692f),
            clampPanelOrigin(Offset(5000f, 5000f), Size(200f, 100f), bounds, 8f),
        )
        assertEquals(
            Offset(8f, 8f),
            clampPanelOrigin(Offset(400f, 400f), Size(2000f, 1600f), bounds, 8f),
        )
    }

    @Test
    fun trainingInfoListsLossAndBothLearningRates() {
        val metrics = mapOf(
            "Train/Avg_Loss" to listOf(MetricPoint(step = 100, value = 0.123456f)),
            "Train/Loss" to listOf(MetricPoint(step = 100, value = 0.2f)),
            "UNet/LR/Effective_Actual_LR" to listOf(MetricPoint(step = 100, value = 2.5e-5f)),
            "TE/LR/Base_Scheduled" to listOf(MetricPoint(step = 100, value = 2.5e-6f)),
        )
        val stats = trainingInfoAt(metrics, step = 101f)

        assertEquals(listOf("Avg Loss", "Loss", "UNet LR", "TE LR"), stats.map { it.label })
        assertEquals("0.1235", stats[0].value)
        assertEquals("0.2000", stats[1].value)
        assertEquals("2.50e-05", stats[2].value)
        assertEquals("2.50e-06", stats[3].value)
        assertEquals(100, stats[0].step)
    }

    @Test
    fun trainingInfoKeepsItsShapeWhenASeriesIsMissing() {
        val stats = trainingInfoAt(mapOf("Train/Avg_Loss" to listOf(MetricPoint(1, 1f))), step = 1f)
        assertEquals(4, stats.size)
        assertEquals("1.0000", stats[0].value)
        assertEquals(listOf("—", "—", "—"), stats.drop(1).map { it.value })
        assertNull(stats[1].step)
    }

    @Test
    fun theGenerateFormAcceptsACompleteEntry() {
        assertNull(generateFormError("a girl", "5.0", "20", "0"))
        assertNull(generateFormError("a girl", " 1 ", " 150 ", " 4294967295 "))
    }

    @Test
    fun theGenerateFormRejectsBadValuesWithAMessage() {
        assertEquals("Enter a prompt first", generateFormError("   ", "5", "20", "0"))
        assertTrue(generateFormError("p", "0.5", "20", "0")!!.contains("CFG"))
        assertTrue(generateFormError("p", "31", "20", "0")!!.contains("CFG"))
        assertTrue(generateFormError("p", "hot", "20", "0")!!.contains("CFG"))
        assertTrue(generateFormError("p", "5", "0", "0")!!.contains("Steps"))
        assertTrue(generateFormError("p", "5", "151", "0")!!.contains("Steps"))
        assertTrue(generateFormError("p", "5", "20", "-1")!!.contains("Seed"))
        assertTrue(generateFormError("p", "5", "20", "4294967296")!!.contains("Seed"))
    }

    @Test
    fun theGenerateFormStartsFromTheConfigsSampleSettings() {
        val config = JsonObject(
            mapOf(
                "sample_prompts" to JsonPrimitive("a girl"),
                "sample_negative" to JsonPrimitive("bad"),
                "guidance_scale" to JsonPrimitive(5.0),
                "sample_steps" to JsonPrimitive(35),
                "sample_seed" to JsonPrimitive(0),
            ),
        )
        val defaults = generateFormDefaults(config)
        assertEquals("a girl", defaults.prompt)
        assertEquals("bad", defaults.negativePrompt)
        assertEquals("5.0", defaults.cfg)
        assertEquals("35", defaults.steps)
        assertEquals("0", defaults.seed)
    }

    @Test
    fun theGenerateFormFallsBackWhenTheConfigLacksTheKeys() {
        val defaults = generateFormDefaults(JsonObject(emptyMap()))
        assertEquals("", defaults.prompt)
        assertEquals("6.0", defaults.cfg)
        assertEquals("55", defaults.steps)
        assertEquals("0", defaults.seed)
    }

    @Test
    fun anEmptyConfiguredPromptDoesNotShadowTheFallback() {
        val config = JsonObject(mapOf("sample_prompts" to JsonPrimitive("")))
        assertEquals("", generateFormDefaults(config).prompt)
        assertEquals("6.0", generateFormDefaults(JsonObject(mapOf("guidance_scale" to JsonPrimitive("")))).cfg)
    }

    @Test
    fun theGenerateFormStartsFromTheFirstSampleSet() {
        val config = JsonObject(
            mapOf(
                "sample_prompts" to JsonPrimitive("flat prompt"),
                "sample_negative" to JsonPrimitive("flat negative"),
                "sample_steps" to JsonPrimitive("35"),
                "sample_sets" to JsonArray(
                    listOf(
                        JsonObject(
                            mapOf(
                                "prompt" to JsonPrimitive("set one"),
                                "negative" to JsonPrimitive("set one negative"),
                                "guidance_scale" to JsonPrimitive("4.0"),
                                "steps" to JsonPrimitive("9"),
                                "seed" to JsonPrimitive("11"),
                            ),
                        ),
                        JsonObject(mapOf("prompt" to JsonPrimitive("set two"))),
                    ),
                ),
            ),
        )
        val defaults = generateFormDefaults(config)
        assertEquals("set one", defaults.prompt)
        assertEquals("set one negative", defaults.negativePrompt)
        assertEquals("4.0", defaults.cfg)
        assertEquals("9", defaults.steps)
        assertEquals("11", defaults.seed)
    }

    @Test
    fun theGenerateFormFallsBackToTheFlatKeysWhenTheSetLacksOne() {
        val config = JsonObject(
            mapOf(
                "sample_negative" to JsonPrimitive("flat negative"),
                "sample_sets" to JsonArray(listOf(JsonObject(mapOf("prompt" to JsonPrimitive("set one"))))),
            ),
        )
        val defaults = generateFormDefaults(config)
        assertEquals("set one", defaults.prompt)
        assertEquals("flat negative", defaults.negativePrompt)
        assertEquals("6.0", defaults.cfg)
    }

    @Test
    fun promptSetBadgesOnlyAppearForRealSets() {
        assertEquals("P1", sampleSetBadge(0))
        assertEquals("P3", sampleSetBadge(2))
        // A generated image belongs to no prompt set.
        assertNull(sampleSetBadge(-1))
    }

    @Test
    fun singleSetRunsShowNoBadges() {
        assertFalse(showsSampleSetBadges(samplesOf(100, 200)))
        assertTrue(showsSampleSetBadges(mapOf("100" to listOf(sample(100), sample(100, index = 1).copy(setIndex = 1)))))
    }

    @Test
    fun aSecondClickInsideTheWindowIsADoubleClick() {
        assertTrue(completesDoubleClick(previousMillis = 1_000, currentMillis = 1_250))
        assertTrue(completesDoubleClick(previousMillis = 1_000, currentMillis = 1_000 + 400))
    }

    @Test
    fun aSlowSecondClickOrNoPendingClickIsNotADoubleClick() {
        assertFalse(completesDoubleClick(previousMillis = 1_000, currentMillis = 1_401))
        assertFalse(completesDoubleClick(previousMillis = 0, currentMillis = 1_100))
        assertFalse(completesDoubleClick(previousMillis = 1_100, currentMillis = 1_000))
    }
}

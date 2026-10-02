package com.acite.axlranko.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The Sampling Prompts section's own rules: the payload-to-form round trip, the editor's
 * validation (shared with the Utils Validation form), the labels both surfaces draw, and the
 * confirmation a card asks before it deletes anything. No window and no helper involved.
 */
class SamplePromptsRulesTest {

    private fun set(
        prompt: String = "1girl, solo",
        name: String = "one",
        width: Int = 1152,
        height: Int = 768,
        steps: Int = 35,
        guidance: Float = 6.0f,
        rescale: Float = 0.6f,
        seed: Long = 0,
        repeat: Int = 2,
    ) = SampleSetInfo(
        name = name,
        prompt = prompt,
        negative = "worst quality",
        width = width,
        height = height,
        steps = steps,
        guidanceScale = guidance,
        guidanceRescale = rescale,
        seed = seed,
        repeat = repeat,
    )

    private fun response(
        sets: List<SampleSetInfo> = listOf(set()),
        edited: Boolean = false,
        file: String? = null,
        configSource: String = "/logs/rein_1/config.toml",
        live: Boolean = false,
        reason: String = "",
    ) = SamplePromptsResponse(
        runId = "rein_1",
        outputName = "rein",
        file = file,
        edited = edited,
        configSource = configSource,
        sets = sets,
        live = live,
        reason = reason,
    )

    @Test
    fun aSetRoundTripsThroughTheEditor() {
        val original = set(prompt = "a prompt", name = "named", seed = 42, repeat = 3)
        val back = sampleSetInfos(listOf(original.toForm()))?.single()
        assertEquals(original, back)
    }

    @Test
    fun wholeNumbersShowWithoutADecimal() {
        assertEquals("6", sampleValueLabel(6.0f))
        assertEquals("0.6", sampleValueLabel(0.6f))
        assertEquals("6", set(guidance = 6.0f).toForm().guidanceScale)
        assertEquals("0.6", set(rescale = 0.6f).toForm().guidanceRescale)
    }

    @Test
    fun aFieldThatDoesNotParseConvertsToNull() {
        val broken = set().toForm().copy(steps = "later")
        assertNull(sampleSetInfos(listOf(broken)))
        assertNull(sampleSetInfos(listOf(set().toForm().copy(seed = ""))))
    }

    @Test
    fun theEditorsRulesAreTheOnesTheUtilsFormUses() {
        val bad = listOf(
            SampleSetForm(prompt = "", width = "640", height = "960", steps = "9", repeat = "1",
                guidanceScale = "6", guidanceRescale = "0", seed = "0"),
            SampleSetForm(prompt = "p", width = "8", height = "960", steps = "9", repeat = "1",
                guidanceScale = "6", guidanceRescale = "0", seed = "0"),
            SampleSetForm(prompt = "p", width = "640", height = "960", steps = "0", repeat = "1",
                guidanceScale = "6", guidanceRescale = "0", seed = "0"),
            SampleSetForm(prompt = "p", width = "640", height = "960", steps = "9", repeat = "99",
                guidanceScale = "6", guidanceRescale = "0", seed = "0"),
            SampleSetForm(prompt = "p", width = "640", height = "960", steps = "9", repeat = "1",
                guidanceScale = "31", guidanceRescale = "0", seed = "0"),
            SampleSetForm(prompt = "p", width = "640", height = "960", steps = "9", repeat = "1",
                guidanceScale = "6", guidanceRescale = "2", seed = "0"),
            SampleSetForm(prompt = "p", width = "640", height = "960", steps = "9", repeat = "1",
                guidanceScale = "6", guidanceRescale = "0", seed = "-1"),
        )
        for ((index, form) in bad.withIndex()) {
            val errors = sampleSetFormErrors(listOf(form))
            assertTrue(errors.isNotEmpty(), "entry $index must be rejected")
        }
        assertTrue(sampleSetFormErrors(listOf(set().toForm())).isEmpty())
        // No sets at all is an error too: a run with no prompts writes no images.
        assertTrue(sampleSetFormErrors(emptyList()).isNotEmpty())
    }

    @Test
    fun theSummaryNamesWhatOneSetDraws() {
        assertEquals("1152×768 · 35 steps · CFG 6 · rescale 0.6 · seed 0 · ×2", sampleSetSummaryLine(set()))
        // The rescale only shows when it is on: it is off by default in most configs.
        assertEquals("640×960 · 9 steps · CFG 4 · seed 11 · ×1",
            sampleSetSummaryLine(set(width = 640, height = 960, steps = 9, guidance = 4.0f, rescale = 0f, seed = 11, repeat = 1)))
        assertTrue(sampleSetSummaryLine(set(rescale = 0.6f)).contains("rescale 0.6"))
    }

    @Test
    fun theSectionSaysHowManyImagesAPassWrites() {
        assertEquals("1 set · 2 images per pass", samplePromptsSummaryLine(response()))
        assertEquals(
            "3 sets · 5 images per pass",
            samplePromptsSummaryLine(response(sets = listOf(set(repeat = 1), set(repeat = 3), set(repeat = 1)))),
        )
    }

    @Test
    fun theSourceLabelFollowsWhatWasEdited() {
        assertEquals("From config.toml", samplePromptsSourceLabel(response()))
        assertEquals(
            "Edited for this run · sample_sets.json",
            samplePromptsSourceLabel(
                response(edited = true, file = "/logs/rein_1/sample_sets.json",
                    configSource = "/logs/rein_1/sample_sets.json")
            ),
        )
        // A payload with neither still says something rather than nothing.
        assertEquals("No config resolved", samplePromptsSourceLabel(response(configSource = "")))
    }

    @Test
    fun onlyALiveRunSaysTheNextSamplePointUsesThem() {
        assertNull(samplePromptsLiveLabel(response()))
        assertTrue(samplePromptsLiveLabel(response(live = true))!!.contains("next sample point"))
        // A run whose prompts could not be read has nothing to promise.
        assertNull(samplePromptsLiveLabel(response(live = true, sets = emptyList())))
    }

    @Test
    fun theEditorTitleCountsTheSets() {
        assertEquals("Sampling prompts · 1 set", samplePromptsEditorTitle(response()))
        assertEquals(
            "Sampling prompts · 2 sets",
            samplePromptsEditorTitle(response(sets = listOf(set(), set()))),
        )
        assertEquals("Sampling prompts · 0 sets", samplePromptsEditorTitle(null))
    }

    @Test
    fun theConfirmationSaysWhatTheButtonRemoves() {
        val both = clearSamplesConfirmText(images = 8, jobs = 2, step = 3050)
        assertTrue(both.startsWith("Removes 8 images at step 3050"))
        assertTrue(both.contains("the 2 pass records that produced them"))
        assertTrue(both.contains("checkpoint itself is kept"))

        val only = clearSamplesConfirmText(images = 1, jobs = 0, step = null)
        assertTrue(only.startsWith("Removes 1 image"))
        assertTrue(!only.contains("pass record"))
        val one = clearSamplesConfirmText(images = 3, jobs = 1, step = 100)
        assertTrue(one.contains("the pass record that produced it"))
        assertTrue(CLEAR_SAMPLES_CONFIRM_TITLE.contains("sample images"))
    }

    @Test
    fun theCardReportsHowManyWentOrWhyItFailed() {
        assertEquals("Cleared 7 images", clearedSamplesLabel(SampleClearResult(images = 7)))
        assertEquals("Cleared 1 image", clearedSamplesLabel(SampleClearResult(images = 1)))
        assertEquals(
            "Could not clear samples: training is using the GPU",
            clearedSamplesLabel(SampleClearResult(error = "training is using the GPU")),
        )
    }
}

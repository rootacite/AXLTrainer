package com.acite.axlranko.pages.components

import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.EvaluationGroup
import com.acite.axlranko.model.EvaluationScores
import com.acite.axlranko.model.EvaluationTagCount
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The evaluation rules the Dashboard reads: the dialog's validation, the labels a card draws, and
 * which of a checkpoint's jobs the card reports on. Composed without a window.
 */
class EvaluationRulesTest {

    private val checkpoint = CheckpointItem(
        path = "/out/rein_20260911_120000/rein_s003050/rein.safetensors",
        runId = "rein_20260911_120000",
        dir = "rein_s003050",
        filename = "rein.safetensors",
        step = 3050,
        final = false,
        sizeBytes = 24_000_000,
        networkDim = 32,
        networkAlpha = 16,
        outputName = "rein",
    )

    private fun evaluation(
        id: String = "evaluate_gen_1",
        state: String = JOB_DONE,
        phase: String = "done",
        startedAt: Double = 1.0,
        files: List<String> = emptyList(),
        scores: EvaluationScores? = null,
    ) = GeneratedSampleJob(
        id = id,
        state = state,
        mode = JOB_MODE_EVALUATE,
        step = 3050,
        checkpoint = checkpoint.path,
        phase = phase,
        files = files,
        startedAt = startedAt,
        scores = scores,
    )

    private fun scores(
        f1: Float = 0.689f,
        precision: Float = 0.769f,
        recall: Float = 0.625f,
        unionF1: Float = 0.75f,
        scored: Int = 21,
        failed: Int = 0,
        skipped: Int = 0,
        groups: List<EvaluationGroup> = emptyList(),
        topFalsePositives: List<EvaluationTagCount> = emptyList(),
        topFalseNegatives: List<EvaluationTagCount> = emptyList(),
    ) = EvaluationScores(
        f1 = f1,
        precision = precision,
        recall = recall,
        unionF1 = unionF1,
        imagesScored = scored,
        imagesFailed = failed,
        imagesSkipped = skipped,
        groups = groups,
        topFalsePositives = topFalsePositives,
        topFalseNegatives = topFalseNegatives,
    )

    @Test
    fun aValidRequestHasNoError() {
        assertNull(evaluationRequestError("20", "0.35"))
        assertNull(evaluationRequestError("1", "1"))
        assertNull(evaluationRequestError("512", "0"))
        assertNull(evaluationRequestError(" 12 ", "0,5"))
    }

    @Test
    fun aBadRequestNamesWhatIsWrong() {
        assertEquals("Enter a depth", evaluationRequestError("", "0.35"))
        assertEquals("Depth must be a whole number", evaluationRequestError("many", "0.35"))
        assertEquals("Depth must be a whole number", evaluationRequestError("1.5", "0.35"))
        assertEquals("Depth must be at least 1", evaluationRequestError("0", "0.35"))
        assertEquals(
            "Depth must be at most $MAX_EVALUATION_DEPTH",
            evaluationRequestError("$MAX_EVALUATION_DEPTH" + "1", "0.35"),
        )
        assertEquals("Enter a threshold", evaluationRequestError("10", ""))
        assertEquals("Threshold must be a number", evaluationRequestError("10", "high"))
        assertEquals("Threshold must be between 0 and 1", evaluationRequestError("10", "1.5"))
    }

    @Test
    fun theDefaultDepthIsWhatTheCardAlreadyShows() {
        assertEquals(7, evaluationDefaultDepth(7))
        assertEquals(1, evaluationDefaultDepth(0))
        assertEquals(1, evaluationDefaultDepth(-3))
    }

    @Test
    fun thePickerPrefillsTheRunsRecordedSelection() {
        assertEquals(
            setOf("1girl", "anal"),
            evaluationPrefillSelection(listOf("1girl", "anal"), listOf("anal", "1girl", "solo")),
        )
    }

    @Test
    fun aTagThisRunsPromptsNoLongerAskForHasNoRowToTick() {
        assertEquals(
            setOf("anal"),
            evaluationPrefillSelection(listOf("retired tag", "anal"), listOf("anal", "1girl")),
        )
    }

    @Test
    fun nothingRecordedAndNothingOfferedAreBothAnEmptyPick() {
        assertEquals(emptySet(), evaluationPrefillSelection(emptyList(), listOf("anal")))
        assertEquals(emptySet(), evaluationPrefillSelection(listOf("anal"), emptyList()))
    }

    @Test
    fun theProgressLabelNamesThePhaseThatOwnsTheCounters() {
        assertEquals(
            "Evaluating · rendering 3/4",
            evaluationProgressLabel(evaluation(state = JOB_RUNNING, phase = "rendering").copy(imagesDone = 3, totalImages = 4)),
        )
        assertEquals(
            "Evaluating · tagging 8/21",
            evaluationProgressLabel(evaluation(state = JOB_RUNNING, phase = "tagging").copy(imagesDone = 8, totalImages = 21)),
        )
        assertEquals(
            "Evaluating · scoring",
            evaluationProgressLabel(evaluation(state = JOB_RUNNING, phase = "scoring").copy(imagesDone = 21, totalImages = 21)),
        )
        // A record from before the phases were written still reports something countable.
        assertEquals(
            "Evaluating · image 2/9",
            evaluationProgressLabel(evaluation(state = JOB_RUNNING, phase = "").copy(imagesDone = 2, totalImages = 9)),
        )
        assertEquals(
            "Evaluating",
            evaluationProgressLabel(evaluation(state = JOB_RUNNING, phase = "").copy(imagesDone = 0, totalImages = 0)),
        )
        // A cancel is what the running row says, whatever the phase.
        assertEquals(
            "cancelling…",
            generatedJobSetProgress(
                evaluation(state = JOB_RUNNING, phase = "tagging").copy(cancelRequested = true),
            ),
        )
        assertNull(generatedJobSetProgress(evaluation(state = JOB_DONE)))
    }

    @Test
    fun recallIsTheHeadlineAndTheRestIsSecondary() {
        assertEquals("Recall 0.62", evaluationRecallHeadline(scores()))
        assertEquals("P 0.77 · F1 0.69 · 21 images", evaluationSecondaryLabel(scores()))
        assertEquals(
            "Recall 1.00",
            evaluationRecallHeadline(scores(f1 = 1f, precision = 1f, recall = 1f, scored = 1)),
        )
        assertEquals(
            "P 1.00 · F1 1.00 · 1 image",
            evaluationSecondaryLabel(scores(f1 = 1f, precision = 1f, recall = 1f, scored = 1)),
        )
        assertEquals(
            "union recall 0.71 · P 0.79",
            evaluationUnionLabel(scores(unionF1 = 0.75f).copy(unionPrecision = 0.789f, unionRecall = 0.714f)),
        )
    }

    @Test
    fun theScoredTagsAreSpeltOutOnlyWhenThereIsASelection() {
        assertNull(evaluationTagsLabel(scores()))
        assertNull(evaluationTagsLabel(null))
        assertEquals("scored tags: anal, pussy", evaluationTagsLabel(scores().copy(tags = listOf("anal", "pussy"))))
    }

    @Test
    fun anEvaluationThatScoredNothingHasNoScoreLine() {
        assertNull(evaluationRecallHeadline(null))
        assertNull(evaluationRecallHeadline(scores(scored = 0)))
        assertNull(evaluationSecondaryLabel(null))
        assertNull(evaluationSecondaryLabel(scores(scored = 0)))
        assertNull(evaluationUnionLabel(null))
        assertNull(evaluationUnionLabel(scores(scored = 0)))
    }

    @Test
    fun theCoverageLineOnlyAppearsWhenSomethingWasNotScored() {
        assertNull(evaluationCoverageLabel(scores()))
        assertEquals("21 scored · 1 failed", evaluationCoverageLabel(scores(failed = 1)))
        assertEquals("21 scored · 2 failed · 1 skipped", evaluationCoverageLabel(scores(failed = 2, skipped = 1)))
        assertNull(evaluationCoverageLabel(scores(scored = 0, failed = 3)))
    }

    @Test
    fun theDetailsFollowTheRecordWithRecallInFront() {
        val rows = evaluationDetailRows(
            scores(
                groups = listOf(
                    EvaluationGroup(prompt = "1girl, solo", images = 4, recall = 0.7f, unionRecall = 0.8f),
                    EvaluationGroup(prompt = "a castle", images = 1, recall = 0f, unionRecall = 0f),
                )
            )
        )
        assertEquals(2, rows.size)
        assertEquals("1girl, solo", rows[0].prompt)
        assertEquals(4, rows[0].images)
        assertEquals(0.7f, rows[0].perImageRecall)
        assertEquals(0.8f, rows[0].unionRecall)
        assertEquals("a castle", rows[1].prompt)
        assertTrue(evaluationDetailRows(null).isEmpty())
    }

    @Test
    fun anOffenderLineCarriesItsCount() {
        assertEquals("solo ×5", evaluationTagLabel("solo", 5))
        assertEquals("long hair ×1", evaluationTagLabel("long hair", 1))
    }

    @Test
    fun theConfigLabelNamesWhereThePromptsCameFrom() {
        assertEquals(
            "run config · rein_20260911_120000",
            evaluationConfigLabel(evaluation().copy(configSource = "/logs/rein_20260911_120000/config.toml")),
        )
        // A run from before the config copies: its own config, read out of the hparams it recorded.
        assertEquals(
            "run config (from its hparams) · rein_20260911_120000",
            evaluationConfigLabel(
                evaluation().copy(
                    configSource = "/logs/rein_20260911_120000/1790780085.6183703/" +
                        "events.out.tfevents.1790780085.acitehost.592336.1",
                ),
            ),
        )
        assertEquals(
            "current config.toml",
            evaluationConfigLabel(evaluation().copy(configSource = "/repo/AxlTrainer/config.toml")),
        )
        assertEquals("config: unknown", evaluationConfigLabel(evaluation()))
    }

    @Test
    fun aScoredEvaluationWithoutImagesStillBelongsToItsCard() {
        // The point of the predicate: a checkpoint that already held enough images renders nothing,
        // and its scores would otherwise never be shown anywhere.
        val job = evaluation(scores = scores())
        assertFalse(jobHasImages(job))
        assertTrue(jobShowsOnCard(job))
        assertTrue(generatedSampleItems(job).isEmpty())
        // A running one is not "on the card" as a finished job; the running row reports it.
        assertFalse(jobShowsOnCard(evaluation(state = JOB_RUNNING, phase = "tagging")))
    }

    @Test
    fun theCardReportsTheNewestEvaluationUnlessOneIsRunning() {
        val older = evaluation(id = "evaluate_gen_1", startedAt = 1.0, scores = scores(f1 = 0.4f))
        val newer = evaluation(id = "evaluate_gen_2", startedAt = 2.0, scores = scores(f1 = 0.9f))
        val row = checkpointRow(jobs = listOf(older, newer))
        assertEquals("evaluate_gen_2", cardEvaluation(row)?.let { it.id })

        val live = evaluation(id = "evaluate_gen_3", state = JOB_RUNNING, phase = "tagging", startedAt = 3.0)
        val running = checkpointRow(jobs = listOf(older, newer), running = live)
        assertEquals("evaluate_gen_3", cardEvaluation(running)?.let { it.id })
    }

    @Test
    fun aCardWithoutAnEvaluationReportsNone() {
        val sets = GeneratedSampleJob(id = "sets_gen_1", state = JOB_DONE, mode = JOB_MODE_SETS, step = 3050)
        assertNull(cardEvaluation(checkpointRow(jobs = listOf(sets))))
    }

    private fun checkpointRow(
        jobs: List<GeneratedSampleJob>,
        running: GeneratedSampleJob? = null,
    ): CheckpointRow = CheckpointRow(
        checkpoint = checkpoint,
        step = 3050,
        samples = listOf(
            SampleItem(
                filename = "rein_003050_p0_0.png",
                setIndex = 0,
                repeatIdx = 0,
                path = "/out/rein_20260911_120000/rein_samples/rein_003050_p0_0.png",
            )
        ),
        generated = jobs,
        running = running,
    )
}

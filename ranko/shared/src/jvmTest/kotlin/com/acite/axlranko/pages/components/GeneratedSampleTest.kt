package com.acite.axlranko.pages.components

import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.pages.previewList
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GeneratedSampleTest {

    private fun job(
        id: String = "lllj_s003050_gen_20260915_161123",
        state: String = JOB_DONE,
        step: Int? = 3050,
        imagePath: String? = "/out/run_samples/generated/$id.png",
        currentStep: Int = 20,
        totalSteps: Int = 20,
        cfg: Float? = 5f,
        steps: Int? = 20,
        seed: Long? = 12345,
        checkpoint: String = checkpoint(3050).path,
    ) = GeneratedSampleJob(
        id = id,
        state = state,
        step = step,
        cfg = cfg,
        steps = steps,
        seed = seed,
        imagePath = imagePath,
        checkpoint = checkpoint,
        currentStep = currentStep,
        totalSteps = totalSteps,
    )

    private fun checkpoint(step: Int) = CheckpointItem(
        path = "/out/run/rein_s$step/rein.safetensors",
        runId = "rein_20260911_120000",
        dir = "rein_s$step",
        filename = "rein.safetensors",
        step = step,
    )

    private fun sample(step: Int, index: Int = 0) = SampleItem(
        filename = "run_${step.toString().padStart(6, '0')}_$index.png",
        repeatIdx = index,
        path = "/out/run_samples/run_${step.toString().padStart(6, '0')}_$index.png",
    )

    @Test
    fun onlyFinishedJobsWithAnImageCountAsSamples() {
        val jobs = listOf(
            job(id = "done", step = 3050),
            job(id = "running", state = JOB_RUNNING, step = 3050),
            job(id = "failed", state = JOB_ERROR, step = 3050, imagePath = null),
            job(id = "other-step", step = 3000),
        )
        assertEquals(listOf("done"), generatedJobsForStep(jobs, 3050).map { it.id })
        assertTrue(generatedJobsForStep(jobs, null).isEmpty())
        assertTrue(generatedJobsForStep(jobs, 999).isEmpty())
    }

    @Test
    fun thePanelRowShowsThePassItsCheckpointOwns() {
        val target = checkpoint(3050)
        val jobs = listOf(
            job(id = "at-step", step = 3050),
            job(id = "legacy-no-step", step = null, checkpoint = target.path),
            job(id = "other-checkpoint", step = 3050, checkpoint = "/elsewhere/rein.safetensors"),
        )
        // Every job recorded at the step (that rule is unchanged, whichever checkpoint it came
        // from), plus the pass recorded without a step that names this checkpoint.
        assertEquals(
            listOf("at-step", "other-checkpoint", "legacy-no-step"),
            panelJobsForStep(jobs, 3050, target).map { it.id },
        )
        assertEquals(
            listOf("at-step", "other-checkpoint"),
            panelJobsForStep(jobs, 3050, null).map { it.id },
        )
    }

    @Test
    fun theRunningJobIsTheOneBeingFollowed() {
        val running = job(id = "live", state = JOB_RUNNING)
        assertEquals(running, runningJob(listOf(job(id = "done"), running)))
        assertNull(runningJob(listOf(job(id = "done"), job(id = "failed", state = JOB_ERROR))))
        assertNull(runningJob(emptyList()))
    }

    @Test
    fun aWholeSetPassBecomesOneSampleItemPerImage() {
        val pass = GeneratedSampleJob(
            id = "rein_s003050_sets_gen_20260929_031500",
            state = JOB_DONE,
            mode = JOB_MODE_SETS,
            step = 3050,
            files = listOf(
                "/out/run_samples/generated/rein_s003050_sets_gen_20260929_031500_p0_0.png",
                "/out/run_samples/generated/rein_s003050_sets_gen_20260929_031500_p2_1.png",
            ),
            imagesDone = 2,
            totalImages = 2,
        )
        val items = generatedSampleItems(pass)
        assertEquals(2, items.size)
        // The `Pn` badge comes from the file name, counting from zero like the run's own samples.
        assertEquals(listOf(0, 2), items.map { it.setIndex })
        assertEquals(listOf(0, 1), items.map { it.repeatIdx })
        assertTrue(items.all { it.path.endsWith(".png") })
    }

    @Test
    fun aJobWithoutAnImageHasNoSampleItem() {
        assertTrue(generatedSampleItems(job(state = JOB_RUNNING, imagePath = null)).isEmpty())
        val item = generatedSampleItems(job(id = "x", imagePath = "/out/generated/x.png")).single()
        assertEquals("x.png", item.filename)
        assertEquals("/out/generated/x.png", item.path)
        assertEquals(-1, item.repeatIdx)
    }

    @Test
    fun aRunningPassReportsItsOwnProgress() {
        val running = GeneratedSampleJob(
            id = "live",
            state = JOB_RUNNING,
            mode = JOB_MODE_SETS,
            imagesDone = 3,
            totalImages = 12,
            currentSet = 2,
            totalSets = 6,
            currentStep = 12,
            totalSteps = 35,
        )
        assertEquals(
            "set 2/6 · image 3/12 · denoising 12/35",
            generatedJobSetProgress(running),
        )
        assertNull(generatedJobSetProgress(running.copy(state = JOB_DONE)))
        assertNull(generatedJobSetProgress(running.copy(mode = JOB_MODE_SINGLE)))
        assertNull(generatedJobSetProgress(null))
    }

    @Test
    fun theRowIsTrainingSamplesThenGeneratedOnesNewestLast() {
        val slots = sampleSlots(
            training = listOf(sample(3050, 0), sample(3050, 1)),
            jobs = listOf(job(id = "newest"), job(id = "older")),
            sessionJobIds = setOf("newest"),
        )
        assertEquals(4, slots.size)
        assertEquals(listOf(null, null, "older", "newest"), slots.map { it.job?.id })
        assertEquals(listOf(false, false, false, true), slots.map { it.isNew })
        assertTrue(slots.take(2).all { it.job == null })
    }

    @Test
    fun theCaptionAndProgressDescribeTheJob() {
        assertEquals("CFG 5 · 20 steps · seed 12345", generatedJobCaption(job()))
        assertEquals("CFG 7.5 · 12 steps", generatedJobCaption(job(cfg = 7.5f, steps = 12, seed = null)))
        assertEquals("generated", generatedJobCaption(job(cfg = null, steps = null, seed = null)))

        assertEquals(
            "denoising 5/20",
            generatedJobProgress(job(state = JOB_RUNNING, currentStep = 5)),
        )
        assertEquals("denoising…", generatedJobProgress(job(state = JOB_RUNNING, totalSteps = 0)))
        assertNull(generatedJobProgress(job(state = JOB_DONE)))
        assertNull(generatedJobProgress(null))
    }

    @Test
    fun thePreviewListIsTheSectionOrder() {
        // The page draws the checkpoints newest first, each card's training samples then the images
        // of the pass that joined it; the fullscreen preview cycles exactly that list.
        val list = previewList(
            checkpoints = listOf(checkpoint(3050), checkpoint(3000)),
            samples = mapOf(
                "3050" to listOf(sample(3050, 0), sample(3050, 1)),
                "3000" to listOf(sample(3000, 0)),
            ),
            jobs = listOf(job(id = "gen", step = 3050)),
        )
        assertEquals(4, list.size)
        assertEquals(sample(3050, 0).path, list[0].path)
        assertEquals(sample(3050, 1).path, list[1].path)
        assertTrue(list[2].path.endsWith("gen.png"))
        assertEquals(sample(3000, 0).path, list[3].path)
    }

    @Test
    fun everyImageOfASetPassIsInThePreviewList() {
        val pass = GeneratedSampleJob(
            id = "pass",
            state = JOB_DONE,
            mode = JOB_MODE_SETS,
            step = 3050,
            checkpoint = checkpoint(3050).path,
            files = listOf("/out/generated/pass_p0_0.png", "/out/generated/pass_p1_0.png"),
        )
        val list = previewList(
            checkpoints = listOf(checkpoint(3050)),
            samples = mapOf("3050" to listOf(sample(3050, 0))),
            jobs = listOf(pass),
        )
        assertEquals(3, list.size)
        assertEquals(sample(3050, 0).path, list[0].path)
        assertEquals(listOf(0, 1), list.drop(1).map { it.setIndex })
    }

    @Test
    fun aPassWithNoStepIsStillInThePreviewList() {
        // The reported bug: the card showed the images (it matches them by checkpoint path) while
        // the preview, built by step alone, did not — so clicking a thumbnail did nothing.
        val target = checkpoint(3050)
        val list = previewList(
            checkpoints = listOf(target),
            samples = emptyMap(),
            jobs = listOf(
                GeneratedSampleJob(
                    id = "legacy_sets_gen_1",
                    state = JOB_DONE,
                    mode = JOB_MODE_SETS,
                    step = null,
                    checkpoint = target.path,
                    files = listOf("/out/generated/legacy_sets_gen_1_p0_0.png"),
                ),
            ),
        )
        assertEquals(listOf("legacy_sets_gen_1_p0_0.png"), list.map { it.filename })
    }

    @Test
    fun aStepWithoutACheckpointKeepsItsImagesInThePreviewList() {
        val list = previewList(
            checkpoints = emptyList(),
            samples = mapOf("3000" to listOf(sample(3000, 0))),
            jobs = listOf(job(id = "orphan", step = 3100)),
        )
        // Newest leftover step first: the orphaned pass's own row, then the step that only has
        // training samples left.
        assertEquals(
            listOf("orphan.png", sample(3000, 0).filename),
            list.map { it.filename },
        )
    }

    @Test
    fun withoutGeneratedImagesThePreviewListIsTheRunsSamples() {
        val list = previewList(
            checkpoints = listOf(checkpoint(3050), checkpoint(3000)),
            samples = mapOf("3050" to listOf(sample(3050, 0)), "3000" to listOf(sample(3000, 1))),
            jobs = listOf(job(state = JOB_RUNNING, imagePath = null)),
        )
        assertEquals(
            listOf(sample(3050, 0).path, sample(3000, 1).path),
            list.map { it.path },
        )
    }
}

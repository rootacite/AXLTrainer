package com.acite.axlranko.pages.components

import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The Dashboard's Checkpoints section: which checkpoint owns which images. */
class CheckpointListTest {

    private fun checkpoint(step: Int?, path: String? = null, final: Boolean = false, dir: String = "") =
        CheckpointItem(
            path = path ?: "/out/run/rein_s${step ?: 0}/${dir.ifBlank { "rein" }}.safetensors",
            runId = "rein_20260911_120000",
            dir = dir.ifBlank { "rein_s${step ?: 0}" },
            filename = "rein.safetensors",
            step = step,
            final = final,
            sizeBytes = 1024,
        )

    private fun sample(step: Int, index: Int = 0) = SampleItem(
        filename = "rein_${step.toString().padStart(6, '0')}_$index.png",
        repeatIdx = index,
        path = "/out/run/rein_samples/rein_${step.toString().padStart(6, '0')}_$index.png",
    )

    private fun job(
        id: String,
        step: Int?,
        checkpointPath: String = "/out/run/rein_s3050/rein.safetensors",
        state: String = JOB_DONE,
        mode: String = JOB_MODE_SETS,
        files: List<String> = listOf("/out/run/rein_samples/generated/${id}_p0_0.png"),
    ) = GeneratedSampleJob(
        id = id,
        state = state,
        mode = mode,
        step = step,
        checkpoint = checkpointPath,
        files = files,
    )

    @Test
    fun eachCheckpointLeadsItsRowWithItsOwnSamples() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050), checkpoint(3000)),
            samples = mapOf(
                "3050" to listOf(sample(3050, 0), sample(3050, 1)),
                "3000" to listOf(sample(3000, 0)),
            ),
            jobs = emptyList(),
        )
        assertEquals(listOf(3050, 3000), rows.map { it.step })
        assertEquals(2, rows[0].samples.size)
        assertEquals(1, rows[1].samples.size)
        assertTrue(rows.all { it.checkpoint != null && it.running == null })
    }

    @Test
    fun aCheckpointWithoutSamplesKeepsItsRow() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = mapOf("3000" to listOf(sample(3000, 0))),
            jobs = emptyList(),
        )
        // The checkpoint first (nothing for it yet), then the step whose weights are gone.
        assertEquals(listOf(3050, 3000), rows.map { it.step })
        assertTrue(rows[0].samples.isEmpty())
        assertNull(rows[1].checkpoint)
        assertEquals(1, rows[1].samples.size)
    }

    @Test
    fun generatedImagesRideWithTheirStep() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = emptyMap(),
            jobs = listOf(job("a_sets_gen_1", 3050), job("b_sets_gen_2", 3050, state = JOB_RUNNING)),
        )
        assertEquals(1, rows.size)
        assertEquals(listOf("a_sets_gen_1"), rows[0].generated.map { it.id })
        assertEquals("b_sets_gen_2", rows[0].running?.id)
    }

    @Test
    fun aStepThatOnlyHasAGeneratedImageIsStillListed() {
        val rows = checkpointRows(
            checkpoints = emptyList(),
            samples = emptyMap(),
            jobs = listOf(job("a_sets_gen_1", 3100)),
        )
        assertEquals(listOf(3100), rows.map { it.step })
        assertNull(rows[0].checkpoint)
        assertEquals(listOf("a_sets_gen_1"), rows[0].generated.map { it.id })
    }

    @Test
    fun aRunningJobThatFailedIsNotASample() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = emptyMap(),
            jobs = listOf(job("failed", 3050, state = JOB_ERROR)),
        )
        assertTrue(rows[0].generated.isEmpty())
        assertNull(rows[0].running)
    }

    @Test
    fun aJobWithoutImagesIsNotListed() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = emptyMap(),
            jobs = listOf(job("empty", 3050, files = emptyList())),
        )
        // A finished job with no file at all is not a sample; with no images the step has no row.
        assertTrue(rows[0].generated.isEmpty())
        assertEquals(listOf(3050), rows.map { it.step })
    }

    @Test
    fun theRowLabelNamesTheStepAndTheFinalOne() {
        assertEquals("step 3050 · final", checkpointRowLabel(
            checkpointRows(listOf(checkpoint(3050, final = true)), emptyMap(), emptyList()).first()
        ))
        assertEquals("step 3050", checkpointRowLabel(
            checkpointRows(listOf(checkpoint(3050)), emptyMap(), emptyList()).first()
        ))
        assertEquals("step unknown", checkpointRowLabel(
            checkpointRows(listOf(checkpoint(null)), emptyMap(), emptyList()).first()
        ))
    }

    @Test
    fun theRowsImagesAreTrainingFirstThenGenerated() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = mapOf("3050" to listOf(sample(3050, 0))),
            jobs = listOf(job("a_sets_gen_1", 3050, files = listOf(
                "/out/run/rein_samples/generated/a_sets_gen_1_p0_0.png",
                "/out/run/rein_samples/generated/a_sets_gen_1_p1_0.png",
            ))),
        )
        val images = rows[0].images()
        assertEquals(3, images.size)
        assertEquals(sample(3050, 0).path, images[0].path)
        assertEquals(0, images[1].setIndex)
        assertEquals(1, images[2].setIndex)
    }
}

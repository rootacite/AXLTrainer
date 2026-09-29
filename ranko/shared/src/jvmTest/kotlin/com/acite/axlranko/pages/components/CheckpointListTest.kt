package com.acite.axlranko.pages.components

import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
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
        checkpointPath: String = checkpoint(3050).path,
        state: String = JOB_DONE,
        mode: String = JOB_MODE_SETS,
        files: List<String> = listOf("/out/run/rein_samples/generated/${id}_p0_0.png"),
        startedAt: Double = 0.0,
    ) = GeneratedSampleJob(
        id = id,
        state = state,
        mode = mode,
        step = step,
        checkpoint = checkpointPath,
        files = files,
        startedAt = startedAt,
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
    fun aJobRidesWithTheCheckpointItNames() {
        // The path decides, whatever step the job recorded: a pass rendered from this checkpoint
        // belongs on its card.
        val target = checkpoint(3050)
        val rows = checkpointRows(
            checkpoints = listOf(target),
            samples = emptyMap(),
            jobs = listOf(job("named_sets_gen_1", step = 999, checkpointPath = target.path)),
        )
        assertEquals(1, rows.size)
        assertEquals(listOf("named_sets_gen_1"), rows[0].generated.map { it.id })
        assertNull(rows[0].samples.firstOrNull())
    }

    @Test
    fun aPassWhoseCheckpointIsGoneStillGetsARow() {
        // The step has a card, but that card is a different checkpoint: the orphaned pass must not
        // vanish, and the card's own samples must not be repeated in the extra row.
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = mapOf("3050" to listOf(sample(3050, 0))),
            jobs = listOf(job("orphan_sets_gen_1", step = 3050, checkpointPath = "/gone/rein.safetensors")),
        )
        assertEquals(2, rows.size)
        assertTrue(rows[0].generated.isEmpty())
        assertEquals(1, rows[0].samples.size)
        assertNull(rows[1].checkpoint)
        assertEquals(3050, rows[1].step)
        assertTrue(rows[1].samples.isEmpty())
        assertEquals(listOf("orphan_sets_gen_1"), rows[1].generated.map { it.id })
    }

    @Test
    fun aJobWithNeitherAPathNorAStepStillGetsARow() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = emptyMap(),
            jobs = listOf(job("nameless_sets_gen_1", step = null, checkpointPath = "")),
        )
        assertEquals(2, rows.size)
        assertNull(rows[1].checkpoint)
        assertNull(rows[1].step)
        assertEquals(listOf("nameless_sets_gen_1"), rows[1].generated.map { it.id })
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
    fun aRunningJobIsShownAsProgressNotAsASample() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = emptyMap(),
            jobs = listOf(job("running", 3050, state = JOB_RUNNING)),
        )
        assertTrue(rows[0].generated.isEmpty())
        assertEquals("running", rows[0].running?.id)
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
    fun theRangeCoversTheCheckpointsInsideItOldestFirst() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3000), checkpoint(1000), checkpoint(2000), checkpoint(null)),
            samples = emptyMap(),
            jobs = emptyList(),
        )
        assertEquals(listOf(1000, 2000, 3000), checkpointSteps(rows))
        assertEquals(
            listOf(1000, 2000),
            checkpointsInRange(rows, 0, 2500).map { it.step },
        )
        assertEquals(listOf(2000, 3000), checkpointsInRange(rows, 2000, 3000).map { it.step })
        assertTrue(checkpointsInRange(rows, 5000, 6000).isEmpty())
        // A checkpoint without a step can neither be placed nor sampled by a range.
        assertTrue(checkpointsInRange(rows, 0, 1_000_000).none { it.step == null })
    }

    @Test
    fun theRangeCheckExplainsItself() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(1000), checkpoint(3000)),
            samples = emptyMap(),
            jobs = emptyList(),
        )
        assertEquals("Enter a step range", batchRangeError(rows, "", "10"))
        assertEquals("Steps must be whole numbers", batchRangeError(rows, "x", "10"))
        assertEquals("Enter a step range", batchRangeError(rows, "10", ""))
        assertEquals("No checkpoints between step 1500 and 2500", batchRangeError(rows, "1500", " 2500 "))
        assertEquals("From must not be greater than To", batchRangeError(rows, "30", "10"))
        assertEquals("No checkpoints between step 1500 and 2500", batchRangeError(rows, "1500", "2500"))
        assertNull(batchRangeError(rows, "1000", "3000"))
        assertNull(batchRangeError(rows, "0", "9999999"))
        // A run with no stepped checkpoints says so rather than offering an empty batch.
        val noSteps = checkpointRows(listOf(checkpoint(null)), emptyMap(), emptyList())
        assertEquals("This run has no checkpoints with a step", batchRangeError(noSteps, "0", "10"))
    }

    @Test
    fun aCancelledOrFailedPassStillShowsTheImagesItWrote() {
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = emptyMap(),
            jobs = listOf(
                job("cancelled_sets_gen_1", step = 3050, state = JOB_CANCELLED, startedAt = 10.0),
                job("failed_sets_gen_2", step = 3050, state = JOB_ERROR, startedAt = 20.0),
            ),
        )
        // Both wrote files before stopping, and files on disk belong on the card.
        assertEquals(
            listOf("failed_sets_gen_2", "cancelled_sets_gen_1"),
            rows[0].generated.map { it.id },
        )
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
        // The card's own assembly: the step's training samples first, then the generated ones.
        val slots = sampleSlots(rows[0].samples, rows[0].generated, emptySet())
        assertEquals(3, slots.size)
        assertEquals(sample(3050, 0).path, slots[0].item.path)
        assertTrue(slots[0].job == null)
        assertEquals(0, slots[1].item.setIndex)
        assertEquals(1, slots[2].item.setIndex)
        assertTrue(slots.drop(1).all { it.job?.id == "a_sets_gen_1" })
    }

    @Test
    fun pinnedCheckpointsLeadTheSection() {
        val checkpoints = listOf(checkpoint(3050), checkpoint(3000), checkpoint(2950))
        val plain = checkpointRows(checkpoints, emptyMap(), emptyList())
        assertEquals(listOf(3050, 3000, 2950), plain.map { it.step })

        val pinned = checkpointRows(
            checkpoints,
            emptyMap(),
            emptyList(),
            pinned = setOf(checkpoint(2950).path),
        )
        assertEquals(listOf(2950, 3050, 3000), pinned.map { it.step })
        assertEquals(listOf(true, false, false), pinned.map { it.pinned })
    }

    @Test
    fun pinningIsAStablePartitionNotAReSort() {
        // Inside the pinned block the section keeps its own newest-step-first order, so a card the
        // user pinned does not move when a new checkpoint is saved.
        val checkpoints = listOf(checkpoint(3100), checkpoint(3050), checkpoint(3000))
        val rows = checkpointRows(
            checkpoints,
            emptyMap(),
            emptyList(),
            pinned = setOf(checkpoint(3000).path, checkpoint(3050).path),
        )
        assertEquals(listOf(3050, 3000, 3100), rows.map { it.step })
    }

    @Test
    fun aPinnedPathWithNoCheckpointIsNotDrawn() {
        // Reset deletes weights and keeps the pin file; the pin survives on disk, but no card can
        // stand for a checkpoint that is gone.
        val rows = checkpointRows(
            checkpoints = listOf(checkpoint(3050)),
            samples = emptyMap(),
            jobs = emptyList(),
            pinned = setOf("/out/run/rein_s0200/rein.safetensors"),
        )
        assertEquals(listOf(3050), rows.map { it.step })
        assertEquals(listOf(false), rows.map { it.pinned })
    }

    @Test
    fun aSamplesOnlyRowIsNeverPinned() {
        val rows = checkpointRows(
            checkpoints = emptyList(),
            samples = mapOf("3000" to listOf(sample(3000, 0))),
            jobs = emptyList(),
            pinned = setOf("samples-only-3000"),
        )
        assertEquals(1, rows.size)
        assertFalse(rows[0].pinned)
    }

    @Test
    fun thePinnedRowsAreTheOnesTheSectionShowsFirst() {
        // The preview list follows the rows, so a pinned card's images open in the order drawn.
        val checkpoints = listOf(checkpoint(3050), checkpoint(3000))
        val rows = checkpointRows(
            checkpoints = checkpoints,
            samples = mapOf("3050" to listOf(sample(3050, 0)), "3000" to listOf(sample(3000, 0))),
            jobs = emptyList(),
            pinned = setOf(checkpoint(3000).path),
        )
        assertEquals(
            listOf(sample(3000, 0).path, sample(3050, 0).path),
            sectionImages(rows).map { it.path },
        )
    }
}

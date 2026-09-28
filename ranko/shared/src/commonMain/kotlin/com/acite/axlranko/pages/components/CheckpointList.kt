package com.acite.axlranko.pages.components

import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem

/**
 * One line of the Dashboard's Checkpoints section: a LoRA checkpoint with the samples written at
 * its step, or — when [checkpoint] is null — a step that only has images left (the weights were
 * deleted by a Reset, or a generation outlived its checkpoint).
 */
internal data class CheckpointRow(
    val checkpoint: CheckpointItem?,
    val step: Int?,
    val samples: List<SampleItem>,
    /** Finished generations at this step, newest first. */
    val generated: List<GeneratedSampleJob>,
    /** A pass still rendering for this checkpoint, if any. */
    val running: GeneratedSampleJob?,
)

/**
 * The Checkpoints section, in the order [checkpoints] arrived (newest step first).
 *
 * The section is checkpoint-driven — an image always belongs to the checkpoint its save wrote — so
 * a job is attached by the checkpoint it names, and the step is only the fallback for a record that
 * names none. A step whose checkpoint is gone keeps its row, so Reset (which deletes weights and
 * keeps samples) never hides images, and any generated image a card did not claim lands in a
 * `samples only` row rather than nowhere at all.
 */
internal fun checkpointRows(
    checkpoints: List<CheckpointItem>,
    samples: Map<String, List<SampleItem>>,
    jobs: List<GeneratedSampleJob>,
): List<CheckpointRow> {
    val done = jobs.filter { it.state == JOB_DONE && generatedSampleItems(it).isNotEmpty() }
    val rows = mutableListOf<CheckpointRow>()
    val claimedSteps = mutableSetOf<Int>()
    val claimedJobs = mutableSetOf<String>()

    for (checkpoint in checkpoints) {
        val step = checkpoint.step
        if (step != null) claimedSteps += step
        val own = done.filter { belongsTo(it, checkpoint) }
        claimedJobs += own.map { it.id }
        rows += CheckpointRow(
            checkpoint = checkpoint,
            step = step,
            samples = step?.let { samples[it.toString()] }.orEmpty(),
            generated = own.sortedByDescending { it.startedAt },
            running = runningJobForCheckpoint(jobs, checkpoint),
        )
    }

    // A row of its own for a step with images whose checkpoint is gone, and for any pass no card
    // claimed — its checkpoint was deleted, or another card sits at the same step under a
    // different path. A step a card already shows does not repeat that step's samples here.
    val unclaimed = done.filter { it.id !in claimedJobs }
    val keys = (
        samples.keys.filter { key -> key.toIntOrNull()?.let { it in claimedSteps } != true } +
            unclaimed.mapNotNull { it.step?.toString() }
        )
        .distinct()
        .sortedByDescending { it.toIntOrNull() ?: Int.MIN_VALUE }
    for (key in keys) {
        val step = key.toIntOrNull()
        val own = if (step != null && step in claimedSteps) emptyList() else samples[key].orEmpty()
        val generated = unclaimed.filter { it.step == step }.sortedByDescending { it.startedAt }
        if (own.isEmpty() && generated.isEmpty()) continue
        rows += CheckpointRow(
            checkpoint = null,
            step = step,
            samples = own,
            generated = generated,
            running = null,
        )
    }

    // Last resort: a pass from a checkpoint whose name and metadata both carry no step, which no
    // row above could claim. It still gets a card, so its images are reachable.
    val homeless = unclaimed.filter { it.step == null }
    if (homeless.isNotEmpty() && rows.none { it.checkpoint == null && it.step == null }) {
        rows += CheckpointRow(
            checkpoint = null,
            step = null,
            samples = emptyList(),
            generated = homeless.sortedByDescending { it.startedAt },
            running = null,
        )
    }
    return rows
}

/** A job belongs to the card it names; the step is the fallback for a record that names none. */
private fun belongsTo(job: GeneratedSampleJob, checkpoint: CheckpointItem): Boolean =
    job.checkpoint == checkpoint.path ||
        (job.checkpoint.isBlank() && job.step != null && job.step == checkpoint.step)

/**
 * Lazy-list key of a row: the checkpoint file, which is unique, or the step of a `samples only`
 * row. Two checkpoints may share a step (a `_s{step:06d}` save that the run's end turned into a
 * `_final` one), so the path is the only safe key.
 */
internal fun checkpointRowKey(row: CheckpointRow): String =
    row.checkpoint?.path ?: "samples-only-${row.step ?: "unknown"}"

/** `step 3050 · final`, or `unknown step` for a file name that carries none. */
internal fun checkpointRowLabel(row: CheckpointRow): String = when (val step = row.step) {
    null -> "step unknown"
    -1 -> "step unknown"
    else -> if (row.checkpoint?.final == true) "step $step · final" else "step $step"
}

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
    val done = jobs.filter { jobHasImages(it) }
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

/**
 * Every image the Checkpoints section shows, in the order the page shows it: a card's training
 * samples first, then the images of the passes that joined it, then the `samples only` rows.
 *
 * The fullscreen preview cycles this list, and a click resolves its index here — so building both
 * from the same rows is what keeps a rendered image openable. A pass whose job records no step (or
 * whose checkpoint is gone) is in a row like any other, rather than being dropped by a step-only
 * lookup and leaving its thumbnail inert.
 */
internal fun sectionImages(rows: List<CheckpointRow>): List<SampleItem> =
    rows.flatMap { row -> sampleSlots(row.samples, row.generated, emptySet()).map { it.item } }

/** A job belongs to the card it names; the step is the fallback for a record that names none. */
private fun belongsTo(job: GeneratedSampleJob, checkpoint: CheckpointItem): Boolean =
    job.checkpoint == checkpoint.path ||
        (job.checkpoint.isBlank() && job.step != null && job.step == checkpoint.step)

/** The steps of the checkpoints that have one, ascending — what a range can cover. */
internal fun checkpointSteps(rows: List<CheckpointRow>): List<Int> =
    rows.mapNotNull { it.checkpoint?.step }.distinct().sorted()

/** The checkpoints a `[from, to]` range covers, oldest step first (the order a batch renders in). */
internal fun checkpointsInRange(rows: List<CheckpointRow>, from: Int, to: Int): List<CheckpointItem> =
    rows.mapNotNull { it.checkpoint }
        .filter { it.step != null && it.step in from..to }
        .sortedWith(compareBy({ it.step ?: 0 }, { it.dir }))

/**
 * Why a range cannot be started, or null when it can. The strings the user typed are checked here
 * so the button can be off with a reason beside it, before anything is sent.
 */
internal fun batchRangeError(rows: List<CheckpointRow>, from: String, to: String): String? {
    val fromStep = from.trim().toIntOrNull()
    val toStep = to.trim().toIntOrNull()
    if (from.trim().isEmpty() || to.trim().isEmpty()) return "Enter a step range"
    if (fromStep == null || toStep == null) return "Steps must be whole numbers"
    if (fromStep < 0 || toStep < 0) return "Steps must be >= 0"
    if (fromStep > toStep) return "From must not be greater than To"
    if (checkpointSteps(rows).isEmpty()) return "This run has no checkpoints with a step"
    if (checkpointsInRange(rows, fromStep, toStep).isEmpty()) {
        return "No checkpoints between step $fromStep and $toStep"
    }
    return null
}

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

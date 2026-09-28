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
 * The section is checkpoint-driven — an image always belongs to the checkpoint its save wrote —
 * but a step whose checkpoint is gone keeps its row, so Reset (which deletes weights and keeps
 * samples) never hides images.
 */
internal fun checkpointRows(
    checkpoints: List<CheckpointItem>,
    samples: Map<String, List<SampleItem>>,
    jobs: List<GeneratedSampleJob>,
): List<CheckpointRow> {
    val done = jobs.filter { it.state == JOB_DONE && generatedSampleItems(it).isNotEmpty() }
    val rows = mutableListOf<CheckpointRow>()
    val claimed = mutableSetOf<Int>()

    for (checkpoint in checkpoints) {
        val step = checkpoint.step
        if (step != null) claimed += step
        rows += CheckpointRow(
            checkpoint = checkpoint,
            step = step,
            samples = step?.let { samples[it.toString()] }.orEmpty(),
            generated = done.filter { it.step == step }.sortedByDescending { it.startedAt },
            running = runningJobForCheckpoint(jobs, checkpoint),
        )
    }

    val leftover = (samples.keys + done.mapNotNull { it.step?.toString() })
        .distinct()
        .filter { key -> key.toIntOrNull()?.let { it in claimed } != true }
        .sortedByDescending { it.toIntOrNull() ?: Int.MIN_VALUE }
    for (key in leftover) {
        val step = key.toIntOrNull()
        val own = samples[key].orEmpty()
        val generated = done.filter { it.step == step }.sortedByDescending { it.startedAt }
        if (own.isEmpty() && generated.isEmpty()) continue
        rows += CheckpointRow(
            checkpoint = null,
            step = step,
            samples = own,
            generated = generated,
            running = null,
        )
    }
    return rows
}

/** Every image of a row, training samples first, then the generated ones in render order. */
internal fun CheckpointRow.images(): List<SampleItem> =
    samples + generated.asReversed().flatMap { generatedSampleItems(it) }

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

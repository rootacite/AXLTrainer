package com.acite.axlranko.pages.components

import com.acite.axlranko.model.EvaluationGroup
import com.acite.axlranko.model.EvaluationScores
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.util.formatFixed

/** The depth bounds api.py enforces; the dialog mirrors them so a bad click costs no GPU time. */
internal const val MAX_EVALUATION_DEPTH = 512

/** The tagger floor the card starts on, same default `dataset_tag` uses. */
internal const val DEFAULT_EVALUATION_THRESHOLD = 0.35f

/** How many offenders of each kind the details block lists. */
internal const val EVALUATION_TOP_TAGS = 8

/**
 * Why an evaluation cannot be sent, or null when it can. The strings the user typed are checked
 * here, so the button can be off with a reason beside it before anything reaches the helper.
 */
internal fun evaluationRequestError(depthText: String, thresholdText: String): String? {
    val depth = depthText.trim()
    if (depth.isEmpty()) return "Enter a depth"
    val count = depth.toIntOrNull() ?: return "Depth must be a whole number"
    if (count < 1) return "Depth must be at least 1"
    if (count > MAX_EVALUATION_DEPTH) return "Depth must be at most $MAX_EVALUATION_DEPTH"

    val threshold = thresholdText.trim().replace(',', '.')
    if (threshold.isEmpty()) return "Enter a threshold"
    val parsed = threshold.toFloatOrNull() ?: return "Threshold must be a number"
    if (parsed < 0f || parsed > 1f) return "Threshold must be between 0 and 1"
    return null
}

/** The depth a dialog opens on: what the checkpoint already shows, at least one image. */
internal fun evaluationDefaultDepth(existingImages: Int): Int = existingImages.coerceAtLeast(1)

/**
 * `Evaluating · rendering 3/4`, then `tagging 8/21` — the phase names what the counters count, since
 * the scored set grows by the rendered images at the handover.
 */
internal fun evaluationProgressLabel(job: GeneratedSampleJob): String {
    val counted = if (job.totalImages > 0) {
        "${job.imagesDone.coerceAtMost(job.totalImages)}/${job.totalImages}"
    } else {
        ""
    }
    val stage = when (job.phase) {
        "rendering" -> "rendering"
        "tagging" -> "tagging"
        "scoring" -> "scoring"
        else -> ""
    }
    val detail = when {
        stage.isEmpty() -> if (counted.isEmpty()) "" else "image $counted"
        stage == "scoring" -> stage
        counted.isEmpty() -> stage
        else -> "$stage $counted"
    }
    return if (detail.isEmpty()) "Evaluating" else "Evaluating · $detail"
}

/** `F1 0.66 · P 0.71 · R 0.62 · 21 images`, the per-image board the card leads with. */
internal fun evaluationScoreLabel(scores: EvaluationScores?): String? {
    if (scores == null) return null
    if (scores.imagesScored <= 0) return null
    val images = if (scores.imagesScored == 1) "1 image" else "${scores.imagesScored} images"
    return "F1 ${formatFixed(scores.f1, 2)} · P ${formatFixed(scores.precision, 2)} · " +
        "R ${formatFixed(scores.recall, 2)} · $images"
}

/** `union F1 0.74 · P 0.78 · R 0.71`, the pooled board, on its own line under the score. */
internal fun evaluationUnionLabel(scores: EvaluationScores?): String? {
    if (scores == null || scores.imagesScored <= 0) return null
    return "union F1 ${formatFixed(scores.unionF1, 2)} · P ${formatFixed(scores.unionPrecision, 2)} · " +
        "R ${formatFixed(scores.unionRecall, 2)}"
}

/** `21 scored · 2 failed · 1 skipped`, or null when every image was scored. */
internal fun evaluationCoverageLabel(scores: EvaluationScores?): String? {
    if (scores == null || scores.imagesScored <= 0) return null
    val parts = mutableListOf("${scores.imagesScored} scored")
    if (scores.imagesFailed > 0) parts += "${scores.imagesFailed} failed"
    if (scores.imagesSkipped > 0) parts += "${scores.imagesSkipped} skipped"
    return if (parts.size == 1) null else parts.joinToString(" · ")
}

/** One prompt's line in the details block: what it asked for, and both boards for it. */
internal data class EvaluationDetailRow(
    val prompt: String,
    val images: Int,
    val perImageF1: Float,
    val unionF1: Float,
)

/** The per-prompt breakdown, the way the record orders it: most images first. */
internal fun evaluationDetailRows(scores: EvaluationScores?): List<EvaluationDetailRow> =
    scores?.groups.orEmpty().map { group: EvaluationGroup ->
        EvaluationDetailRow(
            prompt = group.prompt,
            images = group.images,
            perImageF1 = group.f1,
            unionF1 = group.unionF1,
        )
    }

/** `solo ×5`: a tag that cost precision (an extra) or recall (a miss), with its count. */
internal fun evaluationTagLabel(tag: String, count: Int): String = "$tag ×$count"

/**
 * The evaluation a card reports on: the pass still running for it, else the newest finished one.
 * A checkpoint can hold several over its life; the card shows what happened last.
 */
internal fun cardEvaluation(row: CheckpointRow): GeneratedSampleJob? {
    row.running?.takeIf { isEvaluation(it) }?.let { return it }
    return row.generated.filter { isEvaluation(it) }.maxByOrNull { it.startedAt }
}

/** A run directory is `{output_name}_{YYYYMMDD}_{HHMMSS}`, the same shape the run list shows. */
private val RUN_DIR_NAME = Regex(""".+_\d{8}_\d{6}(?:_\d+)?""")

/**
 * Where an evaluation's prompts came from: `run config · rein_2026…` (the copy the run saved beside
 * its logs), `run config (from its hparams) · rein_2026…` (a run from before those copies, whose
 * config was read back out of the TensorBoard hparams it recorded at startup), `current config.toml`
 * (nothing of the run's own survived), or `config: unknown`. The run is named either way; only the
 * repo-wide fallback cannot say which run the prompts belong to.
 */
internal fun evaluationConfigLabel(job: GeneratedSampleJob): String {
    val source = job.configSource
    if (source.isBlank()) return "config: unknown"
    val runId = source.split('/').lastOrNull { RUN_DIR_NAME.matches(it) } ?: return "current config.toml"
    val hparams = source.substringAfterLast('/').startsWith("events.out.tfevents.")
    return if (hparams) "run config (from its hparams) · $runId" else "run config · $runId"
}

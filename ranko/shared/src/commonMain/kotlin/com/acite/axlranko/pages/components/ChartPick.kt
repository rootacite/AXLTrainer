package com.acite.axlranko.pages.components

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.util.formatFourDecimals
import com.acite.axlranko.util.formatScientificTwoDecimals
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.abs
import kotlin.math.roundToInt

/** Plot insets shared by the chart's draw, pan/zoom math and the Ctrl+click step mapping. */
internal const val PLOT_LEFT_PADDING = 52f
internal const val PLOT_BOTTOM_PADDING = 28f

/** Pointer travel (px) above which a Ctrl+press counts as a drag rather than a click. */
internal const val CLICK_MAX_TRAVEL = 6f

/** Window in which a second click counts as a double click (~the usual desktop default). */
internal const val DOUBLE_CLICK_WINDOW_MILLIS = 400L

/**
 * True when a click at [currentMillis] completes the double click started at [previousMillis], where
 * `0` means "no click pending". Only the interval matters: the two clicks may be anywhere on the
 * chart, and a third quick click starts a new pair instead of picking again.
 */
internal fun completesDoubleClick(
    previousMillis: Long,
    currentMillis: Long,
    windowMillis: Long = DOUBLE_CLICK_WINDOW_MILLIS,
): Boolean {
    if (previousMillis <= 0L) return false
    return currentMillis - previousMillis in 0..windowMillis
}

/** Job states as api.py writes them into the job file. */
internal const val JOB_RUNNING = "running"
internal const val JOB_DONE = "done"
internal const val JOB_ERROR = "error"
/** A job that was asked to stop and finished stopping: its written images still count. */
internal const val JOB_CANCELLED = "cancelled"

/** Sample images shown side by side without scrolling. */
internal const val SAMPLES_PER_ROW = 3

/**
 * Panel sizing: the card is as wide as its content needs, capped by the window. The defaults are
 * intentionally generous (the samples are the point of the panel) and the user can drag the
 * bottom-right grip to override them for the session.
 */
internal val PANEL_CARD_PADDING = 28.dp // PorcelainCard's 14 dp horizontal padding, both sides
internal val PANEL_TEXT_MIN_WIDTH = 760.dp // enough for the checkpoint/path lines to read well
internal val PANEL_MAX_WIDTH = 1440.dp
internal val PANEL_MAX_HEIGHT = 1120.dp
internal val PANEL_MIN_WIDTH = 360.dp
internal val PANEL_MIN_HEIGHT = 260.dp
/** Left between a page-sized overlay (or a dialog) and the window edge, on every side. */
internal val PAGE_PANEL_MARGIN = 24.dp
internal val SAMPLE_SLOT_SPACING = 20.dp
internal val SAMPLE_THUMB_TARGET_WIDTH = 400.dp
internal val SAMPLE_THUMB_MIN_WIDTH = 144.dp
/** Sane ceiling for a dragged-wider panel: a one-image row should not become a full-width poster. */
internal val SAMPLE_THUMB_MAX_WIDTH = 520.dp

/** Height / width of a sample slot. 3:2 matches the usual SDXL sample size, so nothing is letterboxed. */
internal const val SAMPLE_THUMB_ASPECT = 2f / 3f

/** Data-space step under canvas x, or null when the press landed in the axis gutter. */
internal fun stepAtPlotX(x: Float, plotWidth: Float, xMin: Float, xRange: Float): Float? {
    if (plotWidth <= 0f || x < PLOT_LEFT_PADDING) return null
    val fraction = ((x - PLOT_LEFT_PADDING) / plotWidth).coerceIn(0f, 1f)
    return xMin + fraction * xRange
}

/**
 * Width of one sample slot, so that [slots] of them fit inside [maxPanelWidth] alongside the card
 * padding and the gaps between them. Never grows past [SAMPLE_THUMB_TARGET_WIDTH]. Used to derive the
 * panel's *default* width only; a laid-out panel uses [sampleSlotWidth]
 */
internal fun sampleThumbWidth(maxPanelWidth: Dp, slots: Int): Dp {
    if (slots <= 0) return SAMPLE_THUMB_TARGET_WIDTH
    val taken = PANEL_CARD_PADDING + SAMPLE_SLOT_SPACING * (slots - 1)
    return ((maxPanelWidth - taken) / slots).coerceIn(SAMPLE_THUMB_MIN_WIDTH, SAMPLE_THUMB_TARGET_WIDTH)
}

/**
 * Slot width that makes [slots] images fill [panelWidth]. Unlike [sampleThumbWidth] the ceiling is
 * high enough that dragging the panel wider grows the pictures instead of the empty card space.
 */
internal fun sampleSlotWidth(panelWidth: Dp, slots: Int): Dp {
    if (slots <= 0) return SAMPLE_THUMB_TARGET_WIDTH
    val taken = PANEL_CARD_PADDING + SAMPLE_SLOT_SPACING * (slots - 1)
    return ((panelWidth - taken) / slots).coerceIn(SAMPLE_THUMB_MIN_WIDTH, SAMPLE_THUMB_MAX_WIDTH)
}

/** Slot columns for a sample count, so a one-image step does not stretch across the whole panel. */
internal fun sampleColumns(count: Int): Int = count.coerceIn(1, SAMPLES_PER_ROW)

/**
 * Panel width for the resolved content: wide enough for the sample row, never narrower than the
 * text block, never wider than the window allows.
 */
internal fun checkpointPanelWidth(maxPanelWidth: Dp, thumbWidth: Dp, slots: Int): Dp {
    val sampleRow = if (slots <= 0) {
        0.dp
    } else {
        thumbWidth * slots + SAMPLE_SLOT_SPACING * (slots - 1)
    }
    return maxOf(PANEL_TEXT_MIN_WIDTH, sampleRow + PANEL_CARD_PADDING).coerceAtMost(maxPanelWidth)
}

/** Keeps a user-dragged panel on screen and above the minimum readable size. */
internal fun clampPanelSize(width: Dp, height: Dp, maxWidth: Dp, maxHeight: Dp): DpSize {
    val lo = DpSize(minOf(PANEL_MIN_WIDTH, maxWidth), minOf(PANEL_MIN_HEIGHT, maxHeight))
    val hi = DpSize(maxWidth.coerceAtLeast(lo.width), maxHeight.coerceAtLeast(lo.height))
    return DpSize(width.coerceIn(lo.width, hi.width), height.coerceIn(lo.height, hi.height))
}

/**
 * Top-left corner for a panel anchored at [anchor]: right and below the click when a panel of
 * [panelWidth] fits there, otherwise on the other side, leaving [gap] between the two. Clamping to
 * the window is [clampPanelOrigin]'s job.
 *
 * Only the *default* panel width and the minimum readable height take part in the decision, never
 * the size a resize drag is producing: re-deciding against the live size is what made the panel jump
 * to the opposite side of the cursor the moment a drag grew it past the window edge.
 */
internal fun placePanelOrigin(
    anchor: Offset,
    panelWidth: Float,
    minVisibleHeight: Float,
    bounds: Size,
    gap: Float,
    margin: Float,
): Offset {
    val x = if (anchor.x + gap + panelWidth <= bounds.width - margin) {
        anchor.x + gap
    } else {
        anchor.x - gap - panelWidth
    }
    val y = if (anchor.y + gap + minVisibleHeight <= bounds.height - margin) {
        anchor.y + gap
    } else {
        anchor.y - gap - minVisibleHeight
    }
    return Offset(x, y)
}

/** Shifts [origin] by the least amount that keeps a [panelSize] panel inside the bounds. */
internal fun clampPanelOrigin(origin: Offset, panelSize: Size, bounds: Size, margin: Float): Offset {
    val maxX = (bounds.width - margin - panelSize.width).coerceAtLeast(margin)
    val maxY = (bounds.height - margin - panelSize.height).coerceAtLeast(margin)
    return Offset(origin.x.coerceIn(margin, maxX), origin.y.coerceIn(margin, maxY))
}

/** Launch-argument limits for the panel's "generate a sample" form, mirrored by api.py. */
internal const val GENERATE_MIN_CFG = 1f
internal const val GENERATE_MAX_CFG = 30f
internal const val GENERATE_MIN_STEPS = 1
internal const val GENERATE_MAX_STEPS = 150
internal const val GENERATE_MAX_SEED = 4_294_967_295L

/** Job modes as api.py writes them: one ad-hoc image, one image per sample set, that sets pass for
 *  every checkpoint of a step range (one process, one `sets` job per checkpoint), or an evaluation
 *  of one checkpoint (top its samples up, tag them, score them). */
internal const val JOB_MODE_SINGLE = "single"
internal const val JOB_MODE_SETS = "sets"
internal const val JOB_MODE_BATCH = "batch"
internal const val JOB_MODE_EVALUATE = "evaluate"

/** True for an evaluation job, whose record carries `scores` instead of an image of its own. */
internal fun isEvaluation(job: GeneratedSampleJob): Boolean = job.mode == JOB_MODE_EVALUATE

/**
 * Null when the form can be submitted, otherwise the message shown in the panel. Mirrors the
 * validation api.py repeats, so a rejected request costs no GPU time.
 */
internal fun generateFormError(prompt: String, cfg: String, steps: String, seed: String): String? {
    if (prompt.isBlank()) return "Enter a prompt first"
    val parsedCfg = cfg.trim().toFloatOrNull()
    if (parsedCfg == null || parsedCfg < GENERATE_MIN_CFG || parsedCfg > GENERATE_MAX_CFG) {
        return "CFG must be a number between ${GENERATE_MIN_CFG.toInt()} and ${GENERATE_MAX_CFG.toInt()}"
    }
    val parsedSteps = steps.trim().toIntOrNull()
    if (parsedSteps == null || parsedSteps < GENERATE_MIN_STEPS || parsedSteps > GENERATE_MAX_STEPS) {
        return "Steps must be a whole number between $GENERATE_MIN_STEPS and $GENERATE_MAX_STEPS"
    }
    val parsedSeed = seed.trim().toLongOrNull()
    if (parsedSeed == null || parsedSeed < 0 || parsedSeed > GENERATE_MAX_SEED) {
        return "Seed must be 0 (random) or a whole number up to $GENERATE_MAX_SEED"
    }
    return null
}

/** TensorBoard tag → label of the training scalars the panel shows for the clicked step. */
internal val TRAINING_STAT_TAGS = listOf(
    "Train/Avg_Loss" to "Avg Loss",
    "Train/Loss" to "Loss",
    "UNet/LR/Effective_Actual_LR" to "UNet LR",
    "TE/LR/Base_Scheduled" to "TE LR",
)

internal data class TrainingStat(val label: String, val value: String, val step: Int?)

/**
 * Value of every [TRAINING_STAT_TAGS] series nearest to [step]. A series the run never logged (the LR
 * tags are recent) reports `—` rather than vanishing, so the row keeps its shape between runs.
 */
internal fun trainingInfoAt(metrics: Map<String, List<MetricPoint>>, step: Float): List<TrainingStat> =
    TRAINING_STAT_TAGS.map { (tag, label) ->
        val point = nearestMetricPoint(metrics[tag].orEmpty(), step)
        val value = point?.let {
            if (tag.contains("LR")) formatScientificTwoDecimals(it.value) else formatFourDecimals(it.value)
        }
        TrainingStat(label, value ?: "—", point?.step)
    }

/** Sample settings from the flattened config, as the text the generate form starts with. */
internal data class GenerateFormDefaults(
    val prompt: String,
    val negativePrompt: String,
    val cfg: String,
    val steps: String,
    val seed: String,
)

/**
 * Seeds the generate form from `config.toml` so a click on the chart starts from the settings the
 * run's own samples used. The dashboard's resolved `sample_sets` win over the flat `sample_*` keys
 * (which mirror the first set); fallbacks mirror `TrainConfig`'s.
 */
internal fun generateFormDefaults(config: JsonObject): GenerateFormDefaults {
    fun text(key: String, fallback: String = ""): String =
        (config[key] as? JsonPrimitive)?.contentOrNull?.takeIf { it.isNotBlank() } ?: fallback

    val firstSet = config["sample_sets"]?.jsonArray?.firstOrNull()?.jsonObject
    fun fromSet(key: String): String? =
        firstSet?.get(key)?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() }

    return GenerateFormDefaults(
        prompt = fromSet("prompt") ?: text("sample_prompts"),
        negativePrompt = fromSet("negative") ?: text("sample_negative"),
        cfg = fromSet("guidance_scale") ?: text("guidance_scale", "6.0"),
        steps = fromSet("steps") ?: text("sample_steps", "55"),
        seed = fromSet("seed") ?: text("sample_seed", "0"),
    )
}

/** One-line summary of a finished generation: `CFG 5 · 20 steps · seed 12345`. */
internal fun generatedJobCaption(job: GeneratedSampleJob): String {
    val parts = mutableListOf<String>()
    job.cfg?.let { parts += "CFG ${formatCfg(it)}" }
    job.steps?.let { parts += "$it steps" }
    job.seed?.let { parts += "seed $it" }
    return parts.joinToString(" · ").ifBlank { "generated" }
}

/** Progress line while a job runs (`denoising 12/20`), or null once it is no longer running. */
internal fun generatedJobProgress(job: GeneratedSampleJob?): String? {
    if (job == null || job.state != JOB_RUNNING) return null
    if (job.cancelRequested) return "cancelling…"
    val total = job.totalSteps.takeIf { it > 0 } ?: return "denoising…"
    return "denoising ${job.currentStep.coerceAtMost(total)}/$total"
}

/**
 * `3/8 checkpoints · 12/48 images` for a running batch (or `cancelling…`), or null when there is no
 * batch left to report on. The batch record holds no images of its own: they belong to the per
 * checkpoint `sets` jobs its cards show.
 */
internal fun batchProgressLabel(job: GeneratedSampleJob?): String? {
    if (job == null || job.mode != JOB_MODE_BATCH || job.state != JOB_RUNNING) return null
    if (job.cancelRequested) return "cancelling…"
    val parts = mutableListOf("${job.checkpointIndex.coerceAtMost(job.totalCheckpoints)}/${job.totalCheckpoints} checkpoints")
    if (job.totalImages > 0) parts += "${job.imagesDone.coerceAtMost(job.totalImages)}/${job.totalImages} images"
    return parts.joinToString(" · ")
}

/** The running batch of a job list, if one is going. */
internal fun runningBatch(jobs: List<GeneratedSampleJob>): GeneratedSampleJob? =
    jobs.firstOrNull { it.mode == JOB_MODE_BATCH && it.state == JOB_RUNNING }

/**
 * Progress of a whole-set pass: `set 2/6 · image 3/12 · denoising 12/35`, or null when the job is
 * a single image or no longer running. An evaluation reports its own stage instead: `rendering
 * 3/4`, then `tagging 8/21` over the whole set it is scoring.
 */
internal fun generatedJobSetProgress(job: GeneratedSampleJob?): String? {
    if (job == null || job.state != JOB_RUNNING) return null
    if (job.cancelRequested) return "cancelling…"
    if (isEvaluation(job)) return evaluationProgressLabel(job)
    if (job.mode != JOB_MODE_SETS) return null
    val parts = mutableListOf<String>()
    if (job.totalSets > 0 && job.currentSet > 0) parts += "set ${job.currentSet}/${job.totalSets}"
    parts += "image ${job.imagesDone.coerceAtMost(job.totalImages)}/${job.totalImages}"
    generatedJobProgress(job)?.let { parts += it }
    return parts.joinToString(" · ")
}

private fun formatCfg(value: Float): String {
    val rounded = (value * 10).roundToInt() / 10f
    return if (rounded % 1f == 0f) rounded.toInt().toString() else rounded.toString()
}

internal fun checkpointsForRun(items: List<CheckpointItem>, runId: String?): List<CheckpointItem> =
    if (runId.isNullOrBlank()) items else items.filter { it.runId == runId }

/**
 * Closest checkpoint by step. Ties prefer the lower step (the model state that existed at or before
 * the clicked point), then a non-final checkpoint, then the directory name, so the pick never
 * depends on the order the files happened to be discovered in.
 */
internal fun nearestCheckpoint(items: List<CheckpointItem>, step: Float): CheckpointItem? =
    items.mapNotNull { item -> item.step?.let { item to it } }
        .minWithOrNull(
            compareBy({ abs(it.second - step) }, { it.second }, { it.first.final }, { it.first.dir }),
        )
        ?.first

/** Nearest logged point, same lower-step tie-break. */
internal fun nearestMetricPoint(points: List<MetricPoint>, step: Float): MetricPoint? =
    points.minWithOrNull(compareBy({ abs(it.step - step) }, { it.step }))

internal fun samplesForStep(samples: Map<String, List<SampleItem>>, step: Int?): List<SampleItem> =
    if (step == null) emptyList() else samples[step.toString()].orEmpty()

/**
 * Nearest step that actually has sample images. The `"-1"` bucket (files whose name carries no step)
 * is skipped, and a negative step is never a valid answer.
 */
internal fun nearestSampledStep(samples: Map<String, List<SampleItem>>, step: Int?): Int? {
    if (step == null) return null
    return samples.keys
        .mapNotNull { key ->
            key.toIntOrNull()?.takeIf { it >= 0 && samples[key]?.isNotEmpty() == true }
        }
        .minWithOrNull(compareBy({ abs(it - step) }, { it }))
}

/**
 * Whether a job has images to show: anything that is no longer running — finished, cancelled, or
 * one that failed part way. Its `files` are on disk either way, so they belong on the card.
 */
internal fun jobHasImages(job: GeneratedSampleJob): Boolean =
    job.state != JOB_RUNNING && generatedSampleItems(job).isNotEmpty()

/**
 * Whether a job belongs on a checkpoint card: one with images to show, or an evaluation — which may
 * have rendered nothing at all (the checkpoint already held enough images) while still carrying the
 * scores the card is there to show.
 */
internal fun jobShowsOnCard(job: GeneratedSampleJob): Boolean =
    jobHasImages(job) || (isEvaluation(job) && job.state != JOB_RUNNING)

/** Finished generation jobs for [step], in the order they were listed (newest first). */
internal fun generatedJobsForStep(jobs: List<GeneratedSampleJob>, step: Int?): List<GeneratedSampleJob> =
    if (step == null) {
        emptyList()
    } else {
        jobs.filter { it.step == step && jobShowsOnCard(it) }
    }

/**
 * The finished jobs the panel's row for [step] shows: [generatedJobsForStep]'s list — every job
 * recorded at that step — plus any pass rendered from [checkpoint] whatever step its record
 * carries, which is what keeps a job that names no step from going missing for the very checkpoint
 * it belongs to.
 */
internal fun panelJobsForStep(
    jobs: List<GeneratedSampleJob>,
    step: Int?,
    checkpoint: CheckpointItem?,
): List<GeneratedSampleJob> {
    val path = checkpoint?.path
    val own = jobs.filter { jobShowsOnCard(it) && path != null && it.checkpoint == path }
    return (generatedJobsForStep(jobs, step) + own).distinctBy { it.id }
}

/** The pass still rendering for [checkpoint], matched on its path (the job records it). */
internal fun runningJobForCheckpoint(
    jobs: List<GeneratedSampleJob>,
    checkpoint: CheckpointItem?,
): GeneratedSampleJob? {
    val path = checkpoint?.path ?: return null
    return jobs.firstOrNull { it.state == JOB_RUNNING && it.checkpoint == path }
}

/** The generation still denoising, if any: the panel shows its progress and keeps polling for it. */
internal fun runningJob(jobs: List<GeneratedSampleJob>): GeneratedSampleJob? =
    jobs.firstOrNull { it.state == JOB_RUNNING }

/** A generated job as panel slots; empty while it has no image yet. */
internal fun generatedSampleItems(job: GeneratedSampleJob): List<SampleItem> {
    // A `sets` pass and an evaluation both write `{job_id}_p{set}_{repeat}.png`; an evaluation that
    // needed no top-up has an empty list and contributes no thumbnail (its scores are its content).
    if (job.mode == JOB_MODE_SETS || isEvaluation(job)) {
        return job.files.mapNotNull { path ->
            path.takeIf { it.isNotBlank() }?.let { sampleItemForGeneratedFile(path) }
        }
    }
    val path = job.imagePath?.takeIf { it.isNotBlank() } ?: return emptyList()
    return listOf(SampleItem(filename = path.substringAfterLast('/'), setIndex = -1, repeatIdx = -1, path = path))
}

/**
 * `{job_id}_p{set}_{repeat}.png`: the set the pass rendered, counted from zero like the run's own
 * samples, so the thumbnail carries the same `Pn` badge a training sample would.
 */
private fun sampleItemForGeneratedFile(path: String): SampleItem {
    val name = path.substringAfterLast('/')
    val match = GENERATED_SET_NAME.find(name)
    val setIndex = match?.groupValues?.get(1)?.toIntOrNull() ?: -1
    val repeat = match?.groupValues?.get(2)?.toIntOrNull() ?: -1
    return SampleItem(filename = name, setIndex = setIndex, repeatIdx = repeat, path = path)
}

private val GENERATED_SET_NAME = Regex("""_p(\d+)_(\d+)\.png$""")

/**
 * `P2` marker for an image that came from a `[[validation.samples]]` entry, numbered from one
 * like the Utils tabs. A generated image (`setIndex < 0`) and a run from before the prompt sets
 * have nothing to point at.
 */
internal fun sampleSetBadge(setIndex: Int): String? = if (setIndex < 0) null else "P${setIndex + 1}"

/**
 * Whether a run used more than one prompt set. A single-set run numbers nothing: every image
 * would carry `P1` for no reason.
 */
internal fun showsSampleSetBadges(samples: Map<String, List<SampleItem>>): Boolean =
    samples.values.any { group -> group.any { it.setIndex > 0 } }

/** One thumbnail in the panel's sample row: a training sample, or a generated image beside it. */
internal data class SampleSlot(val item: SampleItem, val job: GeneratedSampleJob?, val isNew: Boolean)

/**
 * Panel row contents for a step: the run's own samples in their logged order, then the generated
 * images (newest last, so a fresh one lands at the end of the row where the eye is).
 */
internal fun sampleSlots(
    training: List<SampleItem>,
    jobs: List<GeneratedSampleJob>,
    sessionJobIds: Set<String>,
): List<SampleSlot> {
    val trainingSlots = training.map { SampleSlot(it, job = null, isNew = false) }
    val generatedSlots = jobs.asReversed().flatMap { job ->
        generatedSampleItems(job).map { item -> SampleSlot(item, job = job, isNew = job.id in sessionJobIds) }
    }
    return trainingSlots + generatedSlots
}

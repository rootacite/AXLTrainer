package com.acite.axlranko.model

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.DpSize
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
data class MetricPoint(
    val step: Int,
    val value: Float,
    @SerialName("wall_time") val wallTime: Double? = null,
)

@Serializable
data class DashboardResponse(
    val config: JsonObject = JsonObject(emptyMap()),
    @SerialName("run_id") val runId: String? = null,
    @SerialName("latest_stats") val latestStats: JsonObject = JsonObject(emptyMap()),
    val metrics: Map<String, List<MetricPoint>> = emptyMap(),
)

@Serializable
data class SampleItem(
    val filename: String,
    @SerialName("set_index") val setIndex: Int = 0,
    @SerialName("repeat_idx") val repeatIdx: Int,
    val path: String,
)

@Serializable
data class SamplesResponse(
    @SerialName("run_id") val runId: String? = null,
    val samples: Map<String, List<SampleItem>> = emptyMap(),
)

@Serializable
data class CheckpointItem(
    val path: String = "",
    @SerialName("run_id") val runId: String = "",
    val dir: String = "",
    val filename: String = "",
    val step: Int? = null,
    val epoch: Int? = null,
    val final: Boolean = false,
    @SerialName("size_bytes") val sizeBytes: Long = 0,
    val modified: Double = 0.0,
    @SerialName("network_dim") val networkDim: Int? = null,
    @SerialName("network_alpha") val networkAlpha: Int? = null,
    @SerialName("output_name") val outputName: String = "",
)

@Serializable
data class CheckpointsResponse(
    val checkpoints: List<CheckpointItem> = emptyList(),
)

/**
 * One checkpoint the user pinned in the Checkpoints section, kept in the run's own log directory
 * (`checkpoint_pins.json`). [dir] and [step] are recorded with the path so a pin is still readable
 * when its file is gone.
 */
@Serializable
data class CheckpointPin(
    val path: String = "",
    val dir: String = "",
    val step: Int? = null,
    @SerialName("pinned_at") val pinnedAt: Double? = null,
)

@Serializable
data class CheckpointPinsResponse(
    @SerialName("run_id") val runId: String? = null,
    /** The pin file itself, for the user to find; null when no run resolved. */
    val file: String? = null,
    val pins: List<CheckpointPin> = emptyList(),
)

/**
 * One run in the dashboard's history list, whatever `output_name` it was created with.
 * [current] is the run `state.json` is on — the only one the training controls act on —
 * and [live] that its process is still running.
 */
@Serializable
data class RunSummary(
    @SerialName("run_id") val runId: String = "",
    @SerialName("output_name") val outputName: String = "",
    @SerialName("has_output") val hasOutput: Boolean = false,
    @SerialName("has_log") val hasLog: Boolean = false,
    @SerialName("last_step") val lastStep: Int? = null,
    val samples: Int = 0,
    val checkpoints: Int = 0,
    @SerialName("size_bytes") val sizeBytes: Long = 0,
    val modified: Double = 0.0,
    val current: Boolean = false,
    val live: Boolean = false,
)

@Serializable
data class RunsResponse(
    val runs: List<RunSummary> = emptyList(),
)

/**
 * One "generate a sample with this checkpoint" job. Mirrored by the JSON file the generator writes
 * next to its PNG, so the panel can list jobs from disk and follow a run in progress.
 *
 * [mode] `single` is one image with its own prompt, `sets` one image per `[[validation.samples]]`
 * set and repeat (recorded in [files], with the set index in the file name).
 */
@Serializable
data class GeneratedSampleJob(
    val id: String = "",
    /** `running` while the generator works, then `done` or `error`. */
    val state: String = "running",
    val mode: String = "single",
    val step: Int? = null,
    val prompt: String = "",
    @SerialName("negative_prompt") val negativePrompt: String = "",
    val cfg: Float? = null,
    val steps: Int? = null,
    val seed: Long? = null,
    val checkpoint: String = "",
    @SerialName("image_path") val imagePath: String? = null,
    /** Every image a `sets` job wrote, in render order. */
    val files: List<String> = emptyList(),
    @SerialName("images_done") val imagesDone: Int = 0,
    @SerialName("total_images") val totalImages: Int = 1,
    @SerialName("current_set") val currentSet: Int = 0,
    @SerialName("total_sets") val totalSets: Int = 0,
    val error: String? = null,
    @SerialName("current_step") val currentStep: Int = 0,
    @SerialName("total_steps") val totalSteps: Int = 0,
    /** A cancel was asked for; the process is winding down (the job is still `running`). */
    @SerialName("cancel_requested") val cancelRequested: Boolean = false,
    /** Set on a `sets` job that a `batch` job started, tying it to its range. */
    @SerialName("batch_id") val batchId: String? = null,
    @SerialName("batch_index") val batchIndex: Int = 0,
    @SerialName("batch_total") val batchTotal: Int = 0,
    /** On a `batch` job: how far it is through its own work list. */
    @SerialName("checkpoint_index") val checkpointIndex: Int = 0,
    @SerialName("total_checkpoints") val totalCheckpoints: Int = 0,
    @SerialName("current_checkpoint") val currentCheckpoint: String? = null,
    @SerialName("from_step") val fromStep: Int? = null,
    @SerialName("to_step") val toStep: Int? = null,
    @SerialName("started_at") val startedAt: Double = 0.0,
    /** An evaluation's stage: `rendering` → `tagging` → `scoring` → `done`. */
    val phase: String = "",
    /** The depth it was asked for: a floor, so enough images mean nothing was rendered. */
    val depth: Int? = null,
    val threshold: Float? = null,
    val categories: List<String> = emptyList(),
    /** The tags the scoring was narrowed to; empty means every tag a prompt asks for. */
    val tags: List<String> = emptyList(),
    /** The `config.toml` its prompts came from: the run's own snapshot, or the repo's. */
    @SerialName("config_source") val configSource: String = "",
    /** An evaluation's result, `null` until the pass ends. */
    val scores: EvaluationScores? = null,
)

/**
 * Both scoreboards of one evaluation. The per-image one is the headline: every image's tags are
 * compared with the prompt it was rendered from, and the counts are summed over all of them, so a
 * requested tag that shows up on 1 of 20 images is one true positive and 19 false negatives. The
 * `union*` board pools each prompt's images instead: a requested tag counts as found when any of
 * them shows it.
 */
@Serializable
data class EvaluationScores(
    val tp: Int = 0,
    val fp: Int = 0,
    val fn: Int = 0,
    val precision: Float = 0f,
    val recall: Float = 0f,
    val f1: Float = 0f,
    @SerialName("union_tp") val unionTp: Int = 0,
    @SerialName("union_fp") val unionFp: Int = 0,
    @SerialName("union_fn") val unionFn: Int = 0,
    @SerialName("union_precision") val unionPrecision: Float = 0f,
    @SerialName("union_recall") val unionRecall: Float = 0f,
    @SerialName("union_f1") val unionF1: Float = 0f,
    @SerialName("images_scored") val imagesScored: Int = 0,
    @SerialName("images_failed") val imagesFailed: Int = 0,
    @SerialName("images_skipped") val imagesSkipped: Int = 0,
    /** The tags this evaluation was narrowed to; empty means every tag a prompt asks for. */
    val tags: List<String> = emptyList(),
    /** One row per prompt, both boards, images counted first. */
    val groups: List<EvaluationGroup> = emptyList(),
    /** The tags that cost the most precision / recall, most frequent first. */
    @SerialName("top_false_positives") val topFalsePositives: List<EvaluationTagCount> = emptyList(),
    @SerialName("top_false_negatives") val topFalseNegatives: List<EvaluationTagCount> = emptyList(),
)

/** One prompt of an evaluation: how many images were scored, and both boards for it. */
@Serializable
data class EvaluationGroup(
    val prompt: String = "",
    val images: Int = 0,
    val tp: Int = 0,
    val fp: Int = 0,
    val fn: Int = 0,
    val precision: Float = 0f,
    val recall: Float = 0f,
    val f1: Float = 0f,
    @SerialName("union_precision") val unionPrecision: Float = 0f,
    @SerialName("union_recall") val unionRecall: Float = 0f,
    @SerialName("union_f1") val unionF1: Float = 0f,
)

/** One tag of an evaluation's offender list, with how many images it cost. */
@Serializable
data class EvaluationTagCount(val tag: String = "", val count: Int = 0)

/**
 * The checkpoint the evaluation panel is open for, how many sample images it already shows, and the
 * evaluation job it is reporting on (a running one, the newest finished one, or none when the
 * checkpoint has never been evaluated). UI-only: the depth field starts at [existingImages], since
 * the depth is a floor and not a target.
 */
data class EvaluationTarget(
    val checkpoint: CheckpointItem,
    val existingImages: Int,
    val jobId: String? = null,
)

/** One prompt tag the evaluation picker offers, with how many prompt sets ask for it. */
@Serializable
data class PromptTagCount(
    val tag: String = "",
    val count: Int = 0,
    val frequency: Float = 0f,
)

/** `evaluation_prompts`: what an evaluation of this checkpoint would score against. */
@Serializable
data class EvaluationPromptsResponse(
    @SerialName("run_id") val runId: String = "",
    @SerialName("output_name") val outputName: String = "",
    val checkpoint: String = "",
    @SerialName("config_source") val configSource: String = "",
    /** The tags the prompts ask for, most frequent first; empty with [reason] set when unusable. */
    val tags: List<PromptTagCount> = emptyList(),
    val reason: String = "",
)

@Serializable
data class GeneratedSamplesResponse(
    @SerialName("run_id") val runId: String? = null,
    val jobs: List<GeneratedSampleJob> = emptyList(),
)

@Serializable
data class GenerateSampleResponse(
    val job: GeneratedSampleJob = GeneratedSampleJob(),
    @SerialName("log_path") val logPath: String? = null,
)

@Serializable
data class DatasetTagError(
    val file: String = "",
    val error: String = "",
)

@Serializable
data class DatasetTagResult(
    val directory: String = "",
    val engine: String = "",
    /** `full` (a caption per image) or `partial` (only the requested tags were added). */
    val mode: String = "full",
    val categories: List<String> = emptyList(),
    /** Partial tagging: the tags the pass was asked to add, as they were written. */
    @SerialName("only_tags") val onlyTags: List<String> = emptyList(),
    val threshold: Float = 0.35f,
    val provider: String = "",
    val device: String = "",
    val total: Int = 0,
    val processed: Int = 0,
    val failed: Int = 0,
    val seconds: Float = 0f,
    val errors: List<DatasetTagError> = emptyList(),
    /** Partial tagging: how many images gained each requested tag. */
    val added: Map<String, Int> = emptyMap(),
    /** Partial tagging: requested tags that matched no image at all. */
    val unmatched: List<String> = emptyList(),
)

/** One training folder as the Training form holds it, for `dataset_counts`. */
@Serializable
data class TrainDataCountRequest(val path: String = "", val repeat: Int = 1)

/** One folder's image count, with the reason when it could not be read. */
@Serializable
data class DatasetCountEntry(
    val path: String = "",
    val repeat: Int = 1,
    val images: Int = 0,
    val error: String? = null,
)

/** `dataset_counts`: what one per-epoch pass over the training folders would draw. */
@Serializable
data class DatasetCountsResponse(
    val entries: List<DatasetCountEntry> = emptyList(),
    /** Unique images across the folders. */
    val images: Int = 0,
    /** Images drawn in one epoch, repeats included. */
    val samples: Int = 0,
)

/** One of the tagger's categories, as the model declares it (`tagger_info`). */@Serializable
data class TaggerCategoryInfo(
    val key: String = "",
    val count: Int = 0,
    val calibrated: Float = 0f,
)

@Serializable
data class TaggerInfoResult(
    val available: Boolean = false,
    val engine: String = "",
    val model: String = "",
    @SerialName("model_path") val modelPath: String = "",
    @SerialName("cache_dir") val cacheDir: String = "",
    val categories: List<TaggerCategoryInfo> = emptyList(),
    @SerialName("default_categories") val defaultCategories: List<String> = emptyList(),
    val reason: String = "",
)

@Serializable
data class TrainSwap(
    val stage: String = "",
    val detail: String = "",
    val current: Int = 0,
    val total: Int = 0,
)

@Serializable
data class TrainEncoding(
    val current: Int = 0,
    val total: Int = 0,
    val done: Boolean = false,
)

@Serializable
data class TrainTrainingProgress(
    val step: Int = 0,
    @SerialName("total_steps") val totalSteps: Int = 0,
    val epoch: Int = 0,
    val epochs: Int = 0,
    val loss: Float? = null,
    @SerialName("avg_loss") val avgLoss: Float? = null,
)

@Serializable
data class TrainSampling(
    val active: Boolean = false,
    val repeat: Int = 0,
    val repeats: Int = 0,
    @SerialName("denoise_step") val denoiseStep: Int = 0,
    @SerialName("denoise_steps") val denoiseSteps: Int = 0,
    @SerialName("global_step") val globalStep: Int = 0,
    /** Which `[[validation.samples]]` entry the pass is on, 1-based; 0 when the run has no sets. */
    @SerialName("prompt_set") val promptSet: Int = 0,
    @SerialName("prompt_sets") val promptSets: Int = 0,
)

/**
 * The cadence and sampling switch the run in progress is actually using (`state.json`'s
 * `settings` block, published by the trainer). [nextSaveStep] is the step the next checkpoint
 * is written at.
 */
@Serializable
data class TrainSettings(
    @SerialName("save_every_n_steps") val saveEveryNSteps: Int = 0,
    @SerialName("sampling_enabled") val samplingEnabled: Boolean = true,
    @SerialName("next_save_step") val nextSaveStep: Int = 0,
)
@Serializable
data class TrainResume(
    val path: String = "",
    val filename: String = "",
    val step: Int? = null,
    val epoch: Int? = null,
    val loaded: Int = 0,
    val skipped: Int = 0,
)

@Serializable
data class HardwareGpu(
    val index: Int = 0,
    val name: String = "",
    @SerialName("gpu_clock_mhz") val gpuClockMhz: Double? = null,
    @SerialName("mem_clock_mhz") val memClockMhz: Double? = null,
    @SerialName("fan_pct") val fanPct: Double? = null,
    @SerialName("gpu_util_pct") val gpuUtilPct: Double? = null,
    @SerialName("mem_util_pct") val memUtilPct: Double? = null,
    @SerialName("power_w") val powerW: Double? = null,
    @SerialName("temp_c") val tempC: Double? = null,
    @SerialName("temp_edge_c") val tempEdgeC: Double? = null,
    @SerialName("temp_junction_c") val tempJunctionC: Double? = null,
    @SerialName("temp_mem_c") val tempMemC: Double? = null,
    @SerialName("mem_total_bytes") val memTotalBytes: Long? = null,
    @SerialName("mem_used_bytes") val memUsedBytes: Long? = null,
    @SerialName("mem_free_bytes") val memFreeBytes: Long? = null,
)

@Serializable
data class HardwareCpu(
    val name: String = "",
    @SerialName("n_logical") val nLogical: Int = 0,
    @SerialName("util_pct") val utilPct: Double? = null,
    @SerialName("temp_c") val tempC: Double? = null,
    @SerialName("mem_total_bytes") val memTotalBytes: Long? = null,
    @SerialName("mem_used_bytes") val memUsedBytes: Long? = null,
)

@Serializable
data class HardwareVmmVa(
    val patch: String = "vmm",
    @SerialName("used_bytes") val usedBytes: Long = 0,
    @SerialName("total_bytes") val totalBytes: Long = 0,
    @SerialName("total_source") val totalSource: String = "default",
    val pid: Int? = null,
    val spans: Int = 0,
    @SerialName("never_reuse") val vaNeverReuse: Boolean = false,
)

@Serializable
data class HardwareStatus(
    val available: Boolean = false,
    val error: String? = null,
    val ts: Double = 0.0,
    val gpus: List<HardwareGpu> = emptyList(),
    val cpu: HardwareCpu = HardwareCpu(),
    @SerialName("vmm_va") val vmmVa: HardwareVmmVa? = null,
)

data class HardwareHistory(
    val gpuUtil: List<MetricPoint> = emptyList(),
    val vramGiB: List<MetricPoint> = emptyList(),
    val powerW: List<MetricPoint> = emptyList(),
    val tempEdge: List<MetricPoint> = emptyList(),
    val tempJunction: List<MetricPoint> = emptyList(),
    val cpuUtil: List<MetricPoint> = emptyList(),
    val cpuTemp: List<MetricPoint> = emptyList(),
    val ramGiB: List<MetricPoint> = emptyList(),
)

@Serializable
data class TrainStatus(
    val schema: Int = 1,
    val pid: Int? = null,
    @SerialName("started_at") val startedAt: Double? = null,
    @SerialName("updated_at") val updatedAt: Double? = null,
    val status: String = "idle",
    @SerialName("paused_from") val pausedFrom: String? = null,
    @SerialName("output_name") val outputName: String? = null,
    @SerialName("run_id") val runId: String? = null,
    val resume: TrainResume? = null,
    val encoding: TrainEncoding = TrainEncoding(),
    val training: TrainTrainingProgress = TrainTrainingProgress(),
    val sampling: TrainSampling = TrainSampling(),
    val settings: TrainSettings = TrainSettings(),
    /**
     * A cadence / sampling-switch change the run has been asked for but has not adopted yet
     * (`train_settings`' request in `settings.json`). The trainer reads it once per optimizer
     * step, so a switch flipped during a sample pass stays pending until that pass ends; the
     * card shows the requested value and says when it lands. Null = nothing outstanding.
     */
    val requested: TrainSettings? = null,
    val swap: TrainSwap? = null,
    val error: String? = null,
    val detail: String? = null,
    val alive: Boolean = false,
    @SerialName("log_path") val logPath: String? = null,
)

/**
 * A Ctrl+click on the Avg Loss chart: the step under the pointer, where the panel should open
 * (window-root pixels), the checkpoint resolved for it and the outcome of a "Save As", plus the
 * state of the panel's "generate a sample with this checkpoint" form.
 */
data class ChartPickState(
    val step: Float = 0f,
    val anchor: Offset = Offset.Zero,
    val checkpoint: CheckpointItem? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val isFormOpen: Boolean = false,
    val prompt: String = "",
    val negativePrompt: String = "",
    val cfg: String = "",
    val steps: String = "",
    val seed: String = "0",
    val formError: String? = null,
    val isGenerating: Boolean = false,
)

/**
 * The outcome of one "Save As": [path] is the checkpoint that was copied, so both the Ctrl+click
 * panel and the Checkpoints section's card can show the line only for the checkpoint it belongs to.
 * [savedPath] is already formatted with the size the helper reported.
 */
data class CheckpointExport(
    val path: String,
    val savedPath: String? = null,
    val error: String? = null,
)

data class DashboardUiState(
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val connected: Boolean = false,
    val autoRefresh: Boolean = true,
    val smoothing: Float = 0.90f,
    val chartStroke: Float = 1.5f,
    val sampleThumbSize: Float = 120f,
    val previewIndex: Int? = null,
    val config: JsonObject = JsonObject(emptyMap()),
    val runId: String? = null,
    /** The training history, newest first, as reported by `list_runs`. */
    val runs: List<RunSummary> = emptyList(),
    /** The run the user pinned in the run selector; `null` follows the current one. */
    val selectedRun: RunSummary? = null,
    val latestStats: JsonObject = JsonObject(emptyMap()),
    val metrics: Map<String, List<MetricPoint>> = emptyMap(),
    val samples: Map<String, List<SampleItem>> = emptyMap(),
    /** The displayed run's LoRA checkpoints, newest step first. */
    val checkpoints: List<CheckpointItem> = emptyList(),
    /** Its pinned checkpoints, in the order they were pinned; these lead the section. */
    val checkpointPins: List<CheckpointPin> = emptyList(),
    /** The file the pins live in, shown when one is pinned. */
    val checkpointPinsFile: String? = null,
    /** Checkpoint path whose pin is on its way to the helper, if any. */
    val pinningPath: String? = null,
    val pinsError: String? = null,
    /** Checkpoint whose "Save As" is open or copying; one export at a time (one OS save dialog). */
    val exportInFlightPath: String? = null,
    /** Where the last "Save As" landed, or why it failed. */
    val exportResult: CheckpointExport? = null,
    /** Its one-off generation jobs, newest first, as stored under `{name}_samples/generated/`. */
    val generatedJobs: List<GeneratedSampleJob> = emptyList(),
    val generatedError: String? = null,
    val chartPick: ChartPickState? = null,
    /** Panel size the user dragged to, `null` while the content-derived default applies. */
    val chartPanelSize: DpSize? = null,
    /** Ids of generation jobs started in this session, which the panel marks as new. */
    val sessionJobIds: Set<String> = emptySet(),
    val trainStatus: TrainStatus = TrainStatus(),
    val commandInFlight: Boolean = false,
    val pendingCommand: String? = null,
    /** True while a cadence / sampling-switch change is on its way to the trainer. */
    val settingsInFlight: Boolean = false,
    val settingsError: String? = null,
    /** Checkpoint path whose whole-set sample pass is being started, if any. */
    val isGeneratingCheckpoint: String? = null,
    /** True while a step-range batch is being handed to the helper. */
    val isStartingBatch: Boolean = false,
    val batchError: String? = null,
    /** The checkpoint the evaluation panel is open for, if any. */
    val evaluationTarget: EvaluationTarget? = null,
    /** Checkpoint path whose evaluation is on its way to the helper. */
    val isStartingEvaluation: String? = null,
    val evaluationError: String? = null,
    /** The panel's `Details` block (the per-prompt breakdown) is expanded. */
    val evaluationDetailsOpen: Boolean = false,
    /** What the panel's tag picker offers: the prompts the pass will score against. */
    val evaluationPrompts: EvaluationPromptsResponse? = null,
    val evaluationPromptsLoading: Boolean = false,
    val evaluationPromptsError: String? = null,
    /** The picker's selection; empty scores every tag the prompts ask for. */
    val evaluationTagSelection: Set<String> = emptySet(),
    /** The tagger's own categories, for the dialog; fetched once when one opens. */
    val taggerInfo: TaggerInfoResult? = null,
    val hardware: HardwareStatus = HardwareStatus(),
    val hardwareHistory: HardwareHistory = HardwareHistory(),
)

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
 * One "generate a sample with this checkpoint" job. Mirrored by the JSON file the generator writes
 * next to its PNG, so the panel can list jobs from disk and follow a run in progress.
 */
@Serializable
data class GeneratedSampleJob(
    val id: String = "",
    /** `running` while the generator works, then `done` or `error`. */
    val state: String = "running",
    val step: Int? = null,
    val prompt: String = "",
    @SerialName("negative_prompt") val negativePrompt: String = "",
    val cfg: Float? = null,
    val steps: Int? = null,
    val seed: Long? = null,
    val checkpoint: String = "",
    @SerialName("image_path") val imagePath: String? = null,
    val error: String? = null,
    @SerialName("current_step") val currentStep: Int = 0,
    @SerialName("total_steps") val totalSteps: Int = 0,
    @SerialName("started_at") val startedAt: Double = 0.0,
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
    val threshold: Float = 0.35f,
    val provider: String = "",
    val total: Int = 0,
    val processed: Int = 0,
    val failed: Int = 0,
    val seconds: Float = 0f,
    val errors: List<DatasetTagError> = emptyList(),
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
data class HardwareStatus(
    val available: Boolean = false,
    val error: String? = null,
    val ts: Double = 0.0,
    val gpus: List<HardwareGpu> = emptyList(),
    val cpu: HardwareCpu = HardwareCpu(),
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
    val isSaving: Boolean = false,
    val saveProgress: Float? = null,
    val savedPath: String? = null,
    val saveError: String? = null,
    val isFormOpen: Boolean = false,
    val prompt: String = "",
    val negativePrompt: String = "",
    val cfg: String = "",
    val steps: String = "",
    val seed: String = "0",
    val formError: String? = null,
    val isGenerating: Boolean = false,
    /** Newest first, as stored under the run's `{name}_samples/generated/`. */
    val generatedJobs: List<GeneratedSampleJob> = emptyList(),
    val generatedError: String? = null,
)

data class DashboardUiState(
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val connected: Boolean = false,
    val autoRefresh: Boolean = true,
    val smoothing: Float = 0.90f,
    val chartStroke: Float = 1.5f,
    val sampleThumbSize: Float = 260f,
    val previewIndex: Int? = null,
    val config: JsonObject = JsonObject(emptyMap()),
    val runId: String? = null,
    val latestStats: JsonObject = JsonObject(emptyMap()),
    val metrics: Map<String, List<MetricPoint>> = emptyMap(),
    val samples: Map<String, List<SampleItem>> = emptyMap(),
    val chartPick: ChartPickState? = null,
    /** Panel size the user dragged to, `null` while the content-derived default applies. */
    val chartPanelSize: DpSize? = null,
    /** Ids of generation jobs started in this session, which the panel marks as new. */
    val sessionJobIds: Set<String> = emptySet(),
    val trainStatus: TrainStatus = TrainStatus(),
    val commandInFlight: Boolean = false,
    val pendingCommand: String? = null,
    val hardware: HardwareStatus = HardwareStatus(),
    val hardwareHistory: HardwareHistory = HardwareHistory(),
)

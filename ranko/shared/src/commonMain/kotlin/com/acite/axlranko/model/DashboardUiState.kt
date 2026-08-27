package com.acite.axlranko.model

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
    @SerialName("latest_stats") val latestStats: JsonObject = JsonObject(emptyMap()),
    val metrics: Map<String, List<MetricPoint>> = emptyMap(),
)

@Serializable
data class SampleItem(
    val filename: String,
    @SerialName("repeat_idx") val repeatIdx: Int,
    val path: String,
)

@Serializable
data class SamplesResponse(
    val samples: Map<String, List<SampleItem>> = emptyMap(),
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
    val encoding: TrainEncoding = TrainEncoding(),
    val training: TrainTrainingProgress = TrainTrainingProgress(),
    val sampling: TrainSampling = TrainSampling(),
    val swap: TrainSwap? = null,
    val error: String? = null,
    val detail: String? = null,
    val alive: Boolean = false,
    @SerialName("log_path") val logPath: String? = null,
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
    val latestStats: JsonObject = JsonObject(emptyMap()),
    val metrics: Map<String, List<MetricPoint>> = emptyMap(),
    val samples: Map<String, List<SampleItem>> = emptyMap(),
    val trainStatus: TrainStatus = TrainStatus(),
    val commandInFlight: Boolean = false,
    val pendingCommand: String? = null,
)

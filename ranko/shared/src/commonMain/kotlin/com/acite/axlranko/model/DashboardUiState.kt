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
)

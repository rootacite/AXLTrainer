package dataModel

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RefinementPassOverride(
    val name: String,
    val model: String,
    val denoise: Float,
    @SerialName("guide_size") val guideSize: Int
)

@Serializable
data class PromptOverrides(
    @SerialName("positive_prompt") val positivePrompt: String? = null,
    @SerialName("negative_prompt") val negativePrompt: String? = null,
    @SerialName("base_model_path") val baseModelPath: String? = null,
    @SerialName("lora_path") val loraPath: String? = null,
    @SerialName("lora_scale") val loraScale: Float? = null,
    @SerialName("realesrgan_model_path") val realesrganModelPath: String? = null,
    @SerialName("max_token_length") val maxTokenLength: Int? = null,
    @SerialName("clip_skip") val clipSkip: Int? = null,
    @SerialName("output_filename_prefix") val outputFilenamePrefix: String? = null,
    @SerialName("refinement_passes") val refinementPasses: List<RefinementPassOverride>? = null
)

@Serializable
data class SampleItem(
    val filename: String,
    @SerialName("repeat_idx") val repeatIdx: Int
)

@Serializable
data class SamplesResponse(
    val samples: Map<String, List<SampleItem>>
)

@Serializable
data class HealthzResponse(
    val status: String
)

package dataModel

data class GenerateUiState(
    val isLoadingConfig: Boolean = true,
    val isGenerating: Boolean = false,
    val error: String? = null,

    val positivePrompt: String = "",
    val negativePrompt: String = "",
    val width: String = "",
    val height: String = "",
    val steps: String = "",
    val cfgScale: String = "",
    val loraScale: String = "",
    val stages: String = "3",
    val seed: String = "0",

    val generatedImageBytes: ByteArray? = null,
    val selectedImageBytes: ByteArray? = null,
    val historyImages: List<ByteArray> = emptyList(),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as GenerateUiState

        if (isLoadingConfig != other.isLoadingConfig) return false
        if (isGenerating != other.isGenerating) return false
        if (error != other.error) return false
        if (positivePrompt != other.positivePrompt) return false
        if (negativePrompt != other.negativePrompt) return false
        if (width != other.width) return false
        if (height != other.height) return false
        if (steps != other.steps) return false
        if (cfgScale != other.cfgScale) return false
        if (loraScale != other.loraScale) return false
        if (stages != other.stages) return false
        if (seed != other.seed) return false
        if (!generatedImageBytes.contentEquals(other.generatedImageBytes)) return false
        if (!selectedImageBytes.contentEquals(other.selectedImageBytes)) return false
        if (historyImages != other.historyImages) return false

        return true
    }

    override fun hashCode(): Int {
        var result = isLoadingConfig.hashCode()
        result = 31 * result + isGenerating.hashCode()
        result = 31 * result + (error?.hashCode() ?: 0)
        result = 31 * result + positivePrompt.hashCode()
        result = 31 * result + negativePrompt.hashCode()
        result = 31 * result + width.hashCode()
        result = 31 * result + height.hashCode()
        result = 31 * result + steps.hashCode()
        result = 31 * result + cfgScale.hashCode()
        result = 31 * result + loraScale.hashCode()
        result = 31 * result + stages.hashCode()
        result = 31 * result + seed.hashCode()
        result = 31 * result + (generatedImageBytes?.contentHashCode() ?: 0)
        result = 31 * result + (selectedImageBytes?.contentHashCode() ?: 0)
        result = 31 * result + historyImages.hashCode()
        return result
    }
}
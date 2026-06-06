package generateBoard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dataModel.GenerateUiState
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import services.LoraApiClient
import dataModel.PromptOverrides
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlin.random.Random

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class GenerateBoardViewModel(
    val loraApiClient: LoraApiClient
) : ViewModel() {

    private val _state = MutableStateFlow(GenerateUiState())
    val state: StateFlow<GenerateUiState> = _state.asStateFlow()

    init {
        fetchInitialConfig()
    }

    private fun fetchInitialConfig() {
        viewModelScope.launch {
            _state.update { it.copy(isLoadingConfig = true, error = null) }
            try {
                val config = loraApiClient.getGenerateConfig()
                _state.update {
                    it.copy(
                        isLoadingConfig = false,
                        width = config["WIDTH"]?.jsonPrimitive?.content ?: "512",
                        height = config["HEIGHT"]?.jsonPrimitive?.content ?: "512",
                        steps = config["STEPS"]?.jsonPrimitive?.content ?: "20",
                        cfgScale = config["CFG_SCALE"]?.jsonPrimitive?.content ?: "7.0",
                        loraScale = config["LORA_SCALE"]?.jsonPrimitive?.content ?: "1.0",
                        positivePrompt = config["POSITIVE_PROMPT"]?.jsonPrimitive?.content ?: "",
                        negativePrompt = config["NEGATIVE_PROMPT"]?.jsonPrimitive?.content ?: "",
                        seed = config["SEED"]?.jsonPrimitive?.content ?: "0",
                    )
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(isLoadingConfig = false, error = e.message ?: "Failed to load config")
                }
            }
        }
    }

    fun updateField(field: String, value: String) {
        _state.update { current ->
            when (field) {
                "positivePrompt" -> current.copy(positivePrompt = value)
                "negativePrompt" -> current.copy(negativePrompt = value)
                "width" -> current.copy(width = value)
                "height" -> current.copy(height = value)
                "steps" -> current.copy(steps = value)
                "cfgScale" -> current.copy(cfgScale = value)
                "loraScale" -> current.copy(loraScale = value)
                "stages" -> current.copy(stages = value)
                "seed" -> current.copy(seed = value)
                else -> current
            }
        }
    }

    fun generateQuickImage() {
        executeGeneration(isQuick = true)
    }

    fun generateApiImage() {
        executeGeneration(isQuick = false)
    }

    private fun executeGeneration(isQuick: Boolean) {
        viewModelScope.launch {
            _state.update { it.copy(isGenerating = true, error = null) }
            try {
                val current = _state.value

                val overrides = PromptOverrides(
                    positivePrompt = current.positivePrompt.takeIf { it.isNotBlank() },
                    negativePrompt = current.negativePrompt.takeIf { it.isNotBlank() },
                    loraScale = current.loraScale.toFloatOrNull() ?: 1.0f
                )

                val finalWidth = current.width.toIntOrNull() ?: 512
                val finalHeight = current.height.toIntOrNull() ?: 512
                val finalSteps = current.steps.toIntOrNull() ?: 20
                val finalCfgScale = current.cfgScale.toFloatOrNull() ?: 7.0f
                val finalStages = current.stages.toIntOrNull() ?: 3
                val finalSeed = current.seed.toLongOrNull() ?: Random.nextLong(0, Long.MAX_VALUE)

                val imageBytes = if (isQuick) {
                    loraApiClient.generateQuick(
                        overrides = overrides,
                        steps = finalSteps,
                        cfgScale = finalCfgScale,
                        width = finalWidth,
                        height = finalHeight,
                        seed = finalSeed
                    )
                } else {
                    loraApiClient.generateApi(
                        overrides = overrides,
                        stages = finalStages,
                        steps = finalSteps,
                        cfgScale = finalCfgScale,
                        width = finalWidth,
                        height = finalHeight,
                        seed = finalSeed
                    )
                }

                _state.update {
                    val updatedHistory = it.historyImages + imageBytes
                    it.copy(
                        isGenerating = false,
                        historyImages = updatedHistory
                    )
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(isGenerating = false, error = e.message ?: "Generation failed")
                }
            }
        }
    }
}
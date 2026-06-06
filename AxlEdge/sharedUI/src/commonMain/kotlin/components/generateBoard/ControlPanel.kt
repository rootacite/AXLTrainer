package components.generateBoard

import androidx.compose.animation.*
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import components.publik.PanelHeader
import generateBoard.GenerateBoardViewModel
import utils.toFixed
import kotlin.math.roundToInt

private val FluentPrimary = Color(0xFF0067C0)
private val FluentPrimaryLight = Color(0xFF60CDFF)
private val FluentBorder = Color(0x26000000)
private val FluentRed = Color(0xFFE81123)
private val FluentGreen = Color(0xFF107C10)
private val FluentYellow = Color(0xFFFFB900)
private val CardShape = RoundedCornerShape(12.dp)

// ---------------------------------------------------------------------
// Foreground Controls
// ---------------------------------------------------------------------
@Composable
fun ControlPanel(
    state: dataModel.GenerateUiState,
    viewModel: GenerateBoardViewModel,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.padding(horizontal = 20.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(0.dp)
    ) {
        PanelHeader(title = "Generation", subtitle = "LoRA Inference Parameters")
        Spacer(Modifier.height(20.dp))

        SectionLabel("Prompt")
        Spacer(Modifier.height(8.dp))
        FluentTextField(
            value = state.positivePrompt,
            onValueChange = { viewModel.updateField("positivePrompt", it) },
            label = "Positive Prompt",
            minLines = 3,
            maxLines = 5,
            accentColor = FluentGreen
        )
        Spacer(Modifier.height(10.dp))
        FluentTextField(
            value = state.negativePrompt,
            onValueChange = { viewModel.updateField("negativePrompt", it) },
            label = "Negative Prompt",
            minLines = 2,
            maxLines = 3,
            accentColor = FluentRed
        )

        Spacer(Modifier.height(20.dp))
        FluentDivider()
        Spacer(Modifier.height(16.dp))

        SectionLabel("Resolution")
        Spacer(Modifier.height(12.dp))
        ResolutionControl(
            width = state.width,
            height = state.height,
            onWidthChange = { viewModel.updateField("width", it) },
            onHeightChange = { viewModel.updateField("height", it) }
        )

        Spacer(Modifier.height(20.dp))
        FluentDivider()
        Spacer(Modifier.height(16.dp))

        SectionLabel("Sampling")
        Spacer(Modifier.height(12.dp))

        val stepsValue = state.steps.toIntOrNull() ?: 20
        LabeledSlider(
            label = "Steps",
            valueText = "$stepsValue",
            value = stepsValue.toFloat(),
            onValueChange = { viewModel.updateField("steps", it.roundToInt().toString()) },
            valueRange = 10f..80f,
            steps = 69,
            color = FluentPrimary
        )
        Spacer(Modifier.height(14.dp))

        val cfgValue = state.cfgScale.toFloatOrNull() ?: 7f
        LabeledSlider(
            label = "CFG Scale",
            valueText = cfgValue.toFixed(1),
            value = cfgValue,
            onValueChange = { viewModel.updateField("cfgScale", it.toFixed(2)) },
            valueRange = 1f..15f,
            steps = 0,
            color = FluentPrimaryLight
        )
        Spacer(Modifier.height(14.dp))

        val loraValue = state.loraScale.toFloatOrNull() ?: 1f
        LabeledSlider(
            label = "LoRA Scale",
            valueText = loraValue.toFixed(2),
            value = loraValue,
            onValueChange = { viewModel.updateField("loraScale", it.toFixed(3)) },
            valueRange = 0f..2f,
            steps = 0,
            color = FluentYellow
        )

        Spacer(Modifier.height(20.dp))
        FluentDivider()
        Spacer(Modifier.height(16.dp))

        SectionLabel("Advanced")
        Spacer(Modifier.height(12.dp))
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            FluentSmallTextField(
                value = state.stages,
                onValueChange = { viewModel.updateField("stages", it) },
                label = "Stages",
                modifier = Modifier.weight(1f),
                keyboardType = KeyboardType.Number
            )

            FluentSmallTextField(
                value = state.seed,
                onValueChange = { viewModel.updateField("seed", it) },
                label = "Seed",
                modifier = Modifier.weight(1f),
                keyboardType = KeyboardType.Number
            )
        }

        Spacer(Modifier.height(24.dp))

        AnimatedVisibility(
            visible = state.error != null,
            enter = slideInVertically() + fadeIn(),
            exit = slideOutVertically() + fadeOut()
        ) {
            state.error?.let { ErrorBanner(message = it) }
        }

        GenerateButtons(
            isGenerating = state.isGenerating,
            onQuickGenerate = { viewModel.generateQuickImage() },
            onGenerate = { viewModel.generateApiImage() }
        )
        Spacer(Modifier.height(8.dp))
    }
}


@Composable
private fun SectionLabel(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelLarge,
        fontWeight = FontWeight.SemiBold,
        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
    )
}

@Composable
private fun FluentDivider() {
    HorizontalDivider(color = FluentBorder, thickness = 1.dp)
}

@Composable
private fun FluentTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    minLines: Int = 1,
    maxLines: Int = 4,
    accentColor: Color = FluentPrimary
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label, style = MaterialTheme.typography.labelMedium) },
        modifier = Modifier.fillMaxWidth(),
        minLines = minLines,
        maxLines = maxLines,
        shape = CardShape,
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = accentColor,
            unfocusedBorderColor = FluentBorder,
            focusedLabelColor = accentColor,
            cursorColor = accentColor
        )
    )
}

@Composable
private fun FluentSmallTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    modifier: Modifier = Modifier,
    keyboardType: KeyboardType = KeyboardType.Text
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label, style = MaterialTheme.typography.labelMedium) },
        modifier = modifier,
        singleLine = true,
        shape = CardShape,
        keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = FluentPrimary,
            unfocusedBorderColor = FluentBorder,
            focusedLabelColor = FluentPrimary
        )
    )
}

@Composable
private fun LabeledSlider(
    label: String,
    valueText: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    color: Color
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
            )
            AnimatedContent(
                targetState = valueText,
                transitionSpec = {
                    slideInVertically { -it } + fadeIn() togetherWith
                            slideOutVertically { it } + fadeOut()
                }
            ) { v ->
                Surface(
                    shape = RoundedCornerShape(6.dp),
                    color = color.copy(alpha = 0.12f)
                ) {
                    Text(
                        text = v,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 3.dp),
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = color
                    )
                }
            }
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = steps,
            modifier = Modifier.fillMaxWidth(),
            colors = SliderDefaults.colors(
                thumbColor = color,
                activeTrackColor = color,
                inactiveTrackColor = color.copy(alpha = 0.2f)
            )
        )
    }
}

@Composable
private fun ResolutionControl(
    width: String,
    height: String,
    onWidthChange: (String) -> Unit,
    onHeightChange: (String) -> Unit
) {
    val presets = listOf(
        "512x512" to (512 to 512),
        "768x768" to (768 to 768),
        "1024x1024" to (1024 to 1024),
        "768x1024" to (768 to 1024),
        "1024x768" to (1024 to 768)
    )

    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(presets.size) { i ->
            val (label, dims) = presets[i]
            val isSelected = width == dims.first.toString() && height == dims.second.toString()
            Surface(
                onClick = {
                    onWidthChange(dims.first.toString())
                    onHeightChange(dims.second.toString())
                },
                shape = RoundedCornerShape(8.dp),
                color = if (isSelected) FluentPrimary else MaterialTheme.colorScheme.surfaceVariant,
                modifier = Modifier.border(
                    1.dp,
                    if (isSelected) FluentPrimary else FluentBorder,
                    RoundedCornerShape(8.dp)
                )
            ) {
                Text(
                    text = label,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    style = MaterialTheme.typography.labelMedium,
                    color = if (isSelected) Color.White
                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                    fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal
                )
            }
        }
    }

    Spacer(Modifier.height(10.dp))

    Row(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        FluentSmallTextField(
            value = width,
            onValueChange = onWidthChange,
            label = "Width",
            modifier = Modifier.weight(1f),
            keyboardType = KeyboardType.Number
        )
        FluentSmallTextField(
            value = height,
            onValueChange = onHeightChange,
            label = "Height",
            modifier = Modifier.weight(1f),
            keyboardType = KeyboardType.Number
        )
    }
}

@Composable
private fun GenerateButtons(
    isGenerating: Boolean,
    onQuickGenerate: () -> Unit,
    onGenerate: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        OutlinedButton(
            onClick = onQuickGenerate,
            enabled = !isGenerating,
            modifier = Modifier.weight(1f).height(48.dp),
            shape = CardShape,
            border = androidx.compose.foundation.BorderStroke(
                1.dp,
                if (!isGenerating) FluentPrimary.copy(alpha = 0.6f) else FluentBorder
            )
        ) {
            AnimatedContent(targetState = isGenerating) { gen ->
                if (gen) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(14.dp),
                            strokeWidth = 2.dp,
                            color = FluentPrimary
                        )
                        Text("Working...", style = MaterialTheme.typography.labelMedium)
                    }
                } else {
                    Text(
                        "Flash",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.SemiBold,
                        color = FluentPrimary
                    )
                }
            }
        }

        Button(
            onClick = onGenerate,
            enabled = !isGenerating,
            modifier = Modifier.weight(1f).height(48.dp),
            shape = CardShape,
            colors = ButtonDefaults.buttonColors(
                containerColor = FluentPrimary,
                disabledContainerColor = FluentPrimary.copy(alpha = 0.4f)
            )
        ) {
            AnimatedContent(targetState = isGenerating) { gen ->
                if (gen) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(14.dp),
                            strokeWidth = 2.dp,
                            color = Color.White
                        )
                        Text(
                            "Generating...",
                            style = MaterialTheme.typography.labelMedium,
                            color = Color.White
                        )
                    }
                } else {
                    Text(
                        "Generate",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White
                    )
                }
            }
        }
    }
}

@Composable
private fun ErrorBanner(message: String) {
    Card(
        modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
        shape = CardShape,
        colors = CardDefaults.cardColors(containerColor = FluentRed.copy(alpha = 0.08f)),
        border = androidx.compose.foundation.BorderStroke(1.dp, FluentRed.copy(alpha = 0.3f))
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.Top
        ) {
            Text("!", style = MaterialTheme.typography.bodyMedium, color = FluentRed)
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = FluentRed.copy(alpha = 0.9f)
            )
        }
    }
}

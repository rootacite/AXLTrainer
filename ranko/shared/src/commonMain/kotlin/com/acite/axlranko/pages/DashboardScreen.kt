package com.acite.axlranko.pages

import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import java.awt.Cursor
import coil3.compose.AsyncImage
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.pages.components.ChartCard
import com.acite.axlranko.pages.components.CompactMetric
import com.acite.axlranko.pages.components.DashboardSectionHeader
import com.acite.axlranko.pages.components.MetricCard
import com.acite.axlranko.pages.components.PathChip
import com.acite.axlranko.pages.components.TrainControlCard
import com.acite.axlranko.util.formatFourDecimals
import com.acite.axlranko.util.formatScientificTwoDecimals
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import kotlin.math.roundToInt

@Composable
fun DashboardScreen(
    viewModel: DashboardScreenViewModel = metroViewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()

    if (uiState.errorMessage != null && !uiState.connected && uiState.latestStats.isEmpty()) {
        Box(
            modifier = Modifier.fillMaxSize().padding(32.dp),
            contentAlignment = Alignment.Center
        ) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = "Error",
                        modifier = Modifier.size(64.dp),
                        tint = MaterialTheme.colorScheme.error
                    )
                    Text(
                        text = uiState.errorMessage!!,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Button(onClick = { viewModel.retry() }) {
                        Text("Retry")
                    }
                }
            }
        }
        return
    }

    if (uiState.isLoading && !uiState.connected) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        return
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            DashboardHeader(uiState = uiState, viewModel = viewModel)
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

            Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                val listState = rememberLazyListState()
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp, vertical = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    item {
                        DashboardSectionHeader("Training Control")
                        Spacer(Modifier.height(4.dp))
                        TrainControlCard(
                            status = uiState.trainStatus,
                            commandInFlight = uiState.commandInFlight,
                            pendingCommand = uiState.pendingCommand,
                            outputDir = uiState.config.string("output_dir"),
                            loggingDir = uiState.config.string("logging_dir"),
                            onStart = viewModel::startTraining,
                            onPause = viewModel::pauseTraining,
                            onResume = viewModel::resumeTraining,
                            onStop = viewModel::stopTraining,
                            onReset = viewModel::resetTraining,
                        )
                    }

                    item {
                        PathRow(uiState.config)
                    }

                    item {
                        DashboardSectionHeader("Real-time Metrics")
                        Spacer(Modifier.height(4.dp))
                        MetricsSection(uiState)
                    }

                    item {
                        DashboardSectionHeader("Training Charts")
                        Spacer(Modifier.height(4.dp))
                        ChartsSection(uiState)
                    }

                    item {
                        DashboardSectionHeader("Generated Samples")
                    }

                    val grouped = uiState.samples.entries
                        .sortedByDescending { it.key.toIntOrNull() ?: Int.MIN_VALUE }

                    if (grouped.isEmpty()) {
                        item {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                                ),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Text(
                                    "No sample images generated yet.",
                                    modifier = Modifier.padding(20.dp),
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                    } else {
                        items(grouped, key = { it.key }) { (stepStr, samples) ->
                            SampleGroup(
                                stepStr = stepStr,
                                samples = samples,
                                thumbSize = uiState.sampleThumbSize,
                                onOpen = { viewModel.openPreview(it) },
                            )
                        }
                    }

                    item { Spacer(Modifier.height(24.dp)) }
                }

                VerticalScrollbar(
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .fillMaxHeight()
                        .padding(vertical = 8.dp),
                    adapter = rememberScrollbarAdapter(listState)
                )
            }
        }

        if (uiState.isLoading) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().align(Alignment.TopCenter)
            )
        }

        val previewIndex = uiState.previewIndex
        if (previewIndex != null) {
            val previewSamples = flattenSamples(uiState.samples)
            if (previewSamples.isNotEmpty()) {
                SamplePreviewOverlay(
                    samples = previewSamples,
                    index = previewIndex.coerceIn(previewSamples.indices),
                    onClose = viewModel::closePreview,
                    onPrev = viewModel::previewPrev,
                    onNext = viewModel::previewNext,
                )
            }
        }
    }
}

@Composable
private fun DashboardHeader(
    uiState: DashboardUiState,
    viewModel: DashboardScreenViewModel,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f).padding(end = 16.dp)) {
                Text(
                    text = "Dashboard",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = if (uiState.connected) "Helper connected" else "Helper disconnected",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (uiState.connected) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.error
                )
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = if (uiState.autoRefresh) "3s ON" else "OFF",
                        style = MaterialTheme.typography.labelLarge,
                        color = if (uiState.autoRefresh) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Switch(
                        checked = uiState.autoRefresh,
                        onCheckedChange = { viewModel.toggleAutoRefresh(it) }
                    )
                }
                Button(onClick = { viewModel.refreshNow() }) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Refresh")
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CompactMetric("Dataset", uiState.config.string("train_data_dir"))
            CompactMetric("Target", uiState.config.string("output_name"))
            CompactMetric("Base Model", uiState.config.string("pretrained_model_name_or_path"))
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            HeaderSlider(
                label = "Curve Smoothing: ${((uiState.smoothing * 100).roundToInt() / 100.0)}",
                value = uiState.smoothing,
                range = 0f..0.99f,
                onChange = viewModel::setSmoothing,
                modifier = Modifier.weight(1f),
            )
            HeaderSlider(
                label = "Chart Line: ${((uiState.chartStroke * 10).roundToInt() / 10.0)}",
                value = uiState.chartStroke,
                range = 1f..8f,
                onChange = viewModel::setChartStroke,
                modifier = Modifier.weight(1f),
            )
            HeaderSlider(
                label = "Sample Size: ${uiState.sampleThumbSize.roundToInt()}px",
                value = uiState.sampleThumbSize,
                range = 80f..360f,
                onChange = viewModel::setSampleThumbSize,
                modifier = Modifier.weight(1f),
            )
        }

        uiState.errorMessage?.let { message ->
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                shape = RoundedCornerShape(8.dp)
            ) {
                Text(
                    text = message,
                    modifier = Modifier.padding(12.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}

@Composable
private fun PathRow(config: JsonObject) {
    val loggingDir = config.string("logging_dir")
    val outputName = config.string("output_name")
    val outputDir = config.string("output_dir")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        PathChip("Logs", "$loggingDir/$outputName")
        PathChip("Output", outputDir)
    }
}

@Composable
private fun MetricsSection(uiState: DashboardUiState) {
    val stats = uiState.latestStats
    if (stats.isEmpty()) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            shape = RoundedCornerShape(8.dp)
        ) {
            Text(
                "No TensorBoard logs found yet. Waiting for training to start...",
                modifier = Modifier.padding(20.dp),
                style = MaterialTheme.typography.bodyMedium
            )
        }
        return
    }

    val currentStep = stats["current_step"]?.jsonPrimitive?.intOrNull?.toString() ?: "-"
    val latestLoss = stats["Train/Loss"]?.jsonPrimitive?.floatOrNull?.let { formatFourDecimals(it) } ?: "-"
    val unetLr = stats["UNet/LR/Effective_Actual_LR"]?.jsonPrimitive?.floatOrNull
        ?.let { formatScientificTwoDecimals(it) } ?: "-"
    val teLr = stats["TE/LR/Effective_Actual_LR"]?.jsonPrimitive?.floatOrNull
        ?.let { formatScientificTwoDecimals(it) } ?: "-"

    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val isWide = maxWidth > 720.dp
        if (isWide) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                MetricCard("Current Step", currentStep, MaterialTheme.colorScheme.primary, Modifier.weight(1f))
                MetricCard("Latest Loss", latestLoss, Color(0xFFFF4B4B), Modifier.weight(1f))
                MetricCard("UNet LR", unetLr, Color(0xFF0068C9), Modifier.weight(1f))
                MetricCard("TE Effective LR", teLr, Color(0xFF7B61FF), Modifier.weight(1f))
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard("Current Step", currentStep, MaterialTheme.colorScheme.primary, Modifier.weight(1f))
                    MetricCard("Latest Loss", latestLoss, Color(0xFFFF4B4B), Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard("UNet LR", unetLr, Color(0xFF0068C9), Modifier.weight(1f))
                    MetricCard("TE Effective LR", teLr, Color(0xFF7B61FF), Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun ChartsSection(uiState: DashboardUiState) {
    val metrics = uiState.metrics
    val smoothing = uiState.smoothing
    val stroke = uiState.chartStroke
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val isWide = maxWidth > 720.dp
        Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            ChartCard(
                "Train / Avg Loss",
                metrics["Train/Avg_Loss"].orEmpty(),
                Color(0xFF0D9488),
                smoothing = 0f,
                modifier = Modifier.fillMaxWidth(),
                strokeWidth = stroke,
                chartHeight = 280.dp,
            )
            if (isWide) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    ChartCard("Train / Loss", metrics["Train/Loss"].orEmpty(), Color(0xFFFF4B4B), smoothing, Modifier.weight(1f), strokeWidth = stroke)
                    ChartCard("UNet / LR", metrics["UNet/LR/Effective_Actual_LR"].orEmpty(), Color(0xFF0068C9), smoothing, Modifier.weight(1f), strokeWidth = stroke)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    ChartCard("TE / Base LR", metrics["TE/LR/Base_Scheduled"].orEmpty(), Color(0xFFF5A623), smoothing, Modifier.weight(1f), strokeWidth = stroke)
                    ChartCard("TE / Effective LR", metrics["TE/LR/Effective_Actual_LR"].orEmpty(), Color(0xFF7B61FF), smoothing, Modifier.weight(1f), strokeWidth = stroke)
                }
            } else {
                ChartCard("Train / Loss", metrics["Train/Loss"].orEmpty(), Color(0xFFFF4B4B), smoothing, Modifier.fillMaxWidth(), strokeWidth = stroke)
                ChartCard("UNet / LR", metrics["UNet/LR/Effective_Actual_LR"].orEmpty(), Color(0xFF0068C9), smoothing, Modifier.fillMaxWidth(), strokeWidth = stroke)
                ChartCard("TE / Base LR", metrics["TE/LR/Base_Scheduled"].orEmpty(), Color(0xFFF5A623), smoothing, Modifier.fillMaxWidth(), strokeWidth = stroke)
                ChartCard("TE / Effective LR", metrics["TE/LR/Effective_Actual_LR"].orEmpty(), Color(0xFF7B61FF), smoothing, Modifier.fillMaxWidth(), strokeWidth = stroke)
            }
        }
    }
}

@Composable
private fun HeaderSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = range
        )
    }
}

@Composable
private fun SampleGroup(
    stepStr: String,
    samples: List<SampleItem>,
    thumbSize: Float,
    onOpen: (SampleItem) -> Unit,
) {
    val stepLabel = if (stepStr == "-1") "Other / Unknown Step" else "Step $stepStr"
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(bottom = 10.dp)
        ) {
            Text(
                text = stepLabel,
                fontWeight = FontWeight.SemiBold,
                style = MaterialTheme.typography.titleSmall
            )
            Text(
                text = "(${samples.size} images)",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            items(samples, key = { it.path }) { sample ->
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                        .pointerHoverIcon(PointerIcon(Cursor(Cursor.HAND_CURSOR)))
                        .clickable { onOpen(sample) }
                        .padding(bottom = 8.dp)
                ) {
                    AsyncImage(
                        model = File(sample.path),
                        contentDescription = sample.filename,
                        contentScale = ContentScale.FillHeight,
                        filterQuality = FilterQuality.Low,
                        modifier = Modifier
                            .height(thumbSize.dp)
                            .clip(RoundedCornerShape(topStart = 8.dp, topEnd = 8.dp))
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = sample.filename,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier
                            .padding(horizontal = 8.dp)
                            .width(thumbSize.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun SamplePreviewOverlay(
    samples: List<SampleItem>,
    index: Int,
    onClose: () -> Unit,
    onPrev: () -> Unit,
    onNext: () -> Unit,
) {
    val sample = samples[index]
    val focusRequester = remember { FocusRequester() }
    var dragAccum by remember { mutableFloatStateOf(0f) }

    LaunchedEffect(index) {
        focusRequester.requestFocus()
        dragAccum = 0f
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.82f))
            .focusRequester(focusRequester)
            .focusable()
            .onPreviewKeyEvent { event ->
                if (event.type != KeyEventType.KeyDown) return@onPreviewKeyEvent false
                when (event.key) {
                    Key.Escape -> {
                        onClose()
                        true
                    }
                    Key.DirectionLeft -> {
                        onPrev()
                        true
                    }
                    Key.DirectionRight -> {
                        onNext()
                        true
                    }
                    else -> false
                }
            }
            .clickable(
                indication = null,
                interactionSource = remember { MutableInteractionSource() },
                onClick = onClose,
            ),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .clickable(
                    indication = null,
                    interactionSource = remember { MutableInteractionSource() },
                    onClick = {},
                ),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = sample.filename,
                    style = MaterialTheme.typography.titleSmall,
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = "${index + 1} / ${samples.size}",
                    style = MaterialTheme.typography.labelLarge,
                    color = Color.White.copy(alpha = 0.8f)
                )
                IconButton(onClick = onClose) {
                    Icon(Icons.Default.Close, contentDescription = "Close", tint = Color.White)
                }
            }

            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .pointerInput(index) {
                        detectHorizontalDragGestures(
                            onDragEnd = {
                                when {
                                    dragAccum > 80f -> onPrev()
                                    dragAccum < -80f -> onNext()
                                }
                                dragAccum = 0f
                            },
                            onDragCancel = { dragAccum = 0f },
                            onHorizontalDrag = { change, amount ->
                                change.consume()
                                dragAccum += amount
                            },
                        )
                    },
                contentAlignment = Alignment.Center
            ) {
                AsyncImage(
                    model = File(sample.path),
                    contentDescription = sample.filename,
                    contentScale = ContentScale.Fit,
                    filterQuality = FilterQuality.High,
                    modifier = Modifier.fillMaxSize().padding(horizontal = 72.dp, vertical = 8.dp)
                )

                PreviewNavButton(
                    modifier = Modifier.align(Alignment.CenterStart),
                    icon = Icons.Default.ChevronLeft,
                    description = "Previous",
                    onClick = onPrev,
                )
                PreviewNavButton(
                    modifier = Modifier.align(Alignment.CenterEnd),
                    icon = Icons.Default.ChevronRight,
                    description = "Next",
                    onClick = onNext,
                )
            }
        }
    }
}

@Composable
private fun PreviewNavButton(
    modifier: Modifier,
    icon: ImageVector,
    description: String,
    onClick: () -> Unit,
) {
    IconButton(
        onClick = onClick,
        modifier = modifier
            .size(48.dp)
            .clip(CircleShape)
            .background(Color.Black.copy(alpha = 0.45f))
            .pointerHoverIcon(PointerIcon(Cursor(Cursor.HAND_CURSOR)))
    ) {
        Icon(icon, contentDescription = description, tint = Color.White, modifier = Modifier.size(32.dp))
    }
}

private fun JsonObject.string(key: String): String {
    val value = this[key]?.jsonPrimitive?.content
    return if (value.isNullOrBlank()) "N/A" else value
}

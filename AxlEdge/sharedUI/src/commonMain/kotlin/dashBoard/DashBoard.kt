package dashBoard

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.material3.HorizontalDivider
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.decodeToImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.nestedscroll.NestedScrollConnection
import androidx.compose.ui.input.nestedscroll.NestedScrollSource
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import components.dashBoard.AsyncSampleImageCard
import components.dashBoard.CompactMetric
import components.dashBoard.ControlPanelSection
import components.dashBoard.GeneratedSamplesGrid
import components.publik.ChartCard
import components.publik.MetricCard
import components.publik.PathBadge
import components.publik.SectionHeader
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import utils.formatFourDecimals
import utils.formatScientificTwoDecimals
import kotlin.math.roundToInt

// ─────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────
private val PanelMaxHeight = 85.dp

@Composable
fun DashBoard(
    viewModel: DashBoardViewModel = metroViewModel()
) {
    val density = LocalDensity.current
    val panelMaxHeightPx = with(density) { PanelMaxHeight.toPx() }

    var contentOffsetY by remember { mutableStateOf(panelMaxHeightPx) }

    val nestedScrollConnection = remember {
        object : NestedScrollConnection {
            override fun onPreScroll(available: Offset, source: NestedScrollSource): Offset {
                val deltaY = available.y

                val newOffset = (contentOffsetY + deltaY).coerceIn(0f, panelMaxHeightPx)
                val consumedY = newOffset - contentOffsetY
                contentOffsetY = newOffset

                return if (consumedY != 0f) Offset(0f, consumedY) else Offset.Zero
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f))
            .nestedScroll(nestedScrollConnection)
    ) {
        // ── Top Control Panel (Fixed Height & Positioned Behind) ──
        CollapsibleControlPanel(
            viewModel = viewModel,
            modifier = Modifier
                .fillMaxWidth()
                .height(PanelMaxHeight)
                .align(Alignment.TopCenter)
        )

        // ── Scrollable main content (Overlays the panel) ──
        Box(
            modifier = Modifier
                .fillMaxSize()
                .offset { IntOffset(0, contentOffsetY.roundToInt()) }
                .drawWithCache {
                    val shadowHeight = 20.dp.toPx()
                    val topGradient = Brush.verticalGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color.Black.copy(alpha = 0.15f)
                        ),
                        startY = -shadowHeight,
                        endY = 0f
                    )
                    val dividerColor = Color.Gray.copy(alpha = 0.3f)

                    onDrawWithContent {
                        drawRect(
                            brush = topGradient,
                            topLeft = Offset(0f, -shadowHeight),
                            size = Size(size.width, shadowHeight)
                        )

                        drawLine(
                            color = dividerColor,
                            start = Offset(0f, 0f),
                            end = Offset(size.width, 0f),
                            strokeWidth = 1.dp.toPx()
                        )
                        drawContent()
                    }
                }
                .background(MaterialTheme.colorScheme.background)
        ) {
            MainContent(
                viewModel = viewModel,
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

// ─────────────────────────────────────────────
// Collapsible top panel (Now acts as a fixed background layer)
// ─────────────────────────────────────────────
@Composable
private fun CollapsibleControlPanel(
    viewModel: DashBoardViewModel,
    modifier: Modifier = Modifier
) {
    val config = viewModel.configData
    val scrollState = rememberScrollState()

    Box(
        modifier = modifier
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.8f),
                        MaterialTheme.colorScheme.surface.copy(alpha = 0f)
                    )
                )
            )
            .padding(horizontal = 12.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxHeight()
                .wrapContentWidth()
                .horizontalScroll(scrollState)
                .padding(top = 4.dp, bottom = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Section: Auto-Refresh toggle
            ControlPanelSection(title = "Auto-Refresh") {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = if (viewModel.autoRefresh) "3s  ON" else "OFF",
                        style = MaterialTheme.typography.labelLarge,
                        color = if (viewModel.autoRefresh)
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Switch(
                        checked = viewModel.autoRefresh,
                        onCheckedChange = { viewModel.toggleAutoRefresh(it) },
                        modifier = Modifier.height(24.dp)
                    )
                }
            }

            VerticalDivider(
                modifier = Modifier
                    .height(56.dp)
                    .alpha(0.4f)
            )

            // Section: Smoothing slider
            ControlPanelSection(
                title = "Curve Smoothing: ${((viewModel.smoothing * 100).roundToInt() / 100.0)}",
                modifier = Modifier.width(200.dp)
            ) {
                Slider(
                    value = viewModel.smoothing,
                    onValueChange = { viewModel.smoothing = it },
                    valueRange = 0f..0.99f,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            VerticalDivider(
                modifier = Modifier
                    .height(56.dp)
                    .alpha(0.4f)
            )

            // Section: Training Parameters
            ControlPanelSection(title = "Training Parameters") {
                Row(
                    modifier = Modifier.wrapContentWidth(),
                    horizontalArrangement = Arrangement.spacedBy(24.dp)
                ) {
                    CompactMetric(
                        label = "Dataset",
                        value = config?.get("train_data_dir")?.jsonPrimitive?.content ?: "N/A"
                    )
                    CompactMetric(
                        label = "Target",
                        value = config?.get("output_name")?.jsonPrimitive?.content ?: "N/A"
                    )
                    CompactMetric(
                        label = "Base Model",
                        value = config?.get("pretrained_model_name_or_path")?.jsonPrimitive?.content ?: "N/A"
                    )
                }
            }
        }
        val trackColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
        val thumbColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)

        Canvas(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 6.dp)
                .width(100.dp)
                .height(4.dp)
        ) {
            drawRoundRect(
                color = trackColor,
                size = size,
                cornerRadius = CornerRadius(size.height / 2f)
            )

            if (scrollState.maxValue > 0) {
                val progress = scrollState.value.toFloat() / scrollState.maxValue.toFloat()
                val thumbWidth = size.width * 0.3f
                val availableWidth = size.width - thumbWidth
                val thumbOffset = progress * availableWidth

                drawRoundRect(
                    color = thumbColor,
                    topLeft = Offset(thumbOffset, 0f),
                    size = Size(thumbWidth, size.height),
                    cornerRadius = CornerRadius(size.height / 2f)
                )
            }
        }
    }
}

@Composable
private fun MainContent(
    viewModel: DashBoardViewModel,
    modifier: Modifier = Modifier
) {
    val config = viewModel.configData
    val latestStats = viewModel.latestStats

    BoxWithConstraints(modifier = modifier) {
        val isWideScreen = maxWidth > maxHeight
        val groupedImages = viewModel.samplesData.entries
            .sortedByDescending { it.key.toIntOrNull() ?: Int.MIN_VALUE }

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 28.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(0.dp)
        ) {
            item {
                val loggingDir = config?.get("logging_dir")?.jsonPrimitive?.content ?: "N/A"
                val outputName = config?.get("output_name")?.jsonPrimitive?.content ?: "N/A"
                val outputDir = config?.get("output_dir")?.jsonPrimitive?.content ?: "N/A"

                if (isWideScreen) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.Bottom,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        HeaderTitle()
                        Column(horizontalAlignment = Alignment.End) {
                            PathBadge("Logs", "$loggingDir/$outputName")
                            Spacer(Modifier.height(4.dp))
                            PathBadge("Output", outputDir)
                        }
                    }
                } else {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(16.dp),
                        horizontalAlignment = Alignment.Start
                    ) {
                        HeaderTitle()
                        Column(horizontalAlignment = Alignment.Start) {
                            PathBadge("Logs", "$loggingDir/$outputName")
                            Spacer(Modifier.height(4.dp))
                            PathBadge("Output", outputDir)
                        }
                    }
                }

                Spacer(Modifier.height(32.dp))
            }

            if (latestStats != null) {
                item {
                    SectionHeader("Real-time Metrics")
                    Spacer(Modifier.height(12.dp))
                }

                item {
                    val currentStep = latestStats["current_step"]?.jsonPrimitive?.intOrNull?.toString() ?: "-"
                    val latestLoss = latestStats["Train/Loss"]?.jsonPrimitive?.floatOrNull?.let { formatFourDecimals(it) } ?: "-"
                    val unetLr = latestStats["UNet/LR/Effective_Actual_LR"]?.jsonPrimitive?.floatOrNull?.let { formatScientificTwoDecimals(it) } ?: "-"
                    val teLr = latestStats["TE/LR/Effective_Actual_LR"]?.jsonPrimitive?.floatOrNull?.let { formatScientificTwoDecimals(it) } ?: "-"

                    if (isWideScreen) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            MetricCard(
                                "Current Step",
                                currentStep,
                                accentColor = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.weight(1f)
                            )
                            MetricCard(
                                "Latest Loss",
                                latestLoss,
                                accentColor = Color(0xFFFF4B4B),
                                modifier = Modifier.weight(1f)
                            )
                            MetricCard(
                                "UNet LR",
                                unetLr,
                                accentColor = Color(0xFF0068C9),
                                modifier = Modifier.weight(1f)
                            )
                            MetricCard(
                                "TE Effective LR",
                                teLr,
                                accentColor = Color(0xFF7B61FF),
                                modifier = Modifier.weight(1f)
                            )
                        }
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                MetricCard(
                                    "Current Step",
                                    currentStep,
                                    accentColor = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.weight(1f)
                                )
                                MetricCard(
                                    "Latest Loss",
                                    latestLoss,
                                    accentColor = Color(0xFFFF4B4B),
                                    modifier = Modifier.weight(1f)
                                )
                            }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                MetricCard(
                                    "UNet LR",
                                    unetLr,
                                    accentColor = Color(0xFF0068C9),
                                    modifier = Modifier.weight(1f)
                                )
                                MetricCard(
                                    "TE Effective LR",
                                    teLr,
                                    accentColor = Color(0xFF7B61FF),
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                    }

                    Spacer(Modifier.height(36.dp))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f))
                    Spacer(Modifier.height(28.dp))
                }

                item {
                    SectionHeader("Training Charts")
                    Spacer(Modifier.height(12.dp))
                    ChartsGrid(viewModel, isWideScreen)
                }
            } else {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(20.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                strokeWidth = 2.dp
                            )
                            Text(
                                text = "No TensorBoard logs found yet. Waiting for training to start...",
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }

                    Spacer(Modifier.height(36.dp))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f))
                    Spacer(Modifier.height(28.dp))
                }
            }

            item {
                SectionHeader("Generated Samples")
                Spacer(Modifier.height(12.dp))
            }

            if (groupedImages.isEmpty()) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Text(
                            "No sample images generated yet.",
                            modifier = Modifier.padding(20.dp),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            } else {
                items(
                    items = groupedImages,
                    key = { it.key }
                ) { (stepStr, samples) ->
                    val stepLabel = if (stepStr == "-1") "Other / Unknown Step" else "Step $stepStr"

                    Column(
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.padding(bottom = 10.dp)
                        ) {
                            Text(
                                text = stepLabel,
                                fontWeight = FontWeight.SemiBold,
                                style = MaterialTheme.typography.titleSmall,
                                color = MaterialTheme.colorScheme.onBackground
                            )
                            Text(
                                text = "(${samples.size} images)",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        LazyRow(
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            items(
                                items = samples,
                                key = { it.filename }
                            ) { sample ->
                                AsyncSampleImageCard(sample.filename, viewModel)
                            }
                        }
                    }
                }
            }

            item {
                Spacer(Modifier.height(48.dp))
            }
        }
    }
}
@Composable
private fun HeaderTitle() {
    Column {
        Text(
            text = "LoRA Training",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = "Dashboard",
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.ExtraBold,
            color = MaterialTheme.colorScheme.onBackground
        )
    }
}

@Composable
private fun ChartsGrid(viewModel: DashBoardViewModel, isWideScreen: Boolean) {
    val metrics = viewModel.metricsData
    val smoothing = viewModel.smoothing

    if (isWideScreen) {
        Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
                ChartCard("Train / Loss", metrics?.get("Train/Loss") as? JsonArray, Color(0xFFFF4B4B), smoothing, Modifier.weight(1f))
                ChartCard("UNet / LR", metrics?.get("UNet/LR/Effective_Actual_LR") as? JsonArray, Color(0xFF0068C9), smoothing, Modifier.weight(1f))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
                ChartCard("TE / Base LR", metrics?.get("TE/LR/Base_Scheduled") as? JsonArray, Color(0xFFF5A623), smoothing, Modifier.weight(1f))
                ChartCard("TE / Effective LR", metrics?.get("TE/LR/Effective_Actual_LR") as? JsonArray, Color(0xFF7B61FF), smoothing, Modifier.weight(1f))
            }
        }
    } else {
        Column(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
            ChartCard("Train / Loss", metrics?.get("Train/Loss") as? JsonArray, Color(0xFFFF4B4B), smoothing, Modifier.fillMaxWidth())
            ChartCard("UNet / LR", metrics?.get("UNet/LR/Effective_Actual_LR") as? JsonArray, Color(0xFF0068C9), smoothing, Modifier.fillMaxWidth())
            ChartCard("TE / Base LR", metrics?.get("TE/LR/Base_Scheduled") as? JsonArray, Color(0xFFF5A623), smoothing, Modifier.fillMaxWidth())
            ChartCard("TE / Effective LR", metrics?.get("TE/LR/Effective_Actual_LR") as? JsonArray, Color(0xFF7B61FF), smoothing, Modifier.fillMaxWidth())
        }
    }
}
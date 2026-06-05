package dashBoard

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.geometry.Offset
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
import androidx.compose.ui.unit.dp
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
private val PanelMaxHeight = 80.dp
private val PanelMinHeight = 0.dp
private val PanelCollapseThreshold = 8.dp  // dp above minHeight where alpha hits 0

@Composable
fun DashBoard(
    viewModel: DashBoardViewModel = metroViewModel()
) {
    val density = LocalDensity.current

    // Height of the collapsible panel in dp
    var panelHeight by remember { mutableStateOf(PanelMaxHeight) }

    // Alpha: 1f when fully open, 0f when fully collapsed
    val panelAlpha by animateFloatAsState(
        targetValue = run {
            val dh = panelHeight - PanelMinHeight
            if (dh > PanelCollapseThreshold) 1f
            else (dh.value / PanelCollapseThreshold.value).coerceIn(0f, 1f)
        },
        animationSpec = tween(durationMillis = 80),
        label = "panelAlpha"
    )

    val nestedScrollConnection = remember {
        object : NestedScrollConnection {
            override fun onPreScroll(available: Offset, source: NestedScrollSource): Offset {
                val deltaY = available.y
                val deltaDp = with(density) { deltaY.toDp() }

                return if (deltaY < 0 && panelHeight > PanelMinHeight) {
                    // scrolling up → collapse
                    val newHeight = (panelHeight + deltaDp).coerceIn(PanelMinHeight, PanelMaxHeight)
                    val consumedDp = newHeight - panelHeight
                    panelHeight = newHeight
                    Offset(0f, with(density) { consumedDp.toPx() })
                } else if (deltaY > 0 && panelHeight < PanelMaxHeight) {
                    // scrolling down → expand
                    val newHeight = (panelHeight + deltaDp).coerceIn(PanelMinHeight, PanelMaxHeight)
                    val consumedDp = newHeight - panelHeight
                    panelHeight = newHeight
                    Offset(0f, with(density) { consumedDp.toPx() })
                } else {
                    Offset.Zero
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .nestedScroll(nestedScrollConnection)
    ) {
        // ── Top Control Panel (collapsible) ──
        CollapsibleControlPanel(
            viewModel = viewModel,
            panelHeight = panelHeight,
            panelAlpha = panelAlpha
        )

        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

        // ── Scrollable main content ──
        MainContent(
            viewModel = viewModel,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        )
    }
}

// ─────────────────────────────────────────────
// Collapsible top panel
// ─────────────────────────────────────────────
@Composable
private fun CollapsibleControlPanel(
    viewModel: DashBoardViewModel,
    panelHeight: Dp,
    panelAlpha: Float
) {
    val config = viewModel.configData
    val scrollState = rememberScrollState()

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(panelHeight)
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
                        MaterialTheme.colorScheme.surface.copy(alpha = 0.0f)
                    )
                )
            )
            .alpha(panelAlpha)
            .padding(horizontal = 12.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxHeight()
                .wrapContentWidth()
                .horizontalScroll(scrollState)
                .padding(vertical = 4.dp),
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
                        text = if (viewModel.autoRefresh) "1s  ON" else "OFF",
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
    }
}

// ---------------------------------------------------------
// Main scrollable content
// ---------------------------------------------------------
@Composable
private fun MainContent(
    viewModel: DashBoardViewModel,
    modifier: Modifier = Modifier
) {
    val config = viewModel.configData
    val latestStats = viewModel.latestStats

    BoxWithConstraints(modifier = modifier) {
        val isWideScreen = maxWidth > maxHeight

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 28.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(0.dp)
        ) {
            // -- Header --
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

            if (latestStats != null) {
                // -- Real-time metric cards --
                SectionHeader("Real-time Metrics")
                Spacer(Modifier.height(12.dp))

                val currentStep = latestStats["current_step"]?.jsonPrimitive?.intOrNull?.toString() ?: "-"
                val latestLoss = latestStats["Train/Loss"]?.jsonPrimitive?.floatOrNull?.let { formatFourDecimals(it) } ?: "-"
                val unetLr = latestStats["UNet/LR/Effective_Actual_LR"]?.jsonPrimitive?.floatOrNull?.let { formatScientificTwoDecimals(it) } ?: "-"
                val teLr = latestStats["TE/LR/Effective_Actual_LR"]?.jsonPrimitive?.floatOrNull?.let { formatScientificTwoDecimals(it) } ?: "-"

                if (isWideScreen) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        MetricCard("Current Step", currentStep, accentColor = MaterialTheme.colorScheme.primary, modifier = Modifier.weight(1f))
                        MetricCard("Latest Loss", latestLoss, accentColor = Color(0xFFFF4B4B), modifier = Modifier.weight(1f))
                        MetricCard("UNet LR", unetLr, accentColor = Color(0xFF0068C9), modifier = Modifier.weight(1f))
                        MetricCard("TE Effective LR", teLr, accentColor = Color(0xFF7B61FF), modifier = Modifier.weight(1f))
                    }
                } else {
                    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            MetricCard("Current Step", currentStep, accentColor = MaterialTheme.colorScheme.primary, modifier = Modifier.weight(1f))
                            MetricCard("Latest Loss", latestLoss, accentColor = Color(0xFFFF4B4B), modifier = Modifier.weight(1f))
                        }
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            MetricCard("UNet LR", unetLr, accentColor = Color(0xFF0068C9), modifier = Modifier.weight(1f))
                            MetricCard("TE Effective LR", teLr, accentColor = Color(0xFF7B61FF), modifier = Modifier.weight(1f))
                        }
                    }
                }

                Spacer(Modifier.height(36.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f))
                Spacer(Modifier.height(28.dp))

                // -- Charts --
                SectionHeader("Training Charts")
                Spacer(Modifier.height(12.dp))
                ChartsGrid(viewModel, isWideScreen)

            } else {
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
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                        Text(
                            text = "No TensorBoard logs found yet. Waiting for training to start...",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }

            Spacer(Modifier.height(36.dp))
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f))
            Spacer(Modifier.height(28.dp))

            // -- Generated samples --
            SectionHeader("Generated Samples")
            Spacer(Modifier.height(12.dp))
            GeneratedSamplesGrid(viewModel)

            Spacer(Modifier.height(48.dp))
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

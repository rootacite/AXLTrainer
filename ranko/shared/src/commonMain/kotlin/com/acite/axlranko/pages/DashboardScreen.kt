package com.acite.axlranko.pages

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.detectDragGestures
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.PushPin
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.outlined.PushPin
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.material3.OutlinedTextField
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.foundation.border
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.acite.axlranko.ui.pointerIconHand
import com.acite.axlranko.ui.pointerIconNwseResize
import coil3.compose.AsyncImage
import com.acite.axlranko.model.ChartPickState
import com.acite.axlranko.model.CheckpointExport
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.pages.components.ChartCard
import com.acite.axlranko.pages.components.ChartPickMarkers
import com.acite.axlranko.pages.components.CheckpointRow
import com.acite.axlranko.pages.components.CompactMetric
import com.acite.axlranko.pages.components.DashboardSectionHeader
import com.acite.axlranko.pages.components.EvaluationDialog
import com.acite.axlranko.pages.components.PAGE_PANEL_MARGIN
import com.acite.axlranko.pages.components.evaluationRecallHeadline
import com.acite.axlranko.pages.components.HardwareSection
import com.acite.axlranko.pages.components.ImagePreviewOverlay
import com.acite.axlranko.pages.components.MetricCard
import com.acite.axlranko.pages.components.PANEL_CARD_PADDING
import com.acite.axlranko.pages.components.PreviewImage
import com.acite.axlranko.pages.components.PANEL_MAX_HEIGHT
import com.acite.axlranko.pages.components.PANEL_MAX_WIDTH
import com.acite.axlranko.pages.components.PANEL_MIN_HEIGHT
import com.acite.axlranko.pages.components.PathChip
import com.acite.axlranko.pages.components.RunSelector
import com.acite.axlranko.pages.components.SAMPLES_PER_ROW
import com.acite.axlranko.pages.components.SAMPLE_SLOT_SPACING
import com.acite.axlranko.pages.components.SAMPLE_THUMB_ASPECT
import com.acite.axlranko.pages.components.SampleSlot
import com.acite.axlranko.pages.components.TrainControlCard
import com.acite.axlranko.pages.components.batchProgressLabel
import com.acite.axlranko.pages.components.batchRangeError
import com.acite.axlranko.pages.components.checkpointPanelWidth
import com.acite.axlranko.pages.components.checkpointRowKey
import com.acite.axlranko.pages.components.checkpointSteps
import com.acite.axlranko.pages.components.checkpointsInRange
import com.acite.axlranko.pages.components.checkpointRowLabel
import com.acite.axlranko.pages.components.checkpointRows
import com.acite.axlranko.pages.components.cardEvaluation
import com.acite.axlranko.pages.components.clampPanelOrigin
import com.acite.axlranko.pages.components.clampPanelSize
import com.acite.axlranko.pages.components.displayedRun
import com.acite.axlranko.pages.components.generatedJobCaption
import com.acite.axlranko.pages.components.generatedJobProgress
import com.acite.axlranko.pages.components.generatedJobSetProgress
import com.acite.axlranko.pages.components.isEvaluation
import com.acite.axlranko.pages.components.nearestSampledStep
import com.acite.axlranko.pages.components.panelJobsForStep
import com.acite.axlranko.pages.components.placePanelOrigin
import com.acite.axlranko.pages.components.runningJob
import com.acite.axlranko.pages.components.runningBatch
import com.acite.axlranko.pages.components.sampleColumns
import com.acite.axlranko.pages.components.sampleSetBadge
import com.acite.axlranko.pages.components.sampleSlotWidth
import com.acite.axlranko.pages.components.sampleSlots
import com.acite.axlranko.pages.components.sampleThumbWidth
import com.acite.axlranko.pages.components.samplesForStep
import com.acite.axlranko.pages.components.showsSampleSetBadges
import com.acite.axlranko.pages.components.trainingInfoAt
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.checkpointSubtitle
import com.acite.axlranko.util.formatFourDecimals
import com.acite.axlranko.util.formatScientificTwoDecimals
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality
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
            PorcelainCard {
                Column(
                    modifier = Modifier.padding(18.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = "Error",
                        modifier = Modifier.size(64.dp),
                        tint = rankoColors.qualityRed
                    )
                    Text(
                        text = uiState.errorMessage!!,
                        color = rankoColors.text,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    CapsuleButton(text = "Retry", onClick = { viewModel.retry() }, emphasized = true)
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

    var dashboardOrigin by remember { mutableStateOf(Offset.Zero) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .onGloballyPositioned { dashboardOrigin = it.positionInRoot() },
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            val listState = rememberLazyListState()
            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    // The bar is the list's own first row: scrolling takes it away and brings it
                    // back with everything else, and the list's scroll position is the only one
                    // there is.
                    DashboardHeader(uiState = uiState, viewModel = viewModel)
                    HorizontalDivider(color = rankoColors.stroke.copy(alpha = 0.55f))
                }
                item {
                    DashboardSectionHeader("Training Control")
                    Spacer(Modifier.height(4.dp))
                    TrainControlCard(
                        status = uiState.trainStatus,
                        commandInFlight = uiState.commandInFlight,
                        controlsEnabled = trainingControlsEnabled(uiState),
                        pendingCommand = uiState.pendingCommand,
                        settingsEnabled = liveSettingsEnabled(uiState),
                        settingsInFlight = uiState.settingsInFlight,
                        settingsError = uiState.settingsError,
                        // With no live run the row describes what Start would use.
                        configSaveEveryNSteps = uiState.config.int("save_every_n_steps"),
                        configSamplingEnabled = uiState.config.flag("sampling_enabled"),
                        onApplySettings = viewModel::applyTrainSettings,
                        shownRun = displayedRun(uiState.runs, uiState.selectedRun, uiState.runId),
                        outputDir = uiState.config.string("output_dir"),
                        loggingDir = uiState.config.string("logging_dir"),
                        resumeFrom = uiState.config.string("resume_lora_path"),
                        onStart = viewModel::startTraining,
                        onPause = viewModel::pauseTraining,
                        onResume = viewModel::resumeTraining,
                        onStop = viewModel::stopTraining,
                        onReset = viewModel::resetTraining,
                    )
                }

                item {
                    DashboardSectionHeader("Hardware")
                    Spacer(Modifier.height(4.dp))
                    HardwareSection(uiState)
                }

                item {
                    PathRow(uiState.config, uiState.runId)
                }

                item {
                    DashboardSectionHeader("Real-time Metrics")
                    Spacer(Modifier.height(4.dp))
                    MetricsSection(uiState)
                }

                item {
                    DashboardSectionHeader("Training Charts")
                    Spacer(Modifier.height(4.dp))
                    ChartsSection(
                        uiState = uiState,
                        onPickStep = viewModel::pickCheckpointAt,
                    )
                }

                item {
                    DashboardSectionHeader("Checkpoints")
                    Spacer(Modifier.height(4.dp))
                }

                val pinnedPaths = uiState.checkpointPins.map { it.path }.toSet()
                val checkpointCards = checkpointRows(
                    checkpoints = uiState.checkpoints,
                    samples = uiState.samples,
                    jobs = uiState.generatedJobs,
                    pinned = pinnedPaths,
                )
                // A generation of its own keeps the card busy, so the buttons follow both rules.
                val runningBatchJob = runningBatch(uiState.generatedJobs)
                val gpuFree = generationAllowed(uiState.trainStatus) && runningJob(uiState.generatedJobs) == null
                val showSetBadges = showsSampleSetBadges(uiState.samples)

                item {
                    SampleRangeRow(
                        rows = checkpointCards,
                        batch = runningBatchJob,
                        starting = uiState.isStartingBatch,
                        canStart = gpuFree,
                        note = uiState.batchError ?: uiState.generatedError,
                        onSampleRange = viewModel::startSampleBatch,
                        onCancel = { id -> viewModel.cancelGeneration(id) },
                    )
                }

                if (checkpointCards.any { it.pinned }) {
                    item { PinnedHintLine(uiState.checkpointPinsFile) }
                }
                uiState.pinsError?.let { message -> item { PinsErrorLine(message) } }

                if (checkpointCards.isEmpty()) {
                    item { CheckpointsEmptyCard() }
                } else {
                    // One lazy item per checkpoint: a finished run can hold dozens, and each
                    // card asks for its own thumbnails. The pinned cards are the section's
                    // prefix, so the divider between the two groups is one item too.
                    val pinnedCards = checkpointCards.filter { it.pinned }
                    val restCards = checkpointCards.filterNot { it.pinned }
                    val card: @Composable (CheckpointRow) -> Unit = { row ->
                        CheckpointRowCard(
                            row = row,
                            thumbSize = uiState.sampleThumbSize,
                            showSetBadges = showSetBadges,
                            newJobIds = uiState.sessionJobIds,
                            gpuFree = gpuFree,
                            starting = uiState.isGeneratingCheckpoint == row.checkpoint?.path,
                            busyElsewhere = uiState.isGeneratingCheckpoint != null &&
                                uiState.isGeneratingCheckpoint != row.checkpoint?.path,
                            startingEvaluation = uiState.isStartingEvaluation == row.checkpoint?.path,
                            pinning = uiState.pinningPath == row.checkpoint?.path,
                            pinEnabled = uiState.pinningPath == null,
                            exportInFlightPath = uiState.exportInFlightPath,
                            exportResult = uiState.exportResult,
                            onOpen = { viewModel.openPreview(it) },
                            onGenerate = viewModel::generateCheckpointSamples,
                            onEvaluate = { checkpoint, images, jobId ->
                                viewModel.openEvaluation(checkpoint, images, jobId)
                            },
                            onCancelEvaluation = { id -> viewModel.cancelGeneration(id) },
                            onOpenEvaluation = viewModel::showEvaluation,
                            onTogglePin = viewModel::toggleCheckpointPin,
                            onSaveAs = viewModel::saveCheckpointAs,
                        )
                    }
                    items(pinnedCards, key = { checkpointRowKey(it) }) { row -> card(row) }
                    if (pinnedCards.isNotEmpty() && restCards.isNotEmpty()) {
                        item {
                            CheckpointsDivider(
                                pinned = pinnedCards.size,
                                rest = restCards.size,
                            )
                        }
                    }
                    items(restCards, key = { checkpointRowKey(it) }) { row -> card(row) }
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

        if (uiState.isLoading) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().align(Alignment.TopCenter)
            )
        }

        uiState.evaluationTarget?.let { target ->
            // The panel is a dialog, so it cannot inherit this page's constraints: it is told how
            // much room the window has and scrolls inside that (`PAGE_PANEL_MARGIN` on each side).
            BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
                EvaluationDialog(
                    target = target,
                    job = uiState.generatedJobs.firstOrNull { it.id == target.jobId },
                    tagger = uiState.taggerInfo,
                    prompts = uiState.evaluationPrompts,
                    promptsLoading = uiState.evaluationPromptsLoading,
                    promptsError = uiState.evaluationPromptsError,
                    selectedTags = uiState.evaluationTagSelection,
                    starting = uiState.isStartingEvaluation == target.checkpoint.path,
                    error = uiState.evaluationError,
                    detailsOpen = uiState.evaluationDetailsOpen,
                    maxWidth = maxWidth - PAGE_PANEL_MARGIN,
                    maxHeight = maxHeight - PAGE_PANEL_MARGIN,
                    onToggleTag = viewModel::toggleEvaluationTag,
                    onClearTags = viewModel::clearEvaluationTags,
                    onToggleDetails = viewModel::toggleEvaluationDetails,
                    onCancel = { id -> viewModel.cancelGeneration(id) },
                    onStart = { depth, threshold, categories, tags ->
                        viewModel.startEvaluation(depth, threshold, categories, tags)
                    },
                    onDismiss = viewModel::dismissEvaluation,
                )
            }
        }

        uiState.chartPick?.let { pick ->
            CheckpointPanelOverlay(
                pick = pick,
                metrics = uiState.metrics,
                samples = uiState.samples,
                generatedJobs = uiState.generatedJobs,
                generatedError = uiState.generatedError,
                originInRoot = dashboardOrigin,
                userSize = uiState.chartPanelSize,
                previewOpen = uiState.previewIndex != null,
                // One rule for "may a generation start", shared with the Checkpoints section and
                // mirroring api.py's `_gpu_busy`: a paused run is fine, it has given the card back.
                gpuFree = generationAllowed(uiState.trainStatus),
                newJobIds = uiState.sessionJobIds,
                exportInFlightPath = uiState.exportInFlightPath,
                exportResult = uiState.exportResult,
                onResize = viewModel::setChartPanelSize,
                onOpenSample = { viewModel.openPreview(it) },
                onSaveAs = viewModel::saveCheckpointAs,
                onToggleForm = viewModel::toggleGenerateForm,
                onUpdateForm = viewModel::updateChartPickForm,
                onGenerate = { rowStep -> viewModel.generateSample(rowStep) },
                onClose = viewModel::dismissChartPick,
            )
        }

        val previewIndex = uiState.previewIndex
        if (previewIndex != null) {
            // The section's own list, so a thumbnail the page drew always resolves to an index.
            val previewImages = previewList(uiState.checkpoints, uiState.samples, uiState.generatedJobs)
            if (previewImages.isNotEmpty()) {
                SamplePreviewOverlay(
                    samples = previewImages,
                    index = previewIndex.coerceIn(previewImages.indices),
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
        // The list the bar heads already carries the page's margins.
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 12.dp),
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
                    color = if (uiState.connected) rankoColors.accentPink
                    else rankoColors.qualityRed
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
                        color = if (uiState.autoRefresh) rankoColors.accentPink
                        else rankoColors.textDim
                    )
                    Switch(
                        checked = uiState.autoRefresh,
                        onCheckedChange = { viewModel.toggleAutoRefresh(it) }
                    )
                }
                CapsuleButton(
                    text = "Refresh",
                    onClick = { viewModel.refreshNow() },
                    compact = true,
                    emphasized = true,
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Refresh", fontWeight = FontWeight.SemiBold)
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            RunSelector(
                runs = uiState.runs,
                selected = uiState.selectedRun,
                resolvedRunId = uiState.runId,
                onSelect = viewModel::selectRun,
                modifier = Modifier.weight(1f),
            )
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
            PorcelainCard {
                Text(
                    text = message,
                    color = rankoColors.qualityRed,
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}

@Composable
private fun PathRow(config: JsonObject, runId: String?) {
    val loggingDir = config.string("logging_dir")
    val outputDir = config.string("output_dir")
    val run = runId?.takeIf { it.isNotBlank() }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        PathChip("Run", run ?: "—")
        PathChip("Logs", run?.let { "$loggingDir/$it" } ?: "—")
        PathChip("Output", run?.let { "$outputDir/$it" } ?: "—")
    }
}

@Composable
private fun MetricsSection(uiState: DashboardUiState) {
    val stats = uiState.latestStats
    if (stats.isEmpty()) {
        PorcelainCard {
            Text(
                "No TensorBoard logs found yet. Waiting for training to start...",
                style = MaterialTheme.typography.bodyMedium,
                color = rankoColors.textDim,
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

    val colors = rankoColors
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val isWide = maxWidth > 720.dp
        if (isWide) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                MetricCard("Current Step", currentStep, colors.accentPink, Modifier.weight(1f))
                MetricCard("Latest Loss", latestLoss, colors.qualityRed, Modifier.weight(1f))
                MetricCard("UNet LR", unetLr, colors.accentBlue, Modifier.weight(1f))
                MetricCard("TE Effective LR", teLr, colors.accentLilac, Modifier.weight(1f))
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard("Current Step", currentStep, colors.accentPink, Modifier.weight(1f))
                    MetricCard("Latest Loss", latestLoss, colors.qualityRed, Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard("UNet LR", unetLr, colors.accentBlue, Modifier.weight(1f))
                    MetricCard("TE Effective LR", teLr, colors.accentLilac, Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun ChartsSection(
    uiState: DashboardUiState,
    onPickStep: (Float, Offset) -> Unit,
) {
    val metrics = uiState.metrics
    val smoothing = uiState.smoothing
    val stroke = uiState.chartStroke
    val colors = rankoColors
    val pickMarkers = uiState.chartPick?.let {
        ChartPickMarkers(clickedStep = it.step, matchedStep = it.checkpoint?.step)
    }
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val isWide = maxWidth > 720.dp
        Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            ChartCard(
                "Train / Avg Loss",
                metrics["Train/Avg_Loss"].orEmpty(),
                colors.accentPink,
                smoothing = 0f,
                modifier = Modifier.fillMaxWidth(),
                strokeWidth = stroke,
                chartHeight = 280.dp,
                onPickStep = onPickStep,
                showHoverStep = true,
                pickMarkers = pickMarkers,
            )
            if (isWide) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    TrainingChartCard("Train / Loss", metrics["Train/Loss"].orEmpty(), colors.qualityRed, smoothing, stroke, Modifier.weight(1f))
                    TrainingChartCard("UNet / LR", metrics["UNet/LR/Effective_Actual_LR"].orEmpty(), colors.accentBlue, smoothing, stroke, Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    TrainingChartCard("TE / Base LR", metrics["TE/LR/Base_Scheduled"].orEmpty(), colors.qualityOrange, smoothing, stroke, Modifier.weight(1f))
                    TrainingChartCard("TE / Effective LR", metrics["TE/LR/Effective_Actual_LR"].orEmpty(), colors.accentLilac, smoothing, stroke, Modifier.weight(1f))
                }
            } else {
                TrainingChartCard("Train / Loss", metrics["Train/Loss"].orEmpty(), colors.qualityRed, smoothing, stroke, Modifier.fillMaxWidth())
                TrainingChartCard("UNet / LR", metrics["UNet/LR/Effective_Actual_LR"].orEmpty(), colors.accentBlue, smoothing, stroke, Modifier.fillMaxWidth())
                TrainingChartCard("TE / Base LR", metrics["TE/LR/Base_Scheduled"].orEmpty(), colors.qualityOrange, smoothing, stroke, Modifier.fillMaxWidth())
                TrainingChartCard("TE / Effective LR", metrics["TE/LR/Effective_Actual_LR"].orEmpty(), colors.accentLilac, smoothing, stroke, Modifier.fillMaxWidth())
            }
        }
    }
}

/** Training chart with the always-on hover step readout. */
@Composable
private fun TrainingChartCard(
    title: String,
    points: List<MetricPoint>,
    color: Color,
    smoothing: Float,
    stroke: Float,
    modifier: Modifier,
) {
    ChartCard(
        title = title,
        points = points,
        color = color,
        smoothing = smoothing,
        modifier = modifier,
        strokeWidth = stroke,
        showHoverStep = true,
    )
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
            color = rankoColors.textDim
        )
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = range
        )
    }
}

/**
 * Above the cards: the step range a batch would cover and the button that starts it, plus the
 * progress of the batch in flight with the button that stops it. The range is prefilled with the
 * run's own steps, so "all of them" is one click.
 */
@Composable
internal fun SampleRangeRow(
    rows: List<CheckpointRow>,
    batch: GeneratedSampleJob?,
    starting: Boolean,
    canStart: Boolean,
    note: String?,
    onSampleRange: (Int, Int) -> Unit,
    onCancel: (String) -> Unit,
) {
    val colors = rankoColors
    val steps = checkpointSteps(rows)
    val defaultFrom = steps.firstOrNull()?.toString() ?: ""
    val defaultTo = steps.lastOrNull()?.toString() ?: ""
    var from by remember(defaultFrom) { mutableStateOf(defaultFrom) }
    var to by remember(defaultTo) { mutableStateOf(defaultTo) }
    val error = batchRangeError(rows, from, to)
    val fromStep = from.trim().toIntOrNull()
    val toStep = to.trim().toIntOrNull()

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        batch?.let { running ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = "Sampling steps ${running.fromStep ?: 0}–${running.toStep ?: 0} · " +
                        (batchProgressLabel(running) ?: ""),
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.accentPink,
                    modifier = Modifier.weight(1f),
                )
                CapsuleButton(
                    text = "Stop",
                    onClick = { onCancel(running.id) },
                    enabled = !running.cancelRequested,
                    compact = true,
                ) {
                    Text("Stop", fontWeight = FontWeight.SemiBold)
                }
            }
            if (running.totalImages > 0) {
                LinearProgressIndicator(
                    progress = {
                        running.imagesDone.toFloat().coerceAtMost(running.totalImages.toFloat()) /
                            running.totalImages
                    },
                    modifier = Modifier.fillMaxWidth().height(4.dp),
                )
            }
        }

        if (steps.isNotEmpty()) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = "Sample range",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.textDim,
                )
                val editable = canStart && batch == null
                OutlinedTextField(
                    value = from,
                    onValueChange = { text -> from = text.filter { it.isDigit() }.take(7) },
                    singleLine = true,
                    enabled = editable,
                    modifier = Modifier.width(96.dp),
                    label = { Text("from") },
                    textStyle = MaterialTheme.typography.bodySmall,
                    colors = rankoFieldColors(),
                    shape = rankoTokens.panel,
                )
                OutlinedTextField(
                    value = to,
                    onValueChange = { text -> to = text.filter { it.isDigit() }.take(7) },
                    singleLine = true,
                    enabled = editable,
                    modifier = Modifier.width(96.dp),
                    label = { Text("to") },
                    textStyle = MaterialTheme.typography.bodySmall,
                    colors = rankoFieldColors(),
                    shape = rankoTokens.panel,
                )
                val covered = if (fromStep != null && toStep != null) {
                    checkpointsInRange(rows, fromStep, toStep).size
                } else {
                    0
                }
                CapsuleButton(
                    text = if (starting) "Starting…" else "Sample range",
                    onClick = { if (fromStep != null && toStep != null) onSampleRange(fromStep, toStep) },
                    enabled = editable && !starting && error == null,
                    compact = true,
                ) {
                    Text(if (starting) "Starting…" else "Sample range", fontWeight = FontWeight.SemiBold)
                }
                if (batch == null && covered > 0) {
                    Text(
                        text = "$covered checkpoint(s)",
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.textDim,
                    )
                }
            }

            val reason = when {
                batch != null -> null
                !canStart -> "Pause the run, or stop it, to free the GPU"
                else -> error
            }
            reason?.let { message ->
                Text(
                    text = message,
                    style = MaterialTheme.typography.labelSmall,
                    color = colors.textDim,
                )
            }
        }

        note?.let { message -> GenerationErrorLine(message) }
    }
}

/** The Checkpoints section with nothing to list: no run, or a run that wrote nothing yet. */
@Composable
internal fun CheckpointsEmptyCard() {
    PorcelainCard {
        Text(
            "No checkpoints or sample images for this run yet.",
            style = MaterialTheme.typography.bodyMedium,
            color = rankoColors.textDim,
        )
    }
}

/** A failed generation, above the section it belongs to. */
@Composable
internal fun GenerationErrorLine(message: String) {
    Text(
        text = "Generation failed: $message",
        style = MaterialTheme.typography.labelSmall,
        color = rankoColors.qualityRed,
        maxLines = 3,
        overflow = TextOverflow.Ellipsis,
    )
}

/** A pin the helper refused — no run resolved, or the checkpoint file is not there any more. */
@Composable
internal fun PinsErrorLine(message: String) {
    Text(
        text = "Pin failed: $message",
        style = MaterialTheme.typography.labelSmall,
        color = rankoColors.qualityRed,
        maxLines = 3,
        overflow = TextOverflow.Ellipsis,
    )
}

/** Where the run's pins are kept, so the state the user asked for is findable on disk. */
@Composable
internal fun PinnedHintLine(file: String?) {
    Text(
        text = "Pinned checkpoints first · kept in ${file ?: "checkpoint_pins.json"}",
        style = MaterialTheme.typography.labelSmall,
        color = rankoColors.textDim,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
    )
}

/** The line between the pinned cards and the rest of the section, with the split it makes. */
@Composable
internal fun CheckpointsDivider(pinned: Int, rest: Int) {
    val colors = rankoColors
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        HorizontalDivider(modifier = Modifier.weight(1f), color = colors.accentPink.copy(alpha = 0.35f))
        Text(
            text = "$pinned pinned · $rest more",
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
            color = colors.textDim,
        )
        HorizontalDivider(modifier = Modifier.weight(1f), color = colors.accentPink.copy(alpha = 0.35f))
    }
}

/**
 * One checkpoint's "Save As" state: the copy in flight, where it landed, or why it failed.
 * Shared by the Ctrl+click panel and the Checkpoints section's cards, which are the two places a
 * checkpoint can be saved from.
 */
@Composable
internal fun CheckpointExportStatus(inFlight: Boolean, result: CheckpointExport?) {
    val colors = rankoColors
    if (inFlight) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            LinearProgressIndicator(
                color = colors.accentPink,
                trackColor = colors.accentPink.copy(alpha = 0.18f),
                modifier = Modifier.weight(1f).height(4.dp),
            )
            Text("Saving…", style = MaterialTheme.typography.labelSmall, color = colors.accentPink)
        }
    }
    result?.savedPath?.let { path ->
        Text(
            text = "Saved → $path",
            style = MaterialTheme.typography.labelSmall,
            color = colors.qualityGreen,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
        )
    }
    result?.error?.let { message ->
        Text(
            text = "Save failed: $message",
            style = MaterialTheme.typography.labelSmall,
            color = colors.qualityRed,
            maxLines = 3,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

/**
 * One card per LoRA checkpoint of the run being shown, with the images written at its step (the
 * training ones and any generated pass). A checkpoint with no images yet is still listed — that is
 * the point of a retunable cadence and an optional sampling switch — and can be sampled from here
 * whenever nothing else is using the GPU.
 *
 * The card also carries the checkpoint's evaluation: the pass still running for it (progress and a
 * Cancel), or the newest finished one — its F1, a details block, or why it failed.
 *
 * A pinned checkpoint ([CheckpointRow.pinned]) leads the section, is drawn on an accent-tinted
 * surface so it stands apart from the rest, and carries the pin button that put it there; the pin
 * list is the run's own state, kept in its log directory by the helper.
 */
@Composable
internal fun CheckpointRowCard(
    row: CheckpointRow,
    thumbSize: Float,
    showSetBadges: Boolean,
    newJobIds: Set<String>,
    gpuFree: Boolean,
    starting: Boolean,
    busyElsewhere: Boolean,
    startingEvaluation: Boolean,
    pinning: Boolean,
    pinEnabled: Boolean,
    exportInFlightPath: String?,
    exportResult: CheckpointExport?,
    onOpen: (SampleItem) -> Unit,
    onGenerate: (CheckpointItem) -> Unit,
    onEvaluate: (CheckpointItem, Int, String?) -> Unit,
    onCancelEvaluation: (String) -> Unit,
    /** Opens the evaluation panel on one job of this card. */
    onOpenEvaluation: (String) -> Unit,
    onTogglePin: (CheckpointItem) -> Unit,
    onSaveAs: (CheckpointItem) -> Unit,
) {
    val colors = rankoColors
    val checkpoint = row.checkpoint
    val slots = sampleSlots(row.samples, row.generated, newJobIds)
    val thumbWidth = thumbSize.dp
    val thumbHeight = thumbWidth * SAMPLE_THUMB_ASPECT
    val saving = exportInFlightPath != null && exportInFlightPath == checkpoint?.path
    val saveResult = exportResult?.takeIf { it.path == checkpoint?.path }

    PorcelainCard(emphasized = row.pinned) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Box(
                    modifier = Modifier
                        .clip(rankoTokens.capsule)
                        .background(colors.accentPink.copy(alpha = if (checkpoint == null) 0.4f else 1f))
                        .padding(horizontal = 8.dp, vertical = 2.dp),
                ) {
                    Text(
                        text = checkpointRowLabel(row),
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White,
                    )
                }
                if (row.pinned) {
                    Box(
                        modifier = Modifier
                            .clip(rankoTokens.capsule)
                            .background(colors.accentPink.copy(alpha = 0.18f))
                            .padding(horizontal = 8.dp, vertical = 2.dp),
                    ) {
                        Text(
                            text = "Pinned",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = colors.accentPink,
                        )
                    }
                }
                Text(
                    text = checkpoint?.dir ?: "samples only",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = colors.text,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                if (checkpoint != null) {
                    val saveLabel = if (saving) "Saving…" else "Save As"
                    CapsuleButton(
                        text = saveLabel,
                        onClick = { onSaveAs(checkpoint) },
                        enabled = exportInFlightPath == null,
                        compact = true,
                    ) {
                        Icon(Icons.Default.Save, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(saveLabel, fontWeight = FontWeight.SemiBold)
                    }
                    val label = if (starting) "Starting…" else "Generate samples"
                    CapsuleButton(
                        text = label,
                        onClick = { onGenerate(checkpoint) },
                        enabled = gpuFree && !starting && !busyElsewhere,
                        compact = true,
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(label, fontWeight = FontWeight.SemiBold)
                    }
                    val evaluateLabel = if (startingEvaluation) "Starting…" else "Evaluate"
                    CapsuleButton(
                        text = evaluateLabel,
                        onClick = { onEvaluate(checkpoint, slots.size, cardEvaluation(row)?.id) },
                        enabled = gpuFree && !busyElsewhere && !startingEvaluation,
                        compact = true,
                    ) {
                        Icon(Icons.Default.Star, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(evaluateLabel, fontWeight = FontWeight.SemiBold)
                    }
                    val pinLabel = when {
                        pinning -> "Pinning…"
                        row.pinned -> "Unpin"
                        else -> "Pin"
                    }
                    CapsuleButton(
                        text = pinLabel,
                        onClick = { onTogglePin(checkpoint) },
                        enabled = pinEnabled,
                        emphasized = row.pinned,
                        compact = true,
                    ) {
                        Icon(
                            imageVector = if (row.pinned) Icons.Filled.PushPin else Icons.Outlined.PushPin,
                            contentDescription = pinLabel,
                            modifier = Modifier.size(14.dp),
                        )
                    }
                }
            }

            if (checkpoint != null) {
                Text(
                    text = checkpointSubtitle(checkpoint),
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.text,
                )
                Text(
                    text = checkpoint.path,
                    style = MaterialTheme.typography.labelSmall,
                    color = colors.textDim,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }

            if (slots.isEmpty()) {
                Text(
                    text = if (checkpoint != null) {
                        "No sample images for this checkpoint."
                    } else {
                        "No sample images left for this step."
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.textDim,
                )
            } else {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    slots.forEach { slot ->
                        SampleSlotCard(
                            slot = slot,
                            width = thumbWidth,
                            height = thumbHeight,
                            setBadge = if (showSetBadges) sampleSetBadge(slot.item.setIndex) else null,
                            onOpen = { onOpen(slot.item) },
                        )
                    }
                }
            }

            val running = row.running
            generatedJobSetProgress(running)?.let { progress ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    val total = running?.totalImages ?: 0
                    LinearProgressIndicator(
                        progress = {
                            if (total > 0) {
                                (running?.imagesDone ?: 0).toFloat().coerceAtMost(total.toFloat()) / total
                            } else {
                                0f
                            }
                        },
                        modifier = Modifier.weight(1f).height(4.dp),
                    )
                    Text(progress, style = MaterialTheme.typography.labelSmall, color = colors.accentPink)
                    if (running != null && isEvaluation(running)) {
                        CapsuleButton(
                            text = "Cancel",
                            onClick = { onCancelEvaluation(running.id) },
                            compact = true,
                        )
                    }
                }
            }

            // A score is worth showing wherever its images are — the card, or a `samples only` row
            // for a step whose weights a Reset removed. The card carries the entry point only: the
            // result itself lives in the evaluation panel, where it stays readable.
            val evaluation = cardEvaluation(row)
            if (evaluation != null) {
                val entryLabel = when {
                    evaluation.state == "running" -> "Evaluation · running…"
                    evaluation.error != null -> "Evaluation · failed"
                    evaluation.scores != null -> "Evaluation · " +
                        (evaluationRecallHeadline(evaluation.scores) ?: "scored")
                    else -> "Evaluation"
                }
                CapsuleButton(
                    text = entryLabel,
                    onClick = { onOpenEvaluation(evaluation.id) },
                    compact = true,
                )
            }

            if (checkpoint != null) {
                CheckpointExportStatus(inFlight = saving, result = saveResult)
            }
            if (checkpoint != null && slots.isEmpty() && !gpuFree) {
                Text(
                    text = "Pause or stop the run to render this checkpoint's sample sets.",
                    style = MaterialTheme.typography.labelSmall,
                    color = colors.textDim,
                )
            }
            if (busyElsewhere) {
                Text(
                    text = "Another generation is using the GPU.",
                    style = MaterialTheme.typography.labelSmall,
                    color = colors.textDim,
                )
            }
        }
    }
}

/**
 * Floating panel for a Ctrl+click on the Avg Loss chart: the clicked step with its training scalars,
 * the checkpoint nearest to it, that step's samples, and a form that generates one extra sample from
 * the checkpoint. Sized to its content, resizable by its bottom-right grip. Clicking the scrim, the
 * close button or Esc dismisses.
 */
@Composable
private fun CheckpointPanelOverlay(
    pick: ChartPickState,
    metrics: Map<String, List<MetricPoint>>,
    samples: Map<String, List<SampleItem>>,
    generatedJobs: List<GeneratedSampleJob>,
    generatedError: String?,
    originInRoot: Offset,
    userSize: DpSize?,
    previewOpen: Boolean,
    gpuFree: Boolean,
    newJobIds: Set<String>,
    exportInFlightPath: String?,
    exportResult: CheckpointExport?,
    onResize: (DpSize) -> Unit,
    onOpenSample: (SampleItem) -> Unit,
    onSaveAs: (CheckpointItem) -> Unit,
    onToggleForm: () -> Unit,
    onUpdateForm: (ChartPickState.() -> ChartPickState) -> Unit,
    onGenerate: (Int?) -> Unit,
    onClose: () -> Unit,
) {
    val colors = rankoColors
    val density = LocalDensity.current
    val focusRequester = remember { FocusRequester() }
    LaunchedEffect(pick.step, previewOpen) {
        if (!previewOpen) focusRequester.requestFocus()
    }

    val checkpointStep = pick.checkpoint?.step
    val exactSamples = samplesForStep(samples, checkpointStep)
    val fallbackStep = if (exactSamples.isEmpty()) nearestSampledStep(samples, checkpointStep) else null
    val shownStep = fallbackStep ?: checkpointStep
    val trainingSamples = if (fallbackStep != null) samplesForStep(samples, fallbackStep) else exactSamples
    // Generated images join the row they belong to, so they sit next to the step's own samples.
    val slots = sampleSlots(
        trainingSamples,
        panelJobsForStep(generatedJobs, shownStep, pick.checkpoint),
        newJobIds,
    )

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.bgApp.copy(alpha = 0.35f))
            .focusRequester(focusRequester)
            .focusable()
            .onPreviewKeyEvent { event ->
                if (event.type == KeyEventType.KeyDown && event.key == Key.Escape) {
                    onClose()
                    true
                } else {
                    false
                }
            }
            .clickable(
                indication = null,
                interactionSource = remember { MutableInteractionSource() },
                onClick = onClose,
            ),
    ) {
        val maxPanelWidth = minOf(PANEL_MAX_WIDTH, maxWidth - 24.dp)
        val maxPanelHeight = minOf(PANEL_MAX_HEIGHT, maxHeight - 24.dp)
        val columns = sampleColumns(slots.size.coerceAtLeast(1))
        val defaultWidth = checkpointPanelWidth(maxPanelWidth, sampleThumbWidth(maxPanelWidth, columns), columns)
        val size = userSize?.let { clampPanelSize(it.width, it.height, maxPanelWidth, maxPanelHeight) }
        val panelWidth = size?.width ?: defaultWidth
        val panelHeight = size?.height
        // Slots follow whatever width the panel ended up with, so dragging it scales the samples.
        val slotWidth = sampleSlotWidth(panelWidth, columns)
        val slotHeight = slotWidth * SAMPLE_THUMB_ASPECT

        val gapPx = with(density) { PANEL_GAP.toPx() }
        val marginPx = with(density) { 8.dp.toPx() }
        val bounds = Size(
            with(density) { maxWidth.toPx() },
            with(density) { maxHeight.toPx() },
        )
        val origin = clampPanelOrigin(
            origin = placePanelOrigin(
                anchor = Offset(pick.anchor.x - originInRoot.x, pick.anchor.y - originInRoot.y),
                panelWidth = with(density) { defaultWidth.toPx() },
                minVisibleHeight = with(density) { minOf(PANEL_MIN_HEIGHT, maxPanelHeight).toPx() },
                bounds = bounds,
                gap = gapPx,
                margin = marginPx,
            ),
            panelSize = Size(
                with(density) { panelWidth.toPx() },
                with(density) { (panelHeight ?: maxPanelHeight).toPx() },
            ),
            bounds = bounds,
            margin = marginPx,
        )

        // Measured size, so a resize drag starts from what is actually on screen (the automatic
        // height is content-driven and not known before layout).
        val measured = remember { mutableStateOf(IntSize.Zero) }

        Box(
            modifier = Modifier
                .offset { IntOffset(origin.x.roundToInt(), origin.y.roundToInt()) }
                .width(panelWidth)
                .then(if (panelHeight != null) Modifier.height(panelHeight) else Modifier)
                .onGloballyPositioned { measured.value = it.size }
                // Swallow clicks so the panel does not dismiss itself.
                .clickable(
                    indication = null,
                    interactionSource = remember { MutableInteractionSource() },
                    onClick = {},
                ),
        ) {
            PorcelainCard(
                // A dragged height is a promise: the card has to fill it, otherwise it would wrap its
                // content and leave the resize grip floating below the visible panel.
                modifier = if (panelHeight != null) Modifier.fillMaxHeight() else Modifier,
            ) {
                Column(
                    modifier = Modifier
                        .heightIn(max = (panelHeight ?: maxPanelHeight) - PANEL_CARD_PADDING)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CheckpointPanelBody(
                        pick = pick,
                        metrics = metrics,
                        generatedJobs = generatedJobs,
                        generatedError = generatedError,
                        shownStep = shownStep,
                        slots = slots,
                        fallbackFromStep = fallbackStep?.let { checkpointStep },
                        slotWidth = slotWidth,
                        slotHeight = slotHeight,
                        gpuFree = gpuFree,
                        showSetBadges = showsSampleSetBadges(samples),
                        exportInFlightPath = exportInFlightPath,
                        exportResult = exportResult,
                        onOpenSample = onOpenSample,
                        onSaveAs = onSaveAs,
                        onToggleForm = onToggleForm,
                        onUpdateForm = onUpdateForm,
                        onGenerate = onGenerate,
                        onClose = onClose,
                    )
                }
            }

            ResizeGrip(
                modifier = Modifier.align(Alignment.BottomEnd),
                onDrag = { dragX, dragY ->
                    with(density) {
                        val base = if (measured.value.width > 0 && measured.value.height > 0) {
                            DpSize(measured.value.width.toDp(), measured.value.height.toDp())
                        } else {
                            DpSize(panelWidth, panelHeight ?: maxPanelHeight)
                        }
                        onResize(
                            clampPanelSize(
                                base.width + dragX.toDp(),
                                base.height + dragY.toDp(),
                                maxPanelWidth,
                                maxPanelHeight,
                            ),
                        )
                    }
                },
            )
        }
    }
}

/** Bottom-right grip: drag to resize the panel for the rest of the session. */
@Composable
private fun ResizeGrip(
    modifier: Modifier = Modifier,
    onDrag: (Float, Float) -> Unit,
) {
    val colors = rankoColors
    Box(
        modifier = modifier
            .size(22.dp)
            .pointerHoverIcon(pointerIconNwseResize)
            .pointerInput(Unit) {
                detectDragGestures { change, dragAmount ->
                    change.consume()
                    onDrag(dragAmount.x, dragAmount.y)
                }
            },
        contentAlignment = Alignment.Center,
    ) {
        Canvas(modifier = Modifier.size(11.dp)) {
            val stroke = 1.5f
            val color = colors.textDim.copy(alpha = 0.75f)
            for (i in 1..3) {
                val offset = i * 3.5f
                drawLine(
                    color = color,
                    start = Offset(size.width - offset, size.height),
                    end = Offset(size.width, size.height - offset),
                    strokeWidth = stroke,
                )
            }
        }
    }
}

@Composable
private fun CheckpointPanelBody(
    pick: ChartPickState,
    metrics: Map<String, List<MetricPoint>>,
    generatedJobs: List<GeneratedSampleJob>,
    generatedError: String?,
    shownStep: Int?,
    slots: List<SampleSlot>,
    fallbackFromStep: Int?,
    slotWidth: Dp,
    slotHeight: Dp,
    gpuFree: Boolean,
    showSetBadges: Boolean,
    exportInFlightPath: String?,
    exportResult: CheckpointExport?,
    onOpenSample: (SampleItem) -> Unit,
    onSaveAs: (CheckpointItem) -> Unit,
    onToggleForm: () -> Unit,
    onUpdateForm: (ChartPickState.() -> ChartPickState) -> Unit,
    onGenerate: (Int?) -> Unit,
    onClose: () -> Unit,
) {
    val colors = rankoColors
    val pickedStep = pick.step.roundToInt()

    Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "Step $pickedStep",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = colors.text,
            )
            Text(
                text = "Ctrl+clicked point on the Avg Loss chart",
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
            )
        }
        if (pick.isLoading) {
            CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
        }
        IconButton(onClick = onClose, modifier = Modifier.size(28.dp)) {
            Icon(Icons.Default.Close, contentDescription = "Close", tint = colors.text, modifier = Modifier.size(18.dp))
        }
    }

    MatchedCheckpointBlock(
        checkpoint = pick.checkpoint,
        pickedStep = pickedStep,
        isLoading = pick.isLoading,
    )

    TrainingInfoChips(metrics = metrics, step = pick.step)

    pick.error?.let { message ->
        Text(message, style = MaterialTheme.typography.labelSmall, color = colors.qualityRed)
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        val savingCheckpoint = exportInFlightPath != null &&
            exportInFlightPath == pick.checkpoint?.path
        CapsuleButton(
            text = if (savingCheckpoint) "Saving…" else "Save As",
            onClick = { pick.checkpoint?.let(onSaveAs) },
            enabled = pick.checkpoint != null && exportInFlightPath == null,
            compact = true,
        ) {
            Icon(Icons.Default.Save, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text(
                text = if (savingCheckpoint) "Saving…" else "Save As",
                fontWeight = FontWeight.SemiBold,
            )
        }
        CapsuleButton(
            text = "Generate sample",
            onClick = onToggleForm,
            enabled = pick.checkpoint != null,
            compact = true,
            emphasized = pick.isFormOpen,
        ) {
            Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text("Generate sample", fontWeight = FontWeight.SemiBold)
        }
    }

    CheckpointExportStatus(
        inFlight = exportInFlightPath != null && exportInFlightPath == pick.checkpoint?.path,
        result = exportResult?.takeIf { it.path == pick.checkpoint?.path },
    )

    if (pick.isFormOpen) {
        GenerateSamplePanel(
            pick = pick,
            generatedJobs = generatedJobs,
            generatedError = generatedError,
            gpuFree = gpuFree,
            onUpdateForm = onUpdateForm,
            onGenerate = { onGenerate(shownStep) },
        )
    }

    SampleRow(
        pick = pick,
        shownStep = shownStep,
        fallbackFromStep = fallbackFromStep,
        slots = slots,
        slotWidth = slotWidth,
        slotHeight = slotHeight,
        showSetBadges = showSetBadges,
        onOpenSample = onOpenSample,
    )
}

/** The checkpoint the click matched, leading the panel: name, own step, distance and identity. */
@Composable
private fun MatchedCheckpointBlock(
    checkpoint: CheckpointItem?,
    pickedStep: Int,
    isLoading: Boolean,
) {
    val colors = rankoColors
    val tokens = rankoTokens

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(tokens.panel)
            .background(colors.accentPink.copy(alpha = 0.12f))
            .border(1.dp, colors.accentPink.copy(alpha = 0.35f), tokens.panel)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Text(
            text = "MATCHED CHECKPOINT",
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
            color = colors.accentPink,
        )
        when {
            checkpoint != null -> {
                Text(
                    text = checkpoint.dir.ifBlank { checkpoint.filename },
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = colors.text,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    checkpoint.step?.let { step ->
                        StepBadge(step = step, final = checkpoint.final)
                    }
                    checkpointStepDistance(checkpoint.step, pickedStep)?.let { distance ->
                        Text(
                            text = distance,
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.SemiBold,
                            color = colors.accentPink,
                        )
                    }
                }
                Text(
                    text = checkpointSubtitle(checkpoint),
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.text,
                )
                Text(
                    text = checkpoint.path,
                    style = MaterialTheme.typography.labelSmall,
                    color = colors.textDim,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }

            isLoading -> Text(
                text = "Scanning run checkpoints…",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )

            else -> Text(
                text = "No checkpoints found for this run yet.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
        }
    }
}

@Composable
private fun StepBadge(step: Int, final: Boolean) {
    val colors = rankoColors
    Box(
        modifier = Modifier
            .clip(rankoTokens.panel)
            .background(colors.accentPink)
            .padding(horizontal = 8.dp, vertical = 2.dp),
    ) {
        Text(
            text = if (final) "step $step · final" else "step $step",
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.SemiBold,
            color = Color.White,
        )
    }
}

/** Training scalars at the clicked step: loss and both learning rates the trainer logs. */
@Composable
private fun TrainingInfoChips(metrics: Map<String, List<MetricPoint>>, step: Float) {
    val stats = trainingInfoAt(metrics, step)
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            text = "TRAINING AT STEP ${step.roundToInt()}",
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
            color = rankoColors.textDim,
        )
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(18.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            stats.forEach { stat ->
                CompactMetric(label = stat.label, value = stat.value)
            }
        }
    }
}

/** Prompt/CFG form for one extra sample of the matched checkpoint. */
@Composable
private fun GenerateSamplePanel(
    pick: ChartPickState,
    generatedJobs: List<GeneratedSampleJob>,
    generatedError: String?,
    /** True while a generation may start: no live trainer is using the GPU (a paused one is ok). */
    gpuFree: Boolean,
    onUpdateForm: (ChartPickState.() -> ChartPickState) -> Unit,
    onGenerate: () -> Unit,
) {
    val colors = rankoColors
    val running = runningJob(generatedJobs)
    val busy = pick.isGenerating || running != null

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        HorizontalDivider(color = colors.stroke.copy(alpha = 0.5f))
        GenerateTextField(
            label = "Prompt",
            value = pick.prompt,
            onValueChange = { text -> onUpdateForm { copy(prompt = text) } },
            minLines = 3,
        )
        GenerateTextField(
            label = "Negative prompt",
            value = pick.negativePrompt,
            onValueChange = { text -> onUpdateForm { copy(negativePrompt = text) } },
            minLines = 2,
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            GenerateTextField(
                label = "CFG",
                value = pick.cfg,
                onValueChange = { text -> onUpdateForm { copy(cfg = text) } },
                modifier = Modifier.width(110.dp),
            )
            GenerateTextField(
                label = "Steps",
                value = pick.steps,
                onValueChange = { text -> onUpdateForm { copy(steps = text) } },
                modifier = Modifier.width(110.dp),
            )
            GenerateTextField(
                label = "Seed (0 = random)",
                value = pick.seed,
                onValueChange = { text -> onUpdateForm { copy(seed = text) } },
                modifier = Modifier.weight(1f),
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            CapsuleButton(
                text = "Generate",
                onClick = onGenerate,
                enabled = !busy && gpuFree && pick.checkpoint != null,
                emphasized = true,
                compact = true,
            )
            val note = when {
                busy -> "A generation is running…"
                !gpuFree -> "Pause the run, or stop it, to free the GPU"
                else -> "One extra image, saved next to this run's samples"
            }
            Text(
                text = note,
                style = MaterialTheme.typography.labelSmall,
                color = if (gpuFree) colors.textDim else colors.qualityRed,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        generatedJobProgress(running)?.let { progress ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                val total = running?.totalSteps ?: 0
                val done = running?.currentStep ?: 0
                LinearProgressIndicator(
                    progress = { if (total > 0) (done.toFloat() / total).coerceIn(0f, 1f) else 0f },
                    modifier = Modifier.weight(1f).height(4.dp),
                )
                Text(progress, style = MaterialTheme.typography.labelSmall, color = colors.accentPink)
            }
        }
        pick.formError?.let { message ->
            Text(message, style = MaterialTheme.typography.labelSmall, color = colors.qualityRed)
        }
        generatedError?.let { message ->
            Text(
                text = "Generation failed: $message",
                style = MaterialTheme.typography.labelSmall,
                color = colors.qualityRed,
                maxLines = 4,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun GenerateTextField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    minLines: Int = 1,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier,
        label = { Text(label) },
        singleLine = minLines == 1,
        minLines = minLines,
        textStyle = MaterialTheme.typography.bodySmall,
        colors = rankoFieldColors(),
        shape = rankoTokens.panel,
    )
}

/** This step's samples, with generated images beside them and highlighted. */
@Composable
private fun SampleRow(
    pick: ChartPickState,
    shownStep: Int?,
    fallbackFromStep: Int?,
    slots: List<SampleSlot>,
    slotWidth: Dp,
    slotHeight: Dp,
    showSetBadges: Boolean,
    onOpenSample: (SampleItem) -> Unit,
) {
    val colors = rankoColors
    if (shownStep == null) return

    val training = slots.count { it.job == null }
    val generated = slots.size - training
    Spacer(Modifier.height(2.dp))
    Text(
        text = buildString {
            append("Samples at step ").append(shownStep)
            if (fallbackFromStep != null) append(" (nearest sampled step to ").append(fallbackFromStep).append(")")
            if (slots.isNotEmpty()) {
                append(" · ").append(training).append(" from training")
                if (generated > 0) append(" · ").append(generated).append(" generated")
            }
        },
        style = MaterialTheme.typography.labelSmall,
        color = colors.textDim,
    )
    if (slots.isEmpty()) {
        Text(
            text = "No sample images for this step yet.",
            style = MaterialTheme.typography.bodySmall,
            color = colors.textDim,
        )
        if (pick.checkpoint != null) {
            Text(
                text = "Use Generate sample above to make one from this checkpoint.",
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
            )
        }
        return
    }

    // Wraps instead of scrolling: extra generated images get their own row, nothing is clipped.
    FlowRow(
        modifier = Modifier.fillMaxWidth(),
        maxItemsInEachRow = SAMPLES_PER_ROW,
        horizontalArrangement = Arrangement.spacedBy(SAMPLE_SLOT_SPACING),
        verticalArrangement = Arrangement.spacedBy(SAMPLE_SLOT_SPACING),
    ) {
        slots.forEach { slot ->
            SampleSlotCard(
                slot = slot,
                width = slotWidth,
                height = slotHeight,
                setBadge = if (showSetBadges) sampleSetBadge(slot.item.setIndex) else null,
                onOpen = { onOpenSample(slot.item) },
            )
        }
    }
}

@Composable
private fun SampleSlotCard(
    slot: SampleSlot,
    width: Dp,
    height: Dp,
    setBadge: String?,
    onOpen: () -> Unit,
) {
    val colors = rankoColors
    val generated = slot.job != null
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .width(width)
            .clip(rankoTokens.panel)
            .background(
                if (generated) colors.accentPink.copy(alpha = if (slot.isNew) 0.22f else 0.12f)
                else colors.bgCard.copy(alpha = 0.72f),
            )
            .then(
                if (generated) {
                    Modifier.border(
                        width = if (slot.isNew) 2.dp else 1.dp,
                        color = colors.accentPink.copy(alpha = if (slot.isNew) 1f else 0.6f),
                        shape = rankoTokens.panel,
                    )
                } else {
                    Modifier
                },
            )
            .pointerHoverIcon(pointerIconHand)
            .clickable(onClick = onOpen)
            .padding(bottom = 6.dp),
    ) {
        Box(modifier = Modifier.width(width).height(height)) {
            AsyncImage(
                model = BlobRef(slot.item.path, maxEdge = 512, quality = LocalThumbnailQuality.current),
                contentDescription = slot.item.filename,
                contentScale = ContentScale.Fit,
                filterQuality = FilterQuality.Low,
                modifier = Modifier.fillMaxSize().clip(rankoTokens.panel),
            )
            if (generated) {
                Text(
                    text = if (slot.isNew) "NEW" else "GENERATED",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White,
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(3.dp)
                        .clip(rankoTokens.panel)
                        .background(colors.accentPink)
                        .padding(horizontal = 6.dp, vertical = 1.dp),
                )
            }
            if (setBadge != null) {
                Text(
                    text = setBadge,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White,
                    modifier = Modifier
                        .align(if (generated) Alignment.TopEnd else Alignment.TopStart)
                        .padding(3.dp)
                        .clip(rankoTokens.panel)
                        .background(colors.accentLilac.copy(alpha = 0.85f))
                        .padding(horizontal = 6.dp, vertical = 1.dp),
                )
            }
        }
        Spacer(Modifier.height(4.dp))
        Text(
            text = slot.job?.let { generatedJobCaption(it) } ?: slot.item.filename,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = if (generated) FontWeight.SemiBold else FontWeight.Normal,
            color = if (generated) colors.accentPink else colors.textDim,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(horizontal = 6.dp).width(width),
        )
    }
}

private fun checkpointStepDistance(checkpointStep: Int?, pickedStep: Int): String? {
    if (checkpointStep == null) return null
    val delta = checkpointStep - pickedStep
    return when {
        delta == 0 -> "same step as the click point"
        delta < 0 -> "${-delta} steps before the click point"
        else -> "$delta steps after the click point"
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
    ImagePreviewOverlay(
        images = samples.map { PreviewImage(path = it.path, title = it.filename) },
        index = index,
        onClose = onClose,
        onPrev = onPrev,
        onNext = onNext,
    )
}

private fun JsonObject.string(key: String): String {
    val value = this[key]?.jsonPrimitive?.content
    return if (value.isNullOrBlank()) "N/A" else value
}

/** Flattened config value, or null when the helper has not answered yet. */
private fun JsonObject.int(key: String): Int? = this[key]?.jsonPrimitive?.intOrNull

/** Flattened config flag; a hand-edited `0`/`1` reads as a boolean too. */
private fun JsonObject.flag(key: String): Boolean? =
    this[key]?.jsonPrimitive?.let { it.booleanOrNull ?: (it.intOrNull?.let { value -> value != 0 }) }

private val PANEL_GAP = 16.dp

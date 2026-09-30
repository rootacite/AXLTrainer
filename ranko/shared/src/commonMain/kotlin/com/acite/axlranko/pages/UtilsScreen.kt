package com.acite.axlranko.pages

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Bookmarks
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.GridOn
import androidx.compose.material.icons.filled.Hub
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.OpenInFull
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.PowerSettingsNew
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Restore
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.TextFields
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import com.acite.axlranko.data.ConfigProfile
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.DEFAULT_TAGGER_CATEGORY
import com.acite.axlranko.model.ModelSpecCatalog
import com.acite.axlranko.model.OUTPUT_NAME_HINT
import com.acite.axlranko.model.SAMPLE_SET_ERROR_PREFIX
import com.acite.axlranko.model.SampleSetForm
import com.acite.axlranko.model.TaggerMark
import com.acite.axlranko.model.TRAIN_DATA_ERROR_PREFIX
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.UtilsUiState
import com.acite.axlranko.model.taggerThresholdMarks
import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import com.acite.axlranko.pages.components.DatasetDirBar
import com.acite.axlranko.pages.components.datasetDirLabel
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.RankoChoiceRow
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.AppWindow
import com.acite.axlranko.util.LocalAppWindow
import com.acite.axlranko.util.checkpointSubtitle
import com.acite.axlranko.util.formatBytes
import com.acite.axlranko.util.formatFixed
import dev.zacsweers.metrox.viewmodel.metroViewModel
import com.acite.axlranko.localWallpaperModel
import com.acite.axlranko.data.showsHelperEndpointSettings
import com.acite.axlranko.data.wallpaperImagesSupported
import com.acite.axlranko.ui.pointerIconHorizontalResize
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import kotlin.time.Instant
import kotlin.math.roundToInt

@Composable
public fun UtilsScreen(
    viewModel: UtilsScreenViewModel = metroViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    if (uiState.errorMessage != null &&
        !uiState.isLoading &&
        uiState.configPath.isEmpty() &&
        !showsHelperEndpointSettings
    ) {
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
                    CapsuleButton(text = "Retry", onClick = { viewModel.loadConfig() }, emphasized = true)
                }
            }
        }
        return
    }

    if (uiState.isLoading && !showsHelperEndpointSettings) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        return
    }

    val colors = rankoColors
    val appWindow = LocalAppWindow.current
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val totalWidthPx = constraints.maxWidth.toFloat()

        Row(modifier = Modifier.fillMaxSize()) {
            Box(modifier = Modifier.fillMaxHeight().weight(uiState.leftWeight)) {
                SectionNav(
                    uiState = uiState,
                    appWindow = appWindow,
                    onSelect = viewModel::selectSection
                )
            }

            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(8.dp)
                    .pointerHoverIcon(pointerIconHorizontalResize)
                    .pointerInput(totalWidthPx) {
                        detectHorizontalDragGestures { _, dragAmount ->
                            if (totalWidthPx > 0) {
                                val fraction = dragAmount / totalWidthPx
                                viewModel.updateLeftWeight(uiState.leftWeight + fraction)
                            }
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                VerticalDivider(thickness = 1.dp, color = colors.stroke.copy(alpha = 0.55f))
            }

            Column(
                modifier = Modifier
                    .fillMaxHeight()
                    .weight(1f - uiState.leftWeight)
            ) {
                ConfigHeader(uiState = uiState, viewModel = viewModel)
                HorizontalDivider(color = colors.stroke.copy(alpha = 0.55f))
                Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                    val scroll = rememberScrollState()
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(scroll)
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = uiState.selectedSection.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = uiState.selectedSection.description,
                            style = MaterialTheme.typography.bodyMedium,
                            color = rankoColors.textDim
                        )
                        SectionFields(
                            uiState = uiState,
                            viewModel = viewModel,
                            appWindow = appWindow,
                        )
                    }
                    VerticalScrollbar(
                        modifier = Modifier
                            .align(Alignment.CenterEnd)
                            .fillMaxHeight()
                            .padding(vertical = 8.dp),
                        adapter = rememberScrollbarAdapter(scroll)
                    )
                }
            }
        }

        if (uiState.isSaving || uiState.isTagging) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().align(Alignment.TopCenter)
            )
        }

        if (uiState.checkpointPickerOpen) {
            CheckpointPickerDialog(uiState = uiState, viewModel = viewModel)
        }
    }
}

@Composable
private fun ConfigHeader(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel
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
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "Training Config",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    if (uiState.isDirty) {
                        Spacer(Modifier.width(10.dp))
                        AssistChip(
                            onClick = {},
                            enabled = false,
                            label = { Text("Unsaved") }
                        )
                    }
                }
                Text(
                    text = uiState.configPath.ifBlank { "config.toml" },
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.textDim,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = uiState.summaryLine,
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.accentPink
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CapsuleButton(
                    text = "Reload",
                    onClick = { viewModel.loadConfig() },
                    enabled = !uiState.isDirty && !uiState.isSaving && !uiState.isTagging,
                    compact = true,
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Reload", fontWeight = FontWeight.SemiBold)
                }
                CapsuleButton(
                    text = "Reset",
                    onClick = { viewModel.resetForm() },
                    enabled = uiState.isDirty && !uiState.isSaving && !uiState.isTagging,
                    compact = true,
                ) {
                    Icon(Icons.Default.Restore, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Reset", fontWeight = FontWeight.SemiBold)
                }
                CapsuleButton(
                    text = "Save",
                    onClick = { viewModel.saveConfig() },
                    enabled = uiState.isDirty && !uiState.isSaving && !uiState.isTagging,
                    compact = true,
                    emphasized = true,
                ) {
                    Icon(Icons.Default.Save, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Save", fontWeight = FontWeight.SemiBold)
                }
            }
        }

        uiState.errorMessage?.let { message ->
            StatusBanner(message = message, isError = true)
        }
        uiState.statusMessage?.let { message ->
            StatusBanner(message = message, isError = false)
        }
    }
}

@Composable
private fun StatusBanner(message: String, isError: Boolean) {
    val colors = rankoColors
    val container =
        if (isError) colors.qualityRed.copy(alpha = 0.18f)
        else colors.accentBlue.copy(alpha = 0.16f)
    val content =
        if (isError) colors.qualityRed
        else colors.accentLilac
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(container, rankoTokens.panel)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Icon(
            imageVector = if (isError) Icons.Default.Warning else Icons.Default.CheckCircle,
            contentDescription = null,
            tint = content,
            modifier = Modifier.size(18.dp)
        )
        Text(text = message, color = content, style = MaterialTheme.typography.bodyMedium)
    }
}

/**
 * The sections this build offers: Helper only where the endpoint is editable in-app, WM only where
 * there is a window to command (a desktop build).
 */
internal fun visibleSections(
    helperEndpointSettings: Boolean,
    appWindow: AppWindow?,
): List<ConfigSection> = ConfigSection.entries.filter {
    (it != ConfigSection.Helper || helperEndpointSettings) && (it != ConfigSection.Wm || appWindow != null)
}

@Composable
private fun SectionNav(
    uiState: UtilsUiState,
    appWindow: AppWindow?,
    onSelect: (ConfigSection) -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        val sections = visibleSections(showsHelperEndpointSettings, appWindow)
        items(sections, key = { it.name }) { section ->
            val selected = uiState.selectedSection == section
            val hasError = uiState.fieldErrors.keys.any { section.owns(it) }
            val colors = rankoColors
            RankoChoiceRow(
                selected = selected,
                onClick = { onSelect(section) },
            ) {
                Icon(
                    imageVector = section.icon(),
                    contentDescription = null,
                    modifier = Modifier.size(20.dp),
                    tint = when {
                        hasError -> colors.qualityRed
                        selected -> colors.accentPink
                        else -> colors.textDim
                    }
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    text = section.title,
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                    color = if (selected) colors.accentPink else colors.text,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                if (hasError) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = "Invalid fields",
                        modifier = Modifier.size(16.dp),
                        tint = colors.qualityRed
                    )
                }
            }
        }
    }
}

private fun ConfigSection.icon(): ImageVector = when (this) {
    ConfigSection.Helper -> Icons.Default.Cloud
    ConfigSection.Environment -> Icons.Default.Folder
    ConfigSection.Rocm -> Icons.Default.Memory
    ConfigSection.ModelSpec -> Icons.Default.Info
    ConfigSection.Training -> Icons.Default.Tune
    ConfigSection.Network -> Icons.Default.Hub
    ConfigSection.Bucketing -> Icons.Default.GridOn
    ConfigSection.Optimization -> Icons.Default.Bolt
    ConfigSection.UnetOptimizer -> Icons.Default.Layers
    ConfigSection.TeOptimizer -> Icons.Default.TextFields
    ConfigSection.Infrastructure -> Icons.Default.Settings
    ConfigSection.Validation -> Icons.Default.Photo
    ConfigSection.Appearance -> Icons.Default.Palette
    ConfigSection.Wm -> Icons.Default.OpenInFull
    ConfigSection.Profiles -> Icons.Default.Bookmarks
}

@Composable
private fun SectionFields(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel,
    appWindow: AppWindow?,
) {
    val form = uiState.form
    val errors = uiState.fieldErrors
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        when (uiState.selectedSection) {
            ConfigSection.Helper -> HelperFields(uiState, viewModel)
            ConfigSection.Environment -> EnvironmentFields(uiState, viewModel)
            ConfigSection.Rocm -> RocmFields(uiState, viewModel)
            ConfigSection.ModelSpec -> ModelSpecFields(form, errors, viewModel)
            ConfigSection.Training -> TrainingFields(form, errors, viewModel)
            ConfigSection.Network -> NetworkFields(form, errors, viewModel)
            ConfigSection.Bucketing -> BucketingFields(form, errors, viewModel)
            ConfigSection.Optimization -> OptimizationFields(form, errors, viewModel)
            ConfigSection.UnetOptimizer -> UnetFields(form, errors, viewModel)
            ConfigSection.TeOptimizer -> TeFields(form, errors, viewModel)
            ConfigSection.Infrastructure -> InfrastructureFields(form, errors, viewModel)
            ConfigSection.Validation -> ValidationFields(uiState, form, errors, viewModel)
            ConfigSection.Appearance -> AppearanceFields(uiState, viewModel)
            ConfigSection.Wm -> appWindow?.let { WmFields(it) }
            ConfigSection.Profiles -> ProfilesFields(uiState, viewModel)
        }
    }
}

/**
 * Maximize / restore and quit, for a session whose compositor draws no decorations: cage shows the
 * single application surface and nothing else, so there is no title bar to do either from.
 */
@Composable
private fun WmFields(window: AppWindow) {
    val colors = rankoColors
    // The window state is not observable, so the label follows our own toggle from an initial read.
    var maximized by remember(window) { mutableStateOf(window.maximized) }
    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Window",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "For a kiosk session with no window manager to click — cage, or any other " +
                    "single-window compositor. Maximize fills the screen; Exit quits the app " +
                    "through the same path as closing its window.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CapsuleButton(
                    text = maximizeButtonLabel(maximized),
                    onClick = { maximized = window.toggleMaximized() },
                ) {
                    Icon(
                        Icons.Default.OpenInFull,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(Modifier.width(6.dp))
                    Text(maximizeButtonLabel(maximized), fontWeight = FontWeight.SemiBold, maxLines = 1)
                }
                CapsuleButton(
                    text = "Exit",
                    onClick = { window.exit() },
                    danger = true,
                ) {
                    Icon(
                        Icons.Default.PowerSettingsNew,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(Modifier.width(6.dp))
                    Text("Exit", fontWeight = FontWeight.SemiBold, maxLines = 1)
                }
            }
        }
    }
}

/** `Maximize` fills the screen; the same button puts the remembered size back once it has. */
internal fun maximizeButtonLabel(maximized: Boolean): String =
    if (maximized) "Restore" else "Maximize"

@Composable
private fun HelperFields(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel,
) {
    val colors = rankoColors
    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Dashboard helper",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "WebSocket address of api.py. Defaults 127.0.0.1:18765. Saved in this browser. LAN needs --host 0.0.0.0 and --allow-ip on the helper.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            OutlinedTextField(
                value = uiState.helperHost,
                onValueChange = viewModel::updateHelperHost,
                label = { Text("Host") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                colors = rankoFieldColors(),
            )
            OutlinedTextField(
                value = uiState.helperPort,
                onValueChange = viewModel::updateHelperPort,
                label = { Text("Port") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                colors = rankoFieldColors(),
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CapsuleButton(
                    text = if (uiState.helperBusy) "Connecting" else "Connect",
                    onClick = { viewModel.connectHelper() },
                    enabled = !uiState.helperBusy &&
                        uiState.helperHost.isNotBlank() &&
                        uiState.helperPort.toIntOrNull() != null,
                    compact = true,
                    emphasized = true,
                )
                Text(
                    text = uiState.helperError ?: uiState.helperStatus,
                    style = MaterialTheme.typography.bodySmall,
                    color = if (uiState.helperError != null) colors.qualityRed else colors.textDim,
                )
            }
        }
    }
}

@Composable
private fun EnvironmentFields(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel
) {
    val form = uiState.form
    val errors = uiState.fieldErrors
    ConfigPathField(
        label = "Pretrained model path",
        value = form.pretrainedModelNameOrPath,
        error = errors["pretrained_model_name_or_path"],
        supporting = "Diffusers directory or single-file checkpoint for the selected base",
        onValueChange = { viewModel.updateForm { copy(pretrainedModelNameOrPath = it) } },
        onBrowse = {
            viewModel.browseDirectory(form.pretrainedModelNameOrPath) {
                copy(pretrainedModelNameOrPath = it)
            }
        }
    )
    TrainDataDirsField(uiState, viewModel)
    AutoTagCard(uiState = uiState, viewModel = viewModel)
    ConfigTextField(
        label = "Output name",
        value = form.outputName,
        error = errors["output_name"],
        supporting = "LoRA filename stem written under the output directory; $OUTPUT_NAME_HINT",
        onValueChange = { viewModel.updateForm { copy(outputName = it) } }
    )
    ConfigPathField(
        label = "Output directory",
        value = form.outputDir,
        error = errors["output_dir"],
        onValueChange = { viewModel.updateForm { copy(outputDir = it) } },
        onBrowse = { viewModel.browseDirectory(form.outputDir) { copy(outputDir = it) } }
    )
    ConfigPathField(
        label = "Logging directory",
        value = form.loggingDir,
        error = errors["logging_dir"],
        onValueChange = { viewModel.updateForm { copy(loggingDir = it) } },
        onBrowse = { viewModel.browseDirectory(form.loggingDir) { copy(loggingDir = it) } }
    )
}

@Composable
private fun RocmFields(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel,
) {
    val form = uiState.form
    val errors = uiState.fieldErrors
    val colors = rankoColors
    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text(
                text = "Allocation patch",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "If you are using an RDNA 4 AMD GPU, strongly prefer the Tail or VMM patch. VMM uses less VRAM system-wide because it bypasses ROCr's Memory Pool. The two switches below are the workarounds for the driver bugs the 2026-09 kernel fixed: leave both off unless you are on an older kernel. Takes effect on the next Train start.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TrainingConfigForm.amdfqOptions.forEach { option ->
                    val label = when (option) {
                        "none" -> "None"
                        "tail" -> "Tail"
                        "vmm" -> "VMM"
                        else -> option
                    }
                    CapsuleChoice(
                        text = label,
                        selected = form.amdfq == option,
                        onClick = { viewModel.updateForm { copy(amdfq = option) } },
                    )
                }
            }
            uiState.fieldErrors["amdfq"]?.let { message ->
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.qualityRed,
                )
            }
            ConfigTextField(
                label = "Reserved VRAM (GiB)",
                value = form.amdfqVramReserveGib,
                onValueChange = { viewModel.updateForm { copy(amdfqVramReserveGib = it) } },
                error = errors["amdfq_vram_reserve_gib"],
                supporting = "Used only when Allocation patch is VMM. Optional workaround for the pre-fix kernel: the hook keeps this many GiB free on the card, reports that remaining to the allocator, and returns OOM rather than creating a block that would leave less. Remaining VRAM is the amdgpu sysfs counter, not hipMemGetInfo. Default 0 (off). Takes effect on the next Train start.",
            )
            ConfigSwitch(
                label = "VA never reused (legacy)",
                checked = form.amdfqVaNeverReuse,
                description = "Used only when Allocation patch is VMM. Pre-fix behaviour: a freed range keeps its GPU virtual address for the process lifetime, so the Dashboard's VA bar only grows. Off by default — with the 2026-09 kernel the hook gives the address back on free. Takes effect on the next Train start.",
                onChecked = { viewModel.updateForm { copy(amdfqVaNeverReuse = it) } },
            )
            Text(
                text = "Allocation pool",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Used only when Allocation patch is VMM. Off (default): every hipMalloc gets its own hipMemCreate. On: one hipMemCreate builds a pool of this size, and any request of at most half of it is carved out of one already built — no driver call at all. A pool is released when the upper layer has freed everything carved out of it. Takes effect on the next Train start.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                TrainingConfigForm.amdfqPoolMibOptions.forEach { size ->
                    CapsuleChoice(
                        text = if (size == 0) "Off" else "$size MiB",
                        selected = form.amdfqPoolMib.trim() == size.toString(),
                        onClick = { viewModel.updateForm { copy(amdfqPoolMib = size.toString()) } },
                    )
                }
            }
            uiState.fieldErrors["amdfq_pool_mib"]?.let { message ->
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.qualityRed,
                )
            }
            Text(
                text = "A pool is not better when bigger. It is committed VRAM the driver cannot hand to anything else until its last block is freed, so an oversized pool risks OOM and fragmenting the card, and the savings fall off: the traffic it removes is a bounded share of each training step. 16-64 MiB is the useful range.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
        }
    }
}

@Composable
private fun TrainDataDirsField(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel
) {
    val form = uiState.form
    val errors = uiState.fieldErrors
    val colors = rankoColors
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Train data directories", style = MaterialTheme.typography.labelLarge)
        form.trainDataDirs.forEachIndexed { index, entry ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.Top
            ) {
                ConfigTextField(
                    label = "Folder ${index + 1}",
                    value = entry.path,
                    error = errors["$TRAIN_DATA_ERROR_PREFIX$index.path"],
                    onValueChange = { value ->
                        viewModel.updateTrainDataDir(index) { it.copy(path = value) }
                    },
                    modifier = Modifier.weight(1f),
                    trailingIcon = {
                        IconButton(onClick = { viewModel.browseTrainDataDir(index) }) {
                            Icon(Icons.Default.FolderOpen, contentDescription = "Browse")
                        }
                    }
                )
                ConfigTextField(
                    label = "Repeat",
                    value = entry.repeat,
                    error = errors["$TRAIN_DATA_ERROR_PREFIX$index.repeat"],
                    onValueChange = { value ->
                        viewModel.updateTrainDataDir(index) { it.copy(repeat = value) }
                    },
                    modifier = Modifier.width(120.dp)
                )
                IconButton(
                    onClick = { viewModel.removeTrainDataDir(index) },
                    enabled = form.trainDataDirs.size > 1
                ) {
                    Icon(Icons.Default.Close, contentDescription = "Delete this folder")
                }
            }
        }
        CapsuleButton(
            text = "+ Add folder",
            onClick = { viewModel.addTrainDataDir() },
            compact = true,
        )
        Text(
            text = "Repeat is how often a folder's images are drawn inside one epoch: 3 against 1 " +
                "trains that folder three times as often per epoch. It changes how many times an " +
                "image is drawn, not the weight one image carries, and the epoch length, step " +
                "count and learning-rate schedule all follow the total. A small folder with a " +
                "large repeat can fill a batch with near-copies of the same few images.",
            style = MaterialTheme.typography.bodySmall,
            color = colors.textDim
        )
    }
}

@Composable
private fun AutoTagCard(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel
) {
    val thresholdValue = uiState.tagThreshold.toFloatOrNull()?.coerceIn(0f, 1f) ?: 0.35f
    val dirs = uiState.form.trainDataDirs
    val selectedDir = uiState.datasetDirIndex.coerceIn(0, dirs.lastIndex.coerceAtLeast(0))
    val targetPath = dirs.getOrNull(selectedDir)?.path?.trim().orEmpty()
    // Once per session: the categories and their calibrated thresholds come from the model, and
    // the question is answered locally (no GPU, no network).
    LaunchedEffect(Unit) { viewModel.loadTaggerInfo() }
    PorcelainCard {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Auto-tag dataset",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Runs the Pixai tagger v1 (PyTorch, ROCm GPU) over the selected dataset folder " +
                    "and overwrites every sidecar .txt. Images and Statistics reload when it finishes.",
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.textDim
            )
            DatasetDirBar(
                labels = dirs.map { datasetDirLabel(it.path, it.repeat) },
                selected = selectedDir,
                onSelect = viewModel::selectDatasetDir,
            )
            Text(
                text = "Categories",
                style = MaterialTheme.typography.labelLarge
            )
            val taggerInfo = uiState.taggerInfo
            val taggerCategories = taggerInfo?.categories.orEmpty().filter { it.key.isNotBlank() }
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3,
            ) {
                if (taggerCategories.isEmpty()) {
                    // Without the model's own list the card still has to be runnable: the tagger
                    // writes `general` by itself, and the reason line below says why that is all.
                    CapsuleChoice(text = DEFAULT_TAGGER_CATEGORY, selected = true, onClick = {})
                } else {
                    taggerCategories.forEach { category ->
                        CapsuleChoice(
                            text = category.key,
                            selected = category.key in uiState.tagCategories,
                            onClick = { viewModel.toggleTagCategory(category.key) },
                        )
                    }
                }
            }
            if (taggerInfo != null && !taggerInfo.available) {
                StatusBanner(
                    message = taggerInfo.reason.ifBlank { "The tagger model is not in the local cache." },
                    isError = true,
                )
            }
            Text(
                text = "Confidence  ${formatFixed(thresholdValue, 2)}",
                style = MaterialTheme.typography.labelLarge
            )
            Slider(
                value = thresholdValue,
                onValueChange = { viewModel.updateTagThreshold(formatFixed(it, 2)) },
                valueRange = 0f..1f,
                steps = 19,
                enabled = !uiState.isTagging
            )
            TaggerCalibrationRow(
                marks = taggerThresholdMarks(uiState.tagCategories, taggerCategories, thresholdValue),
                slider = thresholdValue,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                ConfigTextField(
                    label = "Threshold",
                    value = uiState.tagThreshold,
                    error = null,
                    supporting = "0.0 – 1.0  (default 0.35)",
                    onValueChange = viewModel::updateTagThreshold,
                    modifier = Modifier.weight(1f)
                )
                CapsuleButton(
                    text = if (uiState.isTagging) "Tagging…" else "Tag dataset",
                    onClick = { viewModel.runAutoTag() },
                    enabled = !uiState.isTagging && !uiState.isSaving && targetPath.isNotBlank(),
                    compact = true,
                    emphasized = true,
                ) {
                    if (uiState.isTagging) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp
                        )
                    } else {
                        Icon(Icons.Default.Bolt, contentDescription = null, modifier = Modifier.size(18.dp))
                    }
                    Spacer(Modifier.width(6.dp))
                    Text(
                        if (uiState.isTagging) "Tagging…" else "Tag dataset",
                        fontWeight = FontWeight.SemiBold,
                    )
                }
            }
        }
    }
}

/**
 * The model's own calibrated thresholds, under the slider that sits on top of them. A tick the
 * slider has passed is the one the slider has taken over, so it is drawn in the accent colour and
 * spelled out as `general 0.17 → 0.35`.
 */
@Composable
private fun TaggerCalibrationRow(marks: List<TaggerMark>, slider: Float) {
    if (marks.isEmpty()) return
    val colors = rankoColors
    Box(modifier = Modifier.fillMaxWidth().height(12.dp)) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val baseline = colors.textDim.copy(alpha = 0.30f)
            drawLine(
                color = baseline,
                start = Offset(0f, size.height),
                end = Offset(size.width, size.height),
                strokeWidth = 1f,
            )
            marks.forEach { mark ->
                val x = mark.calibrated.coerceIn(0f, 1f) * size.width
                drawLine(
                    color = if (mark.overridden) colors.accentPink else colors.textDim.copy(alpha = 0.65f),
                    start = Offset(x, 0f),
                    end = Offset(x, size.height),
                    strokeWidth = if (mark.overridden) 1.5f else 2f,
                )
            }
        }
    }
    Text(
        text = "Calibrated: " + marks.joinToString("  ·  ") { mark ->
            val value = formatFixed(mark.calibrated, 2)
            if (mark.overridden) "${mark.key} $value → ${formatFixed(slider, 2)}" else "${mark.key} $value"
        },
        style = MaterialTheme.typography.bodySmall,
        color = rankoColors.textDim,
    )
}

@Composable
private fun ModelSpecFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    val preset = ModelSpecCatalog.byVersion(form.baseModelVersion)
    ConfigDropdown(
        label = "Base model",
        value = form.baseModelVersion,
        options = ModelSpecCatalog.presets.map { it.baseModelVersion to it.label },
        onChange = { viewModel.updateForm { withBaseModelVersion(it) } },
        error = errors["base_model_version"],
        supporting = if (preset != null && !preset.trainable) {
            "Listed for future support; training will refuse this family"
        } else {
            "Selects architecture metadata written into checkpoints"
        }
    )
    ConfigTextField(
        label = "Architecture",
        value = form.modelspecArchitecture,
        error = errors["modelspec_architecture"],
        onValueChange = {},
        readOnly = true,
        supporting = "Filled from the selected base model"
    )
    ConfigTextField(
        label = "Implementation URL",
        value = form.modelspecImplementation,
        error = errors["modelspec_implementation"],
        onValueChange = {},
        readOnly = true
    )
    ConfigTextField(
        label = "SAI model spec",
        value = form.modelspecSaiModelSpec,
        error = errors["modelspec_sai_model_spec"],
        onValueChange = {},
        readOnly = true
    )
}

@Composable
private fun TrainingFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Epochs",
            value = form.epoch,
            error = errors["epoch"],
            onValueChange = { viewModel.updateForm { copy(epoch = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Batch size",
            value = form.trainBatchSize,
            error = errors["train_batch_size"],
            onValueChange = { viewModel.updateForm { copy(trainBatchSize = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Grad accumulation",
            value = form.gradientAccumulationSteps,
            error = errors["gradient_accumulation_steps"],
            supporting = effectiveBatchHint(form),
            onValueChange = { viewModel.updateForm { copy(gradientAccumulationSteps = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    Text("Mixed precision", style = MaterialTheme.typography.labelLarge, color = rankoColors.text)
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        TrainingConfigForm.mixedPrecisionOptions.forEach { option ->
            CapsuleChoice(
                text = option,
                selected = form.mixedPrecision == option,
                onClick = { viewModel.updateForm { copy(mixedPrecision = option) } },
            )
        }
    }
    ChoiceChips(
        label = "LR scheduler",
        value = form.lrScheduler,
        options = TrainingConfigForm.lrSchedulerOptions,
        onChange = { viewModel.updateForm { copy(lrScheduler = it) } }
    )
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Learning rate scale",
            value = form.learningRate,
            error = errors["learning_rate"],
            supporting = "Multiplier in front of UNet / TE rates",
            onValueChange = { viewModel.updateForm { copy(learningRate = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "LR warmup steps",
            value = form.lrWarmupSteps,
            error = errors["lr_warmup_steps"],
            onValueChange = { viewModel.updateForm { copy(lrWarmupSteps = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Min SNR gamma",
            value = form.minSnrGamma,
            error = errors["min_snr_gamma"],
            supporting = "epsilon bases only · 0 = off · 5.0 is a common SDXL starting point",
            onValueChange = { viewModel.updateForm { copy(minSnrGamma = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Max grad norm",
            value = form.maxGradNorm,
            error = errors["max_grad_norm"],
            onValueChange = { viewModel.updateForm { copy(maxGradNorm = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Seed",
            value = form.seed,
            error = errors["seed"],
            onValueChange = { viewModel.updateForm { copy(seed = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Save every N epochs",
            value = form.saveEveryNEpochs,
            error = errors["save_every_n_epochs"],
            onValueChange = { viewModel.updateForm { copy(saveEveryNEpochs = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Save every N steps",
            value = form.saveEveryNSteps,
            error = errors["save_every_n_steps"],
            supporting = "Start value; the Dashboard can retune it for the run in progress",
            onValueChange = { viewModel.updateForm { copy(saveEveryNSteps = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    ConfigSwitch(
        label = "Sampling",
        checked = form.samplingEnabled,
        description = "Render the validation samples at every checkpoint save " +
            "(off = checkpoints only; the Dashboard can flip it mid-run, and any checkpoint " +
            "can be sampled later, one pass per checkpoint)",
        onChecked = { viewModel.updateForm { copy(samplingEnabled = it) } }
    )
    ResumeCheckpointCard(form, errors, viewModel)
}

@Composable
private fun ResumeCheckpointCard(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    PorcelainCard {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Resume from LoRA checkpoint",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Loads LoRA weights into the UNet and both text encoders before training. " +
                    "Weights only: this run still counts steps from 0 and writes into its own " +
                    "timestamped run directory, so earlier runs are never overwritten. " +
                    "The checkpoint's network_dim / network_alpha must match this config.",
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.textDim
            )
            ConfigPathField(
                label = "Checkpoint file or its directory",
                value = form.resumeLoraPath,
                error = errors["resume_lora_path"],
                supporting = "Leave empty for a fresh run. Accepts any kohya LoRA .safetensors.",
                onValueChange = { viewModel.updateForm { copy(resumeLoraPath = it) } },
                onBrowse = { viewModel.browseCheckpointPath() }
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                CapsuleButton(
                    text = "Pick from run checkpoints",
                    onClick = { viewModel.openCheckpointPicker() },
                    compact = true,
                ) {
                    Icon(
                        Icons.Default.FolderOpen,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(Modifier.width(6.dp))
                    Text("Pick from run checkpoints", fontWeight = FontWeight.SemiBold)
                }
                CapsuleButton(
                    text = "Clear",
                    onClick = { viewModel.clearCheckpoint() },
                    enabled = form.resumeLoraPath.isNotBlank(),
                    compact = true,
                )
            }
        }
    }
}

@Composable
private fun CheckpointPickerDialog(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel
) {
    val checkpoints = uiState.checkpoints
    AlertDialog(
        onDismissRequest = { viewModel.closeCheckpointPicker() },
        title = { Text("Run checkpoints") },
        text = {
            Column(
                modifier = Modifier.width(620.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = "Checkpoints written by earlier runs of \"${uiState.form.outputName}\"." +
                        " Selecting one only fills the path field — save the config to apply it.",
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.textDim
                )
                when {
                    uiState.isLoadingCheckpoints -> Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                        Text("Scanning run directories…")
                    }
                    uiState.checkpointError != null -> Text(
                        text = uiState.checkpointError,
                        color = rankoColors.qualityRed,
                        style = MaterialTheme.typography.bodyMedium
                    )
                    checkpoints.isEmpty() -> Text(
                        text = "No checkpoints found yet. Finish a run first, or type/browse a path.",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    else -> LazyColumn(
                        modifier = Modifier.heightIn(max = 320.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        items(checkpoints, key = { it.path }) { item ->
                            CheckpointRow(item) { viewModel.selectCheckpoint(item) }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = { viewModel.loadCheckpoints() }) { Text("Refresh") }
        },
        dismissButton = {
            TextButton(onClick = { viewModel.closeCheckpointPicker() }) { Text("Close") }
        }
    )
}

@Composable
private fun CheckpointRow(
    checkpoint: CheckpointItem,
    onSelect: () -> Unit
) {
    RankoChoiceRow(selected = false, onClick = onSelect) {
        Column(verticalArrangement = Arrangement.spacedBy(2.dp), modifier = Modifier.weight(1f)) {
            Text(
                text = checkpoint.dir.ifBlank { checkpoint.filename },
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = checkpointSubtitle(checkpoint),
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.textDim
            )
            Text(
                text = checkpoint.path,
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.textDim,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun NetworkFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    Text("Network type", style = MaterialTheme.typography.labelLarge, color = rankoColors.text)
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        TrainingConfigForm.networkTypeOptions.forEach { option ->
            CapsuleChoice(
                text = if (option == "locon") "LoCon" else "Standard",
                selected = form.networkType == option,
                onClick = {
                    viewModel.updateForm {
                        if (option == "locon") {
                            val convEmpty = convDim.trim().toIntOrNull().let { it == null || it == 0 }
                            val alphaEmpty = convAlpha.trim().toIntOrNull().let { it == null || it == 0 }
                            copy(
                                networkType = option,
                                convDim = if (convEmpty) networkDim else convDim,
                                convAlpha = if (alphaEmpty) networkAlpha else convAlpha,
                            )
                        } else {
                            copy(networkType = option)
                        }
                    }
                },
            )
        }
    }
    errors["network_type"]?.let { message ->
        Text(message, style = MaterialTheme.typography.bodySmall, color = rankoColors.qualityRed)
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Network dim (rank)",
            value = form.networkDim,
            error = errors["network_dim"],
            onValueChange = { viewModel.updateForm { copy(networkDim = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Network alpha",
            value = form.networkAlpha,
            error = errors["network_alpha"],
            supporting = loraScaleHint(form),
            onValueChange = { viewModel.updateForm { copy(networkAlpha = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    if (form.networkType == "locon") {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            ConfigTextField(
                label = "Conv dim",
                value = form.convDim,
                error = errors["conv_dim"],
                supporting = convScaleHint(form),
                onValueChange = { viewModel.updateForm { copy(convDim = it) } },
                modifier = Modifier.weight(1f)
            )
            ConfigTextField(
                label = "Conv alpha",
                value = form.convAlpha,
                error = errors["conv_alpha"],
                supporting = convScaleHint(form),
                onValueChange = { viewModel.updateForm { copy(convAlpha = it) } },
                modifier = Modifier.weight(1f)
            )
        }
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Network dropout",
            value = form.networkDropout,
            error = errors["network_dropout"],
            supporting = "0.0 – 1.0",
            onValueChange = { viewModel.updateForm { copy(networkDropout = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "CLIP skip",
            value = form.clipSkip,
            error = errors["clip_skip"],
            onValueChange = { viewModel.updateForm { copy(clipSkip = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Max token length",
            value = form.maxTokenLength,
            error = errors["max_token_length"],
            onValueChange = { viewModel.updateForm { copy(maxTokenLength = it) } },
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun BucketingFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    ConfigSwitch(
        label = "Enable buckets",
        checked = form.enableBucket,
        description = "Group images by aspect ratio instead of forcing a square crop",
        onChecked = { viewModel.updateForm { copy(enableBucket = it) } }
    )
    ConfigSwitch(
        label = "No upscale",
        checked = form.bucketNoUpscale,
        description = "Never scale images up to reach the training resolution",
        onChecked = { viewModel.updateForm { copy(bucketNoUpscale = it) } }
    )
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Train resolution",
            value = form.trainResolution,
            error = errors["train_resolution"],
            supporting = bucketStepHint(form),
            onValueChange = { viewModel.updateForm { copy(trainResolution = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Bucket step",
            value = form.bucketResoSteps,
            error = errors["bucket_reso_steps"],
            onValueChange = { viewModel.updateForm { copy(bucketResoSteps = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Min bucket resolution",
            value = form.minBucketReso,
            error = errors["min_bucket_reso"],
            onValueChange = { viewModel.updateForm { copy(minBucketReso = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Max bucket resolution",
            value = form.maxBucketReso,
            error = errors["max_bucket_reso"],
            onValueChange = { viewModel.updateForm { copy(maxBucketReso = it) } },
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun OptimizationFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    ConfigSwitch(
        label = "Cache latents",
        checked = form.cacheLatents,
        description = "Encode VAE latents once and reuse them during training",
        onChecked = { viewModel.updateForm { copy(cacheLatents = it) } }
    )
    ConfigSwitch(
        label = "Cache latents to disk",
        checked = form.cacheLatentsToDisk,
        description = "Persist latent cache across runs (uses disk next to the dataset)",
        onChecked = { viewModel.updateForm { copy(cacheLatentsToDisk = it) } }
    )
    ConfigSwitch(
        label = "UNet gradient checkpointing",
        checked = form.gradientCheckpointingUnet,
        description = "Recompute UNet activations in backward to save VRAM; turn off for faster steps if you have headroom",
        onChecked = { viewModel.updateForm { copy(gradientCheckpointingUnet = it) } }
    )
    ConfigSwitch(
        label = "Text encoder gradient checkpointing",
        checked = form.gradientCheckpointingTe,
        description = "Same for CLIP-L / CLIP-G after PEFT wrap; also enables input grads on frozen embeddings",
        onChecked = { viewModel.updateForm { copy(gradientCheckpointingTe = it) } }
    )
    ConfigSwitch(
        label = "Shuffle caption",
        checked = form.shuffleCaption,
        description = "Shuffle comma-separated tags each step; keep_tokens stay fixed",
        onChecked = { viewModel.updateForm { copy(shuffleCaption = it) } }
    )
    ConfigSwitch(
        label = "Flush GPU memory every step",
        checked = form.flushMemoryEveryStep,
        description = "Call empty_cache after each training batch; turn off to reduce ROCm allocator churn",
        onChecked = { viewModel.updateForm { copy(flushMemoryEveryStep = it) } }
    )
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Keep tokens",
            value = form.keepTokens,
            error = errors["keep_tokens"],
            supporting = "Leading tags that are never shuffled",
            onValueChange = { viewModel.updateForm { copy(keepTokens = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Caption extension",
            value = form.captionExtension,
            error = errors["caption_extension"],
            onValueChange = { viewModel.updateForm { copy(captionExtension = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Noise offset",
            value = form.noiseOffset,
            error = errors["noise_offset"],
            supporting = "0.05 is typical for SDXL",
            onValueChange = { viewModel.updateForm { copy(noiseOffset = it) } },
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun UnetFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "UNet learning rate",
            value = form.unetLearningRate,
            error = errors["unet_learning_rate"],
            supporting = "Scientific notation is fine, e.g. 5e-5",
            onValueChange = { viewModel.updateForm { copy(unetLearningRate = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Weight decay",
            value = form.unetWeightDecay,
            error = errors["unet_weight_decay"],
            onValueChange = { viewModel.updateForm { copy(unetWeightDecay = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Beta 1",
            value = form.unetBetas1,
            error = errors["unet_betas_1"],
            onValueChange = { viewModel.updateForm { copy(unetBetas1 = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Beta 2",
            value = form.unetBetas2,
            error = errors["unet_betas_2"],
            onValueChange = { viewModel.updateForm { copy(unetBetas2 = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Epsilon",
            value = form.unetEps,
            error = errors["unet_eps"],
            onValueChange = { viewModel.updateForm { copy(unetEps = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    ConfigTextField(
        label = "UNet warmup steps",
        value = form.unetWarmupSteps,
        error = errors["unet_warmup_steps"],
        onValueChange = { viewModel.updateForm { copy(unetWarmupSteps = it) } }
    )
}

@Composable
private fun TeFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "TE learning rate",
            value = form.teLearningRate,
            error = errors["te_learning_rate"],
            supporting = "Usually 10× lower than the UNet rate",
            onValueChange = { viewModel.updateForm { copy(teLearningRate = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Weight decay",
            value = form.teWeightDecay,
            error = errors["te_weight_decay"],
            onValueChange = { viewModel.updateForm { copy(teWeightDecay = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Beta 1",
            value = form.teBetas1,
            error = errors["te_betas_1"],
            onValueChange = { viewModel.updateForm { copy(teBetas1 = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Beta 2",
            value = form.teBetas2,
            error = errors["te_betas_2"],
            onValueChange = { viewModel.updateForm { copy(teBetas2 = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Max grad norm",
            value = form.teMaxGradNorm,
            error = errors["te_max_grad_norm"],
            onValueChange = { viewModel.updateForm { copy(teMaxGradNorm = it) } },
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun InfrastructureFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    ConfigTextField(
        label = "DataLoader workers",
        value = form.maxDataLoaderNWorkers,
        error = errors["max_data_loader_n_workers"],
        supporting = "CPU workers used while filling batches",
        onValueChange = { viewModel.updateForm { copy(maxDataLoaderNWorkers = it) } }
    )
    ConfigSwitch(
        label = "Persistent workers",
        checked = form.persistentWorkers,
        description = "Keep worker processes alive between epochs",
        onChecked = { viewModel.updateForm { copy(persistentWorkers = it) } }
    )
}

@Composable
private fun ValidationFields(
    uiState: UtilsUiState,
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    val sets = form.sampleSets
    val selected = uiState.selectedSampleSet.coerceIn(0, sets.lastIndex.coerceAtLeast(0))
    val set = sets.getOrNull(selected) ?: SampleSetForm()
    if (sets.isEmpty()) return
    val key = { field: String -> "$SAMPLE_SET_ERROR_PREFIX$selected.$field" }

    SampleSetTabs(
        sets = sets,
        selected = selected,
        errors = errors,
        onSelect = viewModel::selectSampleSet,
        onAdd = viewModel::addSampleSet,
        onRemove = viewModel::removeSampleSet,
    )
    ConfigTextField(
        label = "Label",
        value = set.name,
        supporting = "Tab title; left blank it follows the prompt's first tag",
        onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(name = value) } }
    )
    ConfigTextField(
        label = "Positive prompt",
        value = set.prompt,
        error = errors[key("prompt")],
        onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(prompt = value) } },
        singleLine = false,
        minLines = 4
    )
    ConfigTextField(
        label = "Negative prompt",
        value = set.negative,
        error = errors[key("negative")],
        supporting = "Left blank the set samples without a negative prompt",
        onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(negative = value) } },
        singleLine = false,
        minLines = 3
    )
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Width",
            value = set.width,
            error = errors[key("width")],
            supporting = sampleAspectHint(set),
            onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(width = value) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Height",
            value = set.height,
            error = errors[key("height")],
            onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(height = value) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Steps",
            value = set.steps,
            error = errors[key("steps")],
            onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(steps = value) } },
            modifier = Modifier.weight(1f)
        )
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Guidance scale",
            value = set.guidanceScale,
            error = errors[key("guidance_scale")],
            onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(guidanceScale = value) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Guidance rescale",
            value = set.guidanceRescale,
            error = errors[key("guidance_rescale")],
            supporting = "0 = off, 0.6 = ComfyUI's RescaleCFG",
            onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(guidanceRescale = value) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Seed",
            value = set.seed,
            error = errors[key("seed")],
            supporting = "0 = a random seed per image",
            onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(seed = value) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Repeat",
            value = set.repeat,
            error = errors[key("repeat")],
            onValueChange = { value -> viewModel.updateSampleSet(selected) { it.copy(repeat = value) } },
            modifier = Modifier.weight(1f)
        )
    }
    Text(
        text = "Every checkpoint renders these sets in order; a fixed seed gives each set " +
            "the same starting noise, so only the prompts differ.",
        style = MaterialTheme.typography.bodySmall,
        color = rankoColors.textDim,
    )
}

/** Horizontal `[[validation.samples]]` tab strip with a trailing `+`. */
@Composable
private fun SampleSetTabs(
    sets: List<SampleSetForm>,
    selected: Int,
    errors: Map<String, String>,
    onSelect: (Int) -> Unit,
    onAdd: () -> Unit,
    onRemove: (Int) -> Unit,
) {
    var pendingRemoval by remember { mutableStateOf<Int?>(null) }
    Row(
        modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        sets.forEachIndexed { index, set ->
            SampleSetChip(
                label = sampleSetLabel(set, index),
                selected = index == selected,
                hasError = errors.keys.any { it.startsWith("$SAMPLE_SET_ERROR_PREFIX$index.") },
                onSelect = { onSelect(index) },
                onRemove = if (sets.size > 1) ({ pendingRemoval = index }) else null,
            )
        }
        CapsuleButton(
            text = "+",
            onClick = onAdd,
            compact = true,
        )
    }
    pendingRemoval?.let { index ->
        AlertDialog(
            onDismissRequest = { pendingRemoval = null },
            title = { Text("Delete ${sampleSetLabel(sets[index], index)}?") },
            text = {
                Text(
                    "The set and its prompt are removed from config.toml on save. " +
                        "The other sets are left alone."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    onRemove(index)
                    pendingRemoval = null
                }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { pendingRemoval = null }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun SampleSetChip(
    label: String,
    selected: Boolean,
    hasError: Boolean,
    onSelect: () -> Unit,
    onRemove: (() -> Unit)?,
) {
    val colors = rankoColors
    Row(
        modifier = Modifier
            .clip(rankoTokens.capsule)
            .background(if (selected) colors.accentPink.copy(alpha = 0.22f) else colors.bgPanel)
            .border(
                width = if (selected) 1.dp else 0.dp,
                color = if (selected) colors.accentPink else Color.Transparent,
                shape = rankoTokens.capsule
            )
            .clickable { onSelect() }
            .padding(horizontal = 14.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        if (hasError) {
            Icon(
                Icons.Default.Warning,
                contentDescription = "Invalid fields",
                modifier = Modifier.size(14.dp),
                tint = colors.qualityRed
            )
        }
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
            color = if (selected) colors.accentPink else colors.text,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.widthIn(max = 180.dp)
        )
        if (onRemove != null) {
            Icon(
                Icons.Default.Close,
                contentDescription = "Delete this sample set",
                modifier = Modifier.size(14.dp).clickable { onRemove() },
                tint = colors.textDim
            )
        }
    }
}

/**
 * Tab title: the set's own label, else its prompt's first tag, else the position.
 * Mirrors what the trainer prints, so a set is recognisable from either side.
 */
internal fun sampleSetLabel(set: SampleSetForm, index: Int): String {
    val name = set.name.trim()
    if (name.isNotEmpty()) return name
    val tag = set.prompt.substringBefore(',').trim()
    return tag.ifEmpty { "Set ${index + 1}" }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ConfigDropdown(
    label: String,
    value: String,
    options: List<Pair<String, String>>,
    onChange: (String) -> Unit,
    error: String? = null,
    supporting: String? = null
) {
    var expanded by remember { mutableStateOf(false) }
    val selectedLabel = options.find { it.first == value }?.second ?: value
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it }
    ) {
        OutlinedTextField(
            value = selectedLabel,
            onValueChange = {},
            readOnly = true,
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor(ExposedDropdownMenuAnchorType.PrimaryNotEditable),
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            isError = error != null,
            supportingText = {
                val text = error ?: supporting
                if (text != null) Text(text)
            },
            shape = rankoTokens.panel,
            colors = rankoFieldColors(),
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            options.forEach { (id, optionLabel) ->
                DropdownMenuItem(
                    text = { Text(optionLabel) },
                    onClick = {
                        onChange(id)
                        expanded = false
                    }
                )
            }
        }
    }
}

@Composable
private fun ConfigTextField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    error: String? = null,
    supporting: String? = null,
    singleLine: Boolean = true,
    minLines: Int = 1,
    readOnly: Boolean = false,
    trailingIcon: @Composable (() -> Unit)? = null
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier.fillMaxWidth(),
        label = { Text(label) },
        isError = error != null,
        readOnly = readOnly,
        supportingText = {
            val text = error ?: supporting
            if (text != null) Text(text)
        },
        singleLine = singleLine,
        minLines = if (singleLine) 1 else minLines,
        trailingIcon = trailingIcon,
        shape = rankoTokens.panel,
        colors = rankoFieldColors(),
    )
}

@Composable
private fun ConfigPathField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    onBrowse: () -> Unit,
    error: String? = null,
    supporting: String? = null
) {
    ConfigTextField(
        label = label,
        value = value,
        onValueChange = onValueChange,
        error = error,
        supporting = supporting,
        trailingIcon = {
            IconButton(onClick = onBrowse) {
                Icon(Icons.Default.FolderOpen, contentDescription = "Browse")
            }
        }
    )
}

@Composable
private fun ConfigSwitch(
    label: String,
    checked: Boolean,
    onChecked: (Boolean) -> Unit,
    description: String? = null
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onChecked(!checked) }
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f).padding(end = 16.dp)) {
            Text(label, style = MaterialTheme.typography.bodyLarge)
            if (description != null) {
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.textDim
                )
            }
        }
        Switch(checked = checked, onCheckedChange = onChecked)
    }
}

@Composable
private fun ChoiceChips(
    label: String,
    value: String,
    options: List<String>,
    onChange: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, style = MaterialTheme.typography.labelLarge)
        Row(
            modifier = Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            options.forEach { option ->
                CapsuleChoice(
                    text = option,
                    selected = value == option,
                    onClick = { onChange(option) },
                )
            }
            if (value.isNotBlank() && value !in options) {
                CapsuleChoice(
                    text = value,
                    selected = true,
                    onClick = {},
                )
            }
        }
    }
}

private fun effectiveBatchHint(form: TrainingConfigForm): String? {
    val bs = form.trainBatchSize.toIntOrNull() ?: return null
    val ga = form.gradientAccumulationSteps.toIntOrNull() ?: return null
    return "Effective batch = ${bs * ga}"
}

private fun loraScaleHint(form: TrainingConfigForm): String? {
    val dim = form.networkDim.toIntOrNull() ?: return null
    val alpha = form.networkAlpha.toIntOrNull() ?: return null
    if (dim <= 0) return null
    return "α/dim = ${alpha.toDouble() / dim}"
}

private fun convScaleHint(form: TrainingConfigForm): String? {
    val dim = form.convDim.toIntOrNull() ?: return null
    val alpha = form.convAlpha.toIntOrNull() ?: return null
    if (dim <= 0) return null
    return "α/dim = ${alpha.toDouble() / dim}"
}

private fun bucketStepHint(form: TrainingConfigForm): String? {
    val reso = form.trainResolution.toIntOrNull() ?: return null
    val step = form.bucketResoSteps.toIntOrNull() ?: return null
    if (step <= 0) return null
    return if (reso % step == 0) "Divisible by bucket step"
    else "Not divisible by bucket step $step"
}

private fun sampleAspectHint(set: SampleSetForm): String? {
    val w = set.width.toIntOrNull() ?: return null
    val h = set.height.toIntOrNull() ?: return null
    if (w <= 0 || h <= 0) return null
    val g = gcd(w, h)
    return "${w / g}:${h / g}"
}

private tailrec fun gcd(a: Int, b: Int): Int = if (b == 0) kotlin.math.abs(a) else gcd(b, a % b)

private fun profileSubtitle(profile: ConfigProfile): String {
    val local = Instant.fromEpochMilliseconds(profile.modified)
        .toLocalDateTime(TimeZone.currentSystemDefault())
    val hour = local.hour.toString().padStart(2, '0')
    val minute = local.minute.toString().padStart(2, '0')
    val stamp = "${local.date} $hour:$minute"
    return "$stamp · ${formatBytes(profile.size)}"
}

/**
 * Named `config.toml` presets: the editor writes one into `configs/`, and applying one patches
 * `config.toml` in place and reloads the editor from the result.
 */
@Composable
private fun ProfilesFields(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel,
) {
    val colors = rankoColors
    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Save the current config as a profile",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "A profile is one TOML file in configs/, next to config.toml, and it holds " +
                    "every value in the editor above — save the ones you want it to carry. The " +
                    "folder is tracked by git, the .toml files in it are ignored.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                ConfigTextField(
                    label = "Profile name",
                    value = uiState.profileName,
                    onValueChange = viewModel::updateProfileName,
                    modifier = Modifier.weight(1f),
                )
                CapsuleButton(
                    text = "Save as profile",
                    onClick = { viewModel.requestSaveProfile() },
                    enabled = uiState.profileName.isNotBlank() &&
                        !uiState.isSaving &&
                        !uiState.isTagging,
                    compact = true,
                    emphasized = true,
                ) {
                    Icon(Icons.Default.Save, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Save as profile", fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }

    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Saved profiles",
                    style = MaterialTheme.typography.labelLarge,
                    color = colors.text,
                    modifier = Modifier.weight(1f),
                )
                CapsuleButton(
                    text = "Reload list",
                    onClick = { viewModel.refreshProfiles() },
                    enabled = !uiState.isLoadingProfiles,
                    compact = true,
                )
            }
            when {
                uiState.isLoadingProfiles -> Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text("Reading configs/…", style = MaterialTheme.typography.bodyMedium)
                }
                uiState.profiles.isEmpty() -> Text(
                    text = "No profiles yet. Name one above and press Save as profile.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.textDim,
                )
                else -> uiState.profiles.forEach { profile ->
                    ProfileRow(
                        profile = profile,
                        enabled = !uiState.isSaving,
                        onApply = { viewModel.requestApplyProfile(profile) },
                        onDelete = { viewModel.requestDeleteProfile(profile) },
                    )
                }
            }
            Text(
                text = "Click a profile to apply it: config.toml is patched in place, so comments and " +
                    "every key the profile does not carry stay as they are, and the editor reloads " +
                    "from the result. A run already in flight keeps the settings it started with.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
        }
    }

    uiState.pendingProfileOverwrite?.let { name ->
        AlertDialog(
            onDismissRequest = { viewModel.cancelProfileDialog() },
            title = { Text("Overwrite \"$name\"?") },
            text = {
                Text(
                    "configs/$name.toml is replaced with the values in the editor. The file it holds " +
                        "now is not kept anywhere."
                )
            },
            confirmButton = {
                TextButton(onClick = { viewModel.confirmProfileOverwrite() }) { Text("Overwrite") }
            },
            dismissButton = {
                TextButton(onClick = { viewModel.cancelProfileDialog() }) { Text("Cancel") }
            },
        )
    }
    uiState.pendingProfileApply?.let { profile ->
        AlertDialog(
            onDismissRequest = { viewModel.cancelProfileDialog() },
            title = { Text("Apply \"${profile.name}\"?") },
            text = {
                Text(
                    "The editor has unsaved changes. Applying writes the profile into config.toml " +
                        "and reloads the editor, so those changes are lost."
                )
            },
            confirmButton = {
                TextButton(onClick = { viewModel.confirmApplyProfile() }) { Text("Apply") }
            },
            dismissButton = {
                TextButton(onClick = { viewModel.cancelProfileDialog() }) { Text("Cancel") }
            },
        )
    }
    uiState.pendingProfileDelete?.let { profile ->
        AlertDialog(
            onDismissRequest = { viewModel.cancelProfileDialog() },
            title = { Text("Delete \"${profile.name}\"?") },
            text = { Text("Profile ${profile.name} is deleted. config.toml is not touched.") },
            confirmButton = {
                TextButton(onClick = { viewModel.confirmDeleteProfile() }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { viewModel.cancelProfileDialog() }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun ProfileRow(
    profile: ConfigProfile,
    enabled: Boolean,
    onApply: () -> Unit,
    onDelete: () -> Unit,
) {
    RankoChoiceRow(selected = false, onClick = onApply, enabled = enabled) {
        Column(verticalArrangement = Arrangement.spacedBy(2.dp), modifier = Modifier.weight(1f)) {
            Text(
                text = profile.name,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = profileSubtitle(profile),
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.textDim,
            )
            Text(
                text = profile.name,
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.textDim,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        Spacer(Modifier.width(8.dp))
        CapsuleButton(
            text = "Delete",
            onClick = onDelete,
            enabled = enabled,
            compact = true,
            danger = true,
        )
    }
}

@Composable
private fun AppearanceFields(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel,
) {
    val settings = uiState.appearance
    val colors = rankoColors
    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text(
                text = "Background",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "Solid is the flat purple night. Glow adds the three pink/blue/lilac orbs. Image fills the window with a photo (dimmed so cards stay readable).",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CapsuleChoice(
                    text = "Solid",
                    selected = settings.background == BackgroundStyle.Solid,
                    onClick = { viewModel.updateBackground(BackgroundStyle.Solid) },
                )
                CapsuleChoice(
                    text = "Glow",
                    selected = settings.background == BackgroundStyle.Glow,
                    onClick = { viewModel.updateBackground(BackgroundStyle.Glow) },
                )
                if (wallpaperImagesSupported) {
                    CapsuleChoice(
                        text = "Image",
                        selected = settings.background == BackgroundStyle.Image,
                        onClick = { viewModel.updateBackground(BackgroundStyle.Image) },
                    )
                }
            }
            if (wallpaperImagesSupported) {
                BackgroundImagePicker(settings = settings, viewModel = viewModel)
            }
        }
    }

    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Blur",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "Two independent radii, used when Glow or Image is on. Card blur frosts porcelain cards, the nav rail, and metric chips. Background blur frosts the wallpaper in the gaps between them.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            BlurSlider(
                label = "Card",
                value = settings.cardBlurRadiusDp,
                onChange = { viewModel.updateCardBlur(it) },
            )
            BlurSlider(
                label = "Background",
                value = settings.backgroundBlurRadiusDp,
                onChange = { viewModel.updateBackgroundBlur(it) },
            )
        }
    }

    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                text = "Font scale",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "Text size only (0.50×–2.50×). Independent of icon scale.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            Slider(
                value = settings.fontScale,
                onValueChange = { viewModel.updateFontScale(it) },
                valueRange = AppearanceSettings.MIN_FONT_SCALE..AppearanceSettings.MAX_FONT_SCALE,
            )
            Text(
                text = "${formatFixed(settings.fontScale, 2)} ×",
                style = MaterialTheme.typography.labelSmall,
                color = colors.accentPink,
            )
        }
    }

    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                text = "Icon scale",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "Icons, padding, and component sizes (0.50×–2.50×). Does not change text size.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            Slider(
                value = settings.iconScale,
                onValueChange = { viewModel.updateIconScale(it) },
                valueRange = AppearanceSettings.MIN_ICON_SCALE..AppearanceSettings.MAX_ICON_SCALE,
            )
            Text(
                text = "${formatFixed(settings.iconScale, 2)} ×",
                style = MaterialTheme.typography.labelSmall,
                color = colors.accentPink,
            )
        }
    }

    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                text = "Thumbnail quality",
                style = MaterialTheme.typography.labelLarge,
                color = colors.text,
            )
            Text(
                text = "JPEG quality for dataset and sample thumbnails sent over the helper (1–100). Lower is smaller and faster; the next load after a change uses the new quality.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            Slider(
                value = settings.thumbnailQuality.toFloat(),
                onValueChange = { viewModel.updateThumbnailQuality(it.roundToInt()) },
                valueRange = AppearanceSettings.MIN_THUMBNAIL_QUALITY.toFloat()..
                    AppearanceSettings.MAX_THUMBNAIL_QUALITY.toFloat(),
            )
            Text(
                text = "${settings.thumbnailQuality}",
                style = MaterialTheme.typography.labelSmall,
                color = colors.accentPink,
            )
        }
    }
}

@Composable
private fun BlurSlider(
    label: String,
    value: Float,
    onChange: (Float) -> Unit,
) {
    val colors = rankoColors
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(label, style = MaterialTheme.typography.bodyMedium, color = colors.text)
            Text(
                text = "${value.roundToInt()} dp",
                style = MaterialTheme.typography.labelSmall,
                color = colors.accentPink,
            )
        }
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = AppearanceSettings.MIN_BLUR..AppearanceSettings.MAX_BLUR,
        )
    }
}

@Composable
private fun BackgroundImagePicker(
    settings: AppearanceSettings,
    viewModel: UtilsScreenViewModel,
) {
    val colors = rankoColors
    val tokens = rankoTokens
    val path = settings.backgroundImagePath
    val file = remember(path) { localWallpaperModel(path) }
    val missing = path.isNotEmpty() && file == null
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(tokens.panel)
                .background(colors.bgCard.copy(alpha = 0.42f)),
            contentAlignment = Alignment.Center,
        ) {
            if (file is androidx.compose.ui.graphics.ImageBitmap) {
                androidx.compose.foundation.Image(
                    bitmap = file,
                    contentDescription = "Background preview",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            } else if (file != null) {
                AsyncImage(
                    model = file,
                    contentDescription = "Background preview",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Icon(
                    Icons.Default.Photo,
                    contentDescription = null,
                    tint = colors.textDim,
                )
            }
        }
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(
                text = when {
                    file != null && path.startsWith("data:") -> "Local image"
                    file != null -> path.substringAfterLast('/')
                    missing -> "Image not found"
                    else -> "No image selected"
                },
                style = MaterialTheme.typography.bodyMedium,
                color = if (missing) colors.qualityRed else colors.text,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = if (path.startsWith("data:")) "jpg / png / webp / bmp" else path.ifBlank { "jpg / png / webp / bmp" },
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        CapsuleButton(
            text = "Browse",
            onClick = { viewModel.browseBackgroundImage() },
            compact = true,
        )
        if (path.isNotEmpty()) {
            CapsuleButton(
                text = "Clear",
                onClick = { viewModel.clearBackgroundImage() },
                compact = true,
            )
        }
    }
}

package com.acite.axlranko.pages

import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.background
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
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.GridOn
import androidx.compose.material.icons.filled.Hub
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Photo
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.UtilsUiState
import dev.zacsweers.metrox.viewmodel.metroViewModel
import java.awt.Cursor

@Composable
public fun UtilsScreen(
    viewModel: UtilsScreenViewModel = metroViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    if (uiState.errorMessage != null && !uiState.isLoading && uiState.configPath.isEmpty()) {
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
                    Button(onClick = { viewModel.loadConfig() }) {
                        Text("Retry")
                    }
                }
            }
        }
        return
    }

    if (uiState.isLoading) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        return
    }

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        val totalWidthPx = constraints.maxWidth.toFloat()

        Row(modifier = Modifier.fillMaxSize()) {
            Box(modifier = Modifier.fillMaxHeight().weight(uiState.leftWeight)) {
                SectionNav(
                    uiState = uiState,
                    onSelect = viewModel::selectSection
                )
            }

            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(8.dp)
                    .pointerHoverIcon(PointerIcon(Cursor(Cursor.E_RESIZE_CURSOR)))
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
                VerticalDivider(thickness = 1.dp, color = MaterialTheme.colorScheme.outlineVariant)
            }

            Column(
                modifier = Modifier
                    .fillMaxHeight()
                    .weight(1f - uiState.leftWeight)
            ) {
                ConfigHeader(uiState = uiState, viewModel = viewModel)
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
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
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        SectionFields(uiState = uiState, viewModel = viewModel)
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

        if (uiState.isSaving) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().align(Alignment.TopCenter)
            )
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
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = uiState.summaryLine,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    onClick = { viewModel.loadConfig() },
                    enabled = !uiState.isDirty && !uiState.isSaving
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Reload")
                }
                OutlinedButton(
                    onClick = { viewModel.resetForm() },
                    enabled = uiState.isDirty && !uiState.isSaving
                ) {
                    Icon(Icons.Default.Restore, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Reset")
                }
                Button(
                    onClick = { viewModel.saveConfig() },
                    enabled = uiState.isDirty && !uiState.isSaving
                ) {
                    Icon(Icons.Default.Save, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Save")
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
    val container =
        if (isError) MaterialTheme.colorScheme.errorContainer
        else MaterialTheme.colorScheme.secondaryContainer
    val content =
        if (isError) MaterialTheme.colorScheme.onErrorContainer
        else MaterialTheme.colorScheme.onSecondaryContainer
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(container, RoundedCornerShape(8.dp))
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

@Composable
private fun SectionNav(
    uiState: UtilsUiState,
    onSelect: (ConfigSection) -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        items(ConfigSection.entries.toList(), key = { it.name }) { section ->
            val selected = uiState.selectedSection == section
            val hasError = section.fieldKeys.any { it in uiState.fieldErrors }
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onSelect(section) },
                shape = RoundedCornerShape(8.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer
                    else MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Icon(
                        imageVector = section.icon(),
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                        tint = when {
                            hasError -> MaterialTheme.colorScheme.error
                            selected -> MaterialTheme.colorScheme.primary
                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                    Text(
                        text = section.title,
                        modifier = Modifier.weight(1f),
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                        color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    if (hasError) {
                        Icon(
                            Icons.Default.Warning,
                            contentDescription = "Invalid fields",
                            modifier = Modifier.size(16.dp),
                            tint = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }
        }
    }
}

private fun ConfigSection.icon(): ImageVector = when (this) {
    ConfigSection.Environment -> Icons.Default.Folder
    ConfigSection.ModelSpec -> Icons.Default.Info
    ConfigSection.Training -> Icons.Default.Tune
    ConfigSection.Network -> Icons.Default.Hub
    ConfigSection.Bucketing -> Icons.Default.GridOn
    ConfigSection.Optimization -> Icons.Default.Bolt
    ConfigSection.UnetOptimizer -> Icons.Default.Layers
    ConfigSection.TeOptimizer -> Icons.Default.TextFields
    ConfigSection.Infrastructure -> Icons.Default.Settings
    ConfigSection.Validation -> Icons.Default.Photo
}

@Composable
private fun SectionFields(
    uiState: UtilsUiState,
    viewModel: UtilsScreenViewModel
) {
    val form = uiState.form
    val errors = uiState.fieldErrors
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        when (uiState.selectedSection) {
            ConfigSection.Environment -> EnvironmentFields(form, errors, viewModel)
            ConfigSection.ModelSpec -> ModelSpecFields(form, errors, viewModel)
            ConfigSection.Training -> TrainingFields(form, errors, viewModel)
            ConfigSection.Network -> NetworkFields(form, errors, viewModel)
            ConfigSection.Bucketing -> BucketingFields(form, errors, viewModel)
            ConfigSection.Optimization -> OptimizationFields(form, errors, viewModel)
            ConfigSection.UnetOptimizer -> UnetFields(form, errors, viewModel)
            ConfigSection.TeOptimizer -> TeFields(form, errors, viewModel)
            ConfigSection.Infrastructure -> InfrastructureFields(form, errors, viewModel)
            ConfigSection.Validation -> ValidationFields(form, errors, viewModel)
        }
    }
}

@Composable
private fun EnvironmentFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    ConfigPathField(
        label = "Pretrained model path",
        value = form.pretrainedModelNameOrPath,
        error = errors["pretrained_model_name_or_path"],
        supporting = "Diffusers directory or checkpoint used as the SDXL base",
        onValueChange = { viewModel.updateForm { copy(pretrainedModelNameOrPath = it) } },
        onBrowse = {
            viewModel.browseDirectory(form.pretrainedModelNameOrPath) {
                copy(pretrainedModelNameOrPath = it)
            }
        }
    )
    ConfigPathField(
        label = "Train data directory",
        value = form.trainDataDir,
        error = errors["train_data_dir"],
        supporting = "Image/tag dataset folder used by the Images and Statistics pages",
        onValueChange = { viewModel.updateForm { copy(trainDataDir = it) } },
        onBrowse = { viewModel.browseDirectory(form.trainDataDir) { copy(trainDataDir = it) } }
    )
    ConfigTextField(
        label = "Output name",
        value = form.outputName,
        error = errors["output_name"],
        supporting = "LoRA filename stem written under the output directory",
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
private fun ModelSpecFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    ConfigTextField(
        label = "Base model version",
        value = form.baseModelVersion,
        error = errors["base_model_version"],
        onValueChange = { viewModel.updateForm { copy(baseModelVersion = it) } }
    )
    ConfigTextField(
        label = "Architecture",
        value = form.modelspecArchitecture,
        error = errors["modelspec_architecture"],
        onValueChange = { viewModel.updateForm { copy(modelspecArchitecture = it) } }
    )
    ConfigTextField(
        label = "Implementation URL",
        value = form.modelspecImplementation,
        error = errors["modelspec_implementation"],
        onValueChange = { viewModel.updateForm { copy(modelspecImplementation = it) } }
    )
    ConfigTextField(
        label = "SAI model spec",
        value = form.modelspecSaiModelSpec,
        error = errors["modelspec_sai_model_spec"],
        onValueChange = { viewModel.updateForm { copy(modelspecSaiModelSpec = it) } }
    )
}

@Composable
private fun TrainingFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    ConfigSwitch(
        label = "v-prediction",
        checked = form.isVpred,
        description = "Enable v-pred loss (leave off for standard SDXL epsilon)",
        onChecked = { viewModel.updateForm { copy(isVpred = it) } }
    )
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
    Text("Mixed precision", style = MaterialTheme.typography.labelLarge)
    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
        TrainingConfigForm.mixedPrecisionOptions.forEachIndexed { index, option ->
            SegmentedButton(
                selected = form.mixedPrecision == option,
                onClick = { viewModel.updateForm { copy(mixedPrecision = option) } },
                shape = SegmentedButtonDefaults.itemShape(
                    index = index,
                    count = TrainingConfigForm.mixedPrecisionOptions.size
                )
            ) { Text(option) }
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
            supporting = "5.0 is a common SDXL starting point",
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
            onValueChange = { viewModel.updateForm { copy(saveEveryNSteps = it) } },
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun NetworkFields(
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
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
        label = "Shuffle caption",
        checked = form.shuffleCaption,
        description = "Shuffle comma-separated tags each step; keep_tokens stay fixed",
        onChecked = { viewModel.updateForm { copy(shuffleCaption = it) } }
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
    form: TrainingConfigForm,
    errors: Map<String, String>,
    viewModel: UtilsScreenViewModel
) {
    ConfigTextField(
        label = "Sample prompt",
        value = form.samplePrompts,
        error = errors["sample_prompts"],
        onValueChange = { viewModel.updateForm { copy(samplePrompts = it) } },
        singleLine = false,
        minLines = 4
    )
    ConfigTextField(
        label = "Negative prompt",
        value = form.sampleNegative,
        error = errors["sample_negative"],
        onValueChange = { viewModel.updateForm { copy(sampleNegative = it) } },
        singleLine = false,
        minLines = 3
    )
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Width",
            value = form.sampleWidth,
            error = errors["sample_width"],
            supporting = sampleAspectHint(form),
            onValueChange = { viewModel.updateForm { copy(sampleWidth = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Height",
            value = form.sampleHeight,
            error = errors["sample_height"],
            onValueChange = { viewModel.updateForm { copy(sampleHeight = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Steps",
            value = form.sampleSteps,
            error = errors["sample_steps"],
            onValueChange = { viewModel.updateForm { copy(sampleSteps = it) } },
            modifier = Modifier.weight(1f)
        )
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        ConfigTextField(
            label = "Guidance scale",
            value = form.guidanceScale,
            error = errors["guidance_scale"],
            onValueChange = { viewModel.updateForm { copy(guidanceScale = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Sample seed",
            value = form.sampleSeed,
            error = errors["sample_seed"],
            supporting = "0 typically means a random seed",
            onValueChange = { viewModel.updateForm { copy(sampleSeed = it) } },
            modifier = Modifier.weight(1f)
        )
        ConfigTextField(
            label = "Repeat",
            value = form.sampleRepeat,
            error = errors["sample_repeat"],
            onValueChange = { viewModel.updateForm { copy(sampleRepeat = it) } },
            modifier = Modifier.weight(1f)
        )
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
    trailingIcon: @Composable (() -> Unit)? = null
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier.fillMaxWidth(),
        label = { Text(label) },
        isError = error != null,
        supportingText = {
            val text = error ?: supporting
            if (text != null) Text(text)
        },
        singleLine = singleLine,
        minLines = if (singleLine) 1 else minLines,
        trailingIcon = trailingIcon
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
                    color = MaterialTheme.colorScheme.onSurfaceVariant
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
                FilterChip(
                    selected = value == option,
                    onClick = { onChange(option) },
                    label = { Text(option) }
                )
            }
            if (value.isNotBlank() && value !in options) {
                FilterChip(
                    selected = true,
                    onClick = {},
                    label = { Text(value) }
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

private fun bucketStepHint(form: TrainingConfigForm): String? {
    val reso = form.trainResolution.toIntOrNull() ?: return null
    val step = form.bucketResoSteps.toIntOrNull() ?: return null
    if (step <= 0) return null
    return if (reso % step == 0) "Divisible by bucket step"
    else "Not divisible by bucket step $step"
}

private fun sampleAspectHint(form: TrainingConfigForm): String? {
    val w = form.sampleWidth.toIntOrNull() ?: return null
    val h = form.sampleHeight.toIntOrNull() ?: return null
    if (w <= 0 || h <= 0) return null
    val g = gcd(w, h)
    return "${w / g}:${h / g}"
}

private tailrec fun gcd(a: Int, b: Int): Int = if (b == 0) kotlin.math.abs(a) else gcd(b, a % b)

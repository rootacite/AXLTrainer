package com.acite.axlranko.model

import com.acite.axlranko.data.ConfigProfile

enum class ConfigSection(
    val title: String,
    val description: String,
    val fieldKeys: Set<String>,
    /** Prefixes of the per-entry errors this section renders (`samples.1.steps`, `train_data.0.repeat`). */
    val errorPrefixes: Set<String> = emptySet()
) {
    Helper(
        title = "Helper",
        description = "Dashboard helper WebSocket host and port",
        fieldKeys = emptySet(),
    ),
    Environment(
        title = "Environment",
        description = "Base model, dataset folders, and output paths",
        fieldKeys = setOf(
            "pretrained_model_name_or_path",
            "output_dir",
            "logging_dir",
            "train_data_dir",
            "train_data",
            "output_name",
        ),
        errorPrefixes = setOf(TRAIN_DATA_ERROR_PREFIX)
    ),
    Rocm(
        title = "ROCm",
        description = "RDNA 4 allocation patch: none, tail guard, or VMM",
        fieldKeys = setOf("amdfq", "amdfq_vram_reserve_gib", "amdfq_va_never_reuse", "amdfq_pool_mib"),
    ),
    ModelSpec(
        title = "Model Spec",
        description = "Base-model family and SAI checkpoint metadata",
        fieldKeys = setOf(
            "base_model_version",
            "modelspec_architecture",
            "modelspec_implementation",
            "modelspec_sai_model_spec"
        )
    ),
    Training(
        title = "Training",
        description = "Epochs, batch size, scheduler, and precision",
        fieldKeys = setOf(
            "is_vpred",
            "min_snr_gamma",
            "seed",
            "mixed_precision",
            "train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "lr_scheduler",
            "lr_warmup_steps",
            "max_grad_norm",
            "epoch",
            "save_every_n_epochs",
            "save_every_n_steps",
            "resume_lora_path"
        )
    ),
    Network(
        title = "Network",
        description = "LoRA type, rank, alpha, dropout, and token length",
        fieldKeys = setOf(
            "network_type",
            "network_dim",
            "network_alpha",
            "network_dropout",
            "conv_dim",
            "conv_alpha",
            "clip_skip",
            "max_token_length"
        )
    ),
    Bucketing(
        title = "Bucketing",
        description = "Aspect-ratio buckets and training resolution",
        fieldKeys = setOf(
            "enable_bucket",
            "bucket_no_upscale",
            "train_resolution",
            "bucket_reso_steps",
            "min_bucket_reso",
            "max_bucket_reso"
        )
    ),
    Optimization(
        title = "Optimization",
        description = "Latent cache, gradient checkpointing, caption shuffle, noise offset, and step-end GPU flush",
        fieldKeys = setOf(
            "cache_latents",
            "cache_latents_to_disk",
            "gradient_checkpointing_unet",
            "gradient_checkpointing_te",
            "shuffle_caption",
            "keep_tokens",
            "caption_extension",
            "noise_offset",
            "flush_memory_every_step"
        )
    ),
    UnetOptimizer(
        title = "UNet Optimizer",
        description = "AdamW hyperparameters for the UNet",
        fieldKeys = setOf(
            "unet_learning_rate",
            "unet_weight_decay",
            "unet_betas_1",
            "unet_betas_2",
            "unet_eps",
            "unet_warmup_steps"
        )
    ),
    TeOptimizer(
        title = "Text Encoder",
        description = "AdamW hyperparameters for the text encoder",
        fieldKeys = setOf(
            "te_learning_rate",
            "te_weight_decay",
            "te_betas_1",
            "te_betas_2",
            "te_max_grad_norm"
        )
    ),
    Infrastructure(
        title = "Infrastructure",
        description = "DataLoader workers",
        fieldKeys = setOf(
            "max_data_loader_n_workers",
            "persistent_workers"
        )
    ),
    Validation(
        title = "Validation",
        description = "Sample prompt sets and preview generation",
        fieldKeys = setOf(
            "sample_prompts",
            "sample_negative",
            "sample_width",
            "sample_height",
            "sample_steps",
            "sample_seed",
            "sample_repeat",
            "guidance_scale"
        ),
        errorPrefixes = setOf(SAMPLE_SET_ERROR_PREFIX)
    ),
    Appearance(
        title = "Appearance",
        description = "Background style, blur strength, and font/icon scale",
        fieldKeys = emptySet(),
    ),
    Wm(
        title = "WM",
        description = "Maximize the window and quit, for a session that draws no decorations",
        fieldKeys = emptySet(),
    ),
    Profiles(
        title = "Profiles",
        description = "Named config.toml presets in configs/ — applying one patches config.toml in place",
        fieldKeys = emptySet(),
    ),
    ;

    /**
     * Whether a `fieldErrors` key belongs to this section. Entry errors
     * (`samples.2.steps`, `train_data.0.repeat`) are spelled with the entry index, so they need
     * a prefix match.
     */
    fun owns(key: String): Boolean =
        key in fieldKeys || errorPrefixes.any { key.startsWith(it) }
}

data class UtilsUiState(
    val isLoading: Boolean = true,
    val isSaving: Boolean = false,
    val isTagging: Boolean = false,
    val tagThreshold: String = "0.35",
    val errorMessage: String? = null,
    val statusMessage: String? = null,
    val configPath: String = "",
    val selectedSection: ConfigSection = ConfigSection.Environment,
    /** Open tab in the Validation section's `[[validation.samples]]` editor. */
    val selectedSampleSet: Int = 0,
    /** Which dataset folder the single-folder pages (Images, Statistics, Tag) act on. */
    val datasetDirIndex: Int = 0,
    val form: TrainingConfigForm = TrainingConfigForm(),
    val savedForm: TrainingConfigForm = TrainingConfigForm(),
    val fieldErrors: Map<String, String> = emptyMap(),
    val leftWeight: Float = 0.22f,
    val checkpointPickerOpen: Boolean = false,
    val isLoadingCheckpoints: Boolean = false,
    val checkpoints: List<CheckpointItem> = emptyList(),
    val checkpointError: String? = null,
    val appearance: AppearanceSettings = AppearanceSettings(),
    val helperHost: String = "127.0.0.1",
    val helperPort: String = "18765",
    val helperStatus: String = "disconnected",
    val helperError: String? = null,
    val helperBusy: Boolean = false,
    /** Saved `config.toml` presets under `configs/`, and the dialogs the Profiles section can raise. */
    val profiles: List<ConfigProfile> = emptyList(),
    val isLoadingProfiles: Boolean = false,
    val profileName: String = "",
    val pendingProfileOverwrite: String? = null,
    val pendingProfileApply: ConfigProfile? = null,
    val pendingProfileDelete: ConfigProfile? = null,
) {
    val isDirty: Boolean get() = form != savedForm

    val summaryLine: String
        get() {
            val name = form.outputName.ifBlank { "unnamed" }
            val reso = form.trainResolution.ifBlank { "?" }
            val ep = form.epoch.ifBlank { "?" }
            val bs = form.trainBatchSize.toIntOrNull()
            val ga = form.gradientAccumulationSteps.toIntOrNull()
            val batch = if (bs != null && ga != null) "${bs}×$ga" else "?"
            val resume = if (form.resumeLoraPath.isBlank()) "" else " · resume"
            return "$name · ${reso}px · $ep epochs · batch $batch$resume"
        }
}

package com.acite.axlranko.model

enum class ConfigSection(
    val title: String,
    val description: String,
    val fieldKeys: Set<String>
) {
    Environment(
        title = "Environment",
        description = "Base model, dataset, and output paths",
        fieldKeys = setOf(
            "pretrained_model_name_or_path",
            "output_dir",
            "logging_dir",
            "train_data_dir",
            "output_name"
        )
    ),
    ModelSpec(
        title = "Model Spec",
        description = "SAI model specification metadata",
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
            "save_every_n_steps"
        )
    ),
    Network(
        title = "Network",
        description = "LoRA rank, alpha, dropout, and token length",
        fieldKeys = setOf(
            "network_dim",
            "network_alpha",
            "network_dropout",
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
        description = "Latent cache, caption shuffle, and noise offset",
        fieldKeys = setOf(
            "cache_latents",
            "cache_latents_to_disk",
            "shuffle_caption",
            "keep_tokens",
            "caption_extension",
            "noise_offset"
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
        description = "Sample prompts and preview generation",
        fieldKeys = setOf(
            "sample_prompts",
            "sample_negative",
            "sample_width",
            "sample_height",
            "sample_steps",
            "sample_seed",
            "sample_repeat",
            "guidance_scale"
        )
    )
}

data class UtilsUiState(
    val isLoading: Boolean = true,
    val isSaving: Boolean = false,
    val errorMessage: String? = null,
    val statusMessage: String? = null,
    val configPath: String = "",
    val selectedSection: ConfigSection = ConfigSection.Environment,
    val form: TrainingConfigForm = TrainingConfigForm(),
    val savedForm: TrainingConfigForm = TrainingConfigForm(),
    val fieldErrors: Map<String, String> = emptyMap(),
    val leftWeight: Float = 0.22f
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
            return "$name · ${reso}px · $ep epochs · batch $batch"
        }
}

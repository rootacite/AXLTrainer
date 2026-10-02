package com.acite.axlranko.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class AxlTrainerConfig(
    val environment: EnvironmentConfig,
    @SerialName("model_spec") val modelSpec: ModelSpecConfig,
    val training: TrainingConfig,
    val network: NetworkConfig,
    val bucketing: BucketingConfig,
    val optimization: OptimizationConfig,
    @SerialName("unet_optimizer") val unetOptimizer: UnetOptimizerConfig,
    @SerialName("te_optimizer") val teOptimizer: TeOptimizerConfig,
    val infrastructure: InfrastructureConfig,
    val validation: ValidationConfig
)

@Serializable
data class EnvironmentConfig(
    @SerialName("pretrained_model_name_or_path") val pretrainedModelNameOrPath: String,
    @SerialName("train_data_dir") val trainDataDir: String,
    @SerialName("train_data") val trainData: List<TrainDataEntryConfig> = emptyList(),
    @SerialName("output_name") val outputName: String,
    @SerialName("output_dir") val outputDir: String,
    @SerialName("logging_dir") val loggingDir: String,
    val amdfq: String = "none",
    @SerialName("amdfq_vram_reserve_gib") val amdfqVramReserveGib: Double = 0.0,
    @SerialName("amdfq_va_never_reuse") val amdfqVaNeverReuse: Boolean = false,
    @SerialName("amdfq_pool_mib") val amdfqPoolMib: Int = 64,
)

/**
 * One `[[environment.train_data]]` entry: a dataset folder and how often its images are drawn
 * inside one epoch. `train_data_dir` mirrors the first entry's path.
 */
@Serializable
data class TrainDataEntryConfig(
    val path: String,
    val repeat: Int = 1,
)

/**
 * The dataset folders the trainer trains on: the `[[environment.train_data]]` blocks, or - for a
 * config written before they existed - the single folder `train_data_dir` names, drawn once.
 */
fun EnvironmentConfig.trainDataEntries(): List<TrainDataEntryConfig> {
    val blocks = trainData.filter { it.path.isNotBlank() }
    if (blocks.isNotEmpty()) return blocks
    return listOf(TrainDataEntryConfig(path = trainDataDir, repeat = 1))
}

@Serializable
data class ModelSpecConfig(
    @SerialName("base_model_version") val baseModelVersion: String,
    @SerialName("modelspec_architecture") val modelspecArchitecture: String,
    @SerialName("modelspec_implementation") val modelspecImplementation: String,
    @SerialName("modelspec_sai_model_spec") val modelspecSaiModelSpec: String
)

@Serializable
data class TrainingConfig(
    @SerialName("min_snr_gamma") val minSnrGamma: Double,
    val seed: Long, // 考虑到你的 seed 包含了 1145141919 等较大数值，使用 Long 更安全
    @SerialName("mixed_precision") val mixedPrecision: String,
    @SerialName("train_batch_size") val trainBatchSize: Int,
    @SerialName("gradient_accumulation_steps") val gradientAccumulationSteps: Int,
    @SerialName("learning_rate") val learningRate: Double,
    /**
     * The text encoder's warmup used to live here; it is now `[te_optimizer].te_warmup_steps`.
     * Kept as a read-only fallback for a config that was written before the move.
     */
    @SerialName("lr_warmup_steps") val lrWarmupSteps: Int? = null,
    /**
     * The UNet's gradient-clipping threshold used to live here; it is now
     * `[unet_optimizer].unet_max_grad_norm`. Kept as a read-only fallback for a config that was
     * written before the move.
     */
    @SerialName("max_grad_norm") val maxGradNorm: Double? = null,
    val epoch: Int,
    @SerialName("save_every_n_epochs") val saveEveryNEpochs: Int,
    @SerialName("save_every_n_steps") val saveEveryNSteps: Int,
    /** Render the validation samples at every checkpoint. Optional: older files have no such key. */
    @SerialName("sampling_enabled") val samplingEnabled: Boolean = true,
    @SerialName("resume_lora_path") val resumeLoraPath: String = ""
)

@Serializable
data class NetworkConfig(
    @SerialName("network_type") val networkType: String = "standard",
    @SerialName("network_dim") val networkDim: Int,
    @SerialName("network_alpha") val networkAlpha: Int,
    @SerialName("network_dropout") val networkDropout: Double,
    @SerialName("conv_dim") val convDim: Int = 0,
    @SerialName("conv_alpha") val convAlpha: Int = 0,
    @SerialName("clip_skip") val clipSkip: Int,
    @SerialName("max_token_length") val maxTokenLength: Int
)

@Serializable
data class BucketingConfig(
    @SerialName("enable_bucket") val enableBucket: Boolean,
    @SerialName("bucket_no_upscale") val bucketNoUpscale: Boolean,
    @SerialName("train_resolution") val trainResolution: Int,
    @SerialName("bucket_reso_steps") val bucketResoSteps: Int,
    @SerialName("min_bucket_reso") val minBucketReso: Int,
    @SerialName("max_bucket_reso") val maxBucketReso: Int
)

@Serializable
data class OptimizationConfig(
    @SerialName("cache_latents") val cacheLatents: Boolean,
    @SerialName("cache_latents_to_disk") val cacheLatentsToDisk: Boolean,
    @SerialName("gradient_checkpointing_unet") val gradientCheckpointingUnet: Boolean = true,
    @SerialName("gradient_checkpointing_te") val gradientCheckpointingTe: Boolean = true,
    @SerialName("shuffle_caption") val shuffleCaption: Boolean,
    @SerialName("keep_tokens") val keepTokens: Int,
    @SerialName("caption_extension") val captionExtension: String,
    @SerialName("noise_offset") val noiseOffset: Double,
    @SerialName("flush_memory_every_step") val flushMemoryEveryStep: Boolean = true
)

@Serializable
data class UnetOptimizerConfig(
    @SerialName("unet_learning_rate") val unetLearningRate: Double,
    @SerialName("unet_weight_decay") val unetWeightDecay: Double,
    @SerialName("unet_betas_1") val unetBetas1: Double,
    @SerialName("unet_betas_2") val unetBetas2: Double,
    @SerialName("unet_warmup_steps") val unetWarmupSteps: Int,
    @SerialName("unet_max_grad_norm") val unetMaxGradNorm: Double? = null
)

/** The trainer's own default when a config carries neither the key nor its old home. */
const val DEFAULT_UNET_MAX_GRAD_NORM = 1.0

/**
 * Gradient-clipping threshold of the UNet's LoRA parameters: `[unet_optimizer].unet_max_grad_norm`,
 * the `[training].max_grad_norm` it replaced, or the trainer's default. The trainer resolves the two
 * keys the same way, so the Utils form shows the value the run will actually use.
 */
fun AxlTrainerConfig.effectiveUnetMaxGradNorm(): Double =
    unetOptimizer.unetMaxGradNorm ?: training.maxGradNorm ?: DEFAULT_UNET_MAX_GRAD_NORM

@Serializable
data class TeOptimizerConfig(
    @SerialName("te_learning_rate") val teLearningRate: Double,
    @SerialName("te_weight_decay") val teWeightDecay: Double,
    @SerialName("te_betas_1") val teBetas1: Double,
    @SerialName("te_betas_2") val teBetas2: Double,
    @SerialName("te_max_grad_norm") val teMaxGradNorm: Double,
    @SerialName("te_warmup_steps") val teWarmupSteps: Int? = null
)

/** The trainer's own default when a config carries neither the key nor its old home. */
const val DEFAULT_TE_WARMUP_STEPS = 100

/**
 * Schedule-Free warmup of the text-encoder optimizer: `[te_optimizer].te_warmup_steps`, the
 * `[training].lr_warmup_steps` it replaced, or the trainer's default. The trainer resolves the two
 * keys the same way, so the Utils form shows the value the run will actually use.
 */
fun AxlTrainerConfig.effectiveTeWarmupSteps(): Int =
    teOptimizer.teWarmupSteps ?: training.lrWarmupSteps ?: DEFAULT_TE_WARMUP_STEPS

@Serializable
data class InfrastructureConfig(
    @SerialName("max_data_loader_n_workers") val maxDataLoaderNWorkers: Int,
    @SerialName("persistent_workers") val persistentWorkers: Boolean
)

@Serializable
data class ValidationConfig(
    @SerialName("sample_prompts") val samplePrompts: String,
    @SerialName("sample_negative") val sampleNegative: String,
    @SerialName("sample_width") val sampleWidth: Int,
    @SerialName("sample_height") val sampleHeight: Int,
    @SerialName("sample_steps") val sampleSteps: Int,
    @SerialName("sample_seed") val sampleSeed: Long,
    @SerialName("sample_repeat") val sampleRepeat: Int,
    @SerialName("guidance_scale") val guidanceScale: Double,
    @SerialName("guidance_rescale") val guidanceRescale: Double = 0.0,
    val samples: List<SampleSetConfig> = emptyList()
)

/**
 * One `[[validation.samples]]` entry. `prompt` is the only required key: any other key a set
 * omits falls back to the flat `[validation]` scalar of the same shape (see `resolve_sample_sets`).
 */
@Serializable
data class SampleSetConfig(
    val name: String = "",
    val prompt: String,
    val negative: String? = null,
    val width: Int? = null,
    val height: Int? = null,
    val steps: Int? = null,
    @SerialName("guidance_scale") val guidanceScale: Double? = null,
    @SerialName("guidance_rescale") val guidanceRescale: Double? = null,
    val seed: Long? = null,
    val repeat: Int? = null
)
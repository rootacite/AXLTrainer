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
    @SerialName("output_name") val outputName: String,
    @SerialName("output_dir") val outputDir: String,
    @SerialName("logging_dir") val loggingDir: String
)

@Serializable
data class ModelSpecConfig(
    @SerialName("base_model_version") val baseModelVersion: String,
    @SerialName("modelspec_architecture") val modelspecArchitecture: String,
    @SerialName("modelspec_implementation") val modelspecImplementation: String,
    @SerialName("modelspec_sai_model_spec") val modelspecSaiModelSpec: String
)

@Serializable
data class TrainingConfig(
    @SerialName("is_vpred") val isVpred: Boolean,
    @SerialName("min_snr_gamma") val minSnrGamma: Double,
    val seed: Long, // 考虑到你的 seed 包含了 1145141919 等较大数值，使用 Long 更安全
    @SerialName("mixed_precision") val mixedPrecision: String,
    @SerialName("train_batch_size") val trainBatchSize: Int,
    @SerialName("gradient_accumulation_steps") val gradientAccumulationSteps: Int,
    @SerialName("learning_rate") val learningRate: Double,
    @SerialName("lr_scheduler") val lrScheduler: String,
    @SerialName("lr_warmup_steps") val lrWarmupSteps: Int,
    @SerialName("max_grad_norm") val maxGradNorm: Double,
    val epoch: Int,
    @SerialName("save_every_n_epochs") val saveEveryNEpochs: Int,
    @SerialName("save_every_n_steps") val saveEveryNSteps: Int
)

@Serializable
data class NetworkConfig(
    @SerialName("network_dim") val networkDim: Int,
    @SerialName("network_alpha") val networkAlpha: Int,
    @SerialName("network_dropout") val networkDropout: Double,
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
    @SerialName("shuffle_caption") val shuffleCaption: Boolean,
    @SerialName("keep_tokens") val keepTokens: Int,
    @SerialName("caption_extension") val captionExtension: String,
    @SerialName("noise_offset") val noiseOffset: Double
)

@Serializable
data class UnetOptimizerConfig(
    @SerialName("unet_learning_rate") val unetLearningRate: Double,
    @SerialName("unet_weight_decay") val unetWeightDecay: Double,
    @SerialName("unet_betas_1") val unetBetas1: Double,
    @SerialName("unet_betas_2") val unetBetas2: Double,
    @SerialName("unet_eps") val unetEps: Double,
    @SerialName("unet_warmup_steps") val unetWarmupSteps: Int
)

@Serializable
data class TeOptimizerConfig(
    @SerialName("te_learning_rate") val teLearningRate: Double,
    @SerialName("te_weight_decay") val teWeightDecay: Double,
    @SerialName("te_betas_1") val teBetas1: Double,
    @SerialName("te_betas_2") val teBetas2: Double,
    @SerialName("te_max_grad_norm") val teMaxGradNorm: Double
)

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
    @SerialName("guidance_scale") val guidanceScale: Double
)
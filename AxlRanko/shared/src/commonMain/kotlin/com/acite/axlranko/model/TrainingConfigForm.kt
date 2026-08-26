package com.acite.axlranko.model

import com.acite.axlranko.data.AxlTrainerConfig
import com.acite.axlranko.data.TomlDocumentPatcher

data class TrainingConfigForm(
    val pretrainedModelNameOrPath: String = "",
    val outputDir: String = "",
    val loggingDir: String = "",
    val trainDataDir: String = "",
    val outputName: String = "",

    val baseModelVersion: String = "",
    val modelspecArchitecture: String = "",
    val modelspecImplementation: String = "",
    val modelspecSaiModelSpec: String = "",

    val isVpred: Boolean = false,
    val minSnrGamma: String = "",
    val seed: String = "",
    val mixedPrecision: String = "bf16",
    val trainBatchSize: String = "",
    val gradientAccumulationSteps: String = "",
    val learningRate: String = "",
    val lrScheduler: String = "cosine",
    val lrWarmupSteps: String = "",
    val maxGradNorm: String = "",
    val epoch: String = "",
    val saveEveryNEpochs: String = "",
    val saveEveryNSteps: String = "",

    val networkDim: String = "",
    val networkAlpha: String = "",
    val networkDropout: String = "",
    val clipSkip: String = "",
    val maxTokenLength: String = "",

    val enableBucket: Boolean = true,
    val bucketNoUpscale: Boolean = true,
    val trainResolution: String = "",
    val bucketResoSteps: String = "",
    val minBucketReso: String = "",
    val maxBucketReso: String = "",

    val cacheLatents: Boolean = true,
    val cacheLatentsToDisk: Boolean = true,
    val shuffleCaption: Boolean = true,
    val keepTokens: String = "",
    val captionExtension: String = ".txt",
    val noiseOffset: String = "",

    val unetLearningRate: String = "",
    val unetWeightDecay: String = "",
    val unetBetas1: String = "",
    val unetBetas2: String = "",
    val unetEps: String = "",
    val unetWarmupSteps: String = "",

    val teLearningRate: String = "",
    val teWeightDecay: String = "",
    val teBetas1: String = "",
    val teBetas2: String = "",
    val teMaxGradNorm: String = "",

    val maxDataLoaderNWorkers: String = "",
    val persistentWorkers: Boolean = true,

    val samplePrompts: String = "",
    val sampleNegative: String = "",
    val sampleWidth: String = "",
    val sampleHeight: String = "",
    val sampleSteps: String = "",
    val sampleSeed: String = "",
    val sampleRepeat: String = "",
    val guidanceScale: String = ""
) {
    fun validate(): Map<String, String> {
        val errors = mutableMapOf<String, String>()

        fun requireText(key: String, value: String) {
            if (value.isBlank()) errors[key] = "Required"
        }

        fun requireInt(key: String, value: String, min: Int? = null, max: Int? = null) {
            val parsed = value.trim().toIntOrNull()
            if (parsed == null) {
                errors[key] = "Enter an integer"
                return
            }
            if (min != null && parsed < min) errors[key] = "Min $min"
            if (max != null && parsed > max) errors[key] = "Max $max"
        }

        fun requireLong(key: String, value: String) {
            if (value.trim().toLongOrNull() == null) errors[key] = "Enter an integer"
        }

        fun requireDouble(key: String, value: String, min: Double? = null, max: Double? = null) {
            val parsed = value.trim().toDoubleOrNull()
            if (parsed == null) {
                errors[key] = "Enter a number"
                return
            }
            if (min != null && parsed < min) errors[key] = "Min $min"
            if (max != null && parsed > max) errors[key] = "Max $max"
        }

        requireText("pretrained_model_name_or_path", pretrainedModelNameOrPath)
        requireText("output_dir", outputDir)
        requireText("logging_dir", loggingDir)
        requireText("train_data_dir", trainDataDir)
        requireText("output_name", outputName)

        requireText("base_model_version", baseModelVersion)
        requireText("modelspec_architecture", modelspecArchitecture)
        requireText("modelspec_implementation", modelspecImplementation)
        requireText("modelspec_sai_model_spec", modelspecSaiModelSpec)

        requireDouble("min_snr_gamma", minSnrGamma, min = 0.0)
        requireLong("seed", seed)
        requireText("mixed_precision", mixedPrecision)
        requireInt("train_batch_size", trainBatchSize, min = 1)
        requireInt("gradient_accumulation_steps", gradientAccumulationSteps, min = 1)
        requireDouble("learning_rate", learningRate, min = 0.0)
        requireText("lr_scheduler", lrScheduler)
        requireInt("lr_warmup_steps", lrWarmupSteps, min = 0)
        requireDouble("max_grad_norm", maxGradNorm, min = 0.0)
        requireInt("epoch", epoch, min = 1)
        requireInt("save_every_n_epochs", saveEveryNEpochs, min = 1)
        requireInt("save_every_n_steps", saveEveryNSteps, min = 1)

        requireInt("network_dim", networkDim, min = 1)
        requireInt("network_alpha", networkAlpha, min = 1)
        requireDouble("network_dropout", networkDropout, min = 0.0, max = 1.0)
        requireInt("clip_skip", clipSkip, min = 1)
        requireInt("max_token_length", maxTokenLength, min = 75)

        requireInt("train_resolution", trainResolution, min = 64)
        requireInt("bucket_reso_steps", bucketResoSteps, min = 1)
        requireInt("min_bucket_reso", minBucketReso, min = 64)
        requireInt("max_bucket_reso", maxBucketReso, min = 64)

        requireInt("keep_tokens", keepTokens, min = 0)
        requireText("caption_extension", captionExtension)
        requireDouble("noise_offset", noiseOffset, min = 0.0)

        requireDouble("unet_learning_rate", unetLearningRate, min = 0.0)
        requireDouble("unet_weight_decay", unetWeightDecay, min = 0.0)
        requireDouble("unet_betas_1", unetBetas1, min = 0.0, max = 1.0)
        requireDouble("unet_betas_2", unetBetas2, min = 0.0, max = 1.0)
        requireDouble("unet_eps", unetEps, min = 0.0)
        requireInt("unet_warmup_steps", unetWarmupSteps, min = 0)

        requireDouble("te_learning_rate", teLearningRate, min = 0.0)
        requireDouble("te_weight_decay", teWeightDecay, min = 0.0)
        requireDouble("te_betas_1", teBetas1, min = 0.0, max = 1.0)
        requireDouble("te_betas_2", teBetas2, min = 0.0, max = 1.0)
        requireDouble("te_max_grad_norm", teMaxGradNorm, min = 0.0)

        requireInt("max_data_loader_n_workers", maxDataLoaderNWorkers, min = 0)

        requireText("sample_prompts", samplePrompts)
        requireText("sample_negative", sampleNegative)
        requireInt("sample_width", sampleWidth, min = 64)
        requireInt("sample_height", sampleHeight, min = 64)
        requireInt("sample_steps", sampleSteps, min = 1)
        requireLong("sample_seed", sampleSeed)
        requireInt("sample_repeat", sampleRepeat, min = 1)
        requireDouble("guidance_scale", guidanceScale, min = 0.0)

        val minBucket = minBucketReso.trim().toIntOrNull()
        val maxBucket = maxBucketReso.trim().toIntOrNull()
        if (minBucket != null && maxBucket != null && minBucket > maxBucket) {
            errors["max_bucket_reso"] = "Must be ≥ min bucket"
        }

        return errors
    }

    fun toTomlSections(): Map<String, Map<String, String>> {
        fun q(value: String) = TomlDocumentPatcher.quote(value)
        fun n(value: String) = value.trim()
        fun b(value: Boolean) = if (value) "true" else "false"

        return mapOf(
            "environment" to mapOf(
                "pretrained_model_name_or_path" to q(pretrainedModelNameOrPath.trim()),
                "output_dir" to q(outputDir.trim()),
                "logging_dir" to q(loggingDir.trim()),
                "train_data_dir" to q(trainDataDir.trim()),
                "output_name" to q(outputName.trim())
            ),
            "model_spec" to mapOf(
                "base_model_version" to q(baseModelVersion.trim()),
                "modelspec_architecture" to q(modelspecArchitecture.trim()),
                "modelspec_implementation" to q(modelspecImplementation.trim()),
                "modelspec_sai_model_spec" to q(modelspecSaiModelSpec.trim())
            ),
            "training" to mapOf(
                "is_vpred" to b(isVpred),
                "min_snr_gamma" to n(minSnrGamma),
                "seed" to n(seed),
                "mixed_precision" to q(mixedPrecision.trim()),
                "train_batch_size" to n(trainBatchSize),
                "gradient_accumulation_steps" to n(gradientAccumulationSteps),
                "learning_rate" to n(learningRate),
                "lr_scheduler" to q(lrScheduler.trim()),
                "lr_warmup_steps" to n(lrWarmupSteps),
                "max_grad_norm" to n(maxGradNorm),
                "epoch" to n(epoch),
                "save_every_n_epochs" to n(saveEveryNEpochs),
                "save_every_n_steps" to n(saveEveryNSteps)
            ),
            "network" to mapOf(
                "network_dim" to n(networkDim),
                "network_alpha" to n(networkAlpha),
                "network_dropout" to n(networkDropout),
                "clip_skip" to n(clipSkip),
                "max_token_length" to n(maxTokenLength)
            ),
            "bucketing" to mapOf(
                "enable_bucket" to b(enableBucket),
                "bucket_no_upscale" to b(bucketNoUpscale),
                "train_resolution" to n(trainResolution),
                "bucket_reso_steps" to n(bucketResoSteps),
                "min_bucket_reso" to n(minBucketReso),
                "max_bucket_reso" to n(maxBucketReso)
            ),
            "optimization" to mapOf(
                "cache_latents" to b(cacheLatents),
                "cache_latents_to_disk" to b(cacheLatentsToDisk),
                "shuffle_caption" to b(shuffleCaption),
                "keep_tokens" to n(keepTokens),
                "caption_extension" to q(captionExtension.trim()),
                "noise_offset" to n(noiseOffset)
            ),
            "unet_optimizer" to mapOf(
                "unet_learning_rate" to n(unetLearningRate),
                "unet_weight_decay" to n(unetWeightDecay),
                "unet_betas_1" to n(unetBetas1),
                "unet_betas_2" to n(unetBetas2),
                "unet_eps" to n(unetEps),
                "unet_warmup_steps" to n(unetWarmupSteps)
            ),
            "te_optimizer" to mapOf(
                "te_learning_rate" to n(teLearningRate),
                "te_weight_decay" to n(teWeightDecay),
                "te_betas_1" to n(teBetas1),
                "te_betas_2" to n(teBetas2),
                "te_max_grad_norm" to n(teMaxGradNorm)
            ),
            "infrastructure" to mapOf(
                "max_data_loader_n_workers" to n(maxDataLoaderNWorkers),
                "persistent_workers" to b(persistentWorkers)
            ),
            "validation" to mapOf(
                "sample_prompts" to q(samplePrompts),
                "sample_negative" to q(sampleNegative),
                "sample_width" to n(sampleWidth),
                "sample_height" to n(sampleHeight),
                "sample_steps" to n(sampleSteps),
                "sample_seed" to n(sampleSeed),
                "sample_repeat" to n(sampleRepeat),
                "guidance_scale" to n(guidanceScale)
            )
        )
    }

    companion object {
        val mixedPrecisionOptions = listOf("bf16", "fp16", "no")
        val lrSchedulerOptions = listOf(
            "cosine",
            "cosine_with_restarts",
            "linear",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "adafactor"
        )

        fun from(config: AxlTrainerConfig): TrainingConfigForm {
            val env = config.environment
            val spec = config.modelSpec
            val train = config.training
            val net = config.network
            val bucket = config.bucketing
            val opt = config.optimization
            val unet = config.unetOptimizer
            val te = config.teOptimizer
            val infra = config.infrastructure
            val vali = config.validation

            return TrainingConfigForm(
                pretrainedModelNameOrPath = env.pretrainedModelNameOrPath,
                outputDir = env.outputDir,
                loggingDir = env.loggingDir,
                trainDataDir = env.trainDataDir,
                outputName = env.outputName,
                baseModelVersion = spec.baseModelVersion,
                modelspecArchitecture = spec.modelspecArchitecture,
                modelspecImplementation = spec.modelspecImplementation,
                modelspecSaiModelSpec = spec.modelspecSaiModelSpec,
                isVpred = train.isVpred,
                minSnrGamma = formatNumber(train.minSnrGamma),
                seed = train.seed.toString(),
                mixedPrecision = train.mixedPrecision,
                trainBatchSize = train.trainBatchSize.toString(),
                gradientAccumulationSteps = train.gradientAccumulationSteps.toString(),
                learningRate = formatNumber(train.learningRate),
                lrScheduler = train.lrScheduler,
                lrWarmupSteps = train.lrWarmupSteps.toString(),
                maxGradNorm = formatNumber(train.maxGradNorm),
                epoch = train.epoch.toString(),
                saveEveryNEpochs = train.saveEveryNEpochs.toString(),
                saveEveryNSteps = train.saveEveryNSteps.toString(),
                networkDim = net.networkDim.toString(),
                networkAlpha = net.networkAlpha.toString(),
                networkDropout = formatNumber(net.networkDropout),
                clipSkip = net.clipSkip.toString(),
                maxTokenLength = net.maxTokenLength.toString(),
                enableBucket = bucket.enableBucket,
                bucketNoUpscale = bucket.bucketNoUpscale,
                trainResolution = bucket.trainResolution.toString(),
                bucketResoSteps = bucket.bucketResoSteps.toString(),
                minBucketReso = bucket.minBucketReso.toString(),
                maxBucketReso = bucket.maxBucketReso.toString(),
                cacheLatents = opt.cacheLatents,
                cacheLatentsToDisk = opt.cacheLatentsToDisk,
                shuffleCaption = opt.shuffleCaption,
                keepTokens = opt.keepTokens.toString(),
                captionExtension = opt.captionExtension,
                noiseOffset = formatNumber(opt.noiseOffset),
                unetLearningRate = formatNumber(unet.unetLearningRate),
                unetWeightDecay = formatNumber(unet.unetWeightDecay),
                unetBetas1 = formatNumber(unet.unetBetas1),
                unetBetas2 = formatNumber(unet.unetBetas2),
                unetEps = formatNumber(unet.unetEps),
                unetWarmupSteps = unet.unetWarmupSteps.toString(),
                teLearningRate = formatNumber(te.teLearningRate),
                teWeightDecay = formatNumber(te.teWeightDecay),
                teBetas1 = formatNumber(te.teBetas1),
                teBetas2 = formatNumber(te.teBetas2),
                teMaxGradNorm = formatNumber(te.teMaxGradNorm),
                maxDataLoaderNWorkers = infra.maxDataLoaderNWorkers.toString(),
                persistentWorkers = infra.persistentWorkers,
                samplePrompts = vali.samplePrompts,
                sampleNegative = vali.sampleNegative,
                sampleWidth = vali.sampleWidth.toString(),
                sampleHeight = vali.sampleHeight.toString(),
                sampleSteps = vali.sampleSteps.toString(),
                sampleSeed = vali.sampleSeed.toString(),
                sampleRepeat = vali.sampleRepeat.toString(),
                guidanceScale = formatNumber(vali.guidanceScale)
            )
        }

        private fun formatNumber(value: Double): String {
            if (value.isFinite() &&
                value == value.toLong().toDouble() &&
                value in Long.MIN_VALUE.toDouble()..Long.MAX_VALUE.toDouble()
            ) {
                return value.toLong().toString()
            }
            return value.toString()
        }
    }
}

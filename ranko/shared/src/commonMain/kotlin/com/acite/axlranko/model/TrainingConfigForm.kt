package com.acite.axlranko.model

import com.acite.axlranko.data.AxlTrainerConfig
import com.acite.axlranko.data.EnvironmentConfig
import com.acite.axlranko.data.SampleSetConfig
import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.TrainDataEntryConfig
import com.acite.axlranko.data.ValidationConfig
import com.acite.axlranko.data.effectiveTeWarmupSteps
import com.acite.axlranko.data.effectiveUnetMaxGradNorm
import com.acite.axlranko.data.trainDataEntries

/** The TOML section the sample-set blocks live in. */
const val SAMPLE_SETS_SECTION = "validation.samples"

/** Prefix of the field-error keys that belong to one `[[validation.samples]]` entry. */
const val SAMPLE_SET_ERROR_PREFIX = "samples."

/** The TOML section the dataset-folder blocks live in. */
const val TRAIN_DATA_SECTION = "environment.train_data"

/** Prefix of the field-error keys that belong to one `[[environment.train_data]]` entry. */
const val TRAIN_DATA_ERROR_PREFIX = "train_data."

/** Repeat range shared with `TrainConfig`; the picker refuses anything outside it. */
val TRAIN_DATA_REPEAT_RANGE = 1..512

/**
 * The characters a run name may hold, mirroring `trainer/runs.py::validate_output_name`: the name
 * becomes a run id and the artifact directory names, so a space or a slash would make the id
 * (`re_in_…`) disagree with the directories (`re in_samples`) and a run could not be found again.
 */
internal fun isValidOutputName(value: String): Boolean =
    value.isNotEmpty() && value.all { it.isLetterOrDigit() || it in "-_." }

/** Shown under the field; the trainer refuses the same value again at startup. */
const val OUTPUT_NAME_HINT = "Letters, digits, '-', '_' and '.' only"

/**
 * One dataset folder as editable text. `repeat` is how often its images are drawn inside a single
 * epoch; the trainer reads the same range and refuses the value again before training starts.
 */
data class TrainDataDirForm(
    val path: String = "",
    val repeat: String = "1"
)

/**
 * One `[[validation.samples]]` entry as editable text. A key the file omits is shown with the
 * value it inherits from the flat `[validation]` scalars, and saving writes every key back.
 */
data class SampleSetForm(
    val name: String = "",
    val prompt: String = "",
    val negative: String = "",
    val width: String = "",
    val height: String = "",
    val steps: String = "",
    val guidanceScale: String = "",
    val guidanceRescale: String = "",
    val seed: String = "",
    val repeat: String = ""
)

/**
 * The per-set errors of a list of sample sets, keyed `{prefix}{index}.{field}`.
 *
 * Shared by the Utils Validation form and the Dashboard's Sampling Prompts editor: both send the
 * same fields to api.py's `resolve_sample_sets` ranges, so the two must not drift.
 */
internal fun sampleSetFormErrors(
    sets: List<SampleSetForm>,
    prefix: String = SAMPLE_SET_ERROR_PREFIX,
): Map<String, String> {
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

    fun requireDouble(key: String, value: String, min: Double? = null, max: Double? = null) {
        val parsed = value.trim().toDoubleOrNull()
        if (parsed == null || !parsed.isFinite()) {
            errors[key] = "Enter a number"
            return
        }
        if (min != null && parsed < min) errors[key] = "Min $min"
        if (max != null && parsed > max) errors[key] = "Max $max"
    }

    if (sets.isEmpty()) {
        errors[prefix + "0.prompt"] = "At least one sample set"
    }
    sets.forEachIndexed { index, set ->
        val key = { field: String -> "$prefix$index.$field" }
        requireText(key("prompt"), set.prompt)
        requireInt(key("width"), set.width, min = 64, max = 4096)
        requireInt(key("height"), set.height, min = 64, max = 4096)
        requireInt(key("steps"), set.steps, min = 1, max = 150)
        requireInt(key("repeat"), set.repeat, min = 1, max = 32)
        requireDouble(key("guidance_scale"), set.guidanceScale, min = 0.0, max = 30.0)
        requireDouble(key("guidance_rescale"), set.guidanceRescale, min = 0.0, max = 1.0)
        val seedValue = set.seed.trim().toLongOrNull()
        when {
            seedValue == null -> errors[key("seed")] = "Enter an integer"
            seedValue !in 0..4294967295L -> errors[key("seed")] = "0 – 4294967295"
        }
    }
    return errors
}

data class TrainingConfigForm(
    val pretrainedModelNameOrPath: String = "",
    val outputDir: String = "",
    val loggingDir: String = "",
    /** `[[environment.train_data]]`, in file order. Never empty: one folder always exists. */
    val trainDataDirs: List<TrainDataDirForm> = listOf(TrainDataDirForm()),
    val outputName: String = "",
    val amdfq: String = "none",
    val amdfqVramReserveGib: String = "0",
    val amdfqVaNeverReuse: Boolean = false,
    val amdfqPoolMib: String = "64",

    val baseModelVersion: String = "",
    val modelspecArchitecture: String = "",
    val modelspecImplementation: String = "",
    val modelspecSaiModelSpec: String = "",

    val minSnrGamma: String = "",
    val seed: String = "",
    val mixedPrecision: String = "bf16",
    val trainBatchSize: String = "",
    val gradientAccumulationSteps: String = "",
    val learningRate: String = "",
    val epoch: String = "",
    val saveEveryNEpochs: String = "",
    val saveEveryNSteps: String = "",
    /** Whether a checkpoint save also renders the validation samples. Changeable mid-run. */
    val samplingEnabled: Boolean = true,

    val resumeLoraPath: String = "",

    val networkType: String = "standard",
    val networkDim: String = "",
    val networkAlpha: String = "",
    val networkDropout: String = "",
    val convDim: String = "0",
    val convAlpha: String = "0",
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
    val gradientCheckpointingUnet: Boolean = true,
    val gradientCheckpointingTe: Boolean = true,
    val shuffleCaption: Boolean = true,
    val keepTokens: String = "",
    val captionExtension: String = ".txt",
    val noiseOffset: String = "",
    val flushMemoryEveryStep: Boolean = true,

    val unetLearningRate: String = "",
    val unetWeightDecay: String = "",
    val unetBetas1: String = "",
    val unetBetas2: String = "",
    val unetWarmupSteps: String = "",
    val unetMaxGradNorm: String = "",

    val teLearningRate: String = "",
    val teWeightDecay: String = "",
    val teBetas1: String = "",
    val teBetas2: String = "",
    val teMaxGradNorm: String = "",
    val teWarmupSteps: String = "",

    val maxDataLoaderNWorkers: String = "",
    val persistentWorkers: Boolean = true,

    /** `[[validation.samples]]`, in file order. Never empty: one set always exists. */
    val sampleSets: List<SampleSetForm> = listOf(SampleSetForm()),
) {
    /** The set the flat `[validation]` scalars mirror: deleting every set falls back to it. */
    val primarySampleSet: SampleSetForm get() = sampleSets.firstOrNull() ?: SampleSetForm()

    /** The folder the flat `train_data_dir` scalar mirrors, i.e. what a single-folder reader sees. */
    val primaryTrainDataDir: String get() = trainDataDirs.firstOrNull()?.path?.trim() ?: ""

    fun withSampleSet(index: Int, set: SampleSetForm): TrainingConfigForm {
        if (index !in sampleSets.indices) return this
        return copy(sampleSets = sampleSets.toMutableList().also { it[index] = set })
    }

    fun withTrainDataDir(index: Int, entry: TrainDataDirForm): TrainingConfigForm {
        if (index !in trainDataDirs.indices) return this
        return copy(trainDataDirs = trainDataDirs.toMutableList().also { it[index] = entry })
    }

    /**
     * `+`: a blank row, not a copy of the open one. Two rows naming the same folder would train
     * that folder's images twice as often, which is rarely what a second row is for.
     */
    fun appendTrainDataDir(): TrainingConfigForm = copy(trainDataDirs = trainDataDirs + TrainDataDirForm())

    /** Deletes [index]; the last remaining row stays (training needs one dataset folder). */
    fun removeTrainDataDir(index: Int): TrainingConfigForm {
        if (trainDataDirs.size <= 1 || index !in trainDataDirs.indices) return this
        return copy(trainDataDirs = trainDataDirs.filterIndexed { position, _ -> position != index })
    }

    /** `+`: a new set cloned from [cloneOf], so a variant is one edit away. */
    fun appendSampleSet(cloneOf: Int = sampleSets.lastIndex): TrainingConfigForm {
        val clone = sampleSets.getOrNull(cloneOf) ?: SampleSetForm()
        return copy(sampleSets = sampleSets + clone)
    }

    /** Deletes [index]; the last remaining set stays (the config always needs one prompt). */
    fun removeSampleSet(index: Int): TrainingConfigForm {
        if (sampleSets.size <= 1 || index !in sampleSets.indices) return this
        return copy(sampleSets = sampleSets.filterIndexed { position, _ -> position != index })
    }

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
            if (parsed == null || !parsed.isFinite()) {
                errors[key] = "Enter a number"
                return
            }
            if (min != null && parsed < min) errors[key] = "Min $min"
            if (max != null && parsed > max) errors[key] = "Max $max"
        }

        if (amdfq.trim().lowercase() !in amdfqOptions) {
            errors["amdfq"] = "Choose none, tail, or vmm"
        }
        requireDouble("amdfq_vram_reserve_gib", amdfqVramReserveGib, min = 0.0)
        // 0 is off; the hook clamps anything else into [16, 512] MiB, and the picker only offers
        // values inside that range, so anything else here is a hand-edited config line.
        val poolMib = amdfqPoolMib.trim().toIntOrNull()
        if (poolMib == null) {
            errors["amdfq_pool_mib"] = "Enter an integer"
        } else if (poolMib != 0 && poolMib !in 16..512) {
            errors["amdfq_pool_mib"] = "0, or 16 to 512"
        }

        requireText("pretrained_model_name_or_path", pretrainedModelNameOrPath)
        requireText("output_dir", outputDir)
        requireText("logging_dir", loggingDir)
        requireText("output_name", outputName)
        if (outputName.isNotEmpty() && !isValidOutputName(outputName)) {
            errors["output_name"] = OUTPUT_NAME_HINT
        }

        if (trainDataDirs.isEmpty()) {
            errors[TRAIN_DATA_ERROR_PREFIX + "0.path"] = "At least one dataset folder"
        }
        trainDataDirs.forEachIndexed { index, entry ->
            val key = { field: String -> "$TRAIN_DATA_ERROR_PREFIX$index.$field" }
            requireText(key("path"), entry.path)
            requireInt(
                key("repeat"),
                entry.repeat,
                min = TRAIN_DATA_REPEAT_RANGE.first,
                max = TRAIN_DATA_REPEAT_RANGE.last,
            )
        }

        val preset = ModelSpecCatalog.byVersion(baseModelVersion.trim())
        if (preset == null) {
            errors["base_model_version"] = "Unknown base model"
        } else {
            if (modelspecArchitecture.trim() != preset.architecture) {
                errors["modelspec_architecture"] = "Must match ${preset.baseModelVersion}"
            }
            if (modelspecImplementation.trim() != preset.implementation) {
                errors["modelspec_implementation"] = "Must match ${preset.baseModelVersion}"
            }
            if (modelspecSaiModelSpec.trim() != preset.saiModelSpec) {
                errors["modelspec_sai_model_spec"] = "Must match ${preset.baseModelVersion}"
            }
        }

        requireDouble("min_snr_gamma", minSnrGamma, min = 0.0)
        requireLong("seed", seed)
        requireText("mixed_precision", mixedPrecision)
        requireInt("train_batch_size", trainBatchSize, min = 1)
        requireInt("gradient_accumulation_steps", gradientAccumulationSteps, min = 1)
        requireDouble("learning_rate", learningRate, min = 0.0)
        requireInt("epoch", epoch, min = 1)
        requireInt("save_every_n_epochs", saveEveryNEpochs, min = 1)
        requireInt("save_every_n_steps", saveEveryNSteps, min = 1)

        val type = networkType.trim().lowercase()
        if (type !in networkTypeOptions) {
            errors["network_type"] = "standard or locon"
        }
        requireInt("network_dim", networkDim, min = 1)
        requireInt("network_alpha", networkAlpha, min = 1)
        requireDouble("network_dropout", networkDropout, min = 0.0, max = 1.0)
        if (type == "locon") {
            requireInt("conv_dim", convDim, min = 1)
            requireInt("conv_alpha", convAlpha, min = 1)
        } else {
            requireInt("conv_dim", convDim, min = 0)
            requireInt("conv_alpha", convAlpha, min = 0)
        }
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
        requireInt("unet_warmup_steps", unetWarmupSteps, min = 0)
        requireDouble("unet_max_grad_norm", unetMaxGradNorm, min = 0.0)

        requireDouble("te_learning_rate", teLearningRate, min = 0.0)
        requireDouble("te_weight_decay", teWeightDecay, min = 0.0)
        requireDouble("te_betas_1", teBetas1, min = 0.0, max = 1.0)
        requireDouble("te_betas_2", teBetas2, min = 0.0, max = 1.0)
        requireDouble("te_max_grad_norm", teMaxGradNorm, min = 0.0)
        requireInt("te_warmup_steps", teWarmupSteps, min = 0)

        requireInt("max_data_loader_n_workers", maxDataLoaderNWorkers, min = 0)

        errors.putAll(sampleSetFormErrors(sampleSets))

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
        fun f(value: String) = TomlDocumentPatcher.float(value)
        fun b(value: Boolean) = if (value) "true" else "false"
        val primary = primarySampleSet

        return mapOf(
            "environment" to mapOf(
                "pretrained_model_name_or_path" to q(pretrainedModelNameOrPath.trim()),
                "output_dir" to q(outputDir.trim()),
                "logging_dir" to q(loggingDir.trim()),
                // Mirrors the first `[[environment.train_data]]` block, so the file never carries
                // two contradictory dataset folders.
                "train_data_dir" to q(primaryTrainDataDir),
                "output_name" to q(outputName.trim()),
                "amdfq" to q(amdfq.trim().lowercase().ifBlank { "none" }),
                "amdfq_vram_reserve_gib" to f(amdfqVramReserveGib),
                "amdfq_va_never_reuse" to b(amdfqVaNeverReuse),
                "amdfq_pool_mib" to n(amdfqPoolMib.ifBlank { "64" }),
            ),
            "model_spec" to mapOf(
                "base_model_version" to q(baseModelVersion.trim()),
                "modelspec_architecture" to q(modelspecArchitecture.trim()),
                "modelspec_implementation" to q(modelspecImplementation.trim()),
                "modelspec_sai_model_spec" to q(modelspecSaiModelSpec.trim())
            ),
            "training" to mapOf(
                "min_snr_gamma" to f(minSnrGamma),
                "seed" to n(seed),
                "mixed_precision" to q(mixedPrecision.trim()),
                "train_batch_size" to n(trainBatchSize),
                "gradient_accumulation_steps" to n(gradientAccumulationSteps),
                "learning_rate" to f(learningRate),
                "epoch" to n(epoch),
                "save_every_n_epochs" to n(saveEveryNEpochs),
                "save_every_n_steps" to n(saveEveryNSteps),
                "sampling_enabled" to b(samplingEnabled),
                "resume_lora_path" to q(resumeLoraPath.trim())
            ),
            "network" to mapOf(
                "network_type" to q(networkType.trim().lowercase().ifBlank { "standard" }),
                "network_dim" to n(networkDim),
                "network_alpha" to n(networkAlpha),
                "network_dropout" to f(networkDropout),
                "conv_dim" to n(convDim.ifBlank { "0" }),
                "conv_alpha" to n(convAlpha.ifBlank { "0" }),
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
                "gradient_checkpointing_unet" to b(gradientCheckpointingUnet),
                "gradient_checkpointing_te" to b(gradientCheckpointingTe),
                "shuffle_caption" to b(shuffleCaption),
                "keep_tokens" to n(keepTokens),
                "caption_extension" to q(captionExtension.trim()),
                "noise_offset" to f(noiseOffset),
                "flush_memory_every_step" to b(flushMemoryEveryStep)
            ),
            "unet_optimizer" to mapOf(
                "unet_learning_rate" to f(unetLearningRate),
                "unet_weight_decay" to f(unetWeightDecay),
                "unet_betas_1" to f(unetBetas1),
                "unet_betas_2" to f(unetBetas2),
                "unet_warmup_steps" to n(unetWarmupSteps),
                "unet_max_grad_norm" to f(unetMaxGradNorm)
            ),
            "te_optimizer" to mapOf(
                "te_learning_rate" to f(teLearningRate),
                "te_weight_decay" to f(teWeightDecay),
                "te_betas_1" to f(teBetas1),
                "te_betas_2" to f(teBetas2),
                "te_max_grad_norm" to f(teMaxGradNorm),
                "te_warmup_steps" to n(teWarmupSteps)
            ),
            "infrastructure" to mapOf(
                "max_data_loader_n_workers" to n(maxDataLoaderNWorkers),
                "persistent_workers" to b(persistentWorkers)
            ),
            // The flat scalars are what a `[[validation.samples]]` key falls back to, so they
            // mirror the first set: a file never carries two contradictory prompts.
            "validation" to mapOf(
                "sample_prompts" to q(primary.prompt),
                "sample_negative" to q(primary.negative),
                "sample_width" to n(primary.width),
                "sample_height" to n(primary.height),
                "sample_steps" to n(primary.steps),
                "sample_seed" to n(primary.seed),
                "sample_repeat" to n(primary.repeat),
                "guidance_scale" to f(primary.guidanceScale),
                "guidance_rescale" to f(primary.guidanceRescale)
            )
        )
    }

    /** The `[[validation.samples]]` blocks, key order fixed so a save is a stable diff. */
    fun toTomlArrayBlocks(): Map<String, List<Map<String, String>>> {
        val blocks = sampleSets.map { set ->
            linkedMapOf<String, String>().apply {
                // A blank label is left out entirely; the trainer then names the set after
                // its first prompt tag.
                if (set.name.isNotBlank()) put("name", TomlDocumentPatcher.quote(set.name.trim()))
                put("prompt", TomlDocumentPatcher.quote(set.prompt.trim()))
                put("negative", TomlDocumentPatcher.quote(set.negative.trim()))
                put("width", set.width.trim())
                put("height", set.height.trim())
                put("steps", set.steps.trim())
                put("guidance_scale", TomlDocumentPatcher.float(set.guidanceScale))
                put("guidance_rescale", TomlDocumentPatcher.float(set.guidanceRescale))
                put("seed", set.seed.trim())
                put("repeat", set.repeat.trim())
            }
        }
        val trainDataBlocks = trainDataDirs.map { entry ->
            linkedMapOf<String, String>().apply {
                put("path", TomlDocumentPatcher.quote(entry.path.trim()))
                put("repeat", entry.repeat.trim().ifBlank { "1" })
            }
        }
        return mapOf(
            TRAIN_DATA_SECTION to trainDataBlocks,
            SAMPLE_SETS_SECTION to blocks,
        )
    }

    fun withBaseModelVersion(version: String): TrainingConfigForm {
        val preset = ModelSpecCatalog.byVersion(version) ?: return copy(baseModelVersion = version)
        return copy(
            baseModelVersion = preset.baseModelVersion,
            modelspecArchitecture = preset.architecture,
            modelspecImplementation = preset.implementation,
            modelspecSaiModelSpec = preset.saiModelSpec
        )
    }

    companion object {
        val baseModelVersionOptions = ModelSpecCatalog.versions
        val mixedPrecisionOptions = listOf("bf16", "fp16", "no")
        val networkTypeOptions = listOf("standard", "locon")
        val amdfqOptions = listOf("none", "tail", "vmm")
        /** Pool sizes the picker offers, in MiB; 0 is off. The hook clamps to 16..512 and rounds to
         * an even number of allocation granules, so these are all values it takes as written. */
        val amdfqPoolMibOptions = listOf(0, 16, 32, 64, 128, 256, 512)

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
                trainDataDirs = trainDataDirsOf(env),
                outputName = env.outputName,
                amdfq = env.amdfq.trim().lowercase().ifBlank { "none" },
                amdfqVramReserveGib = formatNumber(env.amdfqVramReserveGib),
                amdfqVaNeverReuse = env.amdfqVaNeverReuse,
                amdfqPoolMib = env.amdfqPoolMib.toString(),
                baseModelVersion = spec.baseModelVersion,
                modelspecArchitecture = spec.modelspecArchitecture,
                modelspecImplementation = spec.modelspecImplementation,
                modelspecSaiModelSpec = spec.modelspecSaiModelSpec,
                minSnrGamma = formatNumber(train.minSnrGamma),
                seed = train.seed.toString(),
                mixedPrecision = train.mixedPrecision,
                trainBatchSize = train.trainBatchSize.toString(),
                gradientAccumulationSteps = train.gradientAccumulationSteps.toString(),
                learningRate = formatNumber(train.learningRate),
                epoch = train.epoch.toString(),
                saveEveryNEpochs = train.saveEveryNEpochs.toString(),
                saveEveryNSteps = train.saveEveryNSteps.toString(),
                samplingEnabled = train.samplingEnabled,
                resumeLoraPath = train.resumeLoraPath,
                networkType = net.networkType.trim().lowercase().ifBlank { "standard" },
                networkDim = net.networkDim.toString(),
                networkAlpha = net.networkAlpha.toString(),
                networkDropout = formatNumber(net.networkDropout),
                convDim = net.convDim.toString(),
                convAlpha = net.convAlpha.toString(),
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
                gradientCheckpointingUnet = opt.gradientCheckpointingUnet,
                gradientCheckpointingTe = opt.gradientCheckpointingTe,
                shuffleCaption = opt.shuffleCaption,
                keepTokens = opt.keepTokens.toString(),
                captionExtension = opt.captionExtension,
                noiseOffset = formatNumber(opt.noiseOffset),
                flushMemoryEveryStep = opt.flushMemoryEveryStep,
                unetLearningRate = formatNumber(unet.unetLearningRate),
                unetWeightDecay = formatNumber(unet.unetWeightDecay),
                unetBetas1 = formatNumber(unet.unetBetas1),
                unetBetas2 = formatNumber(unet.unetBetas2),
                unetWarmupSteps = unet.unetWarmupSteps.toString(),
                unetMaxGradNorm = formatNumber(config.effectiveUnetMaxGradNorm()),
                teLearningRate = formatNumber(te.teLearningRate),
                teWeightDecay = formatNumber(te.teWeightDecay),
                teBetas1 = formatNumber(te.teBetas1),
                teBetas2 = formatNumber(te.teBetas2),
                teMaxGradNorm = formatNumber(te.teMaxGradNorm),
                teWarmupSteps = config.effectiveTeWarmupSteps().toString(),
                maxDataLoaderNWorkers = infra.maxDataLoaderNWorkers.toString(),
                persistentWorkers = infra.persistentWorkers,
                sampleSets = sampleSetsOf(vali)
            )
        }

        /**
         * Every `[[environment.train_data]]` entry. A config without any block is shown as the
         * single folder `train_data_dir` names, with repeat 1 - which is what the trainer then
         * trains on, so the row never claims a repeat the file does not carry.
         */
        private fun trainDataDirsOf(environment: EnvironmentConfig): List<TrainDataDirForm> =
            environment.trainDataEntries().map { entry: TrainDataEntryConfig ->
                TrainDataDirForm(path = entry.path, repeat = entry.repeat.toString())
            }

        /**
         * Every `[[validation.samples]]` entry, with a key the entry omits shown as the
         * `[validation]` scalar it inherits. A config without any entry gets the single set
         * those scalars describe, which is what the trainer samples with in that case.
         */
        private fun sampleSetsOf(validation: ValidationConfig): List<SampleSetForm> {
            if (validation.samples.isEmpty()) {
                return listOf(
                    SampleSetForm(
                        prompt = validation.samplePrompts,
                        negative = validation.sampleNegative,
                        width = validation.sampleWidth.toString(),
                        height = validation.sampleHeight.toString(),
                        steps = validation.sampleSteps.toString(),
                        guidanceScale = formatNumber(validation.guidanceScale),
                        guidanceRescale = formatNumber(validation.guidanceRescale),
                        seed = validation.sampleSeed.toString(),
                        repeat = validation.sampleRepeat.toString()
                    )
                )
            }
            return validation.samples.map { set: SampleSetConfig ->
                SampleSetForm(
                    name = set.name,
                    prompt = set.prompt,
                    negative = set.negative ?: validation.sampleNegative,
                    width = (set.width ?: validation.sampleWidth).toString(),
                    height = (set.height ?: validation.sampleHeight).toString(),
                    steps = (set.steps ?: validation.sampleSteps).toString(),
                    guidanceScale = formatNumber(set.guidanceScale ?: validation.guidanceScale),
                    guidanceRescale = formatNumber(set.guidanceRescale ?: validation.guidanceRescale),
                    seed = (set.seed ?: validation.sampleSeed).toString(),
                    repeat = (set.repeat ?: validation.sampleRepeat).toString()
                )
            }
        }

        /** Display-only: whole-valued doubles show as `5`, not `5.0`. Save uses [TomlDocumentPatcher.float]. */
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

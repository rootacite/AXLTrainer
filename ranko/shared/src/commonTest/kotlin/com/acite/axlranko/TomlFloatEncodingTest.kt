package com.acite.axlranko

import com.acite.axlranko.data.AxlTrainerConfig
import com.acite.axlranko.data.BucketingConfig
import com.acite.axlranko.data.EnvironmentConfig
import com.acite.axlranko.data.InfrastructureConfig
import com.acite.axlranko.data.ModelSpecConfig
import com.acite.axlranko.data.NetworkConfig
import com.acite.axlranko.data.OptimizationConfig
import com.acite.axlranko.data.TeOptimizerConfig
import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.TrainingConfig
import com.acite.axlranko.data.UnetOptimizerConfig
import com.acite.axlranko.data.ValidationConfig
import com.acite.axlranko.model.ModelSpecCatalog
import com.acite.axlranko.model.TrainingConfigForm
import com.akuleshov7.ktoml.Toml
import com.akuleshov7.ktoml.TomlInputConfig
import kotlinx.serialization.Serializable
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

class TomlFloatEncodingTest {

    @Test
    fun wholeValuedFloatsKeepAFractionalPart() {
        assertEquals("5.0", TomlDocumentPatcher.float("5"))
        assertEquals("5.0", TomlDocumentPatcher.float("5.0"))
        assertEquals("1.0", TomlDocumentPatcher.float("1"))
        assertEquals("0.0", TomlDocumentPatcher.float("0"))
        assertEquals("6.0", TomlDocumentPatcher.encodeFloat(6.0))
    }

    @Test
    fun nonIntegralFloatsStayFloats() {
        assertEquals("0.15", TomlDocumentPatcher.float("0.15"))
        assertEquals("0.01", TomlDocumentPatcher.encodeFloat(0.01))
        val encoded = TomlDocumentPatcher.encodeFloat(5e-5)
        assertTrue('.' in encoded || encoded.contains('e', ignoreCase = true))
        assertTrue(!encoded.matches(Regex("-?\\d+")))
    }

    @Test
    fun ktomlRejectsIntegerLiteralForDouble() {
        val toml = "value = 5\n"
        assertFails {
            Toml(
                inputConfig = TomlInputConfig(ignoreUnknownNames = true)
            ).decodeFromString(FloatBox.serializer(), toml)
        }
    }

    @Test
    fun ktomlAcceptsEncodedWholeFloat() {
        val toml = "value = ${TomlDocumentPatcher.float("5")}\n"
        val parsed = Toml(
            inputConfig = TomlInputConfig(ignoreUnknownNames = true)
        ).decodeFromString(FloatBox.serializer(), toml)
        assertEquals(5.0, parsed.value)
    }

    @Test
    fun formSaveEncodesIntegerTypedDoublesAsTomlFloats() {
        val form = TrainingConfigForm.from(wholeValuedFloatConfig())
        assertEquals("5", form.minSnrGamma)
        assertEquals("1", form.learningRate)
        val training = form.toTomlSections()["training"]!!
        assertEquals("5.0", training["min_snr_gamma"])
        assertEquals("1.0", training["learning_rate"])
        assertEquals("1.0", training["max_grad_norm"])
        assertEquals("0.0", form.toTomlSections()["network"]!!["network_dropout"])
        assertEquals("0.0", form.toTomlSections()["optimization"]!!["noise_offset"])
        assertEquals("true", form.toTomlSections()["optimization"]!!["flush_memory_every_step"])
        assertEquals("6.0", form.toTomlSections()["validation"]!!["guidance_scale"])
    }
}

@Serializable
private data class FloatBox(val value: Double)

private fun wholeValuedFloatConfig(): AxlTrainerConfig {
    val sdxl = ModelSpecCatalog.byVersion("sdxl_base_v1-0")!!
    return AxlTrainerConfig(
        environment = EnvironmentConfig(
            pretrainedModelNameOrPath = "/models/sdxl",
            trainDataDir = "/data",
            outputName = "run",
            outputDir = "/out",
            loggingDir = "/logs"
        ),
        modelSpec = ModelSpecConfig(
            baseModelVersion = sdxl.baseModelVersion,
            modelspecArchitecture = sdxl.architecture,
            modelspecImplementation = sdxl.implementation,
            modelspecSaiModelSpec = sdxl.saiModelSpec
        ),
        training = TrainingConfig(
            isVpred = false,
            minSnrGamma = 5.0,
            seed = 1,
            mixedPrecision = "bf16",
            trainBatchSize = 1,
            gradientAccumulationSteps = 1,
            learningRate = 1.0,
            lrScheduler = "cosine",
            lrWarmupSteps = 0,
            maxGradNorm = 1.0,
            epoch = 1,
            saveEveryNEpochs = 1,
            saveEveryNSteps = 100
        ),
        network = NetworkConfig(
            networkDim = 8,
            networkAlpha = 4,
            networkDropout = 0.0,
            clipSkip = 1,
            maxTokenLength = 75
        ),
        bucketing = BucketingConfig(
            enableBucket = true,
            bucketNoUpscale = true,
            trainResolution = 1024,
            bucketResoSteps = 128,
            minBucketReso = 768,
            maxBucketReso = 1280
        ),
        optimization = OptimizationConfig(
            cacheLatents = true,
            cacheLatentsToDisk = true,
            shuffleCaption = true,
            keepTokens = 1,
            captionExtension = ".txt",
            noiseOffset = 0.0
        ),
        unetOptimizer = UnetOptimizerConfig(
            unetLearningRate = 1e-4,
            unetWeightDecay = 0.01,
            unetBetas1 = 0.9,
            unetBetas2 = 0.99,
            unetEps = 1e-8,
            unetWarmupSteps = 0
        ),
        teOptimizer = TeOptimizerConfig(
            teLearningRate = 1e-5,
            teWeightDecay = 0.01,
            teBetas1 = 0.9,
            teBetas2 = 0.99,
            teMaxGradNorm = 0.3
        ),
        infrastructure = InfrastructureConfig(
            maxDataLoaderNWorkers = 0,
            persistentWorkers = false
        ),
        validation = ValidationConfig(
            samplePrompts = "prompt",
            sampleNegative = "neg",
            sampleWidth = 1024,
            sampleHeight = 1024,
            sampleSteps = 20,
            sampleSeed = 0,
            sampleRepeat = 1,
            guidanceScale = 6.0
        )
    )
}

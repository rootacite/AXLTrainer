package com.acite.axlranko

import com.acite.axlranko.data.AxlTrainerConfig
import com.acite.axlranko.data.BucketingConfig
import com.acite.axlranko.data.EnvironmentConfig
import com.acite.axlranko.data.InfrastructureConfig
import com.acite.axlranko.data.ModelSpecConfig
import com.acite.axlranko.data.NetworkConfig
import com.acite.axlranko.data.OptimizationConfig
import com.acite.axlranko.data.TeOptimizerConfig
import com.acite.axlranko.data.TrainingConfig
import com.acite.axlranko.data.UnetOptimizerConfig
import com.acite.axlranko.data.ValidationConfig
import com.acite.axlranko.model.ModelSpecCatalog
import com.acite.axlranko.model.TrainingConfigForm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelSpecCatalogTest {
    @Test
    fun shippedSpecValidates() {
        val form = TrainingConfigForm.from(sampleConfig())
        assertTrue(form.validate().isEmpty())
        val spec = form.toTomlSections()["model_spec"]!!
        assertEquals("\"sdxl_base_v1-0\"", spec["base_model_version"])
        assertEquals("\"stable-diffusion-xl-v1-base/lora\"", spec["modelspec_architecture"])
    }

    @Test
    fun unknownVersionIsRejected() {
        val form = sampleForm().copy(baseModelVersion = "sd1.5")
        val errors = form.validate()
        assertEquals("Unknown base model", errors["base_model_version"])
    }

    @Test
    fun mismatchedArchitectureIsRejected() {
        val form = sampleForm().copy(modelspecArchitecture = "wrong")
        val errors = form.validate()
        assertTrue(errors.containsKey("modelspec_architecture"))
    }

    @Test
    fun changingVersionRewritesCatalogFields() {
        val sd35 = ModelSpecCatalog.byVersion("sd3.5-large")!!
        val form = sampleForm().withBaseModelVersion("sd3.5-large")
        assertEquals(sd35.baseModelVersion, form.baseModelVersion)
        assertEquals(sd35.architecture, form.modelspecArchitecture)
        assertEquals(sd35.implementation, form.modelspecImplementation)
        assertEquals(sd35.saiModelSpec, form.modelspecSaiModelSpec)
        assertTrue(form.validate().isEmpty())
        val spec = form.toTomlSections()["model_spec"]!!
        assertEquals("\"sd3.5-large\"", spec["base_model_version"])
        assertEquals("\"${sd35.architecture}\"", spec["modelspec_architecture"])
        assertEquals(false, sd35.trainable)
    }

    private fun sampleForm(): TrainingConfigForm = TrainingConfigForm.from(sampleConfig())

    private fun sampleConfig(): AxlTrainerConfig {
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
                minSnrGamma = 5.0,
                seed = 1,
                mixedPrecision = "bf16",
                trainBatchSize = 1,
                gradientAccumulationSteps = 1,
                learningRate = 1.0,
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
                unetWarmupSteps = 0,
                unetMaxGradNorm = 1.0
            ),
            teOptimizer = TeOptimizerConfig(
                teLearningRate = 1e-5,
                teWeightDecay = 0.01,
                teBetas1 = 0.9,
                teBetas2 = 0.99,
                teMaxGradNorm = 1.0,
                teWarmupSteps = 100
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
}

package com.acite.axlranko

import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.loadTrainerConfig
import com.acite.axlranko.model.TrainingConfigForm
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import okio.Path.Companion.toPath

/**
 * `resume_lora_path` is optional: config.toml files written before it existed must
 * still parse, and saving from the Utils tab must add it in place.
 */
class ResumeConfigTest {

    private fun writeConfig(trainingExtra: String = ""): java.io.File {
        val file = Files.createTempFile("axl-config", ".toml").toFile()
        file.deleteOnExit()
        file.writeText(LEGACY_CONFIG.replace("RESUME_PLACEHOLDER", trainingExtra))
        return file
    }

    /** The same fixture with the VMM reserve row already sitting in `[environment]`. */
    private fun writeConfigWithReserve(value: String): java.io.File {
        val file = Files.createTempFile("axl-config", ".toml").toFile()
        file.deleteOnExit()
        file.writeText(
            LEGACY_CONFIG
                .replace("RESUME_PLACEHOLDER", "")
                .replace(
                    "output_name = \"haruko\"",
                    "output_name = \"haruko\"\namdfq_vram_reserve_gib = $value",
                ),
        )
        return file
    }

    @Test
    fun legacyConfigWithoutResumeKeyParses() {
        val config = assertNotNull(loadTrainerConfig(writeConfig().absolutePath.toPath()))
        assertEquals("", config.training.resumeLoraPath)
        assertEquals(3, config.training.trainBatchSize)
        assertEquals("haruko", config.environment.outputName)
        assertEquals(true, config.optimization.gradientCheckpointingUnet)
        assertEquals(true, config.optimization.gradientCheckpointingTe)
        val form = TrainingConfigForm.from(config)
        assertEquals(true, form.gradientCheckpointingUnet)
        assertEquals(true, form.gradientCheckpointingTe)
        assertEquals("standard", config.network.networkType)
        assertEquals(0, config.network.convDim)
        assertEquals(0, config.network.convAlpha)
        assertEquals("standard", form.networkType)
        assertEquals("0", form.convDim)
        assertEquals("0", form.convAlpha)
        assertEquals("none", config.environment.amdfq)
        assertEquals("none", form.amdfq)
        assertEquals(0.0, config.environment.amdfqVramReserveGib)
        assertEquals("0", form.amdfqVramReserveGib)
        assertEquals(false, config.environment.amdfqVaNeverReuse)
        assertEquals(false, form.amdfqVaNeverReuse)
        // A config written before the row existed gets the shipped default, not off.
        assertEquals(64, config.environment.amdfqPoolMib)
        assertEquals("64", form.amdfqPoolMib)
    }

    @Test
    fun patcherInsertsNetworkTypeIntoLegacyFile() {
        val source = LEGACY_CONFIG.replace("RESUME_PLACEHOLDER", "")
        val patched = TomlDocumentPatcher.apply(
            source,
            mapOf(
                "network" to mapOf(
                    "network_type" to TomlDocumentPatcher.quote("locon"),
                    "conv_dim" to "16",
                    "conv_alpha" to "8",
                ),
            ),
        )
        assertTrue(patched.contains("network_type = \"locon\""))
        assertTrue(patched.contains("conv_dim = 16"))
        assertTrue(patched.contains("conv_alpha = 8"))
        assertTrue(patched.contains("network_dim = 48"))
        val file = Files.createTempFile("axl-config", ".toml").toFile()
        file.deleteOnExit()
        file.writeText(patched)
        val config = assertNotNull(loadTrainerConfig(file.absolutePath.toPath()))
        assertEquals("locon", config.network.networkType)
        assertEquals(16, config.network.convDim)
        assertEquals(8, config.network.convAlpha)
    }

    @Test
    fun patcherInsertsAmdfqIntoLegacyFile() {
        val source = LEGACY_CONFIG.replace("RESUME_PLACEHOLDER", "")
        val patched = TomlDocumentPatcher.apply(
            source,
            mapOf("environment" to mapOf("amdfq" to TomlDocumentPatcher.quote("vmm"))),
        )
        assertTrue(patched.contains("amdfq = \"vmm\""))
        assertTrue(patched.contains("output_name = \"haruko\""))
        assertTrue(patched.contains("[bookkeeping]"))
    }

    @Test
    fun patcherInsertsVramReserveIntoLegacyFile() {
        val source = LEGACY_CONFIG.replace("RESUME_PLACEHOLDER", "")
        val patched = TomlDocumentPatcher.apply(
            source,
            mapOf("environment" to mapOf("amdfq_vram_reserve_gib" to "0.0")),
        )
        assertTrue(patched.contains("amdfq_vram_reserve_gib = 0.0"))
        assertTrue(patched.contains("output_name = \"haruko\""))
        assertTrue(patched.contains("[bookkeeping]"))
    }

    /** The pool row is not in a legacy config.toml at all: the patcher has to add it, and a second
     * save must not append a second copy. */
    @Test
    fun patcherInsertsPoolSizeIntoLegacyFile() {
        val source = LEGACY_CONFIG.replace("RESUME_PLACEHOLDER", "")
        val patched = TomlDocumentPatcher.apply(
            source,
            mapOf("environment" to mapOf("amdfq_pool_mib" to "64")),
        )
        assertTrue(patched.contains("amdfq_pool_mib = 64"))
        assertTrue(patched.contains("output_name = \"haruko\""))
        assertTrue(patched.contains("[bookkeeping]"))

        val again = TomlDocumentPatcher.apply(
            patched,
            mapOf("environment" to mapOf("amdfq_pool_mib" to "128")),
        )
        assertEquals(1, again.split("amdfq_pool_mib").size - 1)
        assertTrue(again.contains("amdfq_pool_mib = 128"))
    }

    /**
     * The reserve row is a Kotlin `Double`, and ktoml refuses an integer literal for a `Double`:
     * a bare `amdfq_vram_reserve_gib = 0` stops Chromatrix from parsing its own config.toml. Saving the
     * row as 0 therefore has to leave a float literal behind.
     */
    @Test
    fun savingZeroReserveLeavesAFloatLiteral() {
        val file = writeConfigWithReserve("0.5")
        val form = TrainingConfigForm.from(assertNotNull(loadTrainerConfig(file.absolutePath.toPath())))
            .copy(amdfqVramReserveGib = "0")

        val patched = TomlDocumentPatcher.apply(file.readText(), form.toTomlSections())
        assertTrue(patched.contains("amdfq_vram_reserve_gib = 0.0"))
        assertEquals(1, patched.split("amdfq_vram_reserve_gib").size - 1)

        file.writeText(patched)
        val reloaded = assertNotNull(loadTrainerConfig(file.absolutePath.toPath()))
        assertEquals(0.0, reloaded.environment.amdfqVramReserveGib)
        assertEquals("0", TrainingConfigForm.from(reloaded).amdfqVramReserveGib)
    }

    /** A hand-written `0` in that row must not cost Chromatrix its startup. */
    @Test
    fun integerLiteralInTheReserveRowStillParses() {
        val config = assertNotNull(loadTrainerConfig(writeConfigWithReserve("0").absolutePath.toPath()))
        assertEquals(0.0, config.environment.amdfqVramReserveGib)
        assertEquals("0", TrainingConfigForm.from(config).amdfqVramReserveGib)
    }

    @Test
    fun patcherInsertsVaNeverReuseIntoLegacyFile() {
        val source = LEGACY_CONFIG.replace("RESUME_PLACEHOLDER", "")
        val patched = TomlDocumentPatcher.apply(
            source,
            mapOf("environment" to mapOf("amdfq_va_never_reuse" to "true")),
        )
        assertTrue(patched.contains("amdfq_va_never_reuse = true"))
        assertTrue(patched.contains("output_name = \"haruko\""))
        assertTrue(patched.contains("[bookkeeping]"))
    }

    @Test
    fun configWithResumeKeyParses() {
        val config = assertNotNull(
            loadTrainerConfig(
                writeConfig("resume_lora_path = \"/out/rein_final/rein.safetensors\"")
                    .absolutePath
                    .toPath(),
            ),
        )
        assertEquals("/out/rein_final/rein.safetensors", config.training.resumeLoraPath)
        val form = TrainingConfigForm.from(config)
        assertEquals("/out/rein_final/rein.safetensors", form.resumeLoraPath)
    }

    @Test
    fun patcherInsertsResumeKeyIntoLegacyFile() {
        val source = LEGACY_CONFIG.replace("RESUME_PLACEHOLDER", "")
        val patched = TomlDocumentPatcher.apply(
            source,
            mapOf(
                "training" to mapOf(
                    "resume_lora_path" to TomlDocumentPatcher.quote("/out/rein_final/rein.safetensors"),
                    "epoch" to "56",
                ),
            ),
        )
        assertTrue(patched.contains("resume_lora_path = \"/out/rein_final/rein.safetensors\""))
        assertTrue(patched.contains("epoch = 56"))
        assertTrue(patched.contains("# keep this comment"))
        assertTrue(patched.contains("[bookkeeping]"))
        assertEquals(1, patched.split("[training]").size - 1)
    }

    private companion object {
        val LEGACY_CONFIG = """
            [environment]
            pretrained_model_name_or_path = "/opt/models/sdxl"
            output_dir = "/tmp/out"
            logging_dir = "/tmp/logs"
            train_data_dir = "/tmp/data"
            output_name = "haruko"

            [model_spec]
            base_model_version = "sdxl_base_v1-0"
            modelspec_architecture = "stable-diffusion-xl-v1-base/lora"
            modelspec_implementation = "https://github.com/Stability-AI/generative-models"
            modelspec_sai_model_spec = "1.0.0"

            [training]
            # keep this comment
            min_snr_gamma = 5.0
            seed = 1145141919
            mixed_precision = "bf16"
            train_batch_size = 3
            gradient_accumulation_steps = 1
            learning_rate = 1.0
            # A row this version no longer declares, the way an old file still carries it: it has
            # to be ignored rather than made fatal.
            lr_scheduler = "cosine"
            # And one that moved to `[unet_optimizer].unet_max_grad_norm`, which the loader still
            # reads from here when the new key is absent.
            max_grad_norm = 1.0
            epoch = 16
            save_every_n_epochs = 1
            save_every_n_steps = 100
            RESUME_PLACEHOLDER

            [network]
            network_dim = 48
            network_alpha = 24
            network_dropout = 0.15
            clip_skip = 1
            max_token_length = 225

            [bucketing]
            enable_bucket = true
            bucket_no_upscale = true
            train_resolution = 1024
            bucket_reso_steps = 128
            min_bucket_reso = 768
            max_bucket_reso = 1280

            [optimization]
            cache_latents = true
            cache_latents_to_disk = true
            shuffle_caption = true
            keep_tokens = 2
            caption_extension = ".txt"
            noise_offset = 0.05

            [unet_optimizer]
            unet_learning_rate = 5.0E-5
            unet_weight_decay = 0.01
            unet_betas_1 = 0.9
            unet_betas_2 = 0.99
            # Like `lr_scheduler` above: a row this version no longer declares.
            unet_eps = 1.0E-8
            unet_warmup_steps = 100

            [te_optimizer]
            te_learning_rate = 5.0E-6
            te_weight_decay = 0.01
            te_betas_1 = 0.9
            te_betas_2 = 0.99
            te_max_grad_norm = 0.3
            te_warmup_steps = 100

            [infrastructure]
            max_data_loader_n_workers = 20
            persistent_workers = true

            [validation]
            sample_prompts = "prompt"
            sample_negative = "negative"
            sample_width = 1280
            sample_height = 720
            sample_steps = 35
            sample_seed = 0
            sample_repeat = 3
            guidance_scale = 6.0

            [bookkeeping]
        """.trimIndent()
    }
}

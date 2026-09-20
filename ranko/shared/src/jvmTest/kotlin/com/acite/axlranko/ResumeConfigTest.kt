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
        assertEquals("none", config.environment.amdfq)
        assertEquals("none", form.amdfq)
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
            is_vpred = false
            min_snr_gamma = 5.0
            seed = 1145141919
            mixed_precision = "bf16"
            train_batch_size = 3
            gradient_accumulation_steps = 1
            learning_rate = 1.0
            lr_scheduler = "cosine"
            lr_warmup_steps = 100
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
            unet_eps = 1.0E-8
            unet_warmup_steps = 100

            [te_optimizer]
            te_learning_rate = 5.0E-6
            te_weight_decay = 0.01
            te_betas_1 = 0.9
            te_betas_2 = 0.99
            te_max_grad_norm = 0.3

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

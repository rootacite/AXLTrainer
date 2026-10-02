package com.acite.axlranko

import com.acite.axlranko.data.ConfigImporter
import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.effectiveTeWarmupSteps
import com.acite.axlranko.data.effectiveUnetMaxGradNorm
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.TrainingConfigForm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Two keys that moved out of `[training]` into the optimizer sections: the text encoder's warmup
 * (`lr_warmup_steps` → `[te_optimizer].te_warmup_steps`) and the UNet's gradient clipping
 * (`max_grad_norm` → `[unet_optimizer].unet_max_grad_norm`). A config that still carries only the
 * old key has to keep its value — in what the trainer resolves and in what the Utils form shows and
 * saves.
 */
class OptimizerConfigKeysTest {

    private fun document(
        trainingRows: String = "",
        unetRows: String = "",
        teRows: String = "",
    ): String = CONFIG
        .replace("TRAINING_ROWS", trainingRows)
        .replace("UNET_ROWS", unetRows)
        .replace("TE_ROWS", teRows)

    private fun config(
        trainingRows: String = "",
        unetRows: String = "",
        teRows: String = "",
    ) = assertNotNull(ConfigImporter.parseConfig(document(trainingRows, unetRows, teRows)))

    // --- the text encoder's warmup -----------------------------------------------------------------

    @Test
    fun theTeWarmupComesFromItsOwnKey() {
        assertEquals(40, config(teRows = "te_warmup_steps = 40").effectiveTeWarmupSteps())
    }

    @Test
    fun theTeWarmupStillReadsItsOldKey() {
        assertEquals(77, config(trainingRows = "lr_warmup_steps = 77").effectiveTeWarmupSteps())
    }

    @Test
    fun theTeWarmupKeyWinsWhenTheFileCarriesBoth() {
        val both = config(trainingRows = "lr_warmup_steps = 77", teRows = "te_warmup_steps = 40")
        assertEquals(40, both.effectiveTeWarmupSteps())
    }

    @Test
    fun aFileWithNeitherTeWarmupKeyKeepsTheDefault() {
        assertEquals(100, config().effectiveTeWarmupSteps())
    }

    // --- the UNet's gradient clipping --------------------------------------------------------------

    @Test
    fun theUnetClipComesFromItsOwnKey() {
        assertEquals(2.5, config(unetRows = "unet_max_grad_norm = 2.5").effectiveUnetMaxGradNorm())
    }

    @Test
    fun theUnetClipStillReadsItsOldKey() {
        assertEquals(0.8, config(trainingRows = "max_grad_norm = 0.8").effectiveUnetMaxGradNorm())
    }

    @Test
    fun theUnetClipKeyWinsWhenTheFileCarriesBoth() {
        val both = config(trainingRows = "max_grad_norm = 0.8", unetRows = "unet_max_grad_norm = 2.5")
        assertEquals(2.5, both.effectiveUnetMaxGradNorm())
    }

    @Test
    fun aFileWithNeitherClipKeyKeepsTheDefault() {
        assertEquals(1.0, config().effectiveUnetMaxGradNorm())
    }

    // --- what the Utils form shows and writes ------------------------------------------------------

    @Test
    fun theFormShowsTheResolvedWarmupAndClip() {
        val form = TrainingConfigForm.from(
            config(trainingRows = "lr_warmup_steps = 77\nmax_grad_norm = 0.8"),
        )
        assertEquals("77", form.teWarmupSteps)
        assertEquals("0.8", form.unetMaxGradNorm)
    }

    @Test
    fun theFormWritesBothKeysIntoTheirOptimizerSections() {
        val sections = TrainingConfigForm.from(
            config(unetRows = "unet_max_grad_norm = 2.5", teRows = "te_warmup_steps = 40"),
        ).toTomlSections()

        assertEquals("40", sections["te_optimizer"]!!["te_warmup_steps"])
        assertEquals("2.5", sections["unet_optimizer"]!!["unet_max_grad_norm"])
        assertFalse("lr_warmup_steps" in sections["training"]!!)
        assertFalse("max_grad_norm" in sections["training"]!!)
    }

    @Test
    fun theSectionFieldKeysFollowTheMoves() {
        assertTrue("te_warmup_steps" in ConfigSection.TeOptimizer.fieldKeys)
        assertTrue("unet_max_grad_norm" in ConfigSection.UnetOptimizer.fieldKeys)
        assertFalse("lr_warmup_steps" in ConfigSection.Training.fieldKeys)
        assertFalse("max_grad_norm" in ConfigSection.Training.fieldKeys)
    }

    @Test
    fun savingALegacyFileKeepsBothValues() {
        val text = document(trainingRows = "lr_warmup_steps = 77\nmax_grad_norm = 0.8")
        val form = TrainingConfigForm.from(assertNotNull(ConfigImporter.parseConfig(text)))
        val patched = TomlDocumentPatcher.apply(text, form.toTomlSections())

        assertTrue(patched.contains("te_warmup_steps = 77"))
        assertTrue(patched.contains("unet_max_grad_norm = 0.8"))
        // The lines the keys used to live on are left alone (the patcher only touches what it
        // writes), and the reloaded config resolves the same values either way.
        assertTrue(patched.contains("lr_warmup_steps = 77"))
        assertTrue(patched.contains("max_grad_norm = 0.8"))
        val reloaded = assertNotNull(ConfigImporter.parseConfig(patched))
        assertEquals(77, reloaded.effectiveTeWarmupSteps())
        assertEquals(0.8, reloaded.effectiveUnetMaxGradNorm())
    }

    private companion object {
        /** A complete config, in the shape the other config tests use. */
        val CONFIG = """
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
            min_snr_gamma = 5.0
            seed = 1145141919
            mixed_precision = "bf16"
            train_batch_size = 3
            gradient_accumulation_steps = 1
            learning_rate = 1.0
            epoch = 16
            save_every_n_epochs = 1
            save_every_n_steps = 100
            TRAINING_ROWS

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
            unet_warmup_steps = 100
            UNET_ROWS

            [te_optimizer]
            te_learning_rate = 5.0E-6
            te_weight_decay = 0.01
            te_betas_1 = 0.9
            te_betas_2 = 0.99
            te_max_grad_norm = 1.0
            TE_ROWS

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

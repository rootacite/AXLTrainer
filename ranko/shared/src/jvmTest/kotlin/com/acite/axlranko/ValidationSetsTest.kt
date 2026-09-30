package com.acite.axlranko

import com.acite.axlranko.data.SampleSetConfig
import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.ValidationConfig
import com.acite.axlranko.data.loadTrainerConfig
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.pages.sampleSetLabel
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import okio.Path.Companion.toPath

/**
 * `[[validation.samples]]` is an array of tables: ktoml has to decode it into the sectional
 * model, an absent array has to stay an empty list, and the Utils form has to round-trip the
 * blocks through the line-preserving patcher.
 */
class ValidationSetsTest {

    private fun writeConfig(samples: String): java.io.File {
        val file = Files.createTempFile("axl-samples", ".toml").toFile()
        file.deleteOnExit()
        file.writeText(CONFIG.replace("SAMPLES_PLACEHOLDER", samples))
        return file
    }

    private fun load(samples: String = "") =
        assertNotNull(loadTrainerConfig(writeConfig(samples).absolutePath.toPath()))

    private fun formOf(samples: String = "") = TrainingConfigForm.from(load(samples))

    @Test
    fun twoSetsDecodeWithTheirOwnKeys() {
        val samples = load(
            """
            [[validation.samples]]
            name = "close up"
            prompt = "p1"
            steps = 8
            repeat = 1

            [[validation.samples]]
            prompt = "p2"
            width = 512
            """.trimIndent(),
        ).validation.samples
        assertEquals(2, samples.size)
        assertEquals("close up", samples[0].name)
        assertEquals("p1", samples[0].prompt)
        assertEquals(8, samples[0].steps)
        assertEquals(1, samples[0].repeat)
        assertEquals(null, samples[0].width)
        assertEquals(null, samples[0].negative)
        assertEquals("", samples[1].name)
        assertEquals("p2", samples[1].prompt)
        assertEquals(512, samples[1].width)
    }

    @Test
    fun anAbsentArrayDecodesToAnEmptyList() {
        val config = load()
        assertTrue(config.validation.samples.isEmpty())
        // The flat scalars stay as they are; the form shows one set built from them.
        assertEquals("flat prompt", config.validation.samplePrompts)
        val set = TrainingConfigForm.from(config).sampleSets.single()
        assertEquals("flat prompt", set.prompt)
        assertEquals("flat negative", set.negative)
        assertEquals("1280", set.width)
        assertEquals("35", set.steps)
    }

    @Test
    fun theFormShowsTheInheritedValueForAnOmittedKey() {
        val set = formOf(
            """
            [[validation.samples]]
            prompt = "only a prompt"
            """.trimIndent(),
        ).sampleSets.single()
        assertEquals("only a prompt", set.prompt)
        assertEquals("flat negative", set.negative)
        assertEquals("1280", set.width)
        assertEquals("35", set.steps)
        assertEquals("6", set.guidanceScale)
        assertEquals("0.6", set.guidanceRescale)
        assertEquals("0", set.seed)
        assertEquals("3", set.repeat)
    }

    @Test
    fun theRescaleFallsBackToTheFlatScalar() {
        // Config.toml ships 0.6 (ComfyUI's RescaleCFG); a set may ask for its own.
        val withoutSet = formOf().sampleSets.single()
        assertEquals("0.6", withoutSet.guidanceRescale)

        val set = formOf(
            """
            [[validation.samples]]
            prompt = "p"
            guidance_rescale = 0.3
            """.trimIndent(),
        ).sampleSets.single()
        assertEquals("0.3", set.guidanceRescale)
    }

    @Test
    fun aRescaleOutsideZeroToOneIsRejected() {
        val form = formOf().copy(
            sampleSets = listOf(formOf().sampleSets[0].copy(guidanceRescale = "1.5")),
        )
        val errors = form.validate()
        assertEquals("Max 1.0", errors["samples.0.guidance_rescale"])
    }

    @Test
    fun theTabLabelFollowsTheNameThenTheFirstPromptTag() {
        val form = formOf()
        assertEquals("flat prompt", sampleSetLabel(form.sampleSets[0], 0))
        assertEquals("tag", sampleSetLabel(form.sampleSets[0].copy(prompt = "tag, rest"), 0))
        assertEquals("explicit", sampleSetLabel(form.sampleSets[0].copy(name = "explicit"), 0))
    }

    @Test
    fun addClonesTheOpenSetAndRemoveKeepsTheLastOne() {
        val form = formOf().copy(
            sampleSets = listOf(formOf().sampleSets[0].copy(name = "first", prompt = "a")),
        )
        val added = form.appendSampleSet(0)
        assertEquals(2, added.sampleSets.size)
        assertEquals("first", added.sampleSets[1].name)

        val removed = added.removeSampleSet(0)
        assertEquals(1, removed.sampleSets.size)
        assertEquals("first", removed.sampleSets.single().name)
        // Never empty: the config always needs one prompt.
        assertEquals(1, form.removeSampleSet(0).sampleSets.size)
    }

    @Test
    fun validationReportsErrorsPerSet() {
        val form = formOf().copy(
            sampleSets = listOf(
                formOf().sampleSets[0],
                formOf().sampleSets[0].copy(prompt = "", steps = "0"),
            ),
        )
        val errors = form.validate()
        assertEquals("Required", errors["samples.1.prompt"])
        assertEquals("Min 1", errors["samples.1.steps"])
        assertTrue(errors.keys.none { it.startsWith("samples.0.") })
    }

    @Test
    fun savedBlocksRoundTripThroughThePatcher() {
        val base = formOf().sampleSets[0]
        val form = formOf().copy(
            sampleSets = listOf(
                base.copy(name = "close up", prompt = "p1", steps = "8", repeat = "1"),
                base.copy(name = "", prompt = "p2", width = "512"),
            ),
        )
        assertTrue(form.validate().isEmpty())
        val blocks = form.toTomlArrayBlocks().getValue("validation.samples")

        val patched = TomlDocumentPatcher.replaceArrayOfTables(writeConfig("").readText(), "validation.samples", blocks)
        assertEquals(2, patched.split("[[validation.samples]]").size - 1)
        assertTrue(patched.contains("# keep this comment"))
        assertTrue(patched.contains("[bookkeeping]"))
        assertEquals(1, patched.split("[validation]").size - 1)
        assertTrue(patched.contains("sample_prompts = \"flat prompt\""))

        val sets = TrainingConfigForm.from(
            assertNotNull(
                loadTrainerConfig(
                    Files.createTempFile("axl-roundtrip", ".toml").toFile().apply {
                        deleteOnExit()
                        writeText(patched)
                    }.absolutePath.toPath(),
                ),
            ),
        ).sampleSets
        assertEquals(2, sets.size)
        assertEquals("close up", sets[0].name)
        assertEquals("p1", sets[0].prompt)
        assertEquals("8", sets[0].steps)
        assertEquals("1", sets[0].repeat)
        assertEquals("", sets[1].name)
        assertEquals("p2", sets[1].prompt)
        assertEquals("512", sets[1].width)

        // Idempotent: patching the patched file again changes nothing.
        assertEquals(patched, TomlDocumentPatcher.replaceArrayOfTables(patched, "validation.samples", blocks))
    }

    @Test
    fun replacingKeepsTheFollowingTableSeparated() {
        val source = writeConfig(
            """
            [[validation.samples]]
            name = "old"
            prompt = "old"

            [bookkeeping]
            """.trimIndent(),
        ).readText()
        val patched = TomlDocumentPatcher.replaceArrayOfTables(
            source,
            "validation.samples",
            listOf(linkedMapOf("prompt" to "\"new\"")),
        )
        assertTrue(
            patched.contains(
                "guidance_scale = 6.0\nguidance_rescale = 0.6\n\n[[validation.samples]]\nprompt = \"new\"\n\n[bookkeeping]",
            ),
        )
        assertTrue(!patched.contains("old"))
    }

    @Test
    fun anEmptyBlockListLeavesTheFileAlone() {
        val source = writeConfig("").readText()
        assertEquals(source, TomlDocumentPatcher.replaceArrayOfTables(source, "validation.samples", emptyList()))
    }

    @Test
    fun theFlatScalarsMirrorTheFirstSet() {
        val config = load().copy(
            validation = ValidationConfig(
                samplePrompts = "flat prompt",
                sampleNegative = "flat negative",
                sampleWidth = 1280,
                sampleHeight = 720,
                sampleSteps = 35,
                sampleSeed = 0,
                sampleRepeat = 3,
                guidanceScale = 6.0,
                samples = listOf(SampleSetConfig(name = "one", prompt = "p1", steps = 8, repeat = 1)),
            ),
        )
        val validation = TrainingConfigForm.from(config).toTomlSections().getValue("validation")
        assertEquals("\"p1\"", validation.getValue("sample_prompts"))
        assertEquals("8", validation.getValue("sample_steps"))
        assertEquals("1", validation.getValue("sample_repeat"))
        // A key the set omitted keeps the flat scalar it inherits.
        assertEquals("\"flat negative\"", validation.getValue("sample_negative"))
    }

    @Test
    fun sampleSetErrorsBelongToTheValidationSection() {
        assertTrue(ConfigSection.Validation.owns("samples.1.steps"))
        assertTrue(ConfigSection.Validation.owns("sample_prompts"))
        assertFalse(ConfigSection.Network.owns("samples.1.steps"))
        assertFalse(ConfigSection.Validation.owns("network_dim"))
    }

    private companion object {
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
            lr_scheduler = "cosine"
            lr_warmup_steps = 100
            max_grad_norm = 1.0
            epoch = 16
            save_every_n_epochs = 1
            save_every_n_steps = 100

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
            # keep this comment
            sample_prompts = "flat prompt"
            sample_negative = "flat negative"
            sample_width = 1280
            sample_height = 720
            sample_steps = 35
            sample_seed = 0
            sample_repeat = 3
            guidance_scale = 6.0
            guidance_rescale = 0.6
            SAMPLES_PLACEHOLDER

            [bookkeeping]
        """.trimIndent()
    }
}

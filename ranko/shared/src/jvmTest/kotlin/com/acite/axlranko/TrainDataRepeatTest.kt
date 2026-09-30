package com.acite.axlranko

import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.loadTrainerConfig
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.TRAIN_DATA_ERROR_PREFIX
import com.acite.axlranko.model.TRAIN_DATA_SECTION
import com.acite.axlranko.model.TrainDataDirForm
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.pages.components.datasetDirLabel
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import okio.Path.Companion.toPath

/**
 * `[[environment.train_data]]` is an array of tables, like `[[validation.samples]]`: the sectional
 * model has to decode it, an absent array has to stay an empty list (so a config written before the
 * feature trains one folder once), and the Utils form has to write the blocks back through the
 * line-preserving patcher while keeping the `train_data_dir` scalar on the first row.
 */
class TrainDataRepeatTest {

    private fun writeConfig(blocks: String = ""): java.io.File {
        val file = Files.createTempFile("axl-train-data", ".toml").toFile()
        file.deleteOnExit()
        file.writeText(CONFIG.replace("TRAIN_DATA_PLACEHOLDER", blocks))
        return file
    }

    private fun load(blocks: String = "") =
        assertNotNull(loadTrainerConfig(writeConfig(blocks).absolutePath.toPath()))

    private fun formOf(blocks: String = "") = TrainingConfigForm.from(load(blocks))

    private fun reload(configText: String) = assertNotNull(
        loadTrainerConfig(writeConfig().also { it.writeText(configText) }.absolutePath.toPath())
    )

    /** What `saveTrainerConfigPatched` does: array blocks first, then the scalar rows. */
    private fun save(source: String, form: TrainingConfigForm): String {
        var patched = source
        for ((section, blocks) in form.toTomlArrayBlocks()) {
            patched = TomlDocumentPatcher.replaceArrayOfTables(patched, section, blocks)
        }
        return TomlDocumentPatcher.apply(patched, form.toTomlSections())
    }

    @Test
    fun blocksDecodeWithTheirOwnPathAndRepeat() {
        val config = load(
            """
            [[environment.train_data]]
            path = "/data/first"
            repeat = 3

            [[environment.train_data]]
            path = "/data/second"
            """.trimIndent(),
        )
        val entries = config.environment.trainData
        assertEquals(2, entries.size)
        assertEquals("/data/first", entries[0].path)
        assertEquals(3, entries[0].repeat)
        assertEquals("/data/second", entries[1].path)
        assertEquals(1, entries[1].repeat)
    }

    @Test
    fun absentBlocksLeaveAnEmptyList() {
        val config = load()
        assertTrue(config.environment.trainData.isEmpty())
        assertEquals("/tmp/data", config.environment.trainDataDir)
    }

    @Test
    fun theFormListsOneRowPerBlock() {
        val form = formOf(
            """
            [[environment.train_data]]
            path = "/data/first"
            repeat = 8

            [[environment.train_data]]
            path = "/data/second"
            repeat = 2
            """.trimIndent(),
        )
        assertEquals(listOf("/data/first", "/data/second"), form.trainDataDirs.map { it.path })
        assertEquals(listOf("8", "2"), form.trainDataDirs.map { it.repeat })
        assertTrue(form.validate().isEmpty())
    }

    @Test
    fun aConfigWithoutBlocksIsShownAsOneRowDrawnOnce() {
        val form = formOf()
        assertEquals(1, form.trainDataDirs.size)
        assertEquals("/tmp/data", form.trainDataDirs[0].path)
        assertEquals("1", form.trainDataDirs[0].repeat)
    }

    @Test
    fun savingWritesTheBlocksAndKeepsTheMirrorInStep() {
        val form = formOf().copy(
            trainDataDirs = listOf(
                TrainDataDirForm(path = "/data/first", repeat = "4"),
                TrainDataDirForm(path = "/data/second", repeat = "1"),
            )
        )
        val patched = save(writeConfig().readText(), form)

        // The blocks land inside [environment], before the next table, and comments stay.
        assertTrue(patched.contains("# keep this comment"))
        assertTrue(patched.indexOf("[[environment.train_data]]") < patched.indexOf("[model_spec]"))
        assertEquals(2, patched.split("[[environment.train_data]]").size - 1)
        // The scalar mirrors the first row, and only once.
        assertEquals(1, patched.lines().count { it.trim().startsWith("train_data_dir") })
        assertTrue(patched.contains("train_data_dir = \"/data/first\""))

        val reloaded = reload(patched)
        assertEquals(
            listOf("/data/first" to 4, "/data/second" to 1),
            reloaded.environment.trainData.map { it.path to it.repeat },
        )
        assertEquals("/data/first", reloaded.environment.trainDataDir)
        assertEquals(
            listOf("4", "1"),
            TrainingConfigForm.from(reloaded).trainDataDirs.map { it.repeat },
        )
    }

    @Test
    fun savingAgainReplacesTheBlocksInsteadOfAppending() {
        val source = writeConfig().readText()
        val form = formOf(
            """
            [[environment.train_data]]
            path = "/data/first"

            [[environment.train_data]]
            path = "/data/second"
            """.trimIndent(),
        )
        val once = save(source, form)
        val twice = save(once, form)
        assertEquals(2, twice.split("[[environment.train_data]]").size - 1)
        assertEquals(1, twice.split("path = \"/data/second\"").size - 1)
        assertEquals("/data/first", reload(twice).environment.trainDataDir)
    }

    @Test
    fun removingARowDropsItsBlock() {
        val form = formOf(
            """
            [[environment.train_data]]
            path = "/data/first"
            repeat = 3

            [[environment.train_data]]
            path = "/data/second"
            """.trimIndent(),
        )
        val one = form.removeTrainDataDir(0)
        assertEquals(listOf("/data/second"), one.trainDataDirs.map { it.path })

        val patched = save(writeConfig().readText(), one)
        assertEquals(1, patched.split("[[environment.train_data]]").size - 1)
        assertTrue(patched.contains("path = \"/data/second\""))
        assertFalse(patched.contains("/data/first"))
        assertEquals("/data/second", reload(patched).environment.trainDataDir)

        // The last row is never removed: training always needs one folder.
        assertEquals(1, one.removeTrainDataDir(0).trainDataDirs.size)
    }

    @Test
    fun appendingARowStartsBlank() {
        val form = formOf().appendTrainDataDir()
        assertEquals(2, form.trainDataDirs.size)
        assertEquals("", form.trainDataDirs[1].path)
        assertEquals("1", form.trainDataDirs[1].repeat)
        // A blank path is a validation error until it is filled in, so a save cannot lose a row.
        assertEquals("Required", form.validate()["${TRAIN_DATA_ERROR_PREFIX}1.path"])
    }

    @Test
    fun rowErrorsAreNamedAndBelongToTheEnvironmentSection() {
        val form = formOf().copy(
            trainDataDirs = listOf(TrainDataDirForm(path = "/data/first", repeat = "0"))
        )
        assertEquals("Min 1", form.validate()["${TRAIN_DATA_ERROR_PREFIX}0.repeat"])
        assertTrue(ConfigSection.Environment.owns("${TRAIN_DATA_ERROR_PREFIX}0.repeat"))
        assertTrue(ConfigSection.Environment.owns("train_data_dir"))
        assertFalse(ConfigSection.Validation.owns("${TRAIN_DATA_ERROR_PREFIX}0.repeat"))
        assertFalse(ConfigSection.Environment.owns("samples.0.repeat"))
        assertEquals("environment.train_data", TRAIN_DATA_SECTION)
    }

    @Test
    fun aRepeatAboveTheRangeIsRefused() {
        val form = formOf().copy(
            trainDataDirs = listOf(TrainDataDirForm(path = "/data/first", repeat = "513"))
        )
        assertEquals("Max 512", form.validate()["${TRAIN_DATA_ERROR_PREFIX}0.repeat"])
    }

    @Test
    fun pickerLabelsNameTheFolderAndItsRepeat() {
        assertEquals("kanae ×3", datasetDirLabel("/home/acite/Character/kanae", 3))
        assertEquals("kanae ×2", datasetDirLabel("/home/acite/Character/kanae/", "2"))
        assertEquals("(no folder) ×1", datasetDirLabel("", 1))
    }

    private companion object {
        val CONFIG = """
            [environment]
            pretrained_model_name_or_path = "/opt/models/sdxl"
            # keep this comment
            output_dir = "/tmp/out"
            logging_dir = "/tmp/logs"
            train_data_dir = "/tmp/data"
            output_name = "haruko"
            TRAIN_DATA_PLACEHOLDER

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
            sample_prompts = "flat prompt"
            sample_negative = "flat negative"
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

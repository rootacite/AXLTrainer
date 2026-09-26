package com.acite.axlranko

import com.acite.axlranko.data.ConfigImporter
import com.acite.axlranko.data.ConfigProfileDisk
import com.acite.axlranko.data.ConfigProfileStore
import com.acite.axlranko.data.TrainerRepo
import com.acite.axlranko.data.loadTrainerConfig
import com.acite.axlranko.model.TrainDataDirForm
import com.acite.axlranko.model.TrainingConfigForm
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import okio.Path.Companion.toPath

/**
 * The `configs/` folder behind Utils -> Profiles: what a name may be, what the list shows, and that
 * a saved profile is a config again - both when a profile is written and when one is applied on top
 * of `config.toml`.
 *
 * Every case works on a temp directory; only the read side touches the repository's own
 * `config.toml`.
 */
class ProfileStoreTest {

    private fun tempDir(prefix: String): File =
        Files.createTempDirectory(prefix).toFile().apply { deleteOnExit() }

    private fun repoConfig(): File =
        assertNotNull(TrainerRepo.configToml(), "no config.toml in the repository")

    private fun repoConfigText(): String = repoConfig().readText()

    private fun repoForm(): TrainingConfigForm =
        TrainingConfigForm.from(
            assertNotNull(loadTrainerConfig(repoConfig().absolutePath.toPath()))
        )

    private fun commentsOf(text: String): List<String> =
        text.lines().filter { it.trimStart().startsWith("#") }

    @Test
    fun namesAreValidated() {
        assertNull(ConfigProfileStore.validateName("Koharu v2"))
        assertNull(ConfigProfileStore.validateName("コハル"))
        assertNotNull(ConfigProfileStore.validateName(""))
        assertNotNull(ConfigProfileStore.validateName("   "))
        assertNotNull(ConfigProfileStore.validateName("a".repeat(65)))
        assertNotNull(ConfigProfileStore.validateName("a/b"))
        assertNotNull(ConfigProfileStore.validateName(".."))
        assertNotNull(ConfigProfileStore.validateName(".hidden"))
        assertNotNull(ConfigProfileStore.validateName("a:b"))
        assertNotNull(ConfigProfileStore.validateName("tab\there"))
    }

    @Test
    fun savingWritesOneFileAndTheListFindsIt() {
        val dir = ConfigProfileDisk.dir(tempDir("axl-configs-save"))
        val saved = ConfigProfileDisk.save(dir, "Koharu v2", "x = 1\n", overwrite = false)
        assertTrue(saved.isSuccess, "save failed: ${saved.exceptionOrNull()}")
        val file = saved.getOrThrow()

        assertEquals("Koharu v2.toml", file.name)
        assertEquals(dir, file.parentFile)
        assertEquals("x = 1\n", file.readText())
        val listed = ConfigProfileDisk.list(dir)
        assertEquals(listOf("Koharu v2"), listed.map { it.name })
        assertEquals("Koharu v2", listed.single().name)
        assertTrue(listed.single().modified > 0L)
        assertEquals("x = 1\n".length.toLong(), listed.single().size)
    }

    @Test
    fun aTakenNameNeedsOverwriteAndReplacesThatSameFile() {
        val dir = ConfigProfileDisk.dir(tempDir("axl-configs-duplicate"))
        ConfigProfileDisk.save(dir, "a", "first\n", overwrite = false).getOrThrow()

        assertTrue(ConfigProfileDisk.save(dir, "a", "second\n", overwrite = false).isFailure)
        assertEquals("first\n", File(dir, "a.toml").readText())

        // The name check is case-insensitive, so "A" has to overwrite the "a.toml" it matched.
        assertTrue(ConfigProfileDisk.save(dir, "A", "second\n", overwrite = false).isFailure)
        assertTrue(ConfigProfileDisk.save(dir, "A", "second\n", overwrite = true).isSuccess)
        assertEquals("second\n", File(dir, "a.toml").readText())
        assertEquals(listOf("a"), ConfigProfileDisk.list(dir).map { it.name })
    }

    @Test
    fun theListShowsProfilesOnly() {
        val dir = ConfigProfileDisk.dir(tempDir("axl-configs-list"))
        assertTrue(dir.mkdirs())
        File(dir, ".gitkeep").writeText("")
        File(dir, "notes.txt").writeText("")
        File(dir, "b.toml").writeText("")
        File(dir, "a.TOML").writeText("")
        assertTrue(File(dir, "nested").mkdir())
        File(dir, "nested/c.toml").writeText("")

        assertEquals(listOf("a", "b"), ConfigProfileDisk.list(dir).map { it.name })
    }

    @Test
    fun deletingRefusesAPathOutsideTheDirectory() {
        val dir = ConfigProfileDisk.dir(tempDir("axl-configs-delete"))
        val outside = Files.createTempFile("axl-not-a-profile", ".toml").toFile().apply {
            deleteOnExit()
            writeText("keep me")
        }

        assertTrue(ConfigProfileDisk.delete(dir, outside.absolutePath).isFailure)
        assertTrue(outside.exists())

        val inside = ConfigProfileDisk.save(dir, "gone", "x = 1\n", overwrite = false).getOrThrow()
        assertTrue(ConfigProfileDisk.delete(dir, inside.absolutePath).isSuccess)
        assertFalse(inside.exists())
        assertTrue(ConfigProfileDisk.list(dir).isEmpty())
    }

    /** A written profile has to decode as the very config the editor was showing. */
    @Test
    fun aWrittenProfileRoundTripsThroughTheDecoder() {
        val appended = repoForm().appendTrainDataDir()
        val form = appended
            .withTrainDataDir(
                appended.trainDataDirs.lastIndex,
                TrainDataDirForm(path = "/tmp/second folder", repeat = "3"),
            )
            .appendSampleSet(0)
        val document = ConfigProfileStore.document(form.toTomlSections(), form.toTomlArrayBlocks())
        val file = Files.createTempFile("axl-profile", ".toml").toFile().apply {
            deleteOnExit()
            writeText(document)
        }

        val decoded = assertNotNull(
            loadTrainerConfig(file.absolutePath.toPath()),
            "the profile does not decode:\n$document",
        )
        assertEquals(form, TrainingConfigForm.from(decoded))
        assertEquals(form.trainDataDirs.size, decoded.environment.trainData.size)
        assertEquals("/tmp/second folder", decoded.environment.trainData.last().path)
        assertEquals(3, decoded.environment.trainData.last().repeat)
        assertEquals(form.sampleSets.size, decoded.validation.samples.size)
    }

    @Test
    fun applyingAPartialProfileTouchesItsKeysOnly() {
        val config = repoConfigText()
        val before = assertNotNull(ConfigImporter.parseConfig(config))

        val merged = ConfigProfileStore
            .applyToConfig(config, "[training]\ntrain_batch_size = 7\n")
            .getOrThrow()

        assertTrue(merged.text.contains("train_batch_size = 7"))
        assertEquals(1, merged.appliedKeys)
        assertTrue(merged.skippedSections.isEmpty())
        // Comments, blank lines and tables the profile says nothing about stay as they were.
        assertEquals(commentsOf(config), commentsOf(merged.text))
        assertTrue(merged.text.contains("[bookkeeping]"))
        assertEquals(before.environment.trainData.size, ConfigImporter.parseConfig(merged.text)!!
            .environment.trainData.size)
        assertEquals(7, assertNotNull(ConfigImporter.parseConfig(merged.text)).training.trainBatchSize)
    }

    @Test
    fun applyingAFullProfileRewritesEverySection() {
        val form = repoForm().copy(outputName = "snapshot", trainBatchSize = "9")
        val document = ConfigProfileStore.document(form.toTomlSections(), form.toTomlArrayBlocks())

        val merged = ConfigProfileStore.applyToConfig(repoConfigText(), document).getOrThrow()
        val decoded = assertNotNull(ConfigImporter.parseConfig(merged.text))

        assertEquals("snapshot", decoded.environment.outputName)
        assertEquals(9, decoded.training.trainBatchSize)
        assertEquals("snapshot", merged.text.substringAfter("output_name = ").substringBefore("\n")
            .trim()
            .trim('"'))
        // Both arrays came from the profile: same entries the editor held, not the file's.
        assertEquals(form.trainDataDirs.size, decoded.environment.trainData.size)
        assertEquals(form.sampleSets.size, decoded.validation.samples.size)
        assertEquals(
            form.trainDataDirs.map { it.path },
            decoded.environment.trainData.map { it.path },
        )
    }

    @Test
    fun aProfileWithoutArrayBlocksLeavesTheTargetsArraysAlone() {
        val config = repoConfigText()
        val before = assertNotNull(ConfigImporter.parseConfig(config))

        val merged = ConfigProfileStore.applyToConfig(config, "[training]\nseed = 4242\n").getOrThrow()
        val after = assertNotNull(ConfigImporter.parseConfig(merged.text))

        assertEquals(4242L, after.training.seed)
        assertEquals(before.environment.trainData.size, after.environment.trainData.size)
        assertEquals(before.validation.samples.size, after.validation.samples.size)
        assertEquals(
            before.validation.samples.map { it.prompt },
            after.validation.samples.map { it.prompt },
        )
    }

    @Test
    fun aSectionTheConfigHasNoTableForIsReported() {
        val merged = ConfigProfileStore
            .applyToConfig(repoConfigText(), "[nope]\nkey = 1\n")
            .getOrThrow()

        assertEquals(listOf("nope"), merged.skippedSections)
        assertFalse(merged.text.contains("[nope]"))
    }

    @Test
    fun aMergeThatWouldNotDecodeIsRefusedBeforeAnythingIsWritten() {
        val config = repoConfigText()
        val result = ConfigProfileStore.applyToConfig(config, "[training]\ntrain_batch_size = \"seven\"\n")

        assertTrue(result.isFailure)
        assertNotNull(result.exceptionOrNull())
    }

    @Test
    fun aProfileWithoutSettingsIsRefused() {
        val result = ConfigProfileStore.applyToConfig(repoConfigText(), "# only a comment\n")
        assertTrue(result.isFailure)
    }
}

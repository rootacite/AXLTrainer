package com.acite.axlranko

import com.acite.axlranko.data.TrainerRepo
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Repo discovery after `config.toml` moved to the repo root: `api.py` alone is a root, `config.toml`
 * only counts together with `trainer/`, and an unrelated `config.toml` must not be mistaken for one.
 */
class TrainerRepoTest {

    private fun tempDir(prefix: String): File =
        Files.createTempDirectory(prefix).toFile().apply { deleteOnExit() }

    private fun dir(parent: File, name: String): File =
        File(parent, name).apply { assertTrue(mkdir(), "could not create $name") }

    private fun file(parent: File, name: String): File =
        File(parent, name).apply { writeText(""); deleteOnExit() }

    @Test
    fun apiPyAloneMarksTheRepoRoot() {
        val root = tempDir("axl-repo-api")
        file(root, "api.py")
        assertTrue(TrainerRepo.looksLikeRepoRoot(root))
    }

    @Test
    fun configPlusTrainerMarksTheRepoRoot() {
        val root = tempDir("axl-repo-config")
        file(root, "config.toml")
        dir(root, "trainer")
        assertTrue(TrainerRepo.looksLikeRepoRoot(root))
    }

    @Test
    fun aLoneConfigTomlIsNotARepoRoot() {
        val root = tempDir("axl-repo-lone-config")
        file(root, "config.toml")
        assertFalse(TrainerRepo.looksLikeRepoRoot(root))
    }

    @Test
    fun aLoneTrainerDirectoryIsNotARepoRoot() {
        val root = tempDir("axl-repo-lone-trainer")
        dir(root, "trainer")
        assertFalse(TrainerRepo.looksLikeRepoRoot(root))
    }

    @Test
    fun configTomlResolvesToTheRepoRootFile() {
        // The test JVM runs from inside the repo (its classes live under ranko/shared/build), so
        // the upward walk has to land on the repo root and find the config that now lives there.
        val root = TrainerRepo.findRoot()
        assertNotNull(root, "repo root not found from the test JVM")
        val toml = TrainerRepo.configToml()
        assertNotNull(toml, "no config.toml under $root")
        assertEquals(root, toml.parentFile)
        assertEquals("config.toml", toml.name)
    }
}

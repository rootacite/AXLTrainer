package com.acite.axlranko.util

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class FileDialogsTest {

    private fun tempDir() = Files.createTempDirectory("axl-dialogs-test").toFile()

    @Test
    fun aDirectoryOpensInItself() {
        val dir = tempDir()
        assertEquals(dir.absolutePath, initialDirectoryFor(dir.absolutePath))
    }

    @Test
    fun aFilePathOpensInItsParent() {
        val dir = tempDir()
        val file = dir.resolve("lllj_s003050.safetensors")
        file.writeBytes(ByteArray(4))
        assertEquals(dir.absolutePath, initialDirectoryFor(file.absolutePath))
    }

    @Test
    fun aMissingPathOpensInItsExistingParent() {
        val dir = tempDir()
        assertEquals(dir.absolutePath, initialDirectoryFor(dir.resolve("not-written-yet.safetensors").absolutePath))
    }

    @Test
    fun aMissingPathWithoutAParentHasNoStartDirectory() {
        assertNull(initialDirectoryFor(""))
        assertNull(initialDirectoryFor("   "))
        assertNull(initialDirectoryFor("/definitely/not/there/axl-absent/child.safetensors"))
    }

    @Test
    fun aTrailingSlashStillCountsAsADirectory() {
        val dir = tempDir()
        assertEquals(dir.absolutePath, initialDirectoryFor("${dir.absolutePath}/"))
    }

    @Test
    fun thePlaceholderTheSaveDialogCreatesIsRemoved() {
        val dir = tempDir()
        val placeholder = dir.resolve("mylora")
        placeholder.writeBytes(ByteArray(0))

        deleteEmptyPlaceholder(placeholder)

        assertFalse(placeholder.exists())
    }

    @Test
    fun aNonEmptyFileIsNeverRemoved() {
        val dir = tempDir()
        val existing = dir.resolve("mylora.safetensors")
        existing.writeBytes(ByteArray(8) { 1 })

        deleteEmptyPlaceholder(existing)

        assertTrue(existing.isFile)
        assertEquals(8L, existing.length())
    }
}

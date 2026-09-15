package com.acite.axlranko.util

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FileCopyTest {

    private fun tempDir() = Files.createTempDirectory("axl-copy-test").toFile()

    @Test
    fun copiesTheWholeFileAndReportsMonotonicProgress() {
        val dir = tempDir()
        val source = dir.resolve("source.safetensors")
        // Larger than one copy chunk so progress has intermediate steps.
        val payload = ByteArray(9 * 1024 * 1024) { (it % 251).toByte() }
        source.writeBytes(payload)

        val progress = mutableListOf<Float>()
        val target = dir.resolve("nested/deeper/target.safetensors")
        val copied = copyFileWithProgress(source, target) { progress += it }

        assertEquals(payload.size.toLong(), copied)
        assertTrue(target.isFile, "target was not written")
        assertContentEquals(payload, target.readBytes())
        assertEquals(1f, progress.last())
        assertTrue(progress.all { it in 0f..1f })
        assertEquals(progress.sorted(), progress, "progress went backwards: $progress")
        assertTrue(progress.size > 1, "expected intermediate progress, got $progress")
        assertEquals(payload.size.toLong(), source.length(), "the source must be untouched")
    }

    @Test
    fun refusesToCopyAFileOntoItself() {
        val dir = tempDir()
        val source = dir.resolve("source.safetensors")
        source.writeBytes(ByteArray(64) { 7 })

        assertFailsWith<IllegalArgumentException> { copyFileWithProgress(source, source) }
        assertEquals(ByteArray(64) { 7 }.size.toLong(), source.length())
        assertContentEquals(ByteArray(64) { 7 }, source.readBytes())
    }

    @Test
    fun refusesAMissingSource() {
        val dir = tempDir()
        assertFailsWith<IllegalArgumentException> {
            copyFileWithProgress(dir.resolve("absent.safetensors"), dir.resolve("target.safetensors"))
        }
    }

    @Test
    fun copiesAnEmptyFile() {
        val dir = tempDir()
        val source = dir.resolve("empty.safetensors")
        source.writeBytes(ByteArray(0))

        val progress = mutableListOf<Float>()
        assertEquals(0L, copyFileWithProgress(source, dir.resolve("copy.safetensors")) { progress += it })
        assertEquals(listOf(1f), progress)
        assertTrue(dir.resolve("copy.safetensors").isFile)
    }
}

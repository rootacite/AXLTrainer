package com.acite.axlranko.util

import java.io.File
import java.io.IOException
import java.nio.file.Files
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "webp", "bmp")

class DatasetShuffleTest {

    private fun tempDir() = Files.createTempDirectory("axl-shuffle-test").toFile()

    /** One sample: the image carries its origin, so a mismatch is detectable after a rename. */
    private fun sample(dir: File, stem: String, extension: String = "png", caption: Boolean = true, mask: Boolean = true) {
        dir.resolve("$stem.$extension").writeText("IMG:$stem")
        if (caption) dir.resolve("$stem.txt").writeText("tags:$stem")
        if (mask) dir.resolve("$stem.mask.png").writeText("MASK:$stem")
    }

    private fun imageFiles(dir: File): List<File> = dir.listFiles().orEmpty().filter {
        it.isFile && !isMaskSidecar(it.name) && it.extension.lowercase() in IMAGE_EXTENSIONS
    }

    /** Every image keeps its caption and its mask: the pairing survives the shuffle. */
    private fun assertPairsIntact(dir: File) {
        val images = imageFiles(dir)
        assertTrue(images.isNotEmpty(), "no images left in ${dir.name}")
        for (image in images) {
            val stem = image.nameWithoutExtension
            val origin = image.readText().removePrefix("IMG:")

            val caption = dir.resolve("$stem.txt")
            assertEquals("tags:$origin", caption.readText(), "caption ${caption.name} left its image")

            val mask = dir.resolve("$stem.mask.png")
            assertEquals("MASK:$origin", mask.readText(), "mask ${mask.name} left its image")
        }
    }

    private fun originByStem(dir: File): Map<String, String> =
        imageFiles(dir).associate { it.nameWithoutExtension to it.readText().removePrefix("IMG:") }

    @Test
    fun maskAndCaptionFollowTheirImage() {
        val dir = tempDir()
        listOf("alpha", "beta", "gamma", "delta", "epsilon", "zeta").forEach { sample(dir, it) }

        val report = shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(7))

        assertEquals(6, report.groups)
        assertEquals(18, report.renamedFiles)
        assertEquals("0001", report.firstStem)
        assertEquals("0006", report.lastStem)
        assertPairsIntact(dir)
        assertEquals(
            setOf("alpha", "beta", "gamma", "delta", "epsilon", "zeta"),
            originByStem(dir).values.toSet(),
            "the shuffle must not lose or duplicate a sample",
        )
    }

    @Test
    fun renumbersToThePaddedSequence() {
        val dir = tempDir()
        listOf("alpha", "beta", "gamma").forEach { sample(dir, it) }

        shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(3))

        val stems = imageFiles(dir).map { it.nameWithoutExtension }.sorted()
        assertEquals(listOf("0001", "0002", "0003"), stems)
        listOf("0001.txt", "0002.txt", "0003.txt", "0001.mask.png", "0002.mask.png", "0003.mask.png").forEach {
            assertTrue(dir.resolve(it).isFile, "$it is missing")
        }
    }

    @Test
    fun theSameSeedReordersTheSameWayAndASeedDoesReorder() {
        val first = tempDir()
        val second = tempDir()
        val stems = listOf("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")
        stems.forEach { sample(first, it); sample(second, it) }

        shuffleAndRenumber(first, IMAGE_EXTENSIONS, Random(11))
        shuffleAndRenumber(second, IMAGE_EXTENSIONS, Random(11))
        assertEquals(originByStem(first), originByStem(second), "a fixed seed must be reproducible")

        val third = tempDir()
        stems.forEach { sample(third, it) }
        shuffleAndRenumber(third, IMAGE_EXTENSIONS, Random(12))
        assertTrue(
            originByStem(first).values.toList() != originByStem(third).values.toList(),
            "two seeds produced the same order",
        )
    }

    @Test
    fun imagesWithoutSidecarsAreShuffledAndGainNone() {
        val dir = tempDir()
        sample(dir, "lonely", caption = false, mask = false)
        sample(dir, "paired")

        shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(5))

        assertEquals(4, dir.listFiles().orEmpty().count { it.isFile }, "3 files for paired + 1 for lonely")

        val lonelyStem = imageFiles(dir).single { it.readText() == "IMG:lonely" }.nameWithoutExtension
        assertFalse(dir.resolve("$lonelyStem.txt").exists(), "a caption appeared for the lonely sample")
        assertFalse(dir.resolve("$lonelyStem.mask.png").exists(), "a mask appeared for the lonely sample")

        val pairedStem = imageFiles(dir).single { it.readText() == "IMG:paired" }.nameWithoutExtension
        assertEquals("tags:paired", dir.resolve("$pairedStem.txt").readText())
        assertEquals("MASK:paired", dir.resolve("$pairedStem.mask.png").readText())
    }

    @Test
    fun directoriesAndForeignFilesAreLeftAlone() {
        val dir = tempDir()
        sample(dir, "alpha")
        val cache = dir.resolve(".latents_cache").apply { mkdirs() }
        cache.resolve("abc.pt").writeText("latent")
        val trash = dir.resolve("trash").apply { mkdirs() }
        trash.resolve("dropped.png").writeText("IMG:dropped")
        dir.resolve("notes.md").writeText("keep me")
        dir.resolve("meta.json").writeText("{}")

        shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(2))

        assertTrue(cache.isDirectory, ".latents_cache was renamed")
        assertEquals("latent", cache.resolve("abc.pt").readText())
        assertTrue(trash.isDirectory, "trash was renamed")
        assertEquals("IMG:dropped", trash.resolve("dropped.png").readText(), "trash contents were touched")
        assertEquals("keep me", dir.resolve("notes.md").readText())
        assertEquals("{}", dir.resolve("meta.json").readText())
        assertPairsIntact(dir)
    }

    @Test
    fun anOrphanCaptionIsRefusedAndNothingMoves() {
        val dir = tempDir()
        sample(dir, "alpha")
        dir.resolve("lonely.txt").writeText("tags:lonely")

        val failure = assertFailsWith<IllegalStateException> { shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(1)) }

        assertTrue(failure.message!!.contains("lonely.txt"), "the message should name the orphan: ${failure.message}")
        assertTrue(dir.resolve("alpha.png").isFile, "alpha.png was renamed before the check")
        assertTrue(dir.resolve("alpha.mask.png").isFile)
        assertTrue(dir.resolve("lonely.txt").isFile)
    }

    @Test
    fun aStrayMaskWithoutItsImageIsLeftAlone() {
        val dir = tempDir()
        sample(dir, "alpha")
        val stray = dir.resolve("ghost.mask.png").apply { writeText("MASK:ghost") }

        shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(4))

        assertEquals("MASK:ghost", stray.readText())
        assertTrue(stray.isFile, "the stray mask was renamed")
        assertPairsIntact(dir)
    }

    @Test
    fun eachMemberKeepsItsOwnExtension() {
        val dir = tempDir()
        sample(dir, "UPPER", extension = "PNG")
        sample(dir, "lower", extension = "jpg")

        shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(6))

        val names = dir.listFiles().orEmpty().map { it.name }.sorted()
        assertTrue(names.any { it.endsWith(".PNG") }, "uppercase extension was not preserved: $names")
        assertTrue(names.any { it.endsWith(".jpg") }, "jpg extension was not preserved: $names")
        assertEquals(2, names.count { it.endsWith(".mask.png") })
        assertEquals(2, names.count { it.endsWith(".txt") })
        // The caption of the uppercase sample stays a plain .txt next to it, whatever the order is.
        val upperMask = dir.listFiles().orEmpty().single { it.name.endsWith(".mask.png") && it.readText() == "MASK:UPPER" }
        val stem = upperMask.name.removeSuffix(".mask.png")
        assertEquals("IMG:UPPER", dir.resolve("$stem.PNG").readText())
        assertEquals("tags:UPPER", dir.resolve("$stem.txt").readText())
    }

    @Test
    fun anAlreadyNumberedFolderShufflesWithoutCollisions() {
        val dir = tempDir()
        listOf("0001", "0002", "0003", "0004").forEach { sample(dir, it) }

        val report = shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(9))

        assertEquals(4, report.groups)
        val stems = originByStem(dir).keys.sorted()
        assertEquals(listOf("0001", "0002", "0003", "0004"), stems)
        assertPairsIntact(dir)
        assertEquals(4, originByStem(dir).values.toSet().size, "a sample was overwritten")
    }

    @Test
    fun aFailedRenameRollsEveryNameBack() {
        val dir = tempDir()
        sample(dir, "alpha")
        sample(dir, "beta")
        // Occupies the first target of the sequence and is not itself a dataset sample: a stray
        // mask whose image is gone. The rename to 0001.mask.png must fail.
        val blocker = dir.resolve("0001.mask.png").apply { writeText("MASK:blocker") }

        val failure = assertFailsWith<IOException> { shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(1)) }

        assertTrue(failure.message!!.contains("0001.mask.png"), "unexpected message: ${failure.message}")
        assertEquals("MASK:blocker", blocker.readText())
        assertTrue(dir.resolve("alpha.png").isFile, "alpha.png was not restored")
        assertTrue(dir.resolve("beta.txt").isFile, "beta.txt was not restored")
        assertTrue(dir.resolve("beta.mask.png").isFile, "beta.mask.png was not restored")
        assertTrue(
            dir.listFiles().orEmpty().none { it.name.startsWith("axl-shuffle-") },
            "temporary names were left behind: ${dir.listFiles()!!.map { it.name }}",
        )
    }

    @Test
    fun anEmptyFolderIsANoOp() {
        val dir = tempDir()

        val report = shuffleAndRenumber(dir, IMAGE_EXTENSIONS, Random(1))

        assertEquals(ShuffleReport(0, 0, "", ""), report)
        assertEquals(0, dir.listFiles()!!.size)
    }

    @Test
    fun planGroupsImagesWithTheirSidecars() {
        val dir = tempDir()
        sample(dir, "alpha")
        sample(dir, "beta", mask = false)
        dir.resolve("notes.md").writeText("x")

        val groups = planDatasetGroups(dir, IMAGE_EXTENSIONS)

        assertEquals(listOf("alpha", "beta"), groups.map { it.stem })
        assertEquals(
            listOf("alpha.png", "alpha.txt", "alpha.mask.png"),
            groups[0].files.map { it.name },
        )
        assertEquals(listOf("beta.png", "beta.txt"), groups[1].files.map { it.name })
        assertNotNull(groups[0].files.firstOrNull { it.name == "alpha.mask.png" })
    }
}

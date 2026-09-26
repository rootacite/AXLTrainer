package com.acite.axlranko.util

import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.util.UUID
import kotlin.random.Random

/**
 * One dataset sample: its image plus the caption and mask sidecars that share that image's stem.
 *
 * A rename moves the whole group and keeps everything after the stem, so `{stem}.txt` and
 * `{stem}.mask.png` can never land on a different image. (`tools/suf.py` split names on the last
 * dot, which made `x.mask.png` a group of its own and shuffled the mask away from its image.)
 */
data class DatasetFileGroup(val stem: String, val files: List<File>) {
    /** The part of a member's name that follows the stem: `.png`, `.txt`, `.mask.png`. */
    fun suffixOf(file: File): String = file.name.removePrefix(stem)
}

/** What a shuffle did, for the status line. */
data class ShuffleReport(
    val groups: Int,
    val renamedFiles: Int,
    val firstStem: String,
    val lastStem: String,
)

private const val TEMP_PREFIX = "axl-shuffle-"
private const val FIRST_STEM = "0001"
private const val STEM_PAD = 4

/** Only the dataset contract is renamed, so any other file keeps its name. */
private fun maskStemOf(file: File): String = file.name.dropLast(MASK_SIDECAR_SUFFIX.length)

private fun isDatasetImage(file: File, imageExtensions: Set<String>): Boolean =
    !isMaskSidecar(file.name) && file.extension.lowercase() in imageExtensions

/**
 * The samples of one dataset folder, in stem order (non-recursive, like the desktop scan).
 *
 * A group is its image(s) plus the same-stem `.txt` caption and `.mask.png` mask when they exist.
 * Directories (`.latents_cache`, `trash`) and files outside the contract are left alone.
 *
 * @throws IllegalStateException when a `.txt` has no image of the same stem: the same integrity
 * fuse the Statistics scan uses. A `.mask.png` whose image is gone is ignored.
 */
fun planDatasetGroups(dir: File, imageExtensions: Set<String>): List<DatasetFileGroup> {
    val files = dir.listFiles()?.filter { it.isFile }.orEmpty()
    val images = files.filter { isDatasetImage(it, imageExtensions) }.sortedBy { it.name }
    val captions = files.filter { it.extension.lowercase() == "txt" }.sortedBy { it.name }
    val masks = files.filter { isMaskSidecar(it.name) }.sortedBy { it.name }

    val stems = images.map { it.nameWithoutExtension }.toSortedSet()
    val orphans = captions.filter { it.nameWithoutExtension !in stems }
    if (orphans.isNotEmpty()) {
        throw IllegalStateException(
            "Refusing to shuffle: isolated tag file(s) without a matching image: " +
                orphans.joinToString(", ") { it.name }
        )
    }

    return stems.map { stem ->
        val members = images.filter { it.nameWithoutExtension == stem } +
            captions.filter { it.nameWithoutExtension == stem } +
            masks.filter { maskStemOf(it) == stem }
        DatasetFileGroup(stem, members)
    }
}

/**
 * Shuffle the samples of [dir] and renumber them to `0001…`, moving each caption and mask with its
 * image.
 *
 * Every group goes through a unique temporary name first, so a folder that is already numbered
 * shuffles without a collision. If a rename fails, the renames already applied are rolled back and
 * the exception is rethrown; a process killed mid-run leaves `axl-shuffle-*` names that the next
 * run treats as ordinary samples, so rerunning recovers.
 */
fun shuffleAndRenumber(
    dir: File,
    imageExtensions: Set<String>,
    random: Random = Random.Default,
): ShuffleReport {
    val order = planDatasetGroups(dir, imageExtensions).shuffled(random)
    if (order.isEmpty()) return ShuffleReport(0, 0, "", "")

    val applied = mutableListOf<Pair<File, File>>()
    try {
        val staged = order.map { group ->
            val temp = TEMP_PREFIX + UUID.randomUUID().toString().take(12)
            group.files.forEach { move(it, File(dir, temp + group.suffixOf(it)), applied) }
            temp to group
        }
        staged.forEachIndexed { index, (temp, group) ->
            val base = (index + 1).toString().padStart(STEM_PAD, '0')
            group.files.forEach { move(File(dir, temp + group.suffixOf(it)), File(dir, base + group.suffixOf(it)), applied) }
        }
    } catch (e: Exception) {
        rollback(applied)
        throw e
    }

    return ShuffleReport(
        groups = order.size,
        renamedFiles = order.sumOf { it.files.size },
        firstStem = FIRST_STEM,
        lastStem = order.size.toString().padStart(STEM_PAD, '0'),
    )
}

private fun move(from: File, to: File, applied: MutableList<Pair<File, File>>) {
    try {
        Files.move(from.toPath(), to.toPath())
    } catch (e: IOException) {
        throw IOException("could not rename ${from.name} to ${to.name}: ${e.message}", e)
    }
    applied += from to to
}

/** Best effort: put the names back the way they were before the failed run. */
private fun rollback(applied: List<Pair<File, File>>) {
    applied.asReversed().forEach { (from, to) -> to.renameTo(from) }
}

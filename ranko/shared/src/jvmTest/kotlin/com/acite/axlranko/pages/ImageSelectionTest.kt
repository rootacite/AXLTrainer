package com.acite.axlranko.pages

import com.acite.axlranko.model.ImageItem
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * The Statistics -> Images jump. A thumbnail there can belong to a dataset folder Images has not
 * loaded (`[[environment.train_data]]` siblings are separate scans), and a lookup that threw used to
 * surface as "Collection contains no element matching the predicate" instead of the jump.
 */
class ImageSelectionTest {

    private fun item(folder: String, stem: String) = ImageItem(
        imagePath = "$folder/$stem.png",
        txtPath = "$folder/$stem.txt",
        tags = "tag",
        maskPath = "$folder/$stem.mask.png",
    )

    private val first = "/data/first"
    private val second = "/data/second"
    private val loaded = listOf(item(first, "a"), item(first, "b"))

    @Test
    fun aThumbnailFromAnotherFolderIsAMissNotACrash() {
        assertNull(findImageByTxtPath(loaded, "$second/c.txt"))
        assertNull(findImageByTxtPath(emptyList(), "$first/a.txt"))
    }

    @Test
    fun aThumbnailInTheLoadedFolderIsFound() {
        assertEquals("$first/b.png", findImageByTxtPath(loaded, "$first/b.txt")?.imagePath)
    }

    @Test
    fun aPendingJumpWinsOverTheOpenImage() {
        val items = listOf(item(first, "a"), item(first, "b"))
        val selected = selectionAfterRescan(
            items,
            pendingTxtPath = "$first/b.txt",
            previousImagePath = "$first/a.png",
        )
        assertEquals("$first/b.png", selected?.imagePath)
    }

    @Test
    fun withoutAPendingJumpTheOpenImageStays() {
        val selected = selectionAfterRescan(
            loaded,
            pendingTxtPath = null,
            previousImagePath = "$first/b.png",
        )
        assertEquals("$first/b.png", selected?.imagePath)
    }

    @Test
    fun theOpenImageGoesWhenItsFolderDid() {
        val rescanned = listOf(item(second, "c"))
        assertNull(selectionAfterRescan(rescanned, pendingTxtPath = null, previousImagePath = "$first/a.png"))
    }

    @Test
    fun aJumpThatIsStillMissingAfterTheRescanSelectsNothing() {
        val rescanned = listOf(item(second, "c"))
        assertNull(
            selectionAfterRescan(rescanned, pendingTxtPath = "$second/gone.txt", previousImagePath = "$first/a.png")
        )
    }

    @Test
    fun aMissingPendingFileKeepsTheOpenImageThatIsStillThere() {
        val selected = selectionAfterRescan(
            loaded,
            pendingTxtPath = "$first/gone.txt",
            previousImagePath = "$first/a.png",
        )
        assertEquals("$first/a.png", selected?.imagePath)
    }
}

package com.acite.axlranko.pages

import com.acite.axlranko.model.ImageItem

/**
 * The item an Images-screen jump selects for [txtPath], or null when this list does not hold it.
 *
 * A Statistics thumbnail can belong to another `[[environment.train_data]]` folder than the one
 * Images has loaded - that list is only rescanned when its tab is entered - so a miss is normal
 * here. Returning null is the point: `List.first { }` used to abort the jump with
 * "Collection contains no element matching the predicate".
 */
internal fun findImageByTxtPath(items: List<ImageItem>, txtPath: String): ImageItem? =
    items.firstOrNull { it.txtPath == txtPath }

/**
 * The image the Images page opens after a rescan: the pending jump's file when one is waiting,
 * otherwise the image that was already open (null once it is gone, e.g. because the folder
 * changed).
 */
internal fun selectionAfterRescan(
    items: List<ImageItem>,
    pendingTxtPath: String?,
    previousImagePath: String?
): ImageItem? = pendingTxtPath?.let { path -> findImageByTxtPath(items, path) }
    ?: items.find { it.imagePath == previousImagePath }

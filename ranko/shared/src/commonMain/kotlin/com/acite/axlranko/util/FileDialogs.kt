package com.acite.axlranko.util

import io.github.vinceglb.filekit.core.FileKit
import io.github.vinceglb.filekit.core.PickerMode
import io.github.vinceglb.filekit.core.PickerType
import java.io.File

/**
 * OS dialogs through FileKit: the XDG desktop portal on Linux (the KDE/GNOME dialog), `IFileDialog`
 * on Windows, `NSOpenPanel` on macOS. Swing's own chooser ignores the desktop theme and looks
 * foreign on every platform. Each function returns an absolute path, or null when canceled.
 */

/** Directory a dialog should open in: the path itself when it is a directory, its parent otherwise. */
fun initialDirectoryFor(current: String): String? {
    if (current.isBlank()) return null
    val path = File(current)
    val directory = if (path.isDirectory) path else path.parentFile
    return directory?.takeIf { it.isDirectory }?.absolutePath
}

suspend fun pickDirectoryDialog(title: String, current: String = ""): String? =
    FileKit.pickDirectory(title = title, initialDirectory = initialDirectoryFor(current))
        ?.path
        ?.takeIf { it.isNotBlank() }

/** [extensions] is a list of bare suffixes; null accepts any file. */
suspend fun pickFileDialog(title: String, current: String = "", extensions: List<String>? = null): String? =
    FileKit.pickFile(
        type = PickerType.File(extensions),
        mode = PickerMode.Single,
        title = title,
        initialDirectory = initialDirectoryFor(current),
    )?.path?.takeIf { it.isNotBlank() }

/**
 * Save dialog seeded with [suggestedName] in [current]'s directory. FileKit creates an empty file at
 * the chosen path, so the caller overwrites rather than creates it; the path comes back exactly as
 * the user typed it, so a caller that needs an extension still has to apply one.
 */
suspend fun saveFileDialog(suggestedName: String, current: String = ""): String? {
    val baseName = suggestedName.substringBeforeLast('.', suggestedName)
    val extension = suggestedName.substringAfterLast('.', "")
    return FileKit.saveFile(
        baseName = baseName.ifBlank { "checkpoint" },
        extension = extension.ifBlank { "safetensors" },
        initialDirectory = initialDirectoryFor(current),
    )?.path?.takeIf { it.isNotBlank() }
}

/**
 * Removes the empty placeholder the native save dialog creates when the extension the caller
 * appends moves the destination to another name.
 */
fun deleteEmptyPlaceholder(file: File) {
    if (file.isFile && file.length() == 0L) file.delete()
}

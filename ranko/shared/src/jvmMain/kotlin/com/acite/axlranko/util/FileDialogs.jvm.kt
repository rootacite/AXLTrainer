package com.acite.axlranko.util

import java.io.File

/** Directory a dialog should open in: the path itself when it is a directory, its parent otherwise. */
fun initialDirectoryFor(current: String): String? {
    if (current.isBlank()) return null
    val path = File(current)
    val directory = if (path.isDirectory) path else path.parentFile
    return directory?.takeIf { it.isDirectory }?.absolutePath
}

fun deleteEmptyPlaceholder(path: String) {
    val file = File(path)
    if (file.isFile && file.length() == 0L) file.delete()
}

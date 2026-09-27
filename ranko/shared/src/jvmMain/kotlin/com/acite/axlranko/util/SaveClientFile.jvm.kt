package com.acite.axlranko.util

import io.github.vinceglb.filekit.core.FileKit

actual suspend fun saveClientFile(
    suggestedName: String,
    bytes: ByteArray,
    mime: String,
): String? {
    val baseName = suggestedName.substringBeforeLast('.', suggestedName).ifBlank { "export" }
    val extension = suggestedName.substringAfterLast('.', "").ifBlank { "bin" }
    val file = FileKit.saveFile(bytes = bytes, baseName = baseName, extension = extension) ?: return null
    return file.path?.takeIf { it.isNotBlank() }
}

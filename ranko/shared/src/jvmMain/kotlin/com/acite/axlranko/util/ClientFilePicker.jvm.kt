package com.acite.axlranko.util

import io.github.vinceglb.filekit.core.FileKit
import io.github.vinceglb.filekit.core.PickerMode
import io.github.vinceglb.filekit.core.PickerType
import java.awt.Desktop
import java.io.File

actual suspend fun pickClientTextFile(title: String, extensions: List<String>): ClientTextFile? {
    val picked = FileKit.pickFile(
        type = PickerType.File(extensions),
        mode = PickerMode.Single,
        title = title,
    ) ?: return null
    val file = File(picked.path ?: return null)
    if (!file.isFile) return null
    return ClientTextFile(name = file.name, text = file.readText(Charsets.UTF_8))
}

actual fun openLocalDirectory(path: String): Boolean {
    val target = File(path)
    if (!target.exists()) return false
    return try {
        if (!Desktop.isDesktopSupported()) return false
        val desktop = Desktop.getDesktop()
        if (target.isDirectory) desktop.open(target) else desktop.open(target.parentFile)
        true
    } catch (_: Exception) {
        false
    }
}

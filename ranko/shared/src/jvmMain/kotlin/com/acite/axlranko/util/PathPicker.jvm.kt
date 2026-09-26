package com.acite.axlranko.util

import androidx.compose.runtime.Composable
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesBinding
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import io.github.vinceglb.filekit.core.FileKit
import io.github.vinceglb.filekit.core.PickerMode
import io.github.vinceglb.filekit.core.PickerType
import java.io.File

@SingleIn(AppScope::class)
@ContributesBinding(AppScope::class)
@Inject
class JvmPathPicker : PathPicker {
    override suspend fun pickDirectory(title: String, current: String): String? =
        FileKit.pickDirectory(title = title, initialDirectory = initialDirectoryFor(current))
            ?.path
            ?.takeIf { it.isNotBlank() }

    override suspend fun pickFile(title: String, current: String, extensions: List<String>?): String? =
        FileKit.pickFile(
            type = PickerType.File(extensions),
            mode = PickerMode.Single,
            title = title,
            initialDirectory = initialDirectoryFor(current),
        )?.path?.takeIf { it.isNotBlank() }

    override suspend fun saveFile(suggestedName: String, current: String): String? {
        val baseName = suggestedName.substringBeforeLast('.', suggestedName)
        val extension = suggestedName.substringAfterLast('.', "")
        return FileKit.saveFile(
            baseName = baseName.ifBlank { "checkpoint" },
            extension = extension.ifBlank { "safetensors" },
            initialDirectory = initialDirectoryFor(current),
        )?.path?.takeIf { it.isNotBlank() }
    }

    override fun deleteEmptyPlaceholder(path: String) {
        val file = File(path)
        if (file.isFile && file.length() == 0L) file.delete()
    }
}

@Composable
actual fun InstallPathPickerHost(picker: PathPicker) {
}

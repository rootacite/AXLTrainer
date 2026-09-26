package com.acite.axlranko.util

import androidx.compose.runtime.Composable

interface PathPicker {
    suspend fun pickDirectory(title: String, current: String = ""): String?
    suspend fun pickFile(title: String, current: String = "", extensions: List<String>? = null): String?
    suspend fun saveFile(suggestedName: String, current: String = ""): String?
    fun deleteEmptyPlaceholder(path: String) {}
}

@Composable
expect fun InstallPathPickerHost(picker: PathPicker)

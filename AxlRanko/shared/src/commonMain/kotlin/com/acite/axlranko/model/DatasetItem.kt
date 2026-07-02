package com.acite.axlranko.model

import java.io.File

data class DatasetItem(
    val txtFile: File,
    val imageFile: File,
    val tags: List<String>
)
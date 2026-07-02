package com.acite.axlranko.model

data class ImageScreenState(
    val dataDir: String = "",
    val imageItems: List<ImageItem> = emptyList(),
    val selectedItem: ImageItem? = null,
    val editorText: String = "",
    val leftWeight: Float = 0.18f,
    val topWeight: Float = 0.75f
)
package com.acite.axlranko.model

data class ImageScreenState(
    val dataDir: String = "",
    val imageItems: List<ImageItem> = emptyList(),
    val selectedItem: ImageItem? = null,
    val editorText: String = "",
    val leftWeight: Float = 0.18f,
    val topWeight: Float = 0.75f,
    val maskEditEnabled: Boolean = false,
    val maskOnly: Boolean = false,
    val brushRadiusPx: Float = 24f,
    val brushFeather: Float = 0.20f,
    val brushStrength: Float = 1.0f,
    val maskDirty: Boolean = false,
    val maskPreviewRevision: Int = 0,
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
)
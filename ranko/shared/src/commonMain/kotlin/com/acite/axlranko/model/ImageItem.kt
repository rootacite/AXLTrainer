package com.acite.axlranko.model

data class ImageItem(
    val directory: String,
    val stem: String,
    val imagePath: String,
    val txtPath: String,
    val tags: String,
    val draftTags: String? = null,
    val maskPath: String,
    val hasSidecarMask: Boolean = false,
    val hasAlpha: Boolean = false,
    val width: Int = 0,
    val height: Int = 0,
) {
    val hasMask: Boolean
        get() = hasSidecarMask || hasAlpha
    val isDirty: Boolean
        get() = draftTags != null && draftTags != tags

    val currentTags: String
        get() = draftTags ?: tags
}
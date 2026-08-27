package com.acite.axlranko.model

data class ImageItem(
    val imagePath: String,
    val txtPath: String,
    val tags: String,
    val draftTags: String? = null
) {
    val isDirty: Boolean
        get() = draftTags != null && draftTags != tags

    val currentTags: String
        get() = draftTags ?: tags
}
package com.acite.axlranko.model

data class DatasetItem(
    val stem: String,
    val imagePath: String,
    val txtPath: String,
    val maskPath: String,
    val tags: List<String>,
    val width: Int = 0,
    val height: Int = 0,
)

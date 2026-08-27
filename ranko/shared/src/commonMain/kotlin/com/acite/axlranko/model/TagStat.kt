package com.acite.axlranko.model

data class TagStat(
    val tag: String,
    val count: Int,
    val frequency: Float // 0.0 ~ 100.0
)
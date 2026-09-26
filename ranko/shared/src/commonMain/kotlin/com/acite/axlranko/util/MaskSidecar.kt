package com.acite.axlranko.util

const val MASK_SIDECAR_SUFFIX = ".mask.png"

fun isMaskSidecar(name: String): Boolean =
    name.endsWith(MASK_SIDECAR_SUFFIX, ignoreCase = true)

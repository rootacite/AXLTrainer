package com.acite.axlranko.util

import androidx.compose.ui.graphics.ImageBitmap

data class RgbaImage(val width: Int, val height: Int, val argb: IntArray)

expect object ImageCodecs {
    fun decodeRgba(bytes: ByteArray): RgbaImage?

    fun encodeGrayPng(width: Int, height: Int, gray: ByteArray): ByteArray

    fun encodeJpeg(width: Int, height: Int, argb: IntArray, quality: Int): ByteArray

    fun argbToImageBitmap(width: Int, height: Int, argb: IntArray): ImageBitmap
}

fun RgbaImage.fitMaxEdge(maxEdge: Int): RgbaImage {
    val longest = maxOf(width, height)
    if (longest <= maxEdge || maxEdge <= 0) return this
    val dstW = (width * maxEdge / longest).coerceAtLeast(1)
    val dstH = (height * maxEdge / longest).coerceAtLeast(1)
    val out = IntArray(dstW * dstH)
    for (y in 0 until dstH) {
        val sy = y * height / dstH
        val srcRow = sy * width
        val dstRow = y * dstW
        for (x in 0 until dstW) {
            out[dstRow + x] = argb[srcRow + (x * width / dstW)]
        }
    }
    return RgbaImage(dstW, dstH, out)
}

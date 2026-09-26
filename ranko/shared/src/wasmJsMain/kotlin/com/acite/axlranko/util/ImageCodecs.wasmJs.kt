package com.acite.axlranko.util

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import org.jetbrains.skia.Bitmap
import org.jetbrains.skia.ColorAlphaType
import org.jetbrains.skia.ColorType
import org.jetbrains.skia.EncodedImageFormat
import org.jetbrains.skia.Image
import org.jetbrains.skia.ImageInfo

actual object ImageCodecs {
    actual fun decodeRgba(bytes: ByteArray): RgbaImage? {
        val image = Image.makeFromEncoded(bytes) ?: return null
        val w = image.width
        val h = image.height
        val info = ImageInfo(w, h, ColorType.BGRA_8888, ColorAlphaType.UNPREMUL)
        val bitmap = Bitmap()
        if (!bitmap.allocPixels(info)) return null
        if (!image.readPixels(bitmap)) return null
        val raw = bitmap.readPixels() ?: return null
        val argb = IntArray(w * h)
        var i = 0
        var p = 0
        while (i < argb.size) {
            val b = raw[p].toInt() and 0xFF
            val g = raw[p + 1].toInt() and 0xFF
            val r = raw[p + 2].toInt() and 0xFF
            val a = raw[p + 3].toInt() and 0xFF
            argb[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
            i++
            p += 4
        }
        return RgbaImage(w, h, argb)
    }

    actual fun encodeJpeg(width: Int, height: Int, argb: IntArray, quality: Int): ByteArray {
        val raw = ByteArray(width * height * 4)
        var d = 0
        for (c in argb) {
            raw[d] = (c and 0xFF).toByte()
            raw[d + 1] = ((c shr 8) and 0xFF).toByte()
            raw[d + 2] = ((c shr 16) and 0xFF).toByte()
            raw[d + 3] = 0xFF.toByte()
            d += 4
        }
        val info = ImageInfo(width, height, ColorType.BGRA_8888, ColorAlphaType.OPAQUE)
        val image = Image.makeRaster(info, raw, width * 4)
        val data = image.encodeToData(EncodedImageFormat.JPEG, quality.coerceIn(1, 100))
            ?: error("JPEG encode failed")
        return data.bytes
    }

    actual fun encodeGrayPng(width: Int, height: Int, gray: ByteArray): ByteArray {
        val rgba = ByteArray(width * height * 4)
        var s = 0
        var d = 0
        while (s < gray.size) {
            val v = gray[s]
            rgba[d] = v
            rgba[d + 1] = v
            rgba[d + 2] = v
            rgba[d + 3] = 0xFF.toByte()
            s++
            d += 4
        }
        val info = ImageInfo(width, height, ColorType.RGBA_8888, ColorAlphaType.UNPREMUL)
        val image = Image.makeRaster(info, rgba, width * 4)
        val data = image.encodeToData(EncodedImageFormat.PNG) ?: error("PNG encode failed")
        return data.bytes
    }

    actual fun argbToImageBitmap(width: Int, height: Int, argb: IntArray): ImageBitmap {
        val raw = ByteArray(width * height * 4)
        var d = 0
        for (c in argb) {
            raw[d] = (c and 0xFF).toByte()
            raw[d + 1] = ((c shr 8) and 0xFF).toByte()
            raw[d + 2] = ((c shr 16) and 0xFF).toByte()
            raw[d + 3] = ((c ushr 24) and 0xFF).toByte()
            d += 4
        }
        val info = ImageInfo(width, height, ColorType.BGRA_8888, ColorAlphaType.UNPREMUL)
        val image = Image.makeRaster(info, raw, width * 4)
        return image.toComposeImageBitmap()
    }
}

package com.acite.axlranko.util

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

actual object ImageCodecs {
    actual fun decodeRgba(bytes: ByteArray): RgbaImage? {
        val src = ImageIO.read(ByteArrayInputStream(bytes)) ?: return null
        val w = src.width
        val h = src.height
        val argb = IntArray(w * h)
        val rgb = if (src.type == BufferedImage.TYPE_INT_ARGB || src.type == BufferedImage.TYPE_INT_RGB) {
            src
        } else {
            BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB).also { out ->
                val g = out.createGraphics()
                g.drawImage(src, 0, 0, null)
                g.dispose()
            }
        }
        rgb.getRGB(0, 0, w, h, argb, 0, w)
        return RgbaImage(w, h, argb)
    }

    actual fun encodeJpeg(width: Int, height: Int, argb: IntArray, quality: Int): ByteArray {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        image.setRGB(0, 0, width, height, argb, 0, width)
        val buf = ByteArrayOutputStream()
        val writer = ImageIO.getImageWritersByFormatName("jpeg").next()
        val params = writer.defaultWriteParam
        if (params.canWriteCompressed()) {
            params.compressionMode = javax.imageio.ImageWriteParam.MODE_EXPLICIT
            params.compressionQuality = (quality.coerceIn(1, 100) / 100f)
        }
        writer.output = javax.imageio.stream.MemoryCacheImageOutputStream(buf)
        writer.write(null, javax.imageio.IIOImage(image, null, null), params)
        writer.dispose()
        return buf.toByteArray()
    }

    actual fun encodeGrayPng(width: Int, height: Int, gray: ByteArray): ByteArray {
        val image = BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
        val raster = image.raster
        var i = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                raster.setSample(x, y, 0, gray[i].toInt() and 0xFF)
                i++
            }
        }
        val buf = ByteArrayOutputStream()
        ImageIO.write(image, "png", buf)
        return buf.toByteArray()
    }

    actual fun argbToImageBitmap(width: Int, height: Int, argb: IntArray): ImageBitmap {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        image.setRGB(0, 0, width, height, argb, 0, width)
        return image.toComposeImageBitmap()
    }
}

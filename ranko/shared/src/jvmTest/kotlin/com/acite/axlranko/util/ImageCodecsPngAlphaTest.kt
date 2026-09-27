package com.acite.axlranko.util

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class ImageCodecsPngAlphaTest {
    @Test
    fun pngRgbaAlphaSurvivesDecodeAndBecomesTheMask() {
        val w = 16
        val h = 8
        val src = BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val a = if (x < w / 2) 255 else 0
                src.setRGB(x, y, (a shl 24) or 0x002844)
            }
        }
        val buf = ByteArrayOutputStream()
        ImageIO.write(src, "png", buf)
        val decoded = ImageCodecs.decodeRgba(buf.toByteArray())
        assertNotNull(decoded)
        assertEquals(w, decoded.width)
        assertEquals(h, decoded.height)
        assertEquals(255, (decoded.argb[4] ushr 24) and 0xFF)
        assertEquals(0, (decoded.argb[w - 1] ushr 24) and 0xFF)

        val mask = MaskCanvas.fromAlphaRgba(decoded.argb, decoded.width, decoded.height, decoded.width, decoded.height)
        assertEquals(255, mask.pixel(4, 4))
        assertEquals(0, mask.pixel(w - 1, 4))
    }
}

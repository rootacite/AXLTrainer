package com.acite.axlranko.util

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MaskSidecarTest {
    @Test
    fun detectsSidecarNames() {
        assertTrue(isMaskSidecar("x.mask.png"))
        assertTrue(isMaskSidecar("photo.MASK.PNG"))
        assertFalse(isMaskSidecar("x.png"))
        assertFalse(isMaskSidecar("mask.png"))
    }

    @Test
    fun maskPathUsesStem() {
        assertEquals(
            File("/data/cat.mask.png"),
            maskFileFor(File("/data/cat.jpg")),
        )
        assertEquals(
            File("/data/a.mask.png"),
            maskFileFor(File("/data/a.png")),
        )
    }

    @Test
    fun detectsPngRgbaColorType() {
        val rgba = File.createTempFile("alpha-", ".png")
        val rgb = File.createTempFile("opaque-", ".png")
        try {
            rgba.writeBytes(pngIhdr(colorType = 6))
            rgb.writeBytes(pngIhdr(colorType = 2))
            assertTrue(fileHasAlphaChannel(rgba))
            assertFalse(fileHasAlphaChannel(rgb))
        } finally {
            rgba.delete()
            rgb.delete()
        }
    }

    private fun pngIhdr(colorType: Int): ByteArray {
        val out = ArrayList<Byte>(33)
        out += listOf(0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A).map { it.toByte() }
        out += listOf(0, 0, 0, 13).map { it.toByte() }
        out += "IHDR".toByteArray().toList()
        out += listOf(0, 0, 0, 8, 0, 0, 0, 8).map { it.toByte() }
        out += listOf(8, colorType.toByte(), 0, 0, 0)
        return out.toByteArray()
    }
}

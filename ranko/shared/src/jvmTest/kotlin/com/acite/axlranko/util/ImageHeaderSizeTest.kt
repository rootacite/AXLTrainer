package com.acite.axlranko.util

import java.io.File
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ImageHeaderSizeTest {

    private val tempFiles = mutableListOf<File>()

    @AfterTest
    fun tearDown() {
        ImageHeaderSize.clearCache()
        tempFiles.forEach { it.delete() }
        tempFiles.clear()
    }

    @Test
    fun readsPngDimensionsFromIhdr() {
        val file = tempFile(
            "sample.png",
            pngHeader(width = 320, height = 240)
        )
        assertEquals(ImageHeaderSize.PixelSize(320, 240), ImageHeaderSize.size(file))
        assertEquals(320f / 240f, ImageHeaderSize.aspectRatio(file))
    }

    @Test
    fun readsJpegDimensionsFromSof0() {
        val file = tempFile("sample.jpg", jpegSof(width = 1280, height = 720, orientation = null))
        assertEquals(ImageHeaderSize.PixelSize(1280, 720), ImageHeaderSize.size(file))
    }

    @Test
    fun swapsJpegDimensionsForExifOrientation6() {
        val file = tempFile(
            "rotated.jpg",
            jpegSof(width = 1920, height = 1080, orientation = 6)
        )
        assertEquals(ImageHeaderSize.PixelSize(1080, 1920), ImageHeaderSize.size(file))
    }

    @Test
    fun readsBmpDimensions() {
        val file = tempFile("sample.bmp", bmpHeader(width = 10, height = 20))
        assertEquals(ImageHeaderSize.PixelSize(10, 20), ImageHeaderSize.size(file))
    }

    @Test
    fun readsWebpVp8xCanvasSize() {
        val file = tempFile("sample.webp", webpVp8x(width = 100, height = 50))
        assertEquals(ImageHeaderSize.PixelSize(100, 50), ImageHeaderSize.size(file))
    }

    @Test
    fun returnsNullForUnknownBytes() {
        val file = tempFile("notes.txt", "not an image".toByteArray())
        assertNull(ImageHeaderSize.size(file))
    }

    private fun tempFile(name: String, bytes: ByteArray): File {
        val file = File.createTempFile("img-header-", "-$name")
        file.writeBytes(bytes)
        tempFiles += file
        return file
    }

    private fun pngHeader(width: Int, height: Int): ByteArray {
        val out = ArrayList<Byte>(24)
        out += listOf(0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A).map { it.toByte() }
        out += be32(13)
        out += "IHDR".toByteArray().toList()
        out += be32(width)
        out += be32(height)
        out += listOf(8, 2, 0, 0, 0).map { it.toByte() }
        return out.toByteArray()
    }

    private fun jpegSof(width: Int, height: Int, orientation: Int?): ByteArray {
        val out = ArrayList<Byte>()
        out += listOf(0xFF, 0xD8).map { it.toByte() }
        if (orientation != null) {
            val exif = jpegExifApp1(orientation)
            out += exif.toList()
        }
        out += listOf(0xFF, 0xC0).map { it.toByte() }
        out += be16(11)
        out += 8.toByte()
        out += be16(height)
        out += be16(width)
        out += 3.toByte()
        out += listOf(0xFF, 0xD9).map { it.toByte() }
        return out.toByteArray()
    }

    private fun jpegExifApp1(orientation: Int): ByteArray {
        // TIFF IFD0 with a single Orientation tag (0x0112), little-endian.
        val tiff = byteArrayOf(
            'I'.code.toByte(), 'I'.code.toByte(), 0x2A, 0x00,
            0x08, 0x00, 0x00, 0x00,
            0x01, 0x00,
            0x12, 0x01,
            0x03, 0x00,
            0x01, 0x00, 0x00, 0x00,
            orientation.toByte(), 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
        )
        val payload = byteArrayOf(
            'E'.code.toByte(), 'x'.code.toByte(), 'i'.code.toByte(), 'f'.code.toByte(), 0, 0
        ) + tiff
        val length = payload.size + 2
        return byteArrayOf(0xFF.toByte(), 0xE1.toByte()) + be16(length) + payload
    }

    private fun bmpHeader(width: Int, height: Int): ByteArray {
        val bytes = ByteArray(26)
        bytes[0] = 'B'.code.toByte()
        bytes[1] = 'M'.code.toByte()
        writeI32Le(bytes, 14, 40)
        writeI32Le(bytes, 18, width)
        writeI32Le(bytes, 22, height)
        return bytes
    }

    private fun webpVp8x(width: Int, height: Int): ByteArray {
        val payload = ByteArray(10)
        val w = width - 1
        val h = height - 1
        payload[4] = (w and 0xFF).toByte()
        payload[5] = ((w shr 8) and 0xFF).toByte()
        payload[6] = ((w shr 16) and 0xFF).toByte()
        payload[7] = (h and 0xFF).toByte()
        payload[8] = ((h shr 8) and 0xFF).toByte()
        payload[9] = ((h shr 16) and 0xFF).toByte()
        val chunk = "VP8X".toByteArray() + le32(payload.size) + payload
        val riffSize = 4 + chunk.size
        return "RIFF".toByteArray() + le32(riffSize) + "WEBP".toByteArray() + chunk
    }

    private fun be16(value: Int): List<Byte> =
        listOf(((value shr 8) and 0xFF).toByte(), (value and 0xFF).toByte())

    private fun be32(value: Int): List<Byte> =
        listOf(
            ((value shr 24) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            (value and 0xFF).toByte()
        )

    private fun le32(value: Int): ByteArray =
        byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 24) and 0xFF).toByte()
        )

    private fun writeI32Le(bytes: ByteArray, offset: Int, value: Int) {
        bytes[offset] = (value and 0xFF).toByte()
        bytes[offset + 1] = ((value shr 8) and 0xFF).toByte()
        bytes[offset + 2] = ((value shr 16) and 0xFF).toByte()
        bytes[offset + 3] = ((value shr 24) and 0xFF).toByte()
    }
}

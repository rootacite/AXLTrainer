package com.acite.axlranko.util

import java.io.DataInputStream
import java.io.EOFException
import java.io.File
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap

/**
 * Reads pixel size from image file headers without decoding pixel data.
 * Used so Lazy lists can lock item aspect ratio before Coil loads the bitmap.
 */
object ImageHeaderSize {
    data class PixelSize(val width: Int, val height: Int) {
        val aspectRatio: Float
            get() = width.toFloat() / height.toFloat()
    }

    private val cache = ConcurrentHashMap<String, PixelSize>()

    fun size(file: File): PixelSize? {
        if (!file.isFile) return null
        val key = cacheKey(file)
        cache[key]?.let { return it }
        val parsed = try {
            parse(file)
        } catch (_: IOException) {
            null
        } catch (_: EOFException) {
            null
        }
        if (parsed != null && parsed.width > 0 && parsed.height > 0) {
            cache[key] = parsed
            return parsed
        }
        return null
    }

    fun aspectRatio(file: File): Float? = size(file)?.aspectRatio

    internal fun clearCache() {
        cache.clear()
    }

    private fun cacheKey(file: File): String =
        "${file.absolutePath}:${file.length()}:${file.lastModified()}"

    private fun parse(file: File): PixelSize? {
        DataInputStream(file.inputStream().buffered()).use { input ->
            val magic = ByteArray(12)
            val n = input.read(magic)
            if (n < 12) return null
            return when {
                magic[0] == 0xFF.toByte() && magic[1] == 0xD8.toByte() ->
                    parseJpeg(file)
                isPng(magic) -> parsePng(magic, input)
                magic[0] == 'B'.code.toByte() && magic[1] == 'M'.code.toByte() ->
                    parseBmp(magic, input)
                isRiffWebp(magic) -> parseWebp(file)
                else -> null
            }
        }
    }

    private fun isPng(magic: ByteArray): Boolean =
        magic.size >= 8 &&
            magic[0] == 0x89.toByte() &&
            magic[1] == 'P'.code.toByte() &&
            magic[2] == 'N'.code.toByte() &&
            magic[3] == 'G'.code.toByte() &&
            magic[4] == 0x0D.toByte() &&
            magic[5] == 0x0A.toByte() &&
            magic[6] == 0x1A.toByte() &&
            magic[7] == 0x0A.toByte()

    private fun isRiffWebp(magic: ByteArray): Boolean =
        magic.size >= 12 &&
            magic[0] == 'R'.code.toByte() &&
            magic[1] == 'I'.code.toByte() &&
            magic[2] == 'F'.code.toByte() &&
            magic[3] == 'F'.code.toByte() &&
            magic[8] == 'W'.code.toByte() &&
            magic[9] == 'E'.code.toByte() &&
            magic[10] == 'B'.code.toByte() &&
            magic[11] == 'P'.code.toByte()

    private fun parsePng(magic: ByteArray, input: DataInputStream): PixelSize? {
        // Signature (8) + IHDR length (4) already in the 12-byte peek; remaining IHDR follows.
        val ihdrLen = ((magic[8].toInt() and 0xFF) shl 24) or
            ((magic[9].toInt() and 0xFF) shl 16) or
            ((magic[10].toInt() and 0xFF) shl 8) or
            (magic[11].toInt() and 0xFF)
        if (ihdrLen < 8) return null
        val ihdrType = ByteArray(4)
        input.readFully(ihdrType)
        if (String(ihdrType, Charsets.US_ASCII) != "IHDR") return null
        val width = input.readInt()
        val height = input.readInt()
        return validSize(width, height)
    }

    private fun parseBmp(magic: ByteArray, input: DataInputStream): PixelSize? {
        // BMP info header starts at offset 14. We already consumed 12 bytes.
        input.skipFully(2)
        val headerSize = readI32Le(input)
        if (headerSize < 16) return null
        val width = readI32Le(input)
        val height = readI32Le(input)
        return validSize(width, kotlin.math.abs(height))
    }

    private fun parseJpeg(file: File): PixelSize? {
        DataInputStream(file.inputStream().buffered()).use { input ->
            if (input.readUnsignedShort() != 0xFFD8) return null
            var orientation = 1
            while (true) {
                var b = input.readUnsignedByte()
                while (b == 0xFF) {
                    b = input.readUnsignedByte()
                }
                when (b) {
                    0xD9, 0xDA -> return null
                    in 0xD0..0xD7, 0x01 -> continue
                    in 0xC0..0xC3, in 0xC5..0xC7, in 0xC9..0xCB, in 0xCD..0xCF -> {
                        val length = input.readUnsignedShort()
                        if (length < 7) return null
                        input.readUnsignedByte()
                        val height = input.readUnsignedShort()
                        val width = input.readUnsignedShort()
                        return applyExifOrientation(width, height, orientation)
                    }
                    0xE1 -> {
                        val length = input.readUnsignedShort()
                        if (length < 2) return null
                        val payloadLen = length - 2
                        val payload = ByteArray(payloadLen)
                        input.readFully(payload)
                        parseExifOrientation(payload)?.let { orientation = it }
                    }
                    else -> {
                        val length = input.readUnsignedShort()
                        if (length < 2) return null
                        input.skipFully(length - 2)
                    }
                }
            }
        }
        return null
    }

    private fun parseWebp(file: File): PixelSize? {
        DataInputStream(file.inputStream().buffered()).use { input ->
            val riff = ByteArray(12)
            input.readFully(riff)
            if (!isRiffWebp(riff)) return null
            while (true) {
                val fourcc = ByteArray(4)
                val read = input.read(fourcc)
                if (read < 4) return null
                val chunkSize = readU32Le(input).toInt()
                if (chunkSize < 0) return null
                val tag = String(fourcc, Charsets.US_ASCII)
                when (tag) {
                    "VP8X" -> {
                        if (chunkSize < 10) return null
                        val payload = ByteArray(10)
                        input.readFully(payload)
                        val width = 1 + (payload[4].toInt() and 0xFF) +
                            ((payload[5].toInt() and 0xFF) shl 8) +
                            ((payload[6].toInt() and 0xFF) shl 16)
                        val height = 1 + (payload[7].toInt() and 0xFF) +
                            ((payload[8].toInt() and 0xFF) shl 8) +
                            ((payload[9].toInt() and 0xFF) shl 16)
                        return validSize(width, height)
                    }
                    "VP8 " -> {
                        if (chunkSize < 10) return null
                        val payload = ByteArray(10)
                        input.readFully(payload)
                        if (payload[3] != 0x9D.toByte() || payload[4] != 0x01.toByte() || payload[5] != 0x2A.toByte()) {
                            return null
                        }
                        val width = (payload[6].toInt() and 0xFF) or ((payload[7].toInt() and 0x3F) shl 8)
                        val height = (payload[8].toInt() and 0xFF) or ((payload[9].toInt() and 0x3F) shl 8)
                        return validSize(width, height)
                    }
                    "VP8L" -> {
                        if (chunkSize < 5) return null
                        val payload = ByteArray(5)
                        input.readFully(payload)
                        if (payload[0] != 0x2F.toByte()) return null
                        val bits = (payload[1].toInt() and 0xFF) or
                            ((payload[2].toInt() and 0xFF) shl 8) or
                            ((payload[3].toInt() and 0xFF) shl 16) or
                            ((payload[4].toInt() and 0xFF) shl 24)
                        val width = (bits and 0x3FFF) + 1
                        val height = ((bits shr 14) and 0x3FFF) + 1
                        return validSize(width, height)
                    }
                    else -> {
                        input.skipFully(chunkSize)
                        if (chunkSize % 2 == 1) {
                            input.skipFully(1)
                        }
                    }
                }
            }
        }
        return null
    }

    private fun parseExifOrientation(payload: ByteArray): Int? {
        if (payload.size < 14) return null
        if (payload[0] != 'E'.code.toByte() ||
            payload[1] != 'x'.code.toByte() ||
            payload[2] != 'i'.code.toByte() ||
            payload[3] != 'f'.code.toByte() ||
            payload[4] != 0.toByte() ||
            payload[5] != 0.toByte()
        ) {
            return null
        }
        val tiff = payload.copyOfRange(6, payload.size)
        if (tiff.size < 8) return null
        val le = tiff[0] == 'I'.code.toByte() && tiff[1] == 'I'.code.toByte()
        val be = tiff[0] == 'M'.code.toByte() && tiff[1] == 'M'.code.toByte()
        if (!le && !be) return null
        val fortyTwo = u16(tiff, 2, le)
        if (fortyTwo != 42) return null
        val ifd0Long = u32(tiff, 4, le)
        if (ifd0Long < 0L || ifd0Long + 2L > tiff.size.toLong()) return null
        val ifd0 = ifd0Long.toInt()
        val count = u16(tiff, ifd0, le)
        var offset = ifd0 + 2
        repeat(count) {
            if (offset + 12 > tiff.size) return null
            val tag = u16(tiff, offset, le)
            val type = u16(tiff, offset + 2, le)
            val componentCount = u32(tiff, offset + 4, le)
            if (tag == 0x0112 && type == 3 && componentCount == 1L) {
                val value = u16(tiff, offset + 8, le)
                return value.takeIf { it in 1..8 }
            }
            offset += 12
        }
        return null
    }

    private fun applyExifOrientation(width: Int, height: Int, orientation: Int): PixelSize? {
        val swapped = orientation in 5..8
        return if (swapped) validSize(height, width) else validSize(width, height)
    }

    private fun validSize(width: Int, height: Int): PixelSize? {
        if (width <= 0 || height <= 0) return null
        return PixelSize(width, height)
    }

    private fun u16(data: ByteArray, offset: Int, le: Boolean): Int {
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        return if (le) b0 or (b1 shl 8) else (b0 shl 8) or b1
    }

    private fun u32(data: ByteArray, offset: Int, le: Boolean): Long {
        val b0 = (data[offset].toInt() and 0xFF).toLong()
        val b1 = (data[offset + 1].toInt() and 0xFF).toLong()
        val b2 = (data[offset + 2].toInt() and 0xFF).toLong()
        val b3 = (data[offset + 3].toInt() and 0xFF).toLong()
        return if (le) {
            b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
        } else {
            (b0 shl 24) or (b1 shl 16) or (b2 shl 8) or b3
        }
    }

    private fun readI32Le(input: DataInputStream): Int {
        val b0 = input.readUnsignedByte()
        val b1 = input.readUnsignedByte()
        val b2 = input.readUnsignedByte()
        val b3 = input.readUnsignedByte()
        return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }

    private fun readU32Le(input: DataInputStream): Long {
        val b0 = input.readUnsignedByte().toLong()
        val b1 = input.readUnsignedByte().toLong()
        val b2 = input.readUnsignedByte().toLong()
        val b3 = input.readUnsignedByte().toLong()
        return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }

    private fun DataInputStream.skipFully(n: Int) {
        var remaining = n
        while (remaining > 0) {
            val skipped = skipBytes(remaining)
            if (skipped <= 0) {
                if (read() < 0) throw EOFException()
                remaining -= 1
            } else {
                remaining -= skipped
            }
        }
    }
}

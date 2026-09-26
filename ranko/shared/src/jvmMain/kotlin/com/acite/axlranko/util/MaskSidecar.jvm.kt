package com.acite.axlranko.util

import java.io.File

fun isMaskSidecar(file: File): Boolean = isMaskSidecar(file.name)

fun maskFileFor(imageFile: File): File =
    File(imageFile.parentFile, imageFile.nameWithoutExtension + MASK_SIDECAR_SUFFIX)

/** Header-only: PNG color type 4/6, or WebP VP8X alpha flag. Palette+tRNS is ignored. */
fun fileHasAlphaChannel(file: File): Boolean {
    if (!file.isFile) return false
    file.inputStream().buffered().use { input ->
        val magic = ByteArray(12)
        val n = input.read(magic)
        if (n < 8) return false
        if (isPngSignature(magic)) {
            if (n < 12) return false
            val rest = ByteArray(14)
            val got = input.read(rest)
            if (got < 14) return false
            // IHDR: type (4) + width (4) + height (4) + bit depth (1) + color type (1)
            if (String(rest, 0, 4, Charsets.US_ASCII) != "IHDR") return false
            val colorType = rest[13].toInt() and 0xFF
            return colorType == 4 || colorType == 6
        }
        if (isRiffWebp(magic, n)) {
            return webpVp8xHasAlpha(file)
        }
        return false
    }
}

private fun isPngSignature(magic: ByteArray): Boolean =
    magic.size >= 8 &&
        magic[0] == 0x89.toByte() &&
        magic[1] == 'P'.code.toByte() &&
        magic[2] == 'N'.code.toByte() &&
        magic[3] == 'G'.code.toByte() &&
        magic[4] == 0x0D.toByte() &&
        magic[5] == 0x0A.toByte() &&
        magic[6] == 0x1A.toByte() &&
        magic[7] == 0x0A.toByte()

private fun isRiffWebp(magic: ByteArray, n: Int): Boolean =
    n >= 12 &&
        magic[0] == 'R'.code.toByte() &&
        magic[1] == 'I'.code.toByte() &&
        magic[2] == 'F'.code.toByte() &&
        magic[3] == 'F'.code.toByte() &&
        magic[8] == 'W'.code.toByte() &&
        magic[9] == 'E'.code.toByte() &&
        magic[10] == 'B'.code.toByte() &&
        magic[11] == 'P'.code.toByte()

private fun webpVp8xHasAlpha(file: File): Boolean {
    file.inputStream().buffered().use { input ->
        val header = ByteArray(12)
        if (input.read(header) < 12) return false
        while (true) {
            val fourcc = ByteArray(4)
            if (input.read(fourcc) < 4) return false
            val sizeBytes = ByteArray(4)
            if (input.read(sizeBytes) < 4) return false
            val chunkSize = (sizeBytes[0].toInt() and 0xFF) or
                ((sizeBytes[1].toInt() and 0xFF) shl 8) or
                ((sizeBytes[2].toInt() and 0xFF) shl 16) or
                ((sizeBytes[3].toInt() and 0xFF) shl 24)
            if (chunkSize < 0) return false
            val tag = String(fourcc, Charsets.US_ASCII)
            if (tag == "VP8X") {
                if (chunkSize < 1) return false
                val flags = input.read()
                if (flags < 0) return false
                return (flags and 0x10) != 0
            }
            var remaining = chunkSize
            while (remaining > 0) {
                val skipped = input.skip(remaining.toLong()).toInt()
                if (skipped <= 0) return false
                remaining -= skipped
            }
            if (chunkSize % 2 == 1) {
                input.read()
            }
        }
    }
}

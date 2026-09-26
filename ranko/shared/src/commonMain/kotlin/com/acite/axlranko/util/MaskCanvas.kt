package com.acite.axlranko.util

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot

/** Inclusive pixel bounds touched by a mask operation. */
data class MaskDirtyRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val isEmpty: Boolean
        get() = right < left || bottom < top

    fun union(other: MaskDirtyRect): MaskDirtyRect = MaskDirtyRect(
        left = minOf(left, other.left),
        top = minOf(top, other.top),
        right = maxOf(right, other.right),
        bottom = maxOf(bottom, other.bottom),
    )

    companion object {
        val EMPTY = MaskDirtyRect(1, 0, 0, 0)
    }
}

/**
 * 8-bit grayscale loss mask backed by a packed `ByteArray`.
 *
 * Painting is idempotent: a white stroke max-blends toward white and an erase
 * min-blends toward black, so overlapping samples of one stroke do not
 * accumulate and a re-stamped sample is a no-op.
 */
class MaskCanvas(
    val width: Int,
    val height: Int,
    private val pixels: ByteArray = ByteArray(width * height),
) {

    fun pixel(x: Int, y: Int): Int = pixels[y * width + x].toInt() and 0xFF

    fun copyBytes(): ByteArray = pixels.copyOf()

    fun fill(value: Int) {
        val v = value.coerceIn(0, 255).toByte()
        pixels.fill(v)
    }

    fun invert() {
        for (i in pixels.indices) {
            pixels[i] = (255 - (pixels[i].toInt() and 0xFF)).toByte()
        }
    }

    /** One soft round stamp. Returns the touched pixel bounds. */
    fun stamp(
        cx: Float,
        cy: Float,
        radius: Float,
        feather: Float,
        strength: Float,
        erase: Boolean,
    ): MaskDirtyRect {
        if (radius <= 0f) return MaskDirtyRect.EMPTY
        val x0 = floor(cx - radius).toInt().coerceAtLeast(0)
        val y0 = floor(cy - radius).toInt().coerceAtLeast(0)
        val x1 = ceil(cx + radius).toInt().coerceAtMost(width - 1)
        val y1 = ceil(cy + radius).toInt().coerceAtMost(height - 1)
        if (x1 < x0 || y1 < y0) return MaskDirtyRect.EMPTY
        for (y in y0..y1) {
            val row = y * width
            for (x in x0..x1) {
                val falloff = brushFalloff(hypot(x - cx, y - cy), radius, feather)
                if (falloff <= 0f) continue
                val idx = row + x
                val blended = blendMaskSample(pixels[idx].toInt() and 0xFF, falloff, strength, erase)
                pixels[idx] = blended.toByte()
            }
        }
        return MaskDirtyRect(x0, y0, x1, y1)
    }

    /**
     * Paint a segment, sampling the path so a fast pointer jump still leaves a
     * continuous stroke. Both endpoints are stamped.
     */
    fun stroke(
        fromX: Float,
        fromY: Float,
        toX: Float,
        toY: Float,
        radius: Float,
        feather: Float,
        strength: Float,
        erase: Boolean,
    ): MaskDirtyRect {
        var dirty = stamp(fromX, fromY, radius, feather, strength, erase)
        for ((x, y) in interpolateStroke(fromX, fromY, toX, toY, strokeSpacing(radius))) {
            dirty = dirty.union(stamp(x, y, radius, feather, strength, erase))
        }
        return dirty
    }

    companion object {
        fun white(width: Int, height: Int): MaskCanvas =
            MaskCanvas(width, height).also { it.fill(255) }

        fun fromBytes(source: ByteArray, width: Int, height: Int): MaskCanvas {
            require(source.size >= width * height)
            return MaskCanvas(width, height, source.copyOf(width * height))
        }

        fun fromGrayBytes(source: ByteArray, srcW: Int, srcH: Int, width: Int, height: Int): MaskCanvas {
            val canvas = MaskCanvas(width, height)
            if (srcW <= 0 || srcH <= 0) return canvas.also { it.fill(255) }
            for (y in 0 until height) {
                val sy = (y * srcH / height).coerceAtMost(srcH - 1)
                val srcRow = sy * srcW
                val dstRow = y * width
                for (x in 0 until width) {
                    val sx = (x * srcW / width).coerceAtMost(srcW - 1)
                    canvas.pixels[dstRow + x] = source[srcRow + sx]
                }
            }
            return canvas
        }

        /** Alpha channel of packed ARGB as the mask (transparent = ignore). */
        fun fromAlphaRgba(argb: IntArray, srcW: Int, srcH: Int, width: Int, height: Int): MaskCanvas {
            val canvas = MaskCanvas(width, height)
            if (srcW <= 0 || srcH <= 0) return canvas
            for (y in 0 until height) {
                val sy = (y * srcH / height).coerceAtMost(srcH - 1)
                val srcRow = sy * srcW
                val dstRow = y * width
                for (x in 0 until width) {
                    val sx = (x * srcW / width).coerceAtMost(srcW - 1)
                    canvas.pixels[dstRow + x] = ((argb[srcRow + sx] ushr 24) and 0xFF).toByte()
                }
            }
            return canvas
        }
    }
}

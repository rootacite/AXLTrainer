package com.acite.axlranko.util

import java.awt.image.BufferedImage
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
 * 8-bit grayscale loss mask backed by a `TYPE_BYTE_GRAY` raster.
 *
 * Painting is idempotent: a white stroke max-blends toward white and an erase
 * min-blends toward black, so overlapping samples of one stroke do not
 * accumulate and a re-stamped sample is a no-op.
 */
class MaskCanvas(val width: Int, val height: Int) {

    private val image = BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    private val raster = image.raster

    fun pixel(x: Int, y: Int): Int = raster.getSample(x, y, 0)

    fun copyImage(): BufferedImage {
        val out = BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
        out.createGraphics().run {
            drawImage(image, 0, 0, null)
            dispose()
        }
        return out
    }

    fun fill(value: Int) {
        val v = value.coerceIn(0, 255)
        for (y in 0 until height) {
            for (x in 0 until width) {
                raster.setSample(x, y, 0, v)
            }
        }
    }

    fun invert() {
        for (y in 0 until height) {
            for (x in 0 until width) {
                raster.setSample(x, y, 0, 255 - raster.getSample(x, y, 0))
            }
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
            for (x in x0..x1) {
                val falloff = brushFalloff(hypot(x - cx, y - cy), radius, feather)
                if (falloff <= 0f) continue
                val blended = blendMaskSample(raster.getSample(x, y, 0), falloff, strength, erase)
                raster.setSample(x, y, 0, blended)
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

        fun fromGrayImage(source: BufferedImage, width: Int, height: Int): MaskCanvas =
            MaskCanvas(width, height).also { it.draw(source) }

        /** Alpha channel of the training image as the mask (transparent = ignore). */
        fun fromAlpha(source: BufferedImage, width: Int, height: Int): MaskCanvas {
            val canvas = MaskCanvas(width, height)
            val canvasRaster = canvas.image.raster
            val srcW = source.width
            val srcH = source.height
            for (y in 0 until height) {
                val sy = (y * srcH / height).coerceAtMost(srcH - 1)
                for (x in 0 until width) {
                    val sx = (x * srcW / width).coerceAtMost(srcW - 1)
                    val alpha = (source.getRGB(sx, sy) ushr 24) and 0xFF
                    canvasRaster.setSample(x, y, 0, alpha)
                }
            }
            return canvas
        }

        private fun MaskCanvas.draw(source: BufferedImage) {
            image.createGraphics().run {
                drawImage(source, 0, 0, width, height, null)
                dispose()
            }
        }
    }
}

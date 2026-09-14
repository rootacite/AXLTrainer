package com.acite.axlranko.util

import java.awt.Color
import java.awt.image.BufferedImage
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MaskCanvasTest {

    private fun canvas(size: Int = 128) = MaskCanvas.white(size, size)

    @Test
    fun startsWhiteSoEverythingTrains() {
        val mask = canvas(16)
        assertEquals(255, mask.pixel(0, 0))
        assertEquals(255, mask.pixel(15, 15))
    }

    @Test
    fun fastStrokeLeavesNoGaps() {
        val mask = canvas(128)
        // One pointer jump: the old per-event stamping left isolated dots here.
        mask.stroke(10f, 64f, 118f, 64f, radius = 6f, feather = 0.2f, strength = 1f, erase = true)
        for (x in 10..118) {
            val v = mask.pixel(x, 64)
            assertTrue(v < 40, "gap at x=$x (value=$v)")
        }
    }

    @Test
    fun eraseThenPaintRestoresFullWeight() {
        val mask = canvas(64)
        mask.fill(0)
        mask.stroke(32f, 32f, 32f, 32f, radius = 10f, feather = 0f, strength = 1f, erase = false)
        assertEquals(255, mask.pixel(32, 32))
    }

    @Test
    fun strengthScalesThePeak() {
        val mask = canvas(64)
        mask.fill(0)
        mask.stroke(32f, 32f, 32f, 32f, radius = 8f, feather = 0f, strength = 0.5f, erase = false)
        val v = mask.pixel(32, 32)
        assertTrue(v in 126..128, "half strength gave $v")
    }

    @Test
    fun overlappingSamplesDoNotAccumulate() {
        val mask = canvas(64)
        mask.fill(0)
        mask.stroke(20f, 32f, 40f, 32f, radius = 8f, feather = 0f, strength = 1f, erase = false)
        val first = mask.pixel(30, 32)
        mask.stroke(20f, 32f, 40f, 32f, radius = 8f, feather = 0f, strength = 1f, erase = false)
        assertEquals(first, mask.pixel(30, 32))
    }

    @Test
    fun featherFadesToZeroAtTheRim() {
        val mask = canvas(64)
        mask.fill(0)
        mask.stroke(32f, 32f, 32f, 32f, radius = 20f, feather = 0.5f, strength = 1f, erase = false)
        val core = mask.pixel(32, 32)
        val rim = mask.pixel(32 + 19, 32)
        assertEquals(255, core)
        assertTrue(rim < core / 2, "rim should fade, got $rim")
    }

    @Test
    fun strokeReportsDirtyBoundsCoveringThePath() {
        val mask = canvas(128)
        val dirty = mask.stroke(20f, 20f, 100f, 60f, radius = 5f, feather = 0f, strength = 1f, erase = true)
        assertTrue(dirty.left <= 15 && dirty.top <= 15)
        assertTrue(dirty.right >= 105 && dirty.bottom >= 65)
    }

    @Test
    fun alphaBecomesTheMask() {
        val rgba = BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB)
        val g = rgba.createGraphics()
        g.color = Color(0, 0, 0, 0)
        g.fillRect(0, 0, 32, 32)
        g.color = Color(255, 255, 255, 255)
        g.fillRect(0, 0, 16, 32)
        g.dispose()

        val mask = MaskCanvas.fromAlpha(rgba, 32, 32)
        assertEquals(255, mask.pixel(4, 16))
        assertEquals(0, mask.pixel(28, 16))
    }

    @Test
    fun invertSwapsWeights() {
        val mask = canvas(16)
        mask.invert()
        assertEquals(0, mask.pixel(8, 8))
        mask.invert()
        assertEquals(255, mask.pixel(8, 8))
    }
}

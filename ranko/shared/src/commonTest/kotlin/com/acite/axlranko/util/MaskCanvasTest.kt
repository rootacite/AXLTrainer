package com.acite.axlranko.util

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
        val argb = IntArray(32 * 32)
        for (y in 0 until 32) {
            for (x in 0 until 32) {
                val a = if (x < 16) 255 else 0
                argb[y * 32 + x] = (a shl 24)
            }
        }
        val mask = MaskCanvas.fromAlphaRgba(argb, 32, 32, 32, 32)
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

    @Test
    fun copyBytesRoundTrips() {
        val mask = canvas(8)
        mask.fill(40)
        val copy = MaskCanvas.fromBytes(mask.copyBytes(), 8, 8)
        assertEquals(40, copy.pixel(3, 3))
    }
}

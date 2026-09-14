package com.acite.axlranko.util

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MaskBrushTest {
    @Test
    fun coreIsFullAndEdgeIsZero() {
        val radius = 10f
        val feather = 0.20f
        assertEquals(1f, brushFalloff(0f, radius, feather), 1e-5f)
        assertEquals(0f, brushFalloff(radius, radius, feather), 1e-5f)
        assertEquals(0f, brushFalloff(radius + 1f, radius, feather), 1e-5f)
        val mid = brushFalloff(radius - 0.2f * (2f * radius) / 2f, radius, feather)
        assertTrue(mid in 0.01f..0.99f)
    }

    @Test
    fun zeroFeatherIsHardDisk() {
        assertEquals(1f, brushFalloff(9.9f, 10f, 0f), 1e-5f)
        assertEquals(0f, brushFalloff(10f, 10f, 0f), 1e-5f)
    }

    @Test
    fun radiiMatchTheDefaultRim() {
        // 20% of the diameter = 40% of the radius, so the core keeps 60% of it.
        val radii = brushRadii(50f, 0.20f)
        assertEquals(50f, radii.outer, 1e-4f)
        assertEquals(30f, radii.inner, 1e-4f)
    }

    @Test
    fun radiiFollowFeatherExtremes() {
        assertEquals(40f, brushRadii(40f, 0f).inner, 1e-4f)
        assertEquals(0f, brushRadii(40f, 1f).inner, 1e-4f)
        assertEquals(0f, brushRadii(0f, 0.5f).outer, 1e-4f)
    }

    @Test
    fun cursorRadiiBoundThePaintedArea() {
        val radii = brushRadii(12f, 0.25f)
        assertEquals(1f, brushFalloff(radii.inner, 12f, 0.25f), 1e-5f)
        assertEquals(0f, brushFalloff(radii.outer, 12f, 0.25f), 1e-5f)
        assertEquals(0f, brushFalloff(radii.outer + 0.1f, 12f, 0.25f), 1e-5f)
    }

    @Test
    fun wheelNudgesStayInsideSliderRange() {
        assertEquals(26f, nudgeBrushRadius(24f, 1), 1e-4f)
        assertEquals(22f, nudgeBrushRadius(24f, -1), 1e-4f)
        assertEquals(128f, nudgeBrushRadius(120f, 10), 1e-4f)
        assertEquals(2f, nudgeBrushRadius(6f, -10), 1e-4f)
        assertEquals(BRUSH_RADIUS_MAX, nudgeBrushRadius(BRUSH_RADIUS_MAX - 1f, 5), 1e-4f)
        assertEquals(BRUSH_RADIUS_MIN, nudgeBrushRadius(BRUSH_RADIUS_MIN + 1f, -5), 1e-4f)
        assertEquals(24f, nudgeBrushRadius(24f, 0), 1e-4f)
    }

    @Test
    fun interpolatesIncludingEndExcludingStart() {
        val pts = interpolateStroke(0f, 0f, 10f, 0f, spacing = 2.5f)
        assertEquals(4, pts.size)
        assertEquals(10f, pts.last().first, 1e-4f)
        assertTrue(pts.none { it.first == 0f && it.second == 0f })
    }

    @Test
    fun blendRespectsStrengthAndDoesNotAccumulatePastPeak() {
        assertEquals(255, blendMaskSample(0, 1f, 1f, erase = false))
        assertEquals(0, blendMaskSample(255, 1f, 1f, erase = true))
        val half = blendMaskSample(0, 1f, 0.5f, erase = false)
        assertTrue(half in 127..128)
        assertEquals(half, blendMaskSample(half, 1f, 0.5f, erase = false))
    }
}

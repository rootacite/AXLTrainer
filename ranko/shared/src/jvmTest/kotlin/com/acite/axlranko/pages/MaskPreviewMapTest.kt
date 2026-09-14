package com.acite.axlranko.pages

import androidx.compose.ui.geometry.Offset
import com.acite.axlranko.pages.components.mapToSource
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class MaskPreviewMapTest {
    @Test
    fun mapsFitCenter() {
        val mapped = mapToSource(
            pos = Offset(100f, 80f),
            boxW = 200f,
            boxH = 160f,
            srcW = 100,
            srcH = 80,
        )
        assertEquals(50f, mapped!!.first, 0.01f)
        assertEquals(40f, mapped.second, 0.01f)
    }

    @Test
    fun ignoresLetterbox() {
        assertNull(
            mapToSource(
                pos = Offset(5f, 80f),
                boxW = 200f,
                boxH = 160f,
                srcW = 80,
                srcH = 80,
            )
        )
    }

    @Test
    fun ignoresPointsBeyondImageEdges() {
        // Image fills the box exactly, so anything past the edges is outside the image.
        assertNull(mapToSource(Offset(250f, 80f), 200f, 160f, 100, 80))
        assertNull(mapToSource(Offset(-1f, 40f), 200f, 160f, 100, 80))
        assertNull(mapToSource(Offset(100f, 200f), 200f, 160f, 100, 80))
    }

    @Test
    fun keepsPointsOnTheImageEdge() {
        assertEquals(0f, mapToSource(Offset(0f, 0f), 200f, 160f, 100, 80)!!.first, 0.01f)
        assertEquals(99f, mapToSource(Offset(200f, 160f), 200f, 160f, 100, 80)!!.first, 0.01f)
    }
}

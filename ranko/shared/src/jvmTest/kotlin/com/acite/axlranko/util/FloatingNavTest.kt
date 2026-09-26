package com.acite.axlranko.util

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import kotlin.test.Test
import kotlin.test.assertEquals

class FloatingNavTest {

    private val nav = Size(80f, 200f)
    private val bounds = Size(1000f, 800f)

    @Test
    fun clampPullsANegativeOriginToZero() {
        assertEquals(Offset(0f, 0f), clampNavOffset(Offset(-50f, -20f), nav, bounds))
    }

    @Test
    fun clampPullsPastTheFarEdge() {
        assertEquals(
            Offset(920f, 600f),
            clampNavOffset(Offset(5000f, 5000f), nav, bounds),
        )
    }

    @Test
    fun clampWhenNavIsLargerThanTheWindowIsOriginZero() {
        assertEquals(
            Offset(0f, 0f),
            clampNavOffset(Offset(40f, 40f), Size(2000f, 1600f), bounds),
        )
    }

    @Test
    fun nearestDockPicksTheClosestEdge() {
        assertEquals(NavDock.Left, nearestNavDock(Offset(10f, 300f), nav, bounds))
        assertEquals(NavDock.Right, nearestNavDock(Offset(910f, 300f), nav, bounds))
        assertEquals(NavDock.Top, nearestNavDock(Offset(400f, 8f), nav, bounds))
        assertEquals(NavDock.Bottom, nearestNavDock(Offset(400f, 590f), nav, bounds))
    }

    @Test
    fun snapLeftPinsXAndKeepsAFittingY() {
        assertEquals(
            Offset(0f, 240f),
            snapNavOffset(Offset(40f, 240f), nav, bounds, NavDock.Left),
        )
    }

    @Test
    fun peekLeftLeavesTheAskedSliverVisible() {
        val snapped = Offset(0f, 100f)
        val peeked = peekNavOffset(snapped, nav, NavDock.Left, peekPx = 24f)
        assertEquals(-56f, peeked.x)
        assertEquals(100f, peeked.y)
        assertEquals(24f, peeked.x + nav.width)
    }

    @Test
    fun resizeReSnapsARightDockInsteadOfKeepingTheOldX() {
        val origin = Offset(1920f, 100f)
        val size = Size(80f, 80f)
        val wide = snapNavOffset(origin, size, Size(2000f, 1000f), NavDock.Right)
        assertEquals(1920f, wide.x)
        val narrow = snapNavOffset(wide, size, Size(800f, 500f), NavDock.Right)
        assertEquals(720f, narrow.x)
        assertEquals(100f, narrow.y)
    }

    @Test
    fun peekVisibleIsAtLeastTheMinimumAndAtMostTheAxis() {
        assertEquals(55f, navPeekVisiblePx(100f, minPeekPx = 24f))
        assertEquals(24f, navPeekVisiblePx(40f, minPeekPx = 24f))
        assertEquals(20f, navPeekVisiblePx(20f, minPeekPx = 24f))
        assertEquals(0f, navPeekVisiblePx(0f, minPeekPx = 24f))
    }
}

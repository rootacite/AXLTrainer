package com.acite.axlranko.util

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size

enum class NavDock { Left, Right, Top, Bottom }

const val NAV_IDLE_MILLIS = 2000L
const val NAV_PEEK_VISIBLE_FRACTION = 0.55f

/** Shifts [origin] by the least amount that keeps [navSize] inside [bounds]. */
fun clampNavOffset(origin: Offset, navSize: Size, bounds: Size): Offset {
    val maxX = (bounds.width - navSize.width).coerceAtLeast(0f)
    val maxY = (bounds.height - navSize.height).coerceAtLeast(0f)
    return Offset(origin.x.coerceIn(0f, maxX), origin.y.coerceIn(0f, maxY))
}

/** Edge whose window side is closest to the matching side of the nav rect. */
fun nearestNavDock(origin: Offset, navSize: Size, bounds: Size): NavDock {
    val left = origin.x
    val right = bounds.width - origin.x - navSize.width
    val top = origin.y
    val bottom = bounds.height - origin.y - navSize.height
    var dock = NavDock.Left
    var best = left
    if (right < best) {
        dock = NavDock.Right
        best = right
    }
    if (top < best) {
        dock = NavDock.Top
        best = top
    }
    if (bottom < best) {
        dock = NavDock.Bottom
    }
    return dock
}

/** Pins the docked axis to that edge; the other axis stays, then both are clamped. */
fun snapNavOffset(origin: Offset, navSize: Size, bounds: Size, dock: NavDock): Offset {
    val clamped = clampNavOffset(origin, navSize, bounds)
    val maxX = (bounds.width - navSize.width).coerceAtLeast(0f)
    val maxY = (bounds.height - navSize.height).coerceAtLeast(0f)
    return when (dock) {
        NavDock.Left -> Offset(0f, clamped.y)
        NavDock.Right -> Offset(maxX, clamped.y)
        NavDock.Top -> Offset(clamped.x, 0f)
        NavDock.Bottom -> Offset(clamped.x, maxY)
    }
}

/**
 * Slides a snapped nav partly off [dock] so [peekPx] of that axis stays inside the window.
 */
fun peekNavOffset(snapped: Offset, navSize: Size, dock: NavDock, peekPx: Float): Offset {
    val hiddenX = (navSize.width - peekPx).coerceIn(0f, navSize.width)
    val hiddenY = (navSize.height - peekPx).coerceIn(0f, navSize.height)
    return when (dock) {
        NavDock.Left -> Offset(snapped.x - hiddenX, snapped.y)
        NavDock.Right -> Offset(snapped.x + hiddenX, snapped.y)
        NavDock.Top -> Offset(snapped.x, snapped.y - hiddenY)
        NavDock.Bottom -> Offset(snapped.x, snapped.y + hiddenY)
    }
}

/** Visible sliver when peeked: 55% of the docked axis, at least [minPeekPx], never more than the axis. */
fun navPeekVisiblePx(axisPx: Float, minPeekPx: Float): Float {
    if (axisPx <= 0f) return 0f
    return maxOf(minPeekPx, axisPx * NAV_PEEK_VISIBLE_FRACTION).coerceAtMost(axisPx)
}

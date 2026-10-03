package com.acite.axlranko.ui

import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.SubcomposeLayout
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlin.math.abs
import kotlin.math.max
import kotlinx.coroutines.launch

/** A viewport is portrait when it is taller than it is wide. A square stays on the landscape layout. */
internal fun isPortrait(maxWidth: Dp, maxHeight: Dp): Boolean = maxHeight > maxWidth

/**
 * Hardware and Real-time Metrics cards stay on one row while the viewport is at least half as wide
 * as it is tall. Narrower than that, they wrap onto extra rows.
 */
internal fun wideMetricRow(width: Dp, height: Dp): Boolean =
    height.value <= 0f || width >= height * 0.5f

/**
 * Two slots on one row when their single-line widths fit, one above the other when they do not.
 *
 * Each slot must be a single root. Text inside a slot stays one line (`maxLines = 1`,
 * `softWrap = false`); this only chooses the row or the stack, and a stack gives each slot the
 * full width so the ellipsis has a bound. A slot must not use [androidx.compose.foundation.layout.RowScope.weight].
 */
@Composable
internal fun SingleLineOrStacked(
    modifier: Modifier = Modifier,
    spacing: Dp = 8.dp,
    verticalSpacing: Dp = 4.dp,
    first: @Composable () -> Unit,
    second: @Composable () -> Unit,
) {
    SubcomposeLayout(modifier) { constraints ->
        val gap = spacing.roundToPx()
        val loose = Constraints(maxWidth = Constraints.Infinity, maxHeight = Constraints.Infinity)
        val measuredFirst = subcompose("measure-first", first).first().measure(loose)
        val measuredSecond = subcompose("measure-second", second).first().measure(loose)
        val bounded = constraints.hasBoundedWidth
        val stack = bounded && measuredFirst.width + measuredSecond.width + gap > constraints.maxWidth
        if (!stack) {
            val width = if (bounded) constraints.maxWidth else measuredFirst.width + gap + measuredSecond.width
            val height = max(measuredFirst.height, measuredSecond.height)
            layout(width, height) {
                measuredFirst.placeRelative(0, (height - measuredFirst.height) / 2)
                val x2 = if (bounded) width - measuredSecond.width else measuredFirst.width + gap
                measuredSecond.placeRelative(x2, (height - measuredSecond.height) / 2)
            }
        } else {
            val tight = constraints.copy(minWidth = 0, minHeight = 0)
            val placedFirst = subcompose("place-first", first).first().measure(tight)
            val placedSecond = subcompose("place-second", second).first().measure(tight)
            val vGap = verticalSpacing.roundToPx()
            layout(constraints.maxWidth, placedFirst.height + vGap + placedSecond.height) {
                placedFirst.placeRelative(0, 0)
                placedSecond.placeRelative(0, placedFirst.height + vGap)
            }
        }
    }
}

/**
 * A vertical mouse wheel over a horizontal list. Desktop reports a notch as a small delta and a
 * trackpad as pixels; either one moves the row, and the event is consumed so the page does not
 * scroll as well.
 */
@Composable
internal fun Modifier.wheelScrollsHorizontally(state: LazyListState): Modifier {
    val scope = rememberCoroutineScope()
    val density = LocalDensity.current
    return pointerInput(state) {
        awaitPointerEventScope {
            while (true) {
                val event = awaitPointerEvent()
                if (event.type != PointerEventType.Scroll) continue
                val change = event.changes.firstOrNull() ?: continue
                val raw = when {
                    change.scrollDelta.y != 0f -> change.scrollDelta.y
                    else -> change.scrollDelta.x
                }
                if (raw == 0f || !raw.isFinite()) continue
                change.consume()
                val pixels = if (abs(raw) < 10f) with(density) { raw * 56.dp.toPx() } else raw
                scope.launch { state.scroll { scrollBy(-pixels) } }
            }
        }
    }
}

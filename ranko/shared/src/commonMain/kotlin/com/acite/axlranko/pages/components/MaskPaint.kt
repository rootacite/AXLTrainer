package com.acite.axlranko.pages.components

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

/**
 * Pointer input for mask painting, in pixels local to the modified composable.
 *
 * The desktop implementation bypasses Compose pointer handling and listens to
 * raw AWT mouse events, so both buttons are reported and tablet motion is not
 * lost; while a stroke is active it also samples the pointer on a timer, since
 * motion events are coalesced and fast strokes would otherwise become dots.
 *
 * @param onPointerMoved every pointer position while editing is on, whether a
 *   stroke is in progress or not, so a brush cursor can follow the pointer.
 * @param onBrushResize Alt+wheel notches over the canvas; positive grows the brush.
 */
@Composable
expect fun Modifier.maskPaintInput(
    enabled: Boolean,
    onStrokeStart: (boxX: Float, boxY: Float, erase: Boolean) -> Unit,
    onStrokeMove: (boxX: Float, boxY: Float) -> Unit,
    onStrokeEnd: () -> Unit,
    onPointerMoved: (boxX: Float, boxY: Float) -> Unit,
    onBrushResize: (steps: Int) -> Unit,
): Modifier

package com.acite.axlranko.pages.components

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.LocalAwtWindow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import java.awt.AWTEvent
import java.awt.Component
import java.awt.MouseInfo
import java.awt.Point
import java.awt.Toolkit
import java.awt.Window
import java.awt.event.AWTEventListener
import java.awt.event.MouseEvent
import java.awt.event.MouseWheelEvent
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import javax.swing.RootPaneContainer
import javax.swing.SwingUtilities

private const val POINTER_POLL_MS = 4L

/**
 * Mask painting input for the desktop build.
 *
 * Deliberately bypasses the Compose pointer API: Compose delivers one pointer
 * stream and reports the primary button only, so a right-button erase cannot be
 * expressed there. A global AWT listener sees both buttons, and a high-frequency
 * poll of the real pointer keeps fast strokes continuous, because AWT coalesces
 * motion events into far fewer samples than the pointer actually produces.
 */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
actual fun Modifier.maskPaintInput(
    enabled: Boolean,
    onStrokeStart: (boxX: Float, boxY: Float, erase: Boolean) -> Unit,
    onStrokeMove: (boxX: Float, boxY: Float) -> Unit,
    onStrokeEnd: () -> Unit,
    onPointerMoved: (boxX: Float, boxY: Float) -> Unit,
    onBrushResize: (steps: Int) -> Unit,
): Modifier {
    if (!enabled) return this

    val window: Window = LocalAwtWindow.current ?: return this
    var bounds by remember { mutableStateOf(Rect.Zero) }
    val startCb = rememberUpdatedState(onStrokeStart)
    val moveCb = rememberUpdatedState(onStrokeMove)
    val endCb = rememberUpdatedState(onStrokeEnd)
    val pointerCb = rememberUpdatedState(onPointerMoved)
    val resizeCb = rememberUpdatedState(onBrushResize)

    // Screen points must be converted through a component inside the client area,
    // never through the window itself: a Window's screen position is its frame
    // origin, decorations included, so converting through it lands the decoration
    // height off (41 px under KWin/XWayland). Mouse event coordinates, and the
    // Compose bounds compared against them, are client-area based.
    val clientArea: Component = (window as? RootPaneContainer)?.contentPane ?: window

    DisposableEffect(window, enabled) {
        val stroking = AtomicBoolean(false)
        val erasing = AtomicBoolean(false)
        val strokeStarted = AtomicBoolean(false)

        fun toLocal(point: Point): Offset? {
            val b = bounds
            if (b.width <= 0f || b.height <= 0f) return null
            return Offset(point.x - b.left, point.y - b.top)
        }

        fun windowPoint(screenPoint: Point): Point {
            SwingUtilities.convertPointFromScreen(screenPoint, clientArea)
            return screenPoint
        }

        fun begin(local: Offset) {
            strokeStarted.set(true)
            startCb.value(local.x, local.y, erasing.get())
        }

        fun extend(local: Offset) {
            if (strokeStarted.get()) {
                moveCb.value(local.x, local.y)
            } else {
                // Press landed outside the image and the pointer has entered it.
                begin(local)
            }
        }

        fun finish() {
            if (stroking.getAndSet(false) && strokeStarted.getAndSet(false)) {
                endCb.value()
            }
        }

        val listener = AWTEventListener { event ->
            if (event !is MouseEvent) return@AWTEventListener
            val component = event.component ?: return@AWTEventListener
            val owner = if (component is Window) component
            else SwingUtilities.windowForComponent(component)
            if (owner !== window) return@AWTEventListener
            when (event.id) {
                MouseEvent.MOUSE_PRESSED -> {
                    erasing.set(
                        SwingUtilities.isRightMouseButton(event) ||
                            (SwingUtilities.isLeftMouseButton(event) && event.isControlDown)
                    )
                    stroking.set(true)
                    strokeStarted.set(false)
                    val local = toLocal(event.point)
                    if (local != null) {
                        begin(local)
                        pointerCb.value(local.x, local.y)
                    }
                }

                MouseEvent.MOUSE_DRAGGED -> {
                    if (!stroking.get()) return@AWTEventListener
                    val local = toLocal(event.point)
                    if (local != null) {
                        extend(local)
                        pointerCb.value(local.x, local.y)
                    }
                    event.consume()
                }

                MouseEvent.MOUSE_MOVED, MouseEvent.MOUSE_EXITED -> {
                    val local = toLocal(event.point)
                    if (local != null) pointerCb.value(local.x, local.y)
                }

                MouseEvent.MOUSE_WHEEL -> {
                    val wheel = event as MouseWheelEvent
                    if (!wheel.isAltDown || wheel.wheelRotation == 0) return@AWTEventListener
                    val b = bounds
                    val local = toLocal(wheel.point) ?: return@AWTEventListener
                    val overCanvas =
                        local.x >= 0f && local.y >= 0f && local.x <= b.width && local.y <= b.height
                    if (!overCanvas) return@AWTEventListener
                    // Wheel up (negative rotation) grows the brush.
                    resizeCb.value(-wheel.wheelRotation)
                    pointerCb.value(local.x, local.y)
                    wheel.consume()
                }

                MouseEvent.MOUSE_RELEASED -> {
                    finish()
                    event.consume()
                }
            }
        }
        Toolkit.getDefaultToolkit().addAWTEventListener(
            listener,
            AWTEvent.MOUSE_EVENT_MASK or AWTEvent.MOUSE_MOTION_EVENT_MASK or AWTEvent.MOUSE_WHEEL_EVENT_MASK,
        )

        // AWT coalesces motion events; poll the real pointer so fast strokes
        // stay continuous instead of landing as separate dots.
        val poller = Executors.newSingleThreadScheduledExecutor { runnable ->
            Thread(runnable, "chromatrix-mask-pointer").apply { isDaemon = true }
        }
        poller.scheduleAtFixedRate(
            {
                if (!stroking.get()) return@scheduleAtFixedRate
                val info = MouseInfo.getPointerInfo() ?: return@scheduleAtFixedRate
                val screenPoint = info.location ?: return@scheduleAtFixedRate
                SwingUtilities.invokeLater {
                    if (!stroking.get()) return@invokeLater
                    val local = toLocal(windowPoint(Point(screenPoint))) ?: return@invokeLater
                    extend(local)
                    pointerCb.value(local.x, local.y)
                }
            },
            0L,
            POINTER_POLL_MS,
            TimeUnit.MILLISECONDS,
        )

        onDispose {
            Toolkit.getDefaultToolkit().removeAWTEventListener(listener)
            stroking.set(false)
            poller.shutdownNow()
        }
    }

    return this.onGloballyPositioned { bounds = it.boundsInWindow() }
}

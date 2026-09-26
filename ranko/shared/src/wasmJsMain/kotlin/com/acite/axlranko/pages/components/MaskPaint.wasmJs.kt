package com.acite.axlranko.pages.components

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import kotlinx.browser.document
import kotlinx.browser.window
import org.w3c.dom.events.Event
import org.w3c.dom.events.MouseEvent
import org.w3c.dom.events.WheelEvent

private const val POINTER_POLL_MS = 4

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

    var bounds by remember { mutableStateOf(Rect.Zero) }
    val startCb = rememberUpdatedState(onStrokeStart)
    val moveCb = rememberUpdatedState(onStrokeMove)
    val endCb = rememberUpdatedState(onStrokeEnd)
    val pointerCb = rememberUpdatedState(onPointerMoved)
    val resizeCb = rememberUpdatedState(onBrushResize)

    DisposableEffect(enabled) {
        var stroking = false
        var erasing = false
        var strokeStarted = false
        var lastX = 0f
        var lastY = 0f

        fun toLocal(clientX: Float, clientY: Float): Offset? {
            val b = bounds
            if (b.width <= 0f || b.height <= 0f) return null
            return Offset(clientX - b.left, clientY - b.top)
        }

        fun begin(local: Offset) {
            strokeStarted = true
            startCb.value(local.x, local.y, erasing)
        }

        fun extend(local: Offset) {
            if (strokeStarted) moveCb.value(local.x, local.y) else begin(local)
        }

        fun finish() {
            if (stroking && strokeStarted) endCb.value()
            stroking = false
            strokeStarted = false
        }

        val pointerDown: (Event) -> Unit = handler@{ event ->
            val mouse = event as? MouseEvent ?: return@handler
            erasing = mouse.button == 2.toShort() || (mouse.button == 0.toShort() && mouse.ctrlKey)
            stroking = true
            strokeStarted = false
            lastX = mouse.clientX.toFloat()
            lastY = mouse.clientY.toFloat()
            val local = toLocal(lastX, lastY)
            if (local != null) {
                begin(local)
                pointerCb.value(local.x, local.y)
            }
            mouse.preventDefault()
        }

        val pointerMove: (Event) -> Unit = handler@{ event ->
            val mouse = event as? MouseEvent ?: return@handler
            lastX = mouse.clientX.toFloat()
            lastY = mouse.clientY.toFloat()
            val local = toLocal(lastX, lastY) ?: return@handler
            pointerCb.value(local.x, local.y)
            if (stroking) extend(local)
        }

        val pointerUp: (Event) -> Unit = {
            finish()
        }

        val wheel: (Event) -> Unit = handler@{ event ->
            val we = event as? WheelEvent ?: return@handler
            if (!we.altKey || we.deltaY == 0.0) return@handler
            val local = toLocal(we.clientX.toFloat(), we.clientY.toFloat()) ?: return@handler
            val b = bounds
            val over = local.x >= 0f && local.y >= 0f && local.x <= b.width && local.y <= b.height
            if (!over) return@handler
            val steps = if (we.deltaY < 0) 1 else -1
            resizeCb.value(steps)
            pointerCb.value(local.x, local.y)
            we.preventDefault()
        }

        val context: (Event) -> Unit = { it.preventDefault() }

        val doc = document
        doc.addEventListener("pointerdown", pointerDown)
        doc.addEventListener("pointermove", pointerMove)
        doc.addEventListener("pointerup", pointerUp)
        doc.addEventListener("pointerrawupdate", pointerMove)
        doc.addEventListener("wheel", wheel)
        doc.addEventListener("contextmenu", context)

        val interval = window.setInterval({
            if (stroking) {
                val local = toLocal(lastX, lastY)
                if (local != null) {
                    extend(local)
                    pointerCb.value(local.x, local.y)
                }
            }
            null
        }, POINTER_POLL_MS)

        onDispose {
            doc.removeEventListener("pointerdown", pointerDown)
            doc.removeEventListener("pointermove", pointerMove)
            doc.removeEventListener("pointerup", pointerUp)
            doc.removeEventListener("pointerrawupdate", pointerMove)
            doc.removeEventListener("wheel", wheel)
            doc.removeEventListener("contextmenu", context)
            window.clearInterval(interval)
            stroking = false
        }
    }

    return this.onGloballyPositioned { bounds = it.boundsInWindow() }
}

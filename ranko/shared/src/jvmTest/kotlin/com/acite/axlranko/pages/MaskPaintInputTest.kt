package com.acite.axlranko.pages

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import com.acite.axlranko.pages.components.maskPaintInput
import java.awt.AWTEvent
import java.awt.Component
import java.awt.GraphicsEnvironment
import java.awt.Toolkit
import java.awt.event.AWTEventListener
import java.awt.event.InputEvent
import java.awt.event.MouseEvent
import java.awt.event.MouseWheelEvent
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger
import javax.swing.SwingUtilities
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Drives the real AWT path: events go through the system event queue, so the
 * global listener attached by `maskPaintInput` sees them exactly like a mouse.
 * Skipped on a headless JVM.
 */
class MaskPaintInputTest {

    private class Recorder {
        val starts = Collections.synchronizedList(mutableListOf<Triple<Float, Float, Boolean>>())
        val moves = Collections.synchronizedList(mutableListOf<Pair<Float, Float>>())
        val ends = AtomicInteger(0)
        val pointers = Collections.synchronizedList(mutableListOf<Pair<Float, Float>>())
        val resizes = Collections.synchronizedList(mutableListOf<Int>())
    }

    @Test
    fun bothButtonsAndDragsReachTheMaskEditor() {
        if (GraphicsEnvironment.isHeadless()) return

        val rec = Recorder()
        val raw = AtomicInteger(0)
        val rawListener = AWTEventListener { event ->
            if (event is MouseEvent) {
                raw.incrementAndGet()
            }
        }
        Toolkit.getDefaultToolkit().addAWTEventListener(
            rawListener,
            AWTEvent.MOUSE_EVENT_MASK or AWTEvent.MOUSE_MOTION_EVENT_MASK,
        )
        val window = onEdtGet {
            ComposeWindow().apply {
                setContent {
                    Box(
                        Modifier.fillMaxSize().maskPaintInput(
                            enabled = true,
                            onStrokeStart = { x, y, erase -> rec.starts += Triple(x, y, erase) },
                            onStrokeMove = { x, y -> rec.moves += x to y },
                            onStrokeEnd = { rec.ends.incrementAndGet() },
                            onPointerMoved = { x, y -> rec.pointers += x to y },
                            onBrushResize = { rec.resizes += it },
                        )
                    )
                }
                setSize(240, 160)
                setLocation(0, 0)
                isVisible = true
            }
        }

        try {
            val target = onEdtGet { window.contentPane }
            val queue = Toolkit.getDefaultToolkit().systemEventQueue

            postUntil(queue, { buttonEvent(target, MouseEvent.MOUSE_PRESSED, 60, 80, MouseEvent.BUTTON3, true) }) {
                rec.starts.isNotEmpty()
            }
            assertTrue(
                rec.starts.isNotEmpty(),
                "right press never started a stroke (raw mouse events seen by a plain global listener: ${raw.get()})",
            )
            assertEquals(true, rec.starts.first().third, "right button should erase")

            // The high-rate sampler also reports the real cursor, so look for
            // the dispatched position anywhere in the stream rather than last.
            postUntil(queue, { buttonEvent(target, MouseEvent.MOUSE_DRAGGED, 180, 120, MouseEvent.BUTTON3, true) }) {
                rec.moves.any { abs(it.first - 180f) <= 6f && abs(it.second - 120f) <= 6f }
            }
            assertTrue(
                rec.moves.any { abs(it.first - 180f) <= 6f && abs(it.second - 120f) <= 6f },
                "no callback matched the dragged position; got ${rec.moves} " +
                    "(starts=${rec.starts}, pointers=${rec.pointers}, raw=${raw.get()})",
            )

            queue.postEvent(buttonEvent(target, MouseEvent.MOUSE_RELEASED, 180, 120, MouseEvent.BUTTON3, false))
            assertTrue(pumpUntil(2000) { rec.ends.get() > 0 }, "release did not end the stroke")

            rec.starts.clear()
            postUntil(queue, { buttonEvent(target, MouseEvent.MOUSE_PRESSED, 60, 80, MouseEvent.BUTTON1, true) }) {
                rec.starts.isNotEmpty()
            }
            assertTrue(
                rec.starts.isNotEmpty(),
                "left press never started a stroke (raw events: ${raw.get()})",
            )
            assertEquals(false, rec.starts.first().third, "left button should paint")
            queue.postEvent(buttonEvent(target, MouseEvent.MOUSE_RELEASED, 60, 80, MouseEvent.BUTTON1, false))

            // Hover, without any button, must still report the pointer so the
            // brush cursor can follow it.
            rec.pointers.clear()
            postUntil(queue, { buttonEvent(target, MouseEvent.MOUSE_MOVED, 90, 70, MouseEvent.NOBUTTON, false) }) {
                rec.pointers.isNotEmpty()
            }
            assertTrue(rec.pointers.isNotEmpty(), "hover reported no pointer position")

            // Alt+wheel resizes; wheel up grows the brush and plain wheel is ignored.
            rec.resizes.clear()
            postUntil(queue, { wheelEvent(target, 90, 70, altDown = true, rotation = -1) }) {
                rec.resizes.isNotEmpty()
            }
            assertEquals(listOf(1), rec.resizes.toList(), "wheel up should grow the brush by one step")

            rec.resizes.clear()
            postUntil(queue, { wheelEvent(target, 90, 70, altDown = true, rotation = 1) }) {
                rec.resizes.isNotEmpty()
            }
            assertEquals(listOf(-1), rec.resizes.toList(), "wheel down should shrink the brush")

            rec.resizes.clear()
            queue.postEvent(wheelEvent(target, 90, 70, altDown = false, rotation = -1))
            pumpUntil(300) { rec.resizes.isNotEmpty() }
            assertTrue(rec.resizes.isEmpty(), "wheel without Alt must not resize")
        } finally {
            Toolkit.getDefaultToolkit().removeAWTEventListener(rawListener)
            onEdt { window.dispose() }
        }
    }

    /** Synthetic events can be coalesced or delayed, so post until the callback shows up. */
    private fun postUntil(
        queue: java.awt.EventQueue,
        event: () -> AWTEvent,
        recognised: () -> Boolean,
    ) {
        repeat(20) {
            queue.postEvent(event())
            if (pumpUntil(250, recognised)) return
        }
    }

    private fun buttonEvent(
        target: Component,
        id: Int,
        x: Int,
        y: Int,
        button: Int,
        down: Boolean,
    ): MouseEvent = MouseEvent(
        target,
        id,
        System.currentTimeMillis(),
        if (down) MouseEvent.BUTTON3_DOWN_MASK.takeIf { button == MouseEvent.BUTTON3 }
            ?: MouseEvent.BUTTON1_DOWN_MASK else 0,
        x,
        y,
        1,
        false,
        button,
    )

    private fun wheelEvent(
        target: Component,
        x: Int,
        y: Int,
        altDown: Boolean,
        rotation: Int,
    ): MouseWheelEvent = MouseWheelEvent(
        target,
        MouseEvent.MOUSE_WHEEL,
        System.currentTimeMillis(),
        if (altDown) InputEvent.ALT_DOWN_MASK else 0,
        x,
        y,
        1,
        false,
        MouseWheelEvent.WHEEL_UNIT_SCROLL,
        3,
        rotation,
    )

    private fun onEdt(block: () -> Unit) {
        SwingUtilities.invokeAndWait(block)
    }

    private fun <T> onEdtGet(block: () -> T): T {
        var result: T? = null
        SwingUtilities.invokeAndWait { result = block() }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }

    private fun pumpUntil(timeoutMs: Long, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return true
            SwingUtilities.invokeAndWait { }
            Thread.sleep(10)
        }
        return condition()
    }
}

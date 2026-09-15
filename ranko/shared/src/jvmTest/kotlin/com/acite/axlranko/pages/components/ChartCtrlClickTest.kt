package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import com.acite.axlranko.model.MetricPoint
import java.awt.Component
import java.awt.Container
import java.awt.EventQueue
import java.awt.GraphicsEnvironment
import java.awt.Toolkit
import java.awt.event.InputEvent
import java.awt.event.MouseEvent
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Drives the real AWT path, so the chart sees exactly what a mouse produces — including the
 * modifier state, which Compose reads from the AWT event's `modifiersEx`. Skipped on a headless JVM.
 */
class ChartCtrlClickTest {

    private val points = (0..100 step 5).map { MetricPoint(step = it, value = 1f - it / 200f) }

    @Test
    fun ctrlLeftClickReportsAStepInsideThePlot() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }

            postClick(queue, target, x = 200, y = 200, button = MouseEvent.BUTTON1, ctrl = true)
            assertTrue(pumpUntil(3_000) { picks.isNotEmpty() }, "Ctrl+left click never reached the chart")
            assertEquals(1, picks.size, "one click should pick exactly one step, got ${picks.toList()}")

            val (step, anchor) = picks.first()
            assertTrue(step in 0f..100f, "picked step $step is outside the data range")
            assertTrue(anchor.x > 0f && anchor.y > 0f, "anchor $anchor was not measured in the window")
        }
    }

    @Test
    fun laterClicksAlongThePlotPickLaterSteps() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }

            postClick(queue, target, x = 140, y = 200, button = MouseEvent.BUTTON1, ctrl = true)
            assertTrue(pumpUntil(3_000) { picks.isNotEmpty() }, "left-side Ctrl+click never reached the chart")
            postClick(queue, target, x = 380, y = 200, button = MouseEvent.BUTTON1, ctrl = true)
            assertTrue(pumpUntil(3_000) { picks.size > 1 }, "right-side Ctrl+click never reached the chart")

            val left = picks[0].first
            val right = picks[1].first
            assertTrue(left < right, "clicking further right picked an earlier step ($left vs $right)")
            assertTrue(left in 5f..45f, "left click picked $left, expected roughly a quarter of the plot")
            assertTrue(right in 70f..100f, "right click picked $right, expected the far end of the plot")
        }
    }

    @Test
    fun aSinglePlainLeftClickIsIgnored() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }

            postClick(queue, target, x = 200, y = 200, button = MouseEvent.BUTTON1, ctrl = false)
            pumpUntil(500) { picks.isNotEmpty() }
            assertTrue(picks.isEmpty(), "a plain click opened the panel: ${picks.toList()}")
        }
    }

    @Test
    fun twoQuickLeftClicksPickAtTheSecondClick() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }
            val base = System.currentTimeMillis()

            postClick(queue, target, x = 140, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 1_000)
            postClick(queue, target, x = 380, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 880)

            assertTrue(pumpUntil(3_000) { picks.isNotEmpty() }, "a double click never reached the chart")
            assertEquals(1, picks.size, "a double click should pick once, got ${picks.toList()}")

            // The anchor and the step come from the second click (x = 380), not from where the pair started.
            val (step, anchor) = picks.first()
            assertTrue(step in 70f..100f, "the second click's position should win, got step $step")
            assertTrue(anchor.x > 0f && anchor.y > 0f, "anchor $anchor was not measured in the window")
        }
    }

    @Test
    fun theTwoClicksMayBeAnywhereOnTheChart() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }
            val base = System.currentTimeMillis()

            // First click in the y-axis gutter, far from the second one: only the timing matters.
            postClick(queue, target, x = 20, y = 260, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 1_000)
            postClick(queue, target, x = 380, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 900)

            assertTrue(pumpUntil(3_000) { picks.isNotEmpty() }, "the second click should have picked")
            assertEquals(1, picks.size)
            assertTrue(picks.first().first in 70f..100f, "got step ${picks.first().first}")
        }
    }

    @Test
    fun aSecondClickInTheGutterHasNothingToPick() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }
            val base = System.currentTimeMillis()

            postClick(queue, target, x = 200, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 1_000)
            postClick(queue, target, x = 20, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 900)

            pumpUntil(500) { picks.isNotEmpty() }
            assertTrue(picks.isEmpty(), "a click outside the plot picked ${picks.toList()}")
        }
    }

    @Test
    fun leftClicksTooFarApartDoNotPick() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }
            val base = System.currentTimeMillis()

            postClick(queue, target, x = 140, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 2_000)
            postClick(queue, target, x = 380, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 1_000)

            pumpUntil(500) { picks.isNotEmpty() }
            assertTrue(picks.isEmpty(), "slow separate clicks picked ${picks.toList()}")
        }
    }

    @Test
    fun aThirdQuickClickDoesNotPickAgain() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }
            val base = System.currentTimeMillis()

            repeat(3) { index ->
                postClick(
                    queue,
                    target,
                    x = 200,
                    y = 200,
                    button = MouseEvent.BUTTON1,
                    ctrl = false,
                    whenMillis = base - 1_000 + index * 100L,
                )
            }

            assertTrue(pumpUntil(3_000) { picks.isNotEmpty() }, "the pair should have picked once")
            pumpUntil(300) { picks.size > 1 }
            assertEquals(1, picks.size, "a third click picked again: ${picks.toList()}")
        }
    }

    @Test
    fun aCtrlClickDoesNotPairWithAFollowingPlainClick() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }
            val base = System.currentTimeMillis()

            postClick(queue, target, x = 200, y = 200, button = MouseEvent.BUTTON1, ctrl = true, whenMillis = base - 1_000)
            postClick(queue, target, x = 300, y = 200, button = MouseEvent.BUTTON1, ctrl = false, whenMillis = base - 950)

            assertTrue(pumpUntil(3_000) { picks.isNotEmpty() }, "the Ctrl+click never reached the chart")
            pumpUntil(300) { picks.size > 1 }
            assertEquals(1, picks.size, "the plain click paired with the Ctrl+click: ${picks.toList()}")
        }
    }

    @Test
    fun ctrlRightClickIsIgnored() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }

            repeat(3) { postClick(queue, target, x = 200, y = 200, button = MouseEvent.BUTTON3, ctrl = true) }
            pumpUntil(500) { picks.isNotEmpty() }
            assertTrue(picks.isEmpty(), "a right click opened the panel: ${picks.toList()}")
        }
    }

    @Test
    fun ctrlClickInTheAxisGutterIsIgnored() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }

            repeat(3) { postClick(queue, target, x = 20, y = 200, button = MouseEvent.BUTTON1, ctrl = true) }
            pumpUntil(500) { picks.isNotEmpty() }
            assertTrue(picks.isEmpty(), "a click in the y-axis gutter opened the panel: ${picks.toList()}")
        }
    }

    @Test
    fun ctrlDragIsNotAClick() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }

            repeat(2) {
                postClick(
                    queue,
                    target,
                    x = 200,
                    y = 200,
                    button = MouseEvent.BUTTON1,
                    ctrl = true,
                    dragTo = 320,
                )
            }
            pumpUntil(500) { picks.isNotEmpty() }
            assertTrue(picks.isEmpty(), "panning with Ctrl held opened the panel: ${picks.toList()}")
        }
    }

    @Test
    fun hoverDoesNotPickAStep() {
        withChart { window, picks ->
            val queue = Toolkit.getDefaultToolkit().systemEventQueue
            val target = onEdtGet { pointerTarget(window) }

            repeat(5) {
                queue.postEvent(mouseEvent(target, MouseEvent.MOUSE_MOVED, 200, 200, MouseEvent.NOBUTTON, ctrl = false))
                queue.postEvent(mouseEvent(target, MouseEvent.MOUSE_MOVED, 300, 180, MouseEvent.NOBUTTON, ctrl = false))
            }
            pumpUntil(500) { picks.isNotEmpty() }
            assertTrue(picks.isEmpty(), "hovering opened the panel: ${picks.toList()}")
        }
    }

    private fun withChart(block: (ComposeWindow, MutableList<Pair<Float, Offset>>) -> Unit) {
        if (GraphicsEnvironment.isHeadless()) return

        val picks = Collections.synchronizedList(mutableListOf<Pair<Float, Offset>>())
        val window = onEdtGet {
            ComposeWindow().apply {
                setContent {
                    ChartCard(
                        title = "Train / Avg Loss",
                        points = points,
                        color = Color(0xFFE85D4C),
                        smoothing = 0f,
                        modifier = Modifier.fillMaxSize(),
                        onPickStep = { step, anchor -> picks += step to anchor },
                        // Same setup as the Dashboard's Avg Loss card: hover readout plus the markers
                        // a previous pick would have left behind.
                        showHoverStep = true,
                        pickMarkers = ChartPickMarkers(clickedStep = 50f, matchedStep = 40),
                    )
                }
                setSize(420, 320)
                setLocation(0, 0)
                isVisible = true
            }
        }
        try {
            settle()
            block(window, picks)
        } finally {
            onEdt { window.dispose() }
        }
    }

    /**
     * Compose installs its pointer listeners on the Skia layer nested inside the window, not on the
     * content pane, so posted events must be addressed to that component to reach `pointerInput`.
     */
    private fun pointerTarget(window: ComposeWindow): Component {
        var found: Component? = null
        fun walk(component: Component) {
            if (component.mouseListeners.isNotEmpty() && component.mouseMotionListeners.isNotEmpty()) {
                found = component
            }
            (component as? Container)?.components?.forEach { walk(it) }
        }
        walk(window.contentPane)
        return found ?: window.contentPane
    }

    private fun postClick(
        queue: EventQueue,
        target: Component,
        x: Int,
        y: Int,
        button: Int,
        ctrl: Boolean,
        dragTo: Int? = null,
        whenMillis: Long = System.currentTimeMillis(),
    ) {
        queue.postEvent(mouseEvent(target, MouseEvent.MOUSE_PRESSED, x, y, button, ctrl, whenMillis))
        if (dragTo != null) {
            queue.postEvent(mouseEvent(target, MouseEvent.MOUSE_DRAGGED, dragTo, y, button, ctrl, whenMillis + 1))
        }
        queue.postEvent(
            mouseEvent(target, MouseEvent.MOUSE_RELEASED, dragTo ?: x, y, button, ctrl = false, whenMillis = whenMillis + 2),
        )
    }

    private fun mouseEvent(
        target: Component,
        id: Int,
        x: Int,
        y: Int,
        button: Int,
        ctrl: Boolean,
        whenMillis: Long = System.currentTimeMillis(),
    ): MouseEvent {
        var mask = if (ctrl) InputEvent.CTRL_DOWN_MASK else 0
        if (id == MouseEvent.MOUSE_PRESSED || id == MouseEvent.MOUSE_DRAGGED) {
            mask = mask or when (button) {
                MouseEvent.BUTTON1 -> MouseEvent.BUTTON1_DOWN_MASK
                MouseEvent.BUTTON3 -> MouseEvent.BUTTON3_DOWN_MASK
                else -> 0
            }
        }
        return MouseEvent(target, id, whenMillis, mask, x, y, 1, false, button)
    }

    private fun onEdt(block: () -> Unit) {
        onEdtGet { block() }
    }

    /**
     * Runs [block] on the AWT event thread but never waits forever: a wedged event thread (an
     * unusable display) fails the test instead of hanging the whole suite.
     */
    private fun <T> onEdtGet(block: () -> T): T {
        var result: T? = null
        var failure: Throwable? = null
        val done = CountDownLatch(1)
        SwingUtilities.invokeLater {
            try {
                result = block()
            } catch (t: Throwable) {
                failure = t
            } finally {
                done.countDown()
            }
        }
        check(done.await(EDT_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
            "the AWT event thread did not respond within ${EDT_TIMEOUT_MS} ms"
        }
        failure?.let { throw it }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }

    /** Let Compose attach the panel and produce its first frame before posting events. */
    private fun settle() {
        repeat(20) {
            onEdt { }
            Thread.sleep(10)
        }
    }

    private fun pumpUntil(timeoutMs: Long, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return true
            onEdt { }
            Thread.sleep(10)
        }
        return condition()
    }

    private companion object {
        const val EDT_TIMEOUT_MS = 5_000L
    }
}

package com.acite.axlranko.util

import androidx.compose.ui.awt.ComposeWindow
import java.awt.GraphicsEnvironment
import java.awt.Rectangle
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The desktop window the Utils WM tab commands: the toggle sizes the window to its screen and back
 * (a compositor that hosts no window manager never acts on a maximize request), and Exit is the quit
 * callback the entry point hands in — which is where the exit guard is armed.
 *
 * Needs a display to have a frame at all; skipped on a headless JVM.
 */
class DesktopAppWindowTest {

    /** Every screen this JVM sees; the window has to land exactly on one of them. */
    private fun screens(): List<Rectangle> =
        GraphicsEnvironment.getLocalGraphicsEnvironment().screenDevices
            .map { it.defaultConfiguration.bounds }

    private fun <T> onEdt(block: () -> T): T {
        var result: T? = null
        SwingUtilities.invokeAndWait { result = block() }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }

    private fun withWindow(block: (ComposeWindow) -> Unit) {
        val window = onEdt { ComposeWindow() }
        try {
            block(window)
        } finally {
            onEdt { window.dispose() }
        }
    }

    @Test
    fun theToggleSizesTheWindowToAScreenAndBack() {
        if (GraphicsEnvironment.isHeadless()) return
        withWindow { window ->
            val controls = DesktopAppWindow(window) {}
            val original = onEdt { window.bounds }
            assertFalse(controls.maximized)

            assertTrue(controls.toggleMaximized())
            val filled = onEdt { window.bounds }
            assertTrue(filled in screens(), "window $filled is not one of the screens ${screens()}")
            assertTrue(controls.maximized)

            assertFalse(controls.toggleMaximized())
            assertEquals(original, onEdt { window.bounds })
            assertFalse(controls.maximized)
        }
    }

    @Test
    fun aWindowThatAlreadyFillsItsScreenStillOffersMaximize() {
        if (GraphicsEnvironment.isHeadless()) return
        withWindow { window ->
            val controls = DesktopAppWindow(window) {}
            // Nothing was remembered, so there is no Restore to offer and the label stays Maximize.
            onEdt { window.bounds = screens().first() }

            assertFalse(controls.maximized)
            assertTrue(controls.toggleMaximized())
        }
    }

    @Test
    fun exitGoesThroughTheQuitCallback() {
        if (GraphicsEnvironment.isHeadless()) return
        withWindow { window ->
            var quits = 0
            DesktopAppWindow(window) { quits++ }.exit()
            assertEquals(1, quits)
        }
    }
}

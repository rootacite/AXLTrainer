package com.acite.axlranko.util

import androidx.compose.ui.awt.ComposeWindow
import java.awt.EventQueue
import java.awt.Frame
import java.awt.GraphicsEnvironment
import java.awt.Rectangle

/**
 * The desktop window, filled by geometry: the screen's own bounds are applied to the window instead
 * of a maximized frame being asked for. A compositor that hosts no window manager (cage) has nothing
 * that acts on a maximize request — neither `_NET_WM_STATE_MAXIMIZED` nor its xdg-shell equivalent —
 * while a size the client applies itself is a size such a session does have to give.
 */
class DesktopAppWindow(
    private val window: ComposeWindow,
    private val onQuit: () -> Unit,
) : AppWindow {
    /** The bounds the window had before this tab filled it, so Restore has somewhere to go. */
    private var restoreBounds: Rectangle? = null

    override val maximized: Boolean
        get() = onEdt { restoreBounds != null && window.bounds.size == screenBounds().size }

    override fun toggleMaximized(): Boolean = onEdt {
        val screen = screenBounds()
        val before = restoreBounds
        if (before != null && window.bounds.size == screen.size) {
            window.bounds = before
            restoreBounds = null
            false
        } else {
            restoreBounds = window.bounds
            // A window manager that honors maximize would otherwise keep the size it picked.
            window.extendedState = Frame.NORMAL
            window.bounds = screen
            true
        }
    }

    override fun exit() = onQuit()

    /**
     * The screen this window is on: the one under its centre, or the default screen while it has no
     * peer yet. `defaultScreenDevice` alone would not do — on X11 with several monitors it is the
     * whole virtual desktop, and filling that is not what this button means.
     */
    private fun screenBounds(): Rectangle {
        val environment = GraphicsEnvironment.getLocalGraphicsEnvironment()
        val bounds = window.bounds
        val centreX = bounds.x + bounds.width / 2
        val centreY = bounds.y + bounds.height / 2
        val under = environment.screenDevices.firstOrNull {
            it.defaultConfiguration.bounds.contains(centreX, centreY)
        }
        return (under ?: environment.defaultScreenDevice).defaultConfiguration.bounds
    }

    /** Frame state belongs to the event thread, which is where a Compose button already runs. */
    private fun <T> onEdt(block: () -> T): T {
        if (EventQueue.isDispatchThread()) return block()
        var result: T? = null
        EventQueue.invokeAndWait { result = block() }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }
}

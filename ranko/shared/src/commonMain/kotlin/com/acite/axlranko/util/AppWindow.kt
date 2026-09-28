package com.acite.axlranko.util

import androidx.compose.runtime.ProvidableCompositionLocal
import androidx.compose.runtime.staticCompositionLocalOf

/**
 * The window this build runs in, for a session whose compositor draws no decorations — cage, or any
 * other single-window kiosk: with no title bar there is nothing on screen to maximize or close from,
 * so the Utils **WM** tab does it instead.
 *
 * `null` where the build owns no OS window (the wasm companion). The tab is offered only when a
 * window is installed, which is what keeps it desktop-only.
 */
interface AppWindow {
    /**
     * Whether the window fills its screen because this tab filled it — the state the button's label
     * follows. A window that merely happens to fill the screen is not this: Restore would have
     * nowhere to go.
     */
    val maximized: Boolean

    /** Size the window to its screen, or put the size it had before back. */
    fun toggleMaximized(): Boolean

    /** Quit the application, along the same path a window-close request takes. */
    fun exit()
}

val LocalAppWindow: ProvidableCompositionLocal<AppWindow?> = staticCompositionLocalOf { null }

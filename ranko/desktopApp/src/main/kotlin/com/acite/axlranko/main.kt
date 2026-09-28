package com.acite.axlranko

import AppGraph
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import axlranko.desktopapp.generated.resources.Res
import axlranko.desktopapp.generated.resources.app_icon
import com.acite.axlranko.util.DesktopAppWindow
import com.acite.axlranko.util.ProcessExitGuard
import dev.zacsweers.metro.createGraph
import java.awt.Dimension
import org.jetbrains.compose.resources.painterResource

fun main() {
    val appGraph = createGraph<AppGraph>()
    // Covers every exit path a close request does not: SIGTERM, and a shutdown that never returns.
    ProcessExitGuard.install()

    application {
        // The last moment this JVM is known to be healthy — a hang in teardown would otherwise
        // leave a windowless process holding the GPU nodes and its IPC child. The close request and
        // the Utils WM tab's Exit both come through here.
        val quit: () -> Unit = {
            ProcessExitGuard.armOnce()
            exitApplication()
        }
        Window(
            onCloseRequest = quit,
            title = "AxlRanko",
            icon = painterResource(Res.drawable.app_icon),
            state = rememberWindowState(size = DpSize(1600.dp, 900.dp))
        ) {
            window.minimumSize = Dimension(350, 600)

            App(
                metroVmf = appGraph.metroViewModelFactory,
                appearanceRepo = appGraph.appearanceRepository,
                blobStore = appGraph.blobStore,
                pathPicker = appGraph.pathPicker,
                appWindow = DesktopAppWindow(window, quit),
            )
        }
    }
}
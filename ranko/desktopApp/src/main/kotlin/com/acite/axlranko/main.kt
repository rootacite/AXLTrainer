package com.acite.axlranko

import AppGraph
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import axlranko.desktopapp.generated.resources.Res
import axlranko.desktopapp.generated.resources.app_icon
import dev.zacsweers.metro.createGraph
import java.awt.Dimension
import org.jetbrains.compose.resources.painterResource

fun main() {
    val appGraph = createGraph<AppGraph>()

    application {
        Window(
            onCloseRequest = ::exitApplication,
            title = "AxlRanko",
            icon = painterResource(Res.drawable.app_icon),
            state = rememberWindowState(size = DpSize(1600.dp, 900.dp))
        ) {
            window.minimumSize = Dimension(350, 600)

            App(appGraph.metroViewModelFactory, appGraph.appearanceRepository)
        }
    }
}
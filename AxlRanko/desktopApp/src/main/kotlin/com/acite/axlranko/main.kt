package com.acite.axlranko

import AppGraph
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import dev.zacsweers.metro.createGraph
import java.awt.Dimension

fun main() {
    val appGraph = createGraph<AppGraph>()

    application {
        Window(
            onCloseRequest = ::exitApplication,
            title = "AxlRanko",
            state = rememberWindowState(size = DpSize(1600.dp, 900.dp))
        ) {
            window.minimumSize = Dimension(350, 600)

            App(appGraph.metroViewModelFactory)
        }
    }
}


import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import dev.zacsweers.metro.createGraph
import java.awt.Dimension
import org.acitelight.axledge.App
import org.acitelight.axledge.AppGraphCio

fun main() {
    val appGraph = createGraph<AppGraphCio>()

    application {
        Window(
            title = "AxlEdge",
            state = rememberWindowState(width = 800.dp, height = 600.dp),
            onCloseRequest = ::exitApplication,
        ) {
            window.minimumSize = Dimension(350, 600)

            App(appGraph.metroViewModelFactory)
        }
    }
}
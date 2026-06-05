
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.window.ComposeViewport
import dev.zacsweers.metro.createGraph
import org.acitelight.axledge.App
import org.acitelight.axledge.AppGraphJs

@OptIn(ExperimentalComposeUiApi::class)
fun main() {
    val addGraph = createGraph<AppGraphJs>()
    ComposeViewport { App(addGraph.metroViewModelFactory) }
}

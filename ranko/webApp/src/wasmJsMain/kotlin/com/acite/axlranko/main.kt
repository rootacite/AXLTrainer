import AppGraph
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.window.ComposeViewport
import com.acite.axlranko.App
import dev.zacsweers.metro.createGraph

@OptIn(ExperimentalComposeUiApi::class)
fun main() {
    val appGraph = createGraph<AppGraph>()
    ComposeViewport(viewportContainerId = "webApp") {
        App(
            metroVmf = appGraph.metroViewModelFactory,
            appearanceRepo = appGraph.appearanceRepository,
            blobStore = appGraph.blobStore,
            pathPicker = appGraph.pathPicker,
        )
    }
}

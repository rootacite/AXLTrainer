package com.acite.axlranko.pages

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.DatasetCountEntry
import com.acite.axlranko.model.DatasetCountsResponse
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.UtilsUiState
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import java.util.Collections
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The Training section's step line, composed in a real window in each of its shapes. Its arithmetic
 * is `StepEstimateTest`'s; this is for the layout: a two-line estimate, a count in flight, a folder
 * the helper could not read (with its Retry button) and a form whose numbers are not filled in yet.
 *
 * Needs a display; skipped where the JVM has none.
 */
class StepEstimateLineRenderTest {

    private val failures = Collections.synchronizedList(mutableListOf<Throwable>())

    private fun form(epochs: String = "20", batch: String = "2", grad: String = "5") = TrainingConfigForm(
        epoch = epochs,
        trainBatchSize = batch,
        gradientAccumulationSteps = grad,
    )

    private fun counts(samples: Int = 3_100, images: Int = 100, error: String? = null) =
        DatasetCountsResponse(
            entries = listOf(DatasetCountEntry(path = "/data/a", repeat = 3, images = images, error = error)),
            images = images,
            samples = samples,
        )

    private fun render(states: List<Pair<UtilsUiState, TrainingConfigForm>>) {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        try {
            var state by mutableStateOf(states.first())
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(720, 240)
                    setContent {
                        RankoTheme {
                            val (uiState, form) = state
                            Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
                                StepEstimateLine(uiState = uiState, form = form, onRetry = {})
                            }
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            states.forEach { next ->
                onEdtGet { state = next }
                pumpFor(200)
            }
            onEdtGet { window.dispose() }
            assertTrue(
                failures.isEmpty(),
                failures.joinToString("\n\n") { it.stackTraceToString() },
            )
        } finally {
            Thread.setDefaultUncaughtExceptionHandler(previous)
        }
    }

    private fun <T> onEdtGetResult(block: () -> T): T {
        var result: T? = null
        SwingUtilities.invokeAndWait { result = block() }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }

    private fun onEdtGet(block: () -> Unit) {
        SwingUtilities.invokeAndWait(block)
    }

    private fun pumpFor(millis: Long) {
        val deadline = System.currentTimeMillis() + millis
        while (System.currentTimeMillis() < deadline) {
            SwingUtilities.invokeAndWait { }
            Thread.sleep(10)
        }
    }

    @Test
    fun everyShapeOfTheEstimateComposes() {
        render(
            listOf(
                // Nothing counted yet: the line waits rather than guessing.
                UtilsUiState() to form(),
                // Counting.
                UtilsUiState(datasetCountsLoading = true) to form(),
                // The estimate itself.
                UtilsUiState(datasetCounts = counts()) to form(),
                // A long total, and a form whose epoch field is still empty.
                UtilsUiState(datasetCounts = counts(samples = 1_200_000, images = 40_000)) to form(epochs = ""),
                // A folder the helper could not read: the reason plus Retry.
                UtilsUiState(datasetCounts = counts(images = 0, samples = 0, error = "not a directory")) to form(),
                // The count call itself failed.
                UtilsUiState(datasetCountsError = "helper is gone") to form(),
            ),
        )
    }
}

package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.model.TrainTrainingProgress
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import java.util.Collections
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The run selector, its menu entries and the control card are composed for real, in a window,
 * against fixture history: layout crashes that only show up at measurement time (an unbounded
 * scroll, a row measured with infinite width) fail here instead of on the user's screen. Long
 * run ids and the summary line are part of the fixture for that reason.
 *
 * Skipped on a headless JVM; the window is placed far off-screen so it never flashes in view.
 * The menu is drawn by composing its entries the way `DropdownMenu` does — a synthetic AWT click
 * does not reach Compose's pointer input, so opening the popup itself cannot be driven here.
 */
class DashboardRunHistoryTest {

    private val failures = Collections.synchronizedList(mutableListOf<Throwable>())

    private val runs = listOf(
        RunSummary(
            runId = "Tsukuyomi_20260928_110928",
            outputName = "Tsukuyomi",
            hasOutput = true,
            hasLog = false,
            lastStep = 4500,
            samples = 12,
            checkpoints = 46,
            sizeBytes = 11_172_201_792,
            modified = 1_790_587_779.5,
            current = true,
            live = true,
        ),
        RunSummary(
            runId = "Kirika_20260927_225224",
            outputName = "Kirika",
            hasOutput = true,
            hasLog = true,
            lastStep = 3300,
            checkpoints = 34,
            sizeBytes = 8_200_000_000,
            modified = 1_790_500_000.0,
        ),
        // A run id long enough to need the ellipsis, and no step at all.
        RunSummary(
            runId = "a_very_long_character_name_that_keeps_going_20260911_120000_2",
            outputName = "a_very_long_character_name_that_keeps_going",
            modified = 1_790_000_000.0,
        ),
    )

    private fun onEdtGet(block: () -> Unit) = SwingUtilities.invokeAndWait(block)

    private fun <T> onEdtGetResult(block: () -> T): T {
        var result: T? = null
        SwingUtilities.invokeAndWait { result = block() }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }

    private fun pumpFor(millis: Long) {
        val deadline = System.currentTimeMillis() + millis
        while (System.currentTimeMillis() < deadline) {
            SwingUtilities.invokeAndWait { }
            Thread.sleep(10)
        }
    }

    @Test
    fun selectorMenuAndControlCardCompose() {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        try {
            var selected by mutableStateOf<RunSummary?>(null)
            var controlsEnabled by mutableStateOf(true)
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(1000, 760)
                    setContent {
                        RankoTheme {
                            Column(modifier = Modifier.padding(12.dp).width(520.dp)) {
                                RunSelector(
                                    runs = runs,
                                    selected = selected,
                                    resolvedRunId = runs.first().runId,
                                    onSelect = { id -> selected = runs.firstOrNull { it.runId == id } },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                                // The menu's own entries, laid out the way the popup lays them out.
                                RunMenuItem(
                                    title = "Current run",
                                    subtitle = runs.first().runId,
                                    badges = { Text(runStateLabel(runs.first()) ?: "") },
                                    chosen = selected == null,
                                    onClick = { selected = null },
                                )
                                runs.forEach { run ->
                                    RunMenuItem(
                                        title = run.runId,
                                        subtitle = runDetailLabel(run),
                                        badges = { Text(runStateLabel(run) ?: "") },
                                        chosen = selected?.runId == run.runId,
                                        onClick = { selected = run },
                                    )
                                }
                                TrainControlCard(
                                    status = TrainStatus(
                                        status = "finished",
                                        runId = runs.first().runId,
                                        outputName = "Tsukuyomi",
                                        pid = 4242,
                                        startedAt = 1_790_580_000.0,
                                        training = TrainTrainingProgress(
                                            step = 4500,
                                            totalSteps = 4500,
                                            epoch = 16,
                                            epochs = 16,
                                        ),
                                    ),
                                    shownRun = selected ?: runs.first(),
                                    commandInFlight = false,
                                    controlsEnabled = controlsEnabled,
                                    outputDir = "/out",
                                    loggingDir = "/logs",
                                    onStart = {},
                                    onPause = {},
                                    onResume = {},
                                    onStop = {},
                                    onReset = {},
                                )
                            }
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            pumpFor(700)

            // A pinned past run turns the controls off; both states have to lay out.
            onEdtGet {
                selected = runs[1]
                controlsEnabled = false
            }
            pumpFor(600)
            onEdtGet {
                selected = null
                controlsEnabled = true
            }
            pumpFor(600)

            onEdtGet { window.dispose() }
            assertTrue(
                failures.isEmpty(),
                failures.joinToString("\n\n") { it.stackTraceToString() },
            )
        } finally {
            Thread.setDefaultUncaughtExceptionHandler(previous)
        }
    }

    /** The no-run-recorded state, and the control card naming a run that left no logs or samples. */
    @Test
    fun emptyHistoryAndABareRunCompose() {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        try {
            val bare = RunSummary(runId = "Kirika_20260927_225224", outputName = "Kirika")
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(900, 620)
                    setContent {
                        RankoTheme {
                            Column(modifier = Modifier.padding(12.dp).width(520.dp)) {
                                RunSelector(
                                    runs = emptyList(),
                                    selected = null,
                                    resolvedRunId = null,
                                    onSelect = {},
                                    modifier = Modifier.fillMaxWidth(),
                                )
                                RunMenuItem(
                                    title = "Current run",
                                    subtitle = "Nothing started yet",
                                    badges = {},
                                    chosen = true,
                                    onClick = {},
                                )
                                TrainControlCard(
                                    status = TrainStatus(),
                                    shownRun = bare,
                                    commandInFlight = false,
                                    controlsEnabled = false,
                                    outputDir = "/out",
                                    loggingDir = "/logs",
                                    onStart = {},
                                    onPause = {},
                                    onResume = {},
                                    onStop = {},
                                    onReset = {},
                                )
                            }
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            pumpFor(700)
            onEdtGet { window.dispose() }
            assertTrue(
                failures.isEmpty(),
                failures.joinToString("\n\n") { it.stackTraceToString() },
            )
        } finally {
            Thread.setDefaultUncaughtExceptionHandler(previous)
        }
    }
}

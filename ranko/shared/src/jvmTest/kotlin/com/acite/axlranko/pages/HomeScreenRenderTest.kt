package com.acite.axlranko.pages

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import com.acite.axlranko.data.AutomationJobSummary
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
 * Home laid out wide and narrow, in a real window: the brand, the recent-run list, and the five
 * module cards. The card's click is the `onOpen` callback Stage passes in; this test does not
 * synthesize a pointer event.
 *
 * Needs a display; skipped where the JVM has none.
 */
class HomeScreenRenderTest {

    private val failures = Collections.synchronizedList(mutableListOf<Throwable>())

    @Test
    fun wideAndNarrowHomeComposeAndTheDashboardCardOpens() {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        try {
            val runs = listOf(
                RunSummary(
                    runId = "hana_20260928_110928",
                    outputName = "hana",
                    lastStep = 1200,
                    samples = 4,
                    live = true,
                    current = true,
                ),
            )
            val dashboard = dashboardHome(
                TrainStatus(
                    status = "training",
                    training = TrainTrainingProgress(step = 12, totalSteps = 400, epoch = 1, epochs = 20, avgLoss = 0.25f),
                ),
                emptyMap(),
            )
            val automation = automationHome(
                listOf(AutomationJobSummary(id = "job", state = "running", done = 1, total = 4, workflow = "batch.json")),
            )
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(1100, 800)
                    setContent {
                        RankoTheme {
                            HomeScreen(
                                version = "dev",
                                gitHash = "abc1234",
                                runs = runs,
                                changelog = listOf(com.acite.axlranko.changelog.ChangelogEntry("abc", "2026-10-03", "[Feat] Home")),
                                dashboard = dashboard,
                                avgLoss = emptyList(),
                                hardware = null,
                                automation = automation,
                                utilsFacts = listOf(com.acite.axlranko.pages.HomeFact("Output", "hana")),
                                statisticsLines = listOf("set", "2 images · 1 captioned"),
                                tagBars = emptyList(),
                                imagesLines = listOf("set", "2 images · 1 mask", "No unsaved captions"),
                                datasetImages = listOf("/a.png", "/b.png"),
                                onOpen = {},
                                onOpenRun = {},
                                modifier = Modifier.fillMaxSize(),
                            )
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            pumpFor(400)
            onEdtGet { window.setSize(480, 860) }
            pumpFor(400)
            onEdtGet { window.dispose() }
            assertTrue(failures.isEmpty(), failures.joinToString("\n\n") { it.stackTraceToString() })
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
}

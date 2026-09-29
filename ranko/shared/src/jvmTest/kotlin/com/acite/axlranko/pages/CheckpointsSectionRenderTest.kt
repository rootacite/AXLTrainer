package com.acite.axlranko.pages

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.CheckpointExport
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.pages.components.JOB_DONE
import com.acite.axlranko.pages.components.JOB_MODE_BATCH
import com.acite.axlranko.pages.components.JOB_MODE_SETS
import com.acite.axlranko.pages.components.JOB_RUNNING
import com.acite.axlranko.pages.components.CheckpointRow
import com.acite.axlranko.pages.components.checkpointRowKey
import com.acite.axlranko.pages.components.runningBatch
import com.acite.axlranko.pages.components.checkpointRows
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import java.util.Collections
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The Dashboard's Checkpoints section is composed for real, in a window, against fixture state: a
 * card with no images, a card whose pass is still rendering, a `samples only` card and the empty
 * state are all measured here rather than on the user's screen. Skipped on a headless JVM; the
 * window is placed far off-screen so it never flashes in view.
 */
class CheckpointsSectionRenderTest {

    private val failures = Collections.synchronizedList(mutableListOf<Throwable>())

    /** One page state: the run's checkpoints, its samples by step, and its generation jobs. */
    private data class Case(
        val checkpoints: List<CheckpointItem>,
        val samples: Map<String, List<SampleItem>> = emptyMap(),
        val jobs: List<GeneratedSampleJob> = emptyList(),
        val error: String? = null,
        val generatingPath: String? = null,
        val gpuFree: Boolean = true,
        val startingBatch: Boolean = false,
        val pinnedPaths: Set<String> = emptySet(),
        val pinningPath: String? = null,
        val pinsError: String? = null,
        /** The checkpoint whose Save As is open or copying, and where the last one landed. */
        val exportingPath: String? = null,
        val exportResult: CheckpointExport? = null,
    )

    private fun checkpoint(step: Int, final: Boolean = false) = CheckpointItem(
        path = "/out/rein_20260911_120000/rein_s$step/rein.safetensors",
        runId = "rein_20260911_120000",
        dir = if (final) "rein_final" else "rein_s$step",
        filename = "rein.safetensors",
        step = step,
        final = final,
        sizeBytes = 24_000_000,
        networkDim = 32,
        networkAlpha = 16,
        outputName = "rein",
    )

    private fun sample(step: Int, set: Int, repeat: Int = 0) = SampleItem(
        filename = "rein_${step.toString().padStart(6, '0')}_p${set}_$repeat.png",
        setIndex = set,
        repeatIdx = repeat,
        path = "/out/rein_20260911_120000/rein_samples/rein_${step.toString().padStart(6, '0')}_p${set}_$repeat.png",
    )

    private fun batchJob(id: String, index: Int, total: Int, images: Int = 48, done: Int = 12) =
        GeneratedSampleJob(
            id = id,
            state = JOB_RUNNING,
            mode = JOB_MODE_BATCH,
            fromStep = 100,
            toStep = 600,
            checkpointIndex = index,
            totalCheckpoints = total,
            imagesDone = done,
            totalImages = images,
        )

    private fun setsJob(id: String, step: Int, state: String, images: Int = 6, done: Int = 0) =
        GeneratedSampleJob(
            id = id,
            state = state,
            mode = JOB_MODE_SETS,
            step = step,
            checkpoint = checkpoint(step).path,
            files = (0 until done).map {
                "/out/rein_20260911_120000/rein_samples/generated/${id}_p0_$it.png"
            },
            imagesDone = done,
            totalImages = images,
            currentSet = if (state == JOB_RUNNING) 2 else 6,
            totalSets = 6,
            currentStep = if (state == JOB_RUNNING) 17 else 35,
            totalSteps = 35,
        )

    private fun render(cases: List<Case>) {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        try {
            var state by mutableStateOf(cases.first())
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(1280, 900)
                    setContent {
                        RankoTheme {
                            val current = state
                            // The page's own structure: the error line, the empty card, else one
                            // lazy item per checkpoint card.
                            LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
                                val rows = checkpointRows(
                                    current.checkpoints,
                                    current.samples,
                                    current.jobs,
                                    pinned = current.pinnedPaths,
                                )
                                item {
                                    SampleRangeRow(
                                        rows = rows,
                                        batch = runningBatch(current.jobs),
                                        starting = current.startingBatch,
                                        canStart = current.gpuFree,
                                        note = current.error,
                                        onSampleRange = { _, _ -> },
                                        onCancel = {},
                                    )
                                }
                                if (rows.any { it.pinned }) {
                                    item { PinnedHintLine("/logs/rein_20260911_120000/checkpoint_pins.json") }
                                }
                                current.pinsError?.let { message -> item { PinsErrorLine(message) } }
                                if (rows.isEmpty()) {
                                    item { CheckpointsEmptyCard() }
                                } else {
                                    val pinnedCards = rows.filter { it.pinned }
                                    val restCards = rows.filterNot { it.pinned }
                                    val card: @Composable (CheckpointRow) -> Unit = { row ->
                                        CheckpointRowCard(
                                            row = row,
                                            thumbSize = 120f,
                                            showSetBadges = true,
                                            newJobIds = current.jobs.map { it.id }.toSet(),
                                            gpuFree = current.gpuFree,
                                            starting = current.generatingPath == row.checkpoint?.path,
                                            busyElsewhere = current.generatingPath != null &&
                                                current.generatingPath != row.checkpoint?.path,
                                            pinning = current.pinningPath == row.checkpoint?.path,
                                            pinEnabled = current.pinningPath == null,
                                            exportInFlightPath = current.exportingPath,
                                            exportResult = current.exportResult,
                                            onOpen = {},
                                            onGenerate = {},
                                            onTogglePin = {},
                                            onSaveAs = {},
                                        )
                                    }
                                    items(pinnedCards, key = { checkpointRowKey(it) }) { row -> card(row) }
                                    if (pinnedCards.isNotEmpty() && restCards.isNotEmpty()) {
                                        item {
                                            CheckpointsDivider(
                                                pinned = pinnedCards.size,
                                                rest = restCards.size,
                                            )
                                        }
                                    }
                                    items(restCards, key = { checkpointRowKey(it) }) { row -> card(row) }
                                }
                            }
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            cases.forEach { next ->
                onEdtGet { state = next }
                pumpFor(400)
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
    fun everyCardShapeComposes() {
        render(
            listOf(
                // A checkpoint with its own samples and a finished pass riding along.
                Case(
                    checkpoints = listOf(checkpoint(3050)),
                    samples = mapOf("3050" to listOf(sample(3050, 0), sample(3050, 1))),
                    jobs = listOf(setsJob("done_sets_gen_1", 3050, JOB_DONE, done = 6)),
                ),
                // A checkpoint with nothing yet — the card the sampling switch exists for.
                Case(checkpoints = listOf(checkpoint(3100))),
                // A checkpoint with nothing yet while a pass is rendering, and one while the GPU
                // is busy with training (button off, hint instead).
                Case(
                    checkpoints = listOf(checkpoint(3100)),
                    jobs = listOf(setsJob("live_sets_gen_2", 3100, JOB_RUNNING, done = 1)),
                    gpuFree = false,
                ),
                Case(checkpoints = listOf(checkpoint(3100)), gpuFree = false),
                Case(checkpoints = listOf(checkpoint(3100)), generatingPath = checkpoint(3100).path),
                // A final checkpoint, and a step whose weights are gone but whose images stayed.
                Case(checkpoints = listOf(checkpoint(3200, final = true))),
                Case(checkpoints = emptyList(), samples = mapOf("3000" to listOf(sample(3000, 0)))),
                // A failed pass, and an empty page.
                Case(
                    checkpoints = listOf(checkpoint(3050)),
                    samples = mapOf("3050" to listOf(sample(3050, 0))),
                    error = "RuntimeError: base model is gone",
                ),
                Case(checkpoints = emptyList()),
                // A range batch in flight: the progress line with its Stop button, above the cards
                // (each showing the pass that has already reached it).
                Case(
                    checkpoints = listOf(checkpoint(3050), checkpoint(3000)),
                    samples = mapOf("3000" to listOf(sample(3000, 0))),
                    jobs = listOf(
                        batchJob("rein_s100-600_batch_gen_1", index = 3, total = 8),
                        setsJob("done_sets_gen_2", 3000, JOB_DONE, done = 6),
                        setsJob("live_sets_gen_3", 3050, JOB_RUNNING, done = 1),
                    ),
                ),
                // The same batch being stopped, and a range that is only starting.
                Case(
                    checkpoints = listOf(checkpoint(3050)),
                    jobs = listOf(
                        batchJob("rein_s100-600_batch_gen_4", index = 5, total = 8).copy(cancelRequested = true),
                    ),
                ),
                Case(checkpoints = listOf(checkpoint(3050)), startingBatch = true),
                // Pinned cards leading the section, one of them mid-pin and one whose pin the
                // helper refused.
                Case(
                    checkpoints = listOf(checkpoint(3050), checkpoint(3000), checkpoint(2950)),
                    pinnedPaths = setOf(checkpoint(2950).path, checkpoint(3000).path),
                ),
                Case(
                    checkpoints = listOf(checkpoint(3050), checkpoint(3000)),
                    pinnedPaths = setOf(checkpoint(3050).path),
                    pinningPath = checkpoint(3000).path,
                ),
                Case(
                    checkpoints = listOf(checkpoint(3050)),
                    pinsError = "ValueError: no run to pin a checkpoint to",
                ),
                // Save As: open on one card, landed on another, failed on a third.
                Case(
                    checkpoints = listOf(checkpoint(3050), checkpoint(3000)),
                    exportResult = CheckpointExport(
                        path = checkpoint(3050).path,
                        savedPath = "/home/me/loras/rein_s003050.safetensors (24.0 MB)",
                    ),
                    exportingPath = checkpoint(3000).path,
                ),
                Case(
                    checkpoints = listOf(checkpoint(3050)),
                    exportResult = CheckpointExport(
                        path = checkpoint(3050).path,
                        error = "ValueError: destination is the same file as the source",
                    ),
                ),
            ),
        )
    }
}

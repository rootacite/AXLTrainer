package com.acite.axlranko.pages

import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.SamplePromptsResponse
import com.acite.axlranko.model.SampleSetInfo
import com.acite.axlranko.pages.components.PAGE_PANEL_MARGIN
import com.acite.axlranko.pages.components.SamplePromptsEditorDialog
import com.acite.axlranko.pages.components.SamplingPromptsSection
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import java.util.Collections
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The Sampling Prompts section and its editor are composed for real, in a window, against fixture
 * state: a run with one set, one with several, an edited run that says so, a live one, one whose
 * config could not be read, and the editor over a short window where its content has to scroll
 * inside the room the page hands it.
 *
 * Skipped on a headless JVM; the window is placed far off-screen so it never flashes in view.
 */
class SamplingPromptsRenderTest {

    private val failures = Collections.synchronizedList(mutableListOf<Throwable>())

    private fun set(prompt: String = "1girl, solo, serafuku", repeat: Int = 2) = SampleSetInfo(
        name = "cowgirl",
        prompt = prompt,
        negative = "worst quality, low quality",
        width = 1152,
        height = 768,
        steps = 35,
        guidanceScale = 6.0f,
        guidanceRescale = 0.6f,
        seed = 0,
        repeat = repeat,
    )

    private fun run(
        sets: List<SampleSetInfo> = listOf(set()),
        edited: Boolean = false,
        live: Boolean = false,
        reason: String = "",
    ) = SamplePromptsResponse(
        runId = "rein_20260911_120000",
        outputName = "rein",
        file = if (edited) "/logs/rein_20260911_120000/sample_sets.json" else null,
        edited = edited,
        configSource = if (edited) "/logs/rein_20260911_120000/sample_sets.json" else "/logs/rein_20260911_120000/config.toml",
        sets = sets,
        live = live,
        reason = reason,
    )

    private fun onEdtGet(block: () -> Unit) {
        SwingUtilities.invokeAndWait(block)
    }

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

    private fun render(
        states: List<SamplePromptsResponse?>,
        windowSize: IntSize = IntSize(1200, 900),
        editorOpen: Boolean = false,
        saving: Boolean = false,
        error: String? = null,
        /** Close to the editor's own scroll state, so a test can measure what it laid out. */
        scrollState: ScrollState = ScrollState(0),
        inspect: (ComposeWindow) -> Unit = {},
    ) {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        try {
            var state by mutableStateOf(states.first())
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(windowSize.width, windowSize.height)
                    setContent {
                        RankoTheme {
                            val current = state
                            Box(Modifier.fillMaxSize().padding(16.dp)) {
                                SamplingPromptsSection(
                                    prompts = current,
                                    loading = false,
                                    error = error,
                                    saving = saving,
                                    onEdit = {},
                                    onReset = {},
                                )
                            }
                            val open = current
                            if (editorOpen && open != null) {
                                val room = with(LocalDensity.current) {
                                    val width = windowSize.width.toDp() - PAGE_PANEL_MARGIN
                                    val height = windowSize.height.toDp() - PAGE_PANEL_MARGIN
                                    width to height
                                }
                                SamplePromptsEditorDialog(
                                    prompts = open,
                                    saving = saving,
                                    error = error,
                                    maxWidth = room.first,
                                    maxHeight = room.second,
                                    onSave = {},
                                    onDismiss = {},
                                    scrollState = scrollState,
                                )
                            }
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            states.forEach { next ->
                onEdtGet { state = next }
                pumpFor(600)
            }
            onEdtGet { inspect(window) }
            onEdtGet { window.dispose() }
            assertTrue(failures.isEmpty(), failures.joinToString("\n\n") { it.stackTraceToString() })
        } finally {
            Thread.setDefaultUncaughtExceptionHandler(previous)
        }
    }

    @Test
    fun everySectionShapeComposes() {
        render(
            listOf(
                run(),
                run(sets = listOf(set("a prompt"), set("another prompt", repeat = 3), set("a third"))),
                run(edited = true),
                run(edited = true, live = true),
                run(live = true),
                run(sets = emptyList(), reason = "validation.samples[1]: prompt must not be empty"),
                null,
            ),
        )
    }

    @Test
    fun theEditorComposesOverTheSection() {
        render(
            listOf(run(sets = listOf(set(), set("another prompt", repeat = 3)))),
            editorOpen = true,
        )
    }

    @Test
    fun aLongSetListStillSavesWithTheEditorOpen() {
        val many = (1..6).map { set("prompt number $it, with a few more tags to wrap", repeat = it) }
        render(listOf(run(sets = many)), editorOpen = true, windowSize = IntSize(1100, 760))
    }

    /**
     * The editor on a window that cannot hold it. Its height is capped to the room the page hands
     * it and its content scrolls inside that, so the fields and the Save button stay reachable
     * instead of being drawn past the bottom edge (Compose draws this `Dialog` as a layer of the
     * same window here). Measured on the scroll viewport the composition produced.
     */
    @Test
    fun theEditorScrollsInsideAShortWindow() {
        if (GraphicsEnvironment.isHeadless()) return
        val windowSize = IntSize(820, 420)
        val state = ScrollState(0)
        var viewport = 0
        render(
            listOf(run(sets = (1..4).map { set("prompt $it") })),
            windowSize = windowSize,
            editorOpen = true,
            scrollState = state,
        ) {
            viewport = state.viewportSize
        }
        assertTrue(state.maxValue > 0, "the editor's content did not need scrolling at all")
        assertTrue(
            viewport > 0 && viewport <= windowSize.height,
            "the editor's viewport was $viewport in a ${windowSize.height}px window",
        )
    }
}

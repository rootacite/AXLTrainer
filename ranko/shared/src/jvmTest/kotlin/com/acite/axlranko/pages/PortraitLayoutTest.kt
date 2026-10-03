package com.acite.axlranko.pages

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.ui.wideMetricRow
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.AutomationSection
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.StatisticsUiState
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.model.UtilsUiState
import com.acite.axlranko.pages.components.SparkPoint
import com.acite.axlranko.pages.components.TrainControlCard
import com.acite.axlranko.ui.theme.RankoTheme
import com.acite.axlranko.util.PathPicker
import java.awt.GraphicsEnvironment
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Portrait is a taller-than-wide viewport. These scenes measure the four pages that change shape
 * there: what stays on one line, what becomes a second row, and which labels turn into icons.
 * Skipped on a headless JVM.
 */
class PortraitLayoutTest {

    @Test
    fun metricCardsWrapWhenNarrowerThanHalfTheHeight() {
        assertTrue(wideMetricRow(800.dp, 900.dp))
        assertTrue(wideMetricRow(450.dp, 900.dp))
        assertTrue(!wideMetricRow(400.dp, 900.dp))
        assertTrue(wideMetricRow(1400.dp, 800.dp))
    }


    private class NoopPathPicker : PathPicker {
        override suspend fun pickDirectory(title: String, current: String): String? = null
        override suspend fun pickFile(title: String, current: String, extensions: List<String>?): String? = null
        override suspend fun saveFile(suggestedName: String, current: String): String? = null
        override fun deleteEmptyPlaceholder(path: String) = Unit
    }

    private fun scene(width: Int, height: Int, content: @androidx.compose.runtime.Composable () -> Unit) =
        ImageComposeScene(width = width, height = height, density = Density(1f)) {
            RankoTheme { content() }
        }

    private fun walk(node: SemanticsNode, out: MutableList<SemanticsNode>) {
        out += node
        node.children.forEach { walk(it, out) }
    }

    private fun nodes(scene: ImageComposeScene): List<SemanticsNode> {
        scene.render()
        val found = mutableListOf<SemanticsNode>()
        scene.semanticsOwners.forEach { owner -> walk(owner.rootSemanticsNode, found) }
        return found
    }

    private fun texts(nodes: List<SemanticsNode>): List<Pair<String, Float>> =
        nodes.flatMap { node ->
            val y = node.positionInRoot.y
            node.config.getOrNull(SemanticsProperties.Text).orEmpty().map { it.text to y }
        }

    private fun descriptions(nodes: List<SemanticsNode>): List<String> =
        nodes.flatMap { it.config.getOrNull(SemanticsProperties.ContentDescription).orEmpty() }

    @Test
    fun statisticsPagerShowsOnePageAtATime() {
        if (GraphicsEnvironment.isHeadless()) return
        val scene = scene(420, 700) {
            Box(Modifier.width(420.dp).height(700.dp)) {
                StatisticsDetailPager(
                    uiState = StatisticsUiState(isLoading = false),
                    onSearch = {},
                    onToggleTag = {},
                    modifier = Modifier.fillMaxSize(),
                ) {
                    Column(it) { Text("Logic Mode:") }
                }
            }
        }
        try {
            val first = texts(nodes(scene))
            assertTrue(first.any { it.first == "Dataset Tag Distribution" })
            assertTrue(first.none { it.first == "Logic Mode:" }, "control page was composed beside the tags")
            val tab = nodes(scene).first { node ->
                node.config.getOrNull(SemanticsProperties.Text).orEmpty().any { it.text == "Control Panel" } &&
                    node.config.getOrNull(SemanticsActions.OnClick) != null
            }
            assertTrue(tab.config[SemanticsActions.OnClick].action?.invoke() == true)
            val second = texts(nodes(scene))
            assertTrue(second.any { it.first == "Logic Mode:" }, "Control Panel page did not open")
        } finally {
            scene.close()
        }
    }

    @Test
    fun utilsPortraitPutsTabsOnOneRowAndButtonsUnderTheTitle() {
        if (GraphicsEnvironment.isHeadless()) return
        val scene = scene(420, 400) {
            Column(Modifier.width(420.dp)) {
                SectionNav(
                    uiState = UtilsUiState(isLoading = false),
                    appWindow = null,
                    horizontal = true,
                    onSelect = {},
                )
                ConfigHeader(
                    uiState = UtilsUiState(isLoading = false, configPath = "/tmp/config.toml"),
                    portrait = true,
                    onReload = {},
                    onReset = {},
                    onSave = {},
                )
            }
        }
        try {
            val lines = texts(nodes(scene))
            val environment = lines.first { it.first == "Environment" }.second
            val training = lines.first { it.first == "Training" }.second
            assertEquals(environment, training, 2f)
            val title = lines.first { it.first == "Training Config" }.second
            val reload = lines.first { it.first == "Reload" }.second
            assertTrue(reload > title + 8f, "Reload y=$reload title y=$title")
        } finally {
            scene.close()
        }
    }

    @Test
    fun dashboardPortraitUsesIconsAndStacksTheSparkAndPaths() {
        if (GraphicsEnvironment.isHeadless()) return
        val checkpoint = CheckpointItem(
            path = "/out/rein_20260911_120000/rein_s3050/rein.safetensors",
            runId = "rein_20260911_120000",
            dir = "rein_s3050",
            filename = "rein.safetensors",
            step = 3050,
            sizeBytes = 24_000_000,
            networkDim = 32,
            networkAlpha = 16,
            outputName = "rein",
        )
        val spark = listOf(
            SparkPoint(2900f, 0.6f),
            SparkPoint(3050f, 0.4f),
            SparkPoint(3200f, 0.55f),
        )
        val row = com.acite.axlranko.pages.components.CheckpointRow(
            checkpoint = checkpoint,
            step = 3050,
            samples = emptyList(),
            generated = emptyList(),
            running = null,
        )
        val scene = scene(420, 800) {
            Column(Modifier.width(420.dp)) {
                TrainControlCard(
                    iconOnly = true,
                    status = TrainStatus(),
                    commandInFlight = false,
                    outputDir = "/out",
                    loggingDir = "/logs",
                    onStart = {},
                    onPause = {},
                    onResume = {},
                    onStop = {},
                    onReset = {},
                )
                PathRow(
                    config = buildJsonObject {
                        put("logging_dir", "/logs")
                        put("output_dir", "/out")
                    },
                    runId = "rein_20260911_120000",
                    portrait = true,
                )
                com.acite.axlranko.pages.CheckpointRowCard(
                    portrait = true,
                    row = row,
                    spark = spark,
                    saveEveryNSteps = 200,
                    thumbSize = 80f,
                    showSetBadges = false,
                    newJobIds = emptySet(),
                    gpuFree = true,
                    starting = false,
                    busyElsewhere = false,
                    startingEvaluation = false,
                    pinning = false,
                    pinEnabled = true,
                    exportInFlightPath = null,
                    exportResult = null,
                    clearingSamples = false,
                    clearSamplesResult = null,
                    onOpen = {},
                    onGenerate = {},
                    onEvaluate = { _, _, _ -> },
                    onCancelEvaluation = {},
                    onOpenEvaluation = {},
                    onTogglePin = {},
                    onSaveAs = {},
                    onClearSamples = {},
                )
            }
        }
        try {
            val all = nodes(scene)
            val lines = texts(all)
            assertTrue(lines.none { it.first == "Save As" || it.first == "Generate samples" || it.first == "Start" })
            val described = descriptions(all)
            assertTrue("Save As" in described && "Generate samples" in described && "Start" in described)
            val run = lines.first { it.first == "Run" }.second
            val logs = lines.first { it.first == "Logs" }.second
            val output = lines.first { it.first == "Output" }.second
            assertTrue(logs > run + 8f && output > logs + 8f)
            val chart = all.first { "Avg Loss" in descriptions(listOf(it)) }
            assertTrue(chart.size.height in 78..90, "spark height ${chart.size.height}")
        } finally {
            scene.close()
        }
    }

    @Test
    fun automationPortraitTabsShareARowAndTheHeaderStacksWhenNarrow() {
        if (GraphicsEnvironment.isHeadless()) return
        val viewModel = AutomationScreenViewModel(
            ipc = TrainerIpcClient(),
            refreshHub = DatasetRefreshHub(),
            datasetSelection = DatasetSelection(),
            pathPicker = NoopPathPicker(),
        )
        val narrow = scene(280, 200) {
            Column(Modifier.width(280.dp)) {
                AutomationHeader(AutomationUiState(), viewModel)
                AutomationSectionTabs(selected = AutomationSection.Prompts, onSelect = {})
            }
        }
        val wide = scene(900, 160) {
            Box(Modifier.width(900.dp)) {
                AutomationHeader(AutomationUiState(), viewModel)
            }
        }
        try {
            val narrowLines = texts(nodes(narrow))
            val prompts = narrowLines.first { it.first == "Prompts" }.second
            val comfy = narrowLines.first { it.first == "ComfyUI" }.second
            val gallery = narrowLines.first { it.first == "Gallery" }.second
            assertEquals(prompts, comfy, 2f)
            assertEquals(prompts, gallery, 2f)
            val title = narrowLines.first { it.first == "Automation" }.second
            val chip = narrowLines.first { it.first == "中文" }.second
            assertTrue(chip > title + 8f, "language row did not drop below the title")
            val wideLines = texts(nodes(wide))
            val wideTitle = wideLines.first { it.first == "Automation" }.second
            val wideChip = wideLines.first { it.first == "中文" }.second
            assertEquals(wideTitle, wideChip, 8f)
        } finally {
            narrow.close()
            wide.close()
        }
    }
}

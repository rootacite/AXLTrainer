package com.acite.axlranko.pages

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.PointerButton
import androidx.compose.ui.input.pointer.PointerButtons
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.AutomationSection
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.HardwareCpu
import com.acite.axlranko.model.HardwareGpu
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.StatisticsUiState
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.model.UtilsUiState
import com.acite.axlranko.pages.components.CheckpointRow
import com.acite.axlranko.pages.components.HardwareSection
import com.acite.axlranko.pages.components.ImagePreviewOverlay
import com.acite.axlranko.pages.components.PreviewImage
import com.acite.axlranko.pages.components.SparkPoint
import com.acite.axlranko.pages.components.automation.GalleryPane
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
    fun portraitStacksHardwareCardsChartsAndMetrics() {
        if (GraphicsEnvironment.isHeadless()) return
        val hardware = DashboardUiState(
            hardware = HardwareStatus(
                available = true,
                gpus = listOf(
                    HardwareGpu(
                        gpuUtilPct = 17.0,
                        powerW = 222.0,
                        tempEdgeC = 40.0,
                        tempJunctionC = 50.0,
                        memUsedBytes = 8L * 1024 * 1024 * 1024,
                        memTotalBytes = 16L * 1024 * 1024 * 1024,
                    ),
                ),
                cpu = HardwareCpu(
                    utilPct = 7.0,
                    memUsedBytes = (1.5 * 1024 * 1024 * 1024).toLong(),
                    memTotalBytes = 3L * 1024 * 1024 * 1024,
                ),
            ),
        )
        val metrics = DashboardUiState(
            latestStats = buildJsonObject {
                put("current_step", 10)
                put("Train/Loss", 0.25)
                put("UNet/LR/Effective_Actual_LR", 0.0001)
                put("TE/LR/Effective_Actual_LR", 0.00002)
            },
        )
        val portrait = scene(480, 1700) {
            Column(Modifier.width(480.dp)) {
                HardwareSection(hardware, portrait = true)
                MetricsSection(metrics, portrait = true)
            }
        }
        val landscape = scene(1200, 900) {
            Column(Modifier.width(1200.dp)) {
                HardwareSection(hardware, portrait = false)
                MetricsSection(metrics, portrait = false)
            }
        }
        try {
            val tall = texts(nodes(portrait))
            val gpu = tall.first { it.first == "17%" }.second
            val power = tall.first { it.first == "222 W" }.second
            val temp = tall.first { it.first == "40 °C / 50 °C" }.second
            val cpu = tall.first { it.first == "7% · 1.5 / 3.0 GiB" }.second
            assertTrue(power > gpu + 8f, "Power y=$power GPU y=$gpu")
            assertEquals(power, temp, 4f)
            assertTrue(cpu > power + 8f, "CPU y=$cpu Power y=$power")
            val chartRows = bands(tall.filter { it.first == "No Data" }.map { it.second })
            assertEquals(4, chartRows.size, "hardware charts shared rows: $chartRows")
            assertTrue(chartRows.all { it.size == 1 })
            assertTrue(chartRows[1][0] > chartRows[0][0] + 150f)
            assertTrue(chartRows[3][0] > chartRows[2][0] + 150f)
            val step = tall.first { it.first == "Current Step" }.second
            val loss = tall.first { it.first == "Latest Loss" }.second
            val unet = tall.first { it.first == "UNet LR" }.second
            val te = tall.first { it.first == "TE Effective LR" }.second
            assertEquals(step, loss, 4f)
            assertEquals(unet, te, 4f)
            assertTrue(unet > step + 8f, "metrics second row y=$unet first y=$step")

            val wide = texts(nodes(landscape))
            val wideGpu = wide.first { it.first == "17%" }.second
            val widePower = wide.first { it.first == "222 W" }.second
            val wideTemp = wide.first { it.first == "40 °C / 50 °C" }.second
            val wideCpu = wide.first { it.first == "7% · 1.5 / 3.0 GiB" }.second
            assertEquals(wideGpu, widePower, 4f)
            assertEquals(wideGpu, wideTemp, 4f)
            assertEquals(wideGpu, wideCpu, 4f)
            val wideCharts = bands(wide.filter { it.first == "No Data" }.map { it.second })
            assertEquals(2, wideCharts.size, "landscape charts: $wideCharts")
            assertTrue(wideCharts.all { it.size == 2 })
            val wideStep = wide.first { it.first == "Current Step" }.second
            assertEquals(wideStep, wide.first { it.first == "Latest Loss" }.second, 4f)
            assertEquals(wideStep, wide.first { it.first == "UNet LR" }.second, 4f)
            assertEquals(wideStep, wide.first { it.first == "TE Effective LR" }.second, 4f)
        } finally {
            portrait.close()
            landscape.close()
        }
    }

    @Test
    fun portraitSampleRangeSplitsTheButtonOntoTheNextRow() {
        if (GraphicsEnvironment.isHeadless()) return
        val rows = listOf(100, 500).map { step ->
            CheckpointRow(
                checkpoint = CheckpointItem(
                    path = "/out/rein_s$step/rein.safetensors",
                    runId = "rein_20260911_120000",
                    dir = "rein_s$step",
                    filename = "rein.safetensors",
                    step = step,
                    outputName = "rein",
                ),
                step = step,
                samples = emptyList(),
                generated = emptyList(),
                running = null,
            )
        }
        val portrait = scene(420, 240) {
            Column(Modifier.width(420.dp)) {
                SampleRangeRow(
                    rows = rows,
                    batch = null,
                    starting = false,
                    canStart = true,
                    note = null,
                    onSampleRange = { _, _ -> },
                    onCancel = {},
                    portrait = true,
                )
            }
        }
        val landscape = scene(900, 160) {
            Column(Modifier.width(900.dp)) {
                SampleRangeRow(
                    rows = rows,
                    batch = null,
                    starting = false,
                    canStart = true,
                    note = null,
                    onSampleRange = { _, _ -> },
                    onCancel = {},
                    portrait = false,
                )
            }
        }
        try {
            assertSampleRangeStacked(nodes(portrait), stacked = true)
            assertSampleRangeStacked(nodes(landscape), stacked = false)
        } finally {
            portrait.close()
            landscape.close()
        }
    }

    private fun assertSampleRangeStacked(all: List<SemanticsNode>, stacked: Boolean) {
        val lines = texts(all)
        val from = lines.first { it.first == "from" }.second
        val button = sampleButtonY(all)
        val count = lines.first { it.first == "2 checkpoint(s)" }.second
        if (stacked) {
            assertTrue(button > from + 24f, "button y=$button from y=$from")
            assertEquals(button, count, 16f)
        } else {
            assertTrue(button < from + 30f, "button y=$button from y=$from")
            assertEquals(button, count, 16f)
        }
    }

    /** The "Sample range" label and the button share their text; only the button is clickable. */
    private fun sampleButtonY(nodes: List<SemanticsNode>): Float {
        val button = nodes.first { node ->
            node.config.getOrNull(SemanticsProperties.Text).orEmpty().any { it.text == "Sample range" } &&
                hasClick(node)
        }
        return button.positionInRoot.y
    }

    private fun hasClick(node: SemanticsNode): Boolean {
        var current: SemanticsNode? = node
        while (current != null) {
            if (current.config.getOrNull(SemanticsActions.OnClick) != null) return true
            current = current.parent
        }
        return false
    }

    private fun bands(ys: List<Float>, tolerance: Float = 4f): List<List<Float>> {
        val groups = mutableListOf<MutableList<Float>>()
        for (y in ys.sorted()) {
            val last = groups.lastOrNull()
            if (last == null || y - last.last() > tolerance) groups += mutableListOf(y) else last += y
        }
        return groups
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

    private fun nodes(scene: ImageComposeScene): List<SemanticsNode> = semantics(scene, merged = true)

    private fun unmerged(scene: ImageComposeScene): List<SemanticsNode> = semantics(scene, merged = false)

    private fun semantics(scene: ImageComposeScene, merged: Boolean): List<SemanticsNode> {
        scene.render()
        val found = mutableListOf<SemanticsNode>()
        scene.semanticsOwners.forEach { owner ->
            val root = if (merged) owner.rootSemanticsNode else owner.unmergedRootSemanticsNode
            walk(root, found)
        }
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

    @Test
    fun portraitGalleryJobPutsTheNameUnderTheStatus() {
        if (GraphicsEnvironment.isHeadless()) return
        val viewModel = AutomationScreenViewModel(
            ipc = TrainerIpcClient(),
            refreshHub = DatasetRefreshHub(),
            datasetSelection = DatasetSelection(),
            pathPicker = NoopPathPicker(),
        )
        val job = AutomationJobSummary(
            id = "Kirika_20260928_042536",
            state = "done",
            startedAt = 1_790_540_736.0,
            finishedAt = 1_790_540_790.0,
            total = 2,
            done = 1,
            images = 2,
        )
        val state = AutomationUiState(section = AutomationSection.Gallery, jobs = listOf(job))
        val counts = "1/2 · 2 images · 54s"
        val portrait = scene(420, 640) {
            Box(Modifier.width(420.dp).height(640.dp)) {
                GalleryPane(state, viewModel, portrait = true)
            }
        }
        val landscape = scene(1100, 640) {
            Box(Modifier.width(1100.dp).height(640.dp)) {
                GalleryPane(state, viewModel, portrait = false)
            }
        }
        try {
            // The card is clickable, so the merged tree reports every line at the card's top.
            val tallNodes = unmerged(portrait)
            val tall = texts(tallNodes)
            val countsY = tall.first { it.first == counts }.second
            val name = tallNodes.first { node ->
                node.config.getOrNull(SemanticsProperties.Text).orEmpty().map { it.text } == listOf(job.id)
            }
            assertTrue(name.positionInRoot.y > countsY + 8f, "name y=${name.positionInRoot.y} counts y=$countsY")
            assertTrue(name.size.height <= 24, "job name wrapped, height ${name.size.height}")
            assertTrue(tall.any { it.first == "Done" && kotlin.math.abs(it.second - countsY) < 8f })

            val wide = texts(unmerged(landscape))
            val wideCounts = wide.first { it.first == counts }.second
            val wideName = wide.first { it.first == job.id }.second
            assertEquals(wideCounts, wideName, 8f)
        } finally {
            portrait.close()
            landscape.close()
        }
    }

    @Test
    fun portraitPreviewPagesFromTheSidesAndLandscapeKeepsArrows() {
        if (GraphicsEnvironment.isHeadless()) return
        val images = List(3) { index -> PreviewImage(path = "/img/$index.png", title = "frame-$index") }
        val index = mutableStateOf(1)
        val portrait = scene(400, 700) {
            ImagePreviewOverlay(
                images = images,
                index = index.value,
                onClose = {},
                onPrev = { index.value = (index.value - 1).coerceAtLeast(0) },
                onNext = { index.value = (index.value + 1).coerceAtMost(images.lastIndex) },
                portrait = true,
            )
        }
        try {
            val described = descriptions(nodes(portrait))
            assertTrue("Previous" !in described && "Next" !in described)
            assertTrue(texts(nodes(portrait)).any { it.first == "2 / 3" })
            click(portrait, 40f, 400f)
            assertTrue(texts(nodes(portrait)).any { it.first == "1 / 3" }, "left side did not go back")
            click(portrait, 360f, 400f)
            assertTrue(texts(nodes(portrait)).any { it.first == "2 / 3" }, "right side did not go forward")
            swipe(portrait, fromX = 300f, toX = 80f, y = 400f)
            assertTrue(texts(nodes(portrait)).any { it.first == "3 / 3" }, "swipe left did not go forward")
        } finally {
            portrait.close()
        }

        index.value = 1
        val landscape = scene(800, 500) {
            ImagePreviewOverlay(
                images = images,
                index = index.value,
                onClose = {},
                onPrev = { index.value = (index.value - 1).coerceAtLeast(0) },
                onNext = { index.value = (index.value + 1).coerceAtMost(images.lastIndex) },
                portrait = false,
            )
        }
        try {
            val described = descriptions(nodes(landscape))
            assertTrue("Previous" in described && "Next" in described)
            click(landscape, 400f, 280f)
            assertTrue(texts(nodes(landscape)).any { it.first == "2 / 3" }, "a click on the picture paged in landscape")
        } finally {
            landscape.close()
        }
    }

    private fun click(scene: ImageComposeScene, x: Float, y: Float) {
        val at = Offset(x, y)
        val down = PointerButtons(isPrimaryPressed = true)
        scene.sendPointerEvent(PointerEventType.Press, at, button = PointerButton.Primary, buttons = down)
        scene.sendPointerEvent(PointerEventType.Release, at, button = PointerButton.Primary, buttons = PointerButtons())
        scene.render()
    }

    private fun swipe(scene: ImageComposeScene, fromX: Float, toX: Float, y: Float) {
        val down = PointerButtons(isPrimaryPressed = true)
        scene.sendPointerEvent(PointerEventType.Press, Offset(fromX, y), button = PointerButton.Primary, buttons = down)
        scene.sendPointerEvent(PointerEventType.Move, Offset(toX, y), buttons = down)
        scene.sendPointerEvent(PointerEventType.Release, Offset(toX, y), button = PointerButton.Primary, buttons = PointerButtons())
        scene.render()
    }
}

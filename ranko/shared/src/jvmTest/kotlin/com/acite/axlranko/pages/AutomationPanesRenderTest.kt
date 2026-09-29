package com.acite.axlranko.pages

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.layout.onSizeChanged
import com.acite.axlranko.data.AutomationJobDetail
import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.data.AutomationSettings
import com.acite.axlranko.data.AutomationWorkflow
import com.acite.axlranko.data.AutomationWorkflowList
import com.acite.axlranko.data.ComfyCheckedEntry
import com.acite.axlranko.data.ComfyDiscovery
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.JobPass
import com.acite.axlranko.data.JobPromptState
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.data.WorkflowMissingModel
import com.acite.axlranko.data.WorkflowTextNode
import com.acite.axlranko.model.AutomationSection
import com.acite.axlranko.model.AutomationSettingsDraft
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.GalleryImageAction
import com.acite.axlranko.model.GalleryImagePrompt
import com.acite.axlranko.model.GalleryImageRef
import com.acite.axlranko.model.PromptEditDraft
import com.acite.axlranko.model.PromptExtendDraft
import com.acite.axlranko.pages.components.automation.ComfyPane
import com.acite.axlranko.pages.components.automation.GalleryPane
import com.acite.axlranko.pages.components.ImagePreviewOverlay
import com.acite.axlranko.pages.components.PreviewImage
import com.acite.axlranko.pages.components.automation.PromptsPane
import com.acite.axlranko.prompt.defaultSpec
import com.acite.axlranko.ui.theme.RankoTheme
import com.acite.axlranko.util.PathPicker
import java.awt.GraphicsEnvironment
import java.util.Collections
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.CompletableDeferred

/**
 * Every Automation pane is composed for real, in a window, against fixture state: the classes of
 * layout crash that only show up at measurement time (an unbounded scroll inside a scroll, a
 * slider measured with infinite width) fail here instead of on the user's screen.
 *
 * Skipped on a headless JVM; the window is placed far off-screen so it never flashes in view.
 */
class AutomationPanesRenderTest {

    private val failures = Collections.synchronizedList(mutableListOf<Throwable>())

    private class NoopPathPicker : PathPicker {
        override suspend fun pickDirectory(title: String, current: String): String? = null
        override suspend fun pickFile(title: String, current: String, extensions: List<String>?): String? = null
        override suspend fun saveFile(suggestedName: String, current: String): String? = null
        override fun deleteEmptyPlaceholder(path: String) = Unit
    }

    private fun viewModel() = AutomationScreenViewModel(
        ipc = TrainerIpcClient(),
        refreshHub = DatasetRefreshHub(),
        datasetSelection = DatasetSelection(),
        pathPicker = NoopPathPicker(),
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

    private fun render(states: List<AutomationUiState>) {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        try {
            val viewModel = viewModel()
            var state by mutableStateOf(states.first())
            // A ComposeWindow must be built on the AWT event thread.
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(1440, 900)
                    setContent {
                        RankoTheme {
                            val current = state
                            when (current.section) {
                                AutomationSection.Prompts -> PromptsPane(current, viewModel, warning = null)
                                AutomationSection.ComfyUi -> ComfyPane(current, viewModel)
                                AutomationSection.Gallery -> GalleryPane(current, viewModel)
                            }
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            states.forEach { next ->
                onEdtGet { state = next }
                pumpFor(700)
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

    private fun pumpFor(millis: Long) {
        val deadline = System.currentTimeMillis() + millis
        while (System.currentTimeMillis() < deadline) {
            SwingUtilities.invokeAndWait { }
            Thread.sleep(10)
        }
    }

    @Test
    fun everySectionComposesWithRealState() {
        val job = AutomationJobSummary(
            id = "Kirika_20260928_042536",
            state = "done",
            createdAt = 1_790_540_736.0,
            startedAt = 1_790_540_736.0,
            finishedAt = 1_790_540_790.0,
            total = 2,
            done = 1,
            failed = 1,
            images = 2,
            previewPaths = listOf("/repo/automation/jobs/Kirika_20260928_042536/images/p0001_01.png"),
            workflow = "/repo/automation/workflows/Kirika.json",
            outputDir = "/repo/automation/jobs",
            positiveNode = "215",
            comfyUrl = "http://127.0.0.1:8188",
        )
        val workflow = AutomationWorkflow(
            name = "Kirika",
            path = "/repo/automation/workflows/Kirika.json",
            valid = true,
            nodeCount = 25,
            saveImageNodes = listOf("222"),
            batchSizeNodes = listOf("217", "198:196"),
            textNodes = listOf(
                WorkflowTextNode("215", "CLIPTextEncode", "(kirika_character:1.1), 1girl"),
                WorkflowTextNode("216", "CLIPTextEncode", "worst quality"),
                WorkflowTextNode("198:259", "CLIPTextEncode", "best quality"),
            ),
            positiveNode = "215",
        )
        val detail = AutomationJobDetail(
            id = job.id,
            state = "done",
            outputDir = job.outputDir,
            logTail = "[automation] [1] queued stub-1\n[automation] done: 1 prompt(s)",
            prompts = listOf(
                JobPromptState(
                    index = 0,
                    text = "(kirika_character:1.1), 1girl, white sundress",
                    state = "done",
                    seed = 1234,
                    promptId = "stub-1",
                    images = listOf("p0001_01.png", "p0001_02.png"),
                ),
                JobPromptState(index = 1, text = "second prompt", state = "error", error = "ComfyUI execution failed"),
            ),
        )
        val comfy = ComfyDiscovery(
            found = true,
            url = "http://127.0.0.1:8188",
            version = "0.35.0",
            queueRunning = 1,
            queuePending = 2,
        )

        val prompts = AutomationUiState(section = AutomationSection.Prompts, spec = defaultSpec())
        val comfySection = AutomationUiState(
            section = AutomationSection.ComfyUi,
            settings = AutomationSettingsDraft.of(
                AutomationSettings(
                    server = "http://127.0.0.1:8188",
                    workflow = workflow.path,
                    positiveNode = "215",
                    count = 2,
                    poll = 0.5,
                    outputDir = "/repo/automation/jobs",
                ),
            ),
            comfy = comfy,
            workflows = listOf(workflow),
            workflowCheck = true,
            promptSets = emptyList(),
            jobError = null,
        )
        val comfyEmpty = AutomationUiState(
            section = AutomationSection.ComfyUi,
            comfy = ComfyDiscovery(
                found = false,
                checked = listOf(ComfyCheckedEntry("http://127.0.0.1:8188", false, "not a ComfyUI")),
            ),
            workflows = emptyList(),
            jobError = "no prompts to generate",
        )
        val gallery = AutomationUiState(
            section = AutomationSection.Gallery,
            jobs = listOf(job),
            selectedJobId = job.id,
            jobDetail = detail,
        )
        val galleryPreview = gallery.copy(galleryPreviewIndex = 1)
        val galleryEmpty = AutomationUiState(section = AutomationSection.Gallery)
        val galleryFilteredOut = gallery.copy(jobFilter = com.acite.axlranko.model.JobFilter.Failed)

        render(
            listOf(
                prompts,
                comfySection,
                comfyEmpty,
                AutomationUiState(section = AutomationSection.ComfyUi, workflowsLoading = true),
                gallery,
                galleryPreview,
                galleryEmpty,
                galleryFilteredOut,
                AutomationUiState(
                    section = AutomationSection.Gallery,
                    jobs = listOf(job),
                    selectedJobId = job.id,
                    jobDetail = detail,
                    jobActionBusy = job.id,
                ),
                AutomationUiState(
                    section = AutomationSection.Gallery,
                    jobs = listOf(job),
                    selectedJobId = job.id,
                    jobDetail = detail,
                    pendingDeleteJob = job.id,
                ),
                // The per-image dialogs: a redraw (which overwrites the image), a delete, the
                // prompt editor and the "add N images" count.
                AutomationUiState(
                    section = AutomationSection.Gallery,
                    jobs = listOf(job),
                    selectedJobId = job.id,
                    jobDetail = detail,
                    pendingImageAction = GalleryImagePrompt(
                        GalleryImageRef(job.id, 0, "p0001_01.png"),
                        GalleryImageAction.Regenerate,
                    ),
                ),
                AutomationUiState(
                    section = AutomationSection.Gallery,
                    jobs = listOf(job),
                    selectedJobId = job.id,
                    jobDetail = detail,
                    pendingImageAction = GalleryImagePrompt(
                        GalleryImageRef(job.id, 0, "p0001_01.png"),
                        GalleryImageAction.Delete,
                    ),
                ),
                AutomationUiState(
                    section = AutomationSection.Gallery,
                    jobs = listOf(job),
                    selectedJobId = job.id,
                    jobDetail = detail,
                    editingPrompt = PromptEditDraft(job.id, 0, "a longer prompt being edited"),
                ),
                AutomationUiState(
                    section = AutomationSection.Gallery,
                    jobs = listOf(job),
                    selectedJobId = job.id,
                    jobDetail = detail,
                    extendingPrompt = PromptExtendDraft(job.id, 0, count = "4"),
                ),
                // A redraw and an append in flight: the job row says what is happening (a redraw
                // does not move the job's own counters) and so does the card above the thumbnails,
                // which is where the user is looking.
                AutomationUiState(
                    section = AutomationSection.Gallery,
                    jobs = listOf(
                        job.copy(
                            state = "running",
                            pass = JobPass(
                                mode = "append",
                                promptIndex = 0,
                                imagesDone = 2,
                                totalImages = 4,
                                image = "p0001_03.png",
                            ),
                        ),
                    ),
                    selectedJobId = job.id,
                    jobDetail = detail.copy(
                        state = "running",
                        pass = JobPass(
                            mode = "image",
                            promptIndex = 0,
                            imagesDone = 0,
                            totalImages = 1,
                            image = "p0001_01.png",
                        ),
                    ),
                ),
            ),
        )
    }

    @Test
    fun thePreviewOverlayFillsTheBoxItIsGiven() {
        if (GraphicsEnvironment.isHeadless()) return
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, error -> failures += error }
        var overlaySize: androidx.compose.ui.unit.IntSize? = null
        try {
            val images = List(3) { index ->
                PreviewImage(
                    path = "/repo/automation/jobs/job/images/p0001_0${index + 1}.png",
                    title = "p0001_0${index + 1}.png",
                    caption = "seed 1234\nprompt text",
                )
            }
            val window = onEdtGetResult {
                ComposeWindow().apply {
                    setLocation(-3200, -3200)
                    setSize(1200, 800)
                    setContent {
                        RankoTheme {
                            Box(Modifier.fillMaxSize()) {
                                ImagePreviewOverlay(
                                    images = images,
                                    index = 0,
                                    onClose = {},
                                    onPrev = {},
                                    onNext = {},
                                    modifier = Modifier.onSizeChanged { overlaySize = it },
                                )
                            }
                        }
                    }
                }
            }
            onEdtGet { window.isVisible = true }
            pumpFor(900)
            onEdtGet { window.dispose() }
            val size = overlaySize
            assertTrue(size != null, "the overlay never reported a size")
            assertTrue(size!!.width >= 1100, "overlay width was ${size.width}, expected the window width")
            assertTrue(size.height >= 700, "overlay height was ${size.height}: it collapsed instead of filling")
            assertTrue(failures.isEmpty(), failures.joinToString("\n\n") { it.stackTraceToString() })
        } finally {
            Thread.setDefaultUncaughtExceptionHandler(previous)
        }
    }

    @Test
    fun aWorkflowWithMissingModelsAndNoPositiveNodeStillComposes() {
        val broken = AutomationWorkflow(
            name = "broken",
            path = "/repo/automation/workflows/broken.json",
            valid = true,
            nodeCount = 3,
            saveImageNodes = listOf("5"),
            batchSizeNodes = emptyList(),
            textNodes = emptyList(),
            positiveNode = "",
            positiveNodeGuessed = false,
            missingModels = listOf(
                WorkflowMissingModel("198:41", "UpscaleModelLoader", "model_name", "RealESRGAN_x4plus_anime_6B.pth"),
                WorkflowMissingModel("207:219", "LoraLoader", "lora_name", "Kirika_s003100.safetensors"),
            ),
        )
        render(
            listOf(
                AutomationUiState(
                    section = AutomationSection.ComfyUi,
                    workflows = listOf(broken),
                    workflowError = "workflow is not usable: the workflow has no SaveImage node",
                ),
            ),
        )
    }
}

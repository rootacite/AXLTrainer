package com.acite.axlranko.model

import com.acite.axlranko.data.AutomationJobDetail
import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.data.AutomationSettings
import com.acite.axlranko.data.AutomationWorkflow
import com.acite.axlranko.data.ComfyDiscovery
import com.acite.axlranko.data.PromptSetItem
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.prompt.PromptMatrix
import com.acite.axlranko.prompt.PromptSpec
import com.acite.axlranko.prompt.WizardModel
import com.acite.axlranko.prompt.defaultSpec

/** The Automation page's three areas, in rail order. */
enum class AutomationSection(val title: String) {
    Prompts("Prompts"),
    ComfyUi("ComfyUI"),
    Gallery("Gallery"),
}

/** How the Prompts area is showing the current configuration. */
enum class PromptView { Wizard, Manifest }

/** Where a batch job takes its prompts from. */
enum class PromptSource { CurrentResults, SavedSet, Manual }

/** The Gallery's job filter chips. */
enum class JobFilter(val wire: String) {
    All(""),
    Running("running"),
    Done("done"),
    Failed("error"),
    Cancelled("cancelled"),
}

/** The ComfyUI section's editable copy of `automation/settings.json`. */
data class AutomationSettingsDraft(
    val server: String = "",
    val workflow: String = "",
    val positiveNode: String = "",
    val count: String = "1",
    val poll: String = "0.5",
    val outputDir: String = "",
) {
    companion object {
        fun of(settings: AutomationSettings): AutomationSettingsDraft = AutomationSettingsDraft(
            server = settings.server,
            workflow = settings.workflow,
            positiveNode = settings.positiveNode,
            count = settings.count.toString(),
            poll = settings.poll.toString(),
            outputDir = settings.outputDir,
        )
    }

    fun toSettings(): AutomationSettings = AutomationSettings(
        server = server.trim(),
        workflow = workflow.trim(),
        positiveNode = positiveNode.trim(),
        count = count.filter { it.isDigit() }.toIntOrNull() ?: 1,
        poll = poll.toDoubleOrNull() ?: 0.5,
        outputDir = outputDir.trim(),
    )
}

data class PromptProfileItem(
    val name: String,
    val version: Int? = null,
    val modified: Long = 0,
    val size: Long = 0,
    val error: String? = null,
    /** Upgrade notes from loading this profile (v1/v2 → v3), shown on the loaded row. */
    val notes: List<String> = emptyList(),
)

data class AutomationUiState(
    val section: AutomationSection = AutomationSection.Prompts,
    val language: PromptLang = PromptLang.English,
    val matrix: PromptMatrix? = null,
    val matrixPath: String = "",
    val matrixLines: Int = 0,
    val matrixError: String? = null,
    val loadingMatrix: Boolean = false,
    val profiles: List<PromptProfileItem> = emptyList(),
    val profileError: String? = null,
    val loadingProfiles: Boolean = false,
    val activeProfile: String = "",
    val view: PromptView = PromptView.Wizard,
    val spec: PromptSpec = defaultSpec(),
    val pageIndex: Int = 0,
    /** Manifest row opened for editing, as its page key; null while the list itself is shown. */
    val editingRow: String? = null,
    val saveDialogOpen: Boolean = false,
    val saveName: String = "",
    val saveOverwrite: Boolean = false,
    val saveError: String? = null,
    val notice: String? = null,
    val results: List<String> = emptyList(),
    val resultSeed: Long? = null,
    val seedText: String = "",
    val generateError: String? = null,
    val warnings: List<String> = emptyList(),
    val busy: Boolean = false,
    /** The prompt list handed to the ComfyUI area. */
    val batchInput: List<String> = emptyList(),

    // --- ComfyUI section ---
    val settings: AutomationSettingsDraft = AutomationSettingsDraft(),
    val settingsLoaded: Boolean = false,
    val settingsSaving: Boolean = false,
    val settingsError: String? = null,
    val settingsNotice: String? = null,
    val comfy: ComfyDiscovery = ComfyDiscovery(),
    val discovering: Boolean = false,
    val discoverError: String? = null,
    val workflows: List<AutomationWorkflow> = emptyList(),
    val workflowsLoading: Boolean = false,
    val workflowCheck: Boolean = false,
    val workflowError: String? = null,
    val promptSource: PromptSource = PromptSource.CurrentResults,
    val promptSetName: String = "",
    val manualPrompts: String = "",
    val promptSets: List<PromptSetItem> = emptyList(),
    val promptSetError: String? = null,
    val startingJob: Boolean = false,
    val jobError: String? = null,
    val lastJobId: String = "",

    // --- Gallery section ---
    val jobs: List<AutomationJobSummary> = emptyList(),
    val jobsLoading: Boolean = false,
    val jobsError: String? = null,
    val selectedJobId: String = "",
    val jobDetail: AutomationJobDetail? = null,
    val jobFilter: JobFilter = JobFilter.All,
    val jobSearch: String = "",
    val galleryThumbSize: Float = 160f,
    val galleryPreviewIndex: Int? = null,
    val jobActionBusy: String = "",
    /** Job id waiting for the delete confirmation. */
    val pendingDeleteJob: String? = null,
    // --- Gallery: one image / one prompt at a time ---
    /** An image waiting for its confirmation (delete, or a redraw that overwrites it). */
    val pendingImageAction: GalleryImagePrompt? = null,
    /** The prompt whose text is open in the edit dialog (`null` = closed). */
    val editingPrompt: PromptEditDraft? = null,
    /** The prompt the "add N images" dialog is open for (`null` = closed), with its draft count. */
    val extendingPrompt: PromptExtendDraft? = null,
    /** The "add N images to every prompt" dialog, with its draft count (`null` = closed). */
    val appendingAllPrompts: PromptAppendAllDraft? = null,
) {
    /** Keys of the wizard pages that apply to the current spec, in order. */
    val applicablePageKeys: List<String>
        get() = WizardModel.PAGES.filter { it.applies(spec) }.map { it.key }

    /** Index into [applicablePageKeys] of the page now shown, clamped to what applies. */
    val visiblePageIndex: Int
        get() = pageIndex.coerceIn(0, (applicablePageKeys.size - 1).coerceAtLeast(0))

    val currentPageKey: String?
        get() = applicablePageKeys.getOrNull(visiblePageIndex)

    /** The workflow the ComfyUI section is configured for, from the uploaded list. */
    val activeWorkflow: AutomationWorkflow?
        get() = workflows.firstOrNull { it.path == settings.workflow || it.name == settings.workflow }

    /** Jobs after the filter chips and the search box. */
    val visibleJobs: List<AutomationJobSummary>
        get() = jobs.filter { job ->
            val stateOk = when (jobFilter) {
                JobFilter.All -> true
                JobFilter.Failed -> job.state == "error"
                else -> job.state == jobFilter.wire
            }
            if (!stateOk) return@filter false
            if (jobSearch.isBlank()) return@filter true
            val needle = jobSearch.trim().lowercase()
            job.id.lowercase().contains(needle) || job.workflow.lowercase().contains(needle)
        }

    /** Images of the selected job, in the order the runner produced them (absolute paths). */
    val galleryImagePaths: List<String>
        get() {
            val detail = jobDetail ?: return emptyList()
            return detail.prompts.flatMap { prompt ->
                prompt.images.map { name -> jobImagePathFor(detail.id, detail.outputDir, name) }
            }
        }

    val runningJob: AutomationJobSummary?
        get() = jobs.firstOrNull { it.state == "running" }
}

/** `<output_dir>/<job_id>/images/<name>`: what the server allowlists for `blob_*`. */
fun jobImagePathFor(jobId: String, outputDir: String, name: String): String {
    val root = outputDir.trimEnd('/')
    return "$root/$jobId/images/$name"
}

/** One image of one job — what a per-image action needs to name its target. */
data class GalleryImageRef(val jobId: String, val promptIndex: Int, val image: String)

/** What a confirmation dialog over one image would do. */
enum class GalleryImageAction { Delete, Regenerate }

/** The image whose dialog is open, and what confirming it does. */
data class GalleryImagePrompt(val ref: GalleryImageRef, val action: GalleryImageAction)

/** The open "edit this prompt" dialog: which entry, and the text being edited. */
data class PromptEditDraft(val jobId: String, val promptIndex: Int, val text: String)

/** The open "add N images" dialog: which entry, and the count being typed. */
data class PromptExtendDraft(val jobId: String, val promptIndex: Int, val count: String = "1") {
    /** The count to send, or null while the field does not hold a usable number. */
    val images: Int? get() = count.trim().toIntOrNull()?.takeIf { it in 1..16 }
}

/** The open "add N images to every prompt" dialog: which job, and the count being typed. */
data class PromptAppendAllDraft(val jobId: String, val count: String = "1") {
    /** The count to send, or null while the field does not hold a usable number. */
    val images: Int? get() = count.trim().toIntOrNull()?.takeIf { it in 1..16 }
}

/** A job's elapsed seconds: started → finished (or now, while it runs). */
fun jobElapsedSeconds(job: AutomationJobSummary, nowMillis: Long): Double {
    val started = job.startedAt ?: job.createdAt ?: return 0.0
    val finished = job.finishedAt ?: (nowMillis / 1000.0)
    return (finished - started).coerceAtLeast(0.0)
}

/** 0f..1f progress for the job list bars. */
fun jobProgress(job: AutomationJobSummary): Float {
    if (job.total <= 0) return 0f
    return ((job.done + job.failed).toFloat() / job.total.toFloat()).coerceIn(0f, 1f)
}

/** One result export's file name, e.g. `prompts_seed1234_20.txt`. */
fun promptExportFileName(seed: Long?, count: Int): String =
    "prompts_" + (seed?.let { "seed$it" } ?: "random") + "_$count.txt"

/** The whole result list as a downloadable text block. */
fun promptExportText(prompts: List<String>): String = prompts.joinToString("\n", postfix = "\n")

/** A profile row's label: name plus what it says about itself. */
fun promptProfileLabel(item: PromptProfileItem): String {
    val version = item.version?.let { "v$it" } ?: "?"
    return "${item.name}  ·  $version"
}

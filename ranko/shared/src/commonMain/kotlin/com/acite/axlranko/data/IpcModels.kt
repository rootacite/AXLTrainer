package com.acite.axlranko.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ConfigDocument(
    val path: String,
    val text: String,
)

@Serializable
data class ConfigSaveResult(
    val path: String,
)

@Serializable
data class ProfileInfo(
    val name: String,
    val modified: Long = 0,
    val size: Long = 0,
)

@Serializable
data class ProfileListResult(
    val profiles: List<ProfileInfo> = emptyList(),
)

@Serializable
data class ProfileDocument(
    val name: String,
    val text: String,
)

@Serializable
data class ProfileSaveResult(
    val name: String,
)

@Serializable
data class TagLexiconResult(
    val text: String = "",
)

@Serializable
data class DatasetRecord(
    val stem: String,
    val image: String,
    val txt: String,
    val mask: String = "",
    val width: Int = 0,
    val height: Int = 0,
    val tags: List<String> = emptyList(),
    @SerialName("has_sidecar_mask") val hasSidecarMask: Boolean = false,
    @SerialName("has_alpha") val hasAlpha: Boolean = false,
)

@Serializable
data class DatasetListResult(
    val items: List<DatasetRecord> = emptyList(),
    val orphans: List<String> = emptyList(),
)

@Serializable
data class DatasetDropResult(
    val moved: Int = 0,
)

@Serializable
data class DatasetShuffleResult(
    val groups: Int = 0,
    @SerialName("renamed_files") val renamedFiles: Int = 0,
    @SerialName("first_stem") val firstStem: String = "",
    @SerialName("last_stem") val lastStem: String = "",
)

@Serializable
data class MaskGetResult(
    @SerialName("png_base64") val pngBase64: String = "",
)

@Serializable
data class BlobItem(
    val path: String,
    val hash: String? = null,
    val width: Int? = null,
    val height: Int? = null,
    val bytes: Int? = null,
    val mime: String? = null,
    val cache: String? = null,
    val base64: String? = null,
    val error: String? = null,
)

@Serializable
data class BlobListResult(
    val items: List<BlobItem> = emptyList(),
)

@Serializable
data class CheckpointExportResult(
    val bytes: Long = 0,
)

@Serializable
data class FsEntry(
    val name: String,
    val path: String,
    @SerialName("is_dir") val isDir: Boolean = false,
    val size: Long = 0,
    @SerialName("mtime_ms") val mtimeMs: Long = 0,
)

@Serializable
data class FsListResult(
    val path: String = "",
    val parent: String? = null,
    val entries: List<FsEntry> = emptyList(),
)

@Serializable
data class FsRoot(
    val name: String,
    val path: String,
)

@Serializable
data class FsRootsResult(
    val roots: List<FsRoot> = emptyList(),
)

/** Repo-root `input_matrix.txt`: the prompt wizard's tag matrix. */
@Serializable
data class PromptMatrixDocument(
    val path: String = "",
    val text: String = "",
)

/** One named wizard profile under repo-root `prompt_profiles/`; `version` is null when unreadable. */
@Serializable
data class PromptProfileInfo(
    val name: String,
    val version: Int? = null,
    val modified: Long = 0,
    val size: Long = 0,
    val error: String? = null,
)

@Serializable
data class PromptProfileListResult(
    val profiles: List<PromptProfileInfo> = emptyList(),
)

@Serializable
data class PromptProfileDocument(
    val name: String = "",
    val text: String = "",
)

@Serializable
data class PromptProfileSaveResult(
    val name: String = "",
    val path: String = "",
)

// --- Automation: settings, workflows, prompt sets, jobs ---

@Serializable
data class AutomationSettings(
    val server: String = "",
    val workflow: String = "",
    @SerialName("positive_node") val positiveNode: String = "",
    val count: Int = 1,
    val poll: Double = 0.5,
    @SerialName("output_dir") val outputDir: String = "",
)

@Serializable
data class AutomationPaths(
    val root: String = "",
    val workflows: String = "",
    val prompts: String = "",
    val jobs: String = "",
)

@Serializable
data class AutomationConfigResult(
    val settings: AutomationSettings = AutomationSettings(),
    @SerialName("default_output_dir") val defaultOutputDir: String = "",
    val paths: AutomationPaths = AutomationPaths(),
)

@Serializable
data class AutomationConfigSaveResult(
    val settings: AutomationSettings = AutomationSettings(),
)

@Serializable
data class ComfyCheckedEntry(
    val url: String = "",
    val ok: Boolean = false,
    val reason: String = "",
)

@Serializable
data class ComfyDiscovery(
    val found: Boolean = false,
    val url: String = "",
    val version: String = "",
    @SerialName("queue_running") val queueRunning: Int = 0,
    @SerialName("queue_pending") val queuePending: Int = 0,
    val checked: List<ComfyCheckedEntry> = emptyList(),
    @SerialName("probed_all") val probedAll: Boolean = true,
)

@Serializable
data class WorkflowTextNode(
    val id: String = "",
    @SerialName("class_type") val classType: String = "",
    val text: String = "",
)

@Serializable
data class WorkflowMissingModel(
    val node: String = "",
    @SerialName("class_type") val classType: String = "",
    val input: String = "",
    val value: String = "",
)

@Serializable
data class AutomationWorkflow(
    val name: String = "",
    val path: String = "",
    val valid: Boolean = false,
    val error: String? = null,
    @SerialName("node_count") val nodeCount: Int = 0,
    @SerialName("save_image_nodes") val saveImageNodes: List<String> = emptyList(),
    @SerialName("batch_size_nodes") val batchSizeNodes: List<String> = emptyList(),
    @SerialName("text_nodes") val textNodes: List<WorkflowTextNode> = emptyList(),
    @SerialName("positive_node") val positiveNode: String = "",
    @SerialName("positive_node_guessed") val positiveNodeGuessed: Boolean = false,
    @SerialName("missing_models") val missingModels: List<WorkflowMissingModel> = emptyList(),
    @SerialName("model_check") val modelCheck: Boolean = false,
)

@Serializable
data class AutomationWorkflowList(
    val workflows: List<AutomationWorkflow> = emptyList(),
    @SerialName("default_workflow") val defaultWorkflow: String = "",
    @SerialName("model_check") val modelCheck: Boolean = false,
)

@Serializable
data class PromptSetItem(
    val name: String = "",
    val path: String = "",
    val count: Int = 0,
    val text: String = "",
)

@Serializable
data class PromptSetListResult(
    val prompts: List<PromptSetItem> = emptyList(),
)

@Serializable
data class PromptSetSaveResult(
    val name: String = "",
    val path: String = "",
    val count: Int = 0,
)

/** One prompt inside a job: what was sent, what came back, and where the images went. */
@Serializable
data class JobPromptState(
    val index: Int = 0,
    val text: String = "",
    val state: String = "pending",
    val seed: Long? = null,
    @SerialName("prompt_id") val promptId: String = "",
    val images: List<String> = emptyList(),
    /**
     * The seed each image in [images] was drawn with, one per name. Empty on a job from before the
     * Gallery redrew single images, where [seed] (the last pass's) is all there is — see
     * [imageSeedAt].
     */
    @SerialName("image_seeds") val imageSeeds: List<Long?> = emptyList(),
    val error: String? = null,
)

/**
 * The seed behind one of a prompt's images, `null` when it is not known. A record with no
 * per-image list at all falls back to the prompt's single [JobPromptState.seed], which is what the
 * Gallery showed for those images before it could redraw one.
 */
fun imageSeedAt(prompt: JobPromptState, imageIndex: Int): Long? =
    if (prompt.imageSeeds.isEmpty()) prompt.seed else prompt.imageSeeds.getOrNull(imageIndex)

@Serializable
data class AutomationJobSummary(
    val id: String = "",
    val state: String = "",
    @SerialName("created_at") val createdAt: Double? = null,
    @SerialName("started_at") val startedAt: Double? = null,
    @SerialName("updated_at") val updatedAt: Double? = null,
    @SerialName("finished_at") val finishedAt: Double? = null,
    val total: Int = 0,
    val done: Int = 0,
    val failed: Int = 0,
    val images: Int = 0,
    @SerialName("preview_paths") val previewPaths: List<String> = emptyList(),
    /** The last few images the job wrote, newest last. Empty on a helper from before this field. */
    @SerialName("recent_paths") val recentPaths: List<String> = emptyList(),
    val workflow: String = "",
    @SerialName("output_dir") val outputDir: String = "",
    @SerialName("positive_node") val positiveNode: String = "",
    val count: Int = 1,
    @SerialName("comfy_url") val comfyUrl: String = "",
    /** The redraw/append pass in flight for this job, so the job list can say what it is doing. */
    val pass: JobPass? = null,
    val error: String? = null,
)

@Serializable
data class AutomationJobListResult(
    val jobs: List<AutomationJobSummary> = emptyList(),
)

/** A job plus its per-prompt detail; `summary` repeats the list shape for convenience. */
/**
 * What a targeted pass (a redraw or an append from the Gallery) is doing right now. The job's own
 * counters stay still for a redraw — it replaces one image — so the record carries this while the
 * pass runs and drops it (`null`) when the pass ends.
 */
@Serializable
data class JobPass(
    /** `image` (redraw one image in place), `append` (add N images to one prompt) or
     * `append_all` (add N to every prompt). */
    val mode: String = "",
    @SerialName("prompt_index") val promptIndex: Int = 0,
    @SerialName("images_done") val imagesDone: Int = 0,
    @SerialName("total_images") val totalImages: Int = 0,
    /** The image the pass is working on (the redraw's target, or the last one written). */
    val image: String? = null,
)

/** 0f..1f for a targeted pass's progress line. */
fun jobPassProgress(pass: JobPass): Float =
    if (pass.totalImages <= 0) 0f else (pass.imagesDone.toFloat() / pass.totalImages.toFloat()).coerceIn(0f, 1f)

@Serializable
data class AutomationJobDetail(
    val id: String = "",
    val state: String = "",
    val prompts: List<JobPromptState> = emptyList(),
    val summary: AutomationJobSummary? = null,
    @SerialName("log_tail") val logTail: String = "",
    @SerialName("output_dir") val outputDir: String = "",
    @SerialName("comfy_url") val comfyUrl: String = "",
    /** The redraw/append pass in flight, if any. */
    val pass: JobPass? = null,
    val error: String? = null,
)

@Serializable
data class AutomationJobStartResult(
    val job: AutomationJobSummary = AutomationJobSummary(),
    @SerialName("log_path") val logPath: String = "",
)

package com.acite.axlranko.data

import com.acite.axlranko.model.CheckpointPinsResponse
import com.acite.axlranko.model.CheckpointsResponse
import com.acite.axlranko.model.DashboardResponse
import com.acite.axlranko.model.DatasetTagResult
import com.acite.axlranko.model.GenerateSampleResponse
import com.acite.axlranko.model.GeneratedSamplesResponse
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.RunsResponse
import com.acite.axlranko.model.SamplesResponse
import com.acite.axlranko.model.TaggerInfoResult
import com.acite.axlranko.model.TrainStatus
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.random.Random
import kotlin.time.Duration.Companion.seconds
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray


@Serializable
internal data class IpcRequest(
    val id: Long,
    val method: String,
    val params: JsonObject = JsonObject(emptyMap()),
)

@Serializable
internal data class IpcResponse(
    val id: Long? = null,
    val ok: Boolean,
    val result: JsonElement? = null,
    val error: String? = null,
    /** `CLIENT_BUSY` when another client owns the helper (`API.md`), else null. */
    val code: String? = null,
)

/** Another client owns the helper: the connection was refused and the session is not ours. */
internal class HelperBusyException(message: String) : IllegalStateException(message)

private val BLOB_METHODS = setOf("blob_stat", "blob_batch")

/**
 * The connections the client keeps, by what they carry. `Long` exists because one handler can run
 * for minutes (a tagger over a folder) and the connection it is on serves its queue in order.
 */
private enum class LaneKind(val label: String) {
    Control("control"),
    Poll("poll"),
    Blob("blob"),
    Long("long"),
}

private const val DEFAULT_CONNECT_ATTEMPTS = 40
private const val HELLO_TIMEOUT_MILLIS = 10_000L

@Inject
@SingleIn(AppScope::class)
class TrainerIpcClient {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
        // A null where a model declares an Int (a job record written mid-spawn, or by another
        // build) must read as that field's default: a single bad record would otherwise take the
        // whole list down, and the generated images with it.
        coerceInputValues = true
    }
    private val transport = WsTransport()

    /**
     * Who this app is, as the helper sees it: the name shown in a refusal, and an id that makes
     * this run's connections *its* connections — the helper serves one client and turns any other
     * instance away (`API.md`), and this is what tells the two apart.
     */
    internal val instanceId: String = "axlranko-" + Random.nextInt(0, Int.MAX_VALUE).toString(16)

    /**
     * This app's own resource locks (`IpcResources`). api.py takes none: it serves a single client
     * session and leaves ordering to it, so what must not overlap is refused here, before the
     * request reaches the socket.
     */
    internal val resourceTable = ResourceTable()

    /**
     * One connection per kind of traffic — see [laneFor]. They are opened lazily and closed
     * together when the endpoint changes.
     */
    private val lanes: Map<LaneKind, Lane> = LaneKind.entries.associateWith { Lane(it) }
    private val connectMutex = Mutex()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var wsHost = defaultWsHost()
    private var wsPort = defaultWsPort()

    fun endpointHost(): String = wsHost
    fun endpointPort(): Int = wsPort

    suspend fun ping() {
        call("ping", JsonObject(emptyMap()))
    }

    suspend fun getDashboard(
        name: String? = null,
        startStep: Int? = null,
        endStep: Int? = null,
        runId: String? = null,
    ): DashboardResponse {
        val result = call(
            "dashboard",
            buildJsonObject {
                name?.let { put("name", it) }
                startStep?.let { put("start_step", it) }
                endStep?.let { put("end_step", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun listCheckpoints(
        name: String? = null,
        outputDir: String? = null,
    ): CheckpointsResponse {
        val result = call(
            "list_checkpoints",
            buildJsonObject {
                name?.let { put("name", it) }
                outputDir?.let { put("output_dir", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /** The displayed run's pinned checkpoints, from its own `checkpoint_pins.json`. */
    suspend fun checkpointPins(
        name: String? = null,
        runId: String? = null,
    ): CheckpointPinsResponse {
        val result = call(
            "checkpoint_pins",
            buildJsonObject {
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /** Pin or unpin one checkpoint; the reply is that run's whole pin list. */
    suspend fun setCheckpointPin(
        path: String,
        pinned: Boolean,
        dir: String? = null,
        step: Int? = null,
        name: String? = null,
        runId: String? = null,
    ): CheckpointPinsResponse {
        val result = call(
            "checkpoint_pin_set",
            buildJsonObject {
                put("path", path)
                put("pinned", pinned)
                dir?.let { put("dir", it) }
                step?.let { put("step", it) }
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /** The training history: every run directory under the output / log roots, newest first. */
    suspend fun listRuns(): RunsResponse {
        val result = call("list_runs", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun trainStatus(): TrainStatus {
        val result = call("train_status", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun trainStart(): TrainStatus {
        val result = call("train_start", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun trainPause(): TrainStatus {
        val result = call("train_pause", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun trainResume(): TrainStatus {
        val result = call("train_resume", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun trainStop(): TrainStatus {
        val result = call("train_stop", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    /** Clears the run's state so Start can launch a new one. Never deletes artifacts. */
    suspend fun trainReset(name: String? = null): TrainStatus {
        val result = call(
            "train_reset",
            buildJsonObject {
                name?.let { put("name", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /**
     * Retune the checkpoint cadence / sampling switch of the run in progress. The trainer adopts
     * the request at its next optimizer step, so the reply still carries the old effective values.
     */
    suspend fun trainSettings(
        saveEveryNSteps: Int? = null,
        samplingEnabled: Boolean? = null,
    ): TrainStatus {
        val result = call(
            "train_settings",
            buildJsonObject {
                saveEveryNSteps?.let { put("save_every_n_steps", it) }
                samplingEnabled?.let { put("sampling_enabled", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun datasetTag(
        directory: String,
        threshold: Float,
        batchSize: Int? = null,
        categories: List<String>? = null,
    ): DatasetTagResult {
        val result = call(
            "dataset_tag",
            buildJsonObject {
                put("directory", directory)
                put("threshold", threshold.toDouble())
                batchSize?.let { put("batch_size", it) }
                categories?.takeIf { it.isNotEmpty() }?.let { selected ->
                    putJsonArray("categories") { selected.forEach { add(JsonPrimitive(it)) } }
                }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun taggerInfo(): TaggerInfoResult {
        val result = call("tagger_info", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun hardwareStatus(): HardwareStatus {
        val result = call("hardware_status", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun listSamples(name: String? = null, runId: String? = null): SamplesResponse {
        val result = call(
            "list_samples",
            buildJsonObject {
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun listGeneratedSamples(name: String? = null, runId: String? = null): GeneratedSamplesResponse {
        val result = call(
            "list_generated_samples",
            buildJsonObject {
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun generateSample(
        checkpoint: String,
        prompt: String,
        negativePrompt: String,
        cfg: Float,
        steps: Int,
        seed: Long,
        step: Int? = null,
        name: String? = null,
        runId: String? = null,
    ): GenerateSampleResponse {
        val result = call(
            "generate_sample",
            buildJsonObject {
                put("checkpoint", checkpoint)
                put("prompt", prompt)
                put("negative_prompt", negativePrompt)
                put("cfg", cfg.toDouble())
                put("steps", steps)
                put("seed", seed)
                step?.let { put("step", it) }
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /**
     * Render this checkpoint's `[[validation.samples]]` sets as one detached job (one image per
     * set and repeat). Refused while a live trainer is using the GPU.
     */
    suspend fun generateCheckpointSamples(
        checkpoint: String,
        name: String? = null,
        runId: String? = null,
    ): GenerateSampleResponse {
        val result = call(
            "generate_checkpoint_samples",
            buildJsonObject {
                put("checkpoint", checkpoint)
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /**
     * Render the config's sample sets for every checkpoint whose step is inside `fromStep..toStep`,
     * oldest first, as one detached job. Refused while a live trainer is using the GPU and while
     * another generation is running.
     */
    suspend fun generateCheckpointSamplesBatch(
        fromStep: Int,
        toStep: Int,
        name: String? = null,
        runId: String? = null,
    ): GenerateSampleResponse {
        val result = call(
            "generate_checkpoint_samples_batch",
            buildJsonObject {
                put("from_step", fromStep)
                put("to_step", toStep)
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /**
     * Evaluate one checkpoint: top its sample images up to [depth] (a floor — a checkpoint that
     * already holds that many renders nothing), tag every one of them with the Pixai tagger and
     * score the tags against the prompt each image was rendered from. Detached like the generation
     * entries; the job file carries the progress and the scores. [categories] are tagger category
     * keys; an empty list leaves the helper's default (`general`) in place.
     */
    suspend fun evaluateCheckpoint(
        checkpoint: String,
        depth: Int,
        threshold: Float,
        categories: List<String> = emptyList(),
        name: String? = null,
        runId: String? = null,
    ): GenerateSampleResponse {
        val result = call(
            "evaluate_checkpoint",
            buildJsonObject {
                put("checkpoint", checkpoint)
                put("depth", depth)
                put("threshold", threshold)
                if (categories.isNotEmpty()) {
                    putJsonArray("categories") { categories.forEach { add(JsonPrimitive(it)) } }
                }
                name?.let { put("name", it) }
                runId?.let { put("run_id", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /** Ask a running generation job (the batch one included) to stop; [id] null means the running one. */
    suspend fun cancelGeneration(id: String? = null): GenerateSampleResponse {
        val result = call(
            "cancel_generation",
            buildJsonObject { id?.let { put("id", it) } },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun configGet(): ConfigDocument {
        val result = call("config_get", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun parsedConfig(): Pair<String, AxlTrainerConfig> {
        val doc = configGet()
        val parsed = ConfigImporter.parseConfig(doc.text)
            ?: error("Failed to parse config.toml at ${doc.path}")
        return doc.path to parsed
    }

    suspend fun configSave(text: String): ConfigSaveResult {
        val result = call("config_save", buildJsonObject { put("text", text) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun profileList(): ProfileListResult {
        val result = call("profile_list", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun profileGet(name: String): ProfileDocument {
        val result = call("profile_get", buildJsonObject { put("name", name) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun profileSave(name: String, text: String, overwrite: Boolean): ProfileSaveResult {
        val result = call(
            "profile_save",
            buildJsonObject {
                put("name", name)
                put("text", text)
                put("overwrite", overwrite)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun profileDelete(name: String) {
        call("profile_delete", buildJsonObject { put("name", name) })
    }

    suspend fun tagLexicon(): TagLexiconResult {
        val result = call("tag_lexicon", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun promptMatrix(): PromptMatrixDocument {
        val result = call("prompt_matrix", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun promptProfileList(): PromptProfileListResult {
        val result = call("prompt_profile_list", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun promptProfileGet(name: String): PromptProfileDocument {
        val result = call("prompt_profile_get", buildJsonObject { put("name", name) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun promptProfileSave(name: String, text: String, overwrite: Boolean): PromptProfileSaveResult {
        val result = call(
            "prompt_profile_save",
            buildJsonObject {
                put("name", name)
                put("text", text)
                put("overwrite", overwrite)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun promptProfileDelete(name: String) {
        call("prompt_profile_delete", buildJsonObject { put("name", name) })
    }

    // --- Automation ---

    suspend fun automationConfigGet(): AutomationConfigResult {
        val result = call("automation_config_get", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationConfigSave(settings: AutomationSettings): AutomationConfigSaveResult {
        val result = call(
            "automation_config_save",
            buildJsonObject {
                put(
                    "settings",
                    json.encodeToJsonElement(AutomationSettings.serializer(), settings),
                )
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationDiscover(server: String = ""): ComfyDiscovery {
        val params = if (server.isBlank()) JsonObject(emptyMap()) else buildJsonObject { put("server", server) }
        val result = call("automation_discover", params)
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationWorkflowList(): AutomationWorkflowList {
        val result = call("automation_workflow_list", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationWorkflowValidate(path: String, positiveNode: String = ""): AutomationWorkflow {
        val result = call(
            "automation_workflow_validate",
            buildJsonObject {
                put("path", path)
                if (positiveNode.isNotBlank()) put("positive_node", positiveNode)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationWorkflowSave(name: String, text: String): AutomationWorkflow {
        val result = call(
            "automation_workflow_save",
            buildJsonObject {
                put("name", name)
                put("text", text)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationWorkflowDelete(name: String) {
        call("automation_workflow_delete", buildJsonObject { put("name", name) })
    }

    suspend fun automationPromptList(): PromptSetListResult {
        val result = call("automation_prompt_list", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationPromptGet(name: String): PromptSetItem {
        val result = call("automation_prompt_get", buildJsonObject { put("name", name) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationPromptSave(name: String, text: String): PromptSetSaveResult {
        val result = call(
            "automation_prompt_save",
            buildJsonObject {
                put("name", name)
                put("text", text)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationPromptDelete(name: String) {
        call("automation_prompt_delete", buildJsonObject { put("name", name) })
    }

    /** Starts a job; the caller either sends the prompts themselves or names a saved set. */
    suspend fun automationJobStart(
        prompts: List<String> = emptyList(),
        promptSet: String = "",
        overrides: JsonObject = JsonObject(emptyMap()),
    ): AutomationJobStartResult {
        val result = call(
            "automation_job_start",
            buildJsonObject {
                if (prompts.isNotEmpty()) {
                    put("prompts", JsonArray(prompts.map { JsonPrimitive(it) }))
                }
                if (promptSet.isNotBlank()) put("prompt_set", promptSet)
                overrides.forEach { (key, value) -> put(key, value) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationJobList(): AutomationJobListResult {
        val result = call("automation_job_list", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationJobGet(id: String): AutomationJobDetail {
        val result = call("automation_job_get", buildJsonObject { put("id", id) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationJobCancel(id: String): AutomationJobStartResult {
        val result = call("automation_job_cancel", buildJsonObject { put("id", id) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationJobRetryFailed(id: String): AutomationJobStartResult {
        val result = call("automation_job_retry_failed", buildJsonObject { put("id", id) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun automationJobDelete(id: String) {
        call("automation_job_delete", buildJsonObject { put("id", id) })
    }

    /** Deletes one image (and its sidecar) of a job; the reply is that job's whole detail. */
    suspend fun automationImageDelete(id: String, image: String): AutomationJobDetail {
        val result = call(
            "automation_image_delete",
            buildJsonObject {
                put("id", id)
                put("image", image)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /** Draws one image again with a new random seed, writing over it. */
    suspend fun automationImageRegenerate(id: String, image: String): AutomationJobDetail {
        val result = call(
            "automation_image_regenerate",
            buildJsonObject {
                put("id", id)
                put("image", image)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /** Adds [count] images to one prompt entry, each from its own new random seed. */
    suspend fun automationPromptExtend(id: String, promptIndex: Int, count: Int): AutomationJobDetail {
        val result = call(
            "automation_prompt_extend",
            buildJsonObject {
                put("id", id)
                put("prompt_index", promptIndex)
                put("count", count)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    /** Rewrites one prompt entry's text in the job record; nothing is rendered. */
    suspend fun automationJobPromptEdit(id: String, promptIndex: Int, text: String): AutomationJobDetail {
        val result = call(
            "automation_job_prompt_edit",
            buildJsonObject {
                put("id", id)
                put("prompt_index", promptIndex)
                put("text", text)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun datasetList(directory: String): DatasetListResult {
        val result = call("dataset_list", buildJsonObject { put("directory", directory) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun captionWrite(directory: String, stem: String, text: String) {
        call(
            "caption_write",
            buildJsonObject {
                put("directory", directory)
                put("stem", stem)
                put("text", text)
            },
        )
    }

    suspend fun datasetDrop(
        directory: String,
        rate: Float,
        seed: Long? = null,
        stems: List<String>? = null,
    ): DatasetDropResult {
        val result = call(
            "dataset_drop",
            buildJsonObject {
                put("directory", directory)
                put("rate", rate.toDouble())
                seed?.let { put("seed", it) }
                if (stems != null) {
                    put(
                        "stems",
                        buildJsonArray {
                            stems.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) }
                        },
                    )
                }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun datasetShuffle(directory: String, seed: Long? = null): DatasetShuffleResult {
        val result = call(
            "dataset_shuffle",
            buildJsonObject {
                put("directory", directory)
                seed?.let { put("seed", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun maskGet(directory: String, stem: String): MaskGetResult {
        val result = call(
            "mask_get",
            buildJsonObject {
                put("directory", directory)
                put("stem", stem)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun maskWrite(directory: String, stem: String, pngBase64: String) {
        call(
            "mask_write",
            buildJsonObject {
                put("directory", directory)
                put("stem", stem)
                put("png_base64", pngBase64)
            },
        )
    }

    suspend fun maskDelete(directory: String, stem: String) {
        call(
            "mask_delete",
            buildJsonObject {
                put("directory", directory)
                put("stem", stem)
            },
        )
    }

    suspend fun blobStat(
        paths: List<String>,
        maxEdge: Int,
        quality: Int = 80,
        format: String = "jpeg",
    ): BlobListResult {
        val result = call("blob_stat", blobParams(paths, maxEdge, quality, format), blob = true)
        return json.decodeFromJsonElement(result)
    }

    suspend fun blobBatch(
        paths: List<String>,
        maxEdge: Int,
        quality: Int = 80,
        format: String = "jpeg",
    ): BlobListResult {
        val result = call("blob_batch", blobParams(paths, maxEdge, quality, format), blob = true)
        return json.decodeFromJsonElement(result)
    }

    suspend fun checkpointExport(source: String, dest: String): CheckpointExportResult {
        val result = call(
            "checkpoint_export",
            buildJsonObject {
                put("source", source)
                put("dest", dest)
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun fsListdir(path: String): FsListResult {
        val result = call("fs_listdir", buildJsonObject { put("path", path) })
        return json.decodeFromJsonElement(result)
    }

    suspend fun fsRoots(): FsRootsResult {
        val result = call("fs_roots", JsonObject(emptyMap()))
        return json.decodeFromJsonElement(result)
    }

    suspend fun setEndpoint(host: String, port: Int) {
        val trimmed = host.trim().ifBlank { "127.0.0.1" }
        val bounded = port.coerceIn(1, 65535)
        if (trimmed == wsHost && bounded == wsPort && lanes.values.any { it.isConnected() }) return
        wsHost = trimmed
        wsPort = bounded
        persistWsEndpoint(trimmed, bounded)
        closeConnection()
    }

    suspend fun restart() {
        closeConnection()
        stopSpawnedHelper()
        // Bring the polling lane up first: a failure to start the helper should surface here, on
        // the call the Retry button made, rather than on whichever page polls next.
        lanes.getValue(LaneKind.Poll).ensureConnected(attempts = 4)
        ping()
    }

    private fun blobParams(paths: List<String>, maxEdge: Int, quality: Int, format: String): JsonObject =
        buildJsonObject {
            put(
                "paths",
                buildJsonArray {
                    paths.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) }
                },
            )
            put("max_edge", maxEdge)
            put("quality", quality)
            put("format", format)
        }

    private suspend fun call(method: String, params: JsonObject, blob: Boolean = false): JsonElement {
        // The claim is taken before anything touches the socket, and held until the reply lands: a
        // resource released early would only look like ordering. Refused outright when a long call
        // holds it, which is what makes a click answer instead of waiting for that call.
        return resourceTable.with(IpcResources.claimsFor(method, params), method) {
            laneFor(method, params, blob).call(method, params, IpcResources.timeoutFor(method))
        }
    }

    /**
     * Which connection a call goes out on.
     *
     * Each connection is handled in order by one thread on the other side, so anything queued
     * behind a slow call on that connection waits for it — the reason a ten-minute tag used to
     * freeze every page: it shared one connection with the polls. Writes stay together so their
     * order is the order they were made in (Start before Pause); reads, blobs and the calls that
     * can hold a resource for minutes each have their own connection, so none of them can hold the
     * others up. The split is read off the same claims the resource table uses, so a method cannot
     * be long for one and short for the other.
     */
    private fun laneFor(method: String, params: JsonObject, blob: Boolean): Lane {
        val kind = when {
            blob || method in BLOB_METHODS -> LaneKind.Blob
            IpcResources.isLongRunning(method, params) -> LaneKind.Long
            IpcResources.writes(method, params) -> LaneKind.Control
            else -> LaneKind.Poll
        }
        return lanes.getValue(kind)
    }

    /** Which lane a call would take; the test pins the split without a server. */
    internal fun laneKindFor(method: String, params: JsonObject, blob: Boolean = false): String =
        when {
            blob || method in BLOB_METHODS -> LaneKind.Blob.label
            IpcResources.isLongRunning(method, params) -> LaneKind.Long.label
            IpcResources.writes(method, params) -> LaneKind.Control.label
            else -> LaneKind.Poll.label
        }

    /**
     * One WebSocket connection: its own reader, waiter table and id counter, so a reply can never
     * be matched to a request sent on another connection.
     */
    private inner class Lane(val kind: LaneKind) {
        private val writeMutex = Mutex()
        private val waitersMutex = Mutex()
        private val idMutex = Mutex()
        private var nextId = 1L
        private val waiters = mutableMapOf<Long, CompletableDeferred<IpcResponse>>()
        private var connection: WsConnection? = null
        private var readerJob: Job? = null

        fun isConnected(): Boolean = connection != null

        suspend fun call(method: String, params: JsonObject, timeoutMillis: Long): JsonElement {
            val id = idMutex.withLock { nextId++ }
            val deferred = CompletableDeferred<IpcResponse>()
            waitersMutex.withLock { waiters[id] = deferred }
            val request = json.encodeToString(IpcRequest.serializer(), IpcRequest(id, method, params))
            try {
                ensureConnected()
                writeMutex.withLock {
                    val conn = connection ?: error("IPC WebSocket is not available (${kind.label})")
                    conn.send(request)
                }
                // A reply that never arrives — or one whose id was dropped — fails here instead of
                // holding its waiter, and the resources it claimed, for the rest of the session.
                val response = withTimeoutOrNull(timeoutMillis) { deferred.await() }
                    ?: error("$method did not answer within ${timeoutMillis / 1000} s")
                if (!response.ok) {
                    throw IllegalStateException(response.error ?: "IPC call failed")
                }
                return response.result ?: JsonObject(emptyMap())
            } catch (e: Exception) {
                waitersMutex.withLock { waiters.remove(id) }
                throw e
            }
        }

        suspend fun ensureConnected(attempts: Int = DEFAULT_CONNECT_ATTEMPTS) {
            if (connection != null) return
            connectMutex.withLock {
                if (connection != null) return
                withContext(Dispatchers.Default) {
                    val host = wsHost
                    val port = wsPort
                    // Any live lane means the helper already answered a handshake, so only the
                    // first one probes for it (and spawns it): a second lane costs no extra probe.
                    if (lanes.values.none { it.isConnected() } && !helperListening(host, port)) {
                        spawnHelperIfNeeded(host, port)
                    }
                    var last: Exception? = null
                    repeat(attempts.coerceAtLeast(1)) {
                        if (wsHost != host || wsPort != port) {
                            throw IllegalStateException("endpoint changed while connecting")
                        }
                        try {
                            val conn = withTimeout(5.seconds) { transport.connect(host, port) }
                            connection = conn
                            startReader(conn)
                            handshake()
                            return@withContext
                        } catch (e: Exception) {
                            last = e
                            if (e is HelperBusyException) {
                                // The helper is up and someone else owns it: retrying or spawning
                                // another one would only fight the owner.
                                throw e
                            }
                            delay(250)
                        }
                    }
                    throw IllegalStateException(
                        "Could not connect to ws://$host:$port. ${last?.message ?: ""}".trim(),
                        last,
                    )
                }
            }
        }

        private fun startReader(conn: WsConnection) {
            readerJob?.cancel()
            readerJob = scope.launch {
                try {
                    while (isActive) {
                        val line = conn.receive()
                        if (line.isBlank()) continue
                        val response = try {
                            json.decodeFromString(IpcResponse.serializer(), line)
                        } catch (_: Exception) {
                            continue
                        }
                        val id = response.id ?: continue
                        val waiter = waitersMutex.withLock { waiters.remove(id) }
                        waiter?.complete(response)
                    }
                } catch (_: Exception) {
                    close()
                }
            }
        }

        /**
         * Say who we are before the connection is used.
         *
         * The helper admits one client session, so every lane introduces itself; a refusal means
         * another Ranko (or the web companion) is on this helper, and the message says which and
         * since when. Nothing else is sent on a refused connection, and the socket is dropped.
         */
        private suspend fun handshake() {
            val id = idMutex.withLock { nextId++ }
            val deferred = CompletableDeferred<IpcResponse>()
            waitersMutex.withLock { waiters[id] = deferred }
            try {
                val conn = connection ?: error("IPC WebSocket is not available (${kind.label})")
                val hello = buildJsonObject {
                    put("client", clientName)
                    put("instance", instanceId)
                    put("lane", kind.label)
                }
                conn.send(json.encodeToString(IpcRequest.serializer(), IpcRequest(id, "hello", hello)))
                val response = withTimeoutOrNull(HELLO_TIMEOUT_MILLIS) { deferred.await() }
                    ?: error("the helper did not answer the hello within ${HELLO_TIMEOUT_MILLIS / 1000} s")
                if (!response.ok) {
                    if (response.code == "CLIENT_BUSY") {
                        throw HelperBusyException(response.error ?: "another client owns the helper")
                    }
                    // A helper from before this protocol has no sessions: it is nobody's, and there
                    // is nothing to be admitted to. (It only survives until it is restarted.)
                    if (response.error?.startsWith("unknown method") == true) return
                    throw IllegalStateException(response.error ?: "the helper refused this connection")
                }
            } catch (e: Exception) {
                close()
                throw e
            } finally {
                waitersMutex.withLock { waiters.remove(id) }
            }
        }

        suspend fun close() {
            readerJob?.cancel()
            readerJob = null
            val conn = connection
            connection = null
            conn?.close()
            val pending = waitersMutex.withLock { waiters.values.toList().also { waiters.clear() } }
            pending.forEach { waiter ->
                waiter.completeExceptionally(IllegalStateException("Dashboard helper closed unexpectedly"))
            }
        }
    }

    private suspend fun closeConnection() {
        lanes.values.forEach { it.close() }
    }
}

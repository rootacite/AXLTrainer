package com.acite.axlranko.data

import com.acite.axlranko.model.CheckpointsResponse
import com.acite.axlranko.model.DashboardResponse
import com.acite.axlranko.model.DatasetTagResult
import com.acite.axlranko.model.GenerateSampleResponse
import com.acite.axlranko.model.GeneratedSamplesResponse
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.RunsResponse
import com.acite.axlranko.model.SamplesResponse
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
)

private val BLOB_METHODS = setOf("blob_stat", "blob_batch")

@Inject
@SingleIn(AppScope::class)
class TrainerIpcClient {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
    }
    private val transport = WsTransport()
    private val controlMutex = Mutex()
    private val connectMutex = Mutex()
    private val writeMutex = Mutex()
    private val waitersMutex = Mutex()
    private val idMutex = Mutex()
    private var nextIdValue = 1L
    private val waiters = mutableMapOf<Long, CompletableDeferred<IpcResponse>>()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var readerJob: Job? = null
    private var wsHost = defaultWsHost()
    private var wsPort = defaultWsPort()
    private var connection: WsConnection? = null

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

    suspend fun trainReset(deleteWeights: Boolean = false, name: String? = null): TrainStatus {
        val result = call(
            "train_reset",
            buildJsonObject {
                put("delete_weights", deleteWeights)
                name?.let { put("name", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun datasetTag(
        directory: String,
        threshold: Float,
        batchSize: Int? = null,
    ): DatasetTagResult {
        val result = call(
            "dataset_tag",
            buildJsonObject {
                put("directory", directory)
                put("threshold", threshold.toDouble())
                batchSize?.let { put("batch_size", it) }
            },
        )
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
        if (trimmed == wsHost && bounded == wsPort && connection != null) return
        wsHost = trimmed
        wsPort = bounded
        persistWsEndpoint(trimmed, bounded)
        closeConnection()
    }

    suspend fun restart() {
        closeConnection()
        stopSpawnedHelper()
        ensureConnected(attempts = 4)
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
        val invoke = suspend {
            val id = idMutex.withLock { nextIdValue++ }
            val deferred = CompletableDeferred<IpcResponse>()
            waitersMutex.withLock { waiters[id] = deferred }
            val request = json.encodeToString(IpcRequest.serializer(), IpcRequest(id, method, params))
            try {
                writeMutex.withLock {
                    val conn = connection ?: error("IPC WebSocket is not available")
                    conn.send(request)
                }
                val response = deferred.await()
                if (!response.ok) {
                    throw IllegalStateException(response.error ?: "IPC call failed")
                }
                response.result ?: JsonObject(emptyMap())
            } catch (e: Exception) {
                waitersMutex.withLock { waiters.remove(id) }
                throw e
            }
        }
        ensureConnected()
        return if (blob || method in BLOB_METHODS) {
            invoke()
        } else {
            controlMutex.withLock { invoke() }
        }
    }

    private suspend fun ensureConnected(attempts: Int = 40) {
        if (connection != null) return
        connectMutex.withLock {
            if (connection != null) return
            withContext(Dispatchers.Default) {
                val host = wsHost
                val port = wsPort
                if (!helperListening(host, port)) {
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
                        return@withContext
                    } catch (e: Exception) {
                        last = e
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
                closeConnection()
            }
        }
    }

    private fun closeConnection() {
        readerJob?.cancel()
        readerJob = null
        val conn = connection
        connection = null
        conn?.close()
        val pending = waiters.values.toList()
        waiters.clear()
        pending.forEach { waiter ->
            waiter.completeExceptionally(IllegalStateException("Dashboard helper closed unexpectedly"))
        }
    }
}

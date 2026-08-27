package com.acite.axlranko.data

import com.acite.axlranko.model.DashboardResponse
import com.acite.axlranko.model.SamplesResponse
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.put
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.File
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

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

@Inject
@SingleIn(AppScope::class)
class TrainerIpcClient {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
    }
    private val mutex = Mutex()
    private val nextId = AtomicLong(1)
    private val shutdownHookRegistered = AtomicBoolean(false)

    @Volatile private var process: Process? = null
    @Volatile private var writer: BufferedWriter? = null
    @Volatile private var reader: BufferedReader? = null

    init {
        if (shutdownHookRegistered.compareAndSet(false, true)) {
            Runtime.getRuntime().addShutdownHook(Thread { stopProcess() })
        }
    }

    suspend fun ping() {
        call("ping", JsonObject(emptyMap()))
    }

    suspend fun getDashboard(
        name: String? = null,
        startStep: Int? = null,
        endStep: Int? = null,
    ): DashboardResponse {
        val result = call(
            "dashboard",
            buildJsonObject {
                name?.let { put("name", it) }
                startStep?.let { put("start_step", it) }
                endStep?.let { put("end_step", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun listSamples(name: String? = null): SamplesResponse {
        val result = call(
            "list_samples",
            buildJsonObject {
                name?.let { put("name", it) }
            },
        )
        return json.decodeFromJsonElement(result)
    }

    suspend fun restart() {
        mutex.withLock {
            withContext(Dispatchers.IO) {
                stopProcess()
                startProcess()
            }
        }
        ping()
    }

    private suspend fun call(method: String, params: JsonObject): JsonElement {
        mutex.withLock {
            return withContext(Dispatchers.IO) {
                ensureProcess()
                val id = nextId.getAndIncrement()
                val request = json.encodeToString(IpcRequest.serializer(), IpcRequest(id, method, params))
                val out = writer ?: error("IPC writer is not available")
                val inp = reader ?: error("IPC reader is not available")
                out.write(request)
                out.newLine()
                out.flush()
                readResponse(inp, id)
            }
        }
    }

    private fun readResponse(inp: BufferedReader, id: Long): JsonElement {
        while (true) {
            val line = inp.readLine()
                ?: throw IllegalStateException("Dashboard helper closed unexpectedly")
            if (line.isBlank()) continue
            val response = try {
                json.decodeFromString(IpcResponse.serializer(), line)
            } catch (_: Exception) {
                continue
            }
            if (response.id != null && response.id != id) continue
            if (!response.ok) {
                throw IllegalStateException(response.error ?: "IPC call failed")
            }
            return response.result ?: JsonObject(emptyMap())
        }
    }

    private fun ensureProcess() {
        val running = process?.isAlive == true
        if (running && writer != null && reader != null) return
        stopProcess()
        startProcess()
    }

    private fun startProcess() {
        val root = findRepoRoot()
            ?: error("Could not locate api.py. Run AxlRanko from the trainer repo, or set the working directory to the repo root.")
        val python = System.getenv("AXL_PYTHON")?.takeIf { it.isNotBlank() } ?: "python3"
        val script = File(root, "api.py")
        val builder = ProcessBuilder(python, "-u", script.absolutePath)
            .directory(root)
            .redirectError(ProcessBuilder.Redirect.INHERIT)
        builder.environment()["PYTHONUNBUFFERED"] = "1"

        val started = try {
            builder.start()
        } catch (e: Exception) {
            throw IllegalStateException(
                "Failed to start $python ${script.absolutePath}. Set AXL_PYTHON to your interpreter. ${e.message}",
                e,
            )
        }
        process = started
        writer = BufferedWriter(OutputStreamWriter(started.outputStream, StandardCharsets.UTF_8))
        reader = BufferedReader(InputStreamReader(started.inputStream, StandardCharsets.UTF_8))
    }

    private fun stopProcess() {
        runCatching { writer?.close() }
        runCatching { reader?.close() }
        writer = null
        reader = null
        val running = process
        process = null
        if (running != null) {
            running.destroy()
            if (running.isAlive) {
                running.destroyForcibly()
            }
        }
    }

    companion object {
        fun findRepoRoot(): File? = TrainerRepo.findRoot()
    }
}

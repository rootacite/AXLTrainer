package com.acite.axlranko.data

import kotlin.time.TimeMark
import kotlin.time.TimeSource
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

/**
 * What a call needs, and how long it is willing to wait for it.
 *
 * The client is the only place these are enforced: api.py serves one client session and does no
 * locking of its own (see `API.md`), so the app must not put two calls that fight over the same
 * thing on the wire at once. A claim with `waitMillis = 0` is refused immediately
 * ([IpcBusyException]) rather than queued — the point is that a click answers *now*, even when the
 * thing it needs is busy for ten more minutes.
 */
enum class ResourceMode { Read, Write }

data class ResourceClaim(val name: String, val mode: ResourceMode, val waitMillis: Long = 0L)

/** A call that could not have its resource: [holder] is what is using it, for [heldMillis] now. */
class IpcBusyException(
    val resource: String,
    val holder: String,
    val heldMillis: Long,
) : IllegalStateException("$holder is using $resource (${formatHeld(heldMillis)})")

private fun formatHeld(millis: Long): String =
    if (millis >= 1_000L) "${millis / 1_000L} s" else "$millis ms"

/**
 * One read/write lock per named resource: any number of readers, or one writer.
 *
 * A `Semaphore` of [maxReaders] permits is the whole mechanism — a reader takes one, a writer takes
 * all of them — which keeps acquisition cancellation-safe (`acquire` either grants a permit or
 * throws, so a timed-out writer releases exactly what it took). Grants go to the longest-waiting
 * acquirer, so a writer queued behind readers is not starved by later readers.
 */
private class ResourceLock(private val maxReaders: Int) {
    private val permits = Semaphore(maxReaders)

    /** Take the lock now or not at all; no suspension point, so a writer's grab is atomic. */
    fun tryAcquire(mode: ResourceMode): Boolean = when (mode) {
        ResourceMode.Read -> permits.tryAcquire()
        ResourceMode.Write -> {
            var taken = 0
            while (taken < maxReaders && permits.tryAcquire()) taken++
            if (taken == maxReaders) true else { repeat(taken) { permits.release() }; false }
        }
    }

    suspend fun acquire(mode: ResourceMode, waitMillis: Long): Boolean {
        val acquired = withTimeoutOrNull(waitMillis) {
            when (mode) {
                ResourceMode.Read -> permits.acquire()
                ResourceMode.Write -> {
                    var taken = 0
                    try {
                        while (taken < maxReaders) {
                            permits.acquire()
                            taken++
                        }
                    } catch (e: Throwable) {
                        repeat(taken) { permits.release() }
                        throw e
                    }
                }
            }
        }
        return acquired != null
    }

    fun release(mode: ResourceMode) {
        repeat(if (mode == ResourceMode.Write) maxReaders else 1) { permits.release() }
    }
}

private class Holder(val method: String, val since: TimeMark)

/**
 * The app's own locks, one per resource name. Nothing else enforces them: api.py assumes a single
 * client and leaves ordering to it.
 */
class ResourceTable(private val maxReaders: Int = DEFAULT_MAX_READERS) {

    private val guard = Mutex()
    private val locks = mutableMapOf<String, ResourceLock>()
    private val holders = mutableMapOf<String, MutableList<Holder>>()

    /**
     * Run [block] holding every claim, releasing when it returns (or throws).
     *
     * Claims are taken in name order, so two calls that want the same pair can never take them in
     * opposite orders and deadlock. The lock is held across the whole round trip by the caller —
     * a lock released before the reply would only look like ordering.
     */
    suspend fun <T> with(claims: List<ResourceClaim>, holder: String, block: suspend () -> T): T {
        // One claim per resource: a name claimed twice would be taken twice and never released.
        val ordered = claims.distinctBy { it.name }.sortedBy { it.name }
        val acquired = mutableListOf<ResourceClaim>()
        try {
            for (claim in ordered) {
                val lock = lockFor(claim.name)
                val ok = if (claim.waitMillis <= 0L) {
                    lock.tryAcquire(claim.mode)
                } else {
                    lock.acquire(claim.mode, claim.waitMillis)
                }
                if (!ok) throw busy(claim)
                acquired += claim
                record(claim.name, holder)
            }
        } catch (e: Throwable) {
            release(acquired)
            throw e
        }
        try {
            return block()
        } finally {
            release(acquired)
        }
    }

    /** True while [name] is held by anyone; for tests and diagnostics. */
    suspend fun heldBy(name: String): String? =
        guard.withLock { holders[name]?.firstOrNull()?.method }

    private suspend fun lockFor(name: String): ResourceLock =
        guard.withLock { locks.getOrPut(name) { ResourceLock(maxReaders) } }

    private suspend fun record(name: String, method: String) {
        guard.withLock { holders.getOrPut(name) { mutableListOf() } += Holder(method, TimeSource.Monotonic.markNow()) }
    }

    private suspend fun busy(claim: ResourceClaim): IpcBusyException {
        val blocker = guard.withLock { holders[claim.name]?.firstOrNull() }
        val held = blocker?.since?.elapsedNow()?.inWholeMilliseconds ?: 0L
        return IpcBusyException(claim.name, blocker?.method ?: "another request", held)
    }

    private suspend fun release(claims: List<ResourceClaim>) {
        for (claim in claims.asReversed()) {
            lockFor(claim.name).release(claim.mode)
            guard.withLock {
                val list = holders[claim.name] ?: return@withLock
                // The method that took it is the *last* entry for its name; readers may stack.
                list.removeLastOrNull()
                if (list.isEmpty()) holders.remove(claim.name)
            }
        }
    }

    companion object {
        /**
         * Readers per resource. Readers are cheap and short (a poll reads a file), and a writer
         * needs all of them at once; 64 leaves room for every page polling at the same time.
         */
        const val DEFAULT_MAX_READERS = 64
    }
}

/**
 * Which resources each IPC method touches.
 *
 * Every method the client can send has a row here — [IpcResourcesTest] reads
 * `TrainerIpcClient.kt` and fails when one is missing — because a missing row silently means
 * "no lock at all". Waits follow the rule that a *button* may queue a moment while a *long* job
 * may not: the training controls wait for the runtime files (5 s), the small stores for their file
 * (1 s), and anything that can be held for minutes (`gpu`, a dataset folder, an export target) is
 * refused at once so the click answers immediately.
 *
 * Reads only claim a resource that is rewritten wholesale (`config`, `profiles`, `run:` and the
 * automation stores). `dataset_list` and `blob_*` are left out on purpose: they return whatever the
 * files say at that moment, which is the behaviour they have today, and locking them would put a
 * ten-minute tag in front of the Images page.
 */
internal object IpcResources {

    private const val RUNTIME_WAIT = 5_000L
    private const val STORE_WAIT = 1_000L
    private const val READ_WAIT = 500L

    private fun runtime() = ResourceClaim("runtime", ResourceMode.Write, RUNTIME_WAIT)
    private fun gpu() = ResourceClaim("gpu", ResourceMode.Write, 0L)
    private fun config(read: Boolean) =
        ResourceClaim("config", if (read) ResourceMode.Read else ResourceMode.Write, if (read) READ_WAIT else STORE_WAIT)
    private fun profiles(read: Boolean) =
        ResourceClaim("profiles", if (read) ResourceMode.Read else ResourceMode.Write, if (read) READ_WAIT else STORE_WAIT)
    private fun dataset(directory: String?) = ResourceClaim("dataset:${directory.orEmpty()}", ResourceMode.Write, 0L)
    private fun file(path: String?) = ResourceClaim("path:${path.orEmpty()}", ResourceMode.Write, 0L)
    private fun mask(directory: String?, stem: String?) =
        file("$directory/${stem.orEmpty()}.mask.png")
    private fun automation(name: String, read: Boolean = false) = ResourceClaim(
        "automation:$name",
        if (read) ResourceMode.Read else ResourceMode.Write,
        if (read) READ_WAIT else STORE_WAIT,
    )
    private fun job(id: String?) = ResourceClaim("automation:job:${id.orEmpty()}", ResourceMode.Write, STORE_WAIT)

    /**
     * The run a request is about. A request that names none is about "whatever `state.json` is on",
     * which the client cannot resolve before the reply, so it claims that slot by name (`current`):
     * a Reset and a poll of the followed run then take the same lock.
     */
    private fun run(params: JsonObject, mode: ResourceMode) = ResourceClaim(
        "run:${params.text("run_id") ?: "current"}",
        mode,
        if (mode == ResourceMode.Read) READ_WAIT else STORE_WAIT,
    )

    private fun JsonObject.text(key: String): String? =
        (this[key] as? JsonPrimitive)?.contentOrNull?.takeIf { it.isNotBlank() }

    private val policy: Map<String, (JsonObject) -> List<ResourceClaim>> = mapOf(
        // --- runtime state: the training controls write state / command / settings files ---
        // `train_status` claims nothing on purpose. It is the poll the whole app runs on, and the
        // runtime files are written atomically (temp file + rename), so a reader always sees a
        // complete file: the old one or the new one. Giving it a Read claim made it wait behind a
        // settings write that the server was holding for eight seconds, and after its own five it
        // failed with BUSY — a poll that fails because someone pressed a button.
        "train_status" to { _ -> emptyList() },
        "train_start" to { _ -> listOf(runtime()) },
        "train_pause" to { _ -> listOf(runtime()) },
        "train_resume" to { _ -> listOf(runtime()) },
        "train_stop" to { _ -> listOf(runtime()) },
        "train_settings" to { _ -> listOf(runtime()) },
        "train_reset" to { params -> listOf(runtime(), run(params, ResourceMode.Write)) },

        // --- the GPU is single-tenant; a second user of it is refused at once ---
        "generate_sample" to { _ -> listOf(gpu()) },
        "generate_checkpoint_samples" to { _ -> listOf(gpu()) },
        "generate_checkpoint_samples_batch" to { _ -> listOf(gpu()) },
        // An evaluation renders its missing images and then tags them, so it holds the GPU itself.
        "evaluate_checkpoint" to { _ -> listOf(gpu()) },
        // Read-only: the prompts (and their tag frequencies) the evaluation panel's picker draws.
        "evaluation_prompts" to { _ -> emptyList() },
        "dataset_tag" to { params -> listOf(dataset(params.text("directory")), gpu()) },
        "automation_job_start" to { _ -> listOf(gpu()) },
        "automation_job_retry_failed" to { params -> listOf(job(params.text("id")), gpu()) },
        "automation_image_regenerate" to { params -> listOf(job(params.text("id")), gpu()) },
        "automation_prompt_extend" to { params -> listOf(job(params.text("id")), gpu()) },
        "automation_prompt_extend_all" to { params -> listOf(job(params.text("id")), gpu()) },

        // --- one dataset folder: a tag holds it for minutes, so the next writer is refused ---
        "caption_write" to { params -> listOf(dataset(params.text("directory"))) },
        "dataset_drop" to { params -> listOf(dataset(params.text("directory"))) },
        "dataset_shuffle" to { params -> listOf(dataset(params.text("directory"))) },
        "mask_write" to { params ->
            listOf(dataset(params.text("directory")), mask(params.text("directory"), params.text("stem")))
        },
        "mask_delete" to { params ->
            listOf(dataset(params.text("directory")), mask(params.text("directory"), params.text("stem")))
        },

        // --- one file that is written as a whole ---
        "checkpoint_export" to { params -> listOf(file(params.text("dest"))) },

        // --- one run's own files ---
        "dashboard" to { params -> listOf(run(params, ResourceMode.Read)) },
        "list_samples" to { params -> listOf(run(params, ResourceMode.Read)) },
        "list_generated_samples" to { params -> listOf(run(params, ResourceMode.Read)) },
        "checkpoint_pins" to { params -> listOf(run(params, ResourceMode.Read)) },
        "checkpoint_pin_set" to { params -> listOf(run(params, ResourceMode.Write)) },

        // --- config.toml and the preset stores are small files rewritten in place ---
        "config_get" to { _ -> listOf(config(read = true)) },
        "config_save" to { _ -> listOf(config(read = false)) },
        "profile_list" to { _ -> listOf(profiles(read = true)) },
        "profile_get" to { _ -> listOf(profiles(read = true)) },
        "profile_save" to { _ -> listOf(profiles(read = false)) },
        "profile_delete" to { _ -> listOf(profiles(read = false)) },
        "prompt_profile_list" to { _ -> listOf(profiles(read = true)) },
        "prompt_profile_get" to { _ -> listOf(profiles(read = true)) },
        "prompt_profile_save" to { _ -> listOf(profiles(read = false)) },
        "prompt_profile_delete" to { _ -> listOf(profiles(read = false)) },

        // --- the Automation stores: settings, one workflow, one prompt set, one job ---
        "automation_config_get" to { _ -> listOf(automation("config", read = true)) },
        "automation_config_save" to { _ -> listOf(automation("config")) },
        "automation_workflow_list" to { _ -> listOf(automation("workflows", read = true)) },
        "automation_workflow_save" to { params -> listOf(automation("workflow:${params.text("name").orEmpty()}")) },
        "automation_workflow_delete" to { params -> listOf(automation("workflow:${params.text("name").orEmpty()}")) },
        "automation_prompt_save" to { params -> listOf(automation("prompt:${params.text("name").orEmpty()}")) },
        "automation_prompt_delete" to { params -> listOf(automation("prompt:${params.text("name").orEmpty()}")) },
        "automation_job_cancel" to { params -> listOf(job(params.text("id"))) },
        "automation_job_delete" to { params -> listOf(job(params.text("id"))) },
        "automation_image_delete" to { params -> listOf(job(params.text("id"))) },
        "automation_job_prompt_edit" to { params -> listOf(job(params.text("id"))) },

        // --- read-only, or reading the files is the point (see the class comment) ---
        "ping" to { _ -> emptyList() },
        "list_runs" to { _ -> emptyList() },
        "list_checkpoints" to { _ -> emptyList() },
        "tagger_info" to { _ -> emptyList() },
        "hardware_status" to { _ -> emptyList() },
        "cancel_generation" to { _ -> emptyList() },
        "tag_lexicon" to { _ -> emptyList() },
        "prompt_matrix" to { _ -> emptyList() },
        "automation_discover" to { _ -> emptyList() },
        "automation_workflow_validate" to { _ -> emptyList() },
        "automation_prompt_list" to { _ -> emptyList() },
        "automation_prompt_get" to { _ -> emptyList() },
        "automation_job_list" to { _ -> emptyList() },
        "automation_job_get" to { _ -> emptyList() },
        "dataset_list" to { _ -> emptyList() },
        // A directory walk for the Training section's step estimate: read-only, and it must not
        // queue behind a ten-minute tag on the same folder.
        "dataset_counts" to { _ -> emptyList() },
        "mask_get" to { _ -> emptyList() },
        "blob_stat" to { _ -> emptyList() },
        "blob_batch" to { _ -> emptyList() },
        "fs_listdir" to { _ -> emptyList() },
        "fs_roots" to { _ -> emptyList() },
    )

    /** The claims [method] takes with [params]; a method with no row is a bug the test catches. */
    fun claimsFor(method: String, params: JsonObject): List<ResourceClaim> =
        policy[method]?.invoke(params) ?: emptyList()

    internal fun knownMethods(): Set<String> = policy.keys

    /**
     * True when the call can hold its resource for minutes: the GPU, a dataset folder (a tagger
     * over one), an export target, a job record whose runner is starting. Those go on their own
     * connection — a connection serves its queue in order, so a ten-minute tag on the polling
     * connection is what used to freeze every page.
     */
    fun isLongRunning(method: String, params: JsonObject): Boolean =
        claimsFor(method, params).any { claim ->
            claim.name == GPU_RESOURCE || LONG_RESOURCE_PREFIXES.any { claim.name.startsWith(it) }
        }

    /** True when the call changes something. Writes share one connection so their order is kept. */
    fun writes(method: String, params: JsonObject): Boolean =
        claimsFor(method, params).any { it.mode == ResourceMode.Write }

    /**
     * How long a call may stay unanswered before it fails.
     *
     * The default covers the slowest normal call by a wide margin (`dashboard` re-reads a run's
     * whole TensorBoard history: 343 ms measured at 24 k points, and the first `list_runs` of a
     * session scans every run directory). The overrides are the calls that legitimately run for
     * minutes — a tagger over 600 images measured ~1.6 s each — so that a timeout means something
     * is wrong rather than "the work is big". A timeout fails the call and frees its resources; a
     * reply that arrives later is dropped (ids are never reused).
     */
    fun timeoutFor(method: String): Long = timeouts[method] ?: DEFAULT_TIMEOUT_MILLIS

    internal fun timeoutOverrides(): Map<String, Long> = timeouts

    const val DEFAULT_TIMEOUT_MILLIS = 15_000L

    private const val GPU_RESOURCE = "gpu"
    private val LONG_RESOURCE_PREFIXES = listOf("dataset:", "path:", "automation:job:")

    private val timeouts: Map<String, Long> = mapOf(
        "dataset_tag" to 60 * 60_000L,
        "checkpoint_export" to 30 * 60_000L,
        "dataset_shuffle" to 5 * 60_000L,
        "dataset_drop" to 5 * 60_000L,
        "automation_job_delete" to 2 * 60_000L,
        "train_start" to 30_000L,
        "generate_sample" to 60_000L,
        "generate_checkpoint_samples" to 60_000L,
        "generate_checkpoint_samples_batch" to 60_000L,
        "automation_job_start" to 60_000L,
        "automation_job_retry_failed" to 60_000L,
        "automation_image_regenerate" to 60_000L,
        "automation_prompt_extend" to 60_000L,
        "automation_prompt_extend_all" to 60_000L,
        "automation_workflow_validate" to 60_000L,
        "caption_write" to 60_000L,
        "mask_write" to 60_000L,
        "mask_delete" to 60_000L,
        "tagger_info" to 60_000L,
        "blob_batch" to 60_000L,
        "blob_stat" to 30_000L,
        "config_save" to 30_000L,
        "profile_save" to 30_000L,
        "profile_delete" to 30_000L,
        "prompt_profile_save" to 30_000L,
        "prompt_profile_delete" to 30_000L,
    )
}

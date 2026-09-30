package com.acite.axlranko.data

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * The client-side resource table: the only locking in the system, because api.py serves one client
 * and takes none of its own. What these tests pin is the click-answers-now rule — a resource held by
 * a long call refuses the next claim immediately instead of queueing it — plus the policy table
 * covering every method the client can send.
 *
 * Nothing here talks to a helper: a test that reached the transport would connect to (or spawn) one.
 */
class IpcResourcesTest {

    private fun write(name: String, wait: Long = 0L) = ResourceClaim(name, ResourceMode.Write, wait)
    private fun read(name: String, wait: Long = 0L) = ResourceClaim(name, ResourceMode.Read, wait)

    @Test
    fun aHeldResourceIsRefusedAtOnceInsteadOfQueueing() = runBlocking {
        val table = ResourceTable()
        val holding = async { table.with(listOf(write("dataset:/x")), "dataset_tag") { delay(5_000) } }
        delay(50)

        val failure = assertFailsWith<IpcBusyException> {
            withTimeout(1_000) { table.with(listOf(write("dataset:/x")), "dataset_shuffle") { Unit } }
        }
        assertEquals("dataset:/x", failure.resource)
        assertEquals("dataset_tag", failure.holder)
        assertTrue(failure.heldMillis >= 0L, "held ${failure.heldMillis} ms")
        holding.cancel()
    }

    @Test
    fun aClaimThatMayWaitGetsTheLockWhenItIsReleased() = runBlocking {
        val table = ResourceTable()
        val order = mutableListOf<String>()
        val holder = async {
            table.with(listOf(write("runtime")), "train_settings") {
                delay(200)
                order += "first"
            }
        }
        delay(50)
        table.with(listOf(write("runtime", wait = 2_000L)), "train_pause") { order += "second" }
        holder.await()
        assertEquals(listOf("first", "second"), order)
    }

    @Test
    fun readersRunTogetherAndWritersWaitForThem() = runBlocking {
        val table = ResourceTable()
        val first = async { table.with(listOf(read("run:rein_1")), "dashboard") { delay(300) } }
        delay(50)
        // A second reader is not blocked by the first: that is the whole point of Read.
        val second = async { table.with(listOf(read("run:rein_1")), "list_samples") { delay(50) } }
        second.await()
        assertTrue(first.isActive, "the reader waited for another reader")

        // A writer does wait for both of them.
        val started = System.currentTimeMillis()
        table.with(listOf(write("run:rein_1", wait = 2_000L)), "train_reset") { Unit }
        assertTrue(System.currentTimeMillis() - started >= 150, "the writer did not wait for the readers")
        first.await()
    }

    @Test
    fun differentResourcesDoNotBlockEachOther() = runBlocking {
        val table = ResourceTable()
        val holder = async { table.with(listOf(write("dataset:/x")), "dataset_tag") { delay(500) } }
        delay(50)
        // A poll of the run and an export run next to the tag; only the dataset folder is taken.
        table.with(listOf(read("run:current")), "dashboard") { Unit }
        table.with(listOf(write("path:/tmp/out.safetensors")), "checkpoint_export") { Unit }
        holder.await()
    }

    @Test
    fun claimsAreTakenInNameOrderSoTwoCallsCannotDeadlock() = runBlocking {
        val table = ResourceTable()
        val a = async { table.with(listOf(write("gpu"), write("dataset:/x")), "dataset_tag") { delay(200) } }
        delay(20)
        val b = async {
            table.with(listOf(write("dataset:/y"), write("gpu", wait = 2_000L)), "generate_sample") { Unit }
        }
        withTimeout(5_000) { a.await(); b.await() }
    }

    @Test
    fun theLockIsReleasedWhenTheCallFails() = runBlocking {
        val table = ResourceTable()
        assertFailsWith<IllegalStateException> {
            table.with(listOf(write("dataset:/x")), "dataset_tag") { error("server said no") }
        }
        // Still free, so the next caller takes it rather than reporting a phantom holder.
        table.with(listOf(write("dataset:/x")), "dataset_shuffle") { Unit }
        assertEquals(null, table.heldBy("dataset:/x"))
    }

    @Test
    fun theBusyMessageNamesTheHolderAndHowLong() {
        assertEquals(
            "dataset_tag is using dataset:/x (41 s)",
            IpcBusyException("dataset:/x", "dataset_tag", 41_000).message,
        )
        assertEquals(
            "dataset_tag is using dataset:/x (250 ms)",
            IpcBusyException("dataset:/x", "dataset_tag", 250).message,
        )
    }

    @Test
    fun claimsComeFromTheRequestParameters() {
        assertEquals(
            listOf(
                ResourceClaim("dataset:/data/st", ResourceMode.Write, 0L),
                ResourceClaim("gpu", ResourceMode.Write, 0L),
            ),
            IpcResources.claimsFor("dataset_tag", params("directory" to "/data/st")),
        )
        assertEquals(
            listOf(ResourceClaim("path:/tmp/copy.safetensors", ResourceMode.Write, 0L)),
            IpcResources.claimsFor(
                "checkpoint_export",
                params("source" to "/out/a.safetensors", "dest" to "/tmp/copy.safetensors"),
            ),
        )
        assertEquals(
            listOf(ResourceClaim("run:rein_1", ResourceMode.Read, 500L)),
            IpcResources.claimsFor("list_samples", params("run_id" to "rein_1")),
        )
        // A request that names no run is about whatever `state.json` is on: the same slot a Reset
        // takes, so the two cannot cross.
        assertEquals(
            listOf(ResourceClaim("run:current", ResourceMode.Read, 500L)),
            IpcResources.claimsFor("dashboard", JsonObject(emptyMap())),
        )
        // Reset rewrites the runtime state and acts on the followed run, so it takes both slots.
        assertEquals(
            listOf(
                ResourceClaim("runtime", ResourceMode.Write, 5_000L),
                ResourceClaim("run:current", ResourceMode.Write, 1_000L),
            ),
            IpcResources.claimsFor("train_reset", JsonObject(emptyMap())),
        )
        assertEquals(
            listOf(ResourceClaim("automation:job:j1", ResourceMode.Write, 1_000L)),
            IpcResources.claimsFor("automation_job_delete", params("id" to "j1")),
        )
        // The mask sidecar is a file of its own inside the folder it belongs to.
        assertEquals(
            listOf(
                ResourceClaim("dataset:/data/st", ResourceMode.Write, 0L),
                ResourceClaim("path:/data/st/0001.mask.png", ResourceMode.Write, 0L),
            ),
            IpcResources.claimsFor("mask_write", params("directory" to "/data/st", "stem" to "0001")),
        )
        assertEquals(emptyList(), IpcResources.claimsFor("blob_batch", params("paths" to "/x")))
    }

    @Test
    fun theWritersAndReadersOfOneResourceUseTheDocumentedWaits() {
        val tag = IpcResources.claimsFor("dataset_tag", params("directory" to "/data/st"))
        assertEquals(0L, tag.first { it.name == "dataset:/data/st" }.waitMillis)
        assertEquals(0L, tag.first { it.name == "gpu" }.waitMillis)
        assertEquals(
            5_000L,
            IpcResources.claimsFor("train_stop", JsonObject(emptyMap())).single().waitMillis,
        )
        assertEquals(
            1_000L,
            IpcResources.claimsFor("config_save", JsonObject(emptyMap())).single().waitMillis,
        )
        assertEquals(
            ResourceMode.Read,
            IpcResources.claimsFor("config_get", JsonObject(emptyMap())).single().mode,
        )
    }

    /**
     * Every method the client can send has a policy row. A missing row silently means "no lock",
     * which is the failure this table exists to prevent, so the list is read out of the client
     * itself rather than copied by hand.
     */
    @Test
    fun thePolicyCoversEveryMethodTheClientSends() {
        val source = File(repoRoot(), "ranko/shared/src/commonMain/kotlin/com/acite/axlranko/data/TrainerIpcClient.kt")
        assertTrue(source.isFile, "missing ${source.path}")
        val methods = Regex("call\\(\\s*\"([a-z_]+)\"").findAll(source.readText())
            .map { it.groupValues[1] }
            .toSet()
        assertTrue(methods.size > 60, "only found ${methods.size} call sites")
        val missing = methods - IpcResources.knownMethods()
        assertEquals(emptySet(), missing, "methods with no policy row: $missing")
    }

    @Test
    fun theClientRefusesBusyWorkBeforeItTouchesTheSocket() = runBlocking {
        val client = TrainerIpcClient()
        // Nothing is listening in this test, so a call that got past the claim would fail with a
        // transport error: a BUSY here can only come from the table, before any socket work.
        client.resourceTable.with(listOf(write("dataset:/data/st")), "dataset_tag") {
            val failure = assertFailsWith<IpcBusyException> {
                withTimeout(2_000) { client.captionWrite("/data/st", "0001", "tags") }
            }
            assertEquals("dataset:/data/st", failure.resource)
            assertEquals("dataset_tag", failure.holder)
            // The refused call took nothing of its own.
            assertEquals("dataset_tag", client.resourceTable.heldBy("dataset:/data/st"))
        }
        assertEquals(null, client.resourceTable.heldBy("dataset:/data/st"))
    }

    @Test
    fun theStatusPollHoldsNothingSoItCannotWaitBehindAWrite() {
        // A measured failure: with a Read claim on `runtime`, `train_status` spent its whole 5 s
        // budget behind a settings write the server was sitting on, then failed with BUSY. The
        // runtime files are replaced atomically, so the poll needs no lock at all.
        assertEquals(emptyList(), IpcResources.claimsFor("train_status", JsonObject(emptyMap())))
        assertFalse(IpcResources.writes("train_status", JsonObject(emptyMap())))
        assertFalse(IpcResources.isLongRunning("train_status", JsonObject(emptyMap())))
    }

    @Test
    fun theLongCallsAreTheOnesWithAPerMinuteBudget() {
        // A tagger over 600 images runs for ~10 minutes and an export can crawl on a slow disk;
        // the default would cut them off.
        assertTrue(IpcResources.timeoutFor("dataset_tag") >= 30 * 60_000L)
        assertTrue(IpcResources.timeoutFor("checkpoint_export") >= 30 * 60_000L)
        assertTrue(IpcResources.timeoutFor("dataset_shuffle") >= 60_000L)
        assertTrue(IpcResources.timeoutFor("train_start") > IpcResources.DEFAULT_TIMEOUT_MILLIS)
        // Everything else answers in well under the default: `dashboard` is the slow one at 343 ms.
        assertEquals(IpcResources.DEFAULT_TIMEOUT_MILLIS, IpcResources.timeoutFor("dashboard"))
        assertEquals(IpcResources.DEFAULT_TIMEOUT_MILLIS, IpcResources.timeoutFor("train_status"))
        assertEquals(IpcResources.DEFAULT_TIMEOUT_MILLIS, IpcResources.timeoutFor("ping"))
    }

    @Test
    fun everyTimeoutOverrideNamesARealMethodAndExceedsTheDefault() {
        val overrides = IpcResources.timeoutOverrides()
        val unknown = overrides.keys - IpcResources.knownMethods()
        assertEquals(emptySet(), unknown, "timeouts for methods that do not exist: $unknown")
        val tooShort = overrides.filterValues { it <= IpcResources.DEFAULT_TIMEOUT_MILLIS }
        assertEquals(emptyMap(), tooShort, "overrides below the default: $tooShort")
    }

    @Test
    fun theLaneSplitFollowsTheClaims() {
        val data = params("directory" to "/data/st")
        val job = params("id" to "j1")
        // Long: these can hold the GPU, a dataset folder, an export target or a job for minutes.
        for ((method, p) in listOf(
            "dataset_tag" to data,
            "checkpoint_export" to params("dest" to "/tmp/copy.safetensors"),
            "mask_write" to params("directory" to "/data/st", "stem" to "0001"),
            "dataset_shuffle" to data,
            "automation_job_delete" to job,
            "generate_sample" to params("checkpoint" to "/out/a.safetensors"),
        )) {
            assertTrue(IpcResources.isLongRunning(method, p), "$method should ride the long lane")
        }
        // Writes that cannot run long share the control lane, so their order is the order they
        // were made in; reads and blobs take the other two.
        for (method in listOf("train_settings", "train_pause", "train_reset", "config_save", "checkpoint_pin_set")) {
            val p = if (method.startsWith("train")) params("save_every_n_steps" to "50") else JsonObject(emptyMap())
            assertFalse(IpcResources.isLongRunning(method, p), "$method is not long")
            assertTrue(IpcResources.writes(method, p), "$method should ride the control lane")
        }
        for (method in listOf("dashboard", "train_status", "list_runs", "checkpoint_pins", "dataset_list")) {
            assertFalse(IpcResources.writes(method, JsonObject(emptyMap())), "$method is a read")
        }
        assertFalse(IpcResources.writes("blob_batch", params("paths" to "/x")))
        assertFalse(IpcResources.isLongRunning("blob_batch", params("paths" to "/x")))
    }

    @Test
    fun everyCallGoesOnTheLaneItsClaimsImply() {
        val client = TrainerIpcClient()
        val settings = params("save_every_n_steps" to "50")
        // Reads poll; writes share one connection so their order is the order they were made in.
        assertEquals("poll", client.laneKindFor("dashboard", JsonObject(emptyMap())))
        assertEquals("poll", client.laneKindFor("train_status", JsonObject(emptyMap())))
        assertEquals("control", client.laneKindFor("train_settings", settings))
        assertEquals("control", client.laneKindFor("train_pause", JsonObject(emptyMap())))
        assertEquals("control", client.laneKindFor("config_save", JsonObject(emptyMap())))
        // Anything a long job can hold gets its own connection, whatever kind of call it is.
        assertEquals("long", client.laneKindFor("dataset_tag", params("directory" to "/data/st")))
        assertEquals("long", client.laneKindFor("mask_write", params("directory" to "/data/st", "stem" to "0001")))
        assertEquals("long", client.laneKindFor("caption_write", params("directory" to "/data/st")))
        assertEquals("long", client.laneKindFor("checkpoint_export", params("dest" to "/tmp/a.safetensors")))
        assertEquals("long", client.laneKindFor("generate_sample", params("checkpoint" to "/out/a.safetensors")))
        // Blobs keep to themselves, and a write may be sent as one.
        assertEquals("blob", client.laneKindFor("blob_batch", params("paths" to "/x")))
        assertEquals("blob", client.laneKindFor("ping", JsonObject(emptyMap()), blob = true))
    }

    private fun params(vararg pairs: Pair<String, String>): JsonObject =
        buildJsonObject { pairs.forEach { (key, value) -> put(key, value) } }

    private fun repoRoot(): File {
        var dir: File? = File(System.getProperty("user.dir").orEmpty()).absoluteFile
        while (dir != null) {
            if (File(dir, "api.py").isFile && File(dir, "trainer").isDirectory) return dir
            dir = dir.parentFile
        }
        throw AssertionError("could not find the repo root from ${System.getProperty("user.dir")}")
    }
}

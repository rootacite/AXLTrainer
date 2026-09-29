package com.acite.axlranko.data

import androidx.compose.runtime.compositionLocalOf

import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

data class BlobRef(
    val path: String,
    val maxEdge: Int = 256,
    val quality: Int,
    val format: String = "jpeg",
    /**
     * Asks for the bytes to be **revalidated** rather than served from the client cache, and takes
     * part in the request identity, so a caller that knows the file behind [path] can become a
     * different image sets something that changes with it — the Gallery passes the seed the job
     * record credits that image to, and a redraw (same name, new pixels) then fetches the new
     * bytes instead of the ones read the first time.
     */
    val rev: String = "",
)

val LocalThumbnailQuality = compositionLocalOf { 80 }

private data class BlobWaiter(
    val ref: BlobRef,
    val deferred: CompletableDeferred<ByteArray>,
)

@Inject
@SingleIn(AppScope::class)
class BlobStore(private val ipc: TrainerIpcClient) {
    private val hashByKey = mutableMapOf<String, String>()
    private val bytesByHash = mutableMapOf<String, ByteArray>()
    private val cacheLock = Mutex()
    private val mailbox = Channel<BlobWaiter>(Channel.UNLIMITED)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    init {
        scope.launch { drain() }
    }

    suspend fun get(ref: BlobRef): ByteArray {
        // A ref that carries a revision does not trust what is cached: it goes through the mailbox,
        // where the stat's hash (which the server derives from the file's mtime and size) either
        // confirms the bytes we already hold or makes them a miss. Everything else — the whole
        // dataset, the Dashboard's samples — keeps the fast path.
        if (ref.rev.isEmpty()) {
            cacheLock.withLock {
                val hash = hashByKey[key(ref)]
                if (hash != null) {
                    bytesByHash[hash]?.let { return it }
                }
            }
        }
        val waiter = BlobWaiter(ref, CompletableDeferred())
        mailbox.send(waiter)
        return waiter.deferred.await()
    }

    private suspend fun drain() {
        while (true) {
            val first = mailbox.receive()
            val batch = mutableListOf(first)
            delay(16)
            while (true) {
                val next = mailbox.tryReceive().getOrNull() ?: break
                batch.add(next)
            }
            delay(1)
            while (true) {
                val extra = mailbox.tryReceive().getOrNull() ?: break
                batch.add(extra)
            }
            fulfill(batch)
        }
    }

    private suspend fun fulfill(batch: List<BlobWaiter>) {
        batch.groupBy { Triple(it.ref.maxEdge, it.ref.quality, it.ref.format) }.forEach { (params, waiters) ->
            val (maxEdge, quality, format) = params
            val unique = waiters.map { it.ref.path }.distinct()
            try {
                val stat = ipc.blobStat(unique, maxEdge, quality, format)
                val missing = mutableListOf<String>()
                val hashOf = mutableMapOf<String, String>()
                stat.items.forEach { item ->
                    if (item.error != null || item.hash == null) {
                        missing.add(item.path)
                        return@forEach
                    }
                    hashOf[item.path] = item.hash
                    val cached = cacheLock.withLock {
                        val previous = hashByKey[key(item.path, maxEdge, quality, format)]
                        if (previous != null && previous != item.hash) {
                            bytesByHash.remove(previous)
                        }
                        hashByKey[key(item.path, maxEdge, quality, format)] = item.hash
                        bytesByHash[item.hash]
                    }
                    if (cached == null) missing.add(item.path)
                }
                if (missing.isNotEmpty()) {
                    val got = ipc.blobBatch(missing.distinct(), maxEdge, quality, format)
                    got.items.forEach { item ->
                        val payload = item.base64
                        val hash = item.hash
                        if (item.error != null || payload == null || hash == null) return@forEach
                        val bytes = decodeBase64(payload)
                        cacheLock.withLock {
                            hashByKey[key(item.path, maxEdge, quality, format)] = hash
                            bytesByHash[hash] = bytes
                        }
                    }
                }
                waiters.forEach { waiter ->
                    val bytes = cacheLock.withLock {
                        val hash = hashByKey[key(waiter.ref)]
                        hash?.let { bytesByHash[it] }
                    }
                    if (bytes != null) {
                        waiter.deferred.complete(bytes)
                    } else {
                        waiter.deferred.completeExceptionally(
                            IllegalStateException("blob missing: ${waiter.ref.path}"),
                        )
                    }
                }
            } catch (e: Exception) {
                waiters.forEach { it.deferred.completeExceptionally(e) }
            }
        }
    }

    private fun key(ref: BlobRef) = key(ref.path, ref.maxEdge, ref.quality, ref.format)

    private fun key(path: String, maxEdge: Int, quality: Int, format: String) =
        "$path|$maxEdge|$quality|$format"
}

@OptIn(ExperimentalEncodingApi::class)
internal fun decodeBase64(value: String): ByteArray = Base64.decode(value)

@OptIn(ExperimentalEncodingApi::class)
internal fun encodeBase64(bytes: ByteArray): String = Base64.encode(bytes)

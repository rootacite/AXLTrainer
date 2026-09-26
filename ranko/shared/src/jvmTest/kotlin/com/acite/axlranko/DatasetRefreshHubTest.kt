package com.acite.axlranko

import com.acite.axlranko.data.DatasetRefreshHub
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The pages that cache dataset state (Images, Statistics) reload off this hub, so a signal that
 * never arrives leaves one of them showing captions or samples that are already gone.
 */
class DatasetRefreshHubTest {

    @Test
    fun anEmitReachesTheSubscriber() = runBlocking {
        val hub = DatasetRefreshHub()
        var received = 0
        val collector = launch(Dispatchers.Unconfined) { hub.events.collect { received++ } }
        assertTrue(hub.notifyDatasetChanged())
        yield()
        assertEquals(1, received)
        collector.cancel()
    }

    /**
     * A batch of edits emits several times in a row while the subscriber is still busy with the
     * first one. Every one of them must be accepted: with a plain buffered flow the later emits are
     * refused outright, and the dataset change they stand for never reaches the other page.
     */
    @Test
    fun aBurstIsAcceptedWhileTheSubscriberIsBusy() = runBlocking {
        val hub = DatasetRefreshHub()
        val gate = CompletableDeferred<Unit>()
        var received = 0
        val collector = launch(Dispatchers.Unconfined) {
            hub.events.collect {
                received++
                gate.await()
            }
        }

        // Consumed inline, which parks the collector on the gate; the next four land in the buffer.
        assertTrue(hub.notifyDatasetChanged(), "first signal dropped")
        repeat(4) { index -> assertTrue(hub.notifyDatasetChanged(), "burst signal $index dropped") }

        gate.complete(Unit)
        yield()
        assertTrue(received >= 2, "the burst left no signal for the subscriber, got $received")
        collector.cancel()
    }
}

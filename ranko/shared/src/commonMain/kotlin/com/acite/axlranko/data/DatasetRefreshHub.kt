package com.acite.axlranko.data

import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

@Inject
@SingleIn(AppScope::class)
class DatasetRefreshHub() {
    /**
     * `tryEmit` drops a value once the buffer is full, and a burst of edits (a batch of tag
     * rewrites right before a drop) would then lose the signal entirely. DROP_OLDEST keeps
     * "the dataset changed" deliverable: every subscriber rescans the whole folder, so
     * coalescing several signals into one is harmless.
     */
    private val _events = MutableSharedFlow<Unit>(
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val events: SharedFlow<Unit> = _events.asSharedFlow()

    /** False only if the signal could not be handed over at all; see the buffer policy above. */
    fun notifyDatasetChanged(): Boolean = _events.tryEmit(Unit)
}

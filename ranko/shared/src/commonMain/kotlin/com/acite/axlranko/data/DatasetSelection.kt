package com.acite.axlranko.data

import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Which `[[environment.train_data]]` folder the single-folder pages work on: Images, Statistics
 * and the Utils > Environment tag card read the same index, so the folder picked on one of them is
 * the folder the others open.
 *
 * Only the index is held. Every screen reads the folder list from config.toml itself, so a row that
 * was typed but not saved cannot be selected here.
 */
@Inject
@SingleIn(AppScope::class)
class DatasetSelection {
    private val _index = MutableStateFlow(0)
    val index: StateFlow<Int> = _index.asStateFlow()

    fun select(index: Int) {
        if (index < 0) return
        _index.value = index
    }
}

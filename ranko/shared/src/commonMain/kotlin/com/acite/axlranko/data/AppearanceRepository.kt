package com.acite.axlranko.data

import com.acite.axlranko.model.AppearanceSettings
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

@Inject
@SingleIn(AppScope::class)
class AppearanceRepository {
    private val _settings = MutableStateFlow(AppearanceSettings.coerce(loadAppearanceSettings()))
    val settings: StateFlow<AppearanceSettings> = _settings.asStateFlow()

    fun update(transform: (AppearanceSettings) -> AppearanceSettings) {
        val next = AppearanceSettings.coerce(transform(_settings.value))
        _settings.value = next
        persistAppearanceSettings(next)
    }
}

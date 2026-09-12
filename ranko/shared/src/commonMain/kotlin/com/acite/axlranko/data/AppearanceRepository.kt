package com.acite.axlranko.data

import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.prefs.Preferences

@Inject
@SingleIn(AppScope::class)
class AppearanceRepository {
    private val prefs = Preferences.userRoot().node("com/acite/axlranko/appearance")

    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<AppearanceSettings> = _settings.asStateFlow()

    fun update(transform: (AppearanceSettings) -> AppearanceSettings) {
        val next = AppearanceSettings.coerce(transform(_settings.value))
        _settings.value = next
        persist(next)
    }

    private fun load(): AppearanceSettings {
        val bg = prefs.get(KEY_BG, AppearanceSettings().background.name)
            ?.let { runCatching { BackgroundStyle.valueOf(it) }.getOrNull() }
            ?: AppearanceSettings().background
        val cardBlur = prefs.getFloat(KEY_CARD_BLUR, AppearanceSettings().cardBlurRadiusDp)
        val bgBlur = prefs.getFloat(KEY_BG_BLUR, AppearanceSettings().backgroundBlurRadiusDp)
        val font = prefs.getFloat(KEY_FONT, AppearanceSettings().fontScale)
        val icon = prefs.getFloat(KEY_ICON, AppearanceSettings().iconScale)
        val image = prefs.get(KEY_IMAGE, AppearanceSettings().backgroundImagePath).orEmpty()
        return AppearanceSettings.coerce(
            AppearanceSettings(
                background = bg,
                cardBlurRadiusDp = cardBlur,
                backgroundBlurRadiusDp = bgBlur,
                fontScale = font,
                iconScale = icon,
                backgroundImagePath = image,
            )
        )
    }

    private fun persist(value: AppearanceSettings) {
        prefs.put(KEY_BG, value.background.name)
        prefs.putFloat(KEY_CARD_BLUR, value.cardBlurRadiusDp)
        prefs.putFloat(KEY_BG_BLUR, value.backgroundBlurRadiusDp)
        prefs.putFloat(KEY_FONT, value.fontScale)
        prefs.putFloat(KEY_ICON, value.iconScale)
        prefs.put(KEY_IMAGE, value.backgroundImagePath)
    }

    private companion object {
        const val KEY_BG = "background"
        const val KEY_CARD_BLUR = "blur_radius_dp"
        const val KEY_BG_BLUR = "background_blur_radius_dp"
        const val KEY_FONT = "font_scale"
        const val KEY_ICON = "icon_scale"
        const val KEY_IMAGE = "background_image_path"
    }
}
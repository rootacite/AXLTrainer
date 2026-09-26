package com.acite.axlranko.data

import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import kotlinx.browser.window

private const val PREFIX = "axlranko.appearance."
private const val KEY_BG = PREFIX + "background"
private const val KEY_CARD_BLUR = PREFIX + "blur_radius_dp"
private const val KEY_BG_BLUR = PREFIX + "background_blur_radius_dp"
private const val KEY_FONT = PREFIX + "font_scale"
private const val KEY_ICON = PREFIX + "icon_scale"
private const val KEY_IMAGE = PREFIX + "background_image_path"
private const val KEY_THUMB_QUALITY = PREFIX + "thumbnail_quality"

internal actual fun loadAppearanceSettings(): AppearanceSettings {
    val defaults = AppearanceSettings()
    val storage = window.localStorage
    val bg = storage.getItem(KEY_BG)
        ?.let { runCatching { BackgroundStyle.valueOf(it) }.getOrNull() }
        ?: defaults.background
    return AppearanceSettings(
        background = bg,
        cardBlurRadiusDp = storage.getItem(KEY_CARD_BLUR)?.toFloatOrNull() ?: defaults.cardBlurRadiusDp,
        backgroundBlurRadiusDp = storage.getItem(KEY_BG_BLUR)?.toFloatOrNull() ?: defaults.backgroundBlurRadiusDp,
        fontScale = storage.getItem(KEY_FONT)?.toFloatOrNull() ?: defaults.fontScale,
        iconScale = storage.getItem(KEY_ICON)?.toFloatOrNull() ?: defaults.iconScale,
        backgroundImagePath = storage.getItem(KEY_IMAGE).orEmpty(),
        thumbnailQuality = storage.getItem(KEY_THUMB_QUALITY)?.toIntOrNull() ?: defaults.thumbnailQuality,
    )
}

internal actual fun persistAppearanceSettings(value: AppearanceSettings) {
    val storage = window.localStorage
    storage.setItem(KEY_BG, value.background.name)
    storage.setItem(KEY_CARD_BLUR, value.cardBlurRadiusDp.toString())
    storage.setItem(KEY_BG_BLUR, value.backgroundBlurRadiusDp.toString())
    storage.setItem(KEY_FONT, value.fontScale.toString())
    storage.setItem(KEY_ICON, value.iconScale.toString())
    storage.setItem(KEY_THUMB_QUALITY, value.thumbnailQuality.toString())
    try {
        storage.setItem(KEY_IMAGE, value.backgroundImagePath)
    } catch (_: Exception) {
        storage.removeItem(KEY_IMAGE)
    }
}

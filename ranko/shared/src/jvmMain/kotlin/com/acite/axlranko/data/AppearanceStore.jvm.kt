package com.acite.axlranko.data

import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import java.util.prefs.Preferences

private const val KEY_BG = "background"
private const val KEY_CARD_BLUR = "blur_radius_dp"
private const val KEY_BG_BLUR = "background_blur_radius_dp"
private const val KEY_FONT = "font_scale"
private const val KEY_ICON = "icon_scale"
private const val KEY_IMAGE = "background_image_path"
private const val KEY_THUMB_QUALITY = "thumbnail_quality"

internal actual fun loadAppearanceSettings(): AppearanceSettings {
    val prefs = Preferences.userRoot().node("com/acite/axlranko/appearance")
    val defaults = AppearanceSettings()
    val bg = prefs.get(KEY_BG, defaults.background.name)
        ?.let { runCatching { BackgroundStyle.valueOf(it) }.getOrNull() }
        ?: defaults.background
    return AppearanceSettings(
        background = bg,
        cardBlurRadiusDp = prefs.getFloat(KEY_CARD_BLUR, defaults.cardBlurRadiusDp),
        backgroundBlurRadiusDp = prefs.getFloat(KEY_BG_BLUR, defaults.backgroundBlurRadiusDp),
        fontScale = prefs.getFloat(KEY_FONT, defaults.fontScale),
        iconScale = prefs.getFloat(KEY_ICON, defaults.iconScale),
        backgroundImagePath = prefs.get(KEY_IMAGE, defaults.backgroundImagePath).orEmpty(),
        thumbnailQuality = prefs.getInt(KEY_THUMB_QUALITY, defaults.thumbnailQuality),
    )
}

internal actual fun persistAppearanceSettings(value: AppearanceSettings) {
    val prefs = Preferences.userRoot().node("com/acite/axlranko/appearance")
    prefs.put(KEY_BG, value.background.name)
    prefs.putFloat(KEY_CARD_BLUR, value.cardBlurRadiusDp)
    prefs.putFloat(KEY_BG_BLUR, value.backgroundBlurRadiusDp)
    prefs.putFloat(KEY_FONT, value.fontScale)
    prefs.putFloat(KEY_ICON, value.iconScale)
    prefs.put(KEY_IMAGE, value.backgroundImagePath)
    prefs.putInt(KEY_THUMB_QUALITY, value.thumbnailQuality)
}

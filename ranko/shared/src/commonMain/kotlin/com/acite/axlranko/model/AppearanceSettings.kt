package com.acite.axlranko.model

enum class BackgroundStyle { Solid, Glow, Image }

data class AppearanceSettings(
    val background: BackgroundStyle = BackgroundStyle.Solid,
    val cardBlurRadiusDp: Float = 22f,
    val backgroundBlurRadiusDp: Float = 0f,
    val fontScale: Float = 1.0f,
    val iconScale: Float = 1.0f,
    val backgroundImagePath: String = "",
    val thumbnailQuality: Int = 80,
) {
    /**
     * Compose `Density.density`. Icon scale multiplies dp (icons, padding, component size).
     */
    fun scaledDensity(baseDensity: Float): Float = baseDensity * iconScale.coerceAtLeast(0.01f)

    /**
     * Compose `Density.fontScale`. Divide by icon scale so sp (text) does not also grow
     * when density is multiplied — otherwise both sliders appear to change the font.
     */
    fun scaledFontScale(baseFontScale: Float): Float {
        val icon = iconScale.coerceAtLeast(0.01f)
        return baseFontScale * fontScale / icon
    }

    companion object {
        const val MIN_BLUR = 0f
        const val MAX_BLUR = 30f
        const val MIN_FONT_SCALE = 0.50f
        const val MAX_FONT_SCALE = 2.50f
        const val MIN_ICON_SCALE = 0.50f
        const val MAX_ICON_SCALE = 2.50f
        const val MIN_THUMBNAIL_QUALITY = 1
        const val MAX_THUMBNAIL_QUALITY = 100

        fun coerce(value: AppearanceSettings): AppearanceSettings = value.copy(
            cardBlurRadiusDp = value.cardBlurRadiusDp.coerceIn(MIN_BLUR, MAX_BLUR),
            backgroundBlurRadiusDp = value.backgroundBlurRadiusDp.coerceIn(MIN_BLUR, MAX_BLUR),
            fontScale = value.fontScale.coerceIn(MIN_FONT_SCALE, MAX_FONT_SCALE),
            iconScale = value.iconScale.coerceIn(MIN_ICON_SCALE, MAX_ICON_SCALE),
            backgroundImagePath = value.backgroundImagePath.trim(),
            thumbnailQuality = value.thumbnailQuality.coerceIn(MIN_THUMBNAIL_QUALITY, MAX_THUMBNAIL_QUALITY),
        )
    }
}
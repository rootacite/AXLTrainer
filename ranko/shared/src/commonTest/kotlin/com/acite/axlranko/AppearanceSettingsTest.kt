package com.acite.axlranko

import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AppearanceSettingsTest {
    @Test
    fun defaultsAreWithinAllowedRanges() {
        val defaults = AppearanceSettings()
        assertEquals(BackgroundStyle.Solid, defaults.background)
        assertEquals("", defaults.backgroundImagePath)
        assertTrue(defaults.cardBlurRadiusDp in AppearanceSettings.MIN_BLUR..AppearanceSettings.MAX_BLUR)
        assertTrue(defaults.backgroundBlurRadiusDp in AppearanceSettings.MIN_BLUR..AppearanceSettings.MAX_BLUR)
        assertTrue(defaults.fontScale in AppearanceSettings.MIN_FONT_SCALE..AppearanceSettings.MAX_FONT_SCALE)
        assertTrue(defaults.iconScale in AppearanceSettings.MIN_ICON_SCALE..AppearanceSettings.MAX_ICON_SCALE)
    }

    @Test
    fun coerceClampsOutOfRangeValuesAndTrimsImagePath() {
        val clamped = AppearanceSettings.coerce(
            AppearanceSettings(
                background = BackgroundStyle.Image,
                cardBlurRadiusDp = 999f,
                backgroundBlurRadiusDp = -8f,
                fontScale = 0.1f,
                iconScale = 5f,
                backgroundImagePath = "  /tmp/wallpaper.png  ",
            )
        )
        assertEquals(BackgroundStyle.Image, clamped.background)
        assertEquals(AppearanceSettings.MAX_BLUR, clamped.cardBlurRadiusDp)
        assertEquals(AppearanceSettings.MIN_BLUR, clamped.backgroundBlurRadiusDp)
        assertEquals(AppearanceSettings.MIN_FONT_SCALE, clamped.fontScale)
        assertEquals(AppearanceSettings.MAX_ICON_SCALE, clamped.iconScale)
        assertEquals("/tmp/wallpaper.png", clamped.backgroundImagePath)
    }

    @Test
    fun fontPixelsStayIndependentOfIconScale() {
        val baseDensity = 2f
        val baseFont = 1.25f
        val sp = 16f
        val dp = 24f
        val a = AppearanceSettings(fontScale = 1.5f, iconScale = 1.0f)
        val b = AppearanceSettings(fontScale = 1.5f, iconScale = 2.0f)
        val aSp = sp * a.scaledDensity(baseDensity) * a.scaledFontScale(baseFont)
        val bSp = sp * b.scaledDensity(baseDensity) * b.scaledFontScale(baseFont)
        val aDp = dp * a.scaledDensity(baseDensity)
        val bDp = dp * b.scaledDensity(baseDensity)
        assertTrue(abs(aSp - bSp) < 1e-4f)
        assertTrue(abs(aSp - sp * baseDensity * baseFont * 1.5f) < 1e-4f)
        assertTrue(abs(bDp - aDp * 2f) < 1e-4f)
    }

    @Test
    fun fontScaleChangesTextWithoutChangingDp() {
        val baseDensity = 2f
        val small = AppearanceSettings(fontScale = 0.5f, iconScale = 1.2f)
        val large = AppearanceSettings(fontScale = 2.5f, iconScale = 1.2f)
        assertEquals(small.scaledDensity(baseDensity), large.scaledDensity(baseDensity))
        val smallSp = 16f * small.scaledDensity(baseDensity) * small.scaledFontScale(1f)
        val largeSp = 16f * large.scaledDensity(baseDensity) * large.scaledFontScale(1f)
        assertTrue(abs(largeSp - smallSp * 5f) < 1e-4f)
    }
}

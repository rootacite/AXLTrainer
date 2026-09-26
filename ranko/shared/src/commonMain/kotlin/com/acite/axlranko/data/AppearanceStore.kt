package com.acite.axlranko.data

import com.acite.axlranko.model.AppearanceSettings

internal expect fun loadAppearanceSettings(): AppearanceSettings

internal expect fun persistAppearanceSettings(value: AppearanceSettings)

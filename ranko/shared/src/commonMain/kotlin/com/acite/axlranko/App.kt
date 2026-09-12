package com.acite.axlranko

import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import com.acite.axlranko.data.AppearanceRepository
import com.acite.axlranko.ui.components.RankoBackdrop
import com.acite.axlranko.ui.theme.RankoTheme
import dev.zacsweers.metrox.viewmodel.LocalMetroViewModelFactory
import dev.zacsweers.metrox.viewmodel.MetroViewModelFactory

@Composable
fun App(
    metroVmf: MetroViewModelFactory,
    appearanceRepo: AppearanceRepository,
) {
    val settings by appearanceRepo.settings.collectAsState()
    val baseDensity = LocalDensity.current
    val scaledDensity = Density(
        density = settings.scaledDensity(baseDensity.density),
        fontScale = settings.scaledFontScale(baseDensity.fontScale),
    )
    RankoTheme {
        CompositionLocalProvider(LocalMetroViewModelFactory provides metroVmf) {
            RankoBackdrop(settings = settings) {
                CompositionLocalProvider(LocalDensity provides scaledDensity) {
                    Stage()
                }
            }
        }
    }
}
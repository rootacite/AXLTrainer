package com.acite.axlranko

import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import com.acite.axlranko.ui.components.RankoBackdrop
import com.acite.axlranko.ui.theme.RankoTheme
import dev.zacsweers.metrox.viewmodel.LocalMetroViewModelFactory
import dev.zacsweers.metrox.viewmodel.MetroViewModelFactory

@Composable
fun App(
    metroVmf: MetroViewModelFactory,
) {
    RankoTheme {
        CompositionLocalProvider(LocalMetroViewModelFactory provides metroVmf) {
            RankoBackdrop {
                Stage()
            }
        }
    }
}
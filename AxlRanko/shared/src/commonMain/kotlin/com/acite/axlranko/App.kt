package com.acite.axlranko

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import dev.zacsweers.metrox.viewmodel.LocalMetroViewModelFactory
import dev.zacsweers.metrox.viewmodel.MetroViewModelFactory

@Composable
fun App(
    metroVmf: MetroViewModelFactory,
) {
    MaterialTheme {
        CompositionLocalProvider(LocalMetroViewModelFactory provides metroVmf) {
            Stage()
        }
    }
}
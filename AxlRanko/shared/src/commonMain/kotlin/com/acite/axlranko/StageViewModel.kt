package com.acite.axlranko

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.ViewModel
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class StageViewModel : ViewModel()
{
    var navOffset by mutableStateOf(Offset(16f, 200f))
    var currentScreen by mutableStateOf(Screen.Images)
}
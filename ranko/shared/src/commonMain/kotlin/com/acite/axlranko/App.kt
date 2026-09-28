package com.acite.axlranko

import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import com.acite.axlranko.data.AppearanceRepository
import com.acite.axlranko.data.BlobStore
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.ui.components.RankoBackdrop
import com.acite.axlranko.ui.theme.RankoTheme
import com.acite.axlranko.util.AppWindow
import com.acite.axlranko.util.InstallPathPickerHost
import com.acite.axlranko.util.LocalAppWindow
import com.acite.axlranko.util.PathPicker
import dev.zacsweers.metrox.viewmodel.LocalMetroViewModelFactory
import dev.zacsweers.metrox.viewmodel.MetroViewModelFactory

@Composable
expect fun InstallBlobImageLoader(blobStore: BlobStore)

@Composable
fun App(
    metroVmf: MetroViewModelFactory,
    appearanceRepo: AppearanceRepository,
    blobStore: BlobStore,
    pathPicker: PathPicker,
    /** The OS window, where the build owns one; `null` leaves the Utils WM tab out. */
    appWindow: AppWindow? = null,
) {
    InstallBlobImageLoader(blobStore)
    InstallPathPickerHost(pathPicker)
    val settings by appearanceRepo.settings.collectAsState()
    val baseDensity = LocalDensity.current
    val scaledDensity = Density(
        density = settings.scaledDensity(baseDensity.density),
        fontScale = settings.scaledFontScale(baseDensity.fontScale),
    )
    RankoTheme {
        CompositionLocalProvider(
            LocalMetroViewModelFactory provides metroVmf,
            LocalThumbnailQuality provides settings.thumbnailQuality,
            LocalAppWindow provides appWindow,
        ) {
            RankoBackdrop(settings = settings) {
                CompositionLocalProvider(LocalDensity provides scaledDensity) {
                    Stage()
                }
            }
        }
    }
}
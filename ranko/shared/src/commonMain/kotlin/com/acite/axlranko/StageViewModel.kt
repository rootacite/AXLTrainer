package com.acite.axlranko

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.lifecycle.ViewModel
import com.acite.axlranko.util.NavDock
import com.acite.axlranko.util.clampNavOffset
import com.acite.axlranko.util.navPeekVisiblePx
import com.acite.axlranko.util.nearestNavDock
import com.acite.axlranko.util.peekNavOffset
import com.acite.axlranko.util.snapNavOffset
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
    var navExpanded by mutableStateOf(true)
    var navDock by mutableStateOf(NavDock.Left)
    var navDragging by mutableStateOf(false)
    var navPeeking by mutableStateOf(false)
    var currentScreen by mutableStateOf(Screen.Images)

    fun dragNavBy(delta: Offset, navSize: Size, bounds: Size) {
        navDragging = true
        navPeeking = false
        navOffset = clampNavOffset(navOffset + delta, navSize, bounds)
    }

    fun endNavDrag(navSize: Size, bounds: Size) {
        if (navSize.width > 0f && bounds.width > 0f && bounds.height > 0f) {
            navDock = nearestNavDock(navOffset, navSize, bounds)
            navOffset = snapNavOffset(navOffset, navSize, bounds, navDock)
        }
        navDragging = false
    }

    fun layoutNav(navSize: Size, bounds: Size, minPeekPx: Float) {
        if (navDragging) return
        if (navSize.width <= 0f || navSize.height <= 0f) return
        if (bounds.width <= 0f || bounds.height <= 0f) return
        val axis = when (navDock) {
            NavDock.Left, NavDock.Right -> navSize.width
            NavDock.Top, NavDock.Bottom -> navSize.height
        }
        val snapped = snapNavOffset(navOffset, navSize, bounds, navDock)
        navOffset = if (navPeeking) {
            peekNavOffset(snapped, navSize, navDock, navPeekVisiblePx(axis, minPeekPx))
        } else {
            snapped
        }
    }

    fun expandNav() {
        navExpanded = true
        navPeeking = false
    }

    fun collapseAndPeek() {
        navExpanded = false
        navPeeking = true
    }

    fun unpeek() {
        navPeeking = false
    }
}

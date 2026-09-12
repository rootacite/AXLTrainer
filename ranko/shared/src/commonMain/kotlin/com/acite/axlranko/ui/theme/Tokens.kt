package com.acite.axlranko.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

data class RankoTokens(
    val radiusCard: Dp = 22.dp,
    val radiusPanel: Dp = 20.dp,
    val radiusSheet: Dp = 24.dp,
) {
    val card = RoundedCornerShape(radiusCard)
    val panel = RoundedCornerShape(radiusPanel)
    val sheet = RoundedCornerShape(topStart = radiusSheet, topEnd = radiusSheet)
    val capsule = RoundedCornerShape(50)
}

val LocalRankoTokens = staticCompositionLocalOf { RankoTokens() }

val rankoTokens: RankoTokens
    @Composable get() = LocalRankoTokens.current

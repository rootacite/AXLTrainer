package com.acite.axlranko.ui.theme

import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

data class RankoPalette(
    val bgApp: Color,
    val bgPanel: Color,
    val bgCard: Color,
    val stroke: Color,
    val text: Color,
    val textDim: Color,
    val accentPink: Color,
    val accentBlue: Color,
    val accentLilac: Color,
    val boardBg: Color,
    val grid: Color,
    val star: Color,
    val qualityPurple: Color = Color(0xFFB44AC0),
    val qualityRed: Color = Color(0xFFE85D4C),
    val qualityOrange: Color = Color(0xFFF08A3A),
    val qualityYellow: Color = Color(0xFFF2C14E),
    val qualityMint: Color = Color(0xFF7BC67E),
    val qualityGreen: Color = Color(0xFF3DAA6D),
) {
    companion object {
        val SkySakura = RankoPalette(
            bgApp = Color(0xFF12101A),
            bgPanel = Color(0xFF1B1730),
            bgCard = Color(0xFF252042),
            stroke = Color(0xFF3A3460),
            text = Color(0xFFF4F0FF),
            textDim = Color(0xFFA89BC8),
            accentPink = Color(0xFFFF6BA8),
            accentBlue = Color(0xFF7AB8FF),
            accentLilac = Color(0xFFC9B6FF),
            boardBg = Color(0xFF2A2450),
            grid = Color(0xFF6E64A8),
            star = Color(0xFFFF8EC8),
        )
    }
}

val LocalRankoPalette = staticCompositionLocalOf { RankoPalette.SkySakura }

val rankoColors: RankoPalette
    @Composable get() = LocalRankoPalette.current

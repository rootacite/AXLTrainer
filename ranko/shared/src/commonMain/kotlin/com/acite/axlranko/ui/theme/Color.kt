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
        /**
         * Night palette whose primary is the logo amber (`axltrainer.png`, sampled #F8A818).
         * [accentPink] keeps its name: every screen already uses it as the brand accent.
         * [accentBlue] stays a cool secondary so a running phase does not share that amber.
         */
        val Amber = RankoPalette(
            bgApp = Color(0xFF16130F),
            bgPanel = Color(0xFF3E3830),
            bgCard = Color(0xFF514A40),
            stroke = Color(0xFF74685A),
            text = Color(0xFFFFF6E8),
            textDim = Color(0xFFC4B096),
            accentPink = Color(0xFFF8A818),
            accentBlue = Color(0xFF7EB0D4),
            accentLilac = Color(0xFFF0D39A),
            boardBg = Color(0xFF3A342C),
            grid = Color(0xFF8A7B68),
            star = Color(0xFFF8A818),
        )
    }
}

val LocalRankoPalette = staticCompositionLocalOf { RankoPalette.Amber }

/** Checkpoint sparkline: a rising Avg Loss slope, (255, 79, 0). */
val SparkSlopeHigh = Color(0xFFFF4F00)

/** Checkpoint sparkline: a falling Avg Loss slope, (0, 255, 127). */
val SparkSlopeLow = Color(0xFF00FF7F)

val rankoColors: RankoPalette
    @Composable get() = LocalRankoPalette.current

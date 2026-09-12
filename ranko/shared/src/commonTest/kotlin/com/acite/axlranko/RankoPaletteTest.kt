package com.acite.axlranko

import androidx.compose.ui.graphics.Color
import com.acite.axlranko.ui.theme.RankoPalette
import kotlin.test.Test
import kotlin.test.assertEquals

class RankoPaletteTest {
    @Test
    fun skySakuraKeepsKataHanaTokens() {
        val palette = RankoPalette.SkySakura
        assertEquals(Color(0xFF12101A), palette.bgApp)
        assertEquals(Color(0xFF1B1730), palette.bgPanel)
        assertEquals(Color(0xFF252042), palette.bgCard)
        assertEquals(Color(0xFF3A3460), palette.stroke)
        assertEquals(Color(0xFFF4F0FF), palette.text)
        assertEquals(Color(0xFFA89BC8), palette.textDim)
        assertEquals(Color(0xFFFF6BA8), palette.accentPink)
        assertEquals(Color(0xFF7AB8FF), palette.accentBlue)
        assertEquals(Color(0xFFC9B6FF), palette.accentLilac)
        assertEquals(Color(0xFF2A2450), palette.boardBg)
        assertEquals(Color(0xFF6E64A8), palette.grid)
        assertEquals(Color(0xFFFF8EC8), palette.star)
        assertEquals(Color(0xFFE85D4C), palette.qualityRed)
        assertEquals(Color(0xFFF08A3A), palette.qualityOrange)
        assertEquals(Color(0xFFF2C14E), palette.qualityYellow)
        assertEquals(Color(0xFF7BC67E), palette.qualityMint)
        assertEquals(Color(0xFF3DAA6D), palette.qualityGreen)
        assertEquals(Color(0xFFB44AC0), palette.qualityPurple)
    }
}

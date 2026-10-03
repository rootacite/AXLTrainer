package com.acite.axlranko

import androidx.compose.ui.graphics.Color
import com.acite.axlranko.ui.theme.RankoPalette
import kotlin.test.Test
import kotlin.test.assertEquals

class RankoPaletteTest {
    @Test
    fun amberMatchesTheLogo() {
        val palette = RankoPalette.Amber
        assertEquals(Color(0xFF16130F), palette.bgApp)
        assertEquals(Color(0xFF3E3830), palette.bgPanel)
        assertEquals(Color(0xFF514A40), palette.bgCard)
        assertEquals(Color(0xFF74685A), palette.stroke)
        assertEquals(Color(0xFFFFF6E8), palette.text)
        assertEquals(Color(0xFFC4B096), palette.textDim)
        assertEquals(Color(0xFFF8A818), palette.accentPink)
        assertEquals(Color(0xFF7EB0D4), palette.accentBlue)
        assertEquals(Color(0xFFF0D39A), palette.accentLilac)
        assertEquals(Color(0xFF3A342C), palette.boardBg)
        assertEquals(Color(0xFF8A7B68), palette.grid)
        assertEquals(Color(0xFFF8A818), palette.star)
        assertEquals(Color(0xFFE85D4C), palette.qualityRed)
        assertEquals(Color(0xFFF08A3A), palette.qualityOrange)
        assertEquals(Color(0xFFF2C14E), palette.qualityYellow)
        assertEquals(Color(0xFF7BC67E), palette.qualityMint)
        assertEquals(Color(0xFF3DAA6D), palette.qualityGreen)
        assertEquals(Color(0xFFB44AC0), palette.qualityPurple)
    }
}

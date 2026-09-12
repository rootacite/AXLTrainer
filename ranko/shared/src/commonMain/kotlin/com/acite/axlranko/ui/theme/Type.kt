package com.acite.axlranko.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import axlranko.shared.generated.resources.Res
import axlranko.shared.generated.resources.nunito_bold
import axlranko.shared.generated.resources.nunito_medium
import axlranko.shared.generated.resources.nunito_regular
import axlranko.shared.generated.resources.nunito_semibold
import org.jetbrains.compose.resources.Font

val LocalRankoFontFamily = staticCompositionLocalOf<FontFamily> { FontFamily.Default }

val rankoFontFamily: FontFamily
    @Composable get() = LocalRankoFontFamily.current

@Composable
fun rememberRankoFontFamily(): FontFamily {
    val regular = Font(Res.font.nunito_regular, FontWeight.Normal)
    val medium = Font(Res.font.nunito_medium, FontWeight.Medium)
    val semibold = Font(Res.font.nunito_semibold, FontWeight.SemiBold)
    val bold = Font(Res.font.nunito_bold, FontWeight.Bold)
    return remember(regular, medium, semibold, bold) {
        FontFamily(regular, medium, semibold, bold)
    }
}

fun rankoTypography(fontFamily: FontFamily): Typography {
    val text = RankoPalette.SkySakura.text
    val dim = RankoPalette.SkySakura.textDim
    return Typography(
        displayLarge = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.SemiBold,
            fontSize = 42.sp,
            lineHeight = 46.sp,
            letterSpacing = (-0.8).sp,
            color = text,
        ),
        headlineMedium = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.SemiBold,
            fontSize = 22.sp,
            lineHeight = 28.sp,
            color = text,
        ),
        titleLarge = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.SemiBold,
            fontSize = 18.sp,
            lineHeight = 24.sp,
            color = text,
        ),
        titleMedium = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.Medium,
            fontSize = 16.sp,
            lineHeight = 22.sp,
            color = text,
        ),
        titleSmall = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.SemiBold,
            fontSize = 14.sp,
            lineHeight = 20.sp,
            color = text,
        ),
        bodyLarge = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.Normal,
            fontSize = 16.sp,
            lineHeight = 22.sp,
            color = text,
        ),
        bodyMedium = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.Normal,
            fontSize = 14.sp,
            lineHeight = 20.sp,
            color = text,
        ),
        bodySmall = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.Normal,
            fontSize = 12.sp,
            lineHeight = 16.sp,
            color = text,
        ),
        labelLarge = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.Medium,
            fontSize = 14.sp,
            lineHeight = 18.sp,
            color = text,
        ),
        labelMedium = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.Medium,
            fontSize = 12.sp,
            lineHeight = 16.sp,
            color = text,
        ),
        labelSmall = TextStyle(
            fontFamily = fontFamily,
            fontWeight = FontWeight.Medium,
            fontSize = 11.sp,
            lineHeight = 14.sp,
            letterSpacing = 0.4.sp,
            color = dim,
        ),
    )
}

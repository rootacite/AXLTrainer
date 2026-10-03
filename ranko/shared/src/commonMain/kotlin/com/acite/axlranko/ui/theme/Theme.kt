package com.acite.axlranko.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.graphics.Color

@Composable
fun RankoTheme(content: @Composable () -> Unit) {
    val palette = RankoPalette.Amber
    val fonts = rememberRankoFontFamily()
    val typography = remember(fonts) { rankoTypography(fonts) }
    val scheme = remember(palette) {
        darkColorScheme(
            primary = palette.accentPink,
            onPrimary = palette.bgApp,
            primaryContainer = palette.bgCard,
            onPrimaryContainer = palette.text,
            secondary = palette.accentBlue,
            onSecondary = palette.bgApp,
            secondaryContainer = palette.bgPanel,
            onSecondaryContainer = palette.text,
            tertiary = palette.accentLilac,
            onTertiary = palette.bgApp,
            background = palette.bgApp,
            onBackground = palette.text,
            surface = palette.bgPanel,
            onSurface = palette.text,
            surfaceVariant = palette.bgCard,
            onSurfaceVariant = palette.textDim,
            surfaceTint = Color.Transparent,
            outline = palette.stroke,
            outlineVariant = palette.stroke,
            error = palette.qualityRed,
            onError = Color.White,
            errorContainer = palette.qualityRed.copy(alpha = 0.18f),
            onErrorContainer = palette.text,
            inverseSurface = palette.text,
            inverseOnSurface = palette.bgApp,
            inversePrimary = palette.accentPink,
        )
    }
    CompositionLocalProvider(
        LocalRankoTokens provides RankoTokens(),
        LocalRankoPalette provides palette,
        LocalRankoFontFamily provides fonts,
    ) {
        MaterialTheme(
            colorScheme = scheme,
            typography = typography,
            content = content,
        )
    }
}

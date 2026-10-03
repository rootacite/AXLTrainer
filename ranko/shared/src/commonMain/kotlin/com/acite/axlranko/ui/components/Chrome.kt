package com.acite.axlranko.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.material3.Text
import coil3.compose.AsyncImage
import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.chrisbanes.haze.HazeInput
import dev.chrisbanes.haze.HazeSourceSelection
import dev.chrisbanes.haze.HazeState
import dev.chrisbanes.haze.blur.HazeBlurStyle
import dev.chrisbanes.haze.blur.HazeColorEffect
import dev.chrisbanes.haze.blur.hazeBlur
import dev.chrisbanes.haze.hazeSource
import dev.chrisbanes.haze.rememberHazeState
import com.acite.axlranko.data.imageBackgroundUsesHaze
import com.acite.axlranko.localWallpaperModel

data class HazeContext(
    val state: HazeState?,
    val cardBlurRadiusDp: Float,
)

val LocalRankoHaze = staticCompositionLocalOf { HazeContext(state = null, cardBlurRadiusDp = 22f) }

@Composable
fun GlowOrbs(modifier: Modifier = Modifier) {
    val colors = rankoColors
    Canvas(modifier.fillMaxSize()) {
        fun orb(center: Offset, radius: Float, color: Color, core: Float, mid: Float) {
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        color.copy(alpha = core),
                        color.copy(alpha = mid),
                        Color.Transparent,
                    ),
                    center = center,
                    radius = radius,
                ),
                radius = radius,
                center = center,
            )
        }
        orb(
            center = Offset(80.dp.toPx(), 100.dp.toPx()),
            radius = 252.dp.toPx(),
            color = colors.accentPink,
            core = 0.55f,
            mid = 0.16f,
        )
        orb(
            center = Offset(size.width - 70.dp.toPx(), 170.dp.toPx()),
            radius = 240.dp.toPx(),
            color = colors.accentBlue,
            core = 0.48f,
            mid = 0.14f,
        )
        orb(
            center = Offset(110.dp.toPx(), size.height - 90.dp.toPx()),
            radius = 224.dp.toPx(),
            color = colors.accentLilac,
            core = 0.42f,
            mid = 0.12f,
        )
    }
}

@Composable
fun RankoBackdrop(
    settings: AppearanceSettings,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    val colors = rankoColors
    val imageFile = remember(settings.background, settings.backgroundImagePath) {
        if (settings.background != BackgroundStyle.Image) null
        else localWallpaperModel(settings.backgroundImagePath)
    }
    val showOrbs = settings.background == BackgroundStyle.Glow
    val showImage = imageFile != null
    val useHaze = showOrbs || (showImage && imageBackgroundUsesHaze)
    val hazeState = rememberHazeState()
    val hazeContext = HazeContext(
        state = if (useHaze) hazeState else null,
        cardBlurRadiusDp = settings.cardBlurRadiusDp,
    )
    CompositionLocalProvider(LocalRankoHaze provides hazeContext) {
        Box(
            modifier
                .fillMaxSize()
                .background(colors.bgApp),
        ) {
            if (showImage || showOrbs) {
                Box(
                    Modifier
                        .fillMaxSize()
                        .then(if (useHaze) Modifier.hazeSource(hazeState) else Modifier),
                ) {
                    if (imageFile != null) {
                        if (imageFile is ImageBitmap) {
                            Image(
                                bitmap = imageFile,
                                contentDescription = null,
                                contentScale = ContentScale.Crop,
                                filterQuality = FilterQuality.Low,
                                modifier = Modifier.fillMaxSize(),
                            )
                        } else {
                            AsyncImage(
                                model = imageFile,
                                contentDescription = null,
                                contentScale = ContentScale.Crop,
                                filterQuality = FilterQuality.Low,
                                modifier = Modifier.fillMaxSize(),
                            )
                        }
                        Box(Modifier.fillMaxSize().background(colors.bgApp.copy(alpha = 0.48f)))
                    }
                    if (showOrbs) {
                        GlowOrbs()
                    }
                }
                if (useHaze && settings.backgroundBlurRadiusDp > 0f) {
                    Box(
                        Modifier
                            .fillMaxSize()
                            .hazeBlur(
                                input = HazeInput.Sources(
                                    state = hazeState,
                                    selection = HazeSourceSelection.All,
                                ),
                                style = HazeBlurStyle {
                                    blurRadius(settings.backgroundBlurRadiusDp.dp)
                                    backgroundColor(Color.Transparent)
                                    noiseFactor(0.04f)
                                    fallbackColorEffect(HazeColorEffect.tint(Color.Transparent))
                                },
                            ),
                    )
                }
            }
            content()
        }
    }
}



@Composable
fun PorcelainCard(
    modifier: Modifier = Modifier,
    title: String? = null,
    /** Marks the card with the accent — the Dashboard's pinned checkpoint. */
    emphasized: Boolean = false,
    content: @Composable ColumnScope.() -> Unit,
) {
    val tokens = rankoTokens
    val colors = rankoColors
    val hazeContext = LocalRankoHaze.current
    val tint = colors.bgPanel.copy(alpha = 0.62f)
    val cardTint = colors.bgCard.copy(alpha = 0.42f)
    // Solid backdrop has nothing to blur, and a translucent dark tint on it reads as a dirty
    // black. The opaque card color is the fill in that case.
    // The emphasis is a wash over the frosted fill every other card gets, not a different fill: a
    // pinned card is still a card, and only the accent border and the wash mark it apart.
    val wash = if (emphasized) colors.accentPink.copy(alpha = 0.14f) else Color.Transparent
    val border = if (emphasized) {
        colors.accentPink.copy(alpha = 0.50f)
    } else {
        Color.White.copy(alpha = 0.10f)
    }
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(tokens.card)
            .rankoCardBlur(hazeContext, tint, cardTint, colors.bgCard)
            .background(wash)
            .border(if (emphasized) 2.dp else 1.dp, border, tokens.card)
            .padding(horizontal = 14.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        content = {
            if (title != null) {
                Text(title, color = colors.textDim, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
            }
            content()
        },
    )
}

@Composable
fun FrostedSurface(
    modifier: Modifier = Modifier,
    panelAlpha: Float = 0.58f,
    content: @Composable ColumnScope.() -> Unit,
) {
    val tokens = rankoTokens
    val colors = rankoColors
    val hazeContext = LocalRankoHaze.current
    val tint = colors.bgPanel.copy(alpha = panelAlpha)
    val cardTint = colors.bgCard.copy(alpha = 0.42f)
    Column(
        modifier
            .clip(tokens.card)
            .rankoCardBlur(hazeContext, tint, cardTint, colors.bgCard)
            .border(1.dp, Color.White.copy(alpha = 0.10f), tokens.card),
        content = content,
    )
}

private fun Modifier.rankoCardBlur(
    hazeContext: HazeContext,
    tint: Color,
    cardTint: Color,
    solid: Color,
): Modifier {
    val state = hazeContext.state
    val blurDp = hazeContext.cardBlurRadiusDp
    if (state == null || blurDp <= 0f) return background(solid)
    return hazeBlur(
        input = HazeInput.Sources(state),
        style = HazeBlurStyle {
            blurRadius(blurDp.dp)
            backgroundColor(tint)
            colorEffects(listOf(HazeColorEffect.tint(cardTint)))
            noiseFactor(0.05f)
            fallbackColorEffect(HazeColorEffect.tint(solid))
        },
    )
}
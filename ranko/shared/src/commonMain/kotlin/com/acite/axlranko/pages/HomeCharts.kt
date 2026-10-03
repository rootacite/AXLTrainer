package com.acite.axlranko.pages

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.TagStat
import com.acite.axlranko.pages.components.downsampleSpark
import com.acite.axlranko.pages.components.frequencyColor
import com.acite.axlranko.pages.components.slopeColor
import com.acite.axlranko.pages.components.smoothAvgLoss
import com.acite.axlranko.pages.components.sparkColorScale
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.TagTranslations
import kotlin.math.abs

/** The run's smoothed Avg Loss, full width, no ticks and no hit testing. */
@Composable
internal fun HomeAvgLossChart(points: List<MetricPoint>, modifier: Modifier = Modifier) {
    val colors = rankoColors
    val smoothed = remember(points) { downsampleSpark(smoothAvgLoss(points), maxPoints = 160) }
    if (smoothed.size < 2) {
        Text("No loss yet", color = colors.textDim, fontSize = 13.sp)
        return
    }
    val scale = remember(smoothed) { sparkColorScale(smoothed) }
    Canvas(modifier.fillMaxWidth().height(88.dp)) {
        val pad = 4.dp.toPx()
        drawRoundRect(
            // Transparency was 0.45 (alpha 0.55). Cut that by 60%, leaving 0.18, so alpha 0.82.
            color = colors.boardBg.copy(alpha = 0.82f),
            cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx()),
        )
        var minV = smoothed.first().value
        var maxV = minV
        for (point in smoothed) {
            if (point.value < minV) minV = point.value
            if (point.value > maxV) maxV = point.value
        }
        val span = (maxV - minV).let { if (abs(it) < 1e-6f) 1f else it }
        val left = pad
        val right = size.width - pad
        val top = pad
        val bottom = size.height - pad
        val plotW = (right - left).coerceAtLeast(1f)
        val plotH = (bottom - top).coerceAtLeast(1f)
        val x0 = smoothed.first().step
        val xSpan = (smoothed.last().step - x0).let { if (abs(it) < 1e-6f) 1f else it }
        val stroke = 2.dp.toPx()
        for (i in 0 until smoothed.lastIndex) {
            val a = smoothed[i]
            val b = smoothed[i + 1]
            val ax = left + (a.step - x0) / xSpan * plotW
            val bx = left + (b.step - x0) / xSpan * plotW
            val ay = bottom - (a.value - minV) / span * plotH
            val by = bottom - (b.value - minV) / span * plotH
            val slope = if (b.step == a.step) 0f else (b.value - a.value) / (b.step - a.step)
            drawLine(
                color = slopeColor(slope, scale),
                start = Offset(ax, ay),
                end = Offset(bx, by),
                strokeWidth = stroke,
                cap = StrokeCap.Round,
            )
        }
    }
}

/** Statistics' tag bars, shortened and not clickable. */
@Composable
internal fun HomeTagBars(tags: List<TagStat>, modifier: Modifier = Modifier) {
    val colors = rankoColors
    val tokens = rankoTokens
    if (tags.isEmpty()) {
        Text("No tags", color = colors.textDim, fontSize = 13.sp)
        return
    }
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(3.dp)) {
        tags.forEach { stat ->
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(22.dp)
                    .clip(tokens.panel),
            ) {
                Box(
                    Modifier
                        .fillMaxHeight()
                        .fillMaxWidth((stat.frequency / 100f).coerceIn(0.02f, 1f))
                        .background(frequencyColor(stat.frequency, colors).copy(alpha = 0.7f)),
                )
                Row(
                    Modifier.fillMaxSize().padding(horizontal = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        TagTranslations.display(stat.tag),
                        color = colors.text,
                        fontSize = 12.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        stat.count.toString(),
                        color = colors.textDim,
                        fontSize = 11.sp,
                        modifier = Modifier.padding(start = 8.dp),
                    )
                }
            }
        }
    }
}

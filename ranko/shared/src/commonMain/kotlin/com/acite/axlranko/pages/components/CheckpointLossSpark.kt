package com.acite.axlranko.pages.components

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.ui.theme.SparkSlopeHigh
import com.acite.axlranko.ui.theme.SparkSlopeLow
import com.acite.axlranko.ui.theme.rankoColors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/** EMA weight. The Dashboard smoothing slider does not apply to this strip. */
internal const val CHECKPOINT_SPARK_SMOOTHING = 0.85f

/** Drawn vertices. The marker's own step is kept on top of this cap. */
internal const val SPARK_MAX_POINTS = 240

/**
 * Fixed size of the card chart. The width stays put as the card grows. The height is the three
 * header lines at the default type scale: the 36dp button row, bodySmall, labelSmall, and the two
 * 8dp gaps between them. [CheckpointLossSpark] still grows with the text block when the font scale does.
 */
internal val CheckpointSparkWidth = 220.dp
internal val CheckpointSparkMinHeight = 82.dp

private const val SPARK_Y_PAD_FRACTION = 0.08f

internal data class SparkPoint(val step: Float, val value: Float)

internal data class SparkSegment(
    val x0: Float,
    val y0: Float,
    val x1: Float,
    val y1: Float,
    val color: Color,
)

internal data class SparkTick(val at: Float, val label: String)

internal data class SparkLayout(
    val segments: List<SparkSegment>,
    val markerX: Float?,
    val markerY: Float?,
    val markerColor: Color?,
    val plotLeft: Float = 0f,
    val plotTop: Float = 0f,
    val plotRight: Float = 0f,
    val plotBottom: Float = 0f,
    val xTicks: List<SparkTick> = emptyList(),
    val yTicks: List<SparkTick> = emptyList(),
)

/**
 * Same recurrence as the big chart's smoothed stroke: `ema = ema * smoothing + (1 - smoothing) * value`,
 * seeded with the first finite value. Non-finite values are dropped, then what remains is sorted by step.
 */
internal fun smoothAvgLoss(
    points: List<MetricPoint>,
    smoothing: Float = CHECKPOINT_SPARK_SMOOTHING,
): List<SparkPoint> {
    val finite = points.filter { it.value.isFinite() }.sortedBy { it.step }
    if (finite.isEmpty()) return emptyList()
    if (smoothing <= 0f) return finite.map { SparkPoint(it.step.toFloat(), it.value) }
    val out = ArrayList<SparkPoint>(finite.size)
    var ema = finite.first().value
    for (p in finite) {
        ema = ema * smoothing + (1f - smoothing) * p.value
        out += SparkPoint(p.step.toFloat(), ema)
    }
    return out
}

/**
 * Even stride down to [maxPoints], always keeping the first point, the last point, and the point
 * nearest [keepStep]. The kept step takes one slot, so the result is never longer than [maxPoints].
 */
internal fun downsampleSpark(
    points: List<SparkPoint>,
    maxPoints: Int = SPARK_MAX_POINTS,
    keepStep: Float? = null,
): List<SparkPoint> {
    if (maxPoints < 2 || points.size <= maxPoints) return points
    val last = points.lastIndex
    val keep = if (keepStep != null && keepStep.isFinite()) nearestIndex(points, keepStep) else null
    val keepInterior = keep != null && keep != 0 && keep != last
    val budget = if (keepInterior) maxPoints - 1 else maxPoints
    if (budget < 2) return listOf(points.first(), points.last())
    // sortedSetOf is a JVM TreeSet. Wasm has no such builder, so a set plus a sort.
    val indexes = mutableSetOf(0, last)
    val stride = last.toDouble() / (budget - 1)
    for (i in 1 until budget - 1) {
        indexes += (i * stride).roundToInt().coerceIn(1, last - 1)
    }
    if (keep != null) indexes += keep
    return indexes.sorted().map { points[it] }
}

/** One slope per consecutive pair, in value per step. A zero Δstep is a slope of 0. */
internal fun segmentSlopes(points: List<SparkPoint>): FloatArray {
    if (points.size < 2) return FloatArray(0)
    return FloatArray(points.size - 1) { i -> rise(points[i], points[i + 1]) }
}

/** Max |slope| of [slopes], or 0 when there is nothing to scale against. */
internal fun slopeScale(slopes: FloatArray): Float {
    var scale = 0f
    for (slope in slopes) {
        val magnitude = abs(slope)
        if (magnitude.isFinite() && magnitude > scale) scale = magnitude
    }
    return scale
}

/**
 * Slope of the smoothed series at [step]. A step that lands on a vertex uses the central difference
 * (one-sided at either end). A step between vertices uses the segment it sits on. Outside the series
 * the step is clamped to the nearer end first.
 */
internal fun slopeCovering(points: List<SparkPoint>, step: Float): Float {
    if (points.size < 2 || !step.isFinite()) return 0f
    val clamped = step.coerceIn(points.first().step, points.last().step)
    val nearest = nearestIndex(points, clamped)
    if (points[nearest].step == clamped) return vertexSlope(points, nearest)
    val right = points.indexOfFirst { it.step >= clamped }.coerceAtLeast(1)
    return rise(points[right - 1], points[right])
}

/**
 * [scale] is the max |slope| of the segments being colored. +scale is (255, 79, 0), −scale is
 * (0, 255, 127), and 0 is the sRGB midpoint. A scale of 0 (a flat series) is the midpoint.
 *
 * The scale has to be taken from those same segments. The steepest per-step move of the full series
 * is far larger than the slope of a segment that spans many steps, and dividing by it parks every
 * segment on the midpoint — (128, 167, 64), which reads as green whether the segment rises or falls.
 */
internal fun slopeColor(slope: Float, scale: Float): Color {
    if (!slope.isFinite() || !scale.isFinite() || scale <= 1e-12f) return srgbLerp(SparkSlopeLow, SparkSlopeHigh, 0.5f)
    val t = (((slope / scale) + 1f) / 2f).coerceIn(0f, 1f)
    return srgbLerp(SparkSlopeLow, SparkSlopeHigh, t)
}

/** Max |segment slope| of the smoothed polyline that will actually be drawn. */
internal fun sparkColorScale(drawn: List<SparkPoint>): Float = slopeScale(segmentSlopes(drawn))

/**
 * Step window for one checkpoint: at most `± 2 × saveEveryNSteps` around [step], and never the
 * whole run. `0` or a missing cadence is no window — the repo file is not a substitute.
 */
internal fun windowSpark(
    points: List<SparkPoint>,
    step: Float,
    saveEveryNSteps: Int,
): List<SparkPoint> {
    if (saveEveryNSteps <= 0 || !step.isFinite() || points.isEmpty()) return emptyList()
    val half = saveEveryNSteps * 2f
    val lo = step - half
    val hi = step + half
    return points.filter { it.step >= lo && it.step <= hi }
}

/** Step axis. Whole numbers, since a checkpoint step is an integer. */
internal fun sparkStepLabel(step: Float): String =
    if (!step.isFinite()) "" else step.roundToInt().toString()

/** Loss axis. Three decimals below 1, fewer as the value grows. */
internal fun sparkAxisLabel(value: Float): String {
    if (!value.isFinite()) return ""
    val magnitude = abs(value)
    if (magnitude == 0f) return "0"
    val negative = value < 0f
    val decimals = when {
        magnitude >= 100f -> 0
        magnitude >= 10f -> 1
        magnitude >= 1f -> 2
        else -> 3
    }
    var scale = 1
    repeat(decimals) { scale *= 10 }
    val rounded = (magnitude * scale).roundToInt()
    val body = if (decimals == 0) {
        rounded.toString()
    } else {
        val ip = rounded / scale
        val fp = (rounded % scale).toString().padStart(decimals, '0')
        "$ip.$fp"
    }
    if (body.all { it == '0' || it == '.' }) return "0"
    return if (negative) "-$body" else body
}

/**
 * Screen geometry of one card's chart. [points] is the smoothed series already cut to this card's
 * step window (possibly downsampled). [scale] is [sparkColorScale] of those same points.
 * [markerSlope] is the central difference of that polyline at the card's step; when it is omitted
 * the drawn series supplies it. A step outside the points clamps the marker to the nearer edge.
 */
internal fun layoutCheckpointSpark(
    points: List<SparkPoint>,
    step: Float?,
    width: Float,
    height: Float,
    padLeft: Float,
    padTop: Float,
    padRight: Float,
    padBottom: Float,
    scale: Float,
    markerSlope: Float? = null,
): SparkLayout {
    if (points.size < 2 || width <= 0f || height <= 0f) {
        return SparkLayout(emptyList(), null, null, null)
    }
    val plotLeft = padLeft.coerceAtLeast(0f)
    val plotTop = padTop.coerceAtLeast(0f)
    val plotRight = (width - padRight).coerceAtLeast(plotLeft + 1f)
    val plotBottom = (height - padBottom).coerceAtLeast(plotTop + 1f)
    val plotW = plotRight - plotLeft
    val plotH = plotBottom - plotTop

    val xMin = points.first().step
    val xMax = points.last().step
    var yLo = points[0].value
    var yHi = points[0].value
    for (p in points) {
        if (p.value < yLo) yLo = p.value
        if (p.value > yHi) yHi = p.value
    }
    val span = yHi - yLo
    val yPad = if (span > 1e-12f) span * SPARK_Y_PAD_FRACTION else max(abs(yHi) * SPARK_Y_PAD_FRACTION, 1e-6f)
    val yMin = yLo - yPad
    val ySpan = (yHi + yPad - yMin).coerceAtLeast(1e-6f)

    fun xOf(s: Float): Float {
        val denom = xMax - xMin
        val fraction = if (denom <= 1e-6f) 0.5f else ((s - xMin) / denom).coerceIn(0f, 1f)
        return plotLeft + fraction * plotW
    }

    fun yOf(v: Float): Float {
        val fraction = ((v - yMin) / ySpan).coerceIn(0f, 1f)
        return plotBottom - fraction * plotH
    }

    val slopes = segmentSlopes(points)
    val segments = ArrayList<SparkSegment>(slopes.size)
    for (i in slopes.indices) {
        val a = points[i]
        val b = points[i + 1]
        segments += SparkSegment(xOf(a.step), yOf(a.value), xOf(b.step), yOf(b.value), slopeColor(slopes[i], scale))
    }

    val xSpan = xMax - xMin
    val xTicks = listOf(0f, 0.5f, 1f).map { fraction ->
        val stepValue = if (xSpan <= 1e-6f) xMin else xMin + fraction * xSpan
        SparkTick(plotLeft + fraction * plotW, sparkStepLabel(stepValue))
    }
    val yTicks = listOf(0f, 0.5f, 1f).map { fraction ->
        val value = yMin + (1f - fraction) * ySpan
        SparkTick(plotTop + fraction * plotH, sparkAxisLabel(value))
    }
    val frame = SparkLayout(
        segments = segments,
        markerX = null,
        markerY = null,
        markerColor = null,
        plotLeft = plotLeft,
        plotTop = plotTop,
        plotRight = plotRight,
        plotBottom = plotBottom,
        xTicks = xTicks,
        yTicks = yTicks,
    )
    if (step == null || !step.isFinite()) return frame
    val clamped = step.coerceIn(xMin, xMax)
    val slope = markerSlope ?: slopeCovering(points, clamped)
    return frame.copy(
        markerX = xOf(clamped),
        markerY = yOf(valueAt(points, clamped)),
        markerColor = slopeColor(slope, scale),
    )
}

/** Linear value of the series at [step], clamped to the end vertices. */
internal fun valueAt(points: List<SparkPoint>, step: Float): Float {
    if (points.isEmpty()) return 0f
    if (points.size == 1 || step <= points.first().step) return points.first().value
    if (step >= points.last().step) return points.last().value
    val right = points.indexOfFirst { it.step >= step }
    if (right <= 0) return points.first().value
    val a = points[right - 1]
    val b = points[right]
    val dx = b.step - a.step
    if (dx == 0f) return b.value
    val t = ((step - a.step) / dx).coerceIn(0f, 1f)
    return a.value + t * (b.value - a.value)
}

/**
 * The run's smoothed Avg Loss, immediately left of a checkpoint card's Save As button. No hit
 * testing and no tick numbers. [points] is the whole run's 0.85 EMA; the drawn x range is at most
 * `± 2 × [saveEveryNSteps]` around [step].
 */
@Composable
internal fun CheckpointLossSpark(
    points: List<SparkPoint>,
    step: Float,
    saveEveryNSteps: Int,
    modifier: Modifier = Modifier,
) {
    val windowed = remember(points, step, saveEveryNSteps) { windowSpark(points, step, saveEveryNSteps) }
    if (windowed.size < 2) return
    val drawn = remember(windowed, step) { downsampleSpark(windowed, SPARK_MAX_POINTS, step) }
    val scale = remember(drawn) { sparkColorScale(drawn) }
    val markerSlope = remember(drawn, step) { slopeCovering(drawn, step) }
    val colors = rankoColors
    val markerLine = colors.text
    Canvas(modifier) {
        val pad = 4.dp.toPx()
        val corner = CornerRadius(6.dp.toPx(), 6.dp.toPx())
        drawRoundRect(color = colors.boardBg.copy(alpha = 0.35f), cornerRadius = corner)
        val layout = layoutCheckpointSpark(
            points = drawn,
            step = step,
            width = size.width,
            height = size.height,
            padLeft = pad,
            padTop = pad,
            padRight = pad,
            padBottom = pad,
            scale = scale,
            markerSlope = markerSlope,
        )
        val grid = colors.grid.copy(alpha = 0.45f)
        for (tick in layout.xTicks) {
            drawLine(grid, Offset(tick.at, layout.plotTop), Offset(tick.at, layout.plotBottom), strokeWidth = 1f)
        }
        for (tick in layout.yTicks) {
            drawLine(grid, Offset(layout.plotLeft, tick.at), Offset(layout.plotRight, tick.at), strokeWidth = 1f)
        }
        drawLine(colors.stroke, Offset(layout.plotLeft, layout.plotTop), Offset(layout.plotLeft, layout.plotBottom), strokeWidth = 1f)
        drawLine(colors.stroke, Offset(layout.plotLeft, layout.plotBottom), Offset(layout.plotRight, layout.plotBottom), strokeWidth = 1f)
        val markerX = layout.markerX
        val stroke = 2.dp.toPx()
        clipRect(layout.plotLeft, layout.plotTop, layout.plotRight, layout.plotBottom) {
            if (markerX != null) {
                drawLine(
                    color = markerLine,
                    start = Offset(markerX, layout.plotTop),
                    end = Offset(markerX, layout.plotBottom),
                    strokeWidth = 1.dp.toPx(),
                )
            }
            for (segment in layout.segments) {
                drawLine(
                    color = segment.color,
                    start = Offset(segment.x0, segment.y0),
                    end = Offset(segment.x1, segment.y1),
                    strokeWidth = stroke,
                    cap = StrokeCap.Round,
                )
            }
            val markerY = layout.markerY
            val markerColor = layout.markerColor
            if (markerX != null && markerY != null && markerColor != null) {
                val radius = 1.5.dp.toPx()
                drawCircle(Color.White, radius = radius + 1.dp.toPx(), center = Offset(markerX, markerY))
                drawCircle(markerColor, radius = radius, center = Offset(markerX, markerY))
            }
        }
    }
}

private fun nearestIndex(points: List<SparkPoint>, step: Float): Int {
    var best = 0
    var bestDist = abs(points[0].step - step)
    for (i in 1..points.lastIndex) {
        val dist = abs(points[i].step - step)
        if (dist < bestDist) {
            best = i
            bestDist = dist
        }
    }
    return best
}

private fun vertexSlope(points: List<SparkPoint>, index: Int): Float {
    val prev = points.getOrNull(index - 1)
    val next = points.getOrNull(index + 1)
    if (prev != null && next != null) return rise(prev, next)
    if (prev != null) return rise(prev, points[index])
    if (next != null) return rise(points[index], next)
    return 0f
}

private fun rise(a: SparkPoint, b: SparkPoint): Float {
    val dx = b.step - a.step
    if (dx == 0f || !dx.isFinite()) return 0f
    val dy = b.value - a.value
    if (!dy.isFinite()) return 0f
    return dy / dx
}

/** Linear blend in the 0..255 channels the two endpoint colors were named in. */
private fun srgbLerp(from: Color, to: Color, t: Float): Color {
    fun channel(a: Float, b: Float): Int = ((a + (b - a) * t) * 255f).roundToInt().coerceIn(0, 255)
    return Color(channel(from.red, to.red), channel(from.green, to.green), channel(from.blue, to.blue))
}

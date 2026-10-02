package com.acite.axlranko.pages.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.PointerType
import androidx.compose.ui.input.pointer.isCtrlPressed
import androidx.compose.ui.input.pointer.isPrimaryPressed
import androidx.compose.ui.input.pointer.isShiftPressed
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.boundsInRoot
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.theme.rankoColors
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

private data class ChartPoint(val step: Float, val value: Float)

private data class Viewport(
    val xMin: Float,
    val xMax: Float,
    val yMin: Float,
    val yMax: Float,
) {
    val xRange get() = (xMax - xMin).coerceAtLeast(1e-6f)
    val yRange get() = (yMax - yMin).coerceAtLeast(1e-6f)

    fun clampToBounds(bounds: Viewport): Viewport {
        val w = xRange.coerceAtMost(bounds.xRange)
        val h = yRange.coerceAtMost(bounds.yRange)
        val x0 = if (w >= bounds.xRange) bounds.xMin else xMin.coerceIn(bounds.xMin, bounds.xMax - w)
        val y0 = if (h >= bounds.yRange) bounds.yMin else yMin.coerceIn(bounds.yMin, bounds.yMax - h)
        return copy(xMin = x0, xMax = x0 + w, yMin = y0, yMax = y0 + h)
    }
}

// Compose Desktop reports ~1.0 per mouse-wheel notch (preciseWheelRotation), not pixels.
private const val WHEEL_ZOOM_STEP = 1.15f
private const val WHEEL_ZOOM_INTENSITY = 2.5f
private const val JUMP_REJECT_FRACTION = 0.5f

/** A chart whose x axis counts steps opens on its newest [DEFAULT_STEP_SPAN] steps. */
internal const val DEFAULT_STEP_SPAN = 1200f

/** Room the right-hand labels of a dual-axis chart need, mirroring the left padding. */
internal const val PLOT_RIGHT_PADDING = 52f

/**
 * The x window a chart opens on: the whole range, or its newest [maxStepSpan] steps. A null span —
 * every chart whose x axis is not a step count — keeps the full range.
 */
internal fun initialXWindow(xMin: Float, xMax: Float, maxStepSpan: Float?): Pair<Float, Float> {
    val full = xMax - xMin
    if (!full.isFinite() || full <= 1e-9f) return xMin to xMax
    val span = maxStepSpan?.takeIf { it > 0f && it.isFinite() }?.coerceAtMost(full) ?: full
    return (xMax - span) to xMax
}

/**
 * Range of one series inside an x window, padded so a flat curve (a learning rate after its warmup)
 * gets a readable span instead of collapsing onto a single line.
 */
internal fun windowDomain(
    points: List<MetricPoint>,
    xMin: Float,
    xMax: Float,
    padFraction: Float = 0.05f,
): Pair<Float, Float>? {
    val inWindow = points.filter { it.step >= xMin && it.step <= xMax }
    val considered = if (inWindow.isEmpty()) points else inWindow
    if (considered.isEmpty()) return null
    val lo = considered.minOf { it.value }
    val hi = considered.maxOf { it.value }
    val span = hi - lo
    val pad = if (span > 1e-12f) span * padFraction else maxOf(abs(hi) * padFraction, 1e-12f)
    return (lo - pad) to (hi + pad)
}

/**
 * Value a normalized y tick (the 0..100 the viewport spans) stands for on one series' own axis, so a
 * dual-axis chart can label its left and right edges in the units each curve is drawn in.
 */
internal fun axisTickValue(
    viewportMin: Float,
    viewportSpan: Float,
    fraction: Float,
    domainMin: Float,
    domainMax: Float,
): Float = domainMin + ((viewportMin + fraction * viewportSpan) / 100f) * (domainMax - domainMin)

private fun zoomRange(current: Float, factor: Float, minRange: Float, fullRange: Float): Float {
    if (fullRange <= 1e-9f) return 0f
    val lo = minOf(minRange, fullRange)
    return (current * factor).coerceIn(lo, fullRange)
}

private fun List<Float>.percentile(p: Float): Float {
    if (isEmpty()) return 0f
    val idx = (p * (size - 1)).coerceIn(0f, (size - 1).toFloat())
    val lo = idx.toInt()
    val hi = min(lo + 1, size - 1)
    return this[lo] + (idx - lo) * (this[hi] - this[lo])
}

private fun formatAxisValue(v: Float): String {
    val a = abs(v)
    return when {
        a == 0f -> "0"
        a >= 1_000_000f -> formatScientific(v, 1)
        a < 0.01f && a > 0f -> formatScientific(v, 1)
        a >= 1_000f -> formatFixed(v, 0)
        a >= 1f -> formatFixed(v, 2)
        else -> formatFixed(v, 3)
    }
}

private fun formatFixed(v: Float, decimals: Int): String {
    if (decimals == 0) return v.roundToInt().toString()
    val scale = 10.0.pow(decimals).toFloat()
    val rounded = (v * scale).roundToInt()
    val intPart = rounded / scale.toInt()
    val fracPart = abs(rounded) % scale.toInt()
    val fracStr = fracPart.toString().padStart(decimals, '0')
    return if (v < 0 && intPart == 0) "-0.$fracStr" else "$intPart.$fracStr"
}

private fun formatScientific(v: Float, mantissaDecimals: Int): String {
    if (v == 0f) return "0"
    val sign = if (v < 0) "-" else ""
    val absV = abs(v)
    val exp = floor(ln(absV) / ln(10.0)).toInt()
    val mant = absV / 10.0.pow(exp).toFloat()
    val expSign = if (exp >= 0) "+" else "-"
    return "${sign}${formatFixed(mant, mantissaDecimals)}e${expSign}${abs(exp)}"
}

private fun lttbDownsample(points: List<ChartPoint>, maxPoints: Int): List<ChartPoint> {
    val n = points.size
    if (maxPoints !in 3..<n) return points

    val sampled = ArrayList<ChartPoint>(maxPoints)
    sampled.add(points.first())

    val bucketCount = maxPoints - 2
    val bucketSize = (n - 2).toDouble() / bucketCount
    var prevSelected = 0

    for (b in 0 until bucketCount) {
        val nextStart = ((b + 1) * bucketSize + 1).toInt().coerceAtMost(n - 1)
        val nextEnd = ((b + 2) * bucketSize + 1).toInt().coerceAtMost(n)
        var avgX = 0.0
        var avgY = 0.0
        val nextLen = (nextEnd - nextStart).coerceAtLeast(1)
        for (i in nextStart until nextEnd) {
            avgX += points[i].step
            avgY += points[i].value
        }
        avgX /= nextLen
        avgY /= nextLen

        val curStart = (b * bucketSize + 1).toInt().coerceAtMost(n - 1)
        val curEnd = ((b + 1) * bucketSize + 1).toInt().coerceAtMost(n - 1)
        val ax = points[prevSelected].step.toDouble()
        val ay = points[prevSelected].value.toDouble()

        var maxArea = -1.0
        var maxIndex = curStart
        for (i in curStart until curEnd) {
            val area = abs(
                (ax - avgX) * (points[i].value - ay) -
                    (ax - points[i].step) * (avgY - ay)
            )
            if (area > maxArea) {
                maxArea = area
                maxIndex = i
            }
        }

        sampled.add(points[maxIndex])
        prevSelected = maxIndex
    }

    sampled.add(points.last())
    return sampled
}

data class ChartSeries(
    val label: String,
    val points: List<MetricPoint>,
    val color: Color,
    val domainMin: Float? = null,
    val domainMax: Float? = null,
)

/**
 * What a Ctrl+click resolved to: the step under the pointer and the checkpoint step the pick matched,
 * both drawn on the chart so the snapping is visible.
 */
data class ChartPickMarkers(
    val clickedStep: Float,
    val matchedStep: Int?,
)

@Composable
fun ChartCard(
    title: String,
    points: List<MetricPoint>,
    color: Color,
    smoothing: Float,
    modifier: Modifier = Modifier,
    outlierClip: Float = 0.15f,
    strokeWidth: Float = 3f,
    chartHeight: Dp = 220.dp,
    /** Steps the x axis opens on (null = the whole range); see [initialXWindow]. */
    defaultStepSpan: Float? = null,
    /**
     * Receives the step under the picking click and where that click landed (window-root pixels).
     * Fired by `Ctrl`+left click and by a left double click; null keeps the chart a pure display.
     */
    onPickStep: ((step: Float, anchorInRoot: Offset) -> Unit)? = null,
    showHoverStep: Boolean = false,
    pickMarkers: ChartPickMarkers? = null,
) {
    MultiSeriesChartCard(
        title = title,
        series = listOf(ChartSeries(label = title, points = points, color = color)),
        smoothing = smoothing,
        modifier = modifier,
        outlierClip = outlierClip,
        strokeWidth = strokeWidth,
        chartHeight = chartHeight,
        showLegend = false,
        defaultStepSpan = defaultStepSpan,
        onPickStep = onPickStep,
        showHoverStep = showHoverStep,
        pickMarkers = pickMarkers,
    )
}

@Composable
fun MultiSeriesChartCard(
    title: String,
    series: List<ChartSeries>,
    smoothing: Float,
    modifier: Modifier = Modifier,
    outlierClip: Float = 0.15f,
    strokeWidth: Float = 3f,
    chartHeight: Dp = 220.dp,
    showLegend: Boolean = true,
    /** Steps the x axis opens on (null = the whole range); see [initialXWindow]. */
    defaultStepSpan: Float? = null,
    /**
     * Draw the first two series against their own vertical axes: the left edge is labelled in
     * series 0's units, the right edge in series 1's, each in that curve's colour.
     */
    dualAxis: Boolean = false,
    onPickStep: ((step: Float, anchorInRoot: Offset) -> Unit)? = null,
    showHoverStep: Boolean = false,
    pickMarkers: ChartPickMarkers? = null,
) {
    val hasData = series.any { it.points.isNotEmpty() }
    val colors = rankoColors
    PorcelainCard(modifier = modifier.height(chartHeight)) {
        Column(modifier = Modifier.fillMaxSize()) {
            if (showLegend) {
                Text(
                    title,
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                )
                Spacer(modifier = Modifier.height(6.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    series.forEach { item ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(item.color)
                            )
                            Text(
                                item.label,
                                style = MaterialTheme.typography.labelSmall,
                                color = colors.textDim,
                            )
                        }
                    }
                }
            } else {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    series.firstOrNull()?.let { item ->
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(RoundedCornerShape(5.dp))
                                .background(item.color)
                        )
                    }
                    Text(
                        title,
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            if (hasData) {
                InteractiveLineChart(
                    series = series,
                    smoothing = smoothing,
                    outlierClip = outlierClip,
                    strokeWidth = strokeWidth,
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    defaultStepSpan = defaultStepSpan,
                    dualAxis = dualAxis,
                    onPickStep = onPickStep,
                    showHoverStep = showHoverStep,
                    pickMarkers = pickMarkers,
                )
            } else {
                Box(
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        "No Data",
                        color = colors.textDim,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

private const val MAX_DRAW_POINTS = 500

private fun mapSeriesY(value: Float, series: ChartSeries): Float {
    val lo = series.domainMin
    val hi = series.domainMax
    if (lo == null || hi == null) return value
    val span = hi - lo
    if (span <= 1e-9f) return 0f
    return ((value - lo) / span) * 100f
}

private fun smoothPoints(points: List<ChartPoint>, smoothing: Float): List<ChartPoint> {
    if (smoothing <= 0f || points.isEmpty()) return points
    val out = ArrayList<ChartPoint>(points.size)
    var ema = points.first().value
    for (p in points) {
        ema = ema * smoothing + (1f - smoothing) * p.value
        out += ChartPoint(p.step, ema)
    }
    return out
}

private data class PreparedSeries(
    val color: Color,
    val raw: List<ChartPoint>,
    val smooth: List<ChartPoint>,
)

@Composable
private fun InteractiveLineChart(
    series: List<ChartSeries>,
    smoothing: Float,
    outlierClip: Float,
    strokeWidth: Float,
    modifier: Modifier = Modifier,
    defaultStepSpan: Float? = null,
    dualAxis: Boolean = false,
    onPickStep: ((step: Float, anchorInRoot: Offset) -> Unit)? = null,
    showHoverStep: Boolean = false,
    pickMarkers: ChartPickMarkers? = null,
) {
    val rawSeries = remember(series) {
        series.mapNotNull { item ->
            if (item.points.isEmpty()) return@mapNotNull null
            item to item.points.map { ChartPoint(it.step.toFloat(), it.value) }.sortedBy { it.step }
        }
    }
    if (rawSeries.isEmpty()) return

    val fullX = remember(rawSeries) {
        val all = rawSeries.flatMap { it.second }
        all.minOf { it.step } to all.maxOf { it.step }
    }
    // The window the chart opens on — the newest `defaultStepSpan` steps, or the whole range.
    val windowX = remember(fullX, defaultStepSpan) {
        initialXWindow(fullX.first, fullX.second, defaultStepSpan)
    }

    // A dual-axis chart draws each of its first two curves through that curve's own range *inside
    // the opening window*, so the two can be read against a left and a right scale.
    val chartSeries = remember(rawSeries, dualAxis, windowX) {
        rawSeries.mapIndexed { index, (item, _) ->
            if (!dualAxis || index > 1 || (item.domainMin != null && item.domainMax != null)) {
                item
            } else {
                val domain = windowDomain(item.points, windowX.first, windowX.second)
                if (domain == null) item else item.copy(domainMin = domain.first, domainMax = domain.second)
            }
        }
    }
    val axisDomains = remember(chartSeries) {
        chartSeries.mapNotNull { item ->
            val lo = item.domainMin
            val hi = item.domainMax
            if (lo != null && hi != null) lo to hi else null
        }
    }
    val rightDomain = if (dualAxis && axisDomains.size >= 2) axisDomains[1] else null
    // A dual-axis chart labels its left edge in series 0's units even when the second series has
    // nothing to draw (a run from before the other curve was logged).
    val leftDomain = axisDomains.firstOrNull().takeIf { dualAxis }
    val rightPad = if (rightDomain != null) PLOT_RIGHT_PADDING else 0f
    val leftAxisColor = chartSeries.firstOrNull()?.color ?: Color.Unspecified
    val rightAxisColor = chartSeries.getOrNull(1)?.color ?: Color.Unspecified

    val prepared = remember(rawSeries, chartSeries, smoothing) {
        rawSeries.mapIndexed { index, (_, points) ->
            val item = chartSeries[index]
            val raw = points.map { ChartPoint(it.step, mapSeriesY(it.value, item)) }
            PreparedSeries(color = item.color, raw = raw, smooth = smoothPoints(raw, smoothing))
        }
    }

    val allMapped = remember(prepared) { prepared.flatMap { it.raw } }
    val normalized = axisDomains.isNotEmpty()

    val fullBounds = remember(fullX, allMapped, normalized) {
        Viewport(
            xMin = fullX.first,
            xMax = fullX.second,
            yMin = if (normalized) 0f else allMapped.minOf { it.value },
            yMax = if (normalized) 100f else allMapped.maxOf { it.value },
        )
    }

    // The y range is fitted to the points the window actually shows, so an early spike the window
    // has scrolled past cannot flatten the curve that is on screen.
    val initialViewport = remember(allMapped, windowX, outlierClip, normalized) {
        if (normalized) {
            Viewport(xMin = windowX.first, xMax = windowX.second, yMin = 0f, yMax = 100f)
        } else {
            val inWindow = allMapped.filter { it.step >= windowX.first && it.step <= windowX.second }
            val sortedY = (if (inWindow.isEmpty()) allMapped else inWindow).map { it.value }.sorted()
            val half = (outlierClip / 2f).coerceIn(0f, 0.49f)
            val yLo = sortedY.percentile(half)
            val yHi = sortedY.percentile(1f - half)
            val yPad = ((yHi - yLo) * 0.05f).coerceAtLeast(abs(yHi) * 0.01f)
            Viewport(
                xMin = windowX.first,
                xMax = windowX.second,
                yMin = yLo - yPad,
                yMax = yHi + yPad,
            )
        }
    }

    // The pick handler outlives recompositions, so it reads the viewport through the state
    // object instead of capturing the value it was composed with.
    val viewportState = remember(initialViewport) { mutableStateOf(initialViewport) }
    var viewport by viewportState
    val canvasOrigin = remember { mutableStateOf(Offset.Zero) }
    val pickHandler = rememberUpdatedState(onPickStep)
    // Pointer x while the mouse is over the plot. Read from the draw phase, so moving the mouse
    // repaints the canvas without recomposing the card.
    val hoverX = remember { mutableStateOf<Float?>(null) }

    val avgStepGap = remember(allMapped) {
        val xs = allMapped.map { it.step }.distinct().sorted()
        if (xs.size < 2) 0f
        else (xs.last() - xs.first()) / (xs.size - 1).toFloat()
    }
    val minXRange = remember(fullBounds, avgStepGap) {
        maxOf(fullBounds.xRange * 0.01f, avgStepGap * 4f)
            .coerceIn(0f, fullBounds.xRange)
            .coerceAtLeast(1e-6f)
    }
    val minYRange = remember(fullBounds) {
        if (fullBounds.yRange <= 1e-9f) 0f else maxOf(fullBounds.yRange * 0.02f, 1e-6f)
    }

    fun visibleSlice(all: List<ChartPoint>, vp: Viewport): List<ChartPoint> {
        if (all.isEmpty()) return emptyList()
        val lo = vp.xMin - vp.xRange * 0.05f
        val hi = vp.xMax + vp.xRange * 0.05f
        val first = all.indexOfFirst { it.step >= lo }.let { if (it > 0) it - 1 else 0 }
        val last = all.indexOfLast { it.step <= hi }.let { if (it < all.lastIndex) it + 1 else all.lastIndex }
        if (first > last) return emptyList()
        return all.subList(first, last + 1)
    }

    val visibleSeries by remember(prepared) {
        derivedStateOf {
            val vp = viewport
            prepared.map { item ->
                item.copy(
                    raw = lttbDownsample(visibleSlice(item.raw, vp), MAX_DRAW_POINTS),
                    smooth = lttbDownsample(visibleSlice(item.smooth, vp), MAX_DRAW_POINTS),
                )
            }
        }
    }

    val colors = rankoColors
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 9.sp, color = colors.textDim.copy(alpha = 0.85f))
    val gridColor = colors.stroke.copy(alpha = 0.45f)
    val axisColor = colors.stroke

    val gestureModifier = modifier
        .pointerInput(fullBounds, rightPad) {
            detectDragGestures(
                onDrag = { change, dragAmount ->
                    val plotW = (size.width.toFloat() - PLOT_LEFT_PADDING - rightPad).coerceAtLeast(1f)
                    val plotH = (size.height.toFloat() - PLOT_BOTTOM_PADDING).coerceAtLeast(1f)
                    val dx = dragAmount.x
                    val dy = dragAmount.y
                    if (!dx.isFinite() || !dy.isFinite()) return@detectDragGestures
                    if (abs(dx) > plotW * JUMP_REJECT_FRACTION ||
                        abs(dy) > plotH * JUMP_REJECT_FRACTION
                    ) {
                        return@detectDragGestures
                    }
                    change.consume()
                    val vp = viewport
                    val moveX = -(dx / plotW) * vp.xRange
                    val moveY = (dy / plotH) * vp.yRange
                    viewport = Viewport(
                        xMin = vp.xMin + moveX,
                        xMax = vp.xMax + moveX,
                        yMin = vp.yMin + moveY,
                        yMax = vp.yMax + moveY,
                    ).clampToBounds(fullBounds)
                },
            )
        }
        .pointerInput(fullBounds, minXRange, minYRange, rightPad) {
            awaitPointerEventScope {
                while (true) {
                    val event = awaitPointerEvent()
                    val change = event.changes.firstOrNull { c ->
                        c.type == PointerType.Mouse &&
                            (event.type == PointerEventType.Scroll ||
                                c.scrollDelta.x != 0f ||
                                c.scrollDelta.y != 0f)
                    } ?: continue

                    val mods = event.keyboardModifiers
                    val zoomX = mods.isCtrlPressed
                    val zoomY = mods.isShiftPressed
                    if (!zoomX && !zoomY) continue

                    // Desktop remaps Shift+wheel to horizontal delta (y == 0, x != 0).
                    val amount = if (change.scrollDelta.y != 0f) {
                        change.scrollDelta.y
                    } else {
                        change.scrollDelta.x
                    }
                    if (amount == 0f || !amount.isFinite()) continue

                    val plotW = (size.width.toFloat() - PLOT_LEFT_PADDING - rightPad).coerceAtLeast(1f)
                    val plotH = (size.height.toFloat() - PLOT_BOTTOM_PADDING).coerceAtLeast(1f)
                    val factor = WHEEL_ZOOM_STEP.pow(-amount * WHEEL_ZOOM_INTENSITY)
                    val vp = viewport
                    val newXRange = if (zoomX) {
                        zoomRange(vp.xRange, factor, minXRange, fullBounds.xRange)
                    } else {
                        vp.xRange
                    }
                    val newYRange = if (zoomY && fullBounds.yRange > 1e-9f) {
                        zoomRange(vp.yRange, factor, minYRange, fullBounds.yRange)
                    } else {
                        vp.yRange
                    }

                    val xFrac = ((change.position.x - PLOT_LEFT_PADDING) / plotW).coerceIn(0f, 1f)
                    val yFrac = (change.position.y / plotH).coerceIn(0f, 1f)
                    val dataX = vp.xMin + xFrac * vp.xRange
                    val dataY = vp.yMax - yFrac * vp.yRange
                    val newXMin = dataX - xFrac * newXRange
                    val newYMin = dataY - (1f - yFrac) * newYRange

                    change.consume()
                    viewport = Viewport(
                        xMin = newXMin,
                        xMax = newXMin + newXRange,
                        yMin = newYMin,
                        yMax = newYMin + newYRange,
                    ).clampToBounds(fullBounds)
                }
            }
        }
        .pointerInput(fullBounds, rightPad) {
            if (pickHandler.value == null) return@pointerInput
            awaitPointerEventScope {
                var trackedId: PointerId? = null
                var downPosition = Offset.Zero
                var travel = 0f
                var ctrlAtPress = false
                // Uptime of the last plain click. A second one inside the double-click window opens
                // the panel as well; only the interval counts, never the distance between the two.
                var lastPlainClickMillis = 0L
                while (true) {
                    val event = awaitPointerEvent()
                    val change = event.changes.firstOrNull() ?: continue

                    if (change.id == trackedId) {
                        travel = maxOf(travel, (change.position - downPosition).getDistance())
                    }

                    when {
                        event.type == PointerEventType.Press &&
                            change.type == PointerType.Mouse &&
                            event.buttons.isPrimaryPressed -> {
                            trackedId = change.id
                            downPosition = change.position
                            travel = 0f
                            ctrlAtPress = event.keyboardModifiers.isCtrlPressed
                        }

                        event.type == PointerEventType.Press -> trackedId = null

                        event.type == PointerEventType.Release && change.id == trackedId -> {
                            trackedId = null
                            if (travel > CLICK_MAX_TRAVEL) continue

                            val clickedAt = change.uptimeMillis
                            val doubleClick = completesDoubleClick(lastPlainClickMillis, clickedAt)
                            // A Ctrl+click picks at once and never pairs with a later plain click.
                            lastPlainClickMillis = if (ctrlAtPress || doubleClick) 0L else clickedAt
                            if (!ctrlAtPress && !doubleClick) continue

                            val plotW = (size.width.toFloat() - PLOT_LEFT_PADDING - rightPad).coerceAtLeast(1f)
                            val plotH = (size.height.toFloat() - PLOT_BOTTOM_PADDING).coerceAtLeast(1f)
                            if (downPosition.y > plotH) continue
                            val step = stepAtPlotX(
                                downPosition.x,
                                plotW,
                                viewportState.value.xMin,
                                viewportState.value.xRange,
                            ) ?: continue
                            // The anchor is where the picking click landed: for a double click that is
                            // the second click, wherever the first one was.
                            pickHandler.value?.invoke(step, canvasOrigin.value + downPosition)
                        }
                    }
                }
            }
        }

    Canvas(
        modifier = gestureModifier
            .onGloballyPositioned {
                canvasOrigin.value = it.boundsInRoot().topLeft
            }
            .pointerInput(showHoverStep, rightPad) {
                if (!showHoverStep) return@pointerInput
                awaitPointerEventScope {
                    while (true) {
                        val event = awaitPointerEvent()
                        val change = event.changes.firstOrNull() ?: continue
                        if (change.type != PointerType.Mouse) continue
                        hoverX.value = when (event.type) {
                            PointerEventType.Enter,
                            PointerEventType.Move,
                            -> {
                                val plotH = (size.height.toFloat() - PLOT_BOTTOM_PADDING).coerceAtLeast(1f)
                                if (change.position.y > plotH || change.position.x < PLOT_LEFT_PADDING) {
                                    null
                                } else {
                                    change.position.x
                                }
                            }

                            PointerEventType.Exit -> null

                            else -> continue
                        }
                    }
                }
            },
    ) {
        val vp = viewport
        val w = size.width
        val h = size.height
        val leftPad = PLOT_LEFT_PADDING
        val bottomPad = PLOT_BOTTOM_PADDING
        val plotW = w - leftPad - rightPad
        val plotH = h - bottomPad

        fun dataToScreen(step: Float, value: Float) = Offset(
            x = leftPad + ((step - vp.xMin) / vp.xRange) * plotW,
            y = ((vp.yMax - value) / vp.yRange) * plotH,
        )

        drawGridAndLabels(
            vp = vp,
            plotW = plotW,
            plotH = plotH,
            leftPad = leftPad,
            totalW = w,
            gridColor = gridColor,
            axisColor = axisColor,
            textMeasurer = textMeasurer,
            labelStyle = labelStyle,
            leftDomain = leftDomain,
            leftColor = leftAxisColor,
            rightDomain = rightDomain,
            rightColor = rightAxisColor,
        )

        clipRect(left = leftPad, top = 0f, right = leftPad + plotW, bottom = plotH) {
            val rawStroke = (strokeWidth * 0.5f).coerceAtLeast(0.8f)
            for (item in visibleSeries) {
                if (item.raw.isEmpty() && item.smooth.isEmpty()) continue
                drawPath(
                    buildPath(item.raw, ::dataToScreen),
                    item.color.copy(alpha = 0.20f),
                    style = Stroke(width = rawStroke),
                )
                drawPath(
                    buildPath(item.smooth, ::dataToScreen),
                    item.color,
                    style = Stroke(width = strokeWidth),
                )
                item.smooth.lastOrNull()?.let { last ->
                    val pt = dataToScreen(last.step, last.value)
                    drawCircle(item.color, radius = strokeWidth * 1.6f, center = pt)
                    drawCircle(Color.White, radius = strokeWidth * 0.8f, center = pt)
                }
            }

            // The Ctrl+click that the panel is showing: the clicked step stays subtle, the step the
            // pick actually matched gets the loud marker, so the snapping is visible.
            pickMarkers?.let { markers ->
                val clickedX = stepToScreenX(markers.clickedStep, vp, leftPad, plotW)
                if (clickedX != null) {
                    drawLine(
                        colors.textDim.copy(alpha = 0.75f),
                        Offset(clickedX, 0f),
                        Offset(clickedX, plotH),
                        strokeWidth = 1.2f,
                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(5f, 5f)),
                    )
                }
                val matched = markers.matchedStep
                val matchedX = matched?.let { stepToScreenX(it.toFloat(), vp, leftPad, plotW) }
                if (matched != null && matchedX != null) {
                    val accent = colors.accentPink
                    drawLine(accent, Offset(matchedX, 0f), Offset(matchedX, plotH), strokeWidth = 2.5f)
                    drawCircle(accent, radius = 4.5f, center = Offset(matchedX, 5f))
                    // Small flag pointing down at the axis, inside the plot rect so it is not clipped.
                    drawPath(
                        Path().apply {
                            moveTo(matchedX, plotH)
                            lineTo(matchedX - 6f, plotH - 9f)
                            lineTo(matchedX + 6f, plotH - 9f)
                            close()
                        },
                        accent,
                    )
                    drawMarkerLabel(
                        text = "ckpt $matched",
                        centerX = matchedX,
                        topY = 3f,
                        plotLeft = leftPad,
                        plotRight = w,
                        textMeasurer = textMeasurer,
                        style = labelStyle.copy(color = Color.White, fontWeight = FontWeight.Bold),
                        background = accent.copy(alpha = 0.92f),
                    )
                }
            }

            // Always-on readout: the exact step under the pointer.
            hoverX.value?.let { x ->
                val step = stepAtPlotX(x, plotW, vp.xMin, vp.xRange)
                val anchor = step?.let { nearestPointByStep(prepared.first().raw, it) }
                if (anchor != null) {
                    val lineX = stepToScreenX(anchor.step, vp, leftPad, plotW)
                    if (lineX != null) {
                        drawLine(
                            colors.text.copy(alpha = 0.55f),
                            Offset(lineX, 0f),
                            Offset(lineX, plotH),
                            strokeWidth = 1.2f,
                        )
                        drawCircle(colors.text, radius = 3f, center = Offset(lineX, plotH))
                        drawMarkerLabel(
                            text = "step ${anchor.step.roundToInt()}",
                            centerX = lineX,
                            topY = plotH - 20f,
                            plotLeft = leftPad,
                            plotRight = w,
                            textMeasurer = textMeasurer,
                            style = labelStyle.copy(color = colors.text),
                            background = colors.bgCard.copy(alpha = 0.88f),
                        )
                    }
                }
            }
        }
    }
}

/** Screen x of a data step, or null when it is outside the visible viewport. */
private fun stepToScreenX(step: Float, vp: Viewport, leftPad: Float, plotW: Float): Float? {
    if (step < vp.xMin - 1e-3f || step > vp.xMax + 1e-3f) return null
    return leftPad + ((step - vp.xMin) / vp.xRange) * plotW
}

/** Closest logged step, so the readout always names a step that was actually logged. */
private fun nearestPointByStep(points: List<ChartPoint>, step: Float): ChartPoint? =
    points.minByOrNull { abs(it.step - step) }

private fun DrawScope.drawMarkerLabel(
    text: String,
    centerX: Float,
    topY: Float,
    plotLeft: Float,
    plotRight: Float,
    textMeasurer: TextMeasurer,
    style: TextStyle,
    background: Color,
) {
    val layout = textMeasurer.measure(text, style)
    val padX = 5f
    val padY = 3f
    val boxW = layout.size.width + padX * 2
    val boxH = layout.size.height + padY * 2
    val boxX = (centerX - boxW / 2f).coerceIn(plotLeft + 2f, (plotRight - boxW - 2f).coerceAtLeast(plotLeft + 2f))
    drawRoundRect(
        color = background,
        topLeft = Offset(boxX, topY),
        size = Size(boxW, boxH),
        cornerRadius = CornerRadius(6f, 6f),
    )
    drawText(layout, topLeft = Offset(boxX + padX, topY + padY))
}

private fun DrawScope.drawGridAndLabels(
    vp: Viewport,
    plotW: Float,
    plotH: Float,
    leftPad: Float,
    totalW: Float,
    gridColor: Color,
    axisColor: Color,
    textMeasurer: TextMeasurer,
    labelStyle: TextStyle,
    leftDomain: Pair<Float, Float>? = null,
    leftColor: Color = Color.Unspecified,
    rightDomain: Pair<Float, Float>? = null,
    rightColor: Color = Color.Unspecified,
) {
    val xTicks = 5
    val yTicks = 4

    for (i in 0..yTicks) {
        val frac = i.toFloat() / yTicks
        val yScr = plotH - frac * plotH
        drawLine(gridColor, Offset(leftPad, yScr), Offset(leftPad + plotW, yScr), strokeWidth = 1f)

        val leftValue = if (leftDomain != null) {
            axisTickValue(vp.yMin, vp.yRange, frac, leftDomain.first, leftDomain.second)
        } else {
            vp.yMin + frac * vp.yRange
        }
        val leftStyle = if (leftDomain != null) labelStyle.copy(color = leftColor) else labelStyle
        val leftLayout = textMeasurer.measure(formatAxisValue(leftValue), leftStyle)
        drawText(
            leftLayout,
            topLeft = Offset(
                x = (leftPad - leftLayout.size.width - 4f).coerceAtLeast(0f),
                y = yScr - leftLayout.size.height / 2f,
            ),
        )

        if (rightDomain != null) {
            val rightValue = axisTickValue(vp.yMin, vp.yRange, frac, rightDomain.first, rightDomain.second)
            val rightLayout = textMeasurer.measure(
                formatAxisValue(rightValue),
                labelStyle.copy(color = rightColor),
            )
            drawText(
                rightLayout,
                topLeft = Offset(
                    x = totalW - rightLayout.size.width,
                    y = yScr - rightLayout.size.height / 2f,
                ),
            )
        }
    }

    for (i in 0..xTicks) {
        val frac = i.toFloat() / xTicks
        val xVal = vp.xMin + frac * vp.xRange
        val xScr = leftPad + frac * plotW
        drawLine(gridColor, Offset(xScr, 0f), Offset(xScr, plotH), strokeWidth = 1f)
        val layout = textMeasurer.measure(formatAxisValue(xVal), labelStyle)
        drawText(
            layout,
            topLeft = Offset(
                x = xScr - layout.size.width / 2f,
                y = plotH + 4f,
            ),
        )
    }

    drawLine(axisColor, Offset(leftPad, 0f), Offset(leftPad, plotH), strokeWidth = 1f)
    drawLine(axisColor, Offset(leftPad, plotH), Offset(leftPad + plotW, plotH), strokeWidth = 1f)
}

private fun buildPath(
    points: List<ChartPoint>,
    toScreen: (Float, Float) -> Offset,
): Path = Path().apply {
    if (points.isEmpty()) return@apply
    val first = toScreen(points[0].step, points[0].value)
    moveTo(first.x, first.y)
    for (i in 1..points.lastIndex) {
        val pt = toScreen(points[i].step, points[i].value)
        lineTo(pt.x, pt.y)
    }
}

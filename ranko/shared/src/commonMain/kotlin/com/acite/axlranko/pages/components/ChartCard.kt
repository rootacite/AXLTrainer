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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.PointerType
import androidx.compose.ui.input.pointer.isCtrlPressed
import androidx.compose.ui.input.pointer.isShiftPressed
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.model.MetricPoint
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
) {
    Card(
        modifier = modifier.height(chartHeight),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(modifier = Modifier.padding(16.dp).fillMaxSize()) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .clip(RoundedCornerShape(5.dp))
                        .background(color)
                )
                Text(
                    title,
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                )
            }
            Spacer(modifier = Modifier.height(12.dp))
            if (points.isNotEmpty()) {
                InteractiveLineChart(
                    points = points,
                    color = color,
                    smoothing = smoothing,
                    outlierClip = outlierClip,
                    strokeWidth = strokeWidth,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text(
                        "No Data",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

private const val MAX_DRAW_POINTS = 500

@Composable
private fun InteractiveLineChart(
    points: List<MetricPoint>,
    color: Color,
    smoothing: Float,
    outlierClip: Float,
    strokeWidth: Float,
    modifier: Modifier = Modifier,
) {
    val rawPoints = remember(points) {
        points.map { ChartPoint(it.step.toFloat(), it.value) }.sortedBy { it.step }
    }
    if (rawPoints.isEmpty()) return

    val smoothedPoints = remember(rawPoints, smoothing) {
        if (smoothing <= 0f) return@remember rawPoints
        val out = ArrayList<ChartPoint>(rawPoints.size)
        var ema = rawPoints.first().value
        for (p in rawPoints) {
            ema = ema * smoothing + (1f - smoothing) * p.value
            out += ChartPoint(p.step, ema)
        }
        out
    }

    val fullBounds = remember(rawPoints) {
        val sortedY = rawPoints.map { it.value }.sorted()
        Viewport(
            xMin = rawPoints.minOf { it.step },
            xMax = rawPoints.maxOf { it.step },
            yMin = sortedY.first(),
            yMax = sortedY.last(),
        )
    }

    val initialViewport = remember(rawPoints, outlierClip, fullBounds) {
        val sortedY = rawPoints.map { it.value }.sorted()
        val half = (outlierClip / 2f).coerceIn(0f, 0.49f)
        val yLo = sortedY.percentile(half)
        val yHi = sortedY.percentile(1f - half)
        val yPad = ((yHi - yLo) * 0.05f).coerceAtLeast(abs(yHi) * 0.01f)
        Viewport(
            xMin = fullBounds.xMin,
            xMax = fullBounds.xMax,
            yMin = yLo - yPad,
            yMax = yHi + yPad,
        )
    }

    var viewport by remember(initialViewport) { mutableStateOf(initialViewport) }

    val avgStepGap = remember(rawPoints) {
        if (rawPoints.size < 2) 0f
        else (rawPoints.last().step - rawPoints.first().step) / (rawPoints.size - 1).toFloat()
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
        val lo = vp.xMin - vp.xRange * 0.05f
        val hi = vp.xMax + vp.xRange * 0.05f
        val first = all.indexOfFirst { it.step >= lo }.let { if (it > 0) it - 1 else 0 }
        val last = all.indexOfLast { it.step <= hi }.let { if (it < all.lastIndex) it + 1 else all.lastIndex }
        if (first > last) return emptyList()
        return all.subList(first, last + 1)
    }

    data class VisiblePoints(
        val raw: List<ChartPoint>,
        val smooth: List<ChartPoint>,
    )

    val visiblePoints: VisiblePoints by remember(rawPoints, smoothedPoints) {
        derivedStateOf {
            val vp = viewport
            VisiblePoints(
                raw = lttbDownsample(visibleSlice(rawPoints, vp), MAX_DRAW_POINTS),
                smooth = lttbDownsample(visibleSlice(smoothedPoints, vp), MAX_DRAW_POINTS),
            )
        }
    }

    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 9.sp, color = Color.Gray.copy(alpha = 0.7f))
    val gridColor = Color.Gray.copy(alpha = 0.12f)
    val axisColor = Color.Gray.copy(alpha = 0.3f)

    val gestureModifier = modifier
        .pointerInput(fullBounds) {
            detectDragGestures(
                onDrag = { change, dragAmount ->
                    val plotW = (size.width.toFloat() - 52f).coerceAtLeast(1f)
                    val plotH = (size.height.toFloat() - 28f).coerceAtLeast(1f)
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
        .pointerInput(fullBounds, minXRange, minYRange) {
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

                    val plotW = (size.width.toFloat() - 52f).coerceAtLeast(1f)
                    val plotH = (size.height.toFloat() - 28f).coerceAtLeast(1f)
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

                    val xFrac = ((change.position.x - 52f) / plotW).coerceIn(0f, 1f)
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

    Canvas(modifier = gestureModifier) {
        val vp = viewport
        val w = size.width
        val h = size.height
        val leftPad = 52f
        val bottomPad = 28f
        val plotW = w - leftPad
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
        )

        clipRect(left = leftPad, top = 0f, right = w, bottom = plotH) {
            val vp2 = visiblePoints
            val rawStroke = (strokeWidth * 0.5f).coerceAtLeast(0.8f)
            drawPath(buildPath(vp2.raw, ::dataToScreen), color.copy(alpha = 0.20f), style = Stroke(width = rawStroke))
            drawPath(buildPath(vp2.smooth, ::dataToScreen), color, style = Stroke(width = strokeWidth))
            visiblePoints.smooth.lastOrNull()?.let { last ->
                val pt = dataToScreen(last.step, last.value)
                drawCircle(color, radius = strokeWidth * 1.6f, center = pt)
                drawCircle(Color.White, radius = strokeWidth * 0.8f, center = pt)
            }
        }
    }
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
) {
    val xTicks = 5
    val yTicks = 4

    for (i in 0..yTicks) {
        val frac = i.toFloat() / yTicks
        val yVal = vp.yMin + frac * vp.yRange
        val yScr = plotH - frac * plotH
        drawLine(gridColor, Offset(leftPad, yScr), Offset(totalW, yScr), strokeWidth = 1f)
        val layout = textMeasurer.measure(formatAxisValue(yVal), labelStyle)
        drawText(
            layout,
            topLeft = Offset(
                x = (leftPad - layout.size.width - 4f).coerceAtLeast(0f),
                y = yScr - layout.size.height / 2f,
            ),
        )
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
    drawLine(axisColor, Offset(leftPad, plotH), Offset(totalW, plotH), strokeWidth = 1f)
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

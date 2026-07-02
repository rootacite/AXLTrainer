package components.publik

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.exponentialDecay
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
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
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
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
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.time.Duration.Companion.milliseconds

// ─────────────────────────────────────────────────────────────────────────────
// Data classes & helpers
// ─────────────────────────────────────────────────────────────────────────────

private data class ChartPoint(val step: Float, val value: Float)

private data class Viewport(
    val xMin: Float,
    val xMax: Float,
    val yMin: Float,
    val yMax: Float,
) {
    val xRange get() = (xMax - xMin).coerceAtLeast(1e-6f)
    val yRange get() = (yMax - yMin).coerceAtLeast(1e-6f)

    fun clamp(bounds: Viewport, overshoot: Float = 0.2f): Viewport {
        val xPad = bounds.xRange * overshoot
        val yPad = bounds.yRange * overshoot
        val w = xRange
        val h = yRange
        val x0 = xMin.coerceIn(bounds.xMin - xPad, (bounds.xMax + xPad - w))
        val y0 = yMin.coerceIn(bounds.yMin - yPad, (bounds.yMax + yPad - h))
        return copy(xMin = x0, xMax = x0 + w, yMin = y0, yMax = y0 + h)
    }
}

private fun List<Float>.percentile(p: Float): Float {
    if (isEmpty()) return 0f
    val idx = (p * (size - 1)).coerceIn(0f, (size - 1).toFloat())
    val lo = idx.toInt()
    val hi = min(lo + 1, size - 1)
    return this[lo] + (idx - lo) * (this[hi] - this[lo])
}

// ─────────────────────────────────────────────────────────────────────────────
// KMP-safe number formatting
// ─────────────────────────────────────────────────────────────────────────────

private fun formatAxisValue(v: Float): String {
    val a = abs(v)
    return when {
        a == 0f             -> "0"
        a >= 1_000_000f     -> formatScientific(v, 1)
        a < 0.01f && a > 0f -> formatScientific(v, 1)
        a >= 1_000f         -> formatFixed(v, 0)
        a >= 1f             -> formatFixed(v, 2)
        else                -> formatFixed(v, 3)
    }
}

private fun formatFixed(v: Float, decimals: Int): String {
    if (decimals == 0) return v.roundToInt().toString()
    val scale    = 10.0.pow(decimals).toFloat()
    val rounded  = (v * scale).roundToInt()
    val intPart  = rounded / scale.toInt()
    val fracPart = abs(rounded) % scale.toInt()
    val fracStr  = fracPart.toString().padStart(decimals, '0')
    return if (v < 0 && intPart == 0) "-0.$fracStr" else "$intPart.$fracStr"
}

private fun formatScientific(v: Float, mantissaDecimals: Int): String {
    if (v == 0f) return "0"
    val sign    = if (v < 0) "-" else ""
    val absV    = abs(v)
    val exp     = floor(ln(absV) / ln(10.0)).toInt()
    val mant    = absV / 10.0.pow(exp).toFloat()
    val expSign = if (exp >= 0) "+" else "-"
    return "${sign}${formatFixed(mant, mantissaDecimals)}e${expSign}${abs(exp)}"
}

// ─────────────────────────────────────────────────────────────────────────────
// LTTB (Largest-Triangle-Three-Buckets) down-sampling
// Keeps the visually significant points when count > maxPoints.
// Time complexity: O(n), allocations: 1 ArrayList of size maxPoints.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Down-samples [points] to at most [maxPoints] using the LTTB algorithm.
 * Returns the original list unchanged when it is already small enough.
 */
private fun lttbDownsample(points: List<ChartPoint>, maxPoints: Int): List<ChartPoint> {
    val n = points.size
    if (maxPoints !in 3..<n) return points

    val sampled  = ArrayList<ChartPoint>(maxPoints)
    // Always keep first and last
    sampled.add(points.first())

    val bucketCount = maxPoints - 2               // buckets between first and last
    val bucketSize  = (n - 2).toDouble() / bucketCount

    var prevSelected = 0                           // index of the last kept point

    for (b in 0 until bucketCount) {
        // Next-bucket average (used as the "future" anchor)
        val nextStart = ((b + 1) * bucketSize + 1).toInt().coerceAtMost(n - 1)
        val nextEnd   = ((b + 2) * bucketSize + 1).toInt().coerceAtMost(n)
        var avgX = 0.0; var avgY = 0.0
        val nextLen = (nextEnd - nextStart).coerceAtLeast(1)
        for (i in nextStart until nextEnd) { avgX += points[i].step; avgY += points[i].value }
        avgX /= nextLen; avgY /= nextLen

        // Current bucket range
        val curStart = (b       * bucketSize + 1).toInt().coerceAtMost(n - 1)
        val curEnd   = ((b + 1) * bucketSize + 1).toInt().coerceAtMost(n - 1)

        val ax = points[prevSelected].step.toDouble()
        val ay = points[prevSelected].value.toDouble()

        var maxArea   = -1.0
        var maxIndex  = curStart
        for (i in curStart until curEnd) {
            // Triangle area × 2 (sign doesn't matter, we only compare magnitudes)
            val area = abs(
                (ax - avgX) * (points[i].value - ay) -
                        (ax - points[i].step) * (avgY - ay)
            )
            if (area > maxArea) { maxArea = area; maxIndex = i }
        }

        sampled.add(points[maxIndex])
        prevSelected = maxIndex
    }

    sampled.add(points.last())
    return sampled
}

// ─────────────────────────────────────────────────────────────────────────────
// Public composable
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ChartCard(
    title: String,
    dataArray: JsonArray?,
    color: Color,
    smoothing: Float,
    modifier: Modifier = Modifier,
    outlierClip: Float = 0.15f,
) {
    ElevatedCard(
        modifier  = modifier.height(220.dp),
        shape     = RoundedCornerShape(16.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp),
    ) {
        Column(modifier = Modifier.padding(20.dp).fillMaxSize()) {
            Row(
                verticalAlignment     = Alignment.CenterVertically,
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
            Spacer(modifier = Modifier.height(16.dp))
            if (!dataArray.isNullOrEmpty()) {
                InteractiveLineChart(
                    dataArray   = dataArray,
                    color       = color,
                    smoothing   = smoothing,
                    outlierClip = outlierClip,
                    modifier    = Modifier.fillMaxSize(),
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

// ─────────────────────────────────────────────────────────────────────────────
// Interactive chart
// ─────────────────────────────────────────────────────────────────────────────

/** Max screen-space points to hand to the GPU per series. */
private const val MAX_DRAW_POINTS = 500

@Composable
private fun InteractiveLineChart(
    dataArray: JsonArray,
    color: Color,
    smoothing: Float,
    outlierClip: Float,
    modifier: Modifier = Modifier,
) {
    /* ── Parse & sort  (once per dataArray change) ── */
    val rawPoints = remember(dataArray) {
        dataArray.mapNotNull {
            val obj   = it.jsonObject
            val step  = obj["step"]?.jsonPrimitive?.floatOrNull
            val value = obj["value"]?.jsonPrimitive?.floatOrNull
            if (step != null && value != null) ChartPoint(step, value) else null
        }.sortedBy { it.step }
    }
    if (rawPoints.isEmpty()) return

    /* ── EMA smoothing  (once per data/smoothing change) ── */
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

    /* ── Full data bounds ── */
    val fullBounds = remember(rawPoints) {
        val sortedY = rawPoints.map { it.value }.sorted()
        Viewport(
            xMin = rawPoints.minOf { it.step },
            xMax = rawPoints.maxOf { it.step },
            yMin = sortedY.first(),
            yMax = sortedY.last(),
        )
    }

    /* ── Initial viewport ── */
    val initialViewport = remember(rawPoints, outlierClip, fullBounds) {
        val sortedY = rawPoints.map { it.value }.sorted()
        val half    = (outlierClip / 2f).coerceIn(0f, 0.49f)
        val yLo     = sortedY.percentile(half)
        val yHi     = sortedY.percentile(1f - half)
        val yPad    = ((yHi - yLo) * 0.05f).coerceAtLeast(abs(yHi) * 0.01f)
        Viewport(
            xMin = fullBounds.xMin,
            xMax = fullBounds.xMax,
            yMin = yLo - yPad,
            yMax = yHi + yPad,
        )
    }

    /* ── Viewport state ── */
    var viewport by remember(initialViewport) { mutableStateOf(initialViewport) }

    // ─────────────────────────────────────────────────────────────────────────
    // KEY OPTIMIZATION 1: derivedStateOf + LTTB per viewport
    //
    // The visible window changes on every pan/zoom frame, but re-computing
    // which points to draw only needs to happen when `viewport` actually
    // settles to a new value.  derivedStateOf batches rapid updates so we
    // skip redundant work during fast flings.
    //
    // Inside we:
    //   1. Slice to the visible x-range (+ one-point margin each side).
    //   2. Run LTTB so we never hand more than MAX_DRAW_POINTS to the GPU.
    //   3. Build the Path objects once; Canvas just calls drawPath().
    // ─────────────────────────────────────────────────────────────────────────

    /** Slice [all] to points visible in [vp] plus one guard point on each side. */
    fun visibleSlice(all: List<ChartPoint>, vp: Viewport): List<ChartPoint> {
        val lo = vp.xMin - vp.xRange * 0.05f   // 5 % margin
        val hi = vp.xMax + vp.xRange * 0.05f
        var first = all.indexOfFirst { it.step >= lo }.let { if (it > 0) it - 1 else 0 }
        var last  = all.indexOfLast  { it.step <= hi }.let { if (it < all.lastIndex) it + 1 else all.lastIndex }
        if (first > last) return emptyList()
        return all.subList(first, last + 1)
    }

    data class DrawPaths(val raw: Path, val smooth: Path, val lastPt: Offset?)

    val drawPaths: DrawPaths by remember(rawPoints, smoothedPoints) {
        derivedStateOf {
            val vp = viewport

            // 1. Slice to visible x-window
            val visRaw    = visibleSlice(rawPoints,     vp)
            val visSmooth = visibleSlice(smoothedPoints, vp)

            // 2. LTTB down-sample to MAX_DRAW_POINTS
            val sampledRaw    = lttbDownsample(visRaw,    MAX_DRAW_POINTS)
            val sampledSmooth = lttbDownsample(visSmooth, MAX_DRAW_POINTS)

            // 3. Build Paths in data→screen space
            //    (plotW / plotH estimated at 1f per dp; Canvas will use real px)
            //    We defer actual pixel mapping into Canvas to avoid depending on
            //    layout size here; instead we store data coords and let Canvas draw.
            //    → Actually we build paths directly in screen space using a
            //       lambda captured below; paths are value objects so this is fine.

            // Canvas layout constants (must match Canvas block below)
            val leftPad   = 52f
            val bottomPad = 28f

            // We don't know the actual canvas size here, so we use a sentinel
            // size and rebuild when the canvas first measures itself.
            // Simpler approach: store the sampled point lists and build paths
            // inside Canvas (cheap once they're already small).
            //
            // Therefore drawPaths stores the *sampled* lists, not actual Paths.
            // We expose them as Paths built with the real size inside Canvas.
            // → See VisiblePoints below.
            DrawPaths(Path(), Path(), null) // placeholder; real paths built in Canvas
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // KEY OPTIMIZATION 2: Store only the sampled point lists in derived state.
    // Path construction (which walks the list and calls moveTo/lineTo) stays
    // inside the Canvas draw lambda where the real pixel size is known.
    // This avoids the size-estimation problem above.
    // ─────────────────────────────────────────────────────────────────────────

    data class VisiblePoints(
        val raw: List<ChartPoint>,
        val smooth: List<ChartPoint>,
    )

    val visiblePoints: VisiblePoints by remember(rawPoints, smoothedPoints) {
        derivedStateOf {
            val vp        = viewport
            val visRaw    = visibleSlice(rawPoints,     vp)
            val visSmooth = visibleSlice(smoothedPoints, vp)
            VisiblePoints(
                raw    = lttbDownsample(visRaw,    MAX_DRAW_POINTS),
                smooth = lttbDownsample(visSmooth, MAX_DRAW_POINTS),
            )
        }
    }

    /* ── Fling ── */
    val scope      = rememberCoroutineScope()
    val flingAnimX = remember { Animatable(0f) }
    val flingAnimY = remember { Animatable(0f) }
    var flingJob  by remember { mutableStateOf<Job?>(null) }

    /* ── Drawing resources ── */
    val textMeasurer = rememberTextMeasurer()
    val labelStyle   = TextStyle(fontSize = 9.sp, color = Color.Gray.copy(alpha = 0.7f))
    val gridColor    = Color.Gray.copy(alpha = 0.12f)
    val axisColor    = Color.Gray.copy(alpha = 0.3f)

    /* ── Gesture handler (unchanged from original) ── */
    val gestureModifier = modifier.pointerInput(fullBounds) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false)
            flingJob?.cancel()

            val velTracker = VelocityTracker()
            velTracker.addPosition(down.uptimeMillis, down.position)

            var prevCentroid = down.position
            var pointerCount = 1

            do {
                val event   = awaitPointerEvent()
                val pressed = event.changes.filter { it.pressed }
                if (pressed.isEmpty()) break

                pointerCount = pressed.size

                val centroid = event.calculateCentroid(useCurrent = true)
                val pan      = event.calculatePan()
                val zoom     = event.calculateZoom()

                pressed.forEach { c -> if (c.positionChanged()) c.consume() }

                if (pointerCount == 1) {
                    velTracker.addPosition(pressed.first().uptimeMillis, centroid)
                }

                val vp      = viewport
                val canvasW = size.width.toFloat()
                val canvasH = size.height.toFloat()
                val leftPad   = 52f
                val bottomPad = 28f
                val plotW = canvasW - leftPad
                val plotH = canvasH - bottomPad

                if (plotW <= 0f || plotH <= 0f) continue

                val zoomSafe  = zoom.coerceIn(0.5f, 2f)
                val dataX = vp.xMin + ((centroid.x - leftPad) / plotW) * vp.xRange
                val dataY = vp.yMax -  (centroid.y            / plotH) * vp.yRange

                val newXRange = (vp.xRange / zoomSafe)
                    .coerceIn(vp.xRange * 0.001f, fullBounds.xRange * 20f)
                val newYRange = (vp.yRange / zoomSafe)
                    .coerceIn(vp.yRange * 0.001f, fullBounds.yRange * 20f)

                val dxData = -(pan.x / plotW) * newXRange
                val dyData =  (pan.y / plotH) * newYRange

                val xFrac = ((centroid.x - leftPad) / plotW).coerceIn(0f, 1f)
                val yFrac = (centroid.y             / plotH).coerceIn(0f, 1f)

                val newXMin = dataX - xFrac * newXRange + dxData
                val newYMin = dataY - (1f - yFrac) * newYRange + dyData

                viewport = Viewport(
                    xMin = newXMin,
                    xMax = newXMin + newXRange,
                    yMin = newYMin,
                    yMax = newYMin + newYRange,
                ).clamp(fullBounds)

                prevCentroid = centroid
            } while (pressed.any { it.pressed })

            /* ── Fling ── */
            if (pointerCount == 1) {
                val velocity = velTracker.calculateVelocity()
                val velXPx   = velocity.x
                val velYPx   = velocity.y

                val vp      = viewport
                val canvasW = size.width.toFloat()
                val canvasH = size.height.toFloat()
                val plotW   = canvasW - 52f
                val plotH   = canvasH - 28f

                val dataVx = -(velXPx / plotW) * vp.xRange
                val dataVy =  (velYPx / plotH) * vp.yRange

                flingJob = scope.launch {
                    coroutineScope {
                        launch {
                            flingAnimX.snapTo(0f)
                            flingAnimX.animateDecay(
                                initialVelocity = dataVx,
                                animationSpec   = exponentialDecay(frictionMultiplier = 2.5f),
                            )
                        }
                        launch {
                            flingAnimY.snapTo(0f)
                            flingAnimY.animateDecay(
                                initialVelocity = dataVy,
                                animationSpec   = exponentialDecay(frictionMultiplier = 2.5f),
                            )
                        }
                    }
                }

                scope.launch {
                    var prevX = 0f
                    var prevY = 0f
                    while (isActive) {
                        val dx = flingAnimX.value - prevX
                        val dy = flingAnimY.value - prevY
                        prevX  = flingAnimX.value
                        prevY  = flingAnimY.value
                        if (abs(dx) < 1e-5f && abs(dy) < 1e-5f) break
                        viewport = viewport.let { v ->
                            Viewport(
                                xMin = v.xMin + dx,
                                xMax = v.xMax + dx,
                                yMin = v.yMin + dy,
                                yMax = v.yMax + dy,
                            ).clamp(fullBounds)
                        }
                        delay(16L.milliseconds)
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Canvas: Path construction is now O(MAX_DRAW_POINTS) regardless of how
    // many raw data points there are.  derivedStateOf ensures we only rebuild
    // visiblePoints when the viewport actually changes between frames.
    // ─────────────────────────────────────────────────────────────────────────

    Canvas(modifier = gestureModifier) {
        val vp        = viewport
        val w         = size.width
        val h         = size.height
        val leftPad   = 52f
        val bottomPad = 28f
        val plotW     = w - leftPad
        val plotH     = h - bottomPad

        fun dataToScreen(step: Float, value: Float) = Offset(
            x = leftPad + ((step  - vp.xMin) / vp.xRange) * plotW,
            y =           ((vp.yMax - value)  / vp.yRange) * plotH,
        )

        drawGridAndLabels(
            vp           = vp,
            plotW        = plotW,
            plotH        = plotH,
            leftPad      = leftPad,
            totalW       = w,
            gridColor    = gridColor,
            axisColor    = axisColor,
            textMeasurer = textMeasurer,
            labelStyle   = labelStyle,
        )

        clipRect(left = leftPad, top = 0f, right = w, bottom = plotH) {
            // visiblePoints is already sliced + LTTB-sampled; O(MAX_DRAW_POINTS)
            val vp2 = visiblePoints
            val rawPath    = buildPath(vp2.raw,    ::dataToScreen)
            val smoothPath = buildPath(vp2.smooth, ::dataToScreen)

            drawPath(rawPath,    color.copy(alpha = 0.20f), style = Stroke(width = 1.5f))
            drawPath(smoothPath, color,                     style = Stroke(width = 3f))

            visiblePoints.smooth.lastOrNull()?.let { last ->
                val pt = dataToScreen(last.step, last.value)
                drawCircle(color,       radius = 5f,   center = pt)
                drawCircle(Color.White, radius = 2.5f, center = pt)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid + axis labels
// ─────────────────────────────────────────────────────────────────────────────

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
        val frac  = i.toFloat() / yTicks
        val yVal  = vp.yMin + frac * vp.yRange
        val yScr  = plotH - frac * plotH
        drawLine(gridColor, Offset(leftPad, yScr), Offset(totalW, yScr), strokeWidth = 1f)
        val layout = textMeasurer.measure(formatAxisValue(yVal), labelStyle)
        drawText(layout, topLeft = Offset(
            x = (leftPad - layout.size.width - 4f).coerceAtLeast(0f),
            y = yScr - layout.size.height / 2f,
        ))
    }

    for (i in 0..xTicks) {
        val frac  = i.toFloat() / xTicks
        val xVal  = vp.xMin + frac * vp.xRange
        val xScr  = leftPad + frac * plotW
        drawLine(gridColor, Offset(xScr, 0f), Offset(xScr, plotH), strokeWidth = 1f)
        val layout = textMeasurer.measure(formatAxisValue(xVal), labelStyle)
        drawText(layout, topLeft = Offset(
            x = xScr - layout.size.width / 2f,
            y = plotH + 4f,
        ))
    }

    drawLine(axisColor, Offset(leftPad, 0f),    Offset(leftPad, plotH), strokeWidth = 1f)
    drawLine(axisColor, Offset(leftPad, plotH), Offset(totalW,  plotH), strokeWidth = 1f)
}

// ─────────────────────────────────────────────────────────────────────────────
// Path builder  (unchanged API, but input is now already small)
// ─────────────────────────────────────────────────────────────────────────────

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
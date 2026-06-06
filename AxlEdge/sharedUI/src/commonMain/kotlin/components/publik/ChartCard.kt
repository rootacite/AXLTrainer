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
import androidx.compose.runtime.getValue
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

    /**
     * Soft-clamp: allows [overshoot] fraction of extra space beyond full bounds
     * so panning feels springy rather than hard-stopped.
     */
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

/** Linear interpolation between sorted-list percentiles, p ∈ [0, 1]. */
private fun List<Float>.percentile(p: Float): Float {
    if (isEmpty()) return 0f
    val idx = (p * (size - 1)).coerceIn(0f, (size - 1).toFloat())
    val lo = idx.toInt()
    val hi = min(lo + 1, size - 1)
    return this[lo] + (idx - lo) * (this[hi] - this[lo])
}

// ─────────────────────────────────────────────────────────────────────────────
// KMP-safe number formatting  (zero java.util / String.format)
// ─────────────────────────────────────────────────────────────────────────────

private fun formatAxisValue(v: Float): String {
    val a = abs(v)
    return when {
        a == 0f                      -> "0"
        a >= 1_000_000f              -> formatScientific(v, 1)
        a < 0.01f && a > 0f          -> formatScientific(v, 1)
        a >= 1_000f                  -> formatFixed(v, 0)
        a >= 1f                      -> formatFixed(v, 2)
        else                         -> formatFixed(v, 3)
    }
}

private fun formatFixed(v: Float, decimals: Int): String {
    if (decimals == 0) return v.roundToInt().toString()
    val scale   = 10.0.pow(decimals).toFloat()
    val rounded = (v * scale).roundToInt()
    val intPart = rounded / scale.toInt()
    val fracPart = abs(rounded) % scale.toInt()
    val fracStr  = fracPart.toString().padStart(decimals, '0')
    return if (v < 0 && intPart == 0) "-0.$fracStr" else "$intPart.$fracStr"
}

private fun formatScientific(v: Float, mantissaDecimals: Int): String {
    if (v == 0f) return "0"
    val sign  = if (v < 0) "-" else ""
    val absV  = abs(v)
    val exp   = floor(ln(absV) / ln(10.0)).toInt()
    val mant  = absV / 10.0.pow(exp).toFloat()
    val expSign = if (exp >= 0) "+" else "-"
    return "${sign}${formatFixed(mant, mantissaDecimals)}e${expSign}${abs(exp)}"
}

// ─────────────────────────────────────────────────────────────────────────────
// Public composable
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Training-curve card with TensorBoard-style pan/zoom and outlier clipping.
 *
 * @param outlierClip  Fraction of extreme Y values hidden in the initial view.
 *                     Default 0.15 → hides bottom 7.5 % and top 7.5 % of values.
 */
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
                verticalAlignment   = Alignment.CenterVertically,
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

@Composable
private fun InteractiveLineChart(
    dataArray: JsonArray,
    color: Color,
    smoothing: Float,
    outlierClip: Float,
    modifier: Modifier = Modifier,
) {
    /* ── Parse & sort ── */
    val rawPoints = remember(dataArray) {
        dataArray.mapNotNull {
            val obj   = it.jsonObject
            val step  = obj["step"]?.jsonPrimitive?.floatOrNull
            val value = obj["value"]?.jsonPrimitive?.floatOrNull
            if (step != null && value != null) ChartPoint(step, value) else null
        }.sortedBy { it.step }
    }
    if (rawPoints.isEmpty()) return

    /* ── EMA smoothing ── */
    val smoothedPoints = remember(rawPoints, smoothing) {
        if (smoothing <= 0f) return@remember rawPoints
        val out = mutableListOf<ChartPoint>()
        var ema = rawPoints.first().value
        for (p in rawPoints) {
            ema = ema * smoothing + (1f - smoothing) * p.value
            out += ChartPoint(p.step, ema)
        }
        out
    }

    /* ── Full data bounds (includes outliers) ── */
    val fullBounds = remember(rawPoints) {
        val sortedY = rawPoints.map { it.value }.sorted()
        Viewport(
            xMin = rawPoints.minOf { it.step },
            xMax = rawPoints.maxOf { it.step },
            yMin = sortedY.first(),
            yMax = sortedY.last(),
        )
    }

    /* ── Initial viewport: Y axis clips outliers ── */
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

    /* ── State ── */
    var viewport by remember(initialViewport) { mutableStateOf(initialViewport) }

    /* ── Fling animatables (data-space units) ── */
    val scope     = rememberCoroutineScope()
    val flingAnimX = remember { Animatable(0f) }
    val flingAnimY = remember { Animatable(0f) }
    var flingJob  by remember { mutableStateOf<Job?>(null) }

    /* ── Stable drawing resources ── */
    val textMeasurer = rememberTextMeasurer()
    val labelStyle   = TextStyle(fontSize = 9.sp, color = Color.Gray.copy(alpha = 0.7f))
    val gridColor    = Color.Gray.copy(alpha = 0.12f)
    val axisColor    = Color.Gray.copy(alpha = 0.3f)

    /* ──────────────────────────────────────────────────────────
       Gesture handler
       ─────────────────────────────────────────────────────────
       calculateCentroid / calculatePan / calculateZoom are
       extensions on PointerEvent, so we call them on `event`.
       VelocityTracker.calculateVelocity() in KMP returns a
       Velocity whose components are accessed via .x / .y
       (same as on Android — no destructuring needed).
    ────────────────────────────────────────────────────────── */
    val gestureModifier = modifier.pointerInput(fullBounds) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false)
            flingJob?.cancel()

            val velTracker  = VelocityTracker()
            velTracker.addPosition(down.uptimeMillis, down.position)

            var prevCentroid = down.position
            var pointerCount = 1

            do {
                val event   = awaitPointerEvent()
                val pressed = event.changes.filter { it.pressed }
                if (pressed.isEmpty()) break

                pointerCount = pressed.size

                // ── All three helpers are called on `event` (PointerEvent) ──
                val centroid = event.calculateCentroid(useCurrent = true)
                val pan      = event.calculatePan()
                val zoom     = event.calculateZoom()

                // Consume moved pointers so scroll parents don't interfere
                pressed.forEach { c -> if (c.positionChanged()) c.consume() }

                // Only track velocity for single-finger pan
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

                // Guard against degenerate canvas sizes
                if (plotW <= 0f || plotH <= 0f) continue

                /* Zoom anchored on the centroid in data space */
                val zoomSafe  = zoom.coerceIn(0.5f, 2f)
                val dataX = vp.xMin + ((centroid.x - leftPad) / plotW) * vp.xRange
                val dataY = vp.yMax -  (centroid.y            / plotH) * vp.yRange

                val newXRange = (vp.xRange / zoomSafe)
                    .coerceIn(vp.xRange * 0.001f, fullBounds.xRange * 20f)
                val newYRange = (vp.yRange / zoomSafe)
                    .coerceIn(vp.yRange * 0.001f, fullBounds.yRange * 20f)

                /* Pan in data space */
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

            /* ── Fling (single-finger only) ── */
            if (pointerCount == 1) {
                val velocity = velTracker.calculateVelocity()
                // VelocityTracker.calculateVelocity() returns Velocity
                // with .x and .y in px/s on both Android and desktop KMP.
                val velXPx = velocity.x
                val velYPx = velocity.y

                val vp      = viewport
                val canvasW = size.width.toFloat()
                val canvasH = size.height.toFloat()
                val plotW   = canvasW - 52f
                val plotH   = canvasH - 28f

                // Convert px/s → data-units/s
                val dataVx = -(velXPx / plotW) * vp.xRange
                val dataVy =  (velYPx / plotH) * vp.yRange

                flingJob = scope.launch {
                    // Animate two independent decay curves
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

                // Drive viewport from the running fling values
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

    Canvas(modifier = gestureModifier) {
        val vp    = viewport
        val w     = size.width
        val h     = size.height
        val leftPad   = 52f
        val bottomPad = 28f
        val plotW = w - leftPad
        val plotH = h - bottomPad

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
            val rawPath    = buildPath(rawPoints,      vp) { s, v -> dataToScreen(s, v) }
            val smoothPath = buildPath(smoothedPoints, vp) { s, v -> dataToScreen(s, v) }

            drawPath(rawPath,    color.copy(alpha = 0.20f), style = Stroke(width = 1.5f))
            drawPath(smoothPath, color,                     style = Stroke(width = 3f))

            smoothedPoints.lastOrNull()?.let { last ->
                val pt = dataToScreen(last.step, last.value)
                drawCircle(color,       radius = 5f,   center = pt)
                drawCircle(Color.White, radius = 2.5f, center = pt)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid + axis labels  (KMP-safe: TextMeasurer, no android.graphics)
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

    drawLine(axisColor, Offset(leftPad, 0f),    Offset(leftPad, plotH), strokeWidth = 1f)
    drawLine(axisColor, Offset(leftPad, plotH), Offset(totalW,  plotH), strokeWidth = 1f)
}

// ─────────────────────────────────────────────────────────────────────────────
// Path builder
// ─────────────────────────────────────────────────────────────────────────────

private fun buildPath(
    points: List<ChartPoint>,
    vp: Viewport,
    toScreen: (Float, Float) -> Offset,
): Path = Path().apply {
    var started = false
    val xLo = vp.xMin - vp.xRange
    val xHi = vp.xMax + vp.xRange
    for (p in points) {
        val pt = toScreen(p.step, p.value)
        if (p.step in xLo..xHi) {
            if (!started) { moveTo(pt.x, pt.y); started = true }
            else           lineTo(pt.x, pt.y)
        } else if (started) {
            lineTo(pt.x, pt.y) // one extra point to avoid hard clip edge
        }
    }
}
package com.acite.axlranko.util

import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max

/** Brush radius bounds shared by the toolbar slider and the Alt+wheel shortcut. */
const val BRUSH_RADIUS_MIN = 2f
const val BRUSH_RADIUS_MAX = 128f

private const val BRUSH_WHEEL_STEP_PX = 2f

/** Outer radius where the brush reaches zero weight, and the inner radius that still carries full weight. */
data class BrushRadii(val outer: Float, val inner: Float)

/**
 * Radii the brush actually covers; the cursor draws them as a dashed rim and a solid core.
 * `featherOfDiameter` is the soft rim as a fraction of 2*radius, so it never exceeds the radius.
 */
fun brushRadii(radius: Float, featherOfDiameter: Float): BrushRadii {
    val r = radius.coerceAtLeast(0f)
    val featherPx = (2f * r * featherOfDiameter.coerceIn(0f, 1f)).coerceAtMost(r)
    return BrushRadii(outer = r, inner = r - featherPx)
}

/** One Alt+wheel notch per unit of [steps], clamped to the toolbar slider range. */
fun nudgeBrushRadius(current: Float, steps: Int): Float =
    (current + steps * BRUSH_WHEEL_STEP_PX).coerceIn(BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX)

/** Smooth falloff: 1 at the core, 0 at the outer radius. */
fun brushFalloff(distance: Float, radius: Float, featherOfDiameter: Float): Float {
    val (outer, inner) = brushRadii(radius, featherOfDiameter)
    if (outer <= 0f || distance >= outer) return 0f
    if (distance <= inner) return 1f
    val t = (distance - inner) / (outer - inner).coerceAtLeast(1e-6f)
    val s = t * t * (3f - 2f * t)
    return 1f - s
}

/**
 * Sample a segment at ~[spacing] along its length, excluding the start point
 * (already stamped) and including the end.
 */
fun interpolateStroke(
    x0: Float,
    y0: Float,
    x1: Float,
    y1: Float,
    spacing: Float,
): List<Pair<Float, Float>> {
    val dist = hypot(x1 - x0, y1 - y0)
    val step = spacing.coerceAtLeast(0.5f)
    if (dist < 1e-4f) return emptyList()
    if (dist <= step) return listOf(x1 to y1)
    val n = ceil(dist / step).toInt().coerceAtLeast(1)
    return (1..n).map { i ->
        val t = i.toFloat() / n
        x0 + (x1 - x0) * t to y0 + (y1 - y0) * t
    }
}

fun strokeSpacing(radius: Float): Float = max(0.5f, radius * 0.15f)

/** Max-blend toward white, min-blend toward black. Peak = 255 * strength * falloff. */
fun blendMaskSample(old: Int, falloff: Float, strength: Float, erase: Boolean): Int {
    val peak = (255f * strength.coerceIn(0f, 1f) * falloff.coerceIn(0f, 1f))
    return if (erase) {
        minOf(old, (255f - peak).toInt().coerceIn(0, 255))
    } else {
        maxOf(old, peak.toInt().coerceIn(0, 255))
    }
}

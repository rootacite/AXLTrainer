package com.acite.axlranko.pages.components

import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.ui.theme.SparkSlopeHigh
import com.acite.axlranko.ui.theme.SparkSlopeLow
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The checkpoint card's Avg Loss strip, without a window: a fixed 0.85 EMA, a slope color from
 * (255, 79, 0) to (0, 255, 127), and a marker whose x is the step's place on the whole series.
 */
class CheckpointLossSparkTest {

    @Test
    fun smoothingFollowsTheChartRecurrenceAt085() {
        // Unsorted on purpose. The first application of the recurrence to the first value is a no-op.
        val out = smoothAvgLoss(
            listOf(
                MetricPoint(step = 2, value = 0f),
                MetricPoint(step = 0, value = 1f),
                MetricPoint(step = 1, value = Float.NaN),
                MetricPoint(step = 1, value = 2f),
            ),
        )
        assertEquals(listOf(0f, 1f, 2f), out.map { it.step })
        // The complement is `1 - smoothing`, the same term ChartCard uses, which is not the literal 0.15f.
        val smoothing = 0.85f
        var ema = 1f
        ema = ema * smoothing + (1f - smoothing) * 1f
        assertEquals(ema, out[0].value)
        ema = ema * smoothing + (1f - smoothing) * 2f
        assertEquals(ema, out[1].value)
        ema = ema * smoothing + (1f - smoothing) * 0f
        assertEquals(ema, out[2].value)
        assertEquals(1f, out[0].value, 1e-5f)
        assertEquals(1.15f, out[1].value, 1e-4f)
        assertEquals(0.9775f, out[2].value, 1e-4f)
    }

    @Test
    fun theSteepestRiseIsRedAndTheSteepestFallIsGreen() {
        val rising = slopeColor(slope = 2f, scale = 2f)
        val falling = slopeColor(slope = -2f, scale = 2f)
        assertEquals(SparkSlopeHigh, rising)
        assertEquals(SparkSlopeLow, falling)
        assertEquals(255, channel(rising.red))
        assertEquals(79, channel(rising.green))
        assertEquals(0, channel(rising.blue))
        assertEquals(0, channel(falling.red))
        assertEquals(255, channel(falling.green))
        assertEquals(127, channel(falling.blue))
    }

    @Test
    fun aHigherSlopeIsCloserToRedThanALowerOne() {
        val higher = slopeColor(slope = 1f, scale = 2f)
        val lower = slopeColor(slope = -1f, scale = 2f)
        // t = 0.75 and t = 0.25 of the way from (0, 255, 127) to (255, 79, 0).
        assertEquals(191, channel(higher.red))
        assertEquals(123, channel(higher.green))
        assertEquals(32, channel(higher.blue))
        assertEquals(64, channel(lower.red))
        assertEquals(211, channel(lower.green))
        assertEquals(95, channel(lower.blue))
        assertTrue(channel(higher.red) > channel(lower.red))
    }

    @Test
    fun aZeroSlopeAndAFlatSeriesAreTheMidpoint() {
        val zero = slopeColor(slope = 0f, scale = 4f)
        val flat = slopeColor(slope = 0f, scale = 0f)
        assertEquals(128, channel(zero.red))
        assertEquals(167, channel(zero.green))
        assertEquals(64, channel(zero.blue))
        assertEquals(zero, flat)
    }

    @Test
    fun theMarkerSitsAtTheStepsFractionOfTheSeries() {
        val points = listOf(SparkPoint(0f, 0f), SparkPoint(100f, 100f))
        val layout = layoutCheckpointSpark(
            points = points,
            step = 25f,
            width = 100f,
            height = 40f,
            padLeft = 10f,
            padTop = 2f,
            padRight = 10f,
            padBottom = 2f,
            scale = 1f,
        )
        // Plot is [10, 90]. A quarter of the way across the series is x = 30.
        val markerX = checkNotNull(layout.markerX)
        val markerY = checkNotNull(layout.markerY)
        assertEquals(30f, markerX, 0.01f)
        val segment = layout.segments.single()
        val t = (markerX - segment.x0) / (segment.x1 - segment.x0)
        val onLine = segment.y0 + t * (segment.y1 - segment.y0)
        assertEquals(onLine, markerY, 0.01f)
    }

    @Test
    fun aStepOutsideTheSeriesClampsToTheNearerEdge() {
        val points = listOf(SparkPoint(0f, 1f), SparkPoint(100f, 0f))
        val before = layoutCheckpointSpark(points, step = -10f, width = 100f, height = 40f, padLeft = 10f, padTop = 2f, padRight = 10f, padBottom = 2f, scale = 1f)
        val after = layoutCheckpointSpark(points, step = 150f, width = 100f, height = 40f, padLeft = 10f, padTop = 2f, padRight = 10f, padBottom = 2f, scale = 1f)
        assertEquals(before.segments.first().x0, before.markerX!!, 0.01f)
        assertEquals(before.segments.first().y0, before.markerY!!, 0.01f)
        assertEquals(after.segments.last().x1, after.markerX!!, 0.01f)
        assertEquals(after.segments.last().y1, after.markerY!!, 0.01f)
    }

    @Test
    fun fewerThanTwoPointsDrawNothing() {
        val layout = layoutCheckpointSpark(
            points = listOf(SparkPoint(0f, 1f)),
            step = 0f,
            width = 100f,
            height = 40f,
            padLeft = 4f,
            padTop = 2f,
            padRight = 4f,
            padBottom = 2f,
            scale = 1f,
        )
        assertTrue(layout.segments.isEmpty())
        assertNull(layout.markerX)
        assertNull(layout.markerY)
    }

    @Test
    fun aFlatSeriesDrawsALevelLineInTheMidpointColor() {
        val points = listOf(SparkPoint(0f, 1f), SparkPoint(10f, 1f))
        val layout = layoutCheckpointSpark(
            points = points,
            step = 5f,
            width = 100f,
            height = 40f,
            padLeft = 0f,
            padTop = 0f,
            padRight = 0f,
            padBottom = 0f,
            scale = slopeScale(segmentSlopes(points)),
        )
        val segment = layout.segments.single()
        assertEquals(20f, segment.y0, 0.01f)
        assertEquals(20f, segment.y1, 0.01f)
        assertEquals(128, channel(segment.color.red))
        assertEquals(167, channel(segment.color.green))
        assertEquals(64, channel(segment.color.blue))
    }

    @Test
    fun theSharedScaleKeepsAMildSegmentOffTheEndpoint() {
        val points = listOf(SparkPoint(0f, 0f), SparkPoint(1f, 1f))
        val layout = layoutCheckpointSpark(points, step = 0f, width = 80f, height = 20f, padLeft = 0f, padTop = 0f, padRight = 0f, padBottom = 0f, scale = 10f)
        assertEquals(slopeColor(1f, 10f), layout.segments.single().color)
        val marked = layoutCheckpointSpark(
            points,
            step = 0f,
            width = 80f,
            height = 20f,
            padLeft = 0f,
            padTop = 0f,
            padRight = 0f,
            padBottom = 0f,
            scale = 10f,
            markerSlope = -10f,
        )
        assertEquals(SparkSlopeLow, marked.markerColor)
    }

    @Test
    fun slopeAtAVertexIsTheCentralDifference() {
        val points = listOf(SparkPoint(0f, 0f), SparkPoint(10f, 1f), SparkPoint(20f, 0f))
        assertEquals(0.1f, slopeCovering(points, 5f), 1e-6f)
        assertEquals(-0.1f, slopeCovering(points, 15f), 1e-6f)
        assertEquals(0f, slopeCovering(points, 10f), 1e-6f)
        assertEquals(0.1f, slopeCovering(points, 0f), 1e-6f)
        assertEquals(-0.1f, slopeCovering(points, 20f), 1e-6f)
        assertEquals(0.1f, slopeCovering(points, -4f), 1e-6f)
    }

    @Test
    fun aSmoothedRiseAndFallTakeOppositeColors() {
        val smoothed = smoothAvgLoss(
            listOf(
                MetricPoint(0, 0.5f),
                MetricPoint(1, 0.5f),
                MetricPoint(2, 1.5f),
                MetricPoint(3, 1.5f),
                MetricPoint(4, 0.2f),
            ),
        )
        val slopes = segmentSlopes(smoothed)
        val rising = slopes.max()
        val falling = slopes.min()
        assertTrue(rising > 0f, "smoothed series never rises: ${slopes.toList()}")
        assertTrue(falling < 0f, "smoothed series never falls: ${slopes.toList()}")
        val scale = sparkColorScale(smoothed)
        val rise = slopeColor(rising, scale)
        val fall = slopeColor(falling, scale)
        assertTrue(channel(rise.red) > channel(fall.red))
        assertTrue(channel(fall.green) > channel(rise.green))
    }

    @Test
    fun theDrawnSegmentsOwnScaleSeparatesARiseFromAFall() {
        // Slope +0.2 then −0.05. Colored against their own max |slope|, the rise is the red endpoint.
        val points = listOf(SparkPoint(0f, 0f), SparkPoint(10f, 2f), SparkPoint(30f, 1f))
        val scale = sparkColorScale(points)
        assertEquals(0.2f, scale, 1e-6f)
        val layout = layoutCheckpointSpark(
            points = points,
            step = 10f,
            width = 100f,
            height = 40f,
            padLeft = 0f,
            padTop = 0f,
            padRight = 0f,
            padBottom = 0f,
            scale = scale,
        )
        assertEquals(SparkSlopeHigh, layout.segments[0].color)
        assertEquals(slopeColor(-0.05f, 0.2f), layout.segments[1].color)
        assertTrue(channel(layout.segments[1].color.green) > channel(layout.segments[1].color.red))
        // The old scale, a per-step spike far steeper than either drawn segment, pulls both toward
        // the green-reading midpoint.
        val washed = layoutCheckpointSpark(
            points = points,
            step = 10f,
            width = 100f,
            height = 40f,
            padLeft = 0f,
            padTop = 0f,
            padRight = 0f,
            padBottom = 0f,
            scale = 100f,
        )
        val ownGap = channel(layout.segments[0].color.red) - channel(layout.segments[1].color.red)
        val washedGap = channel(washed.segments[0].color.red) - channel(washed.segments[1].color.red)
        assertTrue(ownGap > 100, "own-scale red gap was $ownGap")
        assertTrue(washedGap < 20, "per-step scale still separated the segments by $washedGap")
    }

    @Test
    fun axesFrameThePlot() {
        val points = listOf(SparkPoint(0f, 0f), SparkPoint(100f, 1f))
        val layout = layoutCheckpointSpark(
            points = points,
            step = 50f,
            width = 200f,
            height = 80f,
            padLeft = 30f,
            padTop = 4f,
            padRight = 6f,
            padBottom = 16f,
            scale = 1f,
        )
        assertEquals(30f, layout.plotLeft, 0.01f)
        assertEquals(4f, layout.plotTop, 0.01f)
        assertEquals(194f, layout.plotRight, 0.01f)
        assertEquals(64f, layout.plotBottom, 0.01f)
        assertEquals(listOf("0", "50", "100"), layout.xTicks.map { it.label })
        assertEquals(layout.plotLeft, layout.xTicks.first().at, 0.01f)
        assertEquals((layout.plotLeft + layout.plotRight) / 2f, layout.markerX!!, 0.01f)
        assertEquals(3, layout.yTicks.size)
    }

    @Test
    fun theWindowIsAtMostTwoCadencesEitherSideOfTheStep() {
        val points = (0..1000 step 10).map { SparkPoint(it.toFloat(), 1f) }
        val around = windowSpark(points, step = 500f, saveEveryNSteps = 100)
        assertEquals(300f, around.first().step)
        assertEquals(700f, around.last().step)
        val nearStart = windowSpark(points, step = 50f, saveEveryNSteps = 100)
        assertEquals(0f, nearStart.first().step)
        assertEquals(250f, nearStart.last().step)
        assertTrue(windowSpark(points, step = 500f, saveEveryNSteps = 0).isEmpty())
    }

    @Test
    fun downsampleKeepsBothEndsAndTheCheckpointStep() {
        val points = List(10) { SparkPoint(it.toFloat(), it.toFloat()) }
        val drawn = downsampleSpark(points, maxPoints = 4, keepStep = 3f)
        assertTrue(drawn.size <= 4)
        assertEquals(0f, drawn.first().step)
        assertEquals(9f, drawn.last().step)
        assertTrue(drawn.any { it.step == 3f })
    }

    private fun channel(component: Float): Int = (component * 255f).roundToInt()
}

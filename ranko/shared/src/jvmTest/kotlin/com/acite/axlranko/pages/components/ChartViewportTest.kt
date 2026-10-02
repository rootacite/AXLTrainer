package com.acite.axlranko.pages.components

import com.acite.axlranko.model.MetricPoint
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The chart's windowing and axis maths, which a step chart applies without a window: the newest
 * [DEFAULT_STEP_SPAN] steps by default, and one vertical scale per curve on a dual-axis chart.
 */
class ChartViewportTest {

    @Test
    fun shortHistoryKeepsTheWholeRange() {
        val (lo, hi) = initialXWindow(0f, 400f, DEFAULT_STEP_SPAN)
        assertEquals(0f, lo)
        assertEquals(400f, hi)
    }

    @Test
    fun longHistoryOpensOnTheNewestSteps() {
        val (lo, hi) = initialXWindow(0f, 5_000f, DEFAULT_STEP_SPAN)
        assertEquals(5_000f, hi)
        assertEquals(5_000f - DEFAULT_STEP_SPAN, lo)
    }

    @Test
    fun aSpanAsWideAsTheDataChangesNothing() {
        val (lo, hi) = initialXWindow(120f, 1_320f, DEFAULT_STEP_SPAN)
        assertEquals(120f, lo)
        assertEquals(1_320f, hi)
    }

    @Test
    fun chartsWithoutAStepAxisKeepTheWholeRange() {
        val (lo, hi) = initialXWindow(0f, 9_000f, null)
        assertEquals(0f, lo)
        assertEquals(9_000f, hi)
    }

    @Test
    fun degenerateOnEmptyRangesSurvive() {
        val (lo, hi) = initialXWindow(300f, 300f, DEFAULT_STEP_SPAN)
        assertEquals(300f, lo)
        assertEquals(300f, hi)
        val (slo, shi) = initialXWindow(10f, 20f, 0f)
        assertEquals(10f, slo)
        assertEquals(20f, shi)
    }

    @Test
    fun aDomainIsFittedToTheWindowNotTheWholeHistory() {
        val points = listOf(
            MetricPoint(step = 10, value = 9f),
            MetricPoint(step = 100, value = 1f),
            MetricPoint(step = 1_500, value = 2f),
            MetricPoint(step = 1_900, value = 3f),
        )
        val domain = windowDomain(points, 1_200f, 2_000f)
        assertEquals(1.95f, domain!!.first, 1e-5f)
        assertEquals(3.05f, domain.second, 1e-5f)
    }

    @Test
    fun aFlatSeriesStillGetsAReadableSpan() {
        val points = listOf(MetricPoint(step = 10, value = 7e-6f), MetricPoint(step = 900, value = 7e-6f))
        val (lo, hi) = windowDomain(points, 0f, 1_000f)!!
        assertTrue(lo < 7e-6f && hi > 7e-6f, "flat series collapsed to [$lo, $hi]")
        assertTrue(hi - lo > 1e-9f, "flat series kept a zero span")
    }

    @Test
    fun aSeriesWithNoPointsHasNoDomain() {
        assertNull(windowDomain(emptyList(), 0f, 1_000f))
    }

    @Test
    fun axisTicksFollowEachDomainAndTheViewport() {
        // Full viewport: the fraction maps straight onto the domain.
        assertEquals(1e-5f, axisTickValue(0f, 100f, 0f, 1e-5f, 2e-5f), 1e-12f)
        assertEquals(2e-5f, axisTickValue(0f, 100f, 1f, 1e-5f, 2e-5f), 1e-12f)
        assertEquals(1.5e-5f, axisTickValue(0f, 100f, 0.5f, 1e-5f, 2e-5f), 1e-12f)

        // Zoomed into the top half: the tick at the top edge still names the domain's high end.
        assertEquals(1.5e-5f, axisTickValue(50f, 50f, 0f, 1e-5f, 2e-5f), 1e-12f)
        assertEquals(2e-5f, axisTickValue(50f, 50f, 1f, 1e-5f, 2e-5f), 1e-12f)
    }
}

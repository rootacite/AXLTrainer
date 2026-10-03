package com.acite.axlranko.pages.components

import com.acite.axlranko.model.DashboardUiState
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
        val hiExpected = 120f + DEFAULT_STEP_SPAN
        val (lo, hi) = initialXWindow(120f, hiExpected, DEFAULT_STEP_SPAN)
        assertEquals(120f, lo)
        assertEquals(hiExpected, hi)
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

    @Test
    fun theShippedWindowIs800StepsAndMatchesTheDashboardDefault() {
        assertEquals(800f, DEFAULT_STEP_SPAN)
        assertEquals(0.15f, DEFAULT_OUTLIER_CLIP)
        assertEquals(1.2f, DEFAULT_SMOOTH_EXTRA_DP)
        assertEquals(DEFAULT_STEP_SPAN, DashboardUiState().stepSpan)
        assertEquals(DEFAULT_OUTLIER_CLIP, DashboardUiState().outlierClip)
        assertEquals(DEFAULT_SMOOTH_EXTRA_DP, DashboardUiState().smoothExtraDp)
        // 1.2 dp at density 2 is 2.4 px on top of the raw stroke.
        assertEquals(3.9f, smoothStrokeWidthPx(1.5f, 1.2f, 2f))
    }

    @Test
    fun theNewestPointStaysInsideAClippedRange() {
        val descending = (10 downTo 1).map { it.toFloat() }
        val fitted = fittedYRange(descending, outlierClip = 0.15f, include = listOf(1f))!!
        assertTrue(fitted.first <= 1f, "newest point ${fitted.first} was clipped below 1")
        assertTrue(fitted.second >= 1f)
    }

    @Test
    fun anOldSpikeOutsideThePercentileStaysOutside() {
        val values = List(20) { 2f } + 100f
        val fitted = fittedYRange(values, outlierClip = 0.15f, include = listOf(2f))!!
        assertTrue(fitted.second < 100f, "spike pulled the range to ${fitted.second}")
    }

    @Test
    fun theRangeFollowsTheSmoothedValuesItIsGiven() {
        val fitted = fittedYRange(listOf(1.9f, 2f, 2.1f), outlierClip = 0f, include = listOf(2.1f))!!
        assertTrue(fitted.second < 10f, "smoothed range reached ${fitted.second}")
        assertTrue(fitted.first <= 1.9f && fitted.second >= 2.1f)
    }

    @Test
    fun anEmptyFitIsNull() {
        assertNull(fittedYRange(emptyList(), 0.15f, emptyList()))
    }

    @Test
    fun epochLinesNameTheEpochToTheRightAndSkipTheOrigin() {
        val marks = epochBoundaries(stepsPerEpoch = 10, lastStep = 25f)
        assertEquals(listOf(EpochMark(10f, 2), EpochMark(20f, 3)), marks)
        assertTrue(marks.none { it.step == 0f })
    }

    @Test
    fun aBoundaryOnTheLastStepAndAMissingLengthDrawNothing() {
        assertTrue(epochBoundaries(10, 20f).none { it.step == 20f })
        assertEquals(emptyList(), epochBoundaries(null, 25f))
        assertEquals(emptyList(), epochBoundaries(0, 25f))
        assertEquals(emptyList(), epochBoundaries(10, 0f))
    }
}

package com.acite.axlranko.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * The Training section's step arithmetic: `ceil` once for the batches and once for grad
 * accumulation, exactly as `trainer/setup.py` does it, plus the form-reading overload.
 */
class StepEstimateTest {

    @Test
    fun aBatchThatDividesEvenlyAddsNothing() {
        val estimate = estimatedSteps(samplesPerEpoch = 100, batchSize = 4, gradAccumulation = 1, epochs = 2)
        assertEquals(25, estimate?.batchesPerEpoch)
        assertEquals(25, estimate?.stepsPerEpoch)
        assertEquals(50, estimate?.totalSteps)
        assertEquals(100, estimate?.samplesPerEpoch)
    }

    @Test
    fun aRemainderIsABatchAndGradAccumulationCeilsItsOwnWay() {
        // 100 samples at batch 8 is 12.5 -> 13 batches; 13 batches at GA 4 is 3.25 -> 4 steps.
        val estimate = estimatedSteps(100, 8, 4, 3)
        assertEquals(13, estimate?.batchesPerEpoch)
        assertEquals(4, estimate?.stepsPerEpoch)
        assertEquals(12, estimate?.totalSteps)
    }

    @Test
    fun gradAccumulationLargerThanTheEpochStillTakesOneStep() {
        val estimate = estimatedSteps(samplesPerEpoch = 3, batchSize = 1, gradAccumulation = 16, epochs = 2)
        assertEquals(3, estimate?.batchesPerEpoch)
        assertEquals(1, estimate?.stepsPerEpoch)
        assertEquals(2, estimate?.totalSteps)
    }

    @Test
    fun theFormOverloadReadsTheStringsTheUserTyped() {
        val estimate = estimatedSteps(2_400, batchSizeText = " 6 ", gradAccumulationText = "2", epochsText = "10")
        assertEquals(400, estimate?.batchesPerEpoch)
        assertEquals(200, estimate?.stepsPerEpoch)
        assertEquals(2_000, estimate?.totalSteps)
    }

    @Test
    fun unusableNumbersEstimateNothing() {
        assertNull(estimatedSteps(0, 4, 1, 10))
        assertNull(estimatedSteps(-5, 4, 1, 10))
        assertNull(estimatedSteps(100, 0, 1, 10))
        assertNull(estimatedSteps(100, 4, 0, 10))
        assertNull(estimatedSteps(100, 4, 1, 0))
        assertNull(estimatedSteps(100, "many", "1", "10"))
        assertNull(estimatedSteps(100, "", "1", "10"))
        assertNull(estimatedSteps(100, "4", "2", ""))
    }

    @Test
    fun stepCountsAreGroupedForDisplay() {
        assertEquals("0", formatStepCount(0))
        assertEquals("999", formatStepCount(999))
        assertEquals("1,240", formatStepCount(1_240))
        assertEquals("12,000", formatStepCount(12_000))
        assertEquals("1,234,567", formatStepCount(1_234_567))
    }

    @Test
    fun theEstimateCarriesTheEpochFigureItWasBuiltFrom() {
        val estimate = estimatedSteps(samplesPerEpoch = 3_100, batchSize = 2, gradAccumulation = 5, epochs = 20)
        assertEquals(3_100, estimate?.samplesPerEpoch)
        assertEquals(1_550, estimate?.batchesPerEpoch)
        assertEquals(310, estimate?.stepsPerEpoch)
        assertEquals(6_200, estimate?.totalSteps)
    }
}

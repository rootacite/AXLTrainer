package com.acite.axlranko.model

/**
 * What one run's step count is expected to be, from the dataset's image count and the form's own
 * epoch / batch / GA values.
 *
 * The arithmetic mirrors `trainer/setup.py` and `BucketBatchSampler`, with one deliberate
 * difference: it counts `ceil(samples / batch)` batches for the epoch instead of one `ceil` per
 * aspect-ratio bucket. Bucketing can only add batches (a bucket with one image still yields one
 * batch), so [batchesPerEpoch] is a floor and the trainer's own count is that or a little higher —
 * with `enable_bucket = false` the two agree exactly.
 */
data class StepEstimate(
    /** Images one epoch draws, repeats included. */
    val samplesPerEpoch: Int,
    /** Whole batches one epoch is expected to yield. */
    val batchesPerEpoch: Int,
    /** Optimizer steps one epoch, i.e. `ceil(batches / GA)` (never below one). */
    val stepsPerEpoch: Int,
    /** `stepsPerEpoch × epochs`. */
    val totalSteps: Int,
)

/**
 * The estimate, or null when the form does not hold usable numbers (a blank field, a zero, or a
 * negative). Nothing is guessed from a partial form: an empty line is better than a wrong one.
 */
fun estimatedSteps(
    samplesPerEpoch: Int,
    batchSize: Int,
    gradAccumulation: Int,
    epochs: Int,
): StepEstimate? {
    if (samplesPerEpoch <= 0 || batchSize < 1 || gradAccumulation < 1 || epochs < 1) return null
    val batches = ceilDiv(samplesPerEpoch, batchSize)
    val stepsPerEpoch = ceilDiv(batches, gradAccumulation).coerceAtLeast(1)
    return StepEstimate(
        samplesPerEpoch = samplesPerEpoch,
        batchesPerEpoch = batches,
        stepsPerEpoch = stepsPerEpoch,
        totalSteps = stepsPerEpoch * epochs,
    )
}

/** The estimate for one epoch count, read out of the form as the strings the user typed. */
fun estimatedSteps(
    samplesPerEpoch: Int,
    batchSizeText: String,
    gradAccumulationText: String,
    epochsText: String,
): StepEstimate? {
    val batch = batchSizeText.trim().toIntOrNull() ?: return null
    val grad = gradAccumulationText.trim().toIntOrNull() ?: return null
    val epochs = epochsText.trim().toIntOrNull() ?: return null
    return estimatedSteps(samplesPerEpoch, batch, grad, epochs)
}

private fun ceilDiv(value: Int, divisor: Int): Int =
    if (value <= 0) 0 else (value + divisor - 1) / divisor

/** `1,240` — thousands separated, for the line under the Training fields. */
fun formatStepCount(value: Int): String {
    val digits = value.toString()
    if (digits.length <= 3) return digits
    return digits.reversed().chunked(3).joinToString(",").reversed()
}

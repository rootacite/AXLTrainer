package com.acite.axlranko.model

/** The category a caption takes unless the operator picks others. */
const val DEFAULT_TAGGER_CATEGORY = "general"

/** One calibrated threshold from the tagger's own config, drawn under the slider. */
data class TaggerMark(
    val key: String,
    val calibrated: Float,
    /** The slider sits above this category's calibrated value, so the slider is what applies. */
    val overridden: Boolean,
)

/**
 * The marks the Auto-tag card draws: the model's per-category calibrated thresholds, for the
 * categories the run will actually write. The slider is a floor over them, so a mark whose value
 * is below the slider is the one the slider has taken over.
 */
fun taggerThresholdMarks(
    selected: Set<String>,
    categories: List<TaggerCategoryInfo>,
    slider: Float,
): List<TaggerMark> = categories
    .filter { it.key in selected }
    .map { TaggerMark(it.key, it.calibrated, slider > it.calibrated) }

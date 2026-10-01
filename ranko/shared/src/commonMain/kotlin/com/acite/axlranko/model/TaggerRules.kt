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

/**
 * The tags a partial pass should add. One per `,`, `;` or line break — a tag may hold spaces
 * (`hair between eyes`), so those are kept; surrounding whitespace is dropped and duplicates are
 * folded, first spelling winning.
 */
fun parseOnlyTags(text: String): List<String> {
    val tags = mutableListOf<String>()
    for (part in text.split(',', ';', '\n')) {
        val tag = part.trim()
        if (tag.isNotEmpty() && tags.none { it.equals(tag, ignoreCase = true) }) tags += tag
    }
    return tags
}

package com.acite.axlranko.prompt

/** One step of the wizard; [applies] reproduces the CLI's page gating. */
data class WizardPage(
    val key: String,
    val applies: (PromptSpec) -> Boolean,
)

/**
 * The wizard's page order and navigation rules, ported from `WIZARD_PAGES` / `_previous_page`.
 *
 * The UI is free to render a page however it likes; this only says which pages exist and where
 * "back" lands when the current spec skips some of them.
 */
object WizardModel {

    val PAGES: List<WizardPage> = listOf(
        WizardPage("character") { true },
        WizardPage("mode") { true },
        WizardPage("exposure") { true },
        WizardPage("clothing") { !it.nudeOnly },
        WizardPage("chest") { true },
        WizardPage("belly") { true },
        WizardPage("face") { true },
        WizardPage("scene") { true },
        WizardPage("family") { it.mode == PromptMode.Sex },
        WizardPage("ratio") { it.mode == PromptMode.Sex },
        WizardPage("stages") { it.mode == PromptMode.Sex },
        WizardPage("pose") { true },
        WizardPage("count") { true },
    )

    val PAGE_BY_KEY: Map<String, WizardPage> = PAGES.associateBy { it.key }

    /** Where "back" lands from [index]: the nearest earlier page that still applies. */
    fun previousPage(index: Int, spec: PromptSpec): Int {
        for (candidate in index - 1 downTo 0) {
            if (PAGES[candidate].applies(spec)) return candidate
        }
        return 0
    }

    /** The step to move to when leaving [index] forward, skipping pages the spec does not use. */
    fun nextPage(index: Int, spec: PromptSpec): Int {
        var candidate = index + 1
        while (candidate < PAGES.size && !PAGES[candidate].applies(spec)) candidate++
        return candidate
    }

    /**
     * Switching mode resets the fields whose value is mode-specific, like `_page_mode`: exposure,
     * chest, belly and the whole face page.
     */
    fun applyModeChange(spec: PromptSpec, mode: PromptMode) {
        if (mode == spec.mode) return
        spec.mode = mode
        spec.exposure = defaultExposure(mode)
        spec.chest = CHEST_DEFAULT.getValue(mode)
        spec.belly = BELLY_DEFAULT.getValue(mode)
        spec.face = defaultFace()
    }

    fun needsSfwExposureWarning(spec: PromptSpec): Boolean =
        spec.mode == PromptMode.Sfw && spec.exposure.any { HIGH_EXPOSURE.contains(it) }

    /** SFW with a `breasts out` / `nipples` chest preference; the CLI warns and continues. */
    fun needsSfwBodyWarning(spec: PromptSpec): Boolean =
        spec.mode == PromptMode.Sfw && (spec.chest == "breasts_out" || spec.chest == "nipples")

    fun needsSfwFaceWarning(spec: PromptSpec): Boolean =
        needsSfwFaceWarning(spec.mode, selectedFaceTags(spec.face))

    /** The clothing rows a given exposure setting can offer, in bucket order and without repeats. */
    fun allowedClothing(matrix: PromptMatrix, exposure: List<String>): List<MatrixEntry> {
        val seen = mutableSetOf<List<String>>()
        val out = mutableListOf<MatrixEntry>()
        exposure.forEach { bucket ->
            if (bucket == "nude") return@forEach
            PromptGenerator.clothingPoolForBucket(matrix, bucket).forEach { item ->
                if (seen.add(item.key)) out.add(item)
            }
        }
        return out
    }

    /** Chest/belly/face values a fresh or reset wizard shows; used when a page opens. */
    fun clampCount(value: Int): Int = value.coerceIn(PromptLimits.COUNT_MIN, PromptLimits.COUNT_MAX)

    fun chestIndex(chest: String): Int = CHEST_LEVELS.indexOf(chest).coerceAtLeast(0)

    fun bellyIndex(belly: String): Int = BELLY_LEVELS.indexOf(belly).coerceAtLeast(0)

    fun modeIndex(mode: PromptMode): Int = PromptMode.ORDER.indexOf(mode).coerceAtLeast(0)
}

package com.acite.axlranko.pages

import com.acite.axlranko.model.DEFAULT_TAGGER_CATEGORY
import com.acite.axlranko.model.TaggerCategoryInfo
import com.acite.axlranko.model.UtilsUiState
import com.acite.axlranko.model.taggerThresholdMarks
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private val CATEGORIES = listOf(
    TaggerCategoryInfo(key = "general", count = 15043, calibrated = 0.17f),
    TaggerCategoryInfo(key = "character", count = 8308, calibrated = 0.27f),
    TaggerCategoryInfo(key = "rating", count = 4, calibrated = 0.41f),
)

class TaggerCategoryRulesTest {

    @Test
    fun theDefaultSelectionIsGeneralOnly() {
        assertEquals(setOf(DEFAULT_TAGGER_CATEGORY), UtilsUiState().tagCategories)
        assertEquals("general", DEFAULT_TAGGER_CATEGORY)
        assertEquals(null, UtilsUiState().taggerInfo)
        assertFalse(UtilsUiState().taggerInfoLoaded)
    }

    @Test
    fun theDefaultSelectionConsultsNothingElse() {
        // The state a fresh card shows: one category, no model answer yet, so no marks.
        val state = UtilsUiState()
        assertEquals(emptyList(), taggerThresholdMarks(state.tagCategories, emptyList(), 0.35f))
    }

    @Test
    fun marksOnlyCoverTheSelectedCategories() {
        val marks = taggerThresholdMarks(setOf("general", "rating"), CATEGORIES, 0.0f)
        assertEquals(listOf("general", "rating"), marks.map { it.key })
        assertEquals(0.17f, marks[0].calibrated)
        assertEquals(0.41f, marks[1].calibrated)
    }

    @Test
    fun aMarkBelowTheSliderIsTheOneTheSliderTookOver() {
        val marks = taggerThresholdMarks(setOf("general", "rating"), CATEGORIES, 0.35f)
        // general's own 0.17 is under the slider, so 0.35 is what applies to it; rating keeps 0.41.
        assertTrue(marks.first { it.key == "general" }.overridden)
        assertFalse(marks.first { it.key == "rating" }.overridden)
    }

    @Test
    fun aSliderUnderTheCalibrationLeavesEveryMarkAlone() {
        val marks = taggerThresholdMarks(setOf("general", "character", "rating"), CATEGORIES, 0.1f)
        assertEquals(listOf(false, false, false), marks.map { it.overridden })
    }

    @Test
    fun anEmptySelectionDrawsNothing() {
        assertEquals(emptyList(), taggerThresholdMarks(emptySet(), CATEGORIES, 0.35f))
    }
}

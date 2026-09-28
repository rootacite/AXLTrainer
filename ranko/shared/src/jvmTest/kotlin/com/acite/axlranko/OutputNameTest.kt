package com.acite.axlranko

import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.OUTPUT_NAME_HINT
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.isValidOutputName
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * A run name becomes a run id (`safe_name` + stamp) and the artifact directory names, so the form
 * refuses what the trainer refuses at startup — the same rule as
 * `trainer/runs.py::validate_output_name`: a space or a slash would make the id (`re_in_…`)
 * disagree with the directories (`re in_samples`).
 */
class OutputNameTest {

    private fun form(name: String) = TrainingConfigForm(
        pretrainedModelNameOrPath = "/models/sdxl",
        outputDir = "/out",
        loggingDir = "/logs",
        outputName = name,
    )

    @Test
    fun plainNamesPass() {
        listOf("rein", "towa_2", "babara-v2", "a.b").forEach {
            assertTrue(isValidOutputName(it), it)
            assertNull(form(it).validate()["output_name"], it)
        }
    }

    @Test
    fun spacesAreRefused() {
        listOf("re in", "rein ", " rein").forEach {
            assertFalse(isValidOutputName(it), it)
            assertEquals(OUTPUT_NAME_HINT, form(it).validate()["output_name"], it)
        }
    }

    @Test
    fun slashesAndOtherCharactersAreRefused() {
        listOf("re/in", "re\\in", "rein:v2", "rein·v2").forEach {
            assertFalse(isValidOutputName(it), it)
            assertEquals(OUTPUT_NAME_HINT, form(it).validate()["output_name"], it)
        }
    }

    @Test
    fun anEmptyNameIsRequiredRatherThanInvalid() {
        assertEquals("Required", form("").validate()["output_name"])
    }

    @Test
    fun theErrorBelongsToTheEnvironmentSection() {
        assertTrue(ConfigSection.Environment.owns("output_name"))
    }
}

package com.acite.axlranko

import com.acite.axlranko.data.TomlIntegerLiterals
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * ktoml refuses an integer literal for a Kotlin `Double`, so a hand-edited `key = 0` under a
 * double-typed key has to be read as `0.0` for Ranko to start at all.
 */
class TomlIntegerLiteralsTest {

    @Test
    fun doubleKeysComeFromTheConfigModel() {
        val keys = TomlIntegerLiterals.doubleKeys()
        assertTrue("amdfq_vram_reserve_gib" in keys)
        assertTrue("learning_rate" in keys)
        assertTrue("guidance_scale" in keys, "the [[validation.samples]] row is a Double too")
        assertFalse("seed" in keys)
        assertFalse("epoch" in keys)
        assertFalse("output_name" in keys)
    }

    @Test
    fun integerLiteralUnderADoubleKeyGetsAFractionalPart() {
        assertEquals(
            "amdfq_vram_reserve_gib = 0.0",
            TomlIntegerLiterals.normalize("amdfq_vram_reserve_gib = 0"),
        )
        assertEquals("noise_offset = 1.0", TomlIntegerLiterals.normalize("noise_offset = 1"))
        assertEquals("learning_rate = -2.0", TomlIntegerLiterals.normalize("learning_rate = -2"))
    }

    @Test
    fun indentationAndCommentsSurvive() {
        assertEquals(
            "  amdfq_vram_reserve_gib = 0.0  # 0 = off",
            TomlIntegerLiterals.normalize("  amdfq_vram_reserve_gib = 0  # 0 = off"),
        )
    }

    @Test
    fun integerKeysAndStringsAreLeftAlone() {
        val text = """
            [training]
            seed = 0
            epoch = 12
            output_name = "0"

            [environment]
            amdfq = "vmm"
        """.trimIndent()
        assertEquals(text, TomlIntegerLiterals.normalize(text))
    }

    @Test
    fun floatLiteralsAreLeftAlone() {
        val text = "min_snr_gamma = 5.0\nnoise_offset = 1e-2\nnetwork_dropout = 0.08"
        assertEquals(text, TomlIntegerLiterals.normalize(text))
    }

    @Test
    fun sampleSetRowsAreCovered() {
        val text = """
            [validation]
            guidance_scale = 6

            [[validation.samples]]
            name = "yuzu"
            guidance_scale = 0
            repeat = 1
        """.trimIndent()
        val expected = """
            [validation]
            guidance_scale = 6.0

            [[validation.samples]]
            name = "yuzu"
            guidance_scale = 0.0
            repeat = 1
        """.trimIndent()
        assertEquals(expected, TomlIntegerLiterals.normalize(text))
    }

    @Test
    fun theRestOfTheFileKeepsItsShape() {
        val text = "[environment]\n# a comment\namdfq_vram_reserve_gib = 0\n\n[train]\nx = 1\n"
        assertEquals(
            "[environment]\n# a comment\namdfq_vram_reserve_gib = 0.0\n\n[train]\nx = 1\n",
            TomlIntegerLiterals.normalize(text),
        )
    }
}

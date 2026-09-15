package com.acite.axlranko.util

import com.acite.axlranko.model.CheckpointItem
import kotlin.test.Test
import kotlin.test.assertEquals

class CheckpointFormatTest {

    private fun checkpoint(
        dir: String = "lllj_s003050",
        filename: String = "lllj.safetensors",
        outputName: String = "lllj",
        final: Boolean = false,
    ) = CheckpointItem(
        path = "/out/lllj_20260915_134334/$dir/$filename",
        runId = "lllj_20260915_134334",
        dir = dir,
        filename = filename,
        step = 3050,
        final = final,
        sizeBytes = 237_489_712,
        networkDim = 64,
        networkAlpha = 32,
        outputName = outputName,
    )

    @Test
    fun saveNameUsesTheStepDirectorySoSavedFilesDoNotCollide() {
        assertEquals("lllj_s003050.safetensors", checkpointSaveName(checkpoint()))
        assertEquals("lllj_final.safetensors", checkpointSaveName(checkpoint(dir = "lllj_final")))
    }

    @Test
    fun saveNameFallsBackWhenTheDirectoryIsMissing() {
        assertEquals(
            "lllj.safetensors",
            checkpointSaveName(checkpoint(dir = "", filename = "lllj.safetensors")),
        )
        assertEquals(
            "lllj.safetensors",
            checkpointSaveName(checkpoint(dir = "", filename = "lllj.safetensors", outputName = "lllj")),
        )
        assertEquals(
            "lora.safetensors",
            checkpointSaveName(checkpoint(dir = "", filename = "", outputName = "")),
        )
    }

    @Test
    fun saveNameNeverDoublesTheExtension() {
        assertEquals("lllj_s003050.safetensors", checkpointSaveName(checkpoint(dir = "lllj_s003050.safetensors")))
    }

    @Test
    fun extensionIsAddedOnlyForBareNames() {
        assertEquals("my_lora.safetensors", ensureSafetensorsExtension("my_lora"))
        assertEquals("my_lora.pt", ensureSafetensorsExtension("my_lora.pt"))
        assertEquals("my.lora", ensureSafetensorsExtension("my.lora"))
        assertEquals(".safetensors", ensureSafetensorsExtension(""))
    }

    @Test
    fun subtitleReportsRankAlphaSizeAndFinal() {
        assertEquals(
            "lllj_20260915_134334 · step 3050 · r64/α32 · 226.4 MB",
            checkpointSubtitle(checkpoint()),
        )
        assertEquals(
            "lllj_20260915_134334 · step 3050 · r64/α32 · 226.4 MB · final",
            checkpointSubtitle(checkpoint(dir = "lllj_final", final = true)),
        )
    }
}

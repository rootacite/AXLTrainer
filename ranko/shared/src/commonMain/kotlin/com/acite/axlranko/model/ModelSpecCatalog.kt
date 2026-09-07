package com.acite.axlranko.model

data class ModelSpecPreset(
    val baseModelVersion: String,
    val familyId: String,
    val trainable: Boolean,
    val label: String,
    val architecture: String,
    val implementation: String,
    val saiModelSpec: String
)

object ModelSpecCatalog {
    val presets: List<ModelSpecPreset> = listOf(
        ModelSpecPreset(
            baseModelVersion = "sdxl_base_v1-0",
            familyId = "sdxl",
            trainable = true,
            label = "SDXL 1.0",
            architecture = "stable-diffusion-xl-v1-base/lora",
            implementation = "https://github.com/Stability-AI/generative-models",
            saiModelSpec = "1.0.0"
        ),
        ModelSpecPreset(
            baseModelVersion = "sd3.5-large",
            familyId = "sd3.5",
            trainable = false,
            label = "SD 3.5 Large",
            architecture = "stable-diffusion-v3-5-large/lora",
            implementation = "https://github.com/Stability-AI/sd3.5",
            saiModelSpec = "1.0.0"
        )
    )

    val versions: List<String> = presets.map { it.baseModelVersion }

    fun byVersion(version: String): ModelSpecPreset? =
        presets.find { it.baseModelVersion == version }
}

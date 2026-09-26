package com.acite.axlranko.data

import com.akuleshov7.ktoml.Toml
import com.akuleshov7.ktoml.TomlInputConfig

object ConfigImporter {
    /**
     * Decode a config document that is already in memory, with the same tolerance as the loader:
     * unknown keys are ignored (a file may be newer than this model) and a bare integer literal
     * stands in for a double. Null means "not a config".
     */
    fun parseConfig(text: String): AxlTrainerConfig? = try {
        Toml(
            inputConfig = TomlInputConfig(
                ignoreUnknownNames = true,
                allowEmptyValues = true
            )
        ).decodeFromString(AxlTrainerConfig.serializer(), TomlIntegerLiterals.normalize(text))
    } catch (_: Exception) {
        null
    }
}

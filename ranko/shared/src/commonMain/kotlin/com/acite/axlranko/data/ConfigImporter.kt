package com.acite.axlranko.data

import com.akuleshov7.ktoml.Toml
import com.akuleshov7.ktoml.TomlInputConfig
import okio.Path
import okio.Path.Companion.toPath
expect fun getAppExecutionPath(): String
expect fun loadTrainerConfig(tomlPath: Path): AxlTrainerConfig?
expect fun saveTrainerConfigPatched(
    tomlPath: Path,
    sectionValues: Map<String, Map<String, String>>,
    arrayBlocks: Map<String, List<Map<String, String>>> = emptyMap()
): Result<Unit>

public object ConfigImporter {
    fun getConfig(): AxlTrainerConfig {
        val p = getConfigPath()
            ?: error("Could not locate config.toml (searched upward from the executable and working directory)")
        return loadTrainerConfig(p.toPath())
            ?: error("Failed to parse config.toml at $p")
    }

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
    } catch (e: Exception) {
        null
    }

    fun getConfigPath(): String? = TrainerRepo.configToml()?.absolutePath

    fun loadConfigOrNull(): Pair<String, AxlTrainerConfig>? {
        val path = getConfigPath() ?: return null
        val config = loadTrainerConfig(path.toPath()) ?: return null
        return path to config
    }

    fun savePatched(
        sectionValues: Map<String, Map<String, String>>,
        arrayBlocks: Map<String, List<Map<String, String>>> = emptyMap()
    ): Result<Unit> {
        val path = getConfigPath()
            ?: return Result.failure(IllegalStateException("Could not locate config.toml"))
        return saveTrainerConfigPatched(path.toPath(), sectionValues, arrayBlocks)
    }
}
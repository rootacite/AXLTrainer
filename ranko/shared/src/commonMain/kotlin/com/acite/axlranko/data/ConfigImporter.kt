package com.acite.axlranko.data

import okio.Path
import okio.Path.Companion.toPath
expect fun getAppExecutionPath(): String
expect fun loadTrainerConfig(tomlPath: Path): AxlTrainerConfig?
expect fun saveTrainerConfigPatched(
    tomlPath: Path,
    sectionValues: Map<String, Map<String, String>>
): Result<Unit>

public object ConfigImporter {
    fun getConfig(): AxlTrainerConfig {
        val p = getConfigPath()
            ?: error("Could not locate trainer/config.toml (searched upward from the executable and working directory)")
        return loadTrainerConfig(p.toPath())
            ?: error("Failed to parse trainer/config.toml at $p")
    }

    fun getConfigPath(): String? = TrainerRepo.configToml()?.absolutePath

    fun loadConfigOrNull(): Pair<String, AxlTrainerConfig>? {
        val path = getConfigPath() ?: return null
        val config = loadTrainerConfig(path.toPath()) ?: return null
        return path to config
    }

    fun savePatched(sectionValues: Map<String, Map<String, String>>): Result<Unit> {
        val path = getConfigPath()
            ?: return Result.failure(IllegalStateException("Could not locate trainer/config.toml"))
        return saveTrainerConfigPatched(path.toPath(), sectionValues)
    }
}
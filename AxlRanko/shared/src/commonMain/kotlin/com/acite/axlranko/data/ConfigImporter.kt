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
        val p = getConfigPath()!!
        val c = loadTrainerConfig(p.toPath())!!

        return c
    }

    fun getConfigPath(): String? = findConfigPath(getAppExecutionPath())

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

    private fun findConfigPath(startPathStr: String): String? {
        if (startPathStr.isEmpty()) return null

        var currentPath: Path? = startPathStr.toPath()
        val targetDirName = "AxlRanko"

        while (currentPath != null) {
            if (currentPath.name == targetDirName) {
                break
            }
            currentPath = currentPath.parent
        }

        if (currentPath == null) return null

        val targetPath = currentPath
            .parent
            ?.resolve("trainer")
            ?.resolve("config.toml")

        return targetPath?.normalized()?.toString()
    }
}
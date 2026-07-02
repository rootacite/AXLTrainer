package com.acite.axlranko.data

import okio.Path
import okio.Path.Companion.toPath
expect fun getAppExecutionPath(): String
expect fun loadTrainerConfig(tomlPath: Path): AxlTrainerConfig?

public object ConfigImporter {
    fun getConfig(): AxlTrainerConfig {
        val p = findConfigPath(getAppExecutionPath())!!
        val c = loadTrainerConfig(p.toPath())!!

        return c
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
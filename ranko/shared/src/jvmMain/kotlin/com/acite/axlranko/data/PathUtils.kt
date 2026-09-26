package com.acite.axlranko.data

import java.io.File
import okio.Path

fun getAppExecutionPath(): String {
    return try {
        val codeSource = ::getAppExecutionPath::class.java.protectionDomain.codeSource
        val jarFile = File(codeSource.location.toURI())
        jarFile.parentFile?.absolutePath ?: ""
    } catch (e: Exception) {
        e.printStackTrace()
        ""
    }
}

/** Test helper: decode a config file from a path. Production reads go through IPC `config_get`. */
fun loadTrainerConfig(tomlPath: Path): AxlTrainerConfig? {
    return try {
        ConfigImporter.parseConfig(File(tomlPath.toString()).readText())
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

/** Test helper: patch a config file in place. Production writes go through IPC `config_save`. */
fun saveTrainerConfigPatched(
    tomlPath: Path,
    sectionValues: Map<String, Map<String, String>>,
    arrayBlocks: Map<String, List<Map<String, String>>> = emptyMap(),
): Result<Unit> {
    return try {
        val file = File(tomlPath.toString())
        if (!file.exists()) {
            return Result.failure(IllegalStateException("Config file does not exist: $tomlPath"))
        }
        var patched = file.readText()
        for ((section, blocks) in arrayBlocks) {
            patched = TomlDocumentPatcher.replaceArrayOfTables(patched, section, blocks)
        }
        patched = TomlDocumentPatcher.apply(patched, sectionValues)
        file.writeText(patched)
        Result.success(Unit)
    } catch (e: Exception) {
        e.printStackTrace()
        Result.failure(e)
    }
}

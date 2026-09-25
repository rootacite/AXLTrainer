package com.acite.axlranko.data

import com.akuleshov7.ktoml.Toml
import com.akuleshov7.ktoml.TomlInputConfig
import java.io.File
import com.akuleshov7.ktoml.file.TomlFileReader
import okio.Path

actual fun getAppExecutionPath(): String {
    return try {
        val codeSource = ::getAppExecutionPath::class.java.protectionDomain.codeSource
        val jarFile = File(codeSource.location.toURI())

        jarFile.parentFile?.absolutePath ?: ""
    } catch (e: Exception) {
        e.printStackTrace()
        ""
    }
}

actual fun loadTrainerConfig(tomlPath: Path): AxlTrainerConfig? {
    return try {
        val mt = Toml(
            inputConfig = TomlInputConfig(
                ignoreUnknownNames = true,
                allowEmptyValues = true
            )
        )
        val tomlString = java.io.File(tomlPath.toString()).readText()
        mt.decodeFromString(AxlTrainerConfig.serializer(), TomlIntegerLiterals.normalize(tomlString))
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

actual fun saveTrainerConfigPatched(
    tomlPath: Path,
    sectionValues: Map<String, Map<String, String>>,
    arrayBlocks: Map<String, List<Map<String, String>>>
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
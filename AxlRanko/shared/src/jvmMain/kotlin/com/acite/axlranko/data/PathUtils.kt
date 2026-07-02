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
        mt.decodeFromString(AxlTrainerConfig.serializer(), tomlString)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}
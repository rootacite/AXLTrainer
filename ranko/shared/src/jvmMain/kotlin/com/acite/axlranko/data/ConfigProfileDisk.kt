package com.acite.axlranko.data

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Direct-disk profile helpers for unit tests. Production Chromatrix uses IPC `profile_*`.
 */
object ConfigProfileDisk {
    private const val EXTENSION = "toml"

    fun dir(root: File): File = File(root, ConfigProfileStore.DIRECTORY_NAME)

    fun list(dir: File): List<ConfigProfile> =
        dir.listFiles()
            ?.filter { it.isFile && it.extension.equals(EXTENSION, ignoreCase = true) }
            ?.map {
                ConfigProfile(
                    name = it.nameWithoutExtension,
                    modified = it.lastModified(),
                    size = it.length(),
                )
            }
            ?.sortedBy { it.name.lowercase() }
            ?: emptyList()

    fun findByName(dir: File, name: String): ConfigProfile? =
        list(dir).firstOrNull { it.name.equals(name.trim(), ignoreCase = true) }

    fun save(dir: File, name: String, text: String, overwrite: Boolean): Result<File> {
        val trimmed = name.trim()
        ConfigProfileStore.validateName(trimmed)?.let { return Result.failure(IllegalArgumentException(it)) }
        val existing = dir.listFiles()?.firstOrNull {
            it.isFile && it.extension.equals(EXTENSION, ignoreCase = true) &&
                it.nameWithoutExtension.equals(trimmed, ignoreCase = true)
        }
        if (existing != null && !overwrite) {
            return Result.failure(IllegalStateException("A profile named \"$trimmed\" already exists"))
        }
        return try {
            dir.mkdirs()
            val target = existing ?: File(dir, "$trimmed.$EXTENSION")
            val temporary = File(dir, "${target.name}.tmp")
            temporary.writeText(text, Charsets.UTF_8)
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            Result.success(target)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun delete(dir: File, path: String): Result<Unit> {
        return try {
            val file = File(path)
            if (!file.exists()) {
                Result.failure(IllegalStateException("The profile is already gone: $path"))
            } else if (file.absoluteFile.parentFile?.canonicalFile != dir.canonicalFile) {
                Result.failure(IllegalArgumentException("Not a profile file: $path"))
            } else if (file.delete()) {
                Result.success(Unit)
            } else {
                Result.failure(IllegalStateException("Could not delete $path"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

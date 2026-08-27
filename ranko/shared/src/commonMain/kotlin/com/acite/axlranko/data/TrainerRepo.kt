package com.acite.axlranko.data

import java.io.File

/**
 * Locates the trainer repo root (the directory that contains `api.py` and/or `trainer/config.toml`).
 */
object TrainerRepo {
    fun findRoot(): File? {
        val starts = buildList {
            val exec = getAppExecutionPath()
            if (exec.isNotBlank()) add(File(exec))
            val cwd = System.getProperty("user.dir").orEmpty()
            if (cwd.isNotBlank()) add(File(cwd))
        }
        for (start in starts) {
            var current: File? = start.absoluteFile
            while (current != null) {
                val hasApi = File(current, "api.py").isFile
                val hasConfig = File(File(current, "trainer"), "config.toml").isFile
                if (hasApi || hasConfig) return current
                current = current.parentFile
            }
        }
        return null
    }

    fun configToml(): File? {
        val file = File(findRoot() ?: return null, "trainer${File.separator}config.toml")
        return file.takeIf { it.isFile }
    }
}

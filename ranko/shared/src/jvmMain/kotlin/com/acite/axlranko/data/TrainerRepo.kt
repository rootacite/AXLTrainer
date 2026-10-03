package com.acite.axlranko.data

import java.io.File

/**
 * Locates the trainer repo root (the directory that contains `api.py` and the root `config.toml`).
 *
 * Desktop bootstrap only: after the WebSocket is up, Chromatrix commonMain does not walk the disk.
 */
object TrainerRepo {
    /**
     * A directory is the repo root when it holds `api.py`, or when it holds both `config.toml` and
     * the `trainer/` package. A lone `config.toml` is not enough: any unrelated directory on the way
     * up could own one, and the app would then edit a stranger's file.
     */
    internal fun looksLikeRepoRoot(dir: File): Boolean =
        File(dir, "api.py").isFile ||
            (File(dir, "config.toml").isFile && File(dir, "trainer").isDirectory)

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
                if (looksLikeRepoRoot(current)) return current
                current = current.parentFile
            }
        }
        return null
    }

    fun configToml(): File? {
        val file = File(findRoot() ?: return null, "config.toml")
        return file.takeIf { it.isFile }
    }
}

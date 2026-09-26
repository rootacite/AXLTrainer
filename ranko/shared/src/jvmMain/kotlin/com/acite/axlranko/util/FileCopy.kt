package com.acite.axlranko.util

import java.io.File

private const val COPY_CHUNK_BYTES = 4 * 1024 * 1024

/**
 * Copies [source] onto [target], reporting progress as a 0..1 fraction. Refuses to copy a file onto
 * itself: LoRA checkpoints are the only copy of hours of work, and `copyTo(overwrite = true)` would
 * truncate the source.
 */
fun copyFileWithProgress(source: File, target: File, onProgress: (Float) -> Unit = {}): Long {
    if (!source.isFile) throw IllegalArgumentException("not a file: ${source.absolutePath}")
    if (target.canonicalPath == source.canonicalPath) {
        throw IllegalArgumentException("destination is the same file as the source: ${target.absolutePath}")
    }
    target.parentFile?.mkdirs()

    val total = source.length()
    if (total == 0L) {
        target.outputStream().use { }
        onProgress(1f)
        return 0L
    }

    var copied = 0L
    source.inputStream().use { input ->
        target.outputStream().use { output ->
            val buffer = ByteArray(COPY_CHUNK_BYTES)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                output.write(buffer, 0, read)
                copied += read
                onProgress((copied.toDouble() / total).toFloat().coerceIn(0f, 1f))
            }
        }
    }
    onProgress(1f)
    return copied
}

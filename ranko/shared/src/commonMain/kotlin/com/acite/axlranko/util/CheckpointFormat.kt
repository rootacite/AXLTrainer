package com.acite.axlranko.util

import com.acite.axlranko.model.CheckpointItem

/** One-line summary of a LoRA checkpoint, shared by the Utils picker and the Dashboard panel. */
fun checkpointSubtitle(checkpoint: CheckpointItem): String {
    val parts = mutableListOf<String>()
    parts += checkpoint.runId.ifBlank { "run" }
    checkpoint.step?.let { parts += "step $it" }
    val rank = checkpoint.networkDim
    val alpha = checkpoint.networkAlpha
    if (rank != null && alpha != null) parts += "r$rank/α$alpha"
    parts += formatBytes(checkpoint.sizeBytes)
    if (checkpoint.final) parts += "final"
    return parts.joinToString(" · ")
}

/**
 * File name to offer for "Save As": the artifact directory (`lllj_s003050`) so saving several
 * checkpoints of one run into the same folder does not collide.
 */
fun checkpointSaveName(checkpoint: CheckpointItem): String {
    val stem = checkpoint.dir
        .ifBlank { checkpoint.filename }
        .removeSuffix(".safetensors")
        .ifBlank { checkpoint.outputName }
        .ifBlank { "lora" }
    return "$stem.safetensors"
}

/** Adds the LoRA extension when the user typed a bare name into the save dialog. */
fun ensureSafetensorsExtension(fileName: String): String =
    if (fileName.substringAfterLast('.', "").isEmpty()) "$fileName.safetensors" else fileName

fun formatBytes(bytes: Long): String {
    if (bytes <= 0) return "0 B"
    val units = listOf("B", "KB", "MB", "GB")
    var value = bytes.toDouble()
    var index = 0
    while (value >= 1024 && index < units.lastIndex) {
        value /= 1024
        index++
    }
    return if (index == 0) "$bytes B" else "${(value * 10).toInt() / 10.0} ${units[index]}"
}

package com.acite.axlranko.pages.components

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.SuggestionChipDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.util.formatFourDecimals
import kotlin.math.roundToInt

@Composable
fun TrainControlCard(
    status: TrainStatus,
    commandInFlight: Boolean,
    pendingCommand: String? = null,
    outputDir: String,
    loggingDir: String,
    onStart: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit,
    onReset: (deleteWeights: Boolean) -> Unit,
) {
    var confirmStop by remember { mutableStateOf(false) }
    var confirmReset by remember { mutableStateOf(false) }
    var deleteWeights by remember { mutableStateOf(false) }
    val actual = status.status
    val terminal = actual in setOf("idle", "finished", "error")
    val phase = when {
        terminal -> actual
        pendingCommand == "pause" && actual != "pausing" && actual != "paused" -> "pausing"
        pendingCommand == "resume" && actual != "resuming" -> "resuming"
        pendingCommand == "stop" && actual != "stopping" -> "stopping"
        else -> actual
    }
    val swapping = !terminal && (phase == "pausing" || phase == "resuming")
    val runLocked = commandInFlight || swapping ||
        (!terminal && pendingCommand != null && pendingCommand != "stop") ||
        phase == "starting"
    val running = phase == "encoding" || phase == "training" || phase == "sampling"
    val canStart = !commandInFlight && terminal && !status.alive
    val canPause = !runLocked && running
    val canResume = !runLocked && phase == "paused"
    val canStop = !runLocked && (running || phase == "paused")
    val canReset = !commandInFlight && actual in setOf("idle", "finished", "error", "stopping")
    val runName = status.outputName?.takeIf { it.isNotBlank() } ?: "—"

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (swapping) MaterialTheme.colorScheme.tertiaryContainer
            else MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    StatusChip(phase)
                    if (swapping) {
                        StatusChip(
                            status = if (phase == "resuming") "gpu-in" else "gpu-out",
                            labelOverride = gpuChipLabel(status, resuming = phase == "resuming"),
                        )
                    }
                    if (swapping || phase == "starting" || phase == "stopping") {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                        )
                    }
                    Text(
                        text = runName,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    CompactMetric("PID", status.pid?.toString() ?: "—")
                    CompactMetric("Elapsed", formatElapsed(status.startedAt, phase))
                    CompactMetric("Alive", if (status.alive) "yes" else "no")
                }
            }

            status.detail?.takeIf { it.isNotBlank() }?.let { detail ->
                Text(
                    text = detail.replace('_', ' '),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            status.error?.takeIf { it.isNotBlank() }?.let { message ->
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }

            PhaseBar(
                label = "Latent encode",
                current = status.encoding.current,
                total = status.encoding.total,
                detail = encodingDetail(status),
                active = phase == "encoding",
                pulsing = swapping,
            )
            PhaseBar(
                label = "Training",
                current = status.training.step,
                total = status.training.totalSteps,
                detail = trainingDetail(status),
                active = phase == "training" || (phase == "paused" && status.pausedFrom == "training"),
                pulsing = swapping,
            )
            PhaseBar(
                label = "Sampling",
                current = samplingCurrent(status),
                total = samplingTotal(status),
                detail = samplingDetail(status),
                active = phase == "sampling" || status.sampling.active,
                pulsing = swapping,
            )

            Row(
                modifier = Modifier.fillMaxWidth().then(if (runLocked) Modifier.alpha(0.45f) else Modifier),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ControlButton(
                    text = "Start",
                    icon = Icons.Default.PlayArrow,
                    enabled = canStart,
                    onClick = onStart,
                    modifier = Modifier.weight(1f),
                    tonal = true,
                )
                ControlButton(
                    text = if (phase == "pausing") "Pausing" else "Pause",
                    icon = Icons.Default.Pause,
                    enabled = canPause,
                    inFlight = phase == "pausing",
                    onClick = onPause,
                    modifier = Modifier.weight(1f),
                )
                ControlButton(
                    text = if (phase == "resuming") "Resuming" else "Resume",
                    icon = Icons.Default.PlayArrow,
                    enabled = canResume,
                    inFlight = phase == "resuming",
                    onClick = onResume,
                    modifier = Modifier.weight(1f),
                    tonal = true,
                )
                Button(
                    onClick = { confirmStop = true },
                    enabled = canStop,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                        contentColor = MaterialTheme.colorScheme.onError,
                    )
                ) {
                    Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Early Stop")
                }
                ControlButton(
                    text = "Reset",
                    icon = Icons.Default.RestartAlt,
                    enabled = canReset,
                    onClick = {
                        deleteWeights = false
                        confirmReset = true
                    },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }

    if (confirmStop) {
        AlertDialog(
            onDismissRequest = { confirmStop = false },
            title = { Text("Stop training early?") },
            text = {
                Text(
                    when (phase) {
                        "encoding", "paused" ->
                            if (status.pausedFrom == "encoding" || phase == "encoding") {
                                "Encoding will stop. No LoRA checkpoint will be saved."
                            } else {
                                "Training will stop after the current safe point and a checkpoint will be saved if a step has completed."
                            }
                        "sampling" ->
                            "The current sample will be dropped. Remaining repeats and epochs will be skipped. The last checkpoint is kept."
                        else ->
                            "Training will stop at the next safe point. A LoRA checkpoint is saved if this step has no file yet."
                    }
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmStop = false
                        onStop()
                    }
                ) { Text("Stop") }
            },
            dismissButton = {
                TextButton(onClick = { confirmStop = false }) { Text("Cancel") }
            }
        )
    }

    if (confirmReset) {
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Reset this run?") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "Clears the Finished / Error state so Start can launch a new run. " +
                            "Sample images and TensorBoard logs for \"$runName\" will be deleted."
                    )
                    Text(
                        "Logs: $loggingDir/$runName\nSamples: $outputDir/${runName}_samples",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(checked = deleteWeights, onCheckedChange = { deleteWeights = it })
                        Text("Also delete LoRA checkpoints under $outputDir/${runName}_*")
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        val wipe = deleteWeights
                        confirmReset = false
                        onReset(wipe)
                    }
                ) { Text("Reset") }
            },
            dismissButton = {
                TextButton(onClick = { confirmReset = false }) { Text("Cancel") }
            }
        )
    }
}

@Composable
private fun StatusChip(status: String, labelOverride: String? = null) {
    val (label, color) = statusStyle(status)
    SuggestionChip(
        onClick = {},
        enabled = false,
        label = {
            Text(
                text = labelOverride ?: label,
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.labelMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        },
        colors = SuggestionChipDefaults.suggestionChipColors(
            disabledContainerColor = color.copy(alpha = 0.18f),
            disabledLabelColor = color,
        )
    )
}

@Composable
private fun PhaseBar(
    label: String,
    current: Int,
    total: Int,
    detail: String,
    active: Boolean,
    pulsing: Boolean = false,
    accent: Color = MaterialTheme.colorScheme.primary,
) {
    val fraction = if (total > 0) (current.toFloat() / total.toFloat()).coerceIn(0f, 1f) else 0f
    val pulse = if (pulsing) rememberPulseAlpha() else 1f
    Column(verticalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.alpha(if (pulsing) pulse else 1f)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = if (active) FontWeight.Bold else FontWeight.Medium,
                color = if (active) accent else MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = if (total > 0) "$current / $total" else detail.ifBlank { "idle" },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        if (pulsing) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = if (active) accent else MaterialTheme.colorScheme.outline,
                trackColor = MaterialTheme.colorScheme.surface,
                strokeCap = StrokeCap.Round,
            )
        } else {
            LinearProgressIndicator(
                progress = { if (total > 0) fraction else 0f },
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = if (active) accent else MaterialTheme.colorScheme.outline,
                trackColor = MaterialTheme.colorScheme.surface,
                strokeCap = StrokeCap.Round,
            )
        }
        if (detail.isNotBlank() && total > 0) {
            Text(
                text = detail,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun rememberPulseAlpha(): Float {
    val transition = rememberInfiniteTransition(label = "swap-pulse")
    val alpha by transition.animateFloat(
        initialValue = 0.35f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(700),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "swap-pulse-alpha",
    )
    return alpha
}

@Composable
private fun ControlButton(
    text: String,
    icon: ImageVector,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    inFlight: Boolean = false,
    tonal: Boolean = false,
) {
    val content: @Composable () -> Unit = {
        if (inFlight) {
            CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
        } else {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
        }
        Spacer(Modifier.width(6.dp))
        Text(text)
    }
    if (tonal) {
        FilledTonalButton(onClick = onClick, enabled = enabled, modifier = modifier) {
            content()
        }
    } else {
        OutlinedButton(onClick = onClick, enabled = enabled, modifier = modifier) {
            content()
        }
    }
}

private fun statusStyle(status: String): Pair<String, Color> {
    return when (status) {
        "starting" -> "Starting" to Color(0xFF0068C9)
        "encoding" -> "Encoding" to Color(0xFF0D9488)
        "training" -> "Training" to Color(0xFF0068C9)
        "sampling" -> "Sampling" to Color(0xFF7B61FF)
        "pausing" -> "Pausing" to Color(0xFFB45309)
        "paused" -> "Paused" to Color(0xFFB45309)
        "resuming" -> "Resuming" to Color(0xFFB45309)
        "stopping" -> "Stopping" to Color(0xFFFF4B4B)
        "finished" -> "Finished" to Color(0xFF0D9488)
        "error" -> "Error" to Color(0xFFFF4B4B)
        "gpu-out" -> "Offloading GPU" to Color(0xFFB45309)
        "gpu-in" -> "Reloading GPU" to Color(0xFFB45309)
        else -> "Idle" to Color(0xFF6B7280)
    }
}

private fun gpuChipLabel(status: TrainStatus, resuming: Boolean): String {
    val swap = status.swap
    val detail = swap?.detail?.takeIf { it.isNotBlank() }
    val counts = if (swap != null && swap.total > 0) "${swap.current}/${swap.total}" else null
    return when {
        detail != null && counts != null -> "$counts $detail"
        detail != null -> detail
        resuming -> "Reloading GPU"
        else -> "Offloading GPU"
    }
}

private fun encodingDetail(status: TrainStatus): String {
    val enc = status.encoding
    return when {
        enc.done -> "Done"
        enc.total > 0 -> "Images ${enc.current} / ${enc.total}"
        else -> "Idle"
    }
}

private fun trainingDetail(status: TrainStatus): String {
    val t = status.training
    if (t.totalSteps <= 0 && t.step <= 0) return "Idle"
    val loss = t.loss?.let { "  loss=${formatFourDecimals(it)}" } ?: ""
    val avg = t.avgLoss?.let { "  avg=${formatFourDecimals(it)}" } ?: ""
    return "epoch ${t.epoch}/${t.epochs}$loss$avg"
}

private fun samplingCurrent(status: TrainStatus): Int {
    val s = status.sampling
    if (s.denoiseSteps <= 0) return 0
    return s.repeat * s.denoiseSteps + s.denoiseStep
}

private fun samplingTotal(status: TrainStatus): Int {
    val s = status.sampling
    if (s.repeats <= 0 || s.denoiseSteps <= 0) return 0
    return s.repeats * s.denoiseSteps
}

private fun samplingDetail(status: TrainStatus): String {
    val s = status.sampling
    return when {
        s.active -> "image ${s.repeat + 1}/${s.repeats}  denoise ${s.denoiseStep}/${s.denoiseSteps}"
        s.repeats > 0 && s.globalStep > 0 -> "Last at step ${s.globalStep}"
        else -> "Idle"
    }
}

private fun formatElapsed(startedAt: Double?, status: String): String {
    if (startedAt == null || startedAt <= 0.0) return "—"
    if (status == "idle") return "—"
    val now = System.currentTimeMillis() / 1000.0
    val seconds = (now - startedAt).coerceAtLeast(0.0).roundToInt()
    val h = seconds / 3600
    val m = (seconds % 3600) / 60
    val s = seconds % 60
    return if (h > 0) "${h}h ${m}m" else "${m}m ${s}s"
}

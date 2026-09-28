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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.SuggestionChipDefaults
import androidx.compose.material3.Switch
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
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.TrainSettings
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.formatFourDecimals
import kotlin.math.roundToInt

@Composable
fun TrainControlCard(
    status: TrainStatus,
    commandInFlight: Boolean,
    controlsEnabled: Boolean = true,
    pendingCommand: String? = null,
    /** The run the page shows (pinned or followed); `null` leaves the card to the trainer's own. */
    shownRun: RunSummary? = null,
    outputDir: String,
    loggingDir: String,
    resumeFrom: String? = null,
    /** True while the card may retune the live run's cadence / sampling switch. */
    settingsEnabled: Boolean = false,
    settingsInFlight: Boolean = false,
    settingsError: String? = null,
    /** `[training]` values a run would start with, shown while no run is live. */
    configSaveEveryNSteps: Int? = null,
    configSamplingEnabled: Boolean? = null,
    onApplySettings: (saveEveryNSteps: Int?, samplingEnabled: Boolean?) -> Unit = { _, _ -> },
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
    val canStart = controlsEnabled && !commandInFlight && terminal && !status.alive
    val canPause = controlsEnabled && !runLocked && running
    val canResume = controlsEnabled && !runLocked && phase == "paused"
    val canStop = controlsEnabled && !runLocked && (running || phase == "paused")
    val canReset = controlsEnabled && !commandInFlight && actual in setOf("idle", "finished", "error", "stopping")
    val runName = controlRunName(shownRun, status)

    PorcelainCard {
        Column(
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

            val resumedFrom = status.resume
            val resumeLine = when {
                resumedFrom != null -> buildString {
                    append("Resumed from ")
                    append(resumedFrom.filename.ifBlank { resumedFrom.path })
                    resumedFrom.step?.let { append(" · checkpoint step $it") }
                    if (resumedFrom.loaded > 0) append(" · ${resumedFrom.loaded} tensors")
                }
                terminal && !resumeFrom.isNullOrBlank() ->
                    "Will resume from ${resumeFrom.trimEnd('/').substringAfterLast('/')}"
                else -> null
            }
            // The identity line names the run the page shows; the resume note belongs to the run
            // the trainer is on, so it stays off while a past one is pinned.
            val shownIsTrainers = shownRun == null || shownRun.runId == status.runId
            val runLine = listOfNotNull(
                controlRunId(shownRun, status)?.let { "run $it" },
                resumeLine.takeIf { shownIsTrainers },
            ).joinToString("   ")

            if (runLine.isNotEmpty()) {
                Text(
                    text = runLine,
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.accentPink,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            status.detail?.takeIf { it.isNotBlank() }?.let { detail ->
                Text(
                    text = detail.replace('_', ' '),
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.textDim
                )
            }
            status.error?.takeIf { it.isNotBlank() }?.let { message ->
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.qualityRed
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

            LiveSettingsRow(
                settings = status.settings,
                enabled = settingsEnabled,
                inFlight = settingsInFlight,
                error = settingsError,
                configSaveEveryNSteps = configSaveEveryNSteps,
                configSamplingEnabled = configSamplingEnabled,
                onApply = onApplySettings,
            )

            if (!controlsEnabled) {
                Text(
                    text = "Viewing a past run from the history list, so these controls are off. " +
                        "Pick \"Current run\" there to control training.",
                    style = MaterialTheme.typography.bodySmall,
                    color = rankoColors.qualityOrange,
                )
            }

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
                    emphasized = true,
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
                    emphasized = true,
                )
                ControlButton(
                    text = "Early Stop",
                    icon = Icons.Default.Stop,
                    enabled = canStop,
                    onClick = { confirmStop = true },
                    modifier = Modifier.weight(1f),
                    danger = true,
                )
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
        val runId = status.runId?.takeIf { it.isNotBlank() }
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Reset this run?") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        if (runId != null) {
                            "Clears the Finished / Error state so Start can launch a new run. " +
                                "Run \"$runId\" stays in the run history: its samples and " +
                                "TensorBoard logs are kept."
                        } else {
                            "Clears the Finished / Error state so Start can launch a new run. " +
                                "There is no run directory to delete."
                        }
                    )
                    if (runId != null) {
                        Text(
                            "Kept logs: $loggingDir/$runId\n" +
                                "Kept samples: $outputDir/$runId/${runName}_samples",
                            style = MaterialTheme.typography.bodySmall,
                            color = rankoColors.textDim
                        )
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(checked = deleteWeights, onCheckedChange = { deleteWeights = it })
                        Text(
                            if (runId != null) {
                                "Also delete LoRA checkpoints under $outputDir/$runId/${runName}_*"
                            } else {
                                "Also delete LoRA checkpoints (nothing to delete)"
                            }
                        )
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

/**
 * The live cadence and sampling switch of the run in progress. Both are requests: the trainer
 * adopts them at its next optimizer step, so the summary says what is in force and what is next.
 *
 * With no live run there is nothing to retune, so the row shows what the next Start would use
 * (`config.toml`) instead of the `settings` block — that block describes the run that published
 * it, and a runtime directory with no run on it carries the placeholder `0`, which means
 * "no checkpoints" rather than "the configuration says so".
 */
@Composable
private fun LiveSettingsRow(
    settings: TrainSettings,
    enabled: Boolean,
    inFlight: Boolean,
    error: String?,
    configSaveEveryNSteps: Int?,
    configSamplingEnabled: Boolean?,
    onApply: (saveEveryNSteps: Int?, samplingEnabled: Boolean?) -> Unit,
) {
    val colors = rankoColors
    var draft by remember(settings.saveEveryNSteps) {
        mutableStateOf(settings.saveEveryNSteps.toString())
    }
    val parsed = draft.trim().toIntOrNull()
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            if (enabled) {
                Text(
                    text = "Save every",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.textDim,
                )
                OutlinedTextField(
                    value = draft,
                    onValueChange = { text -> draft = text.filter { it.isDigit() }.take(6) },
                    singleLine = true,
                    modifier = Modifier.width(96.dp),
                    label = { Text("steps") },
                    textStyle = MaterialTheme.typography.bodySmall,
                    colors = rankoFieldColors(),
                    shape = rankoTokens.panel,
                )
                CapsuleButton(
                    text = "Apply",
                    onClick = { parsed?.let { onApply(it, null) } },
                    enabled = !inFlight && parsed != null && parsed >= 1,
                    compact = true,
                ) {
                    Text("Apply", fontWeight = FontWeight.SemiBold)
                }
                Spacer(Modifier.weight(1f))
                Text(
                    text = "Sampling",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.textDim,
                )
                Switch(
                    checked = settings.samplingEnabled,
                    enabled = !inFlight,
                    onCheckedChange = { on -> onApply(null, on) },
                )
            }
        }
        val summary = if (enabled) {
            settingsSummary(settings) + " · changes apply at the next step"
        } else {
            nextRunSummary(configSaveEveryNSteps, configSamplingEnabled)
        }
        summary?.let { text ->
            Text(
                text = text,
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
            )
        }
        error?.let { message ->
            Text(
                text = message,
                style = MaterialTheme.typography.labelSmall,
                color = colors.qualityRed,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

/**
 * `Save every 100 steps · next at step 200 · sampling on` for the run in progress, whose values
 * are what the trainer published.
 */
internal fun settingsSummary(settings: TrainSettings): String {
    val sampling = if (settings.samplingEnabled) "on" else "off"
    if (settings.saveEveryNSteps <= 0) return "Checkpoints off · sampling $sampling"
    val parts = mutableListOf("Save every ${settings.saveEveryNSteps} steps")
    if (settings.nextSaveStep > 0) parts += "next at step ${settings.nextSaveStep}"
    parts += "sampling $sampling"
    return parts.joinToString(" · ")
}

/**
 * `Next run · save every 100 steps · sampling on`, from `config.toml` — what pressing Start would
 * use. Null when the config has not loaded, so the row stays empty rather than guessing.
 */
internal fun nextRunSummary(saveEveryNSteps: Int?, samplingEnabled: Boolean?): String? {
    if (saveEveryNSteps == null && samplingEnabled == null) return null
    val parts = mutableListOf("Next run")
    when {
        saveEveryNSteps == null -> Unit
        saveEveryNSteps <= 0 -> parts += "no checkpoints"
        else -> parts += "save every $saveEveryNSteps steps"
    }
    samplingEnabled?.let { parts += "sampling ${if (it) "on" else "off"}" }
    return parts.joinToString(" · ")
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
    accent: Color = rankoColors.accentPink,
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
                color = if (active) accent else rankoColors.textDim
            )
            Text(
                text = if (total > 0) "$current / $total" else detail.ifBlank { "idle" },
                style = MaterialTheme.typography.labelSmall,
                color = rankoColors.textDim
            )
        }
        if (pulsing) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = if (active) accent else rankoColors.stroke,
                trackColor = rankoColors.bgApp,
                strokeCap = StrokeCap.Round,
            )
        } else {
            LinearProgressIndicator(
                progress = { if (total > 0) fraction else 0f },
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = if (active) accent else rankoColors.stroke,
                trackColor = rankoColors.bgApp,
                strokeCap = StrokeCap.Round,
            )
        }
        if (detail.isNotBlank() && total > 0) {
            Text(
                text = detail,
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.textDim
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
    emphasized: Boolean = false,
    danger: Boolean = false,
) {
    CapsuleButton(
        text = text,
        onClick = onClick,
        enabled = enabled,
        emphasized = emphasized,
        danger = danger,
        modifier = modifier,
        compact = true,
    ) {
        if (inFlight) {
            CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
        } else {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
        }
        Spacer(Modifier.width(6.dp))
        Text(text, fontWeight = FontWeight.SemiBold, maxLines = 1)
    }
}

/**
 * The name on the card: the shown run's, else the trainer's own. A history run carries the name
 * its run id was built from, so this works for a run with no logs and no samples.
 */
internal fun controlRunName(shown: RunSummary?, status: TrainStatus): String =
    shown?.outputName?.takeIf { it.isNotBlank() }
        ?: status.outputName?.takeIf { it.isNotBlank() }
        ?: "—"

/** The run id on the card, `null` when neither the shown run nor the trainer has one. */
internal fun controlRunId(shown: RunSummary?, status: TrainStatus): String? =
    shown?.runId?.takeIf { it.isNotBlank() } ?: status.runId?.takeIf { it.isNotBlank() }

@Composable
private fun statusStyle(status: String): Pair<String, Color> {
    val colors = rankoColors
    return when (status) {
        "starting" -> "Starting" to colors.accentBlue
        "encoding" -> "Encoding" to colors.accentPink
        "training" -> "Training" to colors.accentBlue
        "sampling" -> "Sampling" to colors.accentLilac
        "pausing" -> "Pausing" to colors.qualityOrange
        "paused" -> "Paused" to colors.qualityOrange
        "resuming" -> "Resuming" to colors.qualityOrange
        "stopping" -> "Stopping" to colors.qualityRed
        "finished" -> "Finished" to colors.qualityMint
        "error" -> "Error" to colors.qualityRed
        "gpu-out" -> "Offloading GPU" to colors.qualityOrange
        "gpu-in" -> "Reloading GPU" to colors.qualityOrange
        else -> "Idle" to colors.textDim
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
    val set = if (s.promptSets > 1) "set ${s.promptSet.coerceAtLeast(1)}/${s.promptSets}  " else ""
    return when {
        s.active -> "${set}image ${s.repeat + 1}/${s.repeats}  denoise ${s.denoiseStep}/${s.denoiseSteps}"
        s.repeats > 0 && s.globalStep > 0 -> "Last at step ${s.globalStep}"
        else -> "Idle"
    }
}

private fun formatElapsed(startedAt: Double?, status: String): String {
    if (startedAt == null || startedAt <= 0.0) return "—"
    if (status == "idle") return "—"
    val now = kotlin.time.Clock.System.now().toEpochMilliseconds() / 1000.0
    val seconds = (now - startedAt).coerceAtLeast(0.0).roundToInt()
    val h = seconds / 3600
    val m = (seconds % 3600) / 60
    val s = seconds % 60
    return if (h > 0) "${h}h ${m}m" else "${m}m ${s}s"
}

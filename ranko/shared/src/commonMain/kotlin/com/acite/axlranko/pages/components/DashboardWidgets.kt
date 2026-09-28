package com.acite.axlranko.pages.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.ui.components.FrostedSurface
import com.acite.axlranko.ui.pointerIconHand
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.formatBytes

@Composable
fun DashboardSectionHeader(title: String) {
    val colors = rankoColors
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Box(
            modifier = Modifier
                .width(4.dp)
                .height(20.dp)
                .clip(rankoTokens.capsule)
                .background(colors.accentPink)
        )
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = colors.text
        )
    }
}

/**
 * The dashboard's run history: which run the page shows, and a way to switch to any other.
 *
 * [selected] is the run the user pinned; `null` follows whatever run the helper reports as
 * current, which is what makes the page follow a freshly started run. The training controls
 * only act on the run `state.json` is on, so pinning an older one is a read-only view.
 */
@Composable
fun RunSelector(
    runs: List<RunSummary>,
    selected: RunSummary?,
    resolvedRunId: String?,
    onSelect: (String?) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    val shown = displayedRun(runs, selected, resolvedRunId)
    // Following is a mode, not a run: the box says so, and the run it follows sits on the second line.
    val title = selected?.runId ?: "Current run"
    val subtitle = when {
        selected != null -> runDetailLabel(selected)
        shown != null -> runDetailLabel(shown)
        else -> "Nothing started yet; Start launches a run."
    }

    Box(modifier = modifier) {
        FrostedSurface(
            modifier = Modifier
                .fillMaxWidth()
                .pointerHoverIcon(pointerIconHand)
                .clickable { expanded = true }
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            text = "Run history",
                            style = MaterialTheme.typography.labelSmall,
                            color = rankoColors.textDim,
                        )
                        RunStateBadge(shown)
                    }
                    Text(
                        text = title,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.labelSmall,
                        color = rankoColors.textDim,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Icon(
                    Icons.Default.ArrowDropDown,
                    contentDescription = "Choose a run",
                    tint = rankoColors.accentPink,
                )
            }
        }

        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
            modifier = Modifier
                .widthIn(min = 320.dp, max = 560.dp)
                .heightIn(max = 420.dp),
        ) {
            RunMenuItem(
                title = "Current run",
                subtitle = shown?.runId ?: "Nothing started yet",
                badges = { RunStateBadge(shown) },
                chosen = selected == null,
                onClick = {
                    expanded = false
                    onSelect(null)
                },
            )
            runs.forEach { run ->
                RunMenuItem(
                    title = run.runId,
                    subtitle = runDetailLabel(run),
                    badges = { RunStateBadge(run) },
                    chosen = selected?.runId == run.runId,
                    onClick = {
                        expanded = false
                        onSelect(run.runId)
                    },
                )
            }
        }
    }
}

@Composable
internal fun RunMenuItem(
    title: String,
    subtitle: String,
    badges: @Composable () -> Unit,
    chosen: Boolean,
    onClick: () -> Unit,
) {
    DropdownMenuItem(
        text = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(
                    Icons.Default.Check,
                    contentDescription = null,
                    tint = if (chosen) rankoColors.accentPink else Color.Transparent,
                )
                Column(modifier = Modifier.weight(1f, fill = false)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = if (chosen) FontWeight.Bold else FontWeight.Medium,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f, fill = false),
                        )
                        badges()
                    }
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.labelSmall,
                        color = rankoColors.textDim,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        },
        onClick = onClick,
    )
}

/**
 * The run the page is showing: the one the user pinned, else the run the helper resolved to.
 * `null` means the trainer has no run — the page names none and shows no run's figures.
 */
internal fun displayedRun(
    runs: List<RunSummary>,
    selected: RunSummary?,
    resolvedRunId: String?,
): RunSummary? = selected ?: runs.firstOrNull { it.runId == resolvedRunId }

/**
 * `Live` while that run's process is running, `Stopped` once it is not, and no badge at all when
 * there is no run to speak of yet — the run history's only vocabulary.
 */
internal fun runStateLabel(run: RunSummary?): String? = when {
    run == null -> null
    run.live -> "Live"
    else -> "Stopped"
}

@Composable
private fun RunStateBadge(run: RunSummary?) {
    val label = runStateLabel(run) ?: return
    RunBadge(label, if (run?.live == true) rankoColors.qualityMint else rankoColors.textDim)
}

@Composable
private fun RunBadge(text: String, color: Color) {
    Box(
        modifier = Modifier
            .clip(rankoTokens.capsule)
            .background(color.copy(alpha = 0.18f))
            .padding(horizontal = 6.dp, vertical = 1.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = color,
        )
    }
}

/**
 * `2026-09-28 11:09:28 · step 4500 · 12 samples · 46 checkpoints · 10.4 GB` — the run's own
 * stamp plus what it left on disk. A run with no TensorBoard logs keeps its step count,
 * because both sample and checkpoint names carry it.
 */
internal fun runDetailLabel(run: RunSummary): String {
    val parts = mutableListOf<String>()
    runStampLabel(run.runId)?.let { parts += it }
    run.lastStep?.let { parts += "step $it" }
    if (run.samples > 0) parts += countLabel(run.samples, "sample")
    if (run.checkpoints > 0) parts += countLabel(run.checkpoints, "checkpoint")
    if (run.sizeBytes > 0) parts += formatBytes(run.sizeBytes)
    return parts.joinToString(" · ").ifEmpty { "no artifacts" }
}

/** The `YYYYMMDD_HHMMSS` a run id ends in, as `YYYY-MM-DD HH:MM:SS`; `null` when it has none. */
internal fun runStampLabel(runId: String): String? {
    val match = RUN_STAMP_RE.find(runId) ?: return null
    val date = match.groupValues[1]
    val time = match.groupValues[2]
    return "${date.substring(0, 4)}-${date.substring(4, 6)}-${date.substring(6, 8)} " +
        "${time.substring(0, 2)}:${time.substring(2, 4)}:${time.substring(4, 6)}"
}

private fun countLabel(count: Int, noun: String): String =
    if (count == 1) "1 $noun" else "$count ${noun}s"

private val RUN_STAMP_RE = Regex("""_(\d{8})_(\d{6})(?:_\d+)?$""")

@Composable
fun PathChip(label: String, path: String) {
    val colors = rankoColors
    FrostedSurface {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = colors.accentPink,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = path,
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
fun MetricCard(
    title: String,
    value: String,
    accentColor: Color,
    modifier: Modifier = Modifier,
) {
    val colors = rankoColors
    FrostedSurface(modifier = modifier) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .background(accentColor)
        )
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                title,
                style = MaterialTheme.typography.labelMedium,
                color = colors.textDim
            )
            Spacer(Modifier.height(6.dp))
            Text(
                value,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = accentColor
            )
        }
    }
}

@Composable
fun CompactMetric(label: String, value: String) {
    Column {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = rankoColors.textDim
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

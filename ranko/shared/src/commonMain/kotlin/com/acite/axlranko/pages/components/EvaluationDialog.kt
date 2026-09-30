package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.acite.axlranko.model.DEFAULT_TAGGER_CATEGORY
import com.acite.axlranko.model.EvaluationTarget
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.TaggerInfoResult
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.util.formatFixed

/**
 * The Evaluate dialog: how many images this checkpoint should be scored on, the tagger floor, and
 * which of the tagger's categories count.
 *
 * The field opens on the number of images the card already shows, because the depth is a floor: a
 * checkpoint that already has that many renders nothing and goes straight to tagging. The exact
 * image count of a top-up is the config's own (whole passes of its sample sets), and the card
 * reports it once the pass starts, so the dialog does not promise one.
 */
@Composable
internal fun EvaluationDialog(
    target: EvaluationTarget,
    tagger: TaggerInfoResult?,
    starting: Boolean,
    error: String?,
    onStart: (depth: Int, threshold: Float, categories: List<String>) -> Unit,
    onDismiss: () -> Unit,
) {
    val colors = rankoColors
    val key = target.checkpoint.path
    var depth by remember(key) { mutableStateOf(evaluationDefaultDepth(target.existingImages).toString()) }
    var threshold by remember(key) { mutableStateOf(DEFAULT_EVALUATION_THRESHOLD.toString()) }
    val known = tagger?.categories.orEmpty().map { it.key }.filter { it.isNotBlank() }
    var selected by remember(key) { mutableStateOf(setOf(DEFAULT_TAGGER_CATEGORY)) }

    val requestError = evaluationRequestError(depth, threshold)
    Dialog(onDismissRequest = onDismiss) {
        PorcelainCard {
            Column(
                modifier = Modifier.width(470.dp).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = "Evaluate ${target.checkpoint.dir}",
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = "Tops the sample images up to Depth when there are fewer, tags every one of " +
                        "them with the Pixai tagger and scores the tags against each prompt. Prompts come " +
                        "from the config this run saved; a run that saved none uses today's config.toml.",
                    color = colors.textDim,
                    fontSize = 11.sp,
                )
                OutlinedTextField(
                    value = depth,
                    onValueChange = { depth = it },
                    singleLine = true,
                    label = { Text("Depth") },
                    supportingText = {
                        Text("1 – $MAX_EVALUATION_DEPTH · at least this many images (this card shows ${target.existingImages})")
                    },
                    colors = rankoFieldColors(),
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = threshold,
                    onValueChange = { threshold = it },
                    singleLine = true,
                    label = { Text("Threshold") },
                    supportingText = { Text("0.0 – 1.0 · the model's calibrated value is the floor underneath it") },
                    colors = rankoFieldColors(),
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(text = "Categories", color = colors.text, fontSize = 12.sp)
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    maxItemsInEachRow = 3,
                ) {
                    if (known.isEmpty()) {
                        // Without the model's own list the dialog still has to be runnable: the
                        // tagger writes `general` by itself (and the reason line says why).
                        CapsuleChoice(text = DEFAULT_TAGGER_CATEGORY, selected = true, onClick = {})
                    } else {
                        known.forEach { category ->
                            CapsuleChoice(
                                text = category,
                                selected = category in selected,
                                onClick = {
                                    selected = if (category in selected) selected - category else selected + category
                                },
                            )
                        }
                    }
                }
                if (tagger != null && !tagger.available) {
                    Text(
                        text = tagger.reason.ifBlank { "The tagger model is not in the local cache." },
                        color = colors.qualityRed,
                        fontSize = 11.sp,
                    )
                }
                (requestError ?: error)?.let { message ->
                    Text(text = message, color = colors.qualityRed, fontSize = 12.sp)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CapsuleButton(
                        text = if (starting) "Starting…" else "Evaluate",
                        onClick = {
                            onStart(
                                depth.trim().toIntOrNull() ?: 1,
                                threshold.trim().replace(',', '.').toFloatOrNull()
                                    ?: DEFAULT_EVALUATION_THRESHOLD,
                                if (known.isEmpty()) emptyList() else selected.toList(),
                            )
                        },
                        emphasized = true,
                        enabled = requestError == null && !starting,
                    )
                    CapsuleButton(text = "Cancel", onClick = onDismiss, enabled = !starting)
                }
            }
        }
    }
}

/**
 * What a card says about its evaluation: the failure, the scores with a details block, or nothing
 * while it is still running (the card's progress row is what reports that).
 */
@Composable
internal fun EvaluationCardBlock(
    job: GeneratedSampleJob,
    detailsOpen: Boolean,
    onToggleDetails: (String) -> Unit,
) {
    val colors = rankoColors
    val scores = job.scores
    if (job.state == JOB_RUNNING) return

    if (job.error != null) {
        Text(
            text = "Evaluation failed: ${job.error}",
            style = MaterialTheme.typography.labelSmall,
            color = colors.qualityRed,
            maxLines = 3,
            overflow = TextOverflow.Ellipsis,
        )
        return
    }

    val score = evaluationScoreLabel(scores) ?: return
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                text = score,
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold,
                color = colors.text,
                modifier = Modifier.weight(1f),
            )
            CapsuleButton(
                text = if (detailsOpen) "Hide details" else "Details",
                onClick = { onToggleDetails(job.id) },
                compact = true,
            )
        }
        evaluationUnionLabel(scores)?.let { union ->
            Text(text = union, style = MaterialTheme.typography.labelSmall, color = colors.textDim)
        }
        evaluationCoverageLabel(scores)?.let { coverage ->
            Text(text = coverage, style = MaterialTheme.typography.labelSmall, color = colors.textDim)
        }
        Text(
            text = evaluationConfigLabel(job),
            style = MaterialTheme.typography.labelSmall,
            color = colors.textDim,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        if (detailsOpen) {
            EvaluationDetails(job)
        }
    }
}

/** The per-prompt breakdown and the tags that cost the most, shown under a card's score line. */
@Composable
private fun EvaluationDetails(job: GeneratedSampleJob) {
    val colors = rankoColors
    val scores = job.scores ?: return
    Column(
        modifier = Modifier.fillMaxWidth().heightIn(max = 240.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        evaluationDetailRows(scores).forEach { row ->
            Text(
                text = "${row.images}× F1 ${formatFixed(row.perImageF1, 2)} " +
                    "(union ${formatFixed(row.unionF1, 2)})",
                style = MaterialTheme.typography.labelSmall,
                color = colors.text,
            )
            Text(
                text = row.prompt,
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                fontFamily = FontFamily.Monospace,
            )
        }
        if (scores.topFalsePositives.isNotEmpty()) {
            Text(
                text = "most drawn but not asked: " + scores.topFalsePositives
                    .take(EVALUATION_TOP_TAGS)
                    .joinToString(", ") { evaluationTagLabel(it.tag, it.count) },
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
            )
        }
        if (scores.topFalseNegatives.isNotEmpty()) {
            Text(
                text = "most asked but not drawn: " + scores.topFalseNegatives
                    .take(EVALUATION_TOP_TAGS)
                    .joinToString(", ") { evaluationTagLabel(it.tag, it.count) },
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
            )
        }
    }
}

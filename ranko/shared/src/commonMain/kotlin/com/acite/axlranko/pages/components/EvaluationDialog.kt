package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
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
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.acite.axlranko.model.DEFAULT_TAGGER_CATEGORY
import com.acite.axlranko.model.EvaluationPromptsResponse
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
 * The panel's width when the window has room for it. Narrower windows get a narrower panel: the
 * fields and the prompt list shrink with it rather than running past the edge.
 */
private val EVALUATION_PANEL_WIDTH = 540.dp

/**
 * The evaluation panel: everything about one checkpoint's scoring, and the form that starts another.
 *
 * It is the same surface before and after a pass, which is the point — the score used to be a block
 * of text under the card, and a pass that had run twice left only the newest line of it visible.
 * Here the result is the panel's own content (recall first, precision and F1 under it, the per-prompt
 * breakdown behind `Details`), reachable again at any time from the card's `Evaluation` button, and
 * the fields that start a new pass — depth, threshold, tagger categories, and the prompt tags the
 * scoring should care about — sit under it.
 *
 * The depth field opens on the number of images the card already shows, because the depth is a floor:
 * a checkpoint that already has that many renders nothing and goes straight to tagging. The exact
 * image count of a top-up is the config's own (whole passes of its sample sets), so the dialog does
 * not promise one.
 */
@Composable
internal fun EvaluationDialog(
    target: EvaluationTarget,
    job: GeneratedSampleJob?,
    tagger: TaggerInfoResult?,
    prompts: EvaluationPromptsResponse?,
    promptsLoading: Boolean,
    promptsError: String?,
    selectedTags: Set<String>,
    starting: Boolean,
    error: String?,
    detailsOpen: Boolean,
    /** Room the window leaves for the panel; the content scrolls inside it. */
    maxWidth: Dp,
    maxHeight: Dp,
    onToggleTag: (String) -> Unit,
    onClearTags: () -> Unit,
    onToggleDetails: () -> Unit,
    onCancel: (String) -> Unit,
    onStart: (depth: Int, threshold: Float, categories: List<String>, tags: List<String>) -> Unit,
    onDismiss: () -> Unit,
    /** Hoisted so the panel's own test can measure the viewport it scrolls in. */
    scrollState: ScrollState = rememberScrollState(),
) {
    val colors = rankoColors
    val key = target.checkpoint.path
    var depth by remember(key) { mutableStateOf(evaluationDefaultDepth(target.existingImages).toString()) }
    var threshold by remember(key) { mutableStateOf(DEFAULT_EVALUATION_THRESHOLD.toString()) }
    val known = tagger?.categories.orEmpty().map { it.key }.filter { it.isNotBlank() }
    var selected by remember(key) { mutableStateOf(setOf(DEFAULT_TAGGER_CATEGORY)) }

    val requestError = evaluationRequestError(depth, threshold)
    val running = job?.takeIf { it.state == JOB_RUNNING }

    Dialog(onDismissRequest = onDismiss) {
        PorcelainCard {
            // A dialog window packs itself around its content, so an unbounded column simply runs off
            // the screen and the buttons at the bottom become unreachable. The panel takes the room
            // the window has and scrolls the rest, which is what keeps a long prompt list or an open
            // details block usable on a short window.
            Column(
                modifier = Modifier
                    .width(minOf(EVALUATION_PANEL_WIDTH, maxWidth))
                    .heightIn(max = maxHeight)
                    .verticalScroll(scrollState)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                val step = target.checkpoint.step
                Text(
                    text = "${if (job == null) "Evaluate" else "Evaluation"} ${target.checkpoint.dir}" +
                        (if (step != null) " · step $step" else ""),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )

                if (job != null) {
                    EvaluationResult(
                        job = job,
                        detailsOpen = detailsOpen,
                        onToggleDetails = onToggleDetails,
                        onCancel = onCancel,
                    )
                    HorizontalDivider(color = colors.stroke.copy(alpha = 0.4f))
                }

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

                PromptTagPicker(
                    prompts = prompts,
                    loading = promptsLoading,
                    error = promptsError,
                    selected = selectedTags,
                    onToggle = onToggleTag,
                    onClear = onClearTags,
                )

                (requestError ?: error)?.let { message ->
                    Text(text = message, color = colors.qualityRed, fontSize = 12.sp)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    val startLabel = when {
                        starting -> "Starting…"
                        job != null && running == null -> "Evaluate again"
                        else -> "Evaluate"
                    }
                    CapsuleButton(
                        text = startLabel,
                        onClick = {
                            onStart(
                                depth.trim().toIntOrNull() ?: 1,
                                threshold.trim().replace(',', '.').toFloatOrNull()
                                    ?: DEFAULT_EVALUATION_THRESHOLD,
                                if (known.isEmpty()) emptyList() else selected.toList(),
                                selectedTags.toList(),
                            )
                        },
                        emphasized = true,
                        // One pass at a time: the GPU is single-tenant and the helper refuses a
                        // second job while this one runs, so the button says so instead.
                        enabled = requestError == null && !starting && running == null,
                    )
                    CapsuleButton(text = "Close", onClick = onDismiss, enabled = !starting)
                }
            }
        }
    }
}

/**
 * The tag picker: the tags the run's prompts ask for, with the frequency Statistics draws, one click
 * each. Nothing selected means every tag is scored, which is what the line under the list says — the
 * pass then answers "how well does the caption match the prompt" instead of "were these tags drawn".
 */
@Composable
private fun PromptTagPicker(
    prompts: EvaluationPromptsResponse?,
    loading: Boolean,
    error: String?,
    selected: Set<String>,
    onToggle: (String) -> Unit,
    onClear: () -> Unit,
) {
    val colors = rankoColors
    val tags = prompts?.tags.orEmpty()
    val reason = error ?: prompts?.reason?.takeIf { it.isNotBlank() }
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(text = "Prompt tags to score", color = colors.text, fontSize = 12.sp, modifier = Modifier.weight(1f))
        if (selected.isNotEmpty()) {
            CapsuleButton(text = "Clear", onClick = onClear, compact = true)
        }
    }
    when {
        loading && tags.isEmpty() -> Text(
            text = "Reading this run's prompts…",
            color = colors.textDim,
            fontSize = 11.sp,
        )

        reason != null || tags.isEmpty() -> Text(
            text = reason?.let { "Cannot read this run's prompts: $it" }
                ?: "This run's prompts carry no tags; every tag a prompt asks for will be scored.",
            color = if (reason != null) colors.qualityRed else colors.textDim,
            fontSize = 11.sp,
        )

        else -> {
            TagFrequencyList(tags = tags, selected = selected, onToggle = onToggle)
            Text(
                text = if (selected.isEmpty()) {
                    "Nothing selected: every tag the prompts ask for is scored."
                } else {
                    "Only the ${selected.size} selected tag(s) are scored — every other prompt tag is " +
                        "ignored, so recall answers \"were these drawn\"."
                },
                color = colors.textDim,
                fontSize = 11.sp,
            )
        }
    }
}

/**
 * What the panel says about the job it is showing: the failure, the running phase, or the scores with
 * a details block. Recall is the headline; precision and F1 are deliberately secondary, because the
 * question an evaluation exists to answer is whether the prompt's tags got drawn at all.
 */
@Composable
private fun EvaluationResult(
    job: GeneratedSampleJob,
    detailsOpen: Boolean,
    onToggleDetails: () -> Unit,
    onCancel: (String) -> Unit,
) {
    val colors = rankoColors
    if (job.state == JOB_RUNNING) {
        val total = job.totalImages
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                LinearProgressIndicator(
                    progress = {
                        if (total > 0) {
                            job.imagesDone.toFloat().coerceAtMost(total.toFloat()) / total
                        } else {
                            0f
                        }
                    },
                    modifier = Modifier.weight(1f).height(4.dp),
                )
                Text(evaluationProgressLabel(job), style = MaterialTheme.typography.labelSmall, color = colors.accentPink)
                CapsuleButton(text = "Cancel", onClick = { onCancel(job.id) }, compact = true)
            }
        }
        return
    }

    if (job.error != null) {
        Text(
            text = "Evaluation failed: ${job.error}",
            style = MaterialTheme.typography.labelSmall,
            color = colors.qualityRed,
        )
        return
    }

    val scores = job.scores
    val headline = evaluationRecallHeadline(scores) ?: return
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(
            text = headline,
            color = colors.accentPink,
            fontWeight = FontWeight.Bold,
            fontSize = 22.sp,
        )
        evaluationSecondaryLabel(scores)?.let { secondary ->
            Text(text = secondary, style = MaterialTheme.typography.labelSmall, color = colors.textDim)
        }
        evaluationUnionLabel(scores)?.let { union ->
            Text(text = union, style = MaterialTheme.typography.labelSmall, color = colors.textDim)
        }
        evaluationCoverageLabel(scores)?.let { coverage ->
            Text(text = coverage, style = MaterialTheme.typography.labelSmall, color = colors.textDim)
        }
        evaluationTagsLabel(scores)?.let { tags ->
            Text(text = tags, style = MaterialTheme.typography.labelSmall, color = colors.textDim)
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                text = evaluationConfigLabel(job),
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            CapsuleButton(
                text = if (detailsOpen) "Hide details" else "Details",
                onClick = onToggleDetails,
                compact = true,
            )
        }
        if (detailsOpen) {
            EvaluationDetails(job)
        }
    }
}

/**
 * The per-prompt breakdown and the tags that cost the most, under the score it belongs to. It has no
 * scroll of its own: the panel scrolls as one, so a long list stays reachable instead of nesting a
 * second scroll area the wheel would have to be aimed at.
 */
@Composable
private fun EvaluationDetails(job: GeneratedSampleJob) {
    val colors = rankoColors
    val scores = job.scores ?: return
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        evaluationDetailRows(scores).forEach { row ->
            Text(
                text = "${row.images}× R ${formatFixed(row.perImageRecall, 2)} " +
                    "(union ${formatFixed(row.unionRecall, 2)})",
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

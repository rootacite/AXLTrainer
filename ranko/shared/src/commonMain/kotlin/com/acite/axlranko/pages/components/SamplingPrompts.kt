package com.acite.axlranko.pages.components

import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import com.acite.axlranko.model.SAMPLE_SET_ERROR_PREFIX
import com.acite.axlranko.model.SampleClearResult
import com.acite.axlranko.model.SamplePromptsResponse
import com.acite.axlranko.model.SampleSetForm
import com.acite.axlranko.model.SampleSetInfo
import com.acite.axlranko.model.clearedSamplesLabel
import com.acite.axlranko.model.samplePromptsEditorTitle
import com.acite.axlranko.model.samplePromptsLiveLabel
import com.acite.axlranko.model.samplePromptsSourceLabel
import com.acite.axlranko.model.samplePromptsSummaryLine
import com.acite.axlranko.model.sampleSetFormErrors
import com.acite.axlranko.model.sampleSetSummaryLine
import com.acite.axlranko.model.toForm
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors

/** The editor's width when the window has room for it; a narrower window narrows the panel. */
private val SAMPLE_PROMPTS_PANEL_WIDTH = 620.dp

/**
 * The Sampling Prompts section: the prompts the displayed run's sample points, the cards' "Generate
 * samples" and an evaluation render with, one line per set, plus the way to change them.
 *
 * They are the run's own — the `config.toml` it saved beside its logs, or the sets saved for it
 * here — so a run from weeks ago still says what it drew with, and editing one run never touches
 * another. While that run is live the section says so: the trainer reads the same file before every
 * sample point, so its next checkpoint uses the new prompts.
 */
@Composable
internal fun SamplingPromptsSection(
    prompts: SamplePromptsResponse?,
    loading: Boolean,
    error: String?,
    saving: Boolean,
    onEdit: () -> Unit,
    onReset: () -> Unit,
) {
    val colors = rankoColors
    val readable = prompts?.takeIf { it.reason.isBlank() }
    var confirmReset by remember { mutableStateOf(false) }

    if (confirmReset) {
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Use this run's own config again?") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "The prompts saved for this run are removed, and it samples with the " +
                            "config it started from again."
                    )
                    Text(
                        text = prompts?.file.orEmpty(),
                        style = MaterialTheme.typography.bodySmall,
                        color = colors.textDim,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmReset = false
                        onReset()
                    }
                ) { Text("Reset") }
            },
            dismissButton = {
                TextButton(onClick = { confirmReset = false }) { Text("Cancel") }
            },
        )
    }

    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = "Sampling prompts",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = colors.text,
                    modifier = Modifier.weight(1f),
                )
                CapsuleButton(
                    text = "Edit prompts…",
                    onClick = onEdit,
                    enabled = !loading && !saving && readable?.sets?.isNotEmpty() == true,
                    compact = true,
                ) {
                    Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(14.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Edit prompts…", fontWeight = FontWeight.SemiBold)
                }
                if (readable?.edited == true) {
                    CapsuleButton(
                        text = if (saving) "Resetting…" else "Reset",
                        onClick = { confirmReset = true },
                        enabled = !saving,
                        compact = true,
                    )
                }
            }

            when {
                loading && prompts == null -> Text(
                    text = "Reading this run's prompts…",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.textDim,
                )

                readable == null -> Text(
                    text = prompts?.reason?.takeIf { it.isNotBlank() }
                        ?.let { "Cannot read this run's prompts: $it" }
                        ?: "No run to read sampling prompts for yet.",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (prompts?.reason?.isNotBlank() == true) colors.qualityRed else colors.textDim,
                )

                else -> {
                    Text(
                        text = samplePromptsSummaryLine(readable),
                        style = MaterialTheme.typography.bodySmall,
                        color = colors.text,
                    )
                    Text(
                        text = samplePromptsSourceLabel(readable),
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.textDim,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    samplePromptsLiveLabel(readable)?.let { line ->
                        Text(text = line, style = MaterialTheme.typography.labelSmall, color = colors.accentPink)
                    }
                    error?.let { message ->
                        Text(text = message, style = MaterialTheme.typography.labelSmall, color = colors.qualityRed)
                    }
                    HorizontalDivider(color = colors.stroke.copy(alpha = 0.4f))
                    readable.sets.forEachIndexed { index, set ->
                        SampleSetRow(index = index + 1, set = set)
                    }
                }
            }
        }
    }
}

@Composable
private fun SampleSetRow(index: Int, set: SampleSetInfo) {
    val colors = rankoColors
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = set.name.ifBlank { "set $index" },
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold,
                color = colors.accentPink,
            )
            Text(
                text = sampleSetSummaryLine(set),
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        Text(
            text = set.prompt,
            style = MaterialTheme.typography.bodySmall,
            color = colors.text,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            fontFamily = FontFamily.Monospace,
        )
        if (set.negative.isNotBlank()) {
            Text(
                text = "negative: ${set.negative}",
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

/**
 * The prompt editor: the run's sets, and the button that saves them back for that run.
 *
 * A panel capped to the window it is opened in, scrolling inside it — Compose draws a `Dialog` as a
 * layer of the same window here, so an unbounded column would run off the bottom edge and put its
 * buttons out of reach. Each field carries its own reason when it is rejected, and the same rules
 * run again in api.py, so a refused save costs no round trip.
 */
@Composable
internal fun SamplePromptsEditorDialog(
    prompts: SamplePromptsResponse,
    saving: Boolean,
    error: String?,
    maxWidth: Dp,
    maxHeight: Dp,
    onSave: (List<SampleSetForm>) -> Unit,
    onDismiss: () -> Unit,
    /** Hoisted so the editor's own test can measure the viewport it scrolls in. */
    scrollState: ScrollState = rememberScrollState(),
) {
    val colors = rankoColors
    val key = prompts.runId.orEmpty()
    var forms by remember(key) { mutableStateOf(prompts.sets.map { it.toForm() }) }
    val errors = sampleSetFormErrors(forms)

    Dialog(onDismissRequest = onDismiss) {
        PorcelainCard {
            Column(
                modifier = Modifier
                    .width(minOf(SAMPLE_PROMPTS_PANEL_WIDTH, maxWidth))
                    .heightIn(max = maxHeight)
                    .verticalScroll(scrollState)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = samplePromptsEditorTitle(prompts),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp,
                )
                Text(
                    text = "These are the prompts this run's sample points, the cards' \"Generate " +
                        "samples\" and an evaluation render with. Saving them applies to this run " +
                        "only; a live run uses them at its next sample point.",
                    color = colors.textDim,
                    fontSize = 11.sp,
                )

                forms.forEachIndexed { index, form ->
                    SampleSetEditor(
                        index = index,
                        form = form,
                        errors = errors,
                        removable = forms.size > 1,
                        onChange = { updated -> forms = forms.toMutableList().also { it[index] = updated } },
                        onRemove = { forms = forms.toMutableList().also { it.removeAt(index) } },
                    )
                }

                CapsuleButton(
                    text = "Add set",
                    onClick = { forms = forms + (forms.lastOrNull() ?: SampleSetForm()) },
                    compact = true,
                ) {
                    Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(14.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Add set", fontWeight = FontWeight.SemiBold)
                }

                error?.let { message ->
                    Text(text = message, color = colors.qualityRed, fontSize = 12.sp)
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CapsuleButton(
                        text = if (saving) "Saving…" else "Save",
                        onClick = { onSave(forms) },
                        emphasized = true,
                        enabled = !saving && errors.isEmpty(),
                    )
                    CapsuleButton(text = "Cancel", onClick = onDismiss, enabled = !saving)
                }
            }
        }
    }
}

@Composable
private fun SampleSetEditor(
    index: Int,
    form: SampleSetForm,
    errors: Map<String, String>,
    removable: Boolean,
    onChange: (SampleSetForm) -> Unit,
    onRemove: () -> Unit,
) {
    val colors = rankoColors
    val key = { field: String -> "$SAMPLE_SET_ERROR_PREFIX$index.$field" }
    PorcelainCard {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "Set ${index + 1}",
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.sp,
                    modifier = Modifier.weight(1f),
                )
                CapsuleButton(text = "Remove", onClick = onRemove, enabled = removable, compact = true) {
                    Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(14.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Remove", fontWeight = FontWeight.SemiBold)
                }
            }
            SampleField(
                value = form.name,
                onValueChange = { onChange(form.copy(name = it)) },
                label = "Name",
                error = null,
            )
            SampleField(
                value = form.prompt,
                onValueChange = { onChange(form.copy(prompt = it)) },
                label = "Prompt",
                error = errors[key("prompt")],
                minLines = 3,
            )
            SampleField(
                value = form.negative,
                onValueChange = { onChange(form.copy(negative = it)) },
                label = "Negative",
                error = null,
                minLines = 2,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3,
            ) {
                SampleField(
                    value = form.width,
                    onValueChange = { onChange(form.copy(width = it)) },
                    label = "Width",
                    error = errors[key("width")],
                    small = true,
                )
                SampleField(
                    value = form.height,
                    onValueChange = { onChange(form.copy(height = it)) },
                    label = "Height",
                    error = errors[key("height")],
                    small = true,
                )
                SampleField(
                    value = form.steps,
                    onValueChange = { onChange(form.copy(steps = it)) },
                    label = "Steps",
                    error = errors[key("steps")],
                    small = true,
                )
                SampleField(
                    value = form.guidanceScale,
                    onValueChange = { onChange(form.copy(guidanceScale = it)) },
                    label = "CFG",
                    error = errors[key("guidance_scale")],
                    small = true,
                )
                SampleField(
                    value = form.guidanceRescale,
                    onValueChange = { onChange(form.copy(guidanceRescale = it)) },
                    label = "Rescale",
                    error = errors[key("guidance_rescale")],
                    small = true,
                )
                SampleField(
                    value = form.seed,
                    onValueChange = { onChange(form.copy(seed = it)) },
                    label = "Seed",
                    error = errors[key("seed")],
                    small = true,
                )
                SampleField(
                    value = form.repeat,
                    onValueChange = { onChange(form.copy(repeat = it)) },
                    label = "Repeat",
                    error = errors[key("repeat")],
                    small = true,
                )
            }
        }
    }
}

@Composable
private fun SampleField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    error: String?,
    minLines: Int = 1,
    small: Boolean = false,
) {
    val colors = rankoColors
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        singleLine = minLines == 1,
        minLines = minLines,
        label = { Text(label) },
        isError = error != null,
        supportingText = error?.let { message -> { Text(message, color = colors.qualityRed) } },
        colors = rankoFieldColors(),
        textStyle = MaterialTheme.typography.bodySmall,
        modifier = if (small) Modifier.width(150.dp) else Modifier.fillMaxWidth(),
    )
}

/**
 * What a card reports once its sample images were cleared: how many went, or why the helper
 * refused. The line belongs to the checkpoint that asked, so it is drawn on that card only.
 */
@Composable
internal fun ClearedSamplesStatus(result: SampleClearResult?) {
    if (result == null) return
    val colors = rankoColors
    Text(
        text = clearedSamplesLabel(result),
        style = MaterialTheme.typography.labelSmall,
        color = if (result.error != null) colors.qualityRed else colors.textDim,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
    )
}

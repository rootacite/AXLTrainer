package com.acite.axlranko.pages.components.automation

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import coil3.compose.AsyncImage
import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.data.AutomationJobDetail
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.data.JobPromptState
import com.acite.axlranko.data.JobPass
import com.acite.axlranko.data.imageSeedAt
import com.acite.axlranko.data.jobPassProgress
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.GalleryImageAction
import com.acite.axlranko.model.JobFilter
import com.acite.axlranko.model.jobElapsedSeconds
import com.acite.axlranko.model.jobImagePathFor
import com.acite.axlranko.model.jobProgress
import com.acite.axlranko.pages.AutomationScreenViewModel
import com.acite.axlranko.pages.components.ImagePreviewOverlay
import com.acite.axlranko.pages.components.PreviewImage
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.QuietTextButton
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.copyTextToClipboard
import com.acite.axlranko.util.openLocalDirectory
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlin.time.Clock

/** The Gallery: the jobs, what each one produced, and one image up close. */
@Composable
fun GalleryPane(
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel = metroViewModel(),
) {
    val colors = rankoColors
    val lang = state.language
    // The preview is a sibling of the scrolling column, never a child of it: inside a scrolling
    // column it would be measured with an unbounded height and collapse onto its own header row.
    Box(modifier = Modifier.fillMaxSize()) {
    Column(
        modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        PorcelainCard {
            Column(
                modifier = Modifier.fillMaxWidth().padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = uiText(lang, "jobs"),
                        color = colors.text,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 13.sp,
                        modifier = Modifier.weight(1f),
                    )
                    if (state.jobsLoading) CircularProgressIndicator(modifier = Modifier.padding(2.dp).width(14.dp))
                    QuietTextButton(text = uiText(lang, "refresh"), onClick = viewModel::refreshJobs)
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    JobFilter.entries.forEach { filter ->
                        CapsuleChoice(
                            text = uiText(lang, filterKey(filter)),
                            selected = state.jobFilter == filter,
                            onClick = { viewModel.setJobFilter(filter) },
                        )
                    }
                    OutlinedTextField(
                        value = state.jobSearch,
                        onValueChange = viewModel::setJobSearch,
                        singleLine = true,
                        placeholder = { Text(uiText(lang, "search"), fontSize = 11.sp) },
                        colors = rankoFieldColors(),
                        modifier = Modifier.width(220.dp),
                    )
                }
                if (state.visibleJobs.isEmpty()) {
                    Text(text = uiText(lang, "no_jobs"), color = colors.textDim, fontSize = 11.sp)
                } else {
                    // The history is unbounded, so the list scrolls inside the card instead of
                    // growing with it and pushing the gallery that follows it off the page. The
                    // bounded height is what the inner scroll is measured against.
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 320.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        state.visibleJobs.forEach { job ->
                            JobRow(job, state, viewModel)
                        }
                    }
                }
                state.jobsError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
            }
        }

        val detail = state.jobDetail
        if (detail != null) {
            PorcelainCard {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    // A pass owns the record and the folder while it runs, so every button that
                    // touches either is off for the whole of it.
                    val busy = detail.state == "running" || state.jobActionBusy != ""
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = detail.id,
                            color = colors.text,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 13.sp,
                            modifier = Modifier.weight(1f),
                        )
                        CapsuleButton(
                            text = uiText(lang, "append_all"),
                            onClick = { viewModel.openPromptAppendAll(detail.id) },
                            enabled = !busy,
                            compact = true,
                        )
                        CapsuleButton(
                            text = uiText(lang, "cancel_job"),
                            onClick = { viewModel.cancelJob(detail.id) },
                            enabled = detail.state == "running",
                            danger = true,
                            compact = true,
                        )
                        CapsuleButton(
                            text = uiText(lang, "retry_failed"),
                            onClick = { viewModel.retryFailedJob(detail.id) },
                            enabled = detail.state != "running",
                            compact = true,
                        )
                        CapsuleButton(
                            text = uiText(lang, "save_records"),
                            onClick = viewModel::downloadJobRecord,
                            enabled = state.galleryImagePaths.isNotEmpty(),
                            compact = true,
                        )
                        CapsuleButton(
                            text = uiText(lang, "open_folder"),
                            onClick = { openLocalDirectory(detail.outputDir) },
                            enabled = detail.outputDir.isNotBlank(),
                            compact = true,
                        )
                        CapsuleButton(
                            text = uiText(lang, "delete"),
                            onClick = { viewModel.confirmDeleteJob(detail.id) },
                            enabled = detail.state != "running" && state.jobActionBusy == "",
                            danger = true,
                            compact = true,
                        )
                    }
                    detail.error?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }

                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(text = uiText(lang, "thumb_size"), color = colors.textDim, fontSize = 11.sp)
                        Slider(
                            value = state.galleryThumbSize,
                            onValueChange = viewModel::setGalleryThumbSize,
                            valueRange = 80f..360f,
                            modifier = Modifier.width(220.dp),
                        )
                        Text(
                            text = "${state.galleryThumbSize.toInt()}px",
                            color = colors.textDim,
                            fontSize = 11.sp,
                        )
                    }

                    if (state.galleryImagePaths.isEmpty()) {
                        Text(text = uiText(lang, "no_images_yet"), color = colors.textDim, fontSize = 11.sp)
                    } else {
                        var startIndex = 0
                        detail.prompts.forEach { prompt ->
                            if (prompt.images.isEmpty()) return@forEach
                            val first = startIndex
                            startIndex += prompt.images.size
                            PromptGalleryRow(
                                prompt = prompt,
                                firstIndex = first,
                                outputDir = detail.outputDir,
                                jobId = detail.id,
                                state = state,
                                viewModel = viewModel,
                                busy = busy,
                            )
                        }
                    }
                }
            }
        }

        val pendingDelete = state.pendingDeleteJob
        if (pendingDelete != null) {
            Dialog(onDismissRequest = viewModel::dismissDeleteJob) {
                PorcelainCard {
                    Column(
                        modifier = Modifier.width(420.dp).padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Text(
                            text = uiText(lang, "confirm_delete_job"),
                            color = colors.text,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = pendingDelete,
                            color = colors.textDim,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            CapsuleButton(
                                text = uiText(lang, "yes"),
                                onClick = { viewModel.deleteJob(pendingDelete) },
                                danger = true,
                                compact = true,
                            )
                            CapsuleButton(
                                text = uiText(lang, "no"),
                                onClick = viewModel::dismissDeleteJob,
                                compact = true,
                            )
                        }
                    }
                }
            }
        }

        state.pendingImageAction?.let { pending ->
            val deleting = pending.action == GalleryImageAction.Delete
            Dialog(onDismissRequest = viewModel::dismissImageAction) {
                PorcelainCard {
                    Column(
                        modifier = Modifier.width(470.dp).padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Text(
                            text = uiText(
                                lang,
                                if (deleting) "confirm_delete_image" else "confirm_regenerate_image",
                            ),
                            color = colors.text,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = pending.ref.image,
                            color = colors.textDim,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                        )
                        Text(
                            text = uiText(lang, if (deleting) "delete_last_image_note" else "overwrite_note"),
                            color = colors.textDim,
                            fontSize = 11.sp,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            CapsuleButton(
                                text = uiText(lang, if (deleting) "delete_image" else "regenerate_image"),
                                onClick = viewModel::runImageAction,
                                danger = deleting,
                                compact = true,
                            )
                            CapsuleButton(
                                text = uiText(lang, "no"),
                                onClick = viewModel::dismissImageAction,
                                compact = true,
                            )
                        }
                    }
                }
            }
        }

        state.editingPrompt?.let { draft ->
            Dialog(onDismissRequest = viewModel::dismissPromptEdit) {
                PorcelainCard {
                    Column(
                        modifier = Modifier.width(560.dp).padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Text(
                            text = uiText(lang, "edit_prompt") + "  #${draft.promptIndex + 1}",
                            color = colors.text,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                        )
                        OutlinedTextField(
                            value = draft.text,
                            onValueChange = viewModel::updatePromptEdit,
                            minLines = 4,
                            maxLines = 8,
                            label = { Text(uiText(lang, "prompt_text"), fontSize = 11.sp) },
                            colors = rankoFieldColors(),
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(text = uiText(lang, "prompt_edit_note"), color = colors.textDim, fontSize = 10.sp)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            CapsuleButton(
                                text = uiText(lang, "save"),
                                onClick = viewModel::savePromptEdit,
                                enabled = draft.text.isNotBlank() && state.jobActionBusy == "",
                                emphasized = true,
                                compact = true,
                            )
                            CapsuleButton(
                                text = uiText(lang, "no"),
                                onClick = viewModel::dismissPromptEdit,
                                compact = true,
                            )
                        }
                    }
                }
            }
        }

        state.extendingPrompt?.let { draft ->
            Dialog(onDismissRequest = viewModel::dismissPromptExtend) {
                PorcelainCard {
                    Column(
                        modifier = Modifier.width(470.dp).padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Text(
                            text = uiText(lang, "add_images") + "  #${draft.promptIndex + 1}",
                            color = colors.text,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                        )
                        OutlinedTextField(
                            value = draft.count,
                            onValueChange = viewModel::updatePromptExtend,
                            singleLine = true,
                            label = { Text(uiText(lang, "images_count"), fontSize = 11.sp) },
                            colors = rankoFieldColors(),
                            modifier = Modifier.width(120.dp),
                        )
                        Text(text = uiText(lang, "add_images_note"), color = colors.textDim, fontSize = 10.sp)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            CapsuleButton(
                                text = uiText(lang, "yes"),
                                onClick = viewModel::confirmPromptExtend,
                                enabled = draft.images != null && state.jobActionBusy == "",
                                emphasized = true,
                                compact = true,
                            )
                            CapsuleButton(
                                text = uiText(lang, "no"),
                                onClick = viewModel::dismissPromptExtend,
                                compact = true,
                            )
                        }
                    }
                }
            }
        }

        state.appendingAllPrompts?.let { draft ->
            Dialog(onDismissRequest = viewModel::dismissPromptAppendAll) {
                PorcelainCard {
                    Column(
                        modifier = Modifier.width(470.dp).padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Text(
                            text = uiText(lang, "append_all_title"),
                            color = colors.text,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                        )
                        OutlinedTextField(
                            value = draft.count,
                            onValueChange = viewModel::updatePromptAppendAll,
                            singleLine = true,
                            label = { Text(uiText(lang, "images_count"), fontSize = 11.sp) },
                            colors = rankoFieldColors(),
                            modifier = Modifier.width(120.dp),
                        )
                        Text(text = uiText(lang, "append_all_note"), color = colors.textDim, fontSize = 10.sp)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            CapsuleButton(
                                text = uiText(lang, "yes"),
                                onClick = viewModel::confirmPromptAppendAll,
                                enabled = draft.images != null && state.jobActionBusy == "",
                                emphasized = true,
                                compact = true,
                            )
                            CapsuleButton(
                                text = uiText(lang, "no"),
                                onClick = viewModel::dismissPromptAppendAll,
                                compact = true,
                            )
                        }
                    }
                }
            }
        }

    }

        state.galleryPreviewIndex?.let { index ->
            val images = previewImages(state)
            if (images.isNotEmpty()) {
                val current = images[index.coerceIn(images.indices)]
                // The overlay is a viewer: every action lives on the page itself (the thumbnail
                // discs, the prompt-row buttons), so this stays save/copy-free.
                ImagePreviewOverlay(
                    images = images,
                    index = index.coerceIn(images.indices),
                    onClose = viewModel::closeGalleryPreview,
                    onPrev = viewModel::previewPrev,
                    onNext = viewModel::previewNext,
                )
            }
        }
    }
}

private fun filterKey(filter: JobFilter): String = when (filter) {
    JobFilter.All -> "filter_all"
    JobFilter.Running -> "filter_running"
    JobFilter.Done -> "filter_done"
    JobFilter.Failed -> "filter_failed"
    JobFilter.Cancelled -> "filter_cancelled"
}

/** The per-image captions the shared preview overlay shows. */
private fun previewImages(state: AutomationUiState): List<PreviewImage> {
    val detail = state.jobDetail ?: return emptyList()
    return detail.prompts.flatMap { prompt ->
        prompt.images.mapIndexed { offset, name ->
            PreviewImage(
                path = jobImagePathFor(detail.id, detail.outputDir, name),
                title = name,
                // The seed doubles as the cache revision: a redraw keeps the name and changes
                // this, which is what makes the new bytes visible instead of the cached ones.
                rev = imageSeedAt(prompt, offset)?.toString().orEmpty(),
                caption = buildString {
                    prompt.seed?.let { append("seed $it") }
                    if (prompt.promptId.isNotEmpty()) {
                        if (isNotEmpty()) append("  ·  ")
                        append(prompt.promptId)
                    }
                    if (prompt.text.isNotEmpty()) {
                        if (isNotEmpty()) append("\n")
                        append(prompt.text.take(160))
                    }
                },
                maxEdge = 2048,
            )
        }
    }
}

@Composable
private fun JobRow(job: AutomationJobSummary, state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    val lang = state.language
    val selected = state.selectedJobId == job.id
    val elapsed = jobElapsedSeconds(job, Clock.System.now().toEpochMilliseconds())
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(rankoTokens.panel)
            .background(if (selected) colors.accentPink.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.04f))
            .clickable { viewModel.selectJob(job.id) }
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = stateGlyph(job.state),
                color = stateColor(job.state),
                fontSize = 12.sp,
                modifier = Modifier.width(14.dp),
            )
            Text(
                text = uiText(lang, jobStateKey(job.state)),
                color = stateColor(job.state),
                fontSize = 11.sp,
                modifier = Modifier.width(64.dp),
            )
            Text(
                text = job.id,
                color = colors.text,
                fontSize = 12.sp,
                modifier = Modifier.weight(1f),
            )
            Text(
                text = uiText(lang, "job_counts")
                    .replace("{done}", job.done.toString())
                    .replace("{total}", job.total.toString())
                    .replace("{images}", job.images.toString())
                    .replace("{elapsed}", formatElapsed(elapsed)),
                color = colors.textDim,
                fontSize = 11.sp,
            )
        }
        if (job.state == "running") {
            LinearProgressIndicator(
                progress = { jobProgress(job) },
                color = colors.accentPink,
                trackColor = colors.accentPink.copy(alpha = 0.18f),
                modifier = Modifier.fillMaxWidth().height(3.dp),
            )
        }
        // The bar covers the batch; a redraw/append needs its own line, or the row would look
        // finished while it works (its own counters do not move).
        job.pass?.let { pass ->
            Text(
                text = passLabel(lang, pass),
                color = colors.accentPink,
                fontSize = 10.sp,
            )
        }
        if (job.failed > 0 && job.error != null) {
            Text(text = job.error, color = colors.qualityRed, fontSize = 10.sp)
        }
    }
}

private fun jobStateKey(state: String): String = when (state) {
    "running" -> "filter_running"
    "done" -> "filter_done"
    "error" -> "filter_failed"
    "cancelled" -> "filter_cancelled"
    else -> "filter_all"
}

private fun stateGlyph(state: String): String = when (state) {
    "running" -> "●"
    "done" -> "✓"
    "error" -> "✕"
    else -> "—"
}

@Composable
private fun stateColor(state: String): Color {
    val colors = rankoColors
    return when (state) {
        "running" -> colors.accentBlue
        "done" -> colors.qualityGreen
        "error" -> colors.qualityRed
        else -> colors.textDim
    }
}

/** One prompt's images: a labelled row that wraps instead of scrolling sideways. */
@Composable
private fun PromptGalleryRow(
    prompt: JobPromptState,
    firstIndex: Int,
    outputDir: String,
    jobId: String,
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel,
    busy: Boolean,
) {
    val colors = rankoColors
    val lang = state.language
    Column(verticalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = "#${prompt.index + 1}",
                color = colors.accentPink,
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
            )
            Text(
                text = prompt.text.take(80),
                color = colors.textDim,
                fontSize = 11.sp,
                modifier = Modifier.weight(1f),
            )
            // With a seed per image, the row-level number would be the last pass's and would
            // speak for images it did not draw, so it only shows for a record that has no list.
            if (prompt.imageSeeds.isEmpty()) {
                prompt.seed?.let { Text(text = "seed $it", color = colors.textDim, fontSize = 10.sp) }
            }
            QuietTextButton(
                text = uiText(lang, "copy_prompt"),
                onClick = { copyTextToClipboard(prompt.text) },
            )
            QuietTextButton(
                text = uiText(lang, "edit_prompt"),
                enabled = !busy,
                onClick = { viewModel.openPromptEdit(jobId, prompt.index, prompt.text) },
            )
            QuietTextButton(
                text = uiText(lang, "add_images"),
                enabled = !busy,
                onClick = { viewModel.openPromptExtend(jobId, prompt.index) },
            )
        }
        // The pass running on this record shows itself here, on the record it changes: a redraw
        // does not move the job's own counters, and the group is where the user just clicked.
        state.jobDetail?.pass?.takeIf { it.promptIndex == prompt.index }?.let { pass ->
            JobPassLine(pass, lang)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            prompt.images.forEachIndexed { offset, name ->
                val index = firstIndex + offset
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    // The image's own two actions ride on the thumbnail: a redraw (new seed, over
                    // this file) top-left, a delete top-right. The preview overlay stays a viewer.
                    Box {
                        AsyncImage(
                            model = BlobRef(
                                jobImagePathFor(jobId, outputDir, name),
                                maxEdge = state.galleryThumbSize.toInt(),
                                quality = LocalThumbnailQuality.current,
                                // The record's seed for this image: a redraw writes the same name
                                // with new pixels, and only a changed revision makes the cache
                                // fetch them instead of the bytes it read the first time.
                                rev = imageSeedAt(prompt, offset)?.toString().orEmpty(),
                            ),
                            contentDescription = name,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .width(state.galleryThumbSize.dp)
                                .height(state.galleryThumbSize.dp)
                                .clip(rankoTokens.panel)
                                .background(colors.bgCard.copy(alpha = 0.45f))
                                .clickable { viewModel.openGalleryPreview(index) },
                        )
                        ThumbnailActionBadge(
                            icon = Icons.Default.Refresh,
                            description = uiText(lang, "regenerate_image"),
                            enabled = !busy,
                            modifier = Modifier.align(Alignment.TopStart).padding(4.dp),
                            onClick = {
                                viewModel.confirmImageAction(
                                    jobId,
                                    prompt.index,
                                    name,
                                    GalleryImageAction.Regenerate,
                                )
                            },
                        )
                        ThumbnailActionBadge(
                            icon = Icons.Default.Close,
                            description = uiText(lang, "delete_image"),
                            enabled = !busy,
                            danger = true,
                            modifier = Modifier.align(Alignment.TopEnd).padding(4.dp),
                            onClick = {
                                viewModel.confirmImageAction(
                                    jobId,
                                    prompt.index,
                                    name,
                                    GalleryImageAction.Delete,
                                )
                            },
                        )
                        ThumbnailActionBadge(
                            icon = Icons.Default.Save,
                            description = uiText(lang, "save_image"),
                            enabled = true,
                            accent = colors.accentBlue,
                            modifier = Modifier.align(Alignment.BottomEnd).padding(4.dp),
                            onClick = { viewModel.downloadImage(name) },
                        )
                    }
                    Text(text = name, color = colors.textDim, fontSize = 9.sp)
                    imageSeedAt(prompt, offset)?.let { seed ->
                        Text(text = "seed $seed", color = colors.textDim, fontSize = 9.sp)
                    }
                }
            }
        }
        if (prompt.state == "error" && prompt.error != null) {
            Text(text = prompt.error, color = colors.qualityRed, fontSize = 10.sp)
        }
    }
}

/** The prompt entry (its index and record) one image belongs to, or null when none names it. */
internal fun promptEntryFor(detail: AutomationJobDetail?, image: String): Pair<Int, JobPromptState>? {
    val prompts = detail?.prompts ?: return null
    val index = prompts.indexOfFirst { prompt -> prompt.images.any { it == image } }
    return if (index < 0) null else index to prompts[index]
}

/**
 * One line for a targeted pass: a bar and what it is doing. A redraw names the image it will
 * replace; an append counts the images it has written so far.
 */
@Composable
private fun JobPassLine(pass: JobPass, lang: PromptLang) {
    val colors = rankoColors
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        LinearProgressIndicator(
            progress = { jobPassProgress(pass) },
            color = colors.accentPink,
            trackColor = colors.accentPink.copy(alpha = 0.18f),
            modifier = Modifier.weight(1f).height(4.dp),
        )
        Text(text = passLabel(lang, pass), color = colors.accentPink, fontSize = 11.sp)
    }
}

/** What a pass is doing, in the page's own words. */
private fun passLabel(lang: PromptLang, pass: JobPass): String = when (pass.mode) {
    "image" -> uiText(lang, "pass_redraw").replace("{image}", pass.image ?: "")
    "append" -> uiText(lang, "pass_add")
        .replace("{done}", pass.imagesDone.toString())
        .replace("{total}", pass.totalImages.toString())
    "append_all" -> uiText(lang, "pass_add_all")
        .replace("{done}", pass.imagesDone.toString())
        .replace("{total}", pass.totalImages.toString())
    else -> uiText(lang, "pass_waiting")
}

/**
 * One action hanging in a thumbnail's corner: a small disc over the image, so the button belongs to
 * the image it acts on. `enabled = false` (a job that runs, an action in flight) leaves it visible
 * but inert, like every other control on this page.
 */
@Composable
private fun ThumbnailActionBadge(
    icon: ImageVector,
    description: String,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    danger: Boolean = false,
    accent: Color? = null,
    onClick: () -> Unit,
) {
    val colors = rankoColors
    Box(
        modifier = modifier
            .size(22.dp)
            .clip(CircleShape)
            .background(colors.boardBg.copy(alpha = if (enabled) 0.78f else 0.5f))
            .clickable(enabled = enabled, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = description,
            tint = when {
                !enabled -> colors.textDim.copy(alpha = 0.5f)
                danger -> colors.qualityRed
                accent != null -> accent
                else -> colors.accentPink
            },
            modifier = Modifier.size(14.dp),
        )
    }
}

private fun formatElapsed(seconds: Double): String {
    val total = seconds.toInt()
    val minutes = total / 60
    val rest = total % 60
    return if (minutes > 0) "${minutes}m ${rest.toString().padStart(2, '0')}s" else "${rest}s"
}

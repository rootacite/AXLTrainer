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
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import coil3.compose.AsyncImage
import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.data.JobPromptState
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.JobFilter
import com.acite.axlranko.model.jobElapsedSeconds
import com.acite.axlranko.model.jobImagePathFor
import com.acite.axlranko.model.jobProgress
import com.acite.axlranko.pages.AutomationScreenViewModel
import com.acite.axlranko.pages.components.ImagePreviewOverlay
import com.acite.axlranko.pages.components.PreviewImage
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
                }
                state.visibleJobs.forEach { job ->
                    JobRow(job, state, viewModel)
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
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = detail.id,
                            color = colors.text,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 13.sp,
                            modifier = Modifier.weight(1f),
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

    }

        state.galleryPreviewIndex?.let { index ->
            val images = previewImages(state)
            if (images.isNotEmpty()) {
                val current = images[index.coerceIn(images.indices)]
                ImagePreviewOverlay(
                    images = images,
                    index = index.coerceIn(images.indices),
                    onClose = viewModel::closeGalleryPreview,
                    onPrev = viewModel::previewPrev,
                    onNext = viewModel::previewNext,
                    actions = {
                        CapsuleButton(
                            text = uiText(lang, "save_image"),
                            onClick = { viewModel.downloadImage(current.title) },
                            compact = true,
                        )
                        CapsuleButton(
                            text = uiText(lang, "copy_prompt"),
                            onClick = {
                                val prompt = state.jobDetail?.prompts?.firstOrNull { candidate ->
                                    candidate.images.any { it == current.title }
                                }
                                copyTextToClipboard(prompt?.text.orEmpty())
                            },
                            compact = true,
                        )
                    },
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
        prompt.images.map { name ->
            PreviewImage(
                path = jobImagePathFor(detail.id, detail.outputDir, name),
                title = name,
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
                modifier = Modifier.fillMaxWidth().height(3.dp),
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
) {
    val colors = rankoColors
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
            prompt.seed?.let { Text(text = "seed $it", color = colors.textDim, fontSize = 10.sp) }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            prompt.images.forEachIndexed { offset, name ->
                val index = firstIndex + offset
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    AsyncImage(
                        model = BlobRef(
                            jobImagePathFor(jobId, outputDir, name),
                            maxEdge = state.galleryThumbSize.toInt(),
                            quality = LocalThumbnailQuality.current,
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
                    Text(text = name, color = colors.textDim, fontSize = 9.sp)
                }
            }
        }
        if (prompt.state == "error" && prompt.error != null) {
            Text(text = prompt.error, color = colors.qualityRed, fontSize = 10.sp)
        }
    }
}

private fun formatElapsed(seconds: Double): String {
    val total = seconds.toInt()
    val minutes = total / 60
    val rest = total % 60
    return if (minutes > 0) "${minutes}m ${rest.toString().padStart(2, '0')}s" else "${rest}s"
}

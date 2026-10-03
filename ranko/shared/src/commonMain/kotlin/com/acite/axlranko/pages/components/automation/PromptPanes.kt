package com.acite.axlranko.pages.components.automation

import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.PromptProfileItem
import com.acite.axlranko.model.PromptView
import com.acite.axlranko.model.promptProfileLabel
import com.acite.axlranko.pages.AutomationScreenViewModel
import com.acite.axlranko.prompt.ManifestModel
import com.acite.axlranko.prompt.PromptMode
import com.acite.axlranko.prompt.trimNumber
import com.acite.axlranko.prompt.t
import com.acite.axlranko.ui.SingleLineOrStacked
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.QuietTextButton
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.pointerIconHand
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import kotlin.time.Instant
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime

/** The whole Prompts area: shared resources, the wizard or the configuration list, and the results. */
@Composable
fun PromptsPane(
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel,
    warning: String?,
    portrait: Boolean = false,
) {
    val colors = rankoColors
    Column(
        modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        PromptResourceBar(state, viewModel)

        state.notice?.let { note ->
            Text(text = note, color = colors.accentBlue, fontSize = 12.sp)
        }
        (state.matrixError ?: state.profileError)?.let { message ->
            Text(text = message, color = colors.qualityRed, fontSize = 12.sp)
        }

        PromptProfilesCard(state, viewModel)

        SingleLineOrStacked(
            modifier = Modifier.fillMaxWidth(),
            first = {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    CapsuleChoice(
                        text = uiText(state.language, "wizard"),
                        selected = state.view == PromptView.Wizard,
                        onClick = { viewModel.setView(PromptView.Wizard) },
                    )
                    CapsuleChoice(
                        text = uiText(state.language, "manifest"),
                        selected = state.view == PromptView.Manifest,
                        onClick = { viewModel.setView(PromptView.Manifest) },
                    )
                }
            },
            second = {
                Text(
                    text = uiText(state.language, "mode_count")
                        .replace("{mode}", state.spec.mode.wire)
                        .replace("{count}", state.spec.count.toString()),
                    color = colors.textDim,
                    fontSize = 11.sp,
                    maxLines = 1,
                    softWrap = false,
                    overflow = TextOverflow.Ellipsis,
                )
            },
        )

        if (state.view == PromptView.Wizard) {
            PromptWizardView(state, viewModel, warning, portrait)
        } else {
            PromptManifestView(state, viewModel)
        }

        PromptResultsCard(state, viewModel, portrait)
    }

    if (state.saveDialogOpen) PromptSaveDialog(state, viewModel)
    state.editingRow?.let { row ->
        PromptRowEditorDialog(rowKey = row, state = state, viewModel = viewModel, warning = warning)
    }
}

@Composable
private fun PromptResourceBar(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    PorcelainCard {
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = uiText(state.language, "matrix"),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp,
                )
                Text(
                    text = when {
                        state.loadingMatrix -> "…"
                        state.matrixError != null -> state.matrixError
                        state.matrix != null -> "${state.matrixPath} · ${state.matrixLines} lines"
                        else -> "—"
                    },
                    color = if (state.matrixError != null) colors.qualityRed else colors.textDim,
                    fontSize = 11.sp,
                )
            }
            CapsuleButton(
                text = uiText(state.language, "reload"),
                onClick = viewModel::loadMatrix,
                compact = true,
            )
        }
    }
}

@Composable
private fun PromptProfilesCard(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = t(state.language, "profile_title"),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp,
                    modifier = Modifier.weight(1f),
                )
                if (state.loadingProfiles) Text("…", color = colors.textDim, fontSize = 11.sp)
                QuietTextButton(
                    text = uiText(state.language, "refresh"),
                    onClick = viewModel::refreshProfiles,
                )
            }
            if (state.profiles.isEmpty() && !state.loadingProfiles) {
                Text(
                    text = uiText(state.language, "profiles_empty"),
                    color = colors.textDim,
                    fontSize = 11.sp,
                )
            }
            Column(
                modifier = Modifier.fillMaxWidth().heightIn(max = 200.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                state.profiles.forEach { item -> PromptProfileRow(item, state, viewModel) }
            }
            if (state.activeProfile.isNotEmpty()) {
                Text(
                    text = uiText(state.language, "active_profile") + state.activeProfile,
                    color = colors.accentBlue,
                    fontSize = 11.sp,
                )
            }
        }
    }
}

@Composable
private fun PromptProfileRow(
    item: PromptProfileItem,
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel,
) {
    val colors = rankoColors
    val selected = state.activeProfile == item.name
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(rankoTokens.panel)
            .background(if (selected) colors.accentPink.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.04f))
            .clickable(enabled = item.error == null) { viewModel.loadProfile(item.name) }
            .pointerHoverIcon(pointerIconHand)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = promptProfileLabel(item), color = colors.text, fontSize = 12.sp)
            val detail = buildString {
                if (item.error != null) {
                    append(item.error)
                } else {
                    append(formatProfileStamp(item.modified))
                    append(" · ")
                    append(formatProfileSize(item.size))
                }
                if (item.notes.isNotEmpty()) {
                    append("  ·  ")
                    append(item.notes.joinToString("；"))
                }
            }
            Text(text = detail, color = colors.textDim, fontSize = 10.sp)
        }
        CapsuleButton(
            text = uiText(state.language, "load"),
            onClick = { viewModel.loadProfile(item.name) },
            enabled = item.error == null,
            compact = true,
        )
        CapsuleButton(
            text = uiText(state.language, "delete"),
            onClick = { viewModel.deleteProfile(item.name) },
            danger = true,
            compact = true,
        )
    }
}

@Composable
private fun PromptWizardView(
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel,
    warning: String?,
    portrait: Boolean,
) {
    val colors = rankoColors
    val keys = state.applicablePageKeys
    val index = state.visiblePageIndex
    val pageKey = state.currentPageKey
    val page: @Composable () -> Unit = {
        PorcelainCard {
            Column(
                modifier = Modifier.fillMaxWidth().padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                if (pageKey != null) PromptPageHeader(pageKey, state.language)
                warning?.let { PromptWarningBanner(it) }
                if (pageKey == null) {
                    Text(
                        text = uiText(state.language, "matrix_missing"),
                        color = colors.textDim,
                        fontSize = 12.sp,
                    )
                } else {
                    PromptPageEditor(
                        pageKey = pageKey,
                        spec = state.spec,
                        matrix = state.matrix,
                        lang = state.language,
                        seedText = state.seedText,
                        actions = editorActions(viewModel),
                    )
                }
                PromptWizardFooter(
                    atStart = index == 0,
                    atEnd = index >= keys.size - 1,
                    lang = state.language,
                    onBack = viewModel::previousPage,
                    onNext = viewModel::nextPage,
                )
            }
        }
    }
    if (portrait) {
        Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            PromptStepRail(
                keys = keys,
                currentIndex = index,
                lang = state.language,
                onSelect = viewModel::goToPage,
                horizontal = true,
            )
            page()
        }
    } else {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            Column(modifier = Modifier.width(150.dp)) {
                PromptStepRail(keys = keys, currentIndex = index, lang = state.language, onSelect = viewModel::goToPage)
            }
            page()
        }
    }
}

@Composable
private fun PromptManifestView(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = t(state.language, "manifest_title"),
                color = colors.text,
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp,
            )
            ManifestModel.items(state.spec, state.language).forEach { row ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(rankoTokens.panel)
                        .background(Color.White.copy(alpha = 0.04f))
                        .clickable { viewModel.openManifestRow(row.pageKey) }
                        .pointerHoverIcon(pointerIconHand)
                        .padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text(
                        text = row.label,
                        color = colors.textDim,
                        fontSize = 12.sp,
                        modifier = Modifier.width(130.dp),
                    )
                    Text(
                        text = row.value,
                        color = colors.text,
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier.weight(1f),
                    )
                    Text(text = "✎", color = colors.accentLilac, fontSize = 12.sp)
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                CapsuleButton(
                    text = t(state.language, "manifest_go"),
                    onClick = viewModel::generate,
                    emphasized = true,
                )
                CapsuleButton(
                    text = t(state.language, "manifest_save"),
                    onClick = viewModel::openSaveDialog,
                )
                CapsuleButton(
                    text = uiText(state.language, "wizard"),
                    onClick = { viewModel.setView(PromptView.Wizard) },
                )
                Text(
                    text = "${t(state.language, "item_count")}: ${state.spec.count}",
                    color = colors.textDim,
                    fontSize = 11.sp,
                )
            }
            state.generateError?.let { message ->
                Text(text = message, color = colors.qualityRed, fontSize = 12.sp)
            }
            state.warnings.forEach { note ->
                Text(text = note, color = colors.qualityYellow, fontSize = 11.sp)
            }
        }
    }
}

/** The generated lines, with the three exports the plan asked for. */
@Composable
private fun PromptResultsCard(
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel,
    portrait: Boolean,
) {
    val colors = rankoColors
    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (state.results.isEmpty() || !portrait) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = uiText(state.language, "results"),
                        color = colors.text,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 13.sp,
                        maxLines = 1,
                        softWrap = false,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f),
                    )
                    if (state.results.isNotEmpty()) {
                        Text(
                            text = uiText(state.language, "results_hint")
                                .replace("{n}", state.results.size.toString())
                                .replace("{seed}", (state.resultSeed ?: "—").toString()),
                            color = colors.textDim,
                            fontSize = 11.sp,
                            maxLines = 1,
                            softWrap = false,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            } else {
                SingleLineOrStacked(
                    modifier = Modifier.fillMaxWidth(),
                    first = {
                        Text(
                            text = uiText(state.language, "results"),
                            color = colors.text,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 13.sp,
                            maxLines = 1,
                            softWrap = false,
                            overflow = TextOverflow.Ellipsis,
                        )
                    },
                    second = {
                        Text(
                            text = uiText(state.language, "results_hint")
                                .replace("{n}", state.results.size.toString())
                                .replace("{seed}", (state.resultSeed ?: "—").toString()),
                            color = colors.textDim,
                            fontSize = 11.sp,
                            maxLines = 1,
                            softWrap = false,
                            overflow = TextOverflow.Ellipsis,
                        )
                    },
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CapsuleButton(
                    text = uiText(state.language, "generate"),
                    onClick = viewModel::generate,
                    enabled = state.matrix != null,
                    emphasized = true,
                )
                CapsuleButton(
                    text = uiText(state.language, "copy_all"),
                    onClick = viewModel::copyAll,
                    enabled = state.results.isNotEmpty(),
                )
                CapsuleButton(
                    text = uiText(state.language, "download"),
                    onClick = viewModel::download,
                    enabled = state.results.isNotEmpty(),
                )
                CapsuleButton(
                    text = uiText(state.language, "send_to_batch"),
                    onClick = viewModel::useAsBatchInput,
                    enabled = state.results.isNotEmpty(),
                )
                if (state.results.isNotEmpty()) {
                    QuietTextButton(
                        text = uiText(state.language, "clear"),
                        onClick = viewModel::clearResults,
                    )
                }
            }
            if (state.results.isEmpty()) {
                Text(
                    text = uiText(state.language, "results_empty"),
                    color = colors.textDim,
                    fontSize = 11.sp,
                )
            }
            Column(
                modifier = Modifier.fillMaxWidth().heightIn(max = 420.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                state.results.forEachIndexed { index, line ->
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            text = (index + 1).toString(),
                            color = colors.textDim,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.width(28.dp),
                        )
                        Text(
                            text = line,
                            color = colors.text,
                            fontSize = 12.sp,
                            modifier = Modifier.weight(1f),
                        )
                        QuietTextButton(text = "⧉", onClick = { viewModel.copyLine(line) })
                    }
                }
            }
            if (state.batchInput.isNotEmpty()) {
                Text(
                    text = uiText(state.language, "batch_queue")
                        .replace("{n}", state.batchInput.size.toString()),
                    color = colors.accentBlue,
                    fontSize = 11.sp,
                )
            }
        }
    }
}

@Composable
private fun PromptRowEditorDialog(
    rowKey: String,
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel,
    warning: String?,
) {
    Dialog(onDismissRequest = viewModel::closeManifestRow) {
        PorcelainCard {
            Column(
                modifier = Modifier.width(720.dp).heightIn(max = 720.dp).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                PromptPageHeader(rowKey, state.language)
                warning?.let { PromptWarningBanner(it) }
                Column(
                    modifier = Modifier.fillMaxWidth().heightIn(max = 520.dp).verticalScroll(rememberScrollState()),
                ) {
                    PromptPageEditor(
                        pageKey = rowKey,
                        spec = state.spec,
                        matrix = state.matrix,
                        lang = state.language,
                        seedText = state.seedText,
                        actions = editorActions(viewModel),
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CapsuleButton(
                        text = uiText(state.language, "done"),
                        onClick = viewModel::closeManifestRow,
                        emphasized = true,
                    )
                }
            }
        }
    }
}

@Composable
private fun PromptSaveDialog(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    Dialog(onDismissRequest = viewModel::closeSaveDialog) {
        PorcelainCard {
            Column(
                modifier = Modifier.width(420.dp).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = t(state.language, "profile_save_title"),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = t(state.language, "profile_save_hint"),
                    color = colors.textDim,
                    fontSize = 11.sp,
                )
                OutlinedTextField(
                    value = state.saveName,
                    onValueChange = viewModel::setSaveName,
                    singleLine = true,
                    colors = rankoFieldColors(),
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Switch(checked = state.saveOverwrite, onCheckedChange = viewModel::setSaveOverwrite)
                    Text(
                        text = t(state.language, "profile_overwrite_yes"),
                        color = colors.text,
                        fontSize = 12.sp,
                    )
                }
                state.saveError?.let { message ->
                    Text(text = message, color = colors.qualityRed, fontSize = 12.sp)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CapsuleButton(
                        text = uiText(state.language, "save"),
                        onClick = viewModel::saveProfile,
                        emphasized = true,
                        enabled = !state.busy,
                    )
                    CapsuleButton(
                        text = uiText(state.language, "cancel"),
                        onClick = viewModel::closeSaveDialog,
                    )
                }
            }
        }
    }
}

private fun editorActions(viewModel: AutomationScreenViewModel): PromptEditorActions = PromptEditorActions(
    setCharacter = viewModel::setCharacter,
    setMode = viewModel::setMode,
    setExposure = viewModel::setExposure,
    setClothingAny = viewModel::setClothingAny,
    toggleClothing = viewModel::toggleClothing,
    setChest = viewModel::setChest,
    setBelly = viewModel::setBelly,
    setFigure = viewModel::setFigure,
    setPussyShape = viewModel::setPussyShape,
    setPussyHair = viewModel::setPussyHair,
    setFaceGroup = viewModel::setFaceGroup,
    toggleFaceTag = viewModel::toggleFaceTag,
    setSceneAny = viewModel::setSceneAny,
    toggleScene = viewModel::toggleScene,
    setFamilyAny = viewModel::setFamilyAny,
    toggleFamily = viewModel::toggleFamily,
    setVaginalRatio = viewModel::setVaginalRatio,
    setStageWeight = viewModel::setStageWeight,
    setPoseAny = viewModel::setPoseAny,
    togglePose = viewModel::togglePose,
    setCount = viewModel::setCount,
    setSeedText = viewModel::setSeedText,
)

private fun formatProfileStamp(millis: Long): String {
    if (millis <= 0) return "—"
    val local = Instant.fromEpochMilliseconds(millis).toLocalDateTime(TimeZone.currentSystemDefault())
    val hour = local.hour.toString().padStart(2, '0')
    val minute = local.minute.toString().padStart(2, '0')
    return "${local.date} $hour:$minute"
}

private fun formatProfileSize(bytes: Long): String = when {
    bytes <= 0 -> "—"
    bytes < 1024 -> "$bytes B"
    bytes < 1024 * 1024 -> "${bytes / 1024} KB"
    else -> "${trimNumber(bytes / 1024.0 / 1024.0)} MB"
}

package com.acite.axlranko.pages.components.automation

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.jobProgress
import com.acite.axlranko.model.PromptSource
import com.acite.axlranko.pages.AutomationScreenViewModel
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.QuietTextButton
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.pickClientTextFile
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlinx.coroutines.launch

/** The ComfyUI section: the server, the workflow, and the batch it runs. */
@Composable
fun ComfyPane(
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel = metroViewModel(),
) {
    Column(
        modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        ServerCard(state, viewModel)
        WorkflowCard(state, viewModel)
        BatchCard(state, viewModel)
        JobLogCard(state, viewModel)
    }
}

@Composable
private fun ServerCard(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    val lang = state.language
    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = uiText(lang, "server"),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp,
                    modifier = Modifier.weight(1f),
                )
                if (state.discovering) {
                    CircularProgressIndicator(modifier = Modifier.size(14.dp))
                } else if (state.comfy.found) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = colors.qualityGreen, modifier = Modifier.size(16.dp))
                    Text(
                        text = "${uiText(lang, "connected")} ${state.comfy.version}",
                        color = colors.qualityGreen,
                        fontSize = 12.sp,
                    )
                } else {
                    Icon(Icons.Default.RadioButtonUnchecked, contentDescription = null, tint = colors.textDim, modifier = Modifier.size(16.dp))
                    Text(text = uiText(lang, "not_found"), color = colors.textDim, fontSize = 12.sp)
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = state.settings.server,
                    onValueChange = viewModel::setSettingsServer,
                    singleLine = true,
                    placeholder = { Text("http://127.0.0.1:8188", fontSize = 12.sp) },
                    colors = rankoFieldColors(),
                    modifier = Modifier.weight(1f),
                )
                CapsuleButton(
                    text = if (state.discovering) uiText(lang, "discovering") else uiText(lang, "discover"),
                    onClick = viewModel::discoverComfy,
                    enabled = !state.discovering,
                    compact = true,
                )
            }
            Text(text = uiText(lang, "server_hint"), color = colors.textDim, fontSize = 11.sp)
            if (state.comfy.found) {
                Text(
                    text = uiText(lang, "queue_line")
                        .replace("{running}", state.comfy.queueRunning.toString())
                        .replace("{pending}", state.comfy.queuePending.toString()),
                    color = colors.textDim,
                    fontSize = 11.sp,
                )
            } else if (state.comfy.checked.isNotEmpty()) {
                Text(
                    text = state.comfy.checked.takeLast(3).joinToString(" · ") { "${it.url} — ${it.reason}" },
                    color = colors.textDim,
                    fontSize = 10.sp,
                )
            }
            state.discoverError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
        }
    }
}

@Composable
private fun WorkflowCard(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    val lang = state.language
    val scope = rememberCoroutineScope()
    val active = state.activeWorkflow

    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = uiText(lang, "workflow"),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp,
                    modifier = Modifier.weight(1f),
                )
                if (state.workflowsLoading) CircularProgressIndicator(modifier = Modifier.size(14.dp))
                CapsuleButton(
                    text = uiText(lang, "upload_workflow"),
                    onClick = {
                        scope.launch {
                            val picked = pickClientTextFile("ComfyUI API workflow", listOf("json"))
                            if (picked != null) viewModel.uploadWorkflow(picked.text, picked.name)
                        }
                    },
                    compact = true,
                )
            }
            Text(text = uiText(lang, "upload_failed_hint"), color = colors.textDim, fontSize = 10.sp)

            if (state.workflows.isEmpty()) {
                Text(text = uiText(lang, "no_workflows"), color = colors.textDim, fontSize = 11.sp)
            } else {
                Column(
                    modifier = Modifier.fillMaxWidth().heightIn(max = 180.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    state.workflows.forEach { workflow ->
                        val selected = state.settings.workflow == workflow.path || state.settings.workflow == workflow.name
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(rankoTokens.panel)
                                .background(if (selected) colors.accentPink.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.04f))
                                .clickable { viewModel.selectWorkflow(workflow) }
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(workflow.name, color = colors.text, fontSize = 12.sp)
                                Text(
                                    text = if (workflow.valid) {
                                        uiText(lang, "nodes_line")
                                            .replace("{nodes}", workflow.nodeCount.toString())
                                            .replace("{save}", workflow.saveImageNodes.size.toString())
                                            .replace("{batch}", workflow.batchSizeNodes.size.toString())
                                    } else {
                                        workflow.error ?: ""
                                    },
                                    color = if (workflow.valid) colors.textDim else colors.qualityRed,
                                    fontSize = 10.sp,
                                )
                            }
                            CapsuleButton(
                                text = uiText(lang, "validate"),
                                onClick = { viewModel.validateWorkflow(workflow.path) },
                                compact = true,
                            )
                            CapsuleButton(
                                text = uiText(lang, "delete"),
                                onClick = { viewModel.deleteWorkflow(workflow.name) },
                                danger = true,
                                compact = true,
                            )
                        }
                    }
                }
            }

            if (active != null && active.valid) {
                val missing = active.missingModels
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (missing.isEmpty()) {
                        Icon(Icons.Default.CheckCircle, contentDescription = null, tint = colors.qualityGreen, modifier = Modifier.size(14.dp))
                        Text(
                            text = if (state.workflowCheck) uiText(lang, "model_check_ok") else uiText(lang, "model_check_skipped"),
                            color = colors.textDim,
                            fontSize = 11.sp,
                        )
                    } else {
                        Icon(Icons.Default.Warning, contentDescription = null, tint = colors.qualityYellow, modifier = Modifier.size(14.dp))
                        Text(
                            text = "${uiText(lang, "missing_models")}: " +
                                missing.take(3).joinToString(", ") { "${it.value} (${it.classType})" },
                            color = colors.qualityYellow,
                            fontSize = 11.sp,
                        )
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(text = uiText(lang, "positive_node"), color = colors.textDim, fontSize = 11.sp)
                    if (active.positiveNodeGuessed) {
                        Text(text = uiText(lang, "positive_node_guessed"), color = colors.textDim, fontSize = 10.sp)
                    }
                }
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    active.textNodes.forEach { node ->
                        val selected = state.settings.positiveNode == node.id
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(rankoTokens.panel)
                                .background(if (selected) colors.accentPink.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.04f))
                                .clickable { viewModel.setSettingsPositiveNode(node.id) }
                                .padding(horizontal = 10.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Text(node.id, color = colors.accentPink, fontSize = 11.sp, fontFamily = FontFamily.Monospace, modifier = Modifier.width(64.dp))
                            Text(
                                text = node.text.take(70),
                                color = colors.text,
                                fontSize = 11.sp,
                                modifier = Modifier.weight(1f),
                            )
                        }
                    }
                }
            }
            state.workflowError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
            state.settingsNotice?.let { Text(it, color = colors.accentBlue, fontSize = 11.sp) }
        }
    }
}

@Composable
private fun BatchCard(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    val lang = state.language
    val scope = rememberCoroutineScope()
    var promptSetDraft by remember { mutableStateOf("") }

    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = uiText(lang, "batch"),
                color = colors.text,
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp,
            )
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(text = uiText(lang, "prompt_source"), color = colors.textDim, fontSize = 11.sp)
                CapsuleChoice(
                    text = uiText(lang, "source_results").replace("{n}", state.results.size.toString()),
                    selected = state.promptSource == PromptSource.CurrentResults,
                    onClick = { viewModel.setPromptSource(PromptSource.CurrentResults) },
                )
                CapsuleChoice(
                    text = uiText(lang, "source_set"),
                    selected = state.promptSource == PromptSource.SavedSet,
                    onClick = { viewModel.setPromptSource(PromptSource.SavedSet) },
                )
                CapsuleChoice(
                    text = uiText(lang, "source_manual"),
                    selected = state.promptSource == PromptSource.Manual,
                    onClick = { viewModel.setPromptSource(PromptSource.Manual) },
                )
            }

            when (state.promptSource) {
                PromptSource.SavedSet -> {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (state.promptSets.isEmpty()) {
                            Text(text = uiText(lang, "no_prompt_sets"), color = colors.textDim, fontSize = 11.sp)
                        } else {
                            state.promptSets.forEach { set ->
                                CapsuleChoice(
                                    text = "${set.name} (${set.count})",
                                    selected = state.promptSetName == set.name,
                                    onClick = { viewModel.setPromptSetName(set.name) },
                                )
                            }
                        }
                        QuietTextButton(text = uiText(lang, "refresh"), onClick = viewModel::refreshPromptSets)
                    }
                }
                PromptSource.Manual -> OutlinedTextField(
                    value = state.manualPrompts,
                    onValueChange = viewModel::setManualPrompts,
                    placeholder = { Text(uiText(lang, "manual_hint"), fontSize = 12.sp) },
                    colors = rankoFieldColors(),
                    modifier = Modifier.fillMaxWidth().heightIn(min = 96.dp),
                )
                PromptSource.CurrentResults -> Text(
                    text = uiText(lang, "results_hint")
                        .replace("{n}", state.results.size.toString())
                        .replace("{seed}", (state.resultSeed ?: "—").toString()),
                    color = colors.textDim,
                    fontSize = 11.sp,
                )
            }

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Column(modifier = Modifier.width(120.dp)) {
                    Text(uiText(lang, "images_per_prompt"), color = colors.textDim, fontSize = 10.sp)
                    OutlinedTextField(
                        value = state.settings.count,
                        onValueChange = viewModel::setSettingsCount,
                        singleLine = true,
                        colors = rankoFieldColors(),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Column(modifier = Modifier.width(120.dp)) {
                    Text(uiText(lang, "poll_interval"), color = colors.textDim, fontSize = 10.sp)
                    OutlinedTextField(
                        value = state.settings.poll,
                        onValueChange = viewModel::setSettingsPoll,
                        singleLine = true,
                        colors = rankoFieldColors(),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(uiText(lang, "output_dir"), color = colors.textDim, fontSize = 10.sp)
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        OutlinedTextField(
                            value = state.settings.outputDir,
                            onValueChange = viewModel::setSettingsOutputDir,
                            singleLine = true,
                            colors = rankoFieldColors(),
                            modifier = Modifier.weight(1f),
                        )
                        CapsuleButton(
                            text = uiText(lang, "browse"),
                            onClick = { scope.launch { viewModel.browseOutputDir() } },
                            compact = true,
                        )
                    }
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CapsuleButton(
                    text = if (state.startingJob) "…" else uiText(lang, "start_job"),
                    onClick = viewModel::startJob,
                    enabled = !state.startingJob,
                    emphasized = true,
                )
                CapsuleButton(
                    text = uiText(lang, "save_settings"),
                    onClick = viewModel::saveAutomationSettings,
                    enabled = !state.settingsSaving,
                )
                val running = state.runningJob
                if (running != null) {
                    Text(
                        text = "${running.id} · ${running.done + running.failed}/${running.total}",
                        color = colors.accentBlue,
                        fontSize = 11.sp,
                    )
                    CapsuleButton(
                        text = uiText(lang, "cancel_job"),
                        onClick = { viewModel.cancelJob(running.id) },
                        danger = true,
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = promptSetDraft,
                    onValueChange = { promptSetDraft = it },
                    singleLine = true,
                    placeholder = { Text(uiText(lang, "set_name"), fontSize = 12.sp) },
                    colors = rankoFieldColors(),
                    modifier = Modifier.width(200.dp),
                )
                CapsuleButton(
                    text = uiText(lang, "save_prompt_set"),
                    onClick = { if (promptSetDraft.isNotBlank()) viewModel.savePromptSet(promptSetDraft.trim()) },
                    enabled = promptSetDraft.isNotBlank() &&
                        (state.results.isNotEmpty() || state.manualPrompts.isNotBlank()),
                    compact = true,
                )
                if (state.promptSource == PromptSource.SavedSet && state.promptSetName.isNotBlank()) {
                    CapsuleButton(
                        text = uiText(lang, "delete"),
                        onClick = { viewModel.deletePromptSet(state.promptSetName) },
                        danger = true,
                        compact = true,
                    )
                }
            }

            state.jobError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
            state.settingsError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
            state.promptSetError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
            state.runningJob?.let { running ->
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Slider(
                        value = jobProgress(running),
                        onValueChange = {},
                        enabled = false,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        text = "${running.done + running.failed}/${running.total}",
                        color = colors.textDim,
                        fontSize = 11.sp,
                    )
                }
            }
        }
    }
}

@Composable
private fun JobLogCard(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    val lang = state.language
    val detail = state.jobDetail ?: return
    if (detail.logTail.isBlank()) return
    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = uiText(lang, "log"),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.sp,
                    modifier = Modifier.weight(1f),
                )
                QuietTextButton(text = uiText(lang, "refresh"), onClick = { viewModel.selectJob(detail.id) })
            }
            Text(
                text = detail.logTail,
                color = colors.textDim,
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
            )
        }
    }
}

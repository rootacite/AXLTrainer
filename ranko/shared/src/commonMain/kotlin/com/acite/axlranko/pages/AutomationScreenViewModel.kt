package com.acite.axlranko.pages

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.AutomationSettings
import com.acite.axlranko.data.AutomationJobDetail
import com.acite.axlranko.data.AutomationWorkflow
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.trainDataEntries
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.data.decodeBase64
import com.acite.axlranko.model.AutomationSection
import com.acite.axlranko.model.AutomationSettingsDraft
import com.acite.axlranko.model.GalleryImageAction
import com.acite.axlranko.model.GalleryImagePrompt
import com.acite.axlranko.model.GalleryImageRef
import com.acite.axlranko.model.JobFilter
import com.acite.axlranko.model.PromptEditDraft
import com.acite.axlranko.model.PromptExtendDraft
import com.acite.axlranko.model.PromptSource
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.PromptProfileItem
import com.acite.axlranko.model.PromptView
import com.acite.axlranko.model.promptExportFileName
import com.acite.axlranko.model.promptExportText
import com.acite.axlranko.pages.components.automation.uiText
import com.acite.axlranko.prompt.EXPOSURE_LEVELS
import com.acite.axlranko.prompt.FACE_GROUPS
import com.acite.axlranko.prompt.FacePick
import com.acite.axlranko.prompt.PoseFamily
import com.acite.axlranko.prompt.PromptGenerator
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.prompt.PromptLimits
import com.acite.axlranko.prompt.PromptMode
import com.acite.axlranko.prompt.PromptSpec
import com.acite.axlranko.prompt.ProfileCodec
import com.acite.axlranko.prompt.ProfileException
import com.acite.axlranko.prompt.SexStage
import com.acite.axlranko.prompt.WizardModel
import com.acite.axlranko.prompt.parseMatrix
import com.acite.axlranko.prompt.t
import com.acite.axlranko.util.PathPicker
import com.acite.axlranko.util.copyTextToClipboard
import com.acite.axlranko.util.saveClientFile
import com.acite.axlranko.util.saveClientText
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlin.random.Random
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * The Automation page. Prompts are assembled in `commonMain` (`prompt/`), so this only moves state
 * and talks to the helper for the two shared resources: `input_matrix.txt` and `prompt_profiles/`.
 */
@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class AutomationScreenViewModel(
    private val ipc: TrainerIpcClient,
    private val refreshHub: DatasetRefreshHub,
    private val datasetSelection: DatasetSelection,
    private val pathPicker: PathPicker,
) : ViewModel() {

    private val _uiState = MutableStateFlow(AutomationUiState())
    val uiState: StateFlow<AutomationUiState> = _uiState.asStateFlow()

    private var entered = false
    private var pollJob: Job? = null
    private val watchedJobs = mutableSetOf<String>()

    /** First entry loads what the page needs; later entries keep the wizard's state. */
    fun onEnter() {
        if (entered) return
        entered = true
        loadMatrix()
        refreshProfiles()
        loadAutomationConfig()
    }

    /** Polls the job list while something is running, and stops when nothing is. */
    private fun ensureJobPolling() {
        if (pollJob?.isActive == true) return
        pollJob = viewModelScope.launch {
            while (true) {
                delay(2000)
                if (_uiState.value.runningJob == null) break
                refreshJobs()
            }
            // The tick that found the job stopped: look once more at what it left behind, so the
            // finished state (and the buttons it enables) is on screen without a click.
            refreshJobs()
            pollJob = null
        }
    }

    /**
     * A job this session saw running and that has just finished may have written into the dataset
     * folder Images and Statistics watch; those pages reload off the hub.
     */
    private suspend fun notifyDatasetIfNeeded() {
        _uiState.value.jobs.forEach { job ->
            if (job.state == "running") {
                watchedJobs.add(job.id)
                return@forEach
            }
            if (!watchedJobs.remove(job.id)) return@forEach
            val output = job.outputDir.trimEnd('/')
            if (output.isEmpty()) return@forEach
            val watched = try {
                ipc.parsedConfig().second.environment.trainDataEntries()
                    .getOrNull(datasetSelection.index.value)
                    ?.path
                    ?.trimEnd('/')
            } catch (_: Exception) {
                null
            }
            if (watched != null && watched == output) refreshHub.notifyDatasetChanged()
        }
    }

    fun selectSection(section: AutomationSection) {
        _uiState.update { it.copy(section = section) }
    }

    fun setLanguage(language: PromptLang) {
        _uiState.update { it.copy(language = language, notice = null) }
    }

    // --- shared resources ---

    fun loadMatrix() {
        viewModelScope.launch {
            _uiState.update { it.copy(loadingMatrix = true, matrixError = null) }
            try {
                val document = ipc.promptMatrix()
                val matrix = parseMatrix(document.text)
                _uiState.update {
                    it.copy(
                        loadingMatrix = false,
                        matrix = matrix,
                        matrixPath = document.path,
                        matrixLines = document.text.lines().size,
                        matrixError = null,
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(loadingMatrix = false, matrixError = e.message ?: uiText(it.language, "matrix_failed")) }
            }
        }
    }

    fun refreshProfiles() {
        viewModelScope.launch {
            _uiState.update { it.copy(loadingProfiles = true, profileError = null) }
            try {
                val listed = ipc.promptProfileList()
                val known = _uiState.value.profiles.associateBy { it.name }
                _uiState.update { state ->
                    state.copy(
                        loadingProfiles = false,
                        profiles = listed.profiles.map { info ->
                            PromptProfileItem(
                                name = info.name,
                                version = info.version,
                                modified = info.modified,
                                size = info.size,
                                error = info.error,
                                notes = known[info.name]?.notes.orEmpty(),
                            )
                        },
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(loadingProfiles = false, profileError = e.message ?: uiText(it.language, "profiles_failed")) }
            }
        }
    }

    // --- profiles ---

    fun loadProfile(name: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(busy = true, profileError = null, notice = null) }
            try {
                val document = ipc.promptProfileGet(name)
                val loaded = ProfileCodec.loadProfile(document.text, document.name.ifBlank { name })
                _uiState.update { state ->
                    state.copy(
                        busy = false,
                        spec = loaded.spec,
                        activeProfile = loaded.name,
                        profiles = state.profiles.map { item ->
                            if (item.name == loaded.name) item.copy(notes = loaded.notes) else item
                        },
                        view = PromptView.Manifest,
                        pageIndex = 0,
                        editingRow = null,
                        results = emptyList(),
                        resultSeed = null,
                        warnings = emptyList(),
                        notice = loaded.notes.joinToString("；").ifEmpty { null },
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(busy = false, profileError = e.message ?: uiText(it.language, "load_failed")) }
            }
        }
    }

    fun deleteProfile(name: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(busy = true, profileError = null) }
            try {
                ipc.promptProfileDelete(name)
                _uiState.update { state ->
                    state.copy(
                        busy = false,
                        profiles = state.profiles.filterNot { it.name == name },
                        activeProfile = if (state.activeProfile == name) "" else state.activeProfile,
                        notice = uiText(state.language, "profile_deleted").replace("{name}", name),
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(busy = false, profileError = e.message ?: uiText(it.language, "delete_failed")) }
            }
        }
    }

    fun openSaveDialog() {
        _uiState.update {
            it.copy(
                saveDialogOpen = true,
                saveName = it.activeProfile,
                saveOverwrite = false,
                saveError = null,
                notice = null,
            )
        }
    }

    fun closeSaveDialog() {
        _uiState.update { it.copy(saveDialogOpen = false, saveError = null) }
    }

    fun setSaveName(value: String) {
        _uiState.update { it.copy(saveName = value, saveError = null) }
    }

    fun setSaveOverwrite(overwrite: Boolean) {
        _uiState.update { it.copy(saveOverwrite = overwrite, saveError = null) }
    }

    /** Writes the current spec; a name already on disk comes back as the store's own refusal. */
    fun saveProfile() {
        val state = _uiState.value
        val name = state.saveName.trim()
        if (name.isEmpty()) {
            val message = uiText(state.language, "need_name")
            _uiState.update { it.copy(saveError = message) }
            return
        }
        viewModelScope.launch {
            _uiState.update { it.copy(busy = true, saveError = null) }
            try {
                val text = ProfileCodec.profileText(name, state.spec)
                val saved = ipc.promptProfileSave(name, text, overwrite = state.saveOverwrite)
                _uiState.update {
                    it.copy(
                        busy = false,
                        saveDialogOpen = false,
                        activeProfile = saved.name,
                        notice = "${t(it.language, "profile_saved")}: ${saved.name}",
                    )
                }
                refreshProfiles()
            } catch (e: ProfileException) {
                _uiState.update { it.copy(busy = false, saveError = e.message) }
            } catch (e: Exception) {
                _uiState.update { it.copy(busy = false, saveError = e.message ?: uiText(it.language, "save_failed")) }
            }
        }
    }

    // --- wizard ---

    fun setView(view: PromptView) {
        _uiState.update { it.copy(view = view, editingRow = null, notice = null) }
    }

    fun goToPage(index: Int) {
        _uiState.update { state ->
            val last = (state.applicablePageKeys.size - 1).coerceAtLeast(0)
            state.copy(pageIndex = index.coerceIn(0, last))
        }
    }

    fun nextPage() = goToPage(_uiState.value.visiblePageIndex + 1)

    fun previousPage() = goToPage(_uiState.value.visiblePageIndex - 1)

    fun openManifestRow(pageKey: String) {
        _uiState.update { it.copy(editingRow = pageKey, notice = null) }
    }

    fun closeManifestRow() {
        _uiState.update { it.copy(editingRow = null) }
    }

    private fun edit(block: (PromptSpec) -> Unit) {
        _uiState.update { state ->
            val spec = state.spec.deepCopy()
            block(spec)
            state.copy(spec = spec, notice = null, generateError = null)
        }
    }

    fun setCharacter(value: String) = edit { it.character = value }

    fun setMode(mode: PromptMode) = edit { WizardModel.applyModeChange(it, mode) }

    /** An empty exposure falls back to the mode's default when a prompt is drawn, like the CLI. */
    fun setExposure(level: String, on: Boolean) = edit { spec ->
        val levels = spec.exposure.toMutableList()
        if (on) {
            if (!levels.contains(level)) levels.add(level)
        } else {
            levels.remove(level)
        }
        spec.exposure = levels.sortedBy { EXPOSURE_LEVELS.indexOf(it) }
    }

    /** The "any" chip: clears the specific picks, so the whole pool is drawn from. */
    fun setClothingAny() = edit { spec ->
        spec.clothingAny = true
        spec.clothingKeys = emptySet()
    }

    fun toggleClothing(key: List<String>) = edit { spec ->
        val keys = spec.clothingKeys.toMutableSet()
        if (!keys.add(key)) keys.remove(key)
        spec.clothingKeys = keys
        spec.clothingAny = keys.isEmpty()
    }

    fun setChest(level: String) = edit { it.chest = level }

    fun setBelly(level: String) = edit { it.belly = level }

    fun setFaceGroup(groupId: String, pick: FacePick) = edit { spec -> spec.face = spec.face + (groupId to pick) }

    fun toggleFaceTag(groupId: String, tag: String) = edit { spec ->
        val group = FACE_GROUPS.first { it.id == groupId }
        val tags = (spec.face[groupId] as? FacePick.Tags)?.tags.orEmpty().toMutableList()
        if (!tags.remove(tag)) tags.add(tag)
        val ordered = group.options.map { it.tag }.filter { tags.contains(it) }
        spec.face = spec.face + (groupId to if (ordered.isEmpty()) FacePick.NONE else FacePick.Tags(ordered))
    }

    fun setSceneAny() = edit { spec ->
        spec.sceneAny = true
        spec.sceneKeys = emptySet()
    }

    fun toggleScene(key: List<String>) = edit { spec ->
        val keys = spec.sceneKeys.toMutableSet()
        if (!keys.add(key)) keys.remove(key)
        spec.sceneKeys = keys
        spec.sceneAny = keys.isEmpty()
    }

    fun setFamilyAny() = edit { spec ->
        spec.familyAny = true
        spec.families = emptySet()
    }

    fun toggleFamily(family: PoseFamily) = edit { spec ->
        val families = spec.families.toMutableSet()
        if (!families.add(family)) families.remove(family)
        spec.families = families
        spec.familyAny = families.isEmpty()
    }

    fun setVaginalRatio(value: Double) = edit { it.vaginalRatio = value.coerceIn(0.0, 1.0) }

    fun setStageWeight(stage: SexStage, value: Double) = edit { spec ->
        spec.stageWeights = spec.stageWeights + (stage to value.coerceIn(0.0, 1.0))
    }

    fun setPoseAny() = edit { spec ->
        spec.poseAny = true
        spec.poseKeys = emptySet()
    }

    fun togglePose(key: List<String>) = edit { spec ->
        val keys = spec.poseKeys.toMutableSet()
        if (!keys.add(key)) keys.remove(key)
        spec.poseKeys = keys
        spec.poseAny = keys.isEmpty()
    }

    fun setCount(value: String) {
        val digits = value.filter { it.isDigit() }.take(3)
        val count = digits.toIntOrNull() ?: return
        edit { it.count = count.coerceIn(PromptLimits.COUNT_MIN, PromptLimits.COUNT_MAX) }
    }

    fun setSeedText(value: String) {
        _uiState.update { it.copy(seedText = value.filter { ch -> ch.isDigit() }.take(19), notice = null) }
    }

    /** The warning the pane under the cursor shows, mirroring the CLI's confirm prompts. */
    fun currentWarning(): String? {
        val state = _uiState.value
        val key = state.editingRow ?: state.currentPageKey
        return when (key) {
            "exposure" -> t(state.language, "sfw_warn").takeIf { WizardModel.needsSfwExposureWarning(state.spec) }
            "chest" -> t(state.language, "sfw_body_warn").takeIf { WizardModel.needsSfwBodyWarning(state.spec) }
            "face" -> t(state.language, "sfw_face_warn").takeIf { WizardModel.needsSfwFaceWarning(state.spec) }
            else -> null
        }
    }

    // --- results ---

    fun generate() {
        val state = _uiState.value
        val matrix = state.matrix
        if (matrix == null) {
            _uiState.update { it.copy(generateError = uiText(it.language, "matrix_unavailable")) }
            return
        }
        val seed = state.seedText.trim().toLongOrNull() ?: Random.Default.nextLong(0, Long.MAX_VALUE)
        val warnings = mutableListOf<String>()
        try {
            val prompts = PromptGenerator.generate(state.spec, matrix, seed, warnings)
            _uiState.update {
                it.copy(
                    results = prompts,
                    resultSeed = seed,
                    warnings = warnings,
                    generateError = null,
                    notice = null,
                )
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(results = emptyList(), resultSeed = null, generateError = e.message ?: uiText(it.language, "generate_failed"))
            }
        }
    }

    fun clearResults() {
        _uiState.update { it.copy(results = emptyList(), resultSeed = null, warnings = emptyList()) }
    }

    fun copyLine(line: String) {
        copyTextToClipboard(line)
        _uiState.update { it.copy(notice = uiText(it.language, "copied_one")) }
    }

    fun copyAll() {
        val results = _uiState.value.results
        if (results.isEmpty()) return
        copyTextToClipboard(promptExportText(results))
        _uiState.update { it.copy(notice = uiText(it.language, "copied_many").replace("{n}", results.size.toString())) }
    }

    fun download() {
        val state = _uiState.value
        if (state.results.isEmpty()) return
        val name = promptExportFileName(state.resultSeed, state.results.size)
        viewModelScope.launch {
            val saved = saveClientText(name, promptExportText(state.results))
            _uiState.update { it.copy(notice = if (saved == null) null else uiText(it.language, "saved_to").replace("{path}", saved)) }
        }
    }

    /** Hands the current result list to the ComfyUI area. */
    fun useAsBatchInput() {
        val results = _uiState.value.results
        if (results.isEmpty()) return
        _uiState.update {
            it.copy(
                batchInput = results,
                promptSource = PromptSource.CurrentResults,
                notice = uiText(it.language, "sent_to_batch").replace("{n}", results.size.toString()),
            )
        }
    }

    // --- ComfyUI section ---

    fun setSettingsServer(value: String) = updateSettings { it.copy(server = value) }

    fun setSettingsWorkflow(value: String) = updateSettings { it.copy(workflow = value) }

    fun setSettingsPositiveNode(value: String) = updateSettings { it.copy(positiveNode = value) }

    fun setSettingsCount(value: String) =
        updateSettings { it.copy(count = value.filter { ch -> ch.isDigit() }.take(2)) }

    fun setSettingsPoll(value: String) =
        updateSettings { it.copy(poll = value.filter { ch -> ch.isDigit() || ch == '.' }.take(4)) }

    fun setSettingsOutputDir(value: String) = updateSettings { it.copy(outputDir = value) }

    /** Browse on the machine that runs the helper (the desktop dialog, or the web path picker). */
    suspend fun browseOutputDir() {
        val picked = pathPicker.pickDirectory("ComfyUI output folder", _uiState.value.settings.outputDir)
        if (!picked.isNullOrBlank()) setSettingsOutputDir(picked)
    }

    private fun updateSettings(block: (AutomationSettingsDraft) -> AutomationSettingsDraft) {
        _uiState.update {
            it.copy(settings = block(it.settings), settingsError = null, settingsNotice = null, jobError = null)
        }
    }

    /** Loads `automation/settings.json`, the uploaded workflows, the prompt sets and the job list. */
    fun loadAutomationConfig() {
        viewModelScope.launch {
            _uiState.update { it.copy(settingsSaving = true, settingsError = null) }
            try {
                val config = ipc.automationConfigGet()
                _uiState.update {
                    it.copy(
                        settingsSaving = false,
                        settingsLoaded = true,
                        settings = AutomationSettingsDraft.of(config.settings),
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(settingsSaving = false, settingsError = e.message ?: "读取设置失败") }
            }
            refreshWorkflows()
            refreshPromptSets()
            refreshJobs()
        }
    }

    fun saveAutomationSettings() {
        val draft = _uiState.value.settings
        viewModelScope.launch {
            _uiState.update { it.copy(settingsSaving = true, settingsError = null, settingsNotice = null) }
            try {
                val saved = ipc.automationConfigSave(draft.toSettings())
                _uiState.update {
                    it.copy(
                        settingsSaving = false,
                        settings = AutomationSettingsDraft.of(saved.settings),
                        settingsNotice = uiText(it.language, "settings_saved"),
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(settingsSaving = false, settingsError = e.message ?: "保存设置失败") }
            }
        }
    }

    fun discoverComfy() {
        val server = _uiState.value.settings.server.trim()
        viewModelScope.launch {
            _uiState.update { it.copy(discovering = true, discoverError = null) }
            try {
                val found = ipc.automationDiscover(server)
                _uiState.update {
                    it.copy(
                        discovering = false,
                        comfy = found,
                        settings = if (found.found && it.settings.server.isBlank()) {
                            it.settings.copy(server = found.url)
                        } else {
                            it.settings
                        },
                    )
                }
                if (found.found) refreshWorkflows()
            } catch (e: Exception) {
                _uiState.update { it.copy(discovering = false, discoverError = e.message ?: "探测失败") }
            }
        }
    }

    fun refreshWorkflows() {
        viewModelScope.launch {
            _uiState.update { it.copy(workflowsLoading = true, workflowError = null) }
            try {
                val listed = ipc.automationWorkflowList()
                _uiState.update { state ->
                    val chosen = state.settings.workflow
                    val active = listed.workflows.firstOrNull { it.path == chosen || it.name == chosen }
                    state.copy(
                        workflowsLoading = false,
                        workflows = listed.workflows,
                        workflowCheck = listed.modelCheck,
                        settings = when {
                            active != null && state.settings.positiveNode.isBlank() ->
                                state.settings.copy(positiveNode = active.positiveNode)
                            state.settings.workflow.isBlank() && listed.defaultWorkflow.isNotBlank() ->
                                state.settings.copy(workflow = listed.defaultWorkflow)
                            state.settings.workflow.isBlank() && listed.workflows.size == 1 ->
                                state.settings.copy(workflow = listed.workflows.first().path)
                            else -> state.settings
                        },
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(workflowsLoading = false, workflowError = e.message ?: "读取工作流失败") }
            }
        }
    }

    fun selectWorkflow(workflow: AutomationWorkflow) {
        updateSettings { it.copy(workflow = workflow.path, positiveNode = workflow.positiveNode) }
    }

    /** Uploads a workflow JSON the user picked from their own machine. */
    fun uploadWorkflow(text: String, suggestedName: String) {
        val name = suggestedName.substringAfterLast('/').substringBeforeLast('.')
        viewModelScope.launch {
            _uiState.update { it.copy(workflowsLoading = true, workflowError = null) }
            try {
                val saved = ipc.automationWorkflowSave(name, text)
                _uiState.update {
                    it.copy(
                        workflowsLoading = false,
                        settings = it.settings.copy(workflow = saved.path, positiveNode = saved.positiveNode),
                        settingsNotice = "${uiText(it.language, "workflow_uploaded")}: ${saved.name}",
                    )
                }
                refreshWorkflows()
            } catch (e: Exception) {
                _uiState.update { it.copy(workflowsLoading = false, workflowError = e.message ?: "上传失败") }
            }
        }
    }

    fun validateWorkflow(path: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(workflowsLoading = true, workflowError = null) }
            try {
                val report = ipc.automationWorkflowValidate(path, _uiState.value.settings.positiveNode)
                _uiState.update {
                    it.copy(
                        workflowsLoading = false,
                        settingsNotice = if (report.valid) uiText(it.language, "workflow_valid") else report.error,
                    )
                }
                refreshWorkflows()
            } catch (e: Exception) {
                _uiState.update { it.copy(workflowsLoading = false, workflowError = e.message ?: "校验失败") }
            }
        }
    }

    fun deleteWorkflow(name: String) {
        viewModelScope.launch {
            try {
                ipc.automationWorkflowDelete(name)
                _uiState.update { state ->
                    val wasActive = state.settings.workflow == name ||
                        state.settings.workflow.endsWith("/$name.json")
                    state.copy(
                        workflows = state.workflows.filterNot { it.name == name },
                        settings = if (wasActive) {
                            state.settings.copy(workflow = "", positiveNode = "")
                        } else {
                            state.settings
                        },
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(workflowError = e.message ?: "删除失败") }
            }
        }
    }

    // --- prompts for the batch ---

    fun setPromptSource(source: PromptSource) {
        _uiState.update { it.copy(promptSource = source, jobError = null) }
    }

    fun setPromptSetName(name: String) {
        _uiState.update { it.copy(promptSetName = name, jobError = null) }
    }

    fun setManualPrompts(text: String) {
        _uiState.update { it.copy(manualPrompts = text, jobError = null) }
    }

    fun refreshPromptSets() {
        viewModelScope.launch {
            try {
                val listed = ipc.automationPromptList()
                _uiState.update { state ->
                    state.copy(
                        promptSets = listed.prompts,
                        promptSetName = state.promptSetName.ifBlank { listed.prompts.firstOrNull()?.name.orEmpty() },
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(promptSetError = e.message ?: "读取 prompt 集失败") }
            }
        }
    }

    fun savePromptSet(name: String) {
        val state = _uiState.value
        val text = state.manualPrompts.ifBlank { promptExportText(state.results) }
        viewModelScope.launch {
            try {
                val saved = ipc.automationPromptSave(name, text)
                _uiState.update {
                    it.copy(
                        notice = "${uiText(it.language, "prompt_set_saved")}: ${saved.name} (${saved.count})",
                        promptSetName = saved.name,
                    )
                }
                refreshPromptSets()
            } catch (e: Exception) {
                _uiState.update { it.copy(promptSetError = e.message ?: "保存 prompt 集失败") }
            }
        }
    }

    fun deletePromptSet(name: String) {
        viewModelScope.launch {
            try {
                ipc.automationPromptDelete(name)
                _uiState.update { state ->
                    state.copy(
                        promptSets = state.promptSets.filterNot { it.name == name },
                        promptSetName = if (state.promptSetName == name) "" else state.promptSetName,
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(promptSetError = e.message ?: "删除 prompt 集失败") }
            }
        }
    }

    /** Starts a batch job from the chosen prompt source. */
    fun startJob() {
        val state = _uiState.value
        val (prompts, setName) = when (state.promptSource) {
            PromptSource.CurrentResults -> state.results to ""
            PromptSource.SavedSet -> emptyList<String>() to state.promptSetName
            PromptSource.Manual -> manualPrompts(state.manualPrompts) to ""
        }
        if (prompts.isEmpty() && setName.isBlank()) {
            _uiState.update { it.copy(jobError = uiText(it.language, "no_prompts")) }
            return
        }
        val settings = state.settings.toSettings()
        viewModelScope.launch {
            _uiState.update { it.copy(startingJob = true, jobError = null) }
            try {
                val started = ipc.automationJobStart(
                    prompts = prompts,
                    promptSet = setName,
                    overrides = jsonOverrides(settings),
                )
                _uiState.update {
                    it.copy(
                        startingJob = false,
                        lastJobId = started.job.id,
                        selectedJobId = started.job.id,
                        notice = "${uiText(it.language, "job_started")}: ${started.job.id}",
                    )
                }
                refreshJobs()
            } catch (e: Exception) {
                _uiState.update { it.copy(startingJob = false, jobError = e.message ?: "启动任务失败") }
            }
        }
    }

    private fun manualPrompts(text: String): List<String> =
        text.split("\n").map { it.trim() }.filter { it.isNotEmpty() }

    private fun jsonOverrides(settings: AutomationSettings): JsonObject = buildJsonObject {
        if (settings.server.isNotBlank()) put("server", settings.server)
        if (settings.workflow.isNotBlank()) put("workflow", settings.workflow)
        if (settings.positiveNode.isNotBlank()) put("positive_node", settings.positiveNode)
        put("count", settings.count)
        put("poll", settings.poll)
        if (settings.outputDir.isNotBlank()) put("output_dir", settings.outputDir)
    }

    // --- Gallery ---

    fun refreshJobs() {
        viewModelScope.launch {
            _uiState.update { it.copy(jobsLoading = true, jobsError = null) }
            try {
                val listed = ipc.automationJobList()
                _uiState.update { state ->
                    val selected = state.selectedJobId.takeIf { id -> listed.jobs.any { it.id == id } }
                        ?: listed.jobs.firstOrNull()?.id.orEmpty()
                    state.copy(jobsLoading = false, jobs = listed.jobs, selectedJobId = selected)
                }
                ensureJobPolling()
                notifyDatasetIfNeeded()
                refreshSelectedDetail()
            } catch (e: Exception) {
                _uiState.update { it.copy(jobsLoading = false, jobsError = e.message ?: "读取任务失败") }
            }
        }
    }

    /**
     * Re-reads the selected job's detail, which the poll does on every tick.
     *
     * A job that runs moves while the page looks at it — a pass writes images, a redraw swaps one,
     * its per-image seeds change and its own state ends up `done` — and without this the Gallery
     * kept the copy it loaded when the job was picked: new images stayed invisible and the
     * per-image buttons stayed disabled until the job was clicked again.
     */
    private fun refreshSelectedDetail() {
        if (_uiState.value.jobActionBusy != "") return
        val selected = _uiState.value.selectedJobId
        if (selected.isEmpty()) return
        viewModelScope.launch {
            try {
                val detail = ipc.automationJobGet(selected)
                _uiState.update { state ->
                    if (state.selectedJobId != selected) state else state.copy(jobDetail = detail)
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(jobsError = e.message ?: "读取任务失败") }
            }
        }
    }

    fun selectJob(id: String) {
        if (id.isBlank()) return
        _uiState.update { it.copy(selectedJobId = id, galleryPreviewIndex = null) }
        viewModelScope.launch {
            try {
                val detail = ipc.automationJobGet(id)
                _uiState.update { state ->
                    if (state.selectedJobId != id) state else state.copy(jobDetail = detail)
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(jobsError = e.message ?: "读取任务失败") }
            }
        }
    }

    fun setJobFilter(filter: JobFilter) {
        _uiState.update { it.copy(jobFilter = filter) }
    }

    fun setJobSearch(text: String) {
        _uiState.update { it.copy(jobSearch = text) }
    }

    fun setGalleryThumbSize(value: Float) {
        _uiState.update { it.copy(galleryThumbSize = value.coerceIn(80f, 360f)) }
    }

    fun openGalleryPreview(index: Int) {
        _uiState.update { it.copy(galleryPreviewIndex = index) }
    }

    fun closeGalleryPreview() {
        _uiState.update { it.copy(galleryPreviewIndex = null) }
    }

    fun previewPrev() = stepPreview(-1)

    fun previewNext() = stepPreview(1)

    private fun stepPreview(delta: Int) {
        _uiState.update { state ->
            val count = state.galleryImagePaths.size
            val index = state.galleryPreviewIndex ?: return@update state
            if (count <= 0) return@update state
            state.copy(galleryPreviewIndex = ((index + delta) % count + count) % count)
        }
    }

    fun cancelJob(id: String) = jobAction(id) { ipc.automationJobCancel(id) }

    fun retryFailedJob(id: String) = jobAction(id) { ipc.automationJobRetryFailed(id) }

    fun confirmDeleteJob(id: String) {
        _uiState.update { it.copy(pendingDeleteJob = id) }
    }

    fun dismissDeleteJob() {
        _uiState.update { it.copy(pendingDeleteJob = null) }
    }

    fun deleteJob(id: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(jobActionBusy = id, jobsError = null, pendingDeleteJob = null) }
            try {
                ipc.automationJobDelete(id)
                _uiState.update { state ->
                    state.copy(
                        jobActionBusy = "",
                        jobs = state.jobs.filterNot { it.id == id },
                        selectedJobId = if (state.selectedJobId == id) "" else state.selectedJobId,
                        jobDetail = if (state.jobDetail?.id == id) null else state.jobDetail,
                        galleryPreviewIndex = null,
                        notice = "${uiText(state.language, "job_deleted")}: $id",
                    )
                }
                refreshJobs()
            } catch (e: Exception) {
                _uiState.update { it.copy(jobActionBusy = "", jobsError = e.message ?: "删除任务失败") }
            }
        }
    }

    // --- Gallery: one image / one prompt at a time ---

    /** Asks before an image is overwritten or removed; the dialog is the confirmation. */
    fun confirmImageAction(jobId: String, promptIndex: Int, image: String, action: GalleryImageAction) {
        _uiState.update {
            it.copy(pendingImageAction = GalleryImagePrompt(GalleryImageRef(jobId, promptIndex, image), action))
        }
    }

    fun dismissImageAction() {
        _uiState.update { it.copy(pendingImageAction = null) }
    }

    /** Runs what the open image dialog asked for: a redraw with a new seed, or a delete. */
    fun runImageAction() {
        val pending = _uiState.value.pendingImageAction ?: return
        val ref = pending.ref
        _uiState.update { it.copy(pendingImageAction = null) }
        galleryDetailAction(ref.jobId) {
            when (pending.action) {
                GalleryImageAction.Delete -> ipc.automationImageDelete(ref.jobId, ref.image)
                GalleryImageAction.Regenerate -> ipc.automationImageRegenerate(ref.jobId, ref.image)
            }
        }
    }

    fun openPromptEdit(jobId: String, promptIndex: Int, text: String) {
        _uiState.update { it.copy(editingPrompt = PromptEditDraft(jobId, promptIndex, text), jobsError = null) }
    }

    fun updatePromptEdit(text: String) {
        _uiState.update { state -> state.copy(editingPrompt = state.editingPrompt?.copy(text = text)) }
    }

    fun dismissPromptEdit() {
        _uiState.update { it.copy(editingPrompt = null) }
    }

    /** Saves the edited text into the job record; nothing is rendered by this. */
    fun savePromptEdit() {
        val draft = _uiState.value.editingPrompt ?: return
        if (draft.text.isBlank()) return
        _uiState.update { it.copy(editingPrompt = null) }
        galleryDetailAction(draft.jobId) { ipc.automationJobPromptEdit(draft.jobId, draft.promptIndex, draft.text.trim()) }
    }

    fun openPromptExtend(jobId: String, promptIndex: Int) {
        _uiState.update { it.copy(extendingPrompt = PromptExtendDraft(jobId, promptIndex), jobsError = null) }
    }

    fun updatePromptExtend(count: String) {
        val digits = count.filter { it.isDigit() }.take(2)
        _uiState.update { state -> state.copy(extendingPrompt = state.extendingPrompt?.copy(count = digits)) }
    }

    fun dismissPromptExtend() {
        _uiState.update { it.copy(extendingPrompt = null) }
    }

    /** Adds the drafted number of images to one prompt, each from its own new random seed. */
    fun confirmPromptExtend() {
        val draft = _uiState.value.extendingPrompt ?: return
        val images = draft.images ?: return
        _uiState.update { it.copy(extendingPrompt = null) }
        galleryDetailAction(draft.jobId) { ipc.automationPromptExtend(draft.jobId, draft.promptIndex, images) }
    }

    /**
     * One Gallery action that answers with the job's whole detail: the reply replaces the page's
     * copy of that job (so a redraw's new seed shows at once) and the job list is refreshed to
     * pick up the `running` state. The preview is dropped when the image it showed is gone.
     */
    private fun galleryDetailAction(jobId: String, block: suspend () -> AutomationJobDetail) {
        viewModelScope.launch {
            _uiState.update { it.copy(jobActionBusy = jobId, jobsError = null) }
            try {
                val detail = block()
                _uiState.update { state ->
                    if (state.selectedJobId != jobId) {
                        state.copy(jobActionBusy = "")
                    } else {
                        state.copy(
                            jobActionBusy = "",
                            jobDetail = detail,
                            galleryPreviewIndex = state.galleryPreviewIndex
                                ?.takeIf { index -> index < detail.prompts.sumOf { it.images.size } },
                        )
                    }
                }
                refreshJobs()
            } catch (e: Exception) {
                _uiState.update { it.copy(jobActionBusy = "", jobsError = e.message ?: e.toString()) }
            }
        }
    }

    private fun jobAction(id: String, block: suspend () -> Unit) {
        viewModelScope.launch {
            _uiState.update { it.copy(jobActionBusy = id, jobsError = null) }
            try {
                block()
                _uiState.update { it.copy(jobActionBusy = "") }
                refreshJobs()
                selectJob(id)
            } catch (e: Exception) {
                _uiState.update { it.copy(jobActionBusy = "", jobsError = e.message ?: "任务操作失败") }
            }
        }
    }

    /** Saves one gallery image to the user's machine (desktop dialog or browser download). */
    fun downloadImage(name: String) {
        val path = _uiState.value.galleryImagePaths.firstOrNull { it.endsWith("/$name") } ?: return
        viewModelScope.launch {
            val bytes = try {
                ipc.blobBatch(listOf(path), maxEdge = 4096, quality = 100, format = "png").items.firstOrNull()?.base64
            } catch (e: Exception) {
                null
            }
            if (bytes == null) {
                _uiState.update { it.copy(jobsError = "无法读取图片") }
                return@launch
            }
            val saved = saveClientFile(name, decodeBase64(bytes), mime = "image/png")
            _uiState.update {
                it.copy(notice = if (saved == null) null else uiText(it.language, "saved_to").replace("{path}", saved))
            }
        }
    }

    /** Exports the selected job's per-image records as text. */
    fun downloadJobRecord() {
        val detail = _uiState.value.jobDetail ?: return
        val lines = detail.prompts.flatMap { prompt ->
            prompt.images.map { name ->
                "$name\tseed=${prompt.seed ?: ""}\tprompt_id=${prompt.promptId}\t${prompt.text}"
            }
        }
        if (lines.isEmpty()) return
        viewModelScope.launch {
            val saved = saveClientText("${detail.id}_records.txt", lines.joinToString("\n", postfix = "\n"))
            _uiState.update {
                it.copy(notice = if (saved == null) null else uiText(it.language, "saved_to").replace("{path}", saved))
            }
        }
    }
}

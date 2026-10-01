package com.acite.axlranko.pages

import com.acite.axlranko.IoDispatcher
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.AppearanceRepository
import com.acite.axlranko.data.ConfigProfile
import com.acite.axlranko.data.ConfigProfileStore
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.DatasetTagResult
import com.acite.axlranko.model.SAMPLE_SET_ERROR_PREFIX
import com.acite.axlranko.model.SampleSetForm
import com.acite.axlranko.model.TrainDataCountRequest
import com.acite.axlranko.model.TrainDataDirForm
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.UtilsUiState
import com.acite.axlranko.model.parseOnlyTags
import com.acite.axlranko.data.showsHelperEndpointSettings
import com.acite.axlranko.data.wallpaperImagesSupported
import com.acite.axlranko.pickClientWallpaper
import com.acite.axlranko.util.PathPicker
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class UtilsScreenViewModel(
    private val ipc: TrainerIpcClient,
    private val refreshHub: DatasetRefreshHub,
    private val appearanceRepo: AppearanceRepository,
    private val datasetSelection: DatasetSelection,
    private val pathPicker: PathPicker,
) : ViewModel() {

    private val _uiState = MutableStateFlow(UtilsUiState())
    val uiState: StateFlow<UtilsUiState> = _uiState.asStateFlow()

    /** The in-flight `dataset_counts` call; a new folder edit cancels the one it supersedes. */
    private var estimateJob: Job? = null

    init {
        _uiState.update {
            it.copy(
                helperHost = ipc.endpointHost(),
                helperPort = ipc.endpointPort().toString(),
                selectedSection = if (showsHelperEndpointSettings) {
                    ConfigSection.Helper
                } else {
                    it.selectedSection
                },
            )
        }
        if (showsHelperEndpointSettings) {
            _uiState.update { it.copy(isLoading = false) }
        } else {
            loadConfig()
            refreshProfiles()
        }
        viewModelScope.launch {
            appearanceRepo.settings.collect { value ->
                _uiState.update { it.copy(appearance = value) }
            }
        }
        viewModelScope.launch {
            datasetSelection.index.collect { index ->
                _uiState.update { it.copy(datasetDirIndex = index) }
            }
        }
    }

    fun updateHelperHost(value: String) {
        _uiState.update { it.copy(helperHost = value, helperError = null) }
    }

    fun updateHelperPort(value: String) {
        _uiState.update { it.copy(helperPort = value.filter { ch -> ch.isDigit() }.take(5), helperError = null) }
    }

    fun connectHelper() {
        val host = _uiState.value.helperHost.trim()
        val port = _uiState.value.helperPort.toIntOrNull() ?: return
        if (host.isEmpty()) return
        viewModelScope.launch {
            _uiState.update { it.copy(helperBusy = true, helperError = null, helperStatus = "connecting") }
            try {
                ipc.setEndpoint(host, port)
                ipc.restart()
                _uiState.update { it.copy(helperBusy = false, helperStatus = "connected") }
                loadConfig()
                refreshProfiles()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        helperBusy = false,
                        helperStatus = "failed",
                        helperError = e.message,
                    )
                }
            }
        }
    }

    fun updateBackground(style: BackgroundStyle) {
        if (style == BackgroundStyle.Image && !wallpaperImagesSupported) return
        if (style == BackgroundStyle.Image &&
            _uiState.value.appearance.backgroundImagePath.isBlank()
        ) {
            browseBackgroundImage()
            return
        }
        appearanceRepo.update { it.copy(background = style) }
    }

    fun updateCardBlur(value: Float) {
        appearanceRepo.update { it.copy(cardBlurRadiusDp = value) }
    }

    fun updateBackgroundBlur(value: Float) {
        appearanceRepo.update { it.copy(backgroundBlurRadiusDp = value) }
    }

    fun updateFontScale(value: Float) {
        appearanceRepo.update { it.copy(fontScale = value) }
    }

    fun updateIconScale(value: Float) {
        appearanceRepo.update { it.copy(iconScale = value) }
    }

    fun updateThumbnailQuality(value: Int) {
        appearanceRepo.update { it.copy(thumbnailQuality = value) }
    }

    fun browseBackgroundImage() {
        viewModelScope.launch {
            val selected = pickClientWallpaper() ?: return@launch
            appearanceRepo.update {
                it.copy(background = BackgroundStyle.Image, backgroundImagePath = selected)
            }
        }
    }

    fun clearBackgroundImage() {
        appearanceRepo.update { current ->
            current.copy(
                backgroundImagePath = "",
                background = if (current.background == BackgroundStyle.Image) {
                    BackgroundStyle.Solid
                } else {
                    current.background
                },
            )
        }
    }

    fun reloadFromDiskSafely() {
        if (_uiState.value.isDirty) return
        loadConfig()
    }

    fun loadConfig(statusMessage: String? = null) {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null, statusMessage = null) }
            withContext(IoDispatcher) {
                try {
                    val (path, config) = try {
                        ipc.parsedConfig()
                    } catch (e: Exception) {
                        _uiState.update {
                            it.copy(
                                isLoading = false,
                                errorMessage = e.message ?: "Could not locate or parse config.toml"
                            )
                        }
                        return@withContext
                    }
                    val form = TrainingConfigForm.from(config)
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            configPath = path,
                            form = form,
                            savedForm = form,
                            selectedSampleSet = it.selectedSampleSet.coerceIn(0, form.sampleSets.lastIndex),
                            fieldErrors = emptyMap(),
                            errorMessage = null,
                            statusMessage = statusMessage
                        )
                    }
                    // The file may name other folders than the ones that were counted.
                    invalidateStepEstimate()
                } catch (e: Exception) {
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            errorMessage = e.message ?: "Failed to load config.toml"
                        )
                    }
                }
            }
        }
    }

    fun resetForm() {
        var foldersChanged = false
        _uiState.update { state ->
            foldersChanged = state.savedForm.trainDataDirs != state.form.trainDataDirs
            state.copy(
                form = state.savedForm,
                fieldErrors = emptyMap(),
                errorMessage = null,
                statusMessage = null
            )
        }
        if (foldersChanged) {
            _uiState.update { it.copy(datasetCounts = null, datasetCountsError = null) }
            refreshStepEstimate(debounceMillis = ESTIMATE_DEBOUNCE_MILLIS)
        }
    }

    fun selectSection(section: ConfigSection) {
        _uiState.update { it.copy(selectedSection = section) }
        if (section == ConfigSection.Profiles) refreshProfiles()
        // The Training section shows the step estimate, which needs the folders' image counts once.
        if (section == ConfigSection.Training && _uiState.value.datasetCounts == null) refreshStepEstimate()
    }

    /**
     * The folder list may have changed under the estimate (a config reload, a reset to the saved
     * form): drop the counts and re-read them when the line is on screen.
     */
    private fun invalidateStepEstimate() {
        _uiState.update { it.copy(datasetCounts = null, datasetCountsError = null) }
        if (_uiState.value.selectedSection == ConfigSection.Training) refreshStepEstimate()
    }

    /**
     * How many steps the form's own values would take, as an estimate.
     *
     * The image counts come from the helper (`dataset_counts`, one directory walk), and everything
     * after that is local arithmetic — so editing epoch / batch / GA needs no round trip, while a
     * change to the folders themselves re-counts after [debounceMillis] of quiet.
     */
    fun refreshStepEstimate(debounceMillis: Long = 0L) {
        val dirs = _uiState.value.form.trainDataDirs.map { entry ->
            TrainDataCountRequest(
                path = entry.path.trim(),
                repeat = entry.repeat.trim().toIntOrNull() ?: 1,
            )
        }
        estimateJob?.cancel()
        estimateJob = viewModelScope.launch {
            // Counted as in flight before the quiet period, so the line says so while it waits.
            _uiState.update { it.copy(datasetCountsLoading = true) }
            if (debounceMillis > 0) delay(debounceMillis)
            try {
                val counts = withContext(IoDispatcher) { ipc.datasetCounts(dirs) }
                _uiState.update {
                    it.copy(
                        datasetCounts = counts,
                        datasetCountsLoading = false,
                        datasetCountsError = null,
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        datasetCountsLoading = false,
                        datasetCountsError = e.message ?: "Failed to count the dataset",
                    )
                }
            }
        }
    }

    fun selectSampleSet(index: Int) {
        _uiState.update { state ->
            if (index in state.form.sampleSets.indices) state.copy(selectedSampleSet = index) else state
        }
    }

    fun updateSampleSet(index: Int, transform: (SampleSetForm) -> SampleSetForm) {
        val set = _uiState.value.form.sampleSets.getOrNull(index) ?: return
        updateForm { withSampleSet(index, transform(set)) }
    }

    /** `+` clones the open tab so a second prompt set is one edit away. */
    fun addSampleSet() {
        val from = _uiState.value.selectedSampleSet
        updateForm { appendSampleSet(from) }
        _uiState.update { it.copy(selectedSampleSet = it.form.sampleSets.lastIndex) }
    }

    fun removeSampleSet(index: Int) {
        updateForm { removeSampleSet(index) }
        _uiState.update {
            it.copy(selectedSampleSet = it.selectedSampleSet.coerceIn(0, it.form.sampleSets.lastIndex))
        }
    }

    fun updateLeftWeight(weight: Float) {
        _uiState.update { it.copy(leftWeight = weight.coerceIn(0.16f, 0.4f)) }
    }

    fun updateTrainDataDir(index: Int, transform: (TrainDataDirForm) -> TrainDataDirForm) {
        val entry = _uiState.value.form.trainDataDirs.getOrNull(index) ?: return
        updateForm { withTrainDataDir(index, transform(entry)) }
    }

    fun addTrainDataDir() {
        updateForm { appendTrainDataDir() }
    }

    fun removeTrainDataDir(index: Int) {
        updateForm { removeTrainDataDir(index) }
    }

    fun browseTrainDataDir(index: Int) {
        val current = _uiState.value.form.trainDataDirs.getOrNull(index)?.path ?: return
        viewModelScope.launch {
            val selected = pathPicker.pickDirectory("Select directory", current) ?: return@launch
            updateTrainDataDir(index) { it.copy(path = selected) }
        }
    }

    /** The dataset folder Images, Statistics and the tag card act on. */
    fun selectDatasetDir(index: Int) {
        datasetSelection.select(index)
    }

    fun updateForm(transform: TrainingConfigForm.() -> TrainingConfigForm) {
        var foldersChanged = false
        _uiState.update { state ->
            val newForm = state.form.transform()
            foldersChanged = newForm.trainDataDirs != state.form.trainDataDirs
            state.copy(
                form = newForm,
                fieldErrors = emptyMap(),
                errorMessage = null,
                statusMessage = null
            )
        }
        // Only the folder list feeds the count; epoch / batch / GA are read from the form as it is.
        if (foldersChanged) {
            _uiState.update { it.copy(datasetCounts = null, datasetCountsError = null) }
            refreshStepEstimate(debounceMillis = ESTIMATE_DEBOUNCE_MILLIS)
        }
    }

    fun browseDirectory(current: String, update: TrainingConfigForm.(String) -> TrainingConfigForm) {
        viewModelScope.launch {
            val selected = pathPicker.pickDirectory("Select directory", current) ?: return@launch
            updateForm { update(selected) }
        }
    }

    // The dialog picks a file; the field itself still accepts a directory holding one .safetensors.
    fun browseCheckpointPath() {
        viewModelScope.launch {
            val current = _uiState.value.form.resumeLoraPath
            val selected = pathPicker.pickFile("Select checkpoint", current, listOf("safetensors")) ?: return@launch
            updateForm { copy(resumeLoraPath = selected) }
        }
    }

    fun clearCheckpoint() {
        updateForm { copy(resumeLoraPath = "") }
    }

    fun openCheckpointPicker() {
        _uiState.update { it.copy(checkpointPickerOpen = true, checkpointError = null) }
        loadCheckpoints()
    }

    fun closeCheckpointPicker() {
        _uiState.update { it.copy(checkpointPickerOpen = false) }
    }

    fun loadCheckpoints() {
        val state = _uiState.value
        if (state.isLoadingCheckpoints) return
        viewModelScope.launch {
            _uiState.update { it.copy(isLoadingCheckpoints = true, checkpointError = null) }
            try {
                val response = withContext(IoDispatcher) {
                    ipc.listCheckpoints(
                        name = state.form.outputName.trim().ifBlank { null },
                        outputDir = state.form.outputDir.trim().ifBlank { null }
                    )
                }
                _uiState.update {
                    it.copy(isLoadingCheckpoints = false, checkpoints = response.checkpoints)
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isLoadingCheckpoints = false,
                        checkpointError = e.message ?: "Failed to list checkpoints"
                    )
                }
            }
        }
    }

    fun selectCheckpoint(checkpoint: CheckpointItem) {
        _uiState.update { state ->
            state.copy(
                checkpointPickerOpen = false,
                form = state.form.copy(resumeLoraPath = checkpoint.path),
                fieldErrors = emptyMap(),
                errorMessage = null,
                statusMessage = "Resume checkpoint selected · save to apply"
            )
        }
    }

    fun updateTagThreshold(value: String) {
        _uiState.update { it.copy(tagThreshold = value, errorMessage = null) }
    }

    fun updatePartialTagging(enabled: Boolean) {
        _uiState.update { it.copy(partialTagging = enabled, errorMessage = null) }
    }

    fun updatePartialTags(value: String) {
        _uiState.update { it.copy(partialTags = value, errorMessage = null) }
    }

    /** Category switches: the last selected one cannot be switched off (a caption with none is a no-op). */
    fun toggleTagCategory(key: String) {
        _uiState.update { state ->
            val next = if (key in state.tagCategories) state.tagCategories - key else state.tagCategories + key
            state.copy(tagCategories = next.ifEmpty { setOf(key) }, errorMessage = null)
        }
    }

    /**
     * Ask the helper what the tagger can write. Once per session: the answer is the model's own
     * config, and the card falls back to the plain default when the model is not downloaded yet.
     */
    fun loadTaggerInfo() {
        val state = _uiState.value
        if (state.taggerInfoLoaded || state.isLoading) return
        _uiState.update { it.copy(taggerInfoLoaded = true) }
        viewModelScope.launch {
            val info = try {
                withContext(IoDispatcher) { ipc.taggerInfo() }
            } catch (_: Exception) {
                null
            }
            if (info != null) _uiState.update { it.copy(taggerInfo = info) }
        }
    }

    fun runAutoTag() {
        val state = _uiState.value
        if (state.isTagging || state.isSaving) return
        // The picker's folder, i.e. the one Images and Statistics are showing.
        val dirs = state.form.trainDataDirs
        val selected = state.datasetDirIndex.coerceIn(0, dirs.lastIndex.coerceAtLeast(0))
        val directory = dirs.getOrNull(selected)?.path?.trim().orEmpty()
        if (directory.isEmpty()) {
            _uiState.update { it.copy(errorMessage = "Set a train data directory before tagging") }
            return
        }
        val threshold = state.tagThreshold.toFloatOrNull()
        if (threshold == null || threshold !in 0f..1f) {
            _uiState.update { it.copy(errorMessage = "Tag confidence must be a number between 0.0 and 1.0") }
            return
        }
        val partial = state.partialTagging
        val onlyTags = if (partial) parseOnlyTags(state.partialTags) else emptyList()
        if (partial && onlyTags.isEmpty()) {
            _uiState.update { it.copy(errorMessage = "Partial tagging needs at least one tag") }
            return
        }
        // Partial tagging looks a tag up in every category, so the selection does not take part.
        val categories = if (partial) emptyList() else state.tagCategories.toList()

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isTagging = true,
                    errorMessage = null,
                    statusMessage = if (partial) "Adding tags…" else "Tagging dataset…",
                )
            }
            try {
                val result = withContext(IoDispatcher) {
                    ipc.datasetTag(
                        directory,
                        threshold,
                        categories = categories,
                        onlyTags = onlyTags,
                    )
                }
                val engine = listOf(result.engine, result.device).filter { it.isNotBlank() }.joinToString(", ")
                val provider = engine.ifBlank { result.provider.ifBlank { "tagger" } }
                val summary = if (partial) partialSummary(result) else {
                    "Tagged ${result.processed}/${result.total} images in ${result.seconds}s ($provider)"
                }
                val suffix = if (result.failed > 0) " · ${result.failed} failed" else ""
                _uiState.update {
                    it.copy(
                        isTagging = false,
                        statusMessage = summary + suffix,
                        errorMessage = null
                    )
                }
                refreshHub.notifyDatasetChanged()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isTagging = false,
                        statusMessage = null,
                        errorMessage = e.message ?: "Tagging failed"
                    )
                }
            }
        }
    }

    /** `Added anal ×12 · pussy ×3 · 15/100 images`, plus what never matched. */
    private fun partialSummary(result: DatasetTagResult): String {
        val added = result.added.entries.joinToString(" · ") { (tag, count) -> "$tag ×$count" }
        val head = if (added.isEmpty()) {
            "Added nothing"
        } else {
            "Added $added"
        }
        val unmatched = if (result.unmatched.isEmpty()) "" else " · never matched: ${result.unmatched.joinToString(", ")}"
        return "$head · ${result.processed}/${result.total} images$unmatched"
    }

    fun saveConfig() {
        val form = _uiState.value.form
        val errors = form.validate()
        if (errors.isNotEmpty()) {
            showValidationErrors(form, errors, "saving")
            return
        }

        viewModelScope.launch {
            _uiState.update { it.copy(isSaving = true, errorMessage = null, statusMessage = null) }
            val result = runCatching {
                withContext(IoDispatcher) {
                    val doc = ipc.configGet()
                    var patched = doc.text
                    for ((section, blocks) in form.toTomlArrayBlocks()) {
                        patched = TomlDocumentPatcher.replaceArrayOfTables(patched, section, blocks)
                    }
                    patched = TomlDocumentPatcher.apply(patched, form.toTomlSections())
                    ipc.configSave(patched)
                }
            }
            result.fold(
                onSuccess = {
                    _uiState.update {
                        it.copy(
                            isSaving = false,
                            savedForm = form,
                            fieldErrors = emptyMap(),
                            errorMessage = null,
                            statusMessage = "Saved to config.toml"
                        )
                    }
                },
                onFailure = { e ->
                    _uiState.update {
                        it.copy(
                            isSaving = false,
                            errorMessage = e.message ?: "Failed to save config.toml"
                        )
                    }
                }
            )
        }
    }

    /** Point the editor at the first invalid field, which is what a save or a profile write needs. */
    private fun showValidationErrors(
        form: TrainingConfigForm,
        errors: Map<String, String>,
        action: String,
    ) {
        val firstSection = ConfigSection.entries.firstOrNull { section ->
            errors.keys.any { section.owns(it) }
        }
        val failingSet = errors.keys
            .firstOrNull { it.startsWith(SAMPLE_SET_ERROR_PREFIX) }
            ?.removePrefix(SAMPLE_SET_ERROR_PREFIX)
            ?.substringBefore('.')
            ?.toIntOrNull()
        _uiState.update {
            it.copy(
                fieldErrors = errors,
                selectedSection = firstSection ?: it.selectedSection,
                selectedSampleSet = if (firstSection == ConfigSection.Validation && failingSet != null) {
                    failingSet.coerceIn(0, form.sampleSets.lastIndex)
                } else {
                    it.selectedSampleSet
                },
                statusMessage = null,
                errorMessage = "Fix ${errors.size} invalid field${if (errors.size == 1) "" else "s"} before $action"
            )
        }
    }

    fun refreshProfiles() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoadingProfiles = true) }
            val profiles = withContext(IoDispatcher) {
                ipc.profileList().profiles.map { ConfigProfile(it.name, it.modified, it.size) }
            }
            _uiState.update { it.copy(isLoadingProfiles = false, profiles = profiles) }
        }
    }

    fun updateProfileName(name: String) {
        _uiState.update { it.copy(profileName = name, errorMessage = null, statusMessage = null) }
    }

    /**
     * "Save as profile": the form is validated the way a save is, and a name that is already taken
     * asks for confirmation instead of overwriting.
     */
    fun requestSaveProfile() {
        val state = _uiState.value
        val name = state.profileName.trim()
        ConfigProfileStore.validateName(name)?.let { message ->
            _uiState.update { it.copy(errorMessage = message, statusMessage = null) }
            return
        }
        val form = state.form
        val errors = form.validate()
        if (errors.isNotEmpty()) {
            showValidationErrors(form, errors, "saving a profile")
            return
        }

        viewModelScope.launch {
            val existing = withContext(IoDispatcher) {
                ipc.profileList().profiles.any { it.name.equals(name, ignoreCase = true) }
            }
            if (existing) {
                _uiState.update { it.copy(pendingProfileOverwrite = name) }
            } else {
                writeProfile(name, form, overwrite = false)
            }
        }
    }

    fun confirmProfileOverwrite() {
        val name = _uiState.value.pendingProfileOverwrite ?: return
        val form = _uiState.value.form
        viewModelScope.launch { writeProfile(name, form, overwrite = true) }
    }

    private suspend fun writeProfile(
        name: String,
        form: TrainingConfigForm,
        overwrite: Boolean,
    ) {
        _uiState.update {
            it.copy(
                isSaving = true,
                errorMessage = null,
                statusMessage = null,
                pendingProfileOverwrite = null,
            )
        }
        val result = runCatching {
            withContext(IoDispatcher) {
                val document = ConfigProfileStore.document(form.toTomlSections(), form.toTomlArrayBlocks())
                ipc.profileSave(name, document, overwrite)
                ipc.profileList().profiles.map { ConfigProfile(it.name, it.modified, it.size) }
            }
        }
        result.fold(
            onSuccess = { profiles ->
                _uiState.update {
                    it.copy(
                        isSaving = false,
                        profiles = profiles,
                        profileName = "",
                        errorMessage = null,
                        statusMessage = "Saved profile $name",
                    )
                }
            },
            onFailure = { e ->
                _uiState.update {
                    it.copy(isSaving = false, errorMessage = e.message ?: "Could not save the profile")
                }
            },
        )
    }

    /** Applying writes `config.toml` itself, so unsaved editor changes ask first. */
    fun requestApplyProfile(profile: ConfigProfile) {
        if (_uiState.value.isDirty) {
            _uiState.update { it.copy(pendingProfileApply = profile) }
            return
        }
        applyProfile(profile)
    }

    fun confirmApplyProfile() {
        val profile = _uiState.value.pendingProfileApply ?: return
        applyProfile(profile)
    }

    private fun applyProfile(profile: ConfigProfile) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isSaving = true,
                    pendingProfileApply = null,
                    errorMessage = null,
                    statusMessage = null,
                )
            }
            val result = runCatching {
                withContext(IoDispatcher) {
                    val config = ipc.configGet()
                    val profileDoc = ipc.profileGet(profile.name)
                    val apply = ConfigProfileStore.applyToConfig(config.text, profileDoc.text).getOrThrow()
                    ipc.configSave(apply.text)
                    apply
                }
            }
            result.fold(
                onSuccess = { apply ->
                    _uiState.update { it.copy(isSaving = false) }
                    val skipped = if (apply.skippedSections.isEmpty()) {
                        ""
                    } else {
                        " · skipped ${apply.skippedSections.joinToString(", ")}"
                    }
                    // The editor reloads from the patched file; the status has to be set with it.
                    loadConfig("Applied ${profile.name} · ${apply.appliedKeys} keys$skipped")
                },
                onFailure = { e ->
                    _uiState.update {
                        it.copy(
                            isSaving = false,
                            errorMessage = e.message ?: "Could not apply the profile",
                        )
                    }
                },
            )
        }
    }

    fun requestDeleteProfile(profile: ConfigProfile) {
        _uiState.update { it.copy(pendingProfileDelete = profile) }
    }

    fun confirmDeleteProfile() {
        val profile = _uiState.value.pendingProfileDelete ?: return
        viewModelScope.launch {
            _uiState.update {
                it.copy(pendingProfileDelete = null, errorMessage = null, statusMessage = null)
            }
            val result = runCatching {
                withContext(IoDispatcher) {
                    ipc.profileDelete(profile.name)
                    ipc.profileList().profiles.map { ConfigProfile(it.name, it.modified, it.size) }
                }
            }
            result.fold(
                onSuccess = { profiles ->
                    _uiState.update {
                        it.copy(profiles = profiles, statusMessage = "Deleted profile ${profile.name}")
                    }
                },
                onFailure = { e ->
                    _uiState.update {
                        it.copy(errorMessage = e.message ?: "Could not delete the profile")
                    }
                },
            )
        }
    }

    fun cancelProfileDialog() {
        _uiState.update {
            it.copy(
                pendingProfileOverwrite = null,
                pendingProfileApply = null,
                pendingProfileDelete = null,
            )
        }
    }
}

/** Quiet time before a folder edit re-counts the dataset; typing a path should not walk it per key. */
private const val ESTIMATE_DEBOUNCE_MILLIS = 400L

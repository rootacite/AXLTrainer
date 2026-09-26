package com.acite.axlranko.pages

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.AppearanceRepository
import com.acite.axlranko.data.ConfigImporter
import com.acite.axlranko.data.ConfigProfile
import com.acite.axlranko.data.ConfigProfileStore
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.ProfileApply
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.AppearanceSettings
import com.acite.axlranko.model.BackgroundStyle
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.SAMPLE_SET_ERROR_PREFIX
import com.acite.axlranko.model.SampleSetForm
import com.acite.axlranko.model.TrainDataDirForm
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.UtilsUiState
import com.acite.axlranko.util.pickDirectoryDialog
import com.acite.axlranko.util.pickFileDialog
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

private val IMAGE_EXTENSIONS = listOf("jpg", "jpeg", "png", "webp", "bmp")

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class UtilsScreenViewModel(
    private val ipc: TrainerIpcClient,
    private val refreshHub: DatasetRefreshHub,
    private val appearanceRepo: AppearanceRepository,
    private val datasetSelection: DatasetSelection,
) : ViewModel() {

    private val _uiState = MutableStateFlow(UtilsUiState())
    val uiState: StateFlow<UtilsUiState> = _uiState.asStateFlow()

    init {
        loadConfig()
        refreshProfiles()
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

    fun updateBackground(style: BackgroundStyle) {
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

    fun browseBackgroundImage() {
        viewModelScope.launch {
            val current = _uiState.value.appearance.backgroundImagePath
            val selected = pickFileDialog("Select background image", current, IMAGE_EXTENSIONS) ?: return@launch
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
            withContext(Dispatchers.IO) {
                try {
                    val loaded = ConfigImporter.loadConfigOrNull()
                    if (loaded == null) {
                        _uiState.update {
                            it.copy(
                                isLoading = false,
                                errorMessage = "Could not locate or parse config.toml"
                            )
                        }
                        return@withContext
                    }
                    val (path, config) = loaded
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
        _uiState.update { state ->
            state.copy(
                form = state.savedForm,
                fieldErrors = emptyMap(),
                errorMessage = null,
                statusMessage = null
            )
        }
    }

    fun selectSection(section: ConfigSection) {
        _uiState.update { it.copy(selectedSection = section) }
        if (section == ConfigSection.Profiles) refreshProfiles()
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
            val selected = pickDirectoryDialog("Select directory", current) ?: return@launch
            updateTrainDataDir(index) { it.copy(path = selected) }
        }
    }

    /** The dataset folder Images, Statistics and the tag card act on. */
    fun selectDatasetDir(index: Int) {
        datasetSelection.select(index)
    }

    fun updateForm(transform: TrainingConfigForm.() -> TrainingConfigForm) {
        _uiState.update { state ->
            val newForm = state.form.transform()
            state.copy(
                form = newForm,
                fieldErrors = emptyMap(),
                errorMessage = null,
                statusMessage = null
            )
        }
    }

    fun browseDirectory(current: String, update: TrainingConfigForm.(String) -> TrainingConfigForm) {
        viewModelScope.launch {
            val selected = pickDirectoryDialog("Select directory", current) ?: return@launch
            updateForm { update(selected) }
        }
    }

    // The dialog picks a file; the field itself still accepts a directory holding one .safetensors.
    fun browseCheckpointPath() {
        viewModelScope.launch {
            val current = _uiState.value.form.resumeLoraPath
            val selected = pickFileDialog("Select checkpoint", current, listOf("safetensors")) ?: return@launch
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
                val response = withContext(Dispatchers.IO) {
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

        viewModelScope.launch {
            _uiState.update {
                it.copy(isTagging = true, errorMessage = null, statusMessage = "Tagging dataset…")
            }
            try {
                val result = withContext(Dispatchers.IO) {
                    ipc.datasetTag(directory, threshold)
                }
                val provider = result.provider.ifBlank { "ONNX" }
                val summary =
                    "Tagged ${result.processed}/${result.total} images in ${result.seconds}s ($provider)"
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

    fun saveConfig() {
        val form = _uiState.value.form
        val errors = form.validate()
        if (errors.isNotEmpty()) {
            showValidationErrors(form, errors, "saving")
            return
        }

        viewModelScope.launch {
            _uiState.update { it.copy(isSaving = true, errorMessage = null, statusMessage = null) }
            val result = withContext(Dispatchers.IO) {
                ConfigImporter.savePatched(form.toTomlSections(), form.toTomlArrayBlocks())
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
            val profiles = withContext(Dispatchers.IO) {
                ConfigProfileStore.resolveDir()?.let(ConfigProfileStore::list).orEmpty()
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
            val dir = ConfigProfileStore.resolveDir()
            if (dir == null) {
                _uiState.update { it.copy(errorMessage = "Could not locate the repository root") }
                return@launch
            }
            val existing = withContext(Dispatchers.IO) { ConfigProfileStore.findByName(dir, name) }
            if (existing != null) {
                _uiState.update { it.copy(pendingProfileOverwrite = name) }
            } else {
                writeProfile(dir, name, form, overwrite = false)
            }
        }
    }

    fun confirmProfileOverwrite() {
        val name = _uiState.value.pendingProfileOverwrite ?: return
        val dir = ConfigProfileStore.resolveDir()
        if (dir == null) {
            _uiState.update {
                it.copy(pendingProfileOverwrite = null, errorMessage = "Could not locate the repository root")
            }
            return
        }
        val form = _uiState.value.form
        viewModelScope.launch { writeProfile(dir, name, form, overwrite = true) }
    }

    private suspend fun writeProfile(
        dir: File,
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
        val result = withContext(Dispatchers.IO) {
            val document = ConfigProfileStore.document(form.toTomlSections(), form.toTomlArrayBlocks())
            ConfigProfileStore.save(dir, name, document, overwrite)
        }
        result.fold(
            onSuccess = { file ->
                val profiles = withContext(Dispatchers.IO) { ConfigProfileStore.list(dir) }
                _uiState.update {
                    it.copy(
                        isSaving = false,
                        profiles = profiles,
                        profileName = "",
                        errorMessage = null,
                        statusMessage = "Saved profile ${file.name}",
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
            val result = withContext(Dispatchers.IO) {
                val configPath = ConfigImporter.getConfigPath()
                    ?: return@withContext Result.failure<ProfileApply>(
                        IllegalStateException("Could not locate config.toml")
                    )
                val configFile = File(configPath)
                ConfigProfileStore
                    .applyToConfig(configFile.readText(), File(profile.path).readText())
                    .mapCatching { apply ->
                        configFile.writeText(apply.text)
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
        val dir = ConfigProfileStore.resolveDir()
        viewModelScope.launch {
            _uiState.update {
                it.copy(pendingProfileDelete = null, errorMessage = null, statusMessage = null)
            }
            val result = withContext(Dispatchers.IO) {
                if (dir == null) {
                    Result.failure<Unit>(IllegalStateException("Could not locate the repository root"))
                } else {
                    ConfigProfileStore.delete(dir, profile.path)
                }
            }
            result.fold(
                onSuccess = {
                    val profiles = withContext(Dispatchers.IO) {
                        dir?.let(ConfigProfileStore::list).orEmpty()
                    }
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

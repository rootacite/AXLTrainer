package com.acite.axlranko.pages

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.ConfigImporter
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.UtilsUiState
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
import javax.swing.JFileChooser

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class UtilsScreenViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(UtilsUiState())
    val uiState: StateFlow<UtilsUiState> = _uiState.asStateFlow()

    init {
        loadConfig()
    }

    fun reloadFromDiskSafely() {
        if (_uiState.value.isDirty) return
        loadConfig()
    }

    fun loadConfig() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null, statusMessage = null) }
            withContext(Dispatchers.IO) {
                try {
                    val loaded = ConfigImporter.loadConfigOrNull()
                    if (loaded == null) {
                        _uiState.update {
                            it.copy(
                                isLoading = false,
                                errorMessage = "Could not locate or parse trainer/config.toml"
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
                            fieldErrors = emptyMap(),
                            errorMessage = null,
                            statusMessage = null
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
    }

    fun updateLeftWeight(weight: Float) {
        _uiState.update { it.copy(leftWeight = weight.coerceIn(0.16f, 0.4f)) }
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
        val selected = pickPath(current, directoriesOnly = true) ?: return
        updateForm { update(selected) }
    }

    fun saveConfig() {
        val form = _uiState.value.form
        val errors = form.validate()
        if (errors.isNotEmpty()) {
            val firstSection = ConfigSection.entries.firstOrNull { section ->
                section.fieldKeys.any { it in errors }
            }
            _uiState.update {
                it.copy(
                    fieldErrors = errors,
                    selectedSection = firstSection ?: it.selectedSection,
                    statusMessage = null,
                    errorMessage = "Fix ${errors.size} invalid field${if (errors.size == 1) "" else "s"} before saving"
                )
            }
            return
        }

        viewModelScope.launch {
            _uiState.update { it.copy(isSaving = true, errorMessage = null, statusMessage = null) }
            val result = withContext(Dispatchers.IO) {
                ConfigImporter.savePatched(form.toTomlSections())
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

    private fun pickPath(current: String, directoriesOnly: Boolean): String? {
        val chooser = JFileChooser()
        chooser.fileSelectionMode =
            if (directoriesOnly) JFileChooser.DIRECTORIES_ONLY else JFileChooser.FILES_AND_DIRECTORIES
        chooser.dialogTitle = if (directoriesOnly) "Select directory" else "Select path"
        val start = File(current)
        when {
            start.isDirectory -> chooser.currentDirectory = start
            start.parentFile?.isDirectory == true -> chooser.currentDirectory = start.parentFile
        }
        val result = chooser.showOpenDialog(null)
        return if (result == JFileChooser.APPROVE_OPTION) {
            chooser.selectedFile.absolutePath
        } else {
            null
        }
    }
}

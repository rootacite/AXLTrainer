package com.acite.axlranko.pages

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.ConfigImporter
import com.acite.axlranko.model.DatasetItem
import com.acite.axlranko.model.StatisticsUiState
import com.acite.axlranko.model.TagStat
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
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlin.random.Random

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class StatisticsScreenViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(StatisticsUiState())
    val uiState: StateFlow<StatisticsUiState> = _uiState.asStateFlow()

    private val imageExtensions = setOf("jpg", "jpeg", "png", "webp", "bmp")

    init {
        scanDataset(isInitial = true)
    }

    /**
     * Core: Scan dataset, verify image-text pairs, and calculate frequencies
     */
    fun scanDataset(isInitial: Boolean = false) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.update {
                if (isInitial) it.copy(isLoading = true, errorMessage = null)
                else it.copy(isRefreshing = true, errorMessage = null)
            }

            val dirPath = ConfigImporter.getConfig().environment.trainDataDir
            val dir = File(dirPath)

            if (!dir.exists() || !dir.isDirectory) {
                _uiState.update {
                    it.copy(isLoading = false, isRefreshing = false,
                        errorMessage = "Invalid dataset directory: $dirPath")
                }
                return@launch
            }

            val allFiles = dir.listFiles() ?: emptyArray()
            val txtFiles = allFiles.filter { it.extension.lowercase() == "txt" }
            val imageMap = allFiles.filter { it.extension.lowercase() in imageExtensions }
                .associateBy { it.nameWithoutExtension }

            val items = mutableListOf<DatasetItem>()
            val tagCounter = mutableMapOf<String, Int>()

            for (txt in txtFiles) {
                val baseName = txt.nameWithoutExtension
                val imageFile = imageMap[baseName]

                // Strict verification: abort immediately if isolated txt file is found
                if (imageFile == null) {
                    val err = "Dataset error: Found isolated tag file ${txt.name} without a corresponding image! Execution aborted."
                    _uiState.update { it.copy(isLoading = false, errorMessage = err) }
                    return@launch
                }

                // Read tags, clean format, calculate frequency (deduplicate within single file)
                val content = txt.readText(Charsets.UTF_8)
                val tags = content.split(",")
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }

                val uniqueTags = tags.toSet()
                uniqueTags.forEach { tag ->
                    tagCounter[tag] = tagCounter.getOrDefault(tag, 0) + 1
                }

                items.add(DatasetItem(txt, imageFile, tags))
            }

            val totalFiles = items.size
            val stats = tagCounter.map { (tag, count) ->
                TagStat(tag, count, (count.toFloat() / totalFiles) * 100f)
            }.sortedByDescending { it.count }

            // Retain currently selected tags (if they still exist after rescan)
            val currentSelected = _uiState.value.selectedTags
            val validSelected = currentSelected.intersect(stats.map { it.tag }.toSet())

            _uiState.update {
                it.copy(
                    isLoading = false,
                    isRefreshing = false,
                    datasetItems = items,
                    tagStats = stats,
                    selectedTags = validSelected
                )
            }
        }
    }

    fun toggleTagSelection(tag: String) {
        _uiState.update { state ->
            val newSelection = state.selectedTags.toMutableSet()
            if (newSelection.contains(tag)) {
                newSelection.remove(tag)
            } else {
                newSelection.add(tag)
            }
            state.copy(selectedTags = newSelection)
        }
    }

    fun invertSelection() {
        _uiState.update { state ->
            val allTags = state.tagStats.map { it.tag }.toSet()
            val newSelection = allTags - state.selectedTags
            state.copy(selectedTags = newSelection)
        }
    }

    fun clearSelection() {
        _uiState.update { it.copy(selectedTags = emptySet()) }
    }

    fun updateFilterMode(isAnd: Boolean) {
        _uiState.update { it.copy(isAndMode = isAnd) }
    }

    fun updateNotMode(isNot: Boolean) {
        _uiState.update { it.copy(isNotMode = isNot) }
    }

    fun updateTagSearchQuery(text: String) {
        _uiState.update { it.copy(tagSearchQuery = text) }
    }

    fun updateLeftWeight(weight: Float) {
        _uiState.update { it.copy(leftWeight = weight.coerceIn(0.2f, 0.8f)) }
    }

    fun updateTopWeight(weight: Float) {
        _uiState.update { it.copy(topWeight = weight.coerceIn(0.2f, 0.8f)) }
    }

    fun updateDropRateText(text: String) {
        _uiState.update { it.copy(dropRateText = text) }
    }

    fun updateNewTagText(text: String) {
        _uiState.update { it.copy(newTagText = text) }
    }

    fun updateAddPosition(isStart: Boolean) {
        _uiState.update { it.copy(isAddStart = isStart) }
    }

    /**
     * Rewrite the corresponding txt file (with clean comma formatting)
     */
    private suspend fun writeTagsToFile(txtFile: File, newTags: List<String>) = withContext(Dispatchers.IO) {
        val cleanContent = newTags.joinToString(", ")
        txtFile.writeText(cleanContent, Charsets.UTF_8)
    }

    /**
     * 1. Remove selected tags
     */
    fun removeSelectedTags() {
        val state = _uiState.value
        val targets = state.filteredImages
        if (targets.isEmpty() || state.selectedTags.isEmpty()) return

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.update { it.copy(isRefreshing = true) }
            targets.forEach { item ->
                val updatedTags = item.tags.filterNot { it in state.selectedTags }
                writeTagsToFile(item.txtFile, updatedTags)
            }
            scanDataset()   // isInitial 默认 false
        }
    }

    /**
     * 2. Drop samples based on probability R (move to trash)
     */
    fun dropSamples() {
        val state = _uiState.value
        val rate = state.dropRateText.toFloatOrNull() ?: return
        if (rate <= 0f || rate > 1f) return

        val targets = state.filteredImages
        if (targets.isEmpty()) return

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.update { it.copy(isLoading = true) }
            val trashDir = File("/tmp/axlranko/trash")
            if (!trashDir.exists()) {
                trashDir.mkdirs()
            }

            targets.forEach { item ->
                if (Random.nextFloat() <= rate) {
                    try {
                        // Use Files.move for cross-partition moves and allow replacement
                        Files.move(
                            item.txtFile.toPath(),
                            File(trashDir, item.txtFile.name).toPath(),
                            StandardCopyOption.REPLACE_EXISTING
                        )
                        Files.move(
                            item.imageFile.toPath(),
                            File(trashDir, item.imageFile.name).toPath(),
                            StandardCopyOption.REPLACE_EXISTING
                        )
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }
            }
            scanDataset()
        }
    }

    /**
     * 3. Add specified tag to start/end
     */
    fun addTagToTargets() {
        val state = _uiState.value
        val newTag = state.newTagText.trim()
        val targets = state.filteredImages

        if (newTag.isEmpty() || targets.isEmpty()) return

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.update { it.copy(isLoading = true) }
            targets.forEach { item ->
                // Skip if tag already exists (prevent duplicates)
                if (newTag !in item.tags) {
                    val updatedTags = if (state.isAddStart) {
                        listOf(newTag) + item.tags
                    } else {
                        item.tags + newTag
                    }
                    writeTagsToFile(item.txtFile, updatedTags)
                }
            }
            scanDataset()
        }
    }
}
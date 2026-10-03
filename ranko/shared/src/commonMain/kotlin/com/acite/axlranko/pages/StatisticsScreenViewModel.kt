package com.acite.axlranko.pages

import com.acite.axlranko.IoDispatcher
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.data.trainDataEntries
import com.acite.axlranko.util.TagTranslations
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


@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class StatisticsScreenViewModel(
    private val refreshHub: DatasetRefreshHub,
    private val datasetSelection: DatasetSelection,
    private val ipc: TrainerIpcClient,
) : ViewModel() {
    private val _uiState = MutableStateFlow(StatisticsUiState())
    val uiState: StateFlow<StatisticsUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch(IoDispatcher) {
            runCatching { TagTranslations.install(ipc.tagLexicon().text) }
        }
        scanDataset(isInitial = true)
        viewModelScope.launch {
            refreshHub.events.collect { scanDataset() }
        }
    }

    /**
     * Core: Scan dataset, verify image-text pairs, and calculate frequencies
     */
    fun scanDataset(isInitial: Boolean = false) {
        viewModelScope.launch(IoDispatcher) {
            val entries = try {
                ipc.parsedConfig().second.environment.trainDataEntries()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        isRefreshing = false,
                        errorMessage = e.message ?: "Could not load config.toml",
                    )
                }
                return@launch
            }
            val selected = datasetSelection.index.value.coerceIn(0, entries.lastIndex.coerceAtLeast(0))
            _uiState.update {
                it.copy(
                    datasetDirs = entries,
                    datasetDirIndex = selected,
                    isLoading = isInitial,
                    isRefreshing = !isInitial,
                    errorMessage = null
                )
            }

            val dirPath = entries.getOrNull(selected)?.path.orEmpty()
            if (dirPath.isEmpty()) {
                _uiState.update {
                    it.copy(isLoading = false, isRefreshing = false,
                        errorMessage = "Invalid dataset directory: $dirPath")
                }
                return@launch
            }

            val listed = try {
                ipc.datasetList(dirPath)
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        isRefreshing = false,
                        errorMessage = e.message ?: "Invalid dataset directory: $dirPath",
                    )
                }
                return@launch
            }
            if (listed.orphans.isNotEmpty()) {
                val err = "Dataset error: Found isolated tag file ${listed.orphans.first()} without a corresponding image! Execution aborted."
                _uiState.update { it.copy(isLoading = false, isRefreshing = false, errorMessage = err) }
                return@launch
            }

            val items = mutableListOf<DatasetItem>()
            val tagCounter = mutableMapOf<String, Int>()
            for (record in listed.items) {
                val tags = record.tags
                if (tags.isEmpty()) continue
                val uniqueTags = tags.toSet()
                uniqueTags.forEach { tag ->
                    tagCounter[tag] = (tagCounter[tag] ?: 0) + 1
                }
                items.add(
                    DatasetItem(
                        stem = record.stem,
                        imagePath = record.image,
                        txtPath = record.txt,
                        maskPath = record.mask,
                        tags = tags,
                        width = record.width,
                        height = record.height,
                    )
                )
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
                    imageCount = listed.items.size,
                    tagStats = stats,
                    selectedTags = validSelected
                )
            }
        }
    }

    /** The picker switched dataset folders: rescan the one it landed on. */
    fun selectDatasetDir(index: Int) {
        if (index == _uiState.value.datasetDirIndex) return
        datasetSelection.select(index)
        scanDataset()
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

    fun updatePortraitImageWeight(weight: Float) {
        _uiState.update { it.copy(portraitImageWeight = weight.coerceIn(0.30f, 0.85f)) }
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

    private suspend fun writeTags(item: DatasetItem, newTags: List<String>) {
        val directory = item.imagePath.substringBeforeLast('/', missingDelimiterValue = ".")
        ipc.captionWrite(directory, item.stem, newTags.joinToString(", "))
    }

    /**
     * 1. Remove selected tags
     */
    fun removeSelectedTags() {
        val state = _uiState.value
        val targets = state.filteredImages
        if (targets.isEmpty() || state.selectedTags.isEmpty()) return

        viewModelScope.launch(IoDispatcher) {
            _uiState.update { it.copy(isRefreshing = true) }
            targets.forEach { item ->
                val updatedTags = item.tags.filterNot { it in state.selectedTags }
                writeTags(item, updatedTags)
            }
            // The captions changed on disk: every page holding dataset state reloads off the hub,
            // this one included (its collector rescans).
            refreshHub.notifyDatasetChanged()
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

        val directory = state.datasetDirs.getOrNull(state.datasetDirIndex)?.path.orEmpty()
        if (directory.isEmpty()) return
        viewModelScope.launch(IoDispatcher) {
            _uiState.update { it.copy(isLoading = true) }
            try {
                ipc.datasetDrop(directory, rate, stems = targets.map { it.stem })
            } catch (e: Exception) {
                e.printStackTrace()
            }
            refreshHub.notifyDatasetChanged()
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

        viewModelScope.launch(IoDispatcher) {
            _uiState.update { it.copy(isLoading = true) }
            targets.forEach { item ->
                // Skip if tag already exists (prevent duplicates)
                if (newTag !in item.tags) {
                    val updatedTags = if (state.isAddStart) {
                        listOf(newTag) + item.tags
                    } else {
                        item.tags + newTag
                    }
                    writeTags(item, updatedTags)
                }
            }
            refreshHub.notifyDatasetChanged()
        }
    }

    /**
     * 4. Shuffle the whole folder and renumber every sample to `0001…`, which is what
     * `tools/suf.py` did. The rename walks each sample's group in `DatasetShuffle`, so a caption
     * and a mask sidecar always follow their image; the scan's orphan fuse applies here too.
     */
    fun shuffleDataset() {
        val state = _uiState.value
        if (state.isShuffling || state.isRefreshing) return
        val dirPath = state.datasetDirs.getOrNull(state.datasetDirIndex)?.path.orEmpty()
        if (dirPath.isEmpty()) return
        val dirName = dirPath.trimEnd('/').substringAfterLast('/')

        viewModelScope.launch(IoDispatcher) {
            _uiState.update {
                it.copy(isShuffling = true, statusMessage = "Shuffling $dirName…", statusIsError = false)
            }
            try {
                val report = ipc.datasetShuffle(dirPath)
                _uiState.update {
                    it.copy(
                        isShuffling = false,
                        statusMessage = "Shuffled ${report.groups} samples " +
                            "(${report.renamedFiles} files) to ${report.firstStem}…${report.lastStem}",
                    )
                }
                // Images and Utils hold absolute paths: they have to rescan, not just this page.
                refreshHub.notifyDatasetChanged()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isShuffling = false,
                        statusMessage = "Shuffle failed: ${e.message}",
                        statusIsError = true,
                    )
                }
            }
        }
    }
}
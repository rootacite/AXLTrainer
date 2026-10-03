package com.acite.axlranko.model

import com.acite.axlranko.data.TrainDataEntryConfig


data class StatisticsUiState(
    val isLoading: Boolean = true,
    val isRefreshing: Boolean = false,
    val errorMessage: String? = null,
    /** `[[environment.train_data]]`: the folders the picker offers, and the scanned one. */
    val datasetDirs: List<TrainDataEntryConfig> = emptyList(),
    val datasetDirIndex: Int = 0,
    val datasetItems: List<DatasetItem> = emptyList(),
    /** Images found in the folder, whether or not they have a caption: what a shuffle renames. */
    val imageCount: Int = 0,
    /** Shuffle-and-renumber progress plus its last result; `errorMessage` stays the scan fuse. */
    val isShuffling: Boolean = false,
    val statusMessage: String? = null,
    val statusIsError: Boolean = false,
    val tagStats: List<TagStat> = emptyList(),
    val selectedTags: Set<String> = emptySet(),
    val isAndMode: Boolean = true, // true: Intersection (AND), false: Union (OR)
    val isNotMode: Boolean = false, // true: Negate the AND/OR result (outer NOT)
    val tagSearchQuery: String = "",
    val leftWeight: Float = 0.4f,
    val topWeight: Float = 0.6f,
    /** Portrait only: the image pane's share of the height under the dataset bar. */
    val portraitImageWeight: Float = 0.65f,
    val dropRateText: String = "0.5",
    val newTagText: String = "",
    val isAddStart: Boolean = true // true: Add to start, false: Add to end
)
{
    // Dynamically calculate the list of images matching the current logical conditions
    val filteredImages: List<DatasetItem>
        get() {
            if (selectedTags.isEmpty()) return emptyList()
            return datasetItems.filter { item ->
                val matches = if (isAndMode) {
                    selectedTags.all { it in item.tags }
                } else {
                    selectedTags.any { it in item.tags }
                }
                if (isNotMode) !matches else matches
            }
        }
}
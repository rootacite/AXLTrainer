package com.acite.axlranko.model


data class StatisticsUiState(
    val isLoading: Boolean = true,
    val isRefreshing: Boolean = false,
    val errorMessage: String? = null,
    val datasetItems: List<DatasetItem> = emptyList(),
    val tagStats: List<TagStat> = emptyList(),
    val selectedTags: Set<String> = emptySet(),
    val isAndMode: Boolean = true, // true: Intersection (AND), false: Union (OR)
    val isNotMode: Boolean = false, // true: Negate the AND/OR result (outer NOT)
    val tagSearchQuery: String = "",
    val leftWeight: Float = 0.4f,
    val topWeight: Float = 0.6f,
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
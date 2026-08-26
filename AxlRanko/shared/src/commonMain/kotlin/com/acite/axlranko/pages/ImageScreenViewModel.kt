// ImageViewModel.kt
package com.acite.axlranko.pages

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.ConfigImporter
import com.acite.axlranko.model.ImageItem
import com.acite.axlranko.model.ImageScreenState
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class ImageScreenViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(ImageScreenState())
    val uiState: StateFlow<ImageScreenState> = _uiState.asStateFlow()

    init {
        loadData()
    }

    fun reloadFromDiskSafely() {
        val currentDir = _uiState.value.dataDir
        if (currentDir.isEmpty()) return

        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                val folder = File(currentDir)
                if (!folder.exists() || !folder.isDirectory) return@withContext

                val validExtensions = listOf("jpg", "jpeg", "png", "webp", "bmp")

                val currentItemsMap = _uiState.value.imageItems.associateBy { it.imagePath }
                val currentSelectedPath = _uiState.value.selectedItem?.imagePath

                val updatedItems = folder.listFiles()
                    ?.filter { it.isFile && validExtensions.contains(it.extension.lowercase()) }
                    ?.map { imgFile ->
                        val txtFile = File(folder, "${imgFile.nameWithoutExtension}.txt")
                        val diskTags = if (txtFile.exists()) txtFile.readText() else ""

                        val existingItem = currentItemsMap[imgFile.absolutePath]

                        if (existingItem != null) {
                            existingItem.copy(tags = diskTags)
                        } else {
                            ImageItem(
                                imagePath = imgFile.absolutePath,
                                txtPath = txtFile.absolutePath,
                                tags = diskTags,
                                draftTags = null
                            )
                        }
                    } ?: emptyList()

                _uiState.update { state ->
                    val newSelectedItem = updatedItems.find { it.imagePath == currentSelectedPath }

                    state.copy(
                        imageItems = updatedItems,
                        selectedItem = newSelectedItem,
                        editorText = newSelectedItem?.currentTags ?: ""
                    )
                }
            }
        }
    }

    private fun loadData() {
        viewModelScope.launch {
            try {
                val dir = ConfigImporter.getConfig().environment.trainDataDir
                _uiState.update { it.copy(dataDir = dir) }
                loadImages(dir)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private suspend fun loadImages(dataDir: String) {
        if (dataDir.isEmpty()) return

        withContext(Dispatchers.IO) {
            val folder = File(dataDir)
            if (folder.exists() && folder.isDirectory) {
                val validExtensions = listOf("jpg", "jpeg", "png", "webp", "bmp")
                val items = folder.listFiles()
                    ?.filter { it.isFile && validExtensions.contains(it.extension.lowercase()) }
                    ?.map { imgFile ->
                        val txtFile = File(folder, "${imgFile.nameWithoutExtension}.txt")
                        val initialTags = if (txtFile.exists()) txtFile.readText() else ""
                        ImageItem(
                            imagePath = imgFile.absolutePath,
                            txtPath = txtFile.absolutePath,
                            tags = initialTags,
                            draftTags = null
                        )
                    } ?: emptyList()

                _uiState.update { it.copy(imageItems = items) }
            }
        }
    }

    fun selectItem(item: ImageItem) {
        _uiState.update { state ->
            state.copy(
                selectedItem = item,
                editorText = item.currentTags
            )
        }
    }

    fun selectItemByTxtPath(txtPath: String) {
        val i = _uiState.value.imageItems.first { it.txtPath == txtPath }
        selectItem(i)
    }

    fun updateEditorText(text: String) {
        _uiState.update { state ->
            val selected = state.selectedItem ?: return@update state
            val updatedItem = selected.copy(draftTags = text)

            val updatedList = state.imageItems.map {
                if (it.imagePath == updatedItem.imagePath) updatedItem else it
            }

            state.copy(
                editorText = text,
                selectedItem = updatedItem,
                imageItems = updatedList
            )
        }
    }

    fun resetEditorText() {
        _uiState.update { state ->
            val selected = state.selectedItem ?: return@update state
            val updatedItem = selected.copy(draftTags = null)

            val updatedList = state.imageItems.map {
                if (it.imagePath == updatedItem.imagePath) updatedItem else it
            }

            state.copy(
                editorText = updatedItem.tags,
                selectedItem = updatedItem,
                imageItems = updatedList
            )
        }
    }

    fun saveTags() {
        val currentState = _uiState.value
        val itemToSave = currentState.selectedItem ?: return

        viewModelScope.launch(Dispatchers.IO) {
            try {
                File(itemToSave.txtPath).writeText(itemToSave.currentTags)

                _uiState.update { state ->
                    val currentList = state.imageItems
                    val targetItem = currentList.find { it.imagePath == itemToSave.imagePath } ?: return@update state

                    val updatedItem = targetItem.copy(tags = targetItem.currentTags, draftTags = null)

                    val updatedList = currentList.map {
                        if (it.imagePath == updatedItem.imagePath) updatedItem else it
                    }

                    state.copy(
                        imageItems = updatedList,
                        selectedItem = if (state.selectedItem?.imagePath == updatedItem.imagePath) updatedItem else state.selectedItem
                    )
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun updateLeftWeight(weight: Float) {
        _uiState.update { it.copy(leftWeight = weight.coerceIn(0.1f, 0.9f)) }
    }

    fun updateTopWeight(weight: Float) {
        _uiState.update { it.copy(topWeight = weight.coerceIn(0.1f, 0.9f)) }
    }
}
// ImageViewModel.kt
package com.acite.axlranko.pages

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.ConfigImporter
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.trainDataEntries
import com.acite.axlranko.model.ImageItem
import com.acite.axlranko.model.ImageScreenState
import com.acite.axlranko.util.MaskCanvas
import com.acite.axlranko.util.MaskDirtyRect
import com.acite.axlranko.util.fileHasAlphaChannel
import com.acite.axlranko.util.isMaskSidecar
import com.acite.axlranko.util.maskFileFor
import com.acite.axlranko.util.nudgeBrushRadius
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
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.roundToInt

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class ImageScreenViewModel(
    private val refreshHub: DatasetRefreshHub,
    private val datasetSelection: DatasetSelection,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ImageScreenState())
    val uiState: StateFlow<ImageScreenState> = _uiState.asStateFlow()

    private var photoImage: BufferedImage? = null
    private var mask: MaskCanvas? = null
    private var previewImage: BufferedImage? = null
    private var cachedPreview: ImageBitmap? = null
    private var maskRevision = 0
    private var strokeX: Float? = null
    private var strokeY: Float? = null
    private var strokeErase: Boolean = false
    private var lastPublishNs: Long = 0L

    /** File a Statistics jump asked for, opened once the rescan of its folder finishes. */
    private var pendingSelectTxtPath: String? = null

    init {
        loadData()
        viewModelScope.launch {
            refreshHub.events.collect { reloadFromDisk(resetDrafts = true) }
        }
    }

    fun previewBitmap(): ImageBitmap? = cachedPreview

    fun reloadFromDiskSafely() {
        reloadFromDisk(resetDrafts = false)
    }

    fun reloadFromDisk(resetDrafts: Boolean) {
        viewModelScope.launch {
            val pendingTxtPath = pendingSelectTxtPath
            pendingSelectTxtPath = null
            val entries = try {
                ConfigImporter.getConfig().environment.trainDataEntries()
            } catch (_: Exception) {
                emptyList()
            }
            val selected = datasetSelection.index.value.coerceIn(0, entries.lastIndex.coerceAtLeast(0))
            val currentDir = entries.getOrNull(selected)?.path ?: _uiState.value.dataDir
            if (currentDir.isEmpty()) return@launch
            _uiState.update {
                it.copy(dataDir = currentDir, datasetDirs = entries, datasetDirIndex = selected)
            }

            withContext(Dispatchers.IO) {
                val folder = File(currentDir)
                if (!folder.exists() || !folder.isDirectory) return@withContext

                val currentItemsMap = _uiState.value.imageItems.associateBy { it.imagePath }
                val currentSelectedPath = _uiState.value.selectedItem?.imagePath

                val updatedItems = scanImageItems(folder).map { fresh ->
                    val existing = currentItemsMap[fresh.imagePath]
                    if (existing != null && !resetDrafts) {
                        existing.copy(
                            tags = fresh.tags,
                            maskPath = fresh.maskPath,
                            hasSidecarMask = fresh.hasSidecarMask,
                            hasAlpha = fresh.hasAlpha,
                        )
                    } else {
                        fresh
                    }
                }

                val newSelected = selectionAfterRescan(updatedItems, pendingTxtPath, currentSelectedPath)
                _uiState.update { state ->
                    state.copy(
                        imageItems = updatedItems,
                        selectedItem = newSelected,
                        editorText = newSelected?.currentTags ?: "",
                    )
                }
                if (newSelected != null) {
                    loadMaskBuffers(newSelected)
                } else {
                    clearMaskBuffers()
                }
            }
        }
    }

    private fun loadData() {
        viewModelScope.launch {
            try {
                val entries = ConfigImporter.getConfig().environment.trainDataEntries()
                val selected = datasetSelection.index.value.coerceIn(0, entries.lastIndex.coerceAtLeast(0))
                val dir = entries.getOrNull(selected)?.path.orEmpty()
                _uiState.update {
                    it.copy(dataDir = dir, datasetDirs = entries, datasetDirIndex = selected)
                }
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
                _uiState.update { it.copy(imageItems = scanImageItems(folder)) }
            }
        }
    }

    /** The picker switched dataset folders: unsaved mask strokes belong to the folder left behind. */
    fun selectDatasetDir(index: Int) {
        if (index == _uiState.value.datasetDirIndex) return
        switchDatasetDir(index)
    }

    /**
     * Jump here from a Statistics thumbnail. That page can have another `[[environment.train_data]]`
     * folder open than this one, so an image missing from the loaded list means "scan that folder
     * first": the jump is remembered and completes once the rescan is done.
     */
    fun selectItemByTxtPath(txtPath: String, datasetDirIndex: Int) {
        val current = findImageByTxtPath(_uiState.value.imageItems, txtPath)
        if (current != null) {
            selectItem(current)
            return
        }
        pendingSelectTxtPath = txtPath
        switchDatasetDir(datasetDirIndex)
    }

    private fun switchDatasetDir(index: Int) {
        flushPendingMask()
        datasetSelection.select(index)
        _uiState.update { it.copy(datasetDirIndex = index) }
        reloadFromDisk(resetDrafts = true)
    }

    fun selectItem(item: ImageItem) {
        val previous = _uiState.value.selectedItem
        if (previous != null && previous.imagePath != item.imagePath) flushPendingMask()
        _uiState.update { state ->
            state.copy(
                selectedItem = item,
                editorText = item.currentTags,
                maskDirty = false,
            )
        }
        viewModelScope.launch(Dispatchers.IO) { loadMaskBuffers(item) }
    }

    /** Writes the pending mask for the open image before the selection moves off it. */
    private fun flushPendingMask() {
        val state = _uiState.value
        if (!state.maskDirty) return
        val item = state.selectedItem ?: return
        val copy = (mask ?: return).copyImage()
        val path = item.maskPath
        val imagePath = item.imagePath
        viewModelScope.launch(Dispatchers.IO) {
            try {
                ImageIO.write(copy, "png", File(path))
                updateHasSidecarMask(imagePath, true)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
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

    fun setMaskEditEnabled(enabled: Boolean) {
        _uiState.update { it.copy(maskEditEnabled = enabled) }
    }

    fun setMaskOnly(enabled: Boolean) {
        _uiState.update { it.copy(maskOnly = enabled) }
        rebuildPreview()
    }

    fun setBrushRadius(radius: Float) {
        _uiState.update { it.copy(brushRadiusPx = radius.coerceIn(2f, 64f)) }
    }

    fun setBrushFeather(feather: Float) {
        _uiState.update { it.copy(brushFeather = feather.coerceIn(0f, 1f)) }
    }

    fun setBrushStrength(strength: Float) {
        _uiState.update { it.copy(brushStrength = strength.coerceIn(0.05f, 1f)) }
    }

    fun beginMaskStroke(imageX: Float, imageY: Float, erase: Boolean) {
        val canvas = mask ?: return
        strokeX = imageX
        strokeY = imageY
        strokeErase = erase
        patchPreview(canvas.stroke(imageX, imageY, imageX, imageY, brushRadius(), brushFeather(), brushStrength(), erase))
        publishPreview(force = true)
    }

    fun continueMaskStroke(imageX: Float, imageY: Float) {
        val canvas = mask ?: return
        val x0 = strokeX
        val y0 = strokeY
        if (x0 == null || y0 == null) {
            beginMaskStroke(imageX, imageY, strokeErase)
            return
        }
        patchPreview(
            canvas.stroke(x0, y0, imageX, imageY, brushRadius(), brushFeather(), brushStrength(), strokeErase)
        )
        strokeX = imageX
        strokeY = imageY
        publishPreview(force = false)
    }

    /** Alt+wheel over the canvas: [steps] notches, positive growing the brush. */
    fun resizeBrushBy(steps: Int) {
        if (steps == 0) return
        _uiState.update { it.copy(brushRadiusPx = nudgeBrushRadius(it.brushRadiusPx, steps)) }
    }

    /** Pointer left the image mid-stroke; the next in-image sample starts a new segment. */
    fun leaveMaskImage() {
        strokeX = null
        strokeY = null
    }

    fun endMaskStroke() {
        strokeX = null
        strokeY = null
        publishPreview(force = true)
    }

    fun invertMask() {
        val canvas = mask ?: return
        canvas.invert()
        markMaskChanged()
        rebuildPreview()
    }

    fun fillMask(white: Boolean) {
        val canvas = mask ?: return
        canvas.fill(if (white) 255 else 0)
        markMaskChanged()
        rebuildPreview()
    }

    private fun brushRadius(): Float = _uiState.value.brushRadiusPx.coerceAtLeast(1f)

    private fun brushFeather(): Float = _uiState.value.brushFeather

    private fun brushStrength(): Float = _uiState.value.brushStrength

    fun saveMask(item: ImageItem? = _uiState.value.selectedItem) {
        val target = item ?: return
        val canvas = mask ?: return
        val copy = canvas.copyImage()
        viewModelScope.launch(Dispatchers.IO) {
            try {
                ImageIO.write(copy, "png", File(target.maskPath))
                updateHasSidecarMask(target.imagePath, true)
                _uiState.update { it.copy(maskDirty = false) }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun clearMask() {
        val item = _uiState.value.selectedItem ?: return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                File(item.maskPath).delete()
                updateHasSidecarMask(item.imagePath, false)
                loadMaskBuffers(item.copy(hasSidecarMask = false))
                _uiState.update { it.copy(maskDirty = false) }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun scanImageItems(folder: File): List<ImageItem> {
        val validExtensions = listOf("jpg", "jpeg", "png", "webp", "bmp")
        return folder.listFiles()
            ?.filter {
                it.isFile &&
                    validExtensions.contains(it.extension.lowercase()) &&
                    !isMaskSidecar(it)
            }
            ?.sortedBy { it.name.lowercase() }
            ?.map { imgFile ->
                val txtFile = File(folder, "${imgFile.nameWithoutExtension}.txt")
                val maskFile = maskFileFor(imgFile)
                val tags = if (txtFile.exists()) txtFile.readText() else ""
                ImageItem(
                    imagePath = imgFile.absolutePath,
                    txtPath = txtFile.absolutePath,
                    tags = tags,
                    draftTags = null,
                    maskPath = maskFile.absolutePath,
                    hasSidecarMask = maskFile.exists(),
                    hasAlpha = fileHasAlphaChannel(imgFile),
                )
            }
            ?: emptyList()
    }

    private fun loadMaskBuffers(item: ImageItem) {
        val photo = ImageIO.read(File(item.imagePath)) ?: return
        val rgb = toRgb(photo)
        val maskFile = File(item.maskPath)
        val canvas = when {
            maskFile.exists() -> {
                val loaded = ImageIO.read(maskFile)
                if (loaded != null) MaskCanvas.fromGrayImage(loaded, rgb.width, rgb.height)
                else MaskCanvas.white(rgb.width, rgb.height)
            }
            photo.colorModel.hasAlpha() -> MaskCanvas.fromAlpha(photo, rgb.width, rgb.height)
            else -> MaskCanvas.white(rgb.width, rgb.height)
        }
        photoImage = rgb
        mask = canvas
        rebuildPreview()
        _uiState.update {
            it.copy(
                sourceWidth = rgb.width,
                sourceHeight = rgb.height,
                maskDirty = false,
            )
        }
    }

    private fun clearMaskBuffers() {
        photoImage = null
        mask = null
        previewImage = null
        cachedPreview = null
        _uiState.update {
            it.copy(sourceWidth = 0, sourceHeight = 0, maskPreviewRevision = maskRevision + 1)
        }
        maskRevision += 1
    }

    private fun markMaskChanged() {
        maskRevision += 1
        _uiState.update { it.copy(maskDirty = true, maskPreviewRevision = maskRevision) }
    }

    private fun rebuildPreview() {
        val photo = photoImage ?: return
        val canvas = mask ?: return
        val maskOnly = _uiState.value.maskOnly
        val out = BufferedImage(photo.width, photo.height, BufferedImage.TYPE_INT_RGB)
        for (y in 0 until photo.height) {
            for (x in 0 until photo.width) {
                out.setRGB(x, y, previewPixel(photo.getRGB(x, y), canvas.pixel(x, y), maskOnly))
            }
        }
        previewImage = out
        cachedPreview = out.toComposeImageBitmap()
        maskRevision += 1
        _uiState.update { it.copy(maskPreviewRevision = maskRevision) }
    }

    private fun publishPreview(force: Boolean) {
        val now = System.nanoTime()
        if (!force && now - lastPublishNs < 16_000_000L) return
        lastPublishNs = now
        cachedPreview = previewImage?.toComposeImageBitmap()
        maskRevision += 1
        _uiState.update { it.copy(maskDirty = true, maskPreviewRevision = maskRevision) }
    }

    private fun patchPreview(dirty: MaskDirtyRect) {
        if (dirty.isEmpty) return
        val photo = photoImage ?: return
        val canvas = mask ?: return
        val preview = previewImage ?: return
        val maskOnly = _uiState.value.maskOnly
        val x1 = dirty.right.coerceAtMost(photo.width - 1)
        val y1 = dirty.bottom.coerceAtMost(photo.height - 1)
        for (y in dirty.top..y1) {
            for (x in dirty.left..x1) {
                preview.setRGB(x, y, previewPixel(photo.getRGB(x, y), canvas.pixel(x, y), maskOnly))
            }
        }
    }

    private fun updateHasSidecarMask(imagePath: String, hasSidecarMask: Boolean) {
        _uiState.update { state ->
            val updatedList = state.imageItems.map {
                if (it.imagePath == imagePath) it.copy(hasSidecarMask = hasSidecarMask) else it
            }
            val selected = updatedList.find { it.imagePath == state.selectedItem?.imagePath }
            state.copy(imageItems = updatedList, selectedItem = selected ?: state.selectedItem)
        }
    }

    companion object {
        private fun grayRgb(v: Int): Int {
            val g = v.coerceIn(0, 255)
            return (0xFF shl 24) or (g shl 16) or (g shl 8) or g
        }

        private fun previewPixel(photoRgb: Int, maskV: Int, maskOnly: Boolean): Int {
            if (maskOnly) {
                return grayRgb(maskV)
            }
            val f = 0.25f + 0.75f * (maskV / 255f)
            val r = (((photoRgb shr 16) and 0xFF) * f).roundToInt().coerceIn(0, 255)
            val g = (((photoRgb shr 8) and 0xFF) * f).roundToInt().coerceIn(0, 255)
            val b = ((photoRgb and 0xFF) * f).roundToInt().coerceIn(0, 255)
            return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        private fun toRgb(src: BufferedImage): BufferedImage {
            if (src.type == BufferedImage.TYPE_INT_RGB) return src
            val out = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_RGB)
            val g = out.createGraphics()
            g.drawImage(src, 0, 0, null)
            g.dispose()
            return out
        }
    }
}

// ImageViewModel.kt
package com.acite.axlranko.pages

import androidx.compose.ui.graphics.ImageBitmap
import com.acite.axlranko.IoDispatcher
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.BlobStore
import com.acite.axlranko.data.DatasetRecord
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.data.encodeBase64
import com.acite.axlranko.data.trainDataEntries
import com.acite.axlranko.data.decodeBase64
import com.acite.axlranko.model.ImageItem
import com.acite.axlranko.model.ImageScreenState
import com.acite.axlranko.util.ImageCodecs
import com.acite.axlranko.util.MaskCanvas
import com.acite.axlranko.util.MaskDirtyRect
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
import kotlin.math.roundToInt
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class ImageScreenViewModel(
    private val refreshHub: DatasetRefreshHub,
    private val datasetSelection: DatasetSelection,
    private val ipc: TrainerIpcClient,
    private val blobStore: BlobStore,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ImageScreenState())
    val uiState: StateFlow<ImageScreenState> = _uiState.asStateFlow()

    private var photoArgb: IntArray? = null
    private var photoWidth: Int = 0
    private var photoHeight: Int = 0
    private var mask: MaskCanvas? = null
    private var previewArgb: IntArray? = null
    private var cachedPreview: ImageBitmap? = null
    private var maskRevision = 0
    private var strokeX: Float? = null
    private var strokeY: Float? = null
    private var strokeErase: Boolean = false
    private var lastPublish = TimeSource.Monotonic.markNow()
    private var snapshotPixels: ByteArray? = null
    private var snapshotHasSidecar: Boolean = false
    private var pendingDelete: Boolean = false
    private val drafts = mutableMapOf<String, MaskDraft>()

    private data class MaskDraft(
        val pixels: ByteArray,
        val width: Int,
        val height: Int,
        val pendingDelete: Boolean,
        val photoArgb: IntArray,
        val snapshotPixels: ByteArray?,
        val snapshotHasSidecar: Boolean,
    )

    /** File a Statistics jump asked for, opened once the rescan of its folder finishes. */
    private var pendingSelectTxtPath: String? = null

    init {
        loadData()
        viewModelScope.launch {
            // Another page changed the dataset (tagger, Statistics tag edits, drop, shuffle): rescan
            // and let the merge drop only the drafts whose file no longer matches them.
            refreshHub.events.collect { reloadFromDisk(resetDrafts = false) }
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
                ipc.parsedConfig().second.environment.trainDataEntries()
            } catch (_: Exception) {
                emptyList()
            }
            val selected = datasetSelection.index.value.coerceIn(0, entries.lastIndex.coerceAtLeast(0))
            val currentDir = entries.getOrNull(selected)?.path ?: _uiState.value.dataDir
            if (currentDir.isEmpty()) return@launch
            _uiState.update {
                it.copy(dataDir = currentDir, datasetDirs = entries, datasetDirIndex = selected)
            }

            withContext(IoDispatcher) {
                val currentItemsMap = _uiState.value.imageItems.associateBy { it.imagePath }
                val currentSelectedPath = _uiState.value.selectedItem?.imagePath

                val updatedItems = scanImageItems(currentDir).map { fresh ->
                    val existing = currentItemsMap[fresh.imagePath]
                    if (existing != null && !resetDrafts) {
                        // An unsaved caption draft survives only while its file still holds what the
                        // edit was based on: a caption another page rewrote (tagger, Statistics)
                        // makes the draft stale, and dropping it there is what keeps the editor and
                        // the disk from disagreeing.
                        existing.copy(
                            tags = fresh.tags,
                            draftTags = existing.draftTags.takeIf { fresh.tags == existing.tags },
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
                val entries = ipc.parsedConfig().second.environment.trainDataEntries()
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

        withContext(IoDispatcher) {
            _uiState.update { it.copy(imageItems = scanImageItems(dataDir)) }
        }
    }

    /** The picker switched dataset folders: unsaved mask strokes belong to the folder left behind. */
    fun selectDatasetDir(index: Int) {
        if (index == _uiState.value.datasetDirIndex) return
        stashCurrentMask()
        switchDatasetDir(index)
    }

    /**
     * Jump here from a Statistics thumbnail. That page can have rewritten captions or dropped
     * samples since this list was scanned, so the jump always rescans: [pendingSelectTxtPath] opens
     * the file once it lands, and one that was dropped resolves to nothing instead of a path that
     * is already gone. `resetDrafts = false` keeps an unsaved caption draft and still refreshes
     * the tags read from disk.
     */
    fun selectItemByTxtPath(txtPath: String, datasetDirIndex: Int) {
        stashCurrentMask()
        pendingSelectTxtPath = txtPath
        if (datasetDirIndex != _uiState.value.datasetDirIndex) {
            switchDatasetDir(datasetDirIndex)
        } else {
            reloadFromDisk(resetDrafts = false)
        }
    }

    private fun switchDatasetDir(index: Int) {
        datasetSelection.select(index)
        _uiState.update { it.copy(datasetDirIndex = index) }
        reloadFromDisk(resetDrafts = true)
    }

    fun selectItem(item: ImageItem) {
        val previous = _uiState.value.selectedItem
        if (previous != null && previous.imagePath != item.imagePath) stashCurrentMask()
        _uiState.update { state ->
            state.copy(
                selectedItem = item,
                editorText = item.currentTags,
                maskDirty = drafts[item.imagePath] != null,
            )
        }
        viewModelScope.launch(IoDispatcher) { loadMaskBuffers(item) }
    }

    private fun stashCurrentMask() {
        val item = _uiState.value.selectedItem ?: return
        val canvas = mask ?: return
        val photo = photoArgb ?: return
        if (!_uiState.value.maskDirty) {
            drafts.remove(item.imagePath)
            return
        }
        drafts[item.imagePath] = MaskDraft(
            pixels = canvas.copyBytes(),
            width = canvas.width,
            height = canvas.height,
            pendingDelete = pendingDelete,
            photoArgb = photo,
            snapshotPixels = snapshotPixels,
            snapshotHasSidecar = snapshotHasSidecar,
        )
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

        viewModelScope.launch(IoDispatcher) {
            try {
                ipc.captionWrite(itemToSave.directory, itemToSave.stem, itemToSave.currentTags)

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
                // A saved caption changes the Statistics tag counts: tell the other pages.
                refreshHub.notifyDatasetChanged()
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
        val deleteSidecar = pendingDelete
        val png = if (deleteSidecar) null else ImageCodecs.encodeGrayPng(canvas.width, canvas.height, canvas.copyBytes())
        viewModelScope.launch(IoDispatcher) {
            try {
                if (deleteSidecar) {
                    ipc.maskDelete(target.directory, target.stem)
                    updateHasSidecarMask(target.imagePath, false)
                    snapshotHasSidecar = false
                    snapshotPixels = canvas.copyBytes()
                } else {
                    ipc.maskWrite(target.directory, target.stem, encodeBase64(png!!))
                    updateHasSidecarMask(target.imagePath, true)
                    snapshotHasSidecar = true
                    snapshotPixels = canvas.copyBytes()
                }
                pendingDelete = false
                drafts.remove(target.imagePath)
                _uiState.update { it.copy(maskDirty = false) }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun resetMask() {
        val canvas = mask ?: return
        val snap = snapshotPixels
        if (snap != null && snap.size == canvas.width * canvas.height) {
            mask = MaskCanvas.fromBytes(snap, canvas.width, canvas.height)
        } else {
            mask = MaskCanvas.white(canvas.width, canvas.height)
        }
        pendingDelete = false
        _uiState.value.selectedItem?.imagePath?.let { drafts.remove(it) }
        _uiState.update { it.copy(maskDirty = false) }
        rebuildPreview()
    }

    fun clearMask() {
        val canvas = mask ?: return
        canvas.fill(255)
        pendingDelete = true
        markMaskChanged()
        rebuildPreview()
    }

    private suspend fun scanImageItems(directory: String): List<ImageItem> {
        if (directory.isEmpty()) return emptyList()
        return ipc.datasetList(directory).items.map { it.toImageItem() }
    }

    private fun DatasetRecord.toImageItem(): ImageItem {
        val parent = image.substringBeforeLast('/', missingDelimiterValue = ".")
        return ImageItem(
            directory = parent,
            stem = stem,
            imagePath = image,
            txtPath = txt,
            tags = tags.joinToString(", "),
            draftTags = null,
            maskPath = mask,
            hasSidecarMask = hasSidecarMask,
            hasAlpha = hasAlpha,
            width = width,
            height = height,
        )
    }

    private suspend fun loadMaskBuffers(item: ImageItem) {
        val draft = drafts[item.imagePath]
        if (draft != null) {
            photoArgb = draft.photoArgb
            photoWidth = draft.width
            photoHeight = draft.height
            mask = MaskCanvas.fromBytes(draft.pixels, draft.width, draft.height)
            pendingDelete = draft.pendingDelete
            snapshotPixels = draft.snapshotPixels
            snapshotHasSidecar = draft.snapshotHasSidecar
            previewArgb = IntArray(draft.width * draft.height)
            rebuildPreview()
            _uiState.update {
                it.copy(
                    sourceWidth = draft.width,
                    sourceHeight = draft.height,
                    maskDirty = true,
                )
            }
            return
        }
        val maskBytes = try {
            val b64 = ipc.maskGet(item.directory, item.stem).pngBase64
            if (b64.isNotEmpty()) decodeBase64(b64) else null
        } catch (_: Exception) {
            null
        }
        // JPEG/WebP blobs flatten onto black and drop alpha; PNG keeps it for the no-sidecar fallback.
        val useAlphaFallback = maskBytes == null && item.hasAlpha
        val maxEdge = maxOf(item.width, item.height, 32).coerceIn(32, 4096)
        val photoBytes = try {
            blobStore.get(
                BlobRef(
                    item.imagePath,
                    maxEdge = maxEdge,
                    quality = 90,
                    format = if (useAlphaFallback) "png" else "jpeg",
                ),
            )
        } catch (_: Exception) {
            return
        }
        val rgba = ImageCodecs.decodeRgba(photoBytes) ?: return
        val canvas = when {
            maskBytes != null -> {
                val loaded = ImageCodecs.decodeRgba(maskBytes)
                if (loaded != null) {
                    MaskCanvas.fromGrayBytes(luminance(loaded), loaded.width, loaded.height, rgba.width, rgba.height)
                } else {
                    MaskCanvas.white(rgba.width, rgba.height)
                }
            }
            useAlphaFallback -> MaskCanvas.fromAlphaRgba(
                rgba.argb,
                rgba.width,
                rgba.height,
                rgba.width,
                rgba.height,
            )
            else -> MaskCanvas.white(rgba.width, rgba.height)
        }
        photoArgb = rgba.argb
        photoWidth = rgba.width
        photoHeight = rgba.height
        mask = canvas
        pendingDelete = false
        snapshotPixels = canvas.copyBytes()
        snapshotHasSidecar = maskBytes != null
        previewArgb = IntArray(rgba.width * rgba.height)
        rebuildPreview()
        _uiState.update {
            it.copy(
                sourceWidth = rgba.width,
                sourceHeight = rgba.height,
                maskDirty = false,
            )
        }
    }

    private fun clearMaskBuffers() {
        photoArgb = null
        photoWidth = 0
        photoHeight = 0
        mask = null
        previewArgb = null
        cachedPreview = null
        snapshotPixels = null
        pendingDelete = false
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
        val photo = photoArgb ?: return
        val canvas = mask ?: return
        val out = previewArgb ?: IntArray(photoWidth * photoHeight).also { previewArgb = it }
        val maskOnly = _uiState.value.maskOnly
        val w = photoWidth
        for (y in 0 until photoHeight) {
            val row = y * w
            for (x in 0 until w) {
                out[row + x] = previewPixel(photo[row + x], canvas.pixel(x, y), maskOnly)
            }
        }
        cachedPreview = ImageCodecs.argbToImageBitmap(photoWidth, photoHeight, out)
        maskRevision += 1
        _uiState.update { it.copy(maskPreviewRevision = maskRevision) }
    }

    private fun publishPreview(force: Boolean) {
        if (!force && lastPublish.elapsedNow() < 16.milliseconds) return
        lastPublish = TimeSource.Monotonic.markNow()
        val out = previewArgb ?: return
        cachedPreview = ImageCodecs.argbToImageBitmap(photoWidth, photoHeight, out)
        maskRevision += 1
        _uiState.update { it.copy(maskDirty = true, maskPreviewRevision = maskRevision) }
    }

    private fun patchPreview(dirty: MaskDirtyRect) {
        if (dirty.isEmpty) return
        val photo = photoArgb ?: return
        val canvas = mask ?: return
        val preview = previewArgb ?: return
        val maskOnly = _uiState.value.maskOnly
        val w = photoWidth
        val x1 = dirty.right.coerceAtMost(w - 1)
        val y1 = dirty.bottom.coerceAtMost(photoHeight - 1)
        for (y in dirty.top..y1) {
            val row = y * w
            for (x in dirty.left..x1) {
                preview[row + x] = previewPixel(photo[row + x], canvas.pixel(x, y), maskOnly)
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

        private fun luminance(image: com.acite.axlranko.util.RgbaImage): ByteArray {
            val out = ByteArray(image.width * image.height)
            for (i in out.indices) {
                val c = image.argb[i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                out[i] = ((r + g + b) / 3).toByte()
            }
            return out
        }
    }
}

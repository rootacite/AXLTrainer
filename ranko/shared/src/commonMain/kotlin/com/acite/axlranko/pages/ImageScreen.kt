// ImageScreen.kt
package com.acite.axlranko.pages

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import com.acite.axlranko.model.ImageItem
import com.acite.axlranko.model.ImageScreenState
import com.acite.axlranko.pages.components.AspectLockedAsyncImage
import com.acite.axlranko.pages.components.MaskPreview
import com.acite.axlranko.util.BRUSH_RADIUS_MAX
import com.acite.axlranko.util.BRUSH_RADIUS_MIN
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.zacsweers.metrox.viewmodel.metroViewModel
import com.acite.axlranko.pages.components.DatasetDirBar
import com.acite.axlranko.pages.components.datasetDirLabel
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.ui.isPortrait
import com.acite.axlranko.ui.wheelScrollsHorizontally
import com.acite.axlranko.ui.pointerIconHorizontalResize
import com.acite.axlranko.ui.pointerIconVerticalResize

@Composable
public fun ImagesScreen(
    viewModel: ImageScreenViewModel = metroViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    val colors = rankoColors
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val totalWidthPx = constraints.maxWidth.toFloat()
        val portrait = isPortrait(maxWidth, maxHeight)

        Column(modifier = Modifier.fillMaxSize()) {
            DatasetDirBar(
                labels = uiState.datasetDirs.map { datasetDirLabel(it.path, it.repeat) },
                selected = uiState.datasetDirIndex,
                onSelect = viewModel::selectDatasetDir,
                modifier = Modifier.padding(start = 8.dp, end = 8.dp, top = 6.dp)
            )

            if (portrait) {
                BoxWithConstraints(modifier = Modifier.fillMaxWidth().weight(1f)) {
                    val totalHeightPx = constraints.maxHeight.toFloat()
                    Column(modifier = Modifier.fillMaxSize()) {
                        ImageThumbStrip(
                            uiState = uiState,
                            horizontal = true,
                            onSelect = viewModel::selectItem,
                            modifier = Modifier.fillMaxWidth().weight(uiState.portraitStripWeight),
                        )
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp)
                                .pointerHoverIcon(pointerIconVerticalResize)
                                .pointerInput(totalHeightPx) {
                                    detectVerticalDragGestures { _, dragAmount ->
                                        if (totalHeightPx > 0f) {
                                            viewModel.updatePortraitStripWeight(
                                                uiState.portraitStripWeight + dragAmount / totalHeightPx,
                                            )
                                        }
                                    }
                                },
                            contentAlignment = Alignment.Center,
                        ) {
                            HorizontalDivider(thickness = 1.dp, color = colors.stroke.copy(alpha = 0.55f))
                        }
                        ImageEditorPane(
                            uiState = uiState,
                            viewModel = viewModel,
                            modifier = Modifier.fillMaxWidth().weight(1f - uiState.portraitStripWeight),
                        )
                    }
                }
            } else Row(modifier = Modifier.fillMaxWidth().weight(1f)) {

            ImageThumbStrip(
                uiState = uiState,
                horizontal = false,
                onSelect = viewModel::selectItem,
                modifier = Modifier.fillMaxHeight().weight(uiState.leftWeight),
            )

            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(8.dp)
                    .pointerHoverIcon(pointerIconHorizontalResize)
                    .pointerInput(totalWidthPx) {
                        detectHorizontalDragGestures { _, dragAmount ->
                            if (totalWidthPx > 0) {
                                val fraction = dragAmount / totalWidthPx
                                viewModel.updateLeftWeight(uiState.leftWeight + fraction)
                            }
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                VerticalDivider(
                    thickness = 1.dp,
                    color = colors.stroke.copy(alpha = 0.55f)
                )
            }

            ImageEditorPane(
                uiState = uiState,
                viewModel = viewModel,
                modifier = Modifier.fillMaxHeight().weight(1f - uiState.leftWeight),
            )
        }
        }
    }
}

@Composable
private fun ImageThumbStrip(
    uiState: ImageScreenState,
    horizontal: Boolean,
    onSelect: (ImageItem) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (horizontal) {
        val listState = rememberLazyListState()
        LazyRow(
            state = listState,
            modifier = modifier.wheelScrollsHorizontally(listState),
            contentPadding = PaddingValues(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(uiState.imageItems, key = { it.imagePath }) { item ->
                val ratio = if (item.height > 0) item.width.toFloat() / item.height.toFloat() else 1f
                ImageThumbCell(
                    item = item,
                    selected = uiState.selectedItem?.imagePath == item.imagePath,
                    maskDirty = uiState.maskDirty,
                    onSelect = { onSelect(item) },
                    modifier = Modifier
                        .fillMaxHeight()
                        .aspectRatio(ratio, matchHeightConstraintsFirst = true),
                    scale = ContentScale.FillHeight,
                )
            }
        }
    } else {
        LazyColumn(
            modifier = modifier,
            contentPadding = PaddingValues(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(uiState.imageItems, key = { it.imagePath }) { item ->
                ImageThumbCell(
                    item = item,
                    selected = uiState.selectedItem?.imagePath == item.imagePath,
                    maskDirty = uiState.maskDirty,
                    onSelect = { onSelect(item) },
                    modifier = Modifier.fillMaxWidth(),
                    scale = ContentScale.FillWidth,
                )
            }
        }
    }
}

@Composable
private fun ImageThumbCell(
    item: ImageItem,
    selected: Boolean,
    maskDirty: Boolean,
    onSelect: () -> Unit,
    modifier: Modifier,
    scale: ContentScale,
) {
    val colors = rankoColors
    val tokens = rankoTokens
    val borderColor = when {
        item.isDirty -> colors.qualityRed
        selected -> colors.accentPink.copy(alpha = 0.28f)
        else -> Color.White.copy(alpha = 0.08f)
    }
    Box(
        modifier = modifier
            .clip(tokens.panel)
            .background(
                if (selected) colors.accentPink.copy(alpha = 0.16f)
                else colors.bgCard.copy(alpha = 0.55f),
            )
            .border(1.dp, borderColor, tokens.panel)
            .clickable(onClick = onSelect),
    ) {
        AspectLockedAsyncImage(
            path = item.imagePath,
            width = item.width,
            height = item.height,
            contentScale = scale,
            filterQuality = FilterQuality.High,
            modifier = if (scale == ContentScale.FillWidth) Modifier.fillMaxWidth() else Modifier.fillMaxSize(),
        )
        if (item.hasMask || (selected && maskDirty)) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(6.dp)
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(if (selected && maskDirty) colors.qualityRed else colors.accentPink),
            )
        }
    }
}

@Composable
private fun ImageEditorPane(
    uiState: ImageScreenState,
    viewModel: ImageScreenViewModel,
    modifier: Modifier = Modifier,
) {
    val colors = rankoColors
    val tokens = rankoTokens
    BoxWithConstraints(modifier) {
        val totalHeightPx = constraints.maxHeight.toFloat()
        Column(modifier = Modifier.fillMaxSize()) {

                    val hasSelection = uiState.selectedItem != null
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 36.dp)
                            .padding(start = 8.dp, end = 8.dp, top = 8.dp)
                            .horizontalScroll(rememberScrollState()),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        CapsuleButton(
                            text = "Mask",
                            onClick = { viewModel.setMaskEditEnabled(!uiState.maskEditEnabled) },
                            enabled = hasSelection,
                            compact = true,
                            emphasized = uiState.maskEditEnabled,
                        )
                        CapsuleButton(
                            text = "Mask only",
                            onClick = { viewModel.setMaskOnly(!uiState.maskOnly) },
                            enabled = hasSelection,
                            compact = true,
                            emphasized = uiState.maskOnly,
                        )
                        Text(
                            text = "Brush ${uiState.brushRadiusPx.toInt()}",
                            color = colors.textDim,
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Slider(
                            value = uiState.brushRadiusPx,
                            onValueChange = { viewModel.setBrushRadius(it) },
                            valueRange = BRUSH_RADIUS_MIN..BRUSH_RADIUS_MAX,
                            enabled = hasSelection,
                            modifier = Modifier.width(120.dp),
                        )
                        Text(
                            text = "Feather ${(uiState.brushFeather * 100).toInt()}%",
                            color = colors.textDim,
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Slider(
                            value = uiState.brushFeather,
                            onValueChange = { viewModel.setBrushFeather(it) },
                            valueRange = 0f..1f,
                            enabled = hasSelection,
                            modifier = Modifier.width(100.dp),
                        )
                        Text(
                            text = "Strength ${(uiState.brushStrength * 100).toInt()}%",
                            color = colors.textDim,
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Slider(
                            value = uiState.brushStrength,
                            onValueChange = { viewModel.setBrushStrength(it) },
                            valueRange = 0.05f..1f,
                            enabled = hasSelection,
                            modifier = Modifier.width(100.dp),
                        )
                        CapsuleButton(
                            text = "Invert",
                            onClick = { viewModel.invertMask() },
                            enabled = hasSelection,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Fill white",
                            onClick = { viewModel.fillMask(white = true) },
                            enabled = hasSelection,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Fill black",
                            onClick = { viewModel.fillMask(white = false) },
                            enabled = hasSelection,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Clear",
                            onClick = { viewModel.clearMask() },
                            enabled = hasSelection && (
                                uiState.selectedItem?.hasSidecarMask == true || uiState.maskDirty
                            ),
                            compact = true,
                            danger = true,
                        )
                        CapsuleButton(
                            text = "Reset mask",
                            onClick = { viewModel.resetMask() },
                            enabled = hasSelection && uiState.maskDirty,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Save mask",
                            onClick = { viewModel.saveMask() },
                            enabled = hasSelection && uiState.maskDirty,
                            compact = true,
                            emphasized = true,
                        )
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(uiState.topWeight)
                            .padding(8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        val previewRevision = uiState.maskPreviewRevision
                        val previewBitmap = remember(previewRevision) { viewModel.previewBitmap() }
                        if (uiState.selectedItem != null && previewBitmap != null) {
                            MaskPreview(
                                bitmap = previewBitmap,
                                sourceWidth = uiState.sourceWidth,
                                sourceHeight = uiState.sourceHeight,
                                editing = uiState.maskEditEnabled,
                                brushRadius = uiState.brushRadiusPx,
                                brushFeather = uiState.brushFeather,
                                onStrokeStart = { x, y, erase -> viewModel.beginMaskStroke(x, y, erase) },
                                onStrokeMove = { x, y -> viewModel.continueMaskStroke(x, y) },
                                onStrokeLeaveImage = { viewModel.leaveMaskImage() },
                                onStrokeEnd = { viewModel.endMaskStroke() },
                                onBrushResize = { viewModel.resizeBrushBy(it) },
                                modifier = Modifier.fillMaxSize(),
                            )
                        } else if (uiState.selectedItem != null) {
                            AsyncImage(
                                model = BlobRef(
                                    uiState.selectedItem!!.imagePath,
                                    maxEdge = maxOf(uiState.sourceWidth, uiState.sourceHeight, 256).coerceIn(32, 4096),
                                    quality = LocalThumbnailQuality.current,
                                ),
                                contentDescription = null,
                                contentScale = ContentScale.Fit,
                                filterQuality = FilterQuality.High,
                                modifier = Modifier.fillMaxSize()
                            )
                        } else {
                            Text(
                                text = "Select an image to edit tags",
                                color = colors.textDim
                            )
                        }
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .pointerHoverIcon(pointerIconVerticalResize)
                            .pointerInput(totalHeightPx) {
                                detectVerticalDragGestures { _, dragAmount ->
                                    if (totalHeightPx > 0) {
                                        val fraction = dragAmount / totalHeightPx
                                        viewModel.updateTopWeight(uiState.topWeight + fraction)
                                    }
                                }
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        HorizontalDivider(
                            thickness = 1.dp,
                            color = colors.stroke.copy(alpha = 0.55f)
                        )
                    }

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f - uiState.topWeight)
                            .padding(8.dp)
                    ) {
                        OutlinedTextField(
                            value = uiState.editorText,
                            onValueChange = { viewModel.updateEditorText(it) },
                            modifier = Modifier.weight(1f).fillMaxWidth(),
                            enabled = uiState.selectedItem != null,
                            label = { Text("Tags") },
                            shape = tokens.panel,
                            colors = rankoFieldColors(),
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            horizontalArrangement = Arrangement.End
                        ) {
                            CapsuleButton(
                                text = "Reset",
                                onClick = { viewModel.resetEditorText() },
                                enabled = uiState.selectedItem?.isDirty == true,
                                compact = true,
                                modifier = Modifier.padding(end = 8.dp),
                            )
                            CapsuleButton(
                                text = "Save",
                                onClick = { viewModel.saveTags() },
                                enabled = uiState.selectedItem?.isDirty == true,
                                compact = true,
                                emphasized = true,
                            )
                        }
                    }
                }
            }
}
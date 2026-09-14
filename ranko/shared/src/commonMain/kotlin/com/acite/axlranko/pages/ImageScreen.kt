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
import androidx.compose.foundation.lazy.items
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
import com.acite.axlranko.pages.components.AspectLockedAsyncImage
import com.acite.axlranko.pages.components.MaskPreview
import com.acite.axlranko.util.BRUSH_RADIUS_MAX
import com.acite.axlranko.util.BRUSH_RADIUS_MIN
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.zacsweers.metrox.viewmodel.metroViewModel
import java.awt.Cursor
import java.io.File

@Composable
public fun ImagesScreen(
    viewModel: ImageScreenViewModel = metroViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    val colors = rankoColors
    val tokens = rankoTokens
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val totalWidthPx = constraints.maxWidth.toFloat()

        Row(modifier = Modifier.fillMaxSize()) {

            Box(modifier = Modifier.fillMaxHeight().weight(uiState.leftWeight)) {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(uiState.imageItems, key = { it.imagePath }) { item ->
                        val isSelected = uiState.selectedItem?.imagePath == item.imagePath
                        val isDirty = item.isDirty

                        val borderColor = when {
                            isDirty -> colors.qualityRed
                            isSelected -> colors.accentPink.copy(alpha = 0.28f)
                            else -> Color.White.copy(alpha = 0.08f)
                        }
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(tokens.panel)
                                .background(
                                    if (isSelected) colors.accentPink.copy(alpha = 0.16f)
                                    else colors.bgCard.copy(alpha = 0.55f)
                                )
                                .border(1.dp, borderColor, tokens.panel)
                                .clickable { viewModel.selectItem(item) }
                        ) {
                            AspectLockedAsyncImage(
                                file = File(item.imagePath),
                                contentScale = ContentScale.FillWidth,
                                filterQuality = FilterQuality.High,
                                modifier = Modifier.fillMaxWidth()
                            )
                            val showMaskBadge = item.hasMask || (isSelected && uiState.maskDirty)
                            if (showMaskBadge) {
                                Box(
                                    modifier = Modifier
                                        .align(Alignment.TopEnd)
                                        .padding(6.dp)
                                        .size(8.dp)
                                        .clip(CircleShape)
                                        .background(
                                            if (isSelected && uiState.maskDirty) colors.qualityRed
                                            else colors.accentPink
                                        )
                                )
                            }
                        }
                    }
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(8.dp)
                    .pointerHoverIcon(PointerIcon(Cursor(Cursor.E_RESIZE_CURSOR)))
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

            BoxWithConstraints(modifier = Modifier.fillMaxHeight().weight(1f - uiState.leftWeight)) {
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
                                model = File(uiState.selectedItem!!.imagePath),
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
                            .pointerHoverIcon(PointerIcon(Cursor(Cursor.N_RESIZE_CURSOR)))
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
    }
}
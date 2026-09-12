// ImageScreen.kt
package com.acite.axlranko.pages

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(uiState.topWeight)
                            .padding(8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        uiState.selectedItem?.let { v ->
                            AsyncImage(
                                model = File(v.imagePath),
                                contentDescription = null,
                                contentScale = ContentScale.Fit,
                                filterQuality = FilterQuality.High,
                                modifier = Modifier.fillMaxSize()
                            )
                        } ?: run {
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
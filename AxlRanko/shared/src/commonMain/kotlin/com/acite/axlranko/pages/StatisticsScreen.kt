package com.acite.axlranko.pages

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.lazy.staggeredgrid.LazyVerticalStaggeredGrid
import androidx.compose.foundation.lazy.staggeredgrid.StaggeredGridCells
import androidx.compose.foundation.lazy.staggeredgrid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import dev.zacsweers.metrox.viewmodel.metroViewModel
import java.awt.Cursor

import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.rememberScrollbarAdapter
import com.acite.axlranko.Screen
import com.acite.axlranko.StageViewModel
import com.acite.axlranko.model.StatisticsUiState

/**
 * Map color based on absolute frequency. 0% Blue -> 50% Green -> 100% Red
 */
private fun getFrequencyColor(frequency: Float): Color {
    val normalized = (frequency / 100f).coerceIn(0f, 1f)
    // Hue ranges from 240 (Blue) down to 0 (Red)
    val hue = 240f * (1f - normalized)
    return Color.hsv(hue = hue, saturation = 0.65f, value = 0.85f)
}

@Composable
fun StatisticsScreen(
    viewModel: StatisticsScreenViewModel = metroViewModel(),
    smViewModel: StageViewModel = metroViewModel(),
    iviewModel: ImageScreenViewModel = metroViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    // 1. Critical Error / Safety Fuse State
    if (uiState.errorMessage != null) {
        Box(
            modifier = Modifier.fillMaxSize().padding(32.dp),
            contentAlignment = Alignment.Center
        ) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = "Error",
                        modifier = Modifier.size(64.dp),
                        tint = MaterialTheme.colorScheme.error
                    )
                    Text(
                        text = uiState.errorMessage!!,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
        return
    }

    // 2. Loading State
    if (uiState.isLoading) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        return
    }

    // 3. Main Interface Layout
    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        val totalWidthPx = constraints.maxWidth.toFloat()

        Row(modifier = Modifier.fillMaxSize()) {

            // Left Panel: Tag Bar Chart
            Box(modifier = Modifier.fillMaxHeight().weight(uiState.leftWeight)) {
                Column(modifier = Modifier.fillMaxSize()) {
                    Text(
                        text = "Dataset Tag Distribution",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(16.dp)
                    )

                    val listState = rememberLazyListState()
                    val totalItems = uiState.tagStats.size

                    Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.fillMaxSize(),
                            contentPadding = PaddingValues(start = 16.dp, end = 24.dp, top = 8.dp, bottom = 8.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            items(uiState.tagStats, key = { it.tag }) { stat ->
                                val isSelected = uiState.selectedTags.contains(stat.tag)
                                val offsetX by animateDpAsState(
                                    targetValue = if (isSelected) 16.dp else 0.dp,
                                    animationSpec = tween(300)
                                )
                                val barColor = getFrequencyColor(stat.frequency)

                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(36.dp)
                                        .animateItem()
                                        .offset(x = offsetX)
                                        .clip(RoundedCornerShape(4.dp))
                                        .background(
                                            if (isSelected) MaterialTheme.colorScheme.surfaceVariant
                                            else Color.Transparent
                                        )
                                        .clickable { viewModel.toggleTagSelection(stat.tag) }
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxHeight()
                                            .fillMaxWidth(fraction = (stat.frequency / 100f).coerceIn(0.01f, 1f))
                                            .background(barColor)
                                    )
                                    Row(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .padding(horizontal = 8.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        Text(
                                            text = stat.tag,
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                            color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                                        )
                                        Text(
                                            text = "${stat.count} (%.1f%%)".format(stat.frequency),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                                        )
                                    }
                                }
                            }
                        }

                        if (totalItems > 0) {
                            VerticalScrollbar(
                                modifier = Modifier
                                    .align(Alignment.CenterEnd)
                                    .fillMaxHeight()
                                    .padding(vertical = 8.dp),
                                adapter = rememberScrollbarAdapter(listState)
                            )
                        }
                    }
                }
            }

            // Vertical Draggable Divider
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
                VerticalDivider(thickness = 1.dp, color = MaterialTheme.colorScheme.outlineVariant)
            }

            // Right Panel: Thumbnails Grid + Control Panel
            BoxWithConstraints(modifier = Modifier.fillMaxHeight().weight(1f - uiState.leftWeight)) {
                val totalHeightPx = constraints.maxHeight.toFloat()
                Column(modifier = Modifier.fillMaxSize()) {

                    // Top Right: Staggered Thumbnails Grid (Follows Logic Mode)
                    Box(modifier = Modifier.fillMaxWidth().weight(uiState.topWeight)) {
                        val filteredItems = uiState.filteredImages
                        if (filteredItems.isEmpty()) {
                            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                Text(
                                    text = if (uiState.selectedTags.isEmpty()) "Please select tags on the left" else "No images match the logical conditions",
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        } else {
                            // Using StaggeredGrid to preserve original aspect ratios without cropping
                            LazyVerticalStaggeredGrid(
                                columns = StaggeredGridCells.Adaptive(minSize = 140.dp),
                                modifier = Modifier.fillMaxSize(),
                                contentPadding = PaddingValues(8.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalItemSpacing = 8.dp
                            ) {
                                items(filteredItems, key = { it.txtFile.absolutePath }) { item ->
                                    Card(
                                        modifier = Modifier.fillMaxWidth().wrapContentHeight(),
                                        shape = RoundedCornerShape(8.dp),
                                        onClick = {
                                            smViewModel.currentScreen = Screen.Images
                                            iviewModel.selectItemByTxtPath(item.txtFile.absolutePath)
                                        }
                                    ) {
                                        AsyncImage(
                                            model = item.imageFile,
                                            contentDescription = null,
                                            contentScale = ContentScale.FillWidth, // Preserves ratio
                                            filterQuality = FilterQuality.High,
                                            modifier = Modifier.fillMaxWidth().wrapContentHeight(),
                                        )
                                    }
                                }
                            }
                        }
                    }

                    // Horizontal Draggable Divider
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
                        HorizontalDivider(thickness = 1.dp, color = MaterialTheme.colorScheme.outlineVariant)
                    }

                    // Bottom Right: Control Panel
                    ControlPanel(
                        uiState = uiState,
                        viewModel = viewModel,
                        modifier = Modifier.fillMaxWidth().weight(1f - uiState.topWeight)
                    )
                }
            }
        }

        if (uiState.isRefreshing) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().align(Alignment.TopCenter)
            )
        }
    }
}

@Composable
fun ControlPanel(
    uiState: StatisticsUiState,
    viewModel: StatisticsScreenViewModel,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Control Panel", style = MaterialTheme.typography.titleMedium)

        val hasSelection = uiState.selectedTags.isNotEmpty()

        // 1. Logic Mode Switch & Basic Actions
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Logic Mode:")
                Spacer(Modifier.width(8.dp))
                SingleChoiceSegmentedButtonRow {
                    SegmentedButton(
                        selected = uiState.isAndMode,
                        onClick = { viewModel.updateFilterMode(true) },
                        shape = RoundedCornerShape(topStart = 8.dp, bottomStart = 8.dp)
                    ) { Text("Intersection (AND)") }
                    SegmentedButton(
                        selected = !uiState.isAndMode,
                        onClick = { viewModel.updateFilterMode(false) },
                        shape = RoundedCornerShape(topEnd = 8.dp, bottomEnd = 8.dp)
                    ) { Text("Union (OR)") }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { viewModel.clearSelection() }) {
                    Text("Clear Selection")
                }
                OutlinedButton(onClick = { viewModel.invertSelection() }) {
                    Text("Invert Selection")
                }
                Button(
                    onClick = { viewModel.removeSelectedTags() },
                    enabled = hasSelection && !uiState.isRefreshing,
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                ) {
                    Text("Remove Selected")
                }
            }
        }

        HorizontalDivider()

        // 2. Sample Dropping Section
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            OutlinedTextField(
                value = uiState.dropRateText,
                onValueChange = { viewModel.updateDropRateText(it) },
                label = { Text("Drop Rate r (0.0~1.0)") },
                modifier = Modifier.weight(1f),
                singleLine = true
            )
            Button(
                onClick = { viewModel.dropSamples() },
                enabled = hasSelection && !uiState.isRefreshing && (uiState.dropRateText.toFloatOrNull() ?: 0f) in 0.001f..1f,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.tertiary)
            ) {
                Text("Drop Selected Samples")
            }
        }

        // 3. Append / Prepend Tag Section
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            OutlinedTextField(
                value = uiState.newTagText,
                onValueChange = { viewModel.updateNewTagText(it) },
                label = { Text("New Tag Name") },
                modifier = Modifier.weight(1f),
                singleLine = true
            )

            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = uiState.isAddStart,
                    onClick = { viewModel.updateAddPosition(true) }
                )
                Text("Prepend", modifier = Modifier.clickable { viewModel.updateAddPosition(true) })
                Spacer(Modifier.width(8.dp))
                RadioButton(
                    selected = !uiState.isAddStart,
                    onClick = { viewModel.updateAddPosition(false) }
                )
                Text("Append", modifier = Modifier.clickable { viewModel.updateAddPosition(false) })
            }

            Button(
                onClick = { viewModel.addTagToTargets() },
                enabled = hasSelection && !uiState.isRefreshing && uiState.newTagText.isNotBlank()
            ) {
                Text("Batch Add")
            }
        }
    }
}
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
import androidx.compose.material.icons.filled.Search
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
import com.acite.axlranko.pages.components.AspectLockedAsyncImage
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.RankoPalette
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.zacsweers.metrox.viewmodel.metroViewModel
import java.awt.Cursor

import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.ui.graphics.lerp
import com.acite.axlranko.Screen
import com.acite.axlranko.StageViewModel
import com.acite.axlranko.model.StatisticsUiState

private fun frequencyColor(frequency: Float, colors: RankoPalette): Color {
    val t = (frequency / 100f).coerceIn(0f, 1f)
    val stops = listOf(
        colors.accentBlue,
        colors.qualityMint,
        colors.qualityYellow,
        colors.qualityOrange,
        colors.qualityRed,
    )
    val scaled = t * (stops.lastIndex)
    val index = scaled.toInt().coerceIn(0, stops.lastIndex - 1)
    return lerp(stops[index], stops[index + 1], scaled - index)
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
            PorcelainCard {
                Column(
                    modifier = Modifier.padding(18.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = "Error",
                        modifier = Modifier.size(64.dp),
                        tint = rankoColors.qualityRed
                    )
                    Text(
                        text = uiState.errorMessage!!,
                        color = rankoColors.text,
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
    val colors = rankoColors
    val tokens = rankoTokens
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
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

                    OutlinedTextField(
                        value = uiState.tagSearchQuery,
                        onValueChange = viewModel::updateTagSearchQuery,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 16.dp, end = 16.dp, bottom = 8.dp),
                        placeholder = { Text("Search tags...") },
                        leadingIcon = { Icon(Icons.Default.Search, contentDescription = "Search") },
                        singleLine = true,
                        shape = tokens.panel,
                        colors = rankoFieldColors(),
                    )

                    val listState = rememberLazyListState()
                    val displayedTagStats = remember(uiState.tagStats, uiState.tagSearchQuery) {
                        val q = uiState.tagSearchQuery.trim()
                        if (q.isEmpty()) uiState.tagStats
                        else uiState.tagStats.filter { it.tag.contains(q, ignoreCase = true) }
                    }
                    val totalItems = displayedTagStats.size

                    Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.fillMaxSize(),
                            contentPadding = PaddingValues(start = 16.dp, end = 24.dp, top = 8.dp, bottom = 8.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            items(displayedTagStats, key = { it.tag }) { stat ->
                                val isSelected = uiState.selectedTags.contains(stat.tag)
                                val offsetX by animateDpAsState(
                                    targetValue = if (isSelected) 16.dp else 0.dp,
                                    animationSpec = tween(300)
                                )
                                val barColor = frequencyColor(stat.frequency, colors)

                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(36.dp)
                                        .animateItem()
                                        .offset(x = offsetX)
                                        .clip(tokens.panel)
                                        .background(
                                            if (isSelected) colors.accentPink.copy(alpha = 0.16f)
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
                                            color = if (isSelected) colors.accentPink else colors.text
                                        )
                                        Text(
                                            text = "${stat.count} (%.1f%%)".format(stat.frequency),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = colors.text.copy(alpha = 0.7f)
                                        )
                                    }
                                }
                            }
                        }

                        if (totalItems == 0 && uiState.tagSearchQuery.isNotBlank()) {
                            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                Text(
                                    text = "No tags match \"${uiState.tagSearchQuery.trim()}\"",
                                    color = colors.textDim
                                )
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
                VerticalDivider(thickness = 1.dp, color = colors.stroke.copy(alpha = 0.55f))
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
                                    color = colors.textDim
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
                                        shape = tokens.panel,
                                        colors = CardDefaults.cardColors(
                                            containerColor = colors.bgCard.copy(alpha = 0.72f)
                                        ),
                                        onClick = {
                                            smViewModel.currentScreen = Screen.Images
                                            iviewModel.selectItemByTxtPath(item.txtFile.absolutePath)
                                        }
                                    ) {
                                        AspectLockedAsyncImage(
                                            file = item.imageFile,
                                            contentScale = ContentScale.FillWidth,
                                            filterQuality = FilterQuality.High,
                                            modifier = Modifier.fillMaxWidth(),
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
                        HorizontalDivider(thickness = 1.dp, color = colors.stroke.copy(alpha = 0.55f))
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
    val colors = rankoColors
    val tokens = rankoTokens
    Column(
        modifier = modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Control Panel", style = MaterialTheme.typography.titleMedium, color = colors.text)

        val hasSelection = uiState.selectedTags.isNotEmpty()

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Logic Mode:", color = colors.text)
                Spacer(Modifier.width(8.dp))
                CapsuleChoice(
                    text = "Intersection (AND)",
                    selected = uiState.isAndMode,
                    onClick = { viewModel.updateFilterMode(true) },
                )
                Spacer(Modifier.width(8.dp))
                CapsuleChoice(
                    text = "Union (OR)",
                    selected = !uiState.isAndMode,
                    onClick = { viewModel.updateFilterMode(false) },
                )

                Spacer(Modifier.width(12.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.clickable { viewModel.updateNotMode(!uiState.isNotMode) }
                ) {
                    Checkbox(
                        checked = uiState.isNotMode,
                        onCheckedChange = { viewModel.updateNotMode(it) }
                    )
                    Text("Not", color = colors.text)
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CapsuleButton(
                    text = "Clear Selection",
                    onClick = { viewModel.clearSelection() },
                    compact = true,
                )
                CapsuleButton(
                    text = "Invert Selection",
                    onClick = { viewModel.invertSelection() },
                    compact = true,
                )
                CapsuleButton(
                    text = "Remove Selected",
                    onClick = { viewModel.removeSelectedTags() },
                    enabled = hasSelection && !uiState.isRefreshing && !uiState.isNotMode,
                    compact = true,
                    danger = true,
                )
            }
        }

        HorizontalDivider(color = colors.stroke.copy(alpha = 0.55f))

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
                singleLine = true,
                shape = tokens.panel,
                colors = rankoFieldColors(),
            )
            CapsuleButton(
                text = "Drop Selected Samples",
                onClick = { viewModel.dropSamples() },
                enabled = hasSelection && !uiState.isRefreshing && !uiState.isNotMode && (uiState.dropRateText.toFloatOrNull() ?: 0f) in 0.001f..1f,
                compact = true,
                danger = true,
            )
        }

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
                singleLine = true,
                shape = tokens.panel,
                colors = rankoFieldColors(),
            )

            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = uiState.isAddStart,
                    onClick = { viewModel.updateAddPosition(true) }
                )
                Text("Prepend", color = colors.text, modifier = Modifier.clickable { viewModel.updateAddPosition(true) })
                Spacer(Modifier.width(8.dp))
                RadioButton(
                    selected = !uiState.isAddStart,
                    onClick = { viewModel.updateAddPosition(false) }
                )
                Text("Append", color = colors.text, modifier = Modifier.clickable { viewModel.updateAddPosition(false) })
            }

            CapsuleButton(
                text = "Batch Add",
                onClick = { viewModel.addTagToTargets() },
                enabled = hasSelection && !uiState.isRefreshing && uiState.newTagText.isNotBlank(),
                compact = true,
                emphasized = true,
            )
        }
    }
}
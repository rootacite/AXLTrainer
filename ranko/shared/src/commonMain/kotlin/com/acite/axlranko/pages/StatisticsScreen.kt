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
import androidx.compose.material.icons.filled.ContentCopy
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.util.TagTranslations
import com.acite.axlranko.util.copyTextToClipboard
import com.acite.axlranko.util.formatFixed
import com.acite.axlranko.pages.components.AspectLockedAsyncImage
import com.acite.axlranko.pages.components.DatasetDirBar
import com.acite.axlranko.pages.components.datasetDirLabel
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.zacsweers.metrox.viewmodel.metroViewModel
import com.acite.axlranko.ui.pointerIconHorizontalResize
import com.acite.axlranko.ui.pointerIconVerticalResize

import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.verticalScroll
import com.acite.axlranko.Screen
import com.acite.axlranko.StageViewModel
import com.acite.axlranko.model.StatisticsUiState
import com.acite.axlranko.pages.components.frequencyColor

@Composable
fun StatisticsScreen(
    viewModel: StatisticsScreenViewModel = metroViewModel(),
    smViewModel: StageViewModel = metroViewModel(),
    iviewModel: ImageScreenViewModel = metroViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    Column(modifier = Modifier.fillMaxSize()) {
        // Above the gates: a folder that no longer exists is exactly when switching matters.
        DatasetDirBar(
            labels = uiState.datasetDirs.map { datasetDirLabel(it.path, it.repeat) },
            selected = uiState.datasetDirIndex,
            onSelect = viewModel::selectDatasetDir,
            modifier = Modifier.padding(start = 12.dp, end = 12.dp, top = 8.dp)
        )

        // 1. Critical Error / Safety Fuse State
        if (uiState.errorMessage != null) {
        Box(
            modifier = Modifier.fillMaxWidth().weight(1f).padding(32.dp),
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
        } else if (uiState.isLoading) {
            // 2. Loading State
            Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
        } else {
        // 3. Main Interface Layout
    val colors = rankoColors
    val tokens = rankoTokens
    BoxWithConstraints(
        modifier = Modifier.fillMaxWidth().weight(1f)
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
                        else uiState.tagStats.filter { TagTranslations.matchesQuery(it.tag, q) }
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
                                    ) {
                                        Row(
                                            modifier = Modifier.weight(1f),
                                            verticalAlignment = Alignment.CenterVertically,
                                        ) {
                                            Text(
                                                text = TagTranslations.display(stat.tag),
                                                style = MaterialTheme.typography.bodyMedium,
                                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                                color = if (isSelected) colors.accentPink else colors.text,
                                                maxLines = 1,
                                                overflow = TextOverflow.Ellipsis,
                                                modifier = Modifier.weight(1f, fill = false),
                                            )
                                            IconButton(
                                                onClick = { copyTextToClipboard(stat.tag) },
                                                modifier = Modifier
                                                    .padding(start = 2.dp)
                                                    .size(28.dp)
                                                    .pointerHoverIcon(PointerIcon.Hand),
                                            ) {
                                                Icon(
                                                    Icons.Default.ContentCopy,
                                                    contentDescription = "Copy English tag",
                                                    modifier = Modifier.size(14.dp),
                                                    tint = if (isSelected) colors.accentPink else colors.textDim,
                                                )
                                            }
                                        }
                                        Text(
                                            text = "${stat.count} (${formatFixed(stat.frequency, 1)}%)",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = colors.text.copy(alpha = 0.7f),
                                            modifier = Modifier.padding(start = 8.dp),
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
                                items(filteredItems, key = { it.txtPath }) { item ->
                                    Card(
                                        modifier = Modifier.fillMaxWidth().wrapContentHeight(),
                                        shape = tokens.panel,
                                        colors = CardDefaults.cardColors(
                                            containerColor = colors.bgCard.copy(alpha = 0.72f)
                                        ),
                                        onClick = {
                                            smViewModel.currentScreen = Screen.Images
                                            // Images may have another dataset folder open; the
                                            // jump follows the folder this thumbnail came from.
                                            iviewModel.selectItemByTxtPath(
                                                item.txtPath,
                                                uiState.datasetDirIndex,
                                            )
                                        }
                                    ) {
                                        AspectLockedAsyncImage(
                                            path = item.imagePath,
                                            width = item.width,
                                            height = item.height,
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
    var confirmShuffle by remember { mutableStateOf(false) }
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
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

        HorizontalDivider(color = colors.stroke.copy(alpha = 0.55f))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Shuffle & Renumber", color = colors.text)
                Text(
                    text = "Renames every sample in this folder to 0001…, in random order. Each " +
                        "caption (.txt) and mask (.mask.png) moves with its image. The latent " +
                        "cache re-encodes once afterwards: it is keyed by file path.",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.textDim
                )
            }
            CapsuleButton(
                text = if (uiState.isShuffling) "Shuffling…" else "Shuffle Dataset",
                onClick = { confirmShuffle = true },
                enabled = !uiState.isShuffling && !uiState.isRefreshing && uiState.imageCount > 0,
                compact = true,
                danger = true,
            )
        }

        uiState.statusMessage?.let { message ->
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = if (uiState.statusIsError) colors.qualityRed else colors.accentLilac,
            )
        }
    }

    if (confirmShuffle) {
        val folder = uiState.datasetDirs.getOrNull(uiState.datasetDirIndex)?.path.orEmpty()
        AlertDialog(
            onDismissRequest = { confirmShuffle = false },
            title = { Text("Shuffle and renumber this folder?") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "All ${uiState.imageCount} images in this folder are renamed to a random " +
                            "0001… sequence. Each caption (.txt) and mask (.mask.png) is renamed " +
                            "with its image, so no sidecar can end up on another sample."
                    )
                    Text("Folder: $folder", style = MaterialTheme.typography.bodySmall)
                    Text(
                        "Image count and captions are unchanged. The latent cache re-encodes " +
                            "once on the next run, because its key is the image path.",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmShuffle = false
                        viewModel.shuffleDataset()
                    }
                ) { Text("Shuffle") }
            },
            dismissButton = {
                TextButton(onClick = { confirmShuffle = false }) { Text("Cancel") }
            }
        )
    }
}
package com.acite.axlranko.pages

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.lazy.staggeredgrid.LazyVerticalStaggeredGrid
import androidx.compose.foundation.lazy.staggeredgrid.StaggeredGridCells
import androidx.compose.foundation.lazy.staggeredgrid.StaggeredGridItemSpan
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import axlranko.shared.generated.resources.Res
import axlranko.shared.generated.resources.app_icon
import coil3.compose.AsyncImage
import com.acite.axlranko.Screen
import com.acite.axlranko.changelog.ChangelogEntry
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.data.showsHelperEndpointSettings
import com.acite.axlranko.generated.AppInfo
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.TagStat
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import com.acite.axlranko.pages.components.runDetailLabel
import com.acite.axlranko.pages.components.runStateLabel
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.zacsweers.metrox.viewmodel.metroViewModel
import org.jetbrains.compose.resources.painterResource

private val HomeWide = 840.dp
private val HomeRailWidth = 345.dp
private const val HOME_RUN_LIMIT = 30
private const val HOME_CHANGELOG_LIMIT = 40

@Composable
fun HomeRoute(
    viewModel: HomeScreenViewModel = metroViewModel(),
    imagesViewModel: ImageScreenViewModel = metroViewModel(),
    statisticsViewModel: StatisticsScreenViewModel = metroViewModel(),
    utilsViewModel: UtilsScreenViewModel = metroViewModel(),
    onOpen: (Screen) -> Unit,
    onOpenRun: (String) -> Unit,
) {
    DisposableEffect(Unit) {
        viewModel.onEnter()
        onDispose { viewModel.onLeave() }
    }
    val home by viewModel.uiState.collectAsState()
    val images by imagesViewModel.uiState.collectAsState()
    val statistics by statisticsViewModel.uiState.collectAsState()
    val utils by utilsViewModel.uiState.collectAsState()
    val helperLine = if (showsHelperEndpointSettings && utils.configPath.isBlank()) {
        "${utils.helperHost}:${utils.helperPort} · ${utils.helperStatus}"
    } else {
        null
    }
    HomeScreen(
        version = AppInfo.version,
        gitHash = AppInfo.gitHash,
        runs = home.runs,
        changelog = AppInfo.changelog,
        dashboard = dashboardHome(home.status, emptyMap()).copy(samplePaths = home.samplePaths),
        avgLoss = home.avgLoss,
        hardware = home.hardware,
        automation = automationHome(home.jobs),
        utilsFacts = utilsHome(utils.form, utils.configPath.isNotBlank(), helperLine),
        tagBars = homeTagBars(statistics.tagStats),
        statisticsLines = statisticsHome(
            dir = statistics.datasetDirs.getOrNull(statistics.datasetDirIndex)?.path.orEmpty(),
            imageCount = statistics.imageCount,
            items = statistics.datasetItems,
            tagStats = statistics.tagStats,
            loading = statistics.isLoading,
            error = statistics.errorMessage,
        ),
        imagesLines = imagesHome(images.dataDir, images.imageItems),
        datasetImages = home.datasetImages,
        onOpen = onOpen,
        onOpenRun = onOpenRun,
    )
}

@Composable
fun HomeScreen(
    version: String,
    gitHash: String,
    runs: List<RunSummary>,
    changelog: List<ChangelogEntry>,
    dashboard: DashboardHome,
    avgLoss: List<MetricPoint>,
    hardware: HardwareStatus?,
    automation: AutomationHome,
    utilsFacts: List<HomeFact>,
    statisticsLines: List<String>,
    tagBars: List<TagStat>,
    imagesLines: List<String>,
    datasetImages: List<String>,
    onOpen: (Screen) -> Unit,
    onOpenRun: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    BoxWithConstraints(modifier.fillMaxSize()) {
        val wide = maxWidth >= HomeWide
        if (wide) {
            Row(
                Modifier.fillMaxSize().padding(28.dp),
                horizontalArrangement = Arrangement.spacedBy(22.dp),
            ) {
                Column(
                    Modifier.width(HomeRailWidth).fillMaxHeight(),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    BrandCard(version = version, gitHash = gitHash)
                    ChangelogCard(
                        entries = changelog,
                        fill = true,
                        modifier = Modifier.weight(1f).fillMaxWidth(),
                    )
                }
                WideHomeGrid(
                    runs = runs,
                    onOpenRun = onOpenRun,
                    dashboard = dashboard,
                    avgLoss = avgLoss,
                    hardware = hardware,
                    automation = automation,
                    utilsFacts = utilsFacts,
                    statisticsLines = statisticsLines,
                    tagBars = tagBars,
                    imagesLines = imagesLines,
                    datasetImages = datasetImages,
                    onOpen = onOpen,
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                )
            }
        } else {
            Column(
                Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                BrandCard(version = version, gitHash = gitHash)
                ChangelogCard(entries = changelog)
                RecentRunsCard(runs = runs, onOpenRun = onOpenRun)
                ModuleCards(
                    wide = false,
                    dashboard = dashboard,
                    avgLoss = avgLoss,
                    automation = automation,
                    utilsFacts = utilsFacts,
                    statisticsLines = statisticsLines,
                    tagBars = tagBars,
                    imagesLines = imagesLines,
                    datasetImages = datasetImages,
                    hardware = hardware,
                    onOpen = onOpen,
                )
            }
        }
    }
}

@Composable
private fun BrandCard(version: String, gitHash: String, modifier: Modifier = Modifier) {
    val colors = rankoColors
    PorcelainCard(modifier) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Image(
                painter = painterResource(Res.drawable.app_icon),
                contentDescription = "AxlRanko",
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
                    .border(3.dp, Color.White, CircleShape),
            )
            Column {
                Text(
                    "AxlRanko",
                    color = colors.text,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.SemiBold,
                    lineHeight = 32.sp,
                )
                Text(
                    "$version  ·  $gitHash",
                    color = colors.textDim,
                    fontSize = 12.sp,
                )
            }
        }
    }
}

@Composable
private fun RecentRunsCard(
    runs: List<RunSummary>,
    onOpenRun: (String) -> Unit,
) {
    val colors = rankoColors
    val shown = runs.take(HOME_RUN_LIMIT)
    PorcelainCard {
        Text(
            "Recent runs",
            color = colors.textDim,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold,
        )
        if (shown.isEmpty()) {
            Text("No runs", color = colors.textDim, fontSize = 14.sp)
        } else {
            Column(
                Modifier
                    .fillMaxWidth()
                    .heightIn(max = 320.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                shown.forEach { run ->
                    RunRow(run = run, onOpen = { onOpenRun(run.runId) })
                }
            }
        }
    }
}

@Composable
private fun RunRow(run: RunSummary, onOpen: () -> Unit) {
    val colors = rankoColors
    val tokens = rankoTokens
    val state = runStateLabel(run)
    Row(
        Modifier
            .fillMaxWidth()
            .clip(tokens.panel)
            .background(Color.White.copy(alpha = 0.05f))
            .clickable(onClick = onOpen)
            .pointerHoverIcon(PointerIcon.Hand)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(
                run.outputName.ifBlank { run.runId },
                color = colors.text,
                fontSize = 15.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                runDetailLabel(run).ifBlank { run.runId },
                color = colors.textDim,
                fontSize = 12.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        if (state != null) {
            Text(
                state,
                color = if (run.live) colors.qualityMint else colors.textDim,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
}

@Composable
private fun ChangelogCard(
    entries: List<ChangelogEntry>,
    fill: Boolean = false,
    modifier: Modifier = Modifier,
) {
    val colors = rankoColors
    val shown = entries.take(HOME_CHANGELOG_LIMIT)
    PorcelainCard(modifier) {
        Text(
            "What's new",
            color = colors.textDim,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold,
        )
        Column(
            Modifier
                .fillMaxWidth()
                .then(if (fill) Modifier.weight(1f) else Modifier.heightIn(max = 320.dp))
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (shown.isEmpty()) {
                Text("Nothing bundled yet.", color = colors.textDim, fontSize = 14.sp)
            } else {
                shown.forEach { entry -> ChangelogRow(entry) }
            }
        }
    }
}

@Composable
private fun ChangelogRow(entry: ChangelogEntry) {
    val colors = rankoColors
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            entry.gitTags.forEach { tag -> ChangelogChip(tag, colors.accentPink) }
            entry.kind?.let { kind ->
                val accent = when (kind.lowercase()) {
                    "feat" -> colors.accentPink
                    "fix" -> colors.accentBlue
                    "doc" -> colors.qualityMint
                    else -> colors.accentLilac
                }
                ChangelogChip(kind, accent)
            }
            Text(
                entry.date,
                color = colors.textDim,
                fontSize = 12.sp,
                modifier = Modifier.align(Alignment.CenterVertically),
            )
        }
        Text(entry.title, color = colors.text, fontSize = 14.sp)
    }
}

@Composable
private fun ChangelogChip(label: String, accent: Color) {
    Box(
        Modifier
            .clip(rankoTokens.capsule)
            .background(accent.copy(alpha = 0.18f))
            .padding(horizontal = 8.dp, vertical = 2.dp),
    ) {
        Text(label, color = accent, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ModuleCards(
    wide: Boolean,
    dashboard: DashboardHome,
    avgLoss: List<MetricPoint>,
    hardware: HardwareStatus?,
    automation: AutomationHome,
    utilsFacts: List<HomeFact>,
    statisticsLines: List<String>,
    tagBars: List<TagStat>,
    imagesLines: List<String>,
    datasetImages: List<String>,
    onOpen: (Screen) -> Unit,
) {
    val dashboardExtra: @Composable () -> Unit = {
        if (hardware != null) {
            Text(
                homeHardwareLine(hardware),
                color = rankoColors.textDim,
                fontSize = 12.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        HomeAvgLossChart(avgLoss)
        SampleThumbs(dashboard.samplePaths, emptyLabel = "No samples")
    }
    val automationExtra: @Composable () -> Unit = {
        FittingThumbGrid(automation.imagePaths, emptyLabel = "No images")
    }
    val statisticsExtra: @Composable () -> Unit = { HomeTagBars(tagBars) }
    val imagesExtra: @Composable () -> Unit = {
        FittingThumbGrid(datasetImages, emptyLabel = "No images")
    }
    val utilsExtra: @Composable () -> Unit = { HomeFacts(utilsFacts) }
    if (!wide) {
        DashboardModuleCard(dashboard, onOpen, Modifier.fillMaxWidth(), dashboardExtra)
        TextModuleCard("Automation", Icons.Default.AutoAwesome, automation.statusLabel, automation.statusKey, listOf(automation.detail), extra = automationExtra) { onOpen(Screen.Automation) }
        TextModuleCard("Utils", Icons.Default.Build, "Config", "idle", emptyList(), extra = utilsExtra) { onOpen(Screen.Utils) }
        TextModuleCard("Statistics", Icons.Default.Analytics, "Tags", "idle", statisticsLines, extra = statisticsExtra) { onOpen(Screen.Statistics) }
        TextModuleCard("Images", Icons.Default.Image, "Dataset", "idle", imagesLines, extra = imagesExtra) { onOpen(Screen.Images) }
    }
}

/**
 * Two columns whose cards keep their own height, so a short Automation card does not leave a
 * gap above Statistics. Recent runs and Dashboard span both columns.
 */
@Composable
private fun WideHomeGrid(
    runs: List<RunSummary>,
    onOpenRun: (String) -> Unit,
    dashboard: DashboardHome,
    avgLoss: List<MetricPoint>,
    hardware: HardwareStatus?,
    automation: AutomationHome,
    utilsFacts: List<HomeFact>,
    statisticsLines: List<String>,
    tagBars: List<TagStat>,
    imagesLines: List<String>,
    datasetImages: List<String>,
    onOpen: (Screen) -> Unit,
    modifier: Modifier = Modifier,
) {
    val dashboardExtra: @Composable () -> Unit = {
        if (hardware != null) {
            Text(
                homeHardwareLine(hardware),
                color = rankoColors.textDim,
                fontSize = 12.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        HomeAvgLossChart(avgLoss)
        SampleThumbs(dashboard.samplePaths, emptyLabel = "No samples")
    }
    LazyVerticalStaggeredGrid(
        columns = StaggeredGridCells.Fixed(2),
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalItemSpacing = 16.dp,
    ) {
        item(span = StaggeredGridItemSpan.FullLine) {
            RecentRunsCard(runs = runs, onOpenRun = onOpenRun)
        }
        item(span = StaggeredGridItemSpan.FullLine) {
            DashboardModuleCard(dashboard, onOpen, Modifier.fillMaxWidth(), dashboardExtra)
        }
        item {
            TextModuleCard(
                "Automation",
                Icons.Default.AutoAwesome,
                automation.statusLabel,
                automation.statusKey,
                listOf(automation.detail),
                extra = { FittingThumbGrid(automation.imagePaths, emptyLabel = "No images") },
            ) { onOpen(Screen.Automation) }
        }
        item {
            TextModuleCard(
                "Utils",
                Icons.Default.Build,
                "Config",
                "idle",
                emptyList(),
                extra = { HomeFacts(utilsFacts) },
            ) { onOpen(Screen.Utils) }
        }
        item {
            TextModuleCard(
                "Statistics",
                Icons.Default.Analytics,
                "Tags",
                "idle",
                statisticsLines,
                extra = { HomeTagBars(tagBars) },
            ) { onOpen(Screen.Statistics) }
        }
        item {
            TextModuleCard(
                "Images",
                Icons.Default.Image,
                "Dataset",
                "idle",
                imagesLines,
                extra = { FittingThumbGrid(datasetImages, emptyLabel = "No images") },
            ) { onOpen(Screen.Images) }
        }
    }
}

@Composable
private fun DashboardModuleCard(
    dashboard: DashboardHome,
    onOpen: (Screen) -> Unit,
    modifier: Modifier = Modifier,
    extra: @Composable () -> Unit = {},
) {
    ModuleCard(
        title = "Dashboard",
        icon = Icons.AutoMirrored.Filled.ShowChart,
        status = dashboard.statusLabel,
        statusKey = dashboard.statusKey,
        lines = listOf(dashboard.detail),
        onClick = { onOpen(Screen.Dashboard) },
        modifier = modifier,
        extra = extra,
    )
}

@Composable
private fun TextModuleCard(
    title: String,
    icon: ImageVector,
    status: String,
    statusKey: String,
    lines: List<String>,
    modifier: Modifier = Modifier,
    extra: @Composable () -> Unit = {},
    onClick: () -> Unit,
) {
    ModuleCard(
        title = title,
        icon = icon,
        status = status,
        statusKey = statusKey,
        lines = lines,
        onClick = onClick,
        modifier = modifier,
        extra = extra,
    )
}

@Composable
private fun HomeFacts(facts: List<HomeFact>) {
    val colors = rankoColors
    facts.forEach { fact ->
        Text(
            buildAnnotatedString {
                withStyle(SpanStyle(fontWeight = FontWeight.SemiBold, color = colors.text)) {
                    append(fact.name)
                }
                withStyle(SpanStyle(color = colors.textDim)) {
                    append(": ${fact.value}")
                }
            },
            fontSize = 13.sp,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

/**
 * At most two rows. Each thumb is sized so a full row uses the width this card was given.
 */
@Composable
private fun FittingThumbGrid(paths: List<String>, emptyLabel: String) {
    val colors = rankoColors
    if (paths.isEmpty()) {
        Text(emptyLabel, color = colors.textDim, fontSize = 13.sp)
        return
    }
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val gap = 8.dp
        val minThumb = 72.dp
        val cols = ((maxWidth + gap) / (minThumb + gap)).toInt().coerceAtLeast(1)
        val thumb = (maxWidth - gap * (cols - 1)) / cols
        val shown = paths.take(cols * 2)
        Column(verticalArrangement = Arrangement.spacedBy(gap)) {
            shown.chunked(cols).forEach { row ->
                Row(horizontalArrangement = Arrangement.spacedBy(gap)) {
                    row.forEach { path ->
                        AsyncImage(
                            model = BlobRef(path, maxEdge = 512, quality = LocalThumbnailQuality.current),
                            contentDescription = null,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.size(thumb).clip(rankoTokens.panel),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SampleThumbs(paths: List<String>, emptyLabel: String) {
    val colors = rankoColors
    if (paths.isEmpty()) {
        Text(emptyLabel, color = colors.textDim, fontSize = 13.sp)
        return
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        paths.forEach { path ->
            AsyncImage(
                model = BlobRef(path, maxEdge = 512, quality = LocalThumbnailQuality.current),
                contentDescription = null,
                contentScale = ContentScale.Crop,
                modifier = Modifier.size(72.dp).clip(rankoTokens.panel),
            )
        }
    }
}

@Composable
private fun ModuleCard(
    title: String,
    icon: ImageVector,
    status: String,
    statusKey: String,
    lines: List<String>,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    extra: @Composable () -> Unit = {},
) {
    val colors = rankoColors
    PorcelainCard(
        modifier
            .pointerHoverIcon(PointerIcon.Hand)
            .clickable(onClick = onClick),
    ) {
        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Icon(icon, contentDescription = null, tint = colors.accentPink, modifier = Modifier.size(22.dp))
            Text(
                title,
                color = colors.text,
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.weight(1f),
            )
            Text(
                status,
                color = homeStatusColor(statusKey),
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
            )
        }
        lines.forEach { line ->
            Text(
                line,
                color = colors.textDim,
                fontSize = 13.sp,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        }
        extra()
    }
}

@Composable
private fun homeStatusColor(key: String): Color {
    val colors = rankoColors
    return when (key) {
        "starting", "training", "running" -> colors.accentBlue
        "encoding" -> colors.accentPink
        "sampling" -> colors.accentLilac
        "pausing", "paused", "resuming", "gpu-out", "gpu-in" -> colors.qualityOrange
        "stopping", "error" -> colors.qualityRed
        "finished", "done" -> colors.qualityMint
        else -> colors.textDim
    }
}

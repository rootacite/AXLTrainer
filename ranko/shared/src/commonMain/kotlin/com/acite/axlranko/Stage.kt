package com.acite.axlranko

import androidx.compose.animation.*
import androidx.compose.animation.core.animateOffsetAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.ShowChart
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import chromatrix.shared.generated.resources.Res
import chromatrix.shared.generated.resources.app_icon
import com.acite.axlranko.pages.AutomationScreen
import com.acite.axlranko.pages.AutomationScreenViewModel
import com.acite.axlranko.pages.DashboardScreen
import com.acite.axlranko.pages.DashboardScreenViewModel
import com.acite.axlranko.pages.HomeRoute
import com.acite.axlranko.pages.ImageScreenViewModel
import com.acite.axlranko.pages.ImagesScreen
import com.acite.axlranko.pages.StatisticsScreen
import com.acite.axlranko.pages.StatisticsScreenViewModel
import com.acite.axlranko.pages.UtilsScreen
import com.acite.axlranko.pages.UtilsScreenViewModel
import com.acite.axlranko.ui.components.FrostedSurface
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.util.NAV_IDLE_MILLIS
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlinx.coroutines.delay
import org.jetbrains.compose.resources.painterResource
import kotlin.math.roundToInt

enum class Screen {
    Home, Images, Statistics, Utils, Dashboard, Automation
}

@Composable
public fun Stage(
    viewModel: StageViewModel = metroViewModel(),
    ssViewModel: StatisticsScreenViewModel = metroViewModel(),
    imViewModel: ImageScreenViewModel = metroViewModel(),
    usViewModel: UtilsScreenViewModel = metroViewModel(),
    dsViewModel: DashboardScreenViewModel = metroViewModel(),
    auViewModel: AutomationScreenViewModel = metroViewModel(),
)
{
    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .safeContentPadding()
    )
    {
        val bounds = Size(constraints.maxWidth.toFloat(), constraints.maxHeight.toFloat())
        val open: (Screen) -> Unit = { screen ->
            viewModel.currentScreen = screen
            when (screen) {
                Screen.Home -> dsViewModel.onLeave()
                Screen.Images -> {
                    dsViewModel.onLeave()
                    imViewModel.reloadFromDiskSafely()
                }
                Screen.Statistics -> {
                    dsViewModel.onLeave()
                    ssViewModel.scanDataset()
                }
                Screen.Utils -> {
                    dsViewModel.onLeave()
                    usViewModel.reloadFromDiskSafely()
                }
                Screen.Dashboard -> dsViewModel.onEnter()
                Screen.Automation -> {
                    dsViewModel.onLeave()
                    auViewModel.onEnter()
                }
            }
        }

        AnimatedContent(
            targetState = viewModel.currentScreen,
            transitionSpec = {
                slideIntoContainer(
                    towards = AnimatedContentTransitionScope.SlideDirection.Left,
                    animationSpec = tween(300)
                ) + fadeIn(tween(300)) togetherWith slideOutOfContainer(
                    towards = AnimatedContentTransitionScope.SlideDirection.Left,
                    animationSpec = tween(300)
                ) + fadeOut(tween(300))
            },
            label = "Screen Transition",
            modifier = Modifier.fillMaxSize()
        ) { targetScreen ->
            when (targetScreen) {
                Screen.Home -> HomeRoute(
                    onOpen = { open(it) },
                    onOpenRun = { runId ->
                        dsViewModel.selectRun(runId)
                        open(Screen.Dashboard)
                    },
                )
                Screen.Images -> ImagesScreen()
                Screen.Statistics -> StatisticsScreen()
                Screen.Utils -> UtilsScreen()
                Screen.Dashboard -> DashboardScreen(viewModel = dsViewModel)
                Screen.Automation -> AutomationScreen(viewModel = auViewModel)
            }
        }

        FloatingNavRail(
            viewModel = viewModel,
            bounds = bounds,
            onHome = { open(Screen.Home) },
            onImages = { open(Screen.Images) },
            onStatistics = { open(Screen.Statistics) },
            onUtils = { open(Screen.Utils) },
            onDashboard = { open(Screen.Dashboard) },
            onAutomation = { open(Screen.Automation) },
        )
    }
}

@Composable
private fun FloatingNavRail(
    viewModel: StageViewModel,
    bounds: Size,
    onHome: () -> Unit,
    onImages: () -> Unit,
    onStatistics: () -> Unit,
    onUtils: () -> Unit,
    onDashboard: () -> Unit,
    onAutomation: () -> Unit,
) {
    val density = LocalDensity.current
    val minPeekPx = with(density) { 24.dp.toPx() }
    var navSize by remember { mutableStateOf(Size.Zero) }
    val interactionSource = remember { MutableInteractionSource() }
    val hovered by interactionSource.collectIsHoveredAsState()
    val expanded = viewModel.navExpanded

    LaunchedEffect(bounds, minPeekPx) {
        viewModel.layoutNav(navSize, bounds, minPeekPx)
    }

    LaunchedEffect(hovered, viewModel.navDragging, expanded) {
        if (viewModel.navDragging) return@LaunchedEffect
        if (hovered) {
            if (viewModel.navPeeking) {
                viewModel.unpeek()
                viewModel.layoutNav(navSize, bounds, minPeekPx)
            }
            return@LaunchedEffect
        }
        delay(NAV_IDLE_MILLIS)
        val wasExpanded = viewModel.navExpanded
        viewModel.collapseAndPeek()
        if (!wasExpanded) viewModel.layoutNav(navSize, bounds, minPeekPx)
    }

    val displayedOffset by animateOffsetAsState(
        targetValue = viewModel.navOffset,
        animationSpec = tween(durationMillis = if (viewModel.navDragging) 0 else 220),
        label = "navOffset",
    )
    val offset = if (viewModel.navDragging) viewModel.navOffset else displayedOffset

    FrostedSurface(
        modifier = Modifier
            .offset { IntOffset(offset.x.roundToInt(), offset.y.roundToInt()) }
            .then(if (!expanded) Modifier.clip(CircleShape) else Modifier)
            .alpha(if (viewModel.navPeeking && !hovered) 0.55f else 1f)
            .hoverable(interactionSource)
            .onSizeChanged { size ->
                val measured = Size(size.width.toFloat(), size.height.toFloat())
                navSize = measured
                viewModel.layoutNav(measured, bounds, minPeekPx)
            }
            .then(
                if (!expanded) {
                    Modifier.clickable(
                        interactionSource = interactionSource,
                        indication = null,
                    ) { viewModel.expandNav() }
                } else {
                    Modifier
                },
            )
            .pointerInput(bounds, navSize) {
                detectDragGestures(
                    onDragStart = {
                        viewModel.navDragging = true
                        viewModel.navPeeking = false
                    },
                    onDragEnd = { viewModel.endNavDrag(navSize, bounds) },
                    onDragCancel = { viewModel.endNavDrag(navSize, bounds) },
                ) { change, dragAmount ->
                    change.consume()
                    viewModel.dragNavBy(dragAmount, navSize, bounds)
                }
            },
    ) {
        if (expanded) {
            Column(
                modifier = Modifier.padding(vertical = 12.dp, horizontal = 8.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Image(
                    painter = painterResource(Res.drawable.app_icon),
                    contentDescription = "Chromatrix",
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape),
                )

                StageNavButton(
                    icon = Icons.Default.Home,
                    description = "Home",
                    selected = viewModel.currentScreen == Screen.Home,
                    onClick = onHome,
                )
                StageNavButton(
                    icon = Icons.Default.Image,
                    description = "Images",
                    selected = viewModel.currentScreen == Screen.Images,
                    onClick = onImages,
                )
                StageNavButton(
                    icon = Icons.Default.Analytics,
                    description = "Statistics",
                    selected = viewModel.currentScreen == Screen.Statistics,
                    onClick = onStatistics,
                )
                StageNavButton(
                    icon = Icons.Default.Build,
                    description = "Utils",
                    selected = viewModel.currentScreen == Screen.Utils,
                    onClick = onUtils,
                )
                StageNavButton(
                    icon = Icons.Default.ShowChart,
                    description = "Dashboard",
                    selected = viewModel.currentScreen == Screen.Dashboard,
                    onClick = onDashboard,
                )
                StageNavButton(
                    icon = Icons.Default.AutoAwesome,
                    description = "Automation",
                    selected = viewModel.currentScreen == Screen.Automation,
                    onClick = onAutomation,
                )
            }
        } else {
            Image(
                painter = painterResource(Res.drawable.app_icon),
                contentDescription = "Open navigation",
                modifier = Modifier
                    .padding(10.dp)
                    .size(36.dp)
                    .clip(CircleShape),
            )
        }
    }
}

@Composable
private fun StageNavButton(
    icon: ImageVector,
    description: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    val colors = rankoColors
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        IconButton(onClick = onClick) {
            Icon(
                imageVector = icon,
                contentDescription = description,
                tint = if (selected) colors.accentPink else colors.textDim,
            )
        }
        Box(
            Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(if (selected) colors.accentPink else colors.stroke),
        )
    }
}

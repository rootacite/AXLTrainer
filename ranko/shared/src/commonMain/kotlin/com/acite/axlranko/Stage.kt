package com.acite.axlranko

import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.ShowChart
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import axlranko.shared.generated.resources.Res
import axlranko.shared.generated.resources.app_icon
import org.jetbrains.compose.resources.painterResource
import com.acite.axlranko.pages.DashboardScreen
import com.acite.axlranko.pages.DashboardScreenViewModel
import com.acite.axlranko.pages.ImageScreenViewModel
import com.acite.axlranko.pages.ImagesScreen
import com.acite.axlranko.pages.StatisticsScreen
import com.acite.axlranko.pages.StatisticsScreenViewModel
import com.acite.axlranko.pages.UtilsScreen
import com.acite.axlranko.pages.UtilsScreenViewModel
import com.acite.axlranko.ui.components.FrostedSurface
import com.acite.axlranko.ui.theme.rankoColors
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlin.math.roundToInt

enum class Screen {
    Images, Statistics, Utils, Dashboard
}

@Composable
public fun Stage(
    viewModel: StageViewModel = metroViewModel(),
    ssViewModel: StatisticsScreenViewModel = metroViewModel(),
    imViewModel: ImageScreenViewModel = metroViewModel(),
    usViewModel: UtilsScreenViewModel = metroViewModel(),
    dsViewModel: DashboardScreenViewModel = metroViewModel(),
)
{
    Box(
        modifier = Modifier
            .fillMaxSize()
            .safeContentPadding()
    )
    {
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
                Screen.Images -> ImagesScreen()
                Screen.Statistics -> StatisticsScreen()
                Screen.Utils -> UtilsScreen()
                Screen.Dashboard -> DashboardScreen(viewModel = dsViewModel)
            }
        }

        FrostedSurface(
            modifier = Modifier
                .offset { IntOffset(viewModel.navOffset.x.roundToInt(), viewModel.navOffset.y.roundToInt()) }
                .pointerInput(Unit) {
                    detectDragGestures { change, dragAmount ->
                        change.consume()
                        viewModel.navOffset += dragAmount
                    }
                },
        ) {
            Column(
                modifier = Modifier.padding(vertical = 12.dp, horizontal = 8.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Image(
                    painter = painterResource(Res.drawable.app_icon),
                    contentDescription = "AxlRanko",
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape),
                )

                StageNavButton(
                    icon = Icons.Default.Image,
                    description = "Images",
                    selected = viewModel.currentScreen == Screen.Images,
                    onClick = {
                        viewModel.currentScreen = Screen.Images
                        imViewModel.reloadFromDiskSafely()
                    },
                )
                StageNavButton(
                    icon = Icons.Default.Analytics,
                    description = "Statistics",
                    selected = viewModel.currentScreen == Screen.Statistics,
                    onClick = {
                        viewModel.currentScreen = Screen.Statistics
                        ssViewModel.scanDataset()
                    },
                )
                StageNavButton(
                    icon = Icons.Default.Build,
                    description = "Utils",
                    selected = viewModel.currentScreen == Screen.Utils,
                    onClick = {
                        viewModel.currentScreen = Screen.Utils
                        usViewModel.reloadFromDiskSafely()
                    },
                )
                StageNavButton(
                    icon = Icons.Default.ShowChart,
                    description = "Dashboard",
                    selected = viewModel.currentScreen == Screen.Dashboard,
                    onClick = {
                        viewModel.currentScreen = Screen.Dashboard
                        dsViewModel.onEnter()
                    },
                )
            }
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
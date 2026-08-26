package com.acite.axlranko

import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import com.acite.axlranko.pages.ImageScreenViewModel
import com.acite.axlranko.pages.ImagesScreen
import com.acite.axlranko.pages.StatisticsScreen
import com.acite.axlranko.pages.StatisticsScreenViewModel
import com.acite.axlranko.pages.UtilsScreen
import com.acite.axlranko.pages.UtilsScreenViewModel
import dev.zacsweers.metrox.viewmodel.metroViewModel
import kotlin.math.roundToInt

enum class Screen {
    Images, Statistics, Utils
}

@Composable
public fun Stage(
    viewModel: StageViewModel = metroViewModel(),
    ssViewModel: StatisticsScreenViewModel = metroViewModel(),
    imViewModel: ImageScreenViewModel = metroViewModel(),
    usViewModel: UtilsScreenViewModel = metroViewModel(),
)
{
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.primaryContainer)
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
            }
        }

        Surface(
            modifier = Modifier
                .offset { IntOffset(viewModel.navOffset.x.roundToInt(), viewModel.navOffset.y.roundToInt()) }
                .pointerInput(Unit) {
                    detectDragGestures { change, dragAmount ->
                        change.consume()
                        viewModel.navOffset += dragAmount
                    }
                },
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.surfaceVariant,
            shadowElevation = 8.dp
        ) {
            Column(
                modifier = Modifier.padding(vertical = 12.dp, horizontal = 8.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                IconButton(onClick = {
                    viewModel.currentScreen = Screen.Images
                    imViewModel.reloadFromDiskSafely()
                }) {
                    Icon(
                        imageVector = Icons.Default.Image,
                        contentDescription = "Images",
                        tint = if (viewModel.currentScreen == Screen.Images) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                IconButton(onClick = {
                    viewModel.currentScreen = Screen.Statistics
                    ssViewModel.scanDataset()
                }) {
                    Icon(
                        imageVector = Icons.Default.Analytics,
                        contentDescription = "Statistics",
                        tint = if (viewModel.currentScreen == Screen.Statistics) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                IconButton(onClick = {
                    viewModel.currentScreen = Screen.Utils
                    usViewModel.reloadFromDiskSafely()
                }) {
                    Icon(
                        imageVector = Icons.Default.Build,
                        contentDescription = "Utils",
                        tint = if (viewModel.currentScreen == Screen.Utils) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}
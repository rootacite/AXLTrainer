package org.acitelight.axledge

import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import axledge.sharedui.generated.resources.*
import dashBoard.DashBoard
import dev.zacsweers.metrox.viewmodel.LocalMetroViewModelFactory
import dev.zacsweers.metrox.viewmodel.MetroViewModelFactory
import generateBoard.GenerateBoard
import org.acitelight.axledge.theme.AppTheme
import org.acitelight.axledge.theme.LocalThemeIsDark
import kotlinx.coroutines.isActive
import org.jetbrains.compose.resources.Font
import org.jetbrains.compose.resources.stringResource
import org.jetbrains.compose.resources.vectorResource

enum class AppTab {
    Dashboard, Generator, Library
}

@Composable
fun App(
    metroVmf: MetroViewModelFactory,
    onThemeChanged: @Composable (isDark: Boolean) -> Unit = {}
) = AppTheme(onThemeChanged) {
    CompositionLocalProvider(LocalMetroViewModelFactory provides metroVmf) {
        var currentTab by remember { mutableStateOf(AppTab.Dashboard) }
        var isNavExpanded by remember { mutableStateOf(false) }

        val navWidth by animateDpAsState(
            targetValue = if (isNavExpanded) 240.dp else 72.dp,
            animationSpec = spring(stiffness = Spring.StiffnessLow)
        )

        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing)
        ) {
            val isLandscape = maxWidth > maxHeight

            if (isLandscape) {
                Row(modifier = Modifier.fillMaxSize()) {
                    Column(
                        modifier = Modifier
                            .width(navWidth)
                            .fillMaxHeight()
                            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                            .padding(vertical = 16.dp, horizontal = 8.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        IconButton(
                            onClick = { isNavExpanded = !isNavExpanded },
                            modifier = Modifier.align(if (isNavExpanded) Alignment.End else Alignment.CenterHorizontally)
                        ) {
                            Icon(
                                imageVector = vectorResource(Res.drawable.ic_rotate_right),
                                contentDescription = "Toggle Navigation"
                            )
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        AppTab.entries.forEach { tab ->
                            NavigationItem(
                                tab = tab,
                                isSelected = currentTab == tab,
                                isExpanded = isNavExpanded,
                                isLandscape = true,
                                onClick = { currentTab = tab }
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                    }
                    VerticalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxHeight()
                            .padding(4.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        MainContent(currentTab)
                    }
                }
            }
            else
            {
                Column(modifier = Modifier.fillMaxSize()) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceAround
                    ) {
                        AppTab.entries.forEach { tab ->
                            Box(modifier = Modifier.weight(1f).padding(horizontal = 4.dp).height(40.dp)) {
                                NavigationItem(
                                    tab = tab,
                                    isSelected = currentTab == tab,
                                    isExpanded = false,
                                    isLandscape = false,
                                    onClick = { currentTab = tab },
                                )
                            }
                        }
                    }
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f)
                            .padding(4.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        MainContent(currentTab)
                    }
                }
            }
        }
    }
}

@Composable
fun MainContent(currentTab: AppTab) {
    when (currentTab) {
        AppTab.Dashboard -> DashBoard()
        AppTab.Generator -> GenerateBoard()
        AppTab.Library -> LibraryScreen()
    }
}

@Composable
fun NavigationItem(
    tab: AppTab,
    isSelected: Boolean,
    isExpanded: Boolean,
    isLandscape: Boolean,
    onClick: () -> Unit
) {
    val containerColor = if (isSelected) MaterialTheme.colorScheme.primaryContainer else androidx.compose.ui.graphics.Color.Transparent
    val contentColor = if (isSelected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant

    Surface(
        selected = isSelected,
        onClick = onClick,
        shape = MaterialTheme.shapes.medium,
        color = containerColor,
        contentColor = contentColor,
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
                .animateContentSize(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = if (isLandscape && isExpanded) Arrangement.Start else Arrangement.Center
        ) {
            val iconRes = when (tab) {
                AppTab.Dashboard -> Res.drawable.ic_cyclone
                AppTab.Generator -> Res.drawable.ic_rotate_right
                AppTab.Library -> Res.drawable.ic_dark_mode
            }

            Icon(
                imageVector = vectorResource(iconRes),
                contentDescription = tab.name,
                modifier = Modifier.size(24.dp)
            )

            if (isLandscape && isExpanded) {
                Spacer(modifier = Modifier.width(16.dp))
                Text(
                    text = tab.name,
                    style = MaterialTheme.typography.labelLarge,
                    maxLines = 1
                )
            }
        }
    }
}


@Composable
fun LibraryScreen() {
    val uriHandler = LocalUriHandler.current
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Library Screen", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        TextButton(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp).widthIn(min = 200.dp),
            onClick = { uriHandler.openUri("https://github.com/terrakok") },
        ) {
            Text(stringResource(Res.string.open_github))
        }
    }
}
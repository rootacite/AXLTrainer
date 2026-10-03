package com.acite.axlranko.pages

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.model.AutomationSection
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.pages.components.automation.ComfyPane
import com.acite.axlranko.pages.components.automation.GalleryPane
import com.acite.axlranko.pages.components.automation.PromptsPane
import com.acite.axlranko.pages.components.automation.uiText
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.ui.SingleLineOrStacked
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.isPortrait
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.zacsweers.metrox.viewmodel.metroViewModel

private fun sectionDescription(section: AutomationSection, lang: PromptLang): String = when (section) {
    AutomationSection.Prompts -> uiText(lang, "section_prompts")
    AutomationSection.ComfyUi -> uiText(lang, "section_comfy")
    AutomationSection.Gallery -> uiText(lang, "section_gallery")
}

/**
 * The Automation page: prompts, the ComfyUI batch run, and the gallery of what came out.
 *
 * The Prompts area is live; the other two sections are placeholders until their own steps land.
 */@Composable
public fun AutomationScreen(viewModel: AutomationScreenViewModel = metroViewModel()) {
    val uiState by viewModel.uiState.collectAsState()
    val colors = rankoColors

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val portrait = isPortrait(maxWidth, maxHeight)
        Column(modifier = Modifier.fillMaxSize()) {
            AutomationHeader(uiState, viewModel)
            HorizontalDivider(color = colors.stroke.copy(alpha = 0.55f))

            if (portrait) {
                AutomationSectionTabs(
                    selected = uiState.section,
                    onSelect = viewModel::selectSection,
                )
                Text(
                    text = sectionDescription(uiState.section, uiState.language),
                    color = colors.textDim,
                    fontSize = 11.sp,
                    maxLines = 1,
                    softWrap = false,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
                )
                HorizontalDivider(color = colors.stroke.copy(alpha = 0.55f))
                AutomationBody(uiState, viewModel, portrait = true, modifier = Modifier.weight(1f))
            } else {
                Row(modifier = Modifier.fillMaxSize()) {
                    Column(
                        modifier = Modifier
                            .fillMaxHeight()
                            .width(220.dp)
                            .background(colors.bgPanel.copy(alpha = 0.45f))
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        AutomationSection.entries.forEach { section ->
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(rankoTokens.panel)
                                    .background(
                                        if (uiState.section == section) colors.accentPink.copy(alpha = 0.16f)
                                        else Color.Transparent,
                                    )
                                    .padding(10.dp),
                                verticalArrangement = Arrangement.spacedBy(2.dp),
                            ) {
                                CapsuleChoice(
                                    text = section.title,
                                    selected = uiState.section == section,
                                    onClick = { viewModel.selectSection(section) },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                                Text(
                                    text = sectionDescription(section, uiState.language),
                                    color = colors.textDim,
                                    fontSize = 10.sp,
                                )
                            }
                        }
                    }

                    HorizontalDivider(
                        modifier = Modifier.fillMaxHeight().width(1.dp),
                        color = colors.stroke.copy(alpha = 0.55f),
                    )

                    AutomationBody(uiState, viewModel, portrait = false, modifier = Modifier.fillMaxSize())
                }
            }
        }
    }
}

@Composable
private fun AutomationBody(
    uiState: AutomationUiState,
    viewModel: AutomationScreenViewModel,
    portrait: Boolean,
    modifier: Modifier,
) {
    Box(modifier = modifier) {
        when (uiState.section) {
            AutomationSection.Prompts -> PromptsPane(
                state = uiState,
                viewModel = viewModel,
                warning = viewModel.currentWarning(),
                portrait = portrait,
            )

            AutomationSection.ComfyUi -> ComfyPane(state = uiState, viewModel = viewModel, portrait = portrait)

            AutomationSection.Gallery -> GalleryPane(state = uiState, viewModel = viewModel, portrait = portrait)
        }
        if (uiState.loadingMatrix && uiState.matrix == null) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.TopCenter).padding(8.dp))
        }
    }
}

/** Portrait section rail: one horizontal line of titles, no wrapping. */
@Composable
internal fun AutomationSectionTabs(
    selected: AutomationSection,
    onSelect: (AutomationSection) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        AutomationSection.entries.forEach { section ->
            CapsuleChoice(
                text = section.title,
                selected = selected == section,
                onClick = { onSelect(section) },
            )
        }
    }
}

/** Page header: the wizard's text language belongs to the whole page, not to one section. */
@Composable
internal fun AutomationHeader(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    SingleLineOrStacked(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp),
        spacing = 10.dp,
        first = {
            Text(
                text = "Automation",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = colors.text,
                maxLines = 1,
                softWrap = false,
                overflow = TextOverflow.Ellipsis,
            )
        },
        second = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = uiText(state.language, "language"),
                    color = colors.textDim,
                    fontSize = 11.sp,
                    maxLines = 1,
                    softWrap = false,
                    overflow = TextOverflow.Ellipsis,
                )
                CapsuleChoice(
                    text = "中文",  // the language chips name themselves in their own language
                    selected = state.language == PromptLang.Chinese,
                    onClick = { viewModel.setLanguage(PromptLang.Chinese) },
                )
                CapsuleChoice(
                    text = "EN",
                    selected = state.language == PromptLang.English,
                    onClick = { viewModel.setLanguage(PromptLang.English) },
                )
            }
        },
    )
}

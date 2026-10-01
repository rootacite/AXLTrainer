package com.acite.axlranko.pages.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.PromptTagCount
import com.acite.axlranko.ui.theme.RankoPalette
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import com.acite.axlranko.util.formatFixed

/**
 * The frequency ramp both tag lists share: cool for rare, hot for everywhere. Statistics draws its
 * dataset bars with it, and the evaluation panel draws the prompt-tag picks with it, so a 75 %
 * tag looks the same on either page.
 */
internal fun frequencyColor(frequency: Float, colors: RankoPalette): Color {
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

/**
 * A compact picker over the tags the run's prompts ask for: one row per tag, its frequency as a
 * bar behind it, and a click to add or remove it from the selection. `selected` empty means "every
 * tag" — the caller says so under the list rather than here.
 */
@Composable
internal fun TagFrequencyList(
    tags: List<PromptTagCount>,
    selected: Set<String>,
    onToggle: (String) -> Unit,
    modifier: Modifier = Modifier,
    maxHeight: Int = 200,
) {
    val colors = rankoColors
    val tokens = rankoTokens
    if (tags.isEmpty()) return
    Column(
        modifier = modifier.fillMaxWidth().heightIn(max = maxHeight.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        tags.forEach { row ->
            val isSelected = row.tag in selected
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(28.dp)
                    .clip(tokens.panel)
                    .background(if (isSelected) colors.accentPink.copy(alpha = 0.16f) else Color.Transparent)
                    .clickable { onToggle(row.tag) },
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxHeight()
                        .fillMaxWidth(fraction = (row.frequency / 100f).coerceIn(0.02f, 1f))
                        .background(frequencyColor(row.frequency, colors).copy(alpha = 0.55f)),
                )
                Row(
                    modifier = Modifier.fillMaxSize().padding(horizontal = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = row.tag,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                        color = if (isSelected) colors.accentPink else colors.text,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        text = "${row.count} (${formatFixed(row.frequency, 0)}%)",
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.textDim,
                    )
                }
            }
        }
    }
}

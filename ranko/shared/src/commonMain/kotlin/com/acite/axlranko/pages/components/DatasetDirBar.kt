package com.acite.axlranko.pages.components

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.acite.axlranko.ui.components.CapsuleChoice

/**
 * Picks one `[[environment.train_data]]` folder. A config with a single folder draws nothing: the
 * pages that use it then behave exactly as they did before the list existed.
 */
@Composable
fun DatasetDirBar(
    labels: List<String>,
    selected: Int,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    if (labels.size < 2) return
    Row(
        modifier = modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        labels.forEachIndexed { index, label ->
            CapsuleChoice(
                text = label,
                selected = index == selected,
                onClick = { onSelect(index) }
            )
        }
    }
}

/** `folder ×3`, the last path segment plus the repeat, which is what a chip has room for. */
fun datasetDirLabel(path: String, repeat: Int): String = "${datasetDirName(path)} ×$repeat"

/** The same label while the repeat is still text the user is typing (Utils enum form). */
fun datasetDirLabel(path: String, repeat: String): String =
    "${datasetDirName(path)} ×${repeat.trim().ifBlank { "1" }}"

private fun datasetDirName(path: String): String {
    val trimmed = path.trim().trimEnd('/')
    return trimmed.substringAfterLast('/').ifEmpty { trimmed.ifEmpty { "(no folder)" } }
}

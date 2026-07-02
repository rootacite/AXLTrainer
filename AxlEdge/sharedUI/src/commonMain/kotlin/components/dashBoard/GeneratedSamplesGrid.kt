package components.dashBoard

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.decodeToImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dashBoard.DashBoardViewModel
import kotlin.collections.component1
import kotlin.collections.component2

@Composable
fun GeneratedSamplesGrid(viewModel: DashBoardViewModel) {
    val groupedImages = remember(viewModel.samplesData) {
        viewModel.samplesData.entries
            .sortedBy { it.key.toIntOrNull() ?: Int.MAX_VALUE }
    }

    if (groupedImages.isEmpty()) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
            shape = RoundedCornerShape(16.dp)
        ) {
            Text(
                "No sample images generated yet.",
                modifier = Modifier.padding(20.dp),
                style = MaterialTheme.typography.bodyMedium
            )
        }
        return
    }

    LazyColumn(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        items(
            items = groupedImages,
            key = { it.key }
        ) { (stepStr, samples) ->
            val stepLabel = if (stepStr == "-1") "Other / Unknown Step" else "Step $stepStr"

            Column {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(bottom = 10.dp)
                ) {
                    Text(
                        text = stepLabel,
                        fontWeight = FontWeight.SemiBold,
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "(${samples.size} images)",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    items(
                        items = samples,
                        key = { it.filename }
                    ) { sample ->
                        AsyncSampleImageCard(sample.filename, viewModel)
                    }
                }
            }
        }
    }
}

@Composable
fun AsyncSampleImageCard(filename: String, viewModel: DashBoardViewModel) {
    var imageBitmap by remember { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }

    LaunchedEffect(filename) {
        val bytes = viewModel.fetchSampleImage(filename)
        if (bytes != null) {
            try {
                imageBitmap = bytes.decodeToImageBitmap()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f))
            .padding(bottom = 8.dp)
    ) {
        if (imageBitmap != null) {
            androidx.compose.foundation.Image(
                bitmap = imageBitmap!!,
                contentDescription = filename,
                modifier = Modifier
                    .height(160.dp)
                    .clip(RoundedCornerShape(topStart = 12.dp, topEnd = 12.dp))
            )
        } else {
            Box(
                modifier = Modifier
                    .height(160.dp)
                    .width(160.dp)
                    .clip(RoundedCornerShape(topStart = 12.dp, topEnd = 12.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 2.dp)
            }
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = filename,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 8.dp)
        )
    }
}
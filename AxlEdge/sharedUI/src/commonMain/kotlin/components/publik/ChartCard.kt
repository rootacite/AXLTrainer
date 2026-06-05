package components.publik

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive


@Composable
fun ChartCard(title: String, dataArray: JsonArray?, color: Color, smoothing: Float, modifier: Modifier = Modifier) {
    ElevatedCard(
        modifier = modifier.height(220.dp),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp).fillMaxSize()) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .clip(RoundedCornerShape(5.dp))
                        .background(color)
                )
                Text(title, fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.titleSmall)
            }
            Spacer(modifier = Modifier.height(16.dp))
            if (!dataArray.isNullOrEmpty()) {
                LineChart(dataArray, color, smoothing, modifier = Modifier.fillMaxSize())
            } else {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No Data", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

@Composable
private fun LineChart(dataArray: JsonArray, color: Color, smoothing: Float, modifier: Modifier = Modifier) {
    val points = remember(dataArray) {
        dataArray.mapNotNull {
            val obj = it.jsonObject
            val step = obj["step"]?.jsonPrimitive?.floatOrNull
            val value = obj["value"]?.jsonPrimitive?.floatOrNull
            if (step != null && value != null) Offset(step, value) else null
        }
    }

    if (points.isEmpty()) return

    val smoothedPoints = remember(points, smoothing) {
        if (smoothing <= 0f) return@remember points
        val result = mutableListOf<Offset>()
        var lastVal = points.first().y
        for (p in points) {
            lastVal = lastVal * smoothing + (1 - smoothing) * p.y
            result.add(Offset(p.x, lastVal))
        }
        result
    }

    val gridColor = Color.Gray.copy(alpha = 0.12f)

    Canvas(modifier = modifier) {
        val minX = points.minOf { it.x }
        val maxX = points.maxOf { it.x }
        val minY = points.minOf { it.y }
        val maxY = points.maxOf { it.y }

        val rangeX = if (maxX > minX) maxX - minX else 1f
        val rangeY = if (maxY > minY) maxY - minY else 1f

        fun Offset.toScreen() = Offset(
            x = ((this.x - minX) / rangeX) * size.width,
            y = size.height - ((this.y - minY) / rangeY) * size.height
        )

        // Draw subtle grid lines
        val gridLines = 4
        for (i in 0..gridLines) {
            val y = size.height * i / gridLines
            drawLine(color = gridColor, start = Offset(0f, y), end = Offset(size.width, y), strokeWidth = 1f)
        }

        // Draw Raw Line (low opacity)
        val rawPath = Path().apply {
            points.forEachIndexed { index, point ->
                val screenPt = point.toScreen()
                if (index == 0) moveTo(screenPt.x, screenPt.y) else lineTo(screenPt.x, screenPt.y)
            }
        }
        drawPath(path = rawPath, color = color.copy(alpha = 0.2f), style = Stroke(width = 1.5f))

        // Draw Smoothed Line
        val smoothPath = Path().apply {
            smoothedPoints.forEachIndexed { index, point ->
                val screenPt = point.toScreen()
                if (index == 0) moveTo(screenPt.x, screenPt.y) else lineTo(screenPt.x, screenPt.y)
            }
        }
        drawPath(path = smoothPath, color = color, style = Stroke(width = 3f))

        // Draw endpoint dot
        smoothedPoints.lastOrNull()?.toScreen()?.let { last ->
            drawCircle(color = color, radius = 5f, center = last)
            drawCircle(color = Color.White, radius = 2.5f, center = last)
        }
    }
}

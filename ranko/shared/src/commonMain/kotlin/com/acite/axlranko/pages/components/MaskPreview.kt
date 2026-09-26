package com.acite.axlranko.pages.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.acite.axlranko.util.brushRadii
import kotlin.math.min
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

@Composable
fun MaskPreview(
    bitmap: ImageBitmap?,
    sourceWidth: Int,
    sourceHeight: Int,
    editing: Boolean,
    brushRadius: Float,
    brushFeather: Float,
    onStrokeStart: (imageX: Float, imageY: Float, erase: Boolean) -> Unit,
    onStrokeMove: (imageX: Float, imageY: Float) -> Unit,
    onStrokeLeaveImage: () -> Unit,
    onStrokeEnd: () -> Unit,
    onBrushResize: (steps: Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    var boxSize by remember { mutableStateOf(IntSize.Zero) }
    var cursor by remember { mutableStateOf<Offset?>(null) }
    val lastCursor = remember { arrayOf(TimeSource.Monotonic.markNow()) }

    Box(
        modifier = modifier
            .fillMaxSize()
            .onSizeChanged { boxSize = it }
            .maskPaintInput(
                enabled = editing && bitmap != null && sourceWidth > 0 && sourceHeight > 0,
                onStrokeStart = { x, y, erase ->
                    mapToSource(
                        Offset(x, y),
                        boxSize.width.toFloat(),
                        boxSize.height.toFloat(),
                        sourceWidth,
                        sourceHeight,
                    )?.let { onStrokeStart(it.first, it.second, erase) }
                },
                onStrokeMove = { x, y ->
                    val mapped = mapToSource(
                        Offset(x, y),
                        boxSize.width.toFloat(),
                        boxSize.height.toFloat(),
                        sourceWidth,
                        sourceHeight,
                    )
                    // Outside the image (letterbox or beyond the window): paint nothing.
                    // Clamping here would smear the brush along the border.
                    if (mapped != null) onStrokeMove(mapped.first, mapped.second)
                    else onStrokeLeaveImage()
                },
                onStrokeEnd = onStrokeEnd,
                onPointerMoved = { x, y ->
                    val inside = mapToSource(
                        Offset(x, y),
                        boxSize.width.toFloat(),
                        boxSize.height.toFloat(),
                        sourceWidth,
                        sourceHeight,
                    ) != null
                    when {
                        !inside -> if (cursor != null) cursor = null
                        lastCursor[0].elapsedNow() >= 16.milliseconds -> {
                            lastCursor[0] = TimeSource.Monotonic.markNow()
                            cursor = Offset(x, y)
                        }
                    }
                },
                onBrushResize = onBrushResize,
            ),
        contentAlignment = Alignment.Center,
    ) {
        if (bitmap == null || sourceWidth <= 0 || sourceHeight <= 0) {
            return@Box
        }
        Image(
            bitmap = bitmap,
            contentDescription = null,
            contentScale = ContentScale.Fit,
            filterQuality = FilterQuality.High,
            modifier = Modifier.fillMaxSize(),
        )
        val hint = cursor
        if (editing && hint != null) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val scale = min(size.width / sourceWidth, size.height / sourceHeight)
                val radii = brushRadii(brushRadius, brushFeather)
                val dash = PathEffect.dashPathEffect(floatArrayOf(5.dp.toPx(), 4.dp.toPx()))
                drawBrushCircle(hint, radii.outer * scale, dash)
                drawBrushCircle(hint, radii.inner * scale, null)
            }
        }
    }
}

/**
 * A dark halo under a light line, so the brush cursor stays visible on both
 * bright and dark artwork.
 */
private fun DrawScope.drawBrushCircle(center: Offset, radius: Float, pathEffect: PathEffect?) {
    if (radius <= 0.5f) return
    val halo = Stroke(width = 3f, pathEffect = pathEffect)
    val line = Stroke(width = 1.2f, pathEffect = pathEffect)
    drawCircle(Color.Black.copy(alpha = 0.45f), radius, center, style = halo)
    drawCircle(Color.White.copy(alpha = 0.95f), radius, center, style = line)
}

/** Fit-and-center image coordinates, or null when [pos] falls outside the drawn image. */
internal fun mapToSource(
    pos: Offset,
    boxW: Float,
    boxH: Float,
    srcW: Int,
    srcH: Int,
): Pair<Float, Float>? {
    if (boxW <= 0f || boxH <= 0f || srcW <= 0 || srcH <= 0) return null
    val scale = min(boxW / srcW, boxH / srcH)
    val dw = srcW * scale
    val dh = srcH * scale
    val left = (boxW - dw) / 2f
    val top = (boxH - dh) / 2f
    val inside = pos.x >= left && pos.y >= top && pos.x <= left + dw && pos.y <= top + dh
    if (!inside) return null
    val x = ((pos.x - left) / dw * srcW).coerceIn(0f, (srcW - 1).toFloat())
    val y = ((pos.y - top) / dh * srcH).coerceIn(0f, (srcH - 1).toFloat())
    return x to y
}

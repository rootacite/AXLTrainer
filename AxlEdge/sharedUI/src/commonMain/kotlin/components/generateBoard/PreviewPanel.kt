package components.generateBoard

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil3.compose.LocalPlatformContext
import coil3.compose.rememberAsyncImagePainter
import coil3.request.ImageRequest
import components.publik.PanelHeader

import coil3.size.Size as CoilSize

private val FluentPrimary = Color(0xFF0067C0)
private val FluentPrimaryLight = Color(0xFF60CDFF)

// ---------------------------------------------------------------------
// Top Panel - Preview with Pager
// ---------------------------------------------------------------------
@Composable
fun PreviewPanel(
    state: dataModel.GenerateUiState,
    isFullscreen: Boolean,
    onToggleLock: () -> Unit,
    onDoubleTap: () -> Unit,
    modifier: Modifier = Modifier
) {
    val padding by animateDpAsState(
        targetValue = if (isFullscreen) 0.dp else 20.dp,
        label = "paddingAnim"
    )
    val bottomPadding by animateDpAsState(
        targetValue = if (isFullscreen) 0.dp else 8.dp,
        label = "bottomPaddingAnim"
    )
    val cornerRadius by animateDpAsState(
        targetValue = if (isFullscreen) 0.dp else 12.dp,
        label = "cornerAnim"
    )

    val dynamicShape = RoundedCornerShape(cornerRadius)

    Column(
        modifier = modifier.padding(
            top = padding,
            start = padding,
            end = padding,
            bottom = bottomPadding
        )
    ) {
        // Main Image Display
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .clip(dynamicShape)
                .background(Color.Black.copy(alpha = 0.3f))
                .border(1.dp, Color.White.copy(alpha = 0.1f), dynamicShape)
                .pointerInput(Unit) {
                    detectTapGestures(
                        onDoubleTap = { onDoubleTap() }
                    )
                },
            contentAlignment = Alignment.Center
        ) {
            val pageCount = state.historyImages.size + if (state.isGenerating) 1 else 0

            if (pageCount == 0) {
                EmptyPreview()
            } else {
                val pagerState = rememberPagerState(
                    pageCount = { pageCount }
                )

                LaunchedEffect(state.historyImages.size, state.isGenerating) {
                    val targetPage = pageCount - 1
                    if (targetPage >= 0 && pagerState.currentPage != targetPage) {
                        pagerState.scrollToPage(targetPage)
                    }
                }

                HorizontalPager(
                    state = pagerState,
                    modifier = Modifier.fillMaxSize()
                ) { page ->
                    if (state.isGenerating && page == pageCount - 1) {
                        GeneratingOverlay()
                    } else {
                        ImageResult(state.historyImages[page])
                    }
                }

                // Page indicator badge
                if (pageCount > 1) {
                    Text(
                        text = "${pagerState.currentPage + 1} / $pageCount",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.White.copy(alpha = 0.8f),
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .padding(bottom = 8.dp)
                            .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Image Result
// ---------------------------------------------------------------------
@Composable
private fun ImageResult(bytes: ByteArray) {
    val context = LocalPlatformContext.current

    val painter = rememberAsyncImagePainter(
        model = remember(bytes) {
            ImageRequest.Builder(context)
                .data(bytes)
                .size(CoilSize.ORIGINAL)
                .build()
        }
    )

    Column(
        modifier = Modifier.fillMaxSize().padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Image(
            painter = painter,
            contentDescription = "Generated image",
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp)),
            contentScale = ContentScale.Fit
        )
    }
}

@Composable
private fun EmptyPreview() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Canvas(modifier = Modifier.size(64.dp)) {
            val c = Color.White.copy(alpha = 0.1f)
            val r = 8.dp.toPx()
            val cell = size.width / 2.5f
            val gap = 2.dp.toPx()
            for (row in 0..2) for (col in 0..2) {
                drawRoundRect(
                    color = c,
                    topLeft = Offset(col * (cell + gap), row * (cell + gap)),
                    size = Size(cell, cell),
                    cornerRadius = CornerRadius(r)
                )
            }
        }
        Text(
            "No image yet",
            style = MaterialTheme.typography.bodyMedium,
            color = Color.White.copy(alpha = 0.35f)
        )
    }
}

@Composable
private fun GeneratingOverlay() {
    val infiniteTransition = rememberInfiniteTransition()
    val sweep by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(1400, easing = LinearEasing))
    )
    val pulse by infiniteTransition.animateFloat(
        initialValue = 0.6f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(900), repeatMode = RepeatMode.Reverse)
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = Modifier.fillMaxSize(),
    ) {
        Spacer(Modifier.weight(1f))
        Box(contentAlignment = Alignment.Center, modifier = Modifier.size(64.dp)) {
            Canvas(modifier = Modifier.size(64.dp)) {
                drawCircle(color = FluentPrimary.copy(alpha = 0.08f * pulse), radius = size.minDimension / 2f)
                drawArc(
                    color = FluentPrimary.copy(alpha = 0.3f),
                    startAngle = sweep,
                    sweepAngle = 270f,
                    useCenter = false,
                    style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round)
                )
                drawArc(
                    color = FluentPrimaryLight,
                    startAngle = sweep + 90f,
                    sweepAngle = 90f,
                    useCenter = false,
                    style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round)
                )
            }
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(FluentPrimary.copy(alpha = pulse), CircleShape)
            )
        }

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                "Generating...",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = Color.White.copy(alpha = 0.8f)
            )
        }
        Spacer(Modifier.weight(1f))
    }
}
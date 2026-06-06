package generateBoard

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import components.generateBoard.ControlPanel
import components.generateBoard.PreviewPanel
import dev.zacsweers.metrox.viewmodel.metroViewModel

private val FluentPrimary = Color(0xFF0067C0)
private val FluentPrimaryLight = Color(0xFF60CDFF)

// ---------------------------------------------------------------------
// Root Composable
// ---------------------------------------------------------------------
@Composable
fun GenerateBoard(
    viewModel: GenerateBoardViewModel = metroViewModel()
) {
    val state by viewModel.state.collectAsState()

    AnimatedVisibility(
        visible = state.isLoadingConfig,
        enter = fadeIn(),
        exit = fadeOut(tween(400))
    ) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                FluentProgressRing(size = 56.dp, strokeWidth = 3.dp, color = FluentPrimary)
                Text(
                    "Loading configuration...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
            }
        }
    }

    if (!state.isLoadingConfig) {
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF121212))
        ) {
            val totalHeight = maxHeight
            val previewHeight = totalHeight * 0.3f

            Column(Modifier.fillMaxSize()) {
                PreviewPanel(
                    state = state,
                    onToggleLock = {  },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(previewHeight)
                )

                // Top glowing boundary line
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(2.dp)
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    FluentPrimaryLight.copy(alpha = 0.6f),
                                    Color.Transparent
                                )
                            )
                        )
                )

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .background(MaterialTheme.colorScheme.surface)
                        .verticalScroll(rememberScrollState())
                ) {
                    ControlPanel(
                        state = state,
                        viewModel = viewModel,
                        modifier = Modifier.fillMaxWidth().padding(bottom = 60.dp)
                    )
                }
            }
        }
    }
}


@Composable
private fun FluentProgressRing(
    size: Dp,
    strokeWidth: Dp,
    color: Color
) {
    val infiniteTransition = rememberInfiniteTransition()
    val sweep by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(1200, easing = LinearEasing))
    )
    Canvas(modifier = Modifier.size(size)) {
        drawArc(
            color = color.copy(alpha = 0.15f),
            startAngle = 0f,
            sweepAngle = 360f,
            useCenter = false,
            style = Stroke(width = strokeWidth.toPx(), cap = StrokeCap.Round)
        )
        drawArc(
            color = color,
            startAngle = sweep,
            sweepAngle = 240f,
            useCenter = false,
            style = Stroke(width = strokeWidth.toPx(), cap = StrokeCap.Round)
        )
    }
}

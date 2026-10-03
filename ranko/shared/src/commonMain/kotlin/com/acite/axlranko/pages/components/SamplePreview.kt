package com.acite.axlranko.pages.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import kotlin.math.abs
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.ui.pointerIconHand
import com.acite.axlranko.ui.theme.rankoColors

/** One image in the full-screen preview: what it is called, and the line under the title. */
data class PreviewImage(
    val path: String,
    val title: String,
    val caption: String = "",
    val maxEdge: Int = 1024,
    /** Cache revision of [path]: changes when the file behind it becomes a different image. */
    val rev: String = "",
)

/**
 * The full-screen image preview both the Dashboard and the Automation gallery use: arrow keys,
 * a drag to change image, a click outside to dismiss, and room for caller-specific buttons.
 *
 * In portrait the side arrows are left off. A click on the left half of the picture goes back,
 * the right half goes forward, and a horizontal drag still changes the image.
 *
 * Place it where it can fill a window-sized box — a sibling of a scrolling container, not a child
 * of one: inside a `verticalScroll` it is measured with an unbounded height, the weighted image
 * area collapses to zero and all that is left is the header row.
 */
@Composable
fun ImagePreviewOverlay(
    images: List<PreviewImage>,
    index: Int,
    onClose: () -> Unit,
    onPrev: () -> Unit,
    onNext: () -> Unit,
    modifier: Modifier = Modifier,
    portrait: Boolean = false,
    actions: @Composable RowScope.() -> Unit = {},
) {
    if (images.isEmpty()) return
    val current = images[index.coerceIn(images.indices)]
    val focusRequester = remember { FocusRequester() }

    LaunchedEffect(index) {
        focusRequester.requestFocus()
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(rankoColors.bgApp.copy(alpha = 0.92f))
            .focusRequester(focusRequester)
            .focusable()
            .onPreviewKeyEvent { event ->
                if (event.type != KeyEventType.KeyDown) return@onPreviewKeyEvent false
                when (event.key) {
                    Key.Escape -> {
                        onClose()
                        true
                    }
                    Key.DirectionLeft -> {
                        onPrev()
                        true
                    }
                    Key.DirectionRight -> {
                        onNext()
                        true
                    }
                    else -> false
                }
            }
            .clickable(
                indication = null,
                interactionSource = remember { MutableInteractionSource() },
                onClick = onClose,
            ),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .clickable(
                    indication = null,
                    interactionSource = remember { MutableInteractionSource() },
                    onClick = {},
                ),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = current.title,
                        style = MaterialTheme.typography.titleSmall,
                        color = rankoColors.text,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    if (current.caption.isNotEmpty()) {
                        Text(
                            text = current.caption,
                            style = MaterialTheme.typography.labelSmall,
                            color = rankoColors.textDim,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
                actions()
                Text(
                    text = "${index + 1} / ${images.size}",
                    style = MaterialTheme.typography.labelLarge,
                    color = rankoColors.textDim,
                )
                IconButton(onClick = onClose) {
                    Icon(Icons.Default.Close, contentDescription = "Close", tint = rankoColors.text)
                }
            }

            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .pageByDragOrSideTap(portrait, onPrev, onNext),
                contentAlignment = Alignment.Center,
            ) {
                AsyncImage(
                    model = BlobRef(
                        current.path,
                        maxEdge = current.maxEdge,
                        quality = LocalThumbnailQuality.current,
                        rev = current.rev,
                    ),
                    contentDescription = current.title,
                    contentScale = ContentScale.Fit,
                    filterQuality = FilterQuality.High,
                    modifier = Modifier.fillMaxSize().padding(
                        horizontal = if (portrait) 0.dp else 72.dp,
                        vertical = 8.dp,
                    ),
                )

                if (!portrait) {
                    PreviewNavButton(
                        modifier = Modifier.align(Alignment.CenterStart),
                        icon = Icons.Default.ChevronLeft,
                        description = "Previous",
                        onClick = onPrev,
                    )
                    PreviewNavButton(
                        modifier = Modifier.align(Alignment.CenterEnd),
                        icon = Icons.Default.ChevronRight,
                        description = "Next",
                        onClick = onNext,
                    )
                }
            }
        }
    }
}

/**
 * Landscape keeps a horizontal drag. Portrait adds a tap on either half of the picture, and the
 * same drag: moving right goes back, moving left goes forward.
 */
private fun Modifier.pageByDragOrSideTap(
    portrait: Boolean,
    onPrev: () -> Unit,
    onNext: () -> Unit,
): Modifier = pointerInput(portrait) {
    if (!portrait) {
        var dragAccum = 0f
        detectHorizontalDragGestures(
            onDragEnd = {
                when {
                    dragAccum > 80f -> onPrev()
                    dragAccum < -80f -> onNext()
                }
                dragAccum = 0f
            },
            onDragCancel = { dragAccum = 0f },
            onHorizontalDrag = { change, amount ->
                change.consume()
                dragAccum += amount
            },
        )
        return@pointerInput
    }
    val slop = viewConfiguration.touchSlop
    awaitEachGesture {
        val down = awaitFirstDown()
        val startX = down.position.x
        var total = 0f
        var pastSlop = false
        while (true) {
            val event = awaitPointerEvent()
            val change = event.changes.firstOrNull { it.id == down.id } ?: break
            if (!change.pressed) {
                change.consume()
                if (!pastSlop) {
                    if (startX < size.width / 2f) onPrev() else onNext()
                } else {
                    when {
                        total > 80f -> onPrev()
                        total < -80f -> onNext()
                    }
                }
                break
            }
            val dx = change.position.x - change.previousPosition.x
            total += dx
            if (!pastSlop && abs(change.position.x - startX) > slop) pastSlop = true
            if (pastSlop) change.consume()
        }
    }
}

@Composable
private fun PreviewNavButton(
    modifier: Modifier,
    icon: ImageVector,
    description: String,
    onClick: () -> Unit,
) {
    IconButton(
        onClick = onClick,
        modifier = modifier
            .size(48.dp)
            .clip(CircleShape)
            .background(rankoColors.bgCard.copy(alpha = 0.72f))
            .pointerHoverIcon(pointerIconHand),
    ) {
        Icon(icon, contentDescription = description, tint = rankoColors.text, modifier = Modifier.size(32.dp))
    }
}

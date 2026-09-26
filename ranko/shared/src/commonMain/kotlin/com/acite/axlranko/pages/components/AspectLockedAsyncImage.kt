package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.layout.ContentScale
import coil3.compose.AsyncImage
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality

/**
 * AsyncImage whose measured size is known before the bitmap decodes.
 * LazyColumn / LazyVerticalStaggeredGrid otherwise treat unloaded images as 0-height,
 * which jumps the scroll position when scrolling upward into not-yet-composed items.
 */
@Composable
fun AspectLockedAsyncImage(
    path: String,
    width: Int,
    height: Int,
    modifier: Modifier = Modifier,
    maxEdge: Int = 1024,
    contentDescription: String? = null,
    contentScale: ContentScale = ContentScale.FillWidth,
    filterQuality: FilterQuality = FilterQuality.High,
) {
    val ratio = if (width > 0 && height > 0) width.toFloat() / height.toFloat() else null
    val quality = LocalThumbnailQuality.current
    AsyncImage(
        model = BlobRef(path, maxEdge = maxEdge, quality = quality),
        contentDescription = contentDescription,
        contentScale = contentScale,
        filterQuality = filterQuality,
        modifier = if (ratio != null && ratio > 0f) {
            modifier.aspectRatio(ratio)
        } else {
            modifier
        },
    )
}

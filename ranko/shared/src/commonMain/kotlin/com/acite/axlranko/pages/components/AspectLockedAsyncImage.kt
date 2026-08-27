package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.layout.ContentScale
import coil3.compose.AsyncImage
import com.acite.axlranko.util.ImageHeaderSize
import java.io.File

/**
 * AsyncImage whose measured size is known before the bitmap decodes.
 * LazyColumn / LazyVerticalStaggeredGrid otherwise treat unloaded images as 0-height,
 * which jumps the scroll position when scrolling upward into not-yet-composed items.
 */
@Composable
fun AspectLockedAsyncImage(
    file: File,
    modifier: Modifier = Modifier,
    contentDescription: String? = null,
    contentScale: ContentScale = ContentScale.FillWidth,
    filterQuality: FilterQuality = FilterQuality.High,
) {
    val ratio = remember(file.absolutePath, file.length(), file.lastModified()) {
        ImageHeaderSize.aspectRatio(file)
    }
    AsyncImage(
        model = file,
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

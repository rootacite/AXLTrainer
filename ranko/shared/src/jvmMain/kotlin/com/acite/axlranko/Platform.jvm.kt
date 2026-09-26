package com.acite.axlranko

import androidx.compose.runtime.Composable
import coil3.ImageLoader
import coil3.compose.setSingletonImageLoaderFactory
import com.acite.axlranko.data.BlobFetcher
import com.acite.axlranko.data.BlobStore

class JVMPlatform : Platform {
    override val name: String = "Java ${System.getProperty("java.version")}"
}

actual fun getPlatform(): Platform = JVMPlatform()

actual val IoDispatcher = kotlinx.coroutines.Dispatchers.IO

actual fun localWallpaperModel(path: String): Any? {
    if (path.isEmpty()) return null
    val file = java.io.File(path)
    return file.takeIf { it.isFile }
}

actual suspend fun pickClientWallpaper(): String? {
    return io.github.vinceglb.filekit.core.FileKit.pickFile(
        type = io.github.vinceglb.filekit.core.PickerType.File(
            listOf("jpg", "jpeg", "png", "webp", "bmp"),
        ),
        mode = io.github.vinceglb.filekit.core.PickerMode.Single,
        title = "Select background image",
    )?.path?.takeIf { it.isNotBlank() }
}

@Composable
actual fun InstallBlobImageLoader(blobStore: BlobStore) {
    setSingletonImageLoaderFactory { context ->
        ImageLoader.Builder(context)
            .components { add(BlobFetcher.Factory(blobStore)) }
            .build()
    }
}
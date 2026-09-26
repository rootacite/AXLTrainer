package com.acite.axlranko

import androidx.compose.runtime.Composable
import coil3.ImageLoader
import coil3.compose.setSingletonImageLoaderFactory
import com.acite.axlranko.data.BlobFetcher
import com.acite.axlranko.data.BlobStore
import com.acite.axlranko.data.decodeBase64
import com.acite.axlranko.data.encodeBase64
import com.acite.axlranko.util.ImageCodecs
import com.acite.axlranko.util.WallpaperCache
import com.acite.axlranko.util.fitMaxEdge
import kotlinx.browser.document
import kotlinx.coroutines.CompletableDeferred
import org.w3c.dom.HTMLInputElement
import org.w3c.dom.events.Event
import org.w3c.files.FileReader

class WasmPlatform : Platform {
    override val name: String = "Web"
}

actual fun getPlatform(): Platform = WasmPlatform()

actual val IoDispatcher = kotlinx.coroutines.Dispatchers.Default

private const val WALLPAPER_MAX_EDGE = 1280
private const val WALLPAPER_JPEG_QUALITY = 78

actual fun localWallpaperModel(path: String): Any? {
    return WallpaperCache.remember(path) { key ->
        if (!key.startsWith("data:image/")) return@remember null
        val comma = key.indexOf(',')
        if (comma < 0) return@remember null
        val bytes = runCatching { decodeBase64(key.substring(comma + 1)) }.getOrNull() ?: return@remember null
        val rgba = ImageCodecs.decodeRgba(bytes) ?: return@remember null
        ImageCodecs.argbToImageBitmap(rgba.width, rgba.height, rgba.argb)
    }
}

actual suspend fun pickClientWallpaper(): String? {
    val deferred = CompletableDeferred<String?>()
    val input = document.createElement("input") as HTMLInputElement
    input.type = "file"
    input.accept = "image/jpeg,image/png,image/webp,image/bmp,.jpg,.jpeg,.png,.webp,.bmp"
    input.onchange = { _: Event ->
        val file = input.files?.item(0)
        if (file == null) {
            deferred.complete(null)
        } else {
            val reader = FileReader()
            reader.onload = {
                deferred.complete(reader.result?.toString())
                null
            }
            reader.onerror = {
                deferred.complete(null)
                null
            }
            reader.readAsDataURL(file)
        }
        null
    }
    input.click()
    val raw = deferred.await() ?: return null
    val comma = raw.indexOf(',')
    if (comma < 0) return null
    val bytes = runCatching { decodeBase64(raw.substring(comma + 1)) }.getOrNull() ?: return null
    val rgba = ImageCodecs.decodeRgba(bytes)?.fitMaxEdge(WALLPAPER_MAX_EDGE) ?: return null
    val jpeg = ImageCodecs.encodeJpeg(rgba.width, rgba.height, rgba.argb, WALLPAPER_JPEG_QUALITY)
    val stored = "data:image/jpeg;base64,${encodeBase64(jpeg)}"
    WallpaperCache.prime(stored, ImageCodecs.argbToImageBitmap(rgba.width, rgba.height, rgba.argb))
    return stored
}

@Composable
actual fun InstallBlobImageLoader(blobStore: BlobStore) {
    setSingletonImageLoaderFactory { context ->
        ImageLoader.Builder(context)
            .components { add(BlobFetcher.Factory(blobStore)) }
            .build()
    }
}

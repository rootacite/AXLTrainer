package com.acite.axlranko

import kotlinx.coroutines.CoroutineDispatcher

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform

expect val IoDispatcher: CoroutineDispatcher

/** Wallpaper is user-local chrome, not trainer data; the helper must not serve it. */
expect fun localWallpaperModel(path: String): Any?

/** Desktop: a filesystem path. Web: a `data:` URL from a client file. */
expect suspend fun pickClientWallpaper(): String?
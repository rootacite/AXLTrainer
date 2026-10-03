package com.acite.axlranko.data

internal interface WsConnection {
    suspend fun send(text: String)
    suspend fun receive(): String
    fun close()
}

internal expect class WsTransport() {
    suspend fun connect(host: String, port: Int): WsConnection
}

internal expect fun defaultWsHost(): String

/**
 * Host the wasm client opens. A query or a saved value wins; otherwise the page's own hostname,
 * so a browser that loaded the site from this machine talks to this machine. `127.0.0.1` is only
 * the fallback when that hostname is empty.
 */
internal fun wasmHelperHost(queryHost: String?, storedHost: String?, pageHostname: String): String {
    queryHost?.takeIf { it.isNotBlank() }?.let { return it }
    storedHost?.takeIf { it.isNotBlank() }?.let { return it }
    pageHostname.takeIf { it.isNotBlank() }?.let { return it }
    return "127.0.0.1"
}
internal expect fun defaultWsPort(): Int
internal expect fun persistWsEndpoint(host: String, port: Int)
internal expect fun spawnHelperIfNeeded(host: String, port: Int)
internal expect fun helperListening(host: String, port: Int): Boolean
internal expect fun stopSpawnedHelper()

/** What this client calls itself when it takes the helper (`hello`); shown in a refusal. */
internal expect val clientName: String

expect val showsHelperEndpointSettings: Boolean
expect val wallpaperImagesSupported: Boolean
expect val imageBackgroundUsesHaze: Boolean

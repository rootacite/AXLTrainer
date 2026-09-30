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

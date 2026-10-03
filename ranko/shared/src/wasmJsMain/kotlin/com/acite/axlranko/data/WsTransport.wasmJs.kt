package com.acite.axlranko.data

import kotlinx.browser.window
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.channels.Channel
import org.w3c.dom.MessageEvent
import org.w3c.dom.WebSocket
import org.w3c.dom.events.Event

private const val HOST_KEY = "axlranko.ws.host"
private const val PORT_KEY = "axlranko.ws.port"

internal actual class WsTransport actual constructor() {
    actual suspend fun connect(host: String, port: Int): WsConnection {
        val conn = BrowserWsConnection()
        conn.open("ws://$host:$port")
        return conn
    }
}

private class BrowserWsConnection : WsConnection {
    private val incoming = Channel<String>(Channel.UNLIMITED)
    private val opened = CompletableDeferred<Unit>()
    private var socket: WebSocket? = null

    suspend fun open(url: String) {
        val ws = WebSocket(url)
        socket = ws
        ws.onopen = { _: Event ->
            opened.complete(Unit)
            null
        }
        ws.onmessage = { event: MessageEvent ->
            incoming.trySend(event.data.toString())
            null
        }
        ws.onerror = { _: Event ->
            if (!opened.isCompleted) {
                opened.completeExceptionally(IllegalStateException("WebSocket error opening $url"))
            }
            incoming.close()
            null
        }
        ws.onclose = { _: Event ->
            if (!opened.isCompleted) {
                opened.completeExceptionally(IllegalStateException("WebSocket closed before open: $url"))
            }
            incoming.close()
            null
        }
        try {
            opened.await()
        } catch (e: Exception) {
            runCatching { ws.close() }
            throw e
        }
    }

    override suspend fun send(text: String) {
        val ws = socket ?: error("WebSocket is not connected")
        ws.send(text)
    }

    override suspend fun receive(): String = incoming.receive()

    override fun close() {
        runCatching { socket?.close() }
        if (!opened.isCompleted) {
            opened.completeExceptionally(IllegalStateException("WebSocket closed"))
        }
        incoming.close()
    }
}

internal actual fun defaultWsHost(): String =
    wasmHelperHost(queryParam("host"), window.localStorage.getItem(HOST_KEY), window.location.hostname)

internal actual fun defaultWsPort(): Int {
    queryParam("port")?.toIntOrNull()?.let { return it }
    window.localStorage.getItem(PORT_KEY)?.toIntOrNull()?.let { return it }
    return 18765
}

internal actual fun persistWsEndpoint(host: String, port: Int) {
    window.localStorage.setItem(HOST_KEY, host)
    window.localStorage.setItem(PORT_KEY, port.toString())
}

internal actual fun helperListening(host: String, port: Int): Boolean = false

internal actual fun spawnHelperIfNeeded(host: String, port: Int) {
}

internal actual fun stopSpawnedHelper() {
}

internal actual val clientName: String = "chromatrix-web"

actual val showsHelperEndpointSettings: Boolean = true
actual val wallpaperImagesSupported: Boolean = true
actual val imageBackgroundUsesHaze: Boolean = false

private fun queryParam(name: String): String? {
    val search = window.location.search
    if (search.isEmpty()) return null
    val query = search.removePrefix("?")
    for (part in query.split("&")) {
        val eq = part.indexOf('=')
        if (eq <= 0) continue
        if (part.substring(0, eq) == name) {
            return part.substring(eq + 1).takeIf { it.isNotBlank() }
        }
    }
    return null
}

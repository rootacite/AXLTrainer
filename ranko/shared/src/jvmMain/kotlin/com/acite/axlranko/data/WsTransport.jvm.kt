package com.acite.axlranko.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.future.await
import kotlinx.coroutines.withContext
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Proxy
import java.net.ProxySelector
import java.net.Socket
import java.net.SocketAddress
import java.net.URI
import java.net.http.HttpClient
import java.net.http.WebSocket
import java.nio.ByteBuffer
import java.time.Duration
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CompletionStage

private object NoProxySelector : ProxySelector() {
    override fun select(uri: URI?): MutableList<Proxy> = mutableListOf(Proxy.NO_PROXY)
    override fun connectFailed(uri: URI?, sa: SocketAddress?, ioe: IOException?) {}
}

internal actual class WsTransport actual constructor() {
    actual suspend fun connect(host: String, port: Int): WsConnection {
        val listener = JvmWsConnection()
        val client = HttpClient.newBuilder()
            .proxy(NoProxySelector)
            .connectTimeout(Duration.ofSeconds(2))
            .build()
        val ws = client.newWebSocketBuilder()
            .buildAsync(URI.create("ws://$host:$port"), listener)
            .await()
        listener.attach(ws)
        return listener
    }
}

private class JvmWsConnection : WebSocket.Listener, WsConnection {
    private val incoming = Channel<String>(Channel.UNLIMITED)
    private val parts = StringBuilder()
    private var socket: WebSocket? = null

    fun attach(ws: WebSocket) {
        socket = ws
    }

    override fun onOpen(webSocket: WebSocket) {
        webSocket.request(1)
    }

    override fun onText(webSocket: WebSocket, data: CharSequence, last: Boolean): CompletionStage<*> {
        parts.append(data)
        if (last) {
            incoming.trySend(parts.toString())
            parts.setLength(0)
        }
        webSocket.request(1)
        return CompletableFuture.completedFuture(null)
    }

    override fun onBinary(webSocket: WebSocket, data: ByteBuffer, last: Boolean): CompletionStage<*> {
        webSocket.request(1)
        return CompletableFuture.completedFuture(null)
    }

    override fun onClose(webSocket: WebSocket, statusCode: Int, reason: String?): CompletionStage<*> {
        incoming.close()
        return CompletableFuture.completedFuture(null)
    }

    override fun onError(webSocket: WebSocket, error: Throwable) {
        incoming.close(error)
    }

    override suspend fun send(text: String) {
        val ws = socket ?: error("WebSocket is not connected")
        withContext(Dispatchers.IO) {
            ws.sendText(text, true).join()
        }
    }

    override suspend fun receive(): String {
        return incoming.receive()
    }

    override fun close() {
        runCatching { socket?.sendClose(WebSocket.NORMAL_CLOSURE, "") }
        incoming.close()
    }
}

internal actual fun defaultWsHost(): String =
    System.getenv("AXL_WS_HOST")?.takeIf { it.isNotBlank() } ?: "127.0.0.1"

internal actual fun defaultWsPort(): Int =
    System.getenv("AXL_WS_PORT")?.toIntOrNull() ?: 18765

internal actual fun persistWsEndpoint(host: String, port: Int) {
}

actual val showsHelperEndpointSettings: Boolean = false
actual val wallpaperImagesSupported: Boolean = true
actual val imageBackgroundUsesHaze: Boolean = true

internal actual fun helperListening(host: String, port: Int): Boolean {
    val connectHost = if (host == "0.0.0.0" || host == "::") "127.0.0.1" else host
    return try {
        val done = java.util.concurrent.CountDownLatch(1)
        val ok = java.util.concurrent.atomic.AtomicBoolean(false)
        val client = java.net.http.HttpClient.newBuilder()
            .proxy(NoProxySelector)
            .connectTimeout(java.time.Duration.ofMillis(250))
            .build()
        client.newWebSocketBuilder()
            .buildAsync(
                java.net.URI.create("ws://$connectHost:$port"),
                object : java.net.http.WebSocket.Listener {
                    override fun onOpen(webSocket: java.net.http.WebSocket) {
                        ok.set(true)
                        webSocket.sendClose(java.net.http.WebSocket.NORMAL_CLOSURE, "")
                        done.countDown()
                    }

                    override fun onError(webSocket: java.net.http.WebSocket?, error: Throwable?) {
                        done.countDown()
                    }
                },
            )
        done.await(400, java.util.concurrent.TimeUnit.MILLISECONDS)
        ok.get()
    } catch (_: Exception) {
        false
    }
}

internal actual fun spawnHelperIfNeeded(host: String, port: Int) {
    JvmHelperProcess.ensure(host, port)
}

internal actual fun stopSpawnedHelper() {
    JvmHelperProcess.stopSpawned()
}

internal actual val clientName: String = "chromatrix-desktop"

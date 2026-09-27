package com.acite.axlranko.util

import kotlinx.browser.document
import kotlinx.coroutines.CompletableDeferred
import org.w3c.dom.HTMLInputElement
import org.w3c.dom.events.Event
import org.w3c.files.FileReader

actual suspend fun pickClientTextFile(title: String, extensions: List<String>): ClientTextFile? {
    val deferred = CompletableDeferred<ClientTextFile?>()
    val input = document.createElement("input") as HTMLInputElement
    input.type = "file"
    input.accept = extensions.joinToString(",") { extension ->
        if (extension.startsWith(".")) extension else ".$extension"
    }
    input.onchange = { _: Event ->
        val file = input.files?.item(0)
        if (file == null) {
            deferred.complete(null)
        } else {
            val reader = FileReader()
            reader.onload = {
                deferred.complete(ClientTextFile(name = file.name, text = reader.result?.toString().orEmpty()))
                null
            }
            reader.onerror = {
                deferred.complete(null)
                null
            }
            reader.readAsText(file)
        }
        null
    }
    input.click()
    return deferred.await()
}

/** The browser has no file manager to open a server path in. */
actual fun openLocalDirectory(path: String): Boolean = false

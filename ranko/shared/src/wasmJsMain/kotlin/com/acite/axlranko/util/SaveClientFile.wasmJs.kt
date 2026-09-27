package com.acite.axlranko.util

import com.acite.axlranko.data.encodeBase64
import kotlinx.browser.document
import org.w3c.dom.HTMLAnchorElement

/**
 * A browser download: the bytes travel as a data URL, which keeps this to string APIs (no
 * typed-array interop). Fine for prompt lists and single images; a very large file would be
 * better off as a server-side copy.
 */
actual suspend fun saveClientFile(
    suggestedName: String,
    bytes: ByteArray,
    mime: String,
): String? {
    val anchor = document.createElement("a") as HTMLAnchorElement
    anchor.href = "data:$mime;base64,${encodeBase64(bytes)}"
    anchor.download = suggestedName
    anchor.style.display = "none"
    document.body?.appendChild(anchor)
    anchor.click()
    anchor.remove()
    return suggestedName
}

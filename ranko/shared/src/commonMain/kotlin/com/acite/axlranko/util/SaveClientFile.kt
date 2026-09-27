package com.acite.axlranko.util

/**
 * Hands one finished file to the user: desktop saves it wherever they pick, web downloads it.
 *
 * Returns the destination (a path, or the suggested name on web) or null when the user cancelled.
 */
expect suspend fun saveClientFile(suggestedName: String, bytes: ByteArray, mime: String = "application/octet-stream"): String?

/** The same for a text file. */
suspend fun saveClientText(suggestedName: String, text: String): String? =
    saveClientFile(suggestedName, text.encodeToByteArray(), mime = "text/plain;charset=utf-8")

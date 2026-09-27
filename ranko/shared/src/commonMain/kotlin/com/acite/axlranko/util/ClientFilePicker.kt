package com.acite.axlranko.util

/** A text file the user picks on **their** machine (a ComfyUI API workflow, a prompt list). */
data class ClientTextFile(val name: String, val text: String)

/**
 * Desktop: the file's text, read locally after the OS dialog. Web: the browser's file input
 * (the server can never see a client path, so the upload sends the text).
 */
expect suspend fun pickClientTextFile(title: String, extensions: List<String>): ClientTextFile?

/** Desktop: reveal a folder in the file manager. Web: unsupported, returns false. */
expect fun openLocalDirectory(path: String): Boolean

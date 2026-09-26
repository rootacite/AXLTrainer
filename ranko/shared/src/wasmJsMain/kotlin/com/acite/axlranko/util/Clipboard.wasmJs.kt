package com.acite.axlranko.util

import kotlinx.browser.window

actual fun copyTextToClipboard(text: String) {
    window.navigator.clipboard.writeText(text)
}

package com.acite.axlranko.util

import java.awt.Toolkit
import java.awt.datatransfer.StringSelection

actual fun copyTextToClipboard(text: String) {
    val clipboard = Toolkit.getDefaultToolkit().systemClipboard
    clipboard.setContents(StringSelection(text), null)
}

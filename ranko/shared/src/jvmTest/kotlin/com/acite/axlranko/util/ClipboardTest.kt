package com.acite.axlranko.util

import java.awt.GraphicsEnvironment
import java.awt.Toolkit
import java.awt.datatransfer.DataFlavor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ClipboardTest {

    @Test
    fun copiesPlainText() {
        if (GraphicsEnvironment.isHeadless()) return
        copyTextToClipboard("1girl")
        val contents = Toolkit.getDefaultToolkit().systemClipboard.getContents(null)
        assertTrue(contents.isDataFlavorSupported(DataFlavor.stringFlavor))
        assertEquals("1girl", contents.getTransferData(DataFlavor.stringFlavor) as String)
    }
}

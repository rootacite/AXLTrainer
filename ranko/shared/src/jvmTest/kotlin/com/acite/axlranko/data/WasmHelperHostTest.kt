package com.acite.axlranko.data

import kotlin.test.Test
import kotlin.test.assertEquals

/** The wasm client's host fallback. The desktop default stays 127.0.0.1 in its own actual. */
class WasmHelperHostTest {
    @Test
    fun queryThenStoredThenThePageThenLoopback() {
        assertEquals("10.0.0.8", wasmHelperHost("10.0.0.8", "192.168.1.4", "192.168.1.9"))
        assertEquals("192.168.1.4", wasmHelperHost(null, "192.168.1.4", "192.168.1.9"))
        assertEquals("192.168.1.4", wasmHelperHost("  ", "192.168.1.4", "192.168.1.9"))
        assertEquals("192.168.1.9", wasmHelperHost(null, null, "192.168.1.9"))
        assertEquals("192.168.1.9", wasmHelperHost(null, "", "192.168.1.9"))
        assertEquals("127.0.0.1", wasmHelperHost(null, null, ""))
        assertEquals("127.0.0.1", wasmHelperHost(null, null, "   "))
    }
}

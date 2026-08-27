package com.acite.axlranko

import com.acite.axlranko.data.IpcRequest
import com.acite.axlranko.data.IpcResponse
import com.acite.axlranko.model.DashboardResponse
import com.acite.axlranko.model.SamplesResponse
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

class DashboardIpcTest {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
    }

    @Test
    fun requestRoundTrip() {
        val encoded = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(id = 7, method = "dashboard", params = buildJsonObject { put("name", "run") }),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        assertEquals(7, decoded.id)
        assertEquals("dashboard", decoded.method)
        assertEquals("run", decoded.params["name"]?.jsonPrimitive?.content)
    }

    @Test
    fun dashboardResponseParses() {
        val raw = """
            {
              "config": { "output_name": "kanae", "logging_dir": "/tmp/logs" },
              "latest_stats": { "current_step": 10, "Train/Loss": 0.5, "Train/Avg_Loss": 0.4 },
              "metrics": {
                "Train/Loss": [ { "step": 1, "value": 0.5, "wall_time": 1.0 } ],
                "Train/Avg_Loss": [ { "step": 1, "value": 0.4, "wall_time": 1.0 } ]
              }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(DashboardResponse.serializer(), raw)
        assertEquals("kanae", parsed.config["output_name"]?.jsonPrimitive?.content)
        assertEquals(10, parsed.latestStats["current_step"]?.jsonPrimitive?.content?.toInt())
        assertEquals(1, parsed.metrics["Train/Loss"]?.size)
        assertEquals(0.5f, parsed.metrics["Train/Loss"]?.first()?.value)
        assertEquals(0.4f, parsed.metrics["Train/Avg_Loss"]?.first()?.value)
    }

    @Test
    fun samplesResponseParsesPaths() {
        val raw = """
            {
              "samples": {
                "1000": [
                  { "filename": "a_1000_0.png", "repeat_idx": 0, "path": "/tmp/a_1000_0.png" }
                ]
              }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(SamplesResponse.serializer(), raw)
        assertEquals("/tmp/a_1000_0.png", parsed.samples["1000"]?.first()?.path)
    }

    @Test
    fun errorEnvelopeParses() {
        val raw = """{"id": 3, "ok": false, "error": "unknown method: generate"}"""
        val parsed = json.decodeFromString(IpcResponse.serializer(), raw)
        assertEquals(3, parsed.id)
        assertTrue(!parsed.ok)
        assertEquals("unknown method: generate", parsed.error)
    }
}

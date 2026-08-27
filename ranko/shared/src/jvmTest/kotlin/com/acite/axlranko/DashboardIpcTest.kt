package com.acite.axlranko

import com.acite.axlranko.data.IpcRequest
import com.acite.axlranko.data.IpcResponse
import com.acite.axlranko.model.DashboardResponse
import com.acite.axlranko.model.SamplesResponse
import com.acite.axlranko.model.TrainStatus
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
    fun trainStatusParsesSwapAndPhases() {
        val raw = """
            {
              "schema": 1,
              "pid": 4242,
              "started_at": 1000.0,
              "updated_at": 1001.5,
              "status": "pausing",
              "paused_from": "training",
              "output_name": "rein",
              "encoding": { "current": 10, "total": 20, "done": true },
              "training": { "step": 12, "total_steps": 100, "epoch": 1, "epochs": 16, "loss": 0.25, "avg_loss": 0.3 },
              "sampling": { "active": false, "repeat": 0, "repeats": 3, "denoise_step": 0, "denoise_steps": 55, "global_step": 0 },
              "swap": { "stage": "offload_unet", "detail": "Moving UNet to CPU", "current": 1, "total": 5 },
              "error": null,
              "detail": null,
              "alive": true,
              "log_path": "/tmp/train.log"
            }
        """.trimIndent()
        val parsed = json.decodeFromString(TrainStatus.serializer(), raw)
        assertEquals("pausing", parsed.status)
        assertEquals(4242, parsed.pid)
        assertEquals("rein", parsed.outputName)
        assertEquals("training", parsed.pausedFrom)
        assertEquals(12, parsed.training.step)
        assertEquals(0.25f, parsed.training.loss)
        assertEquals("offload_unet", parsed.swap?.stage)
        assertEquals(1, parsed.swap?.current)
        assertTrue(parsed.alive)
        assertEquals("/tmp/train.log", parsed.logPath)
        assertEquals(true, parsed.encoding.done)
    }

    @Test
    fun trainResetRequestRoundTrip() {
        val encoded = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 9,
                method = "train_reset",
                params = buildJsonObject { put("delete_weights", true) },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        assertEquals("train_reset", decoded.method)
        assertEquals("true", decoded.params["delete_weights"]?.jsonPrimitive?.content)
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

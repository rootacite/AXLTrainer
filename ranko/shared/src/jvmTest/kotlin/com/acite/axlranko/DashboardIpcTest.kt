package com.acite.axlranko

import com.acite.axlranko.data.IpcRequest
import com.acite.axlranko.data.IpcResponse
import com.acite.axlranko.model.CheckpointsResponse
import com.acite.axlranko.model.DashboardResponse
import com.acite.axlranko.model.DatasetTagResult
import com.acite.axlranko.model.HardwareStatus
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
              "run_id": "rein_20260911_120000",
              "resume": { "path": "/out/rein_final/rein.safetensors", "filename": "rein.safetensors", "step": 300, "epoch": 7, "loaded": 96, "skipped": 2 },
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
        assertEquals("rein_20260911_120000", parsed.runId)
        assertEquals(300, parsed.resume?.step)
        assertEquals(96, parsed.resume?.loaded)
        assertEquals("rein.safetensors", parsed.resume?.filename)
    }

    @Test
    fun trainStatusWithoutRunOrResumeDefaultsToNull() {
        val parsed = json.decodeFromString(TrainStatus.serializer(), """{"status": "idle"}""")
        assertEquals(null, parsed.runId)
        assertEquals(null, parsed.resume)
    }

    @Test
    fun dashboardResponseParsesRunId() {
        val raw = """
            {
              "config": { "output_name": "rein" },
              "run_id": "rein_20260911_120000",
              "latest_stats": {},
              "metrics": {}
            }
        """.trimIndent()
        val parsed = json.decodeFromString(DashboardResponse.serializer(), raw)
        assertEquals("rein_20260911_120000", parsed.runId)
    }

    @Test
    fun samplesResponseParsesRunId() {
        val parsed = json.decodeFromString(
            SamplesResponse.serializer(),
            """{"run_id": "rein_20260911_120000", "samples": {}}""",
        )
        assertEquals("rein_20260911_120000", parsed.runId)
    }

    @Test
    fun checkpointsResponseParses() {
        val raw = """
            {
              "checkpoints": [
                {
                  "path": "/out/rein_20260911_120000/rein_final/rein.safetensors",
                  "run_id": "rein_20260911_120000",
                  "dir": "rein_final",
                  "filename": "rein.safetensors",
                  "step": 300,
                  "epoch": 7,
                  "final": true,
                  "size_bytes": 12345678,
                  "modified": 1757500000.0,
                  "network_dim": 48,
                  "network_alpha": 24,
                  "output_name": "rein"
                }
              ]
            }
        """.trimIndent()
        val parsed = json.decodeFromString(CheckpointsResponse.serializer(), raw)
        assertEquals(1, parsed.checkpoints.size)
        val item = parsed.checkpoints.first()
        assertEquals("rein_final", item.dir)
        assertEquals(300, item.step)
        assertEquals(true, item.final)
        assertEquals(12345678L, item.sizeBytes)
        assertEquals(48, item.networkDim)
        assertEquals(24, item.networkAlpha)
    }

    @Test
    fun listCheckpointsRequestRoundTrip() {
        val encoded = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 21,
                method = "list_checkpoints",
                params = buildJsonObject {
                    put("name", "rein")
                    put("output_dir", "/out")
                },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        assertEquals("list_checkpoints", decoded.method)
        assertEquals("/out", decoded.params["output_dir"]?.jsonPrimitive?.content)
    }

    @Test
    fun hardwareStatusParsesNvtopSnapshot() {
        val raw = """
            {
              "available": true,
              "error": null,
              "ts": 1710000000.12,
              "gpus": [
                {
                  "index": 0,
                  "name": "AMD Radeon RX 9070 XT",
                  "gpu_clock_mhz": 2165.0,
                  "mem_clock_mhz": 2500.0,
                  "fan_pct": 30.0,
                  "gpu_util_pct": 92.0,
                  "mem_util_pct": 76.0,
                  "power_w": 303.0,
                  "temp_c": 72.0,
                  "temp_edge_c": 72.0,
                  "temp_junction_c": 85.0,
                  "temp_mem_c": 80.0,
                  "mem_total_bytes": 17095983104,
                  "mem_used_bytes": 13000000000,
                  "mem_free_bytes": 4095983104
                }
              ],
              "cpu": {
                "name": "Test CPU",
                "n_logical": 28,
                "util_pct": 41.2,
                "temp_c": 41.0,
                "mem_total_bytes": 67108864000,
                "mem_used_bytes": 22020096000
              }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(HardwareStatus.serializer(), raw)
        assertEquals(true, parsed.available)
        assertEquals(null, parsed.error)
        assertEquals("AMD Radeon RX 9070 XT", parsed.gpus.first().name)
        assertEquals(92.0, parsed.gpus.first().gpuUtilPct)
        assertEquals(85.0, parsed.gpus.first().tempJunctionC)
        assertEquals(17095983104L, parsed.gpus.first().memTotalBytes)
        assertEquals(28, parsed.cpu.nLogical)
        assertEquals(41.2, parsed.cpu.utilPct)
        assertEquals(22020096000L, parsed.cpu.memUsedBytes)
        assertEquals(null, parsed.vmmVa)
    }

    @Test
    fun hardwareStatusVmmVaParses() {
        val raw = """
            {
              "available": true,
              "error": null,
              "ts": 1.0,
              "gpus": [],
              "cpu": { "name": "", "n_logical": 8, "util_pct": null, "temp_c": null },
              "vmm_va": {
                "patch": "vmm",
                "used_bytes": 8388608,
                "total_bytes": 281474976710656,
                "total_source": "journal",
                "pid": 1234,
                "spans": 4
              }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(HardwareStatus.serializer(), raw)
        val va = parsed.vmmVa
        assertEquals("vmm", va?.patch)
        assertEquals(8388608L, va?.usedBytes)
        assertEquals(281474976710656L, va?.totalBytes)
        assertEquals("journal", va?.totalSource)
        assertEquals(1234, va?.pid)
        assertEquals(4, va?.spans)
    }

    @Test
    fun hardwareStatusUnavailableStillParses() {
        val raw = """
            {
              "available": false,
              "error": "nvtop not found on PATH",
              "ts": 1.0,
              "gpus": [],
              "cpu": { "name": "", "n_logical": 8, "util_pct": null, "temp_c": null }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(HardwareStatus.serializer(), raw)
        assertEquals(false, parsed.available)
        assertEquals("nvtop not found on PATH", parsed.error)
        assertTrue(parsed.gpus.isEmpty())
        assertEquals(null, parsed.cpu.utilPct)
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
    fun datasetTagResultParses() {
        val raw = """
            {
              "directory": "/tmp/alice",
              "threshold": 0.35,
              "provider": "MIGraphXExecutionProvider",
              "total": 4,
              "processed": 4,
              "failed": 0,
              "seconds": 1.25,
              "errors": []
            }
        """.trimIndent()
        val parsed = json.decodeFromString(DatasetTagResult.serializer(), raw)
        assertEquals("/tmp/alice", parsed.directory)
        assertEquals(0.35f, parsed.threshold)
        assertEquals("MIGraphXExecutionProvider", parsed.provider)
        assertEquals(4, parsed.processed)
        assertEquals(0, parsed.failed)
        assertEquals(1.25f, parsed.seconds)
    }

    @Test
    fun datasetTagRequestRoundTrip() {
        val encoded = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 11,
                method = "dataset_tag",
                params = buildJsonObject {
                    put("directory", "/tmp/alice")
                    put("threshold", 0.4)
                },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        assertEquals("dataset_tag", decoded.method)
        assertEquals("/tmp/alice", decoded.params["directory"]?.jsonPrimitive?.content)
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

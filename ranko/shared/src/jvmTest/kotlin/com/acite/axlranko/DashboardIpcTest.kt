package com.acite.axlranko

import com.acite.axlranko.data.IpcRequest
import com.acite.axlranko.data.IpcResponse
import com.acite.axlranko.model.CheckpointPinsResponse
import com.acite.axlranko.model.CheckpointsResponse
import com.acite.axlranko.model.DashboardResponse
import com.acite.axlranko.model.DatasetTagResult
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.RunsResponse
import com.acite.axlranko.model.SamplesResponse
import com.acite.axlranko.model.TrainStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

class DashboardIpcTest {
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
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
    fun trainStatusParsesTheLiveSettings() {
        val raw = """
            {
              "status": "training",
              "settings": { "save_every_n_steps": 50, "sampling_enabled": false, "next_save_step": 1250 }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(TrainStatus.serializer(), raw)
        assertEquals(50, parsed.settings.saveEveryNSteps)
        assertFalse(parsed.settings.samplingEnabled)
        assertEquals(1250, parsed.settings.nextSaveStep)

        // An older helper (or a run that has not published anything yet) keeps the defaults.
        val bare = json.decodeFromString(TrainStatus.serializer(), """{"status": "idle"}""")
        assertEquals(0, bare.settings.saveEveryNSteps)
        assertTrue(bare.settings.samplingEnabled)
        assertEquals(0, bare.settings.nextSaveStep)
    }

    @Test
    fun generatedSampleJobParsesASetsPass() {
        val raw = """
            {
              "id": "rein_s003050_sets_gen_20260929_031500",
              "state": "running",
              "mode": "sets",
              "step": 3050,
              "checkpoint": "/out/rein_20260911_120000/rein_s003050/rein.safetensors",
              "files": [
                "/out/rein_20260911_120000/rein_samples/generated/rein_s003050_sets_gen_20260929_031500_p0_0.png"
              ],
              "images_done": 1,
              "total_images": 6,
              "current_set": 1,
              "total_sets": 6,
              "current_step": 12,
              "total_steps": 35
            }
        """.trimIndent()
        val parsed = json.decodeFromString(
            com.acite.axlranko.model.GeneratedSampleJob.serializer(),
            raw,
        )
        assertEquals("sets", parsed.mode)
        assertEquals(1, parsed.files.size)
        assertEquals(1, parsed.imagesDone)
        assertEquals(6, parsed.totalImages)
        assertEquals(1, parsed.currentSet)
        assertEquals(6, parsed.totalSets)

        // A job file written before the sets mode reads as one image with no files.
        val old = json.decodeFromString(
            com.acite.axlranko.model.GeneratedSampleJob.serializer(),
            """{"id": "rein_s000100_gen", "state": "done", "image_path": "/out/x.png"}""",
        )
        assertEquals("single", old.mode)
        assertTrue(old.files.isEmpty())
        assertEquals(1, old.totalImages)
    }

    @Test
    fun aNullInANumericFieldReadsAsItsDefault() {
        // The record api.py writes before the generator touches it can carry `null` where the model
        // declares an Int; one such job must not take the whole generated-samples list down.
        val raw = """
            {
              "id": "rein_s003050_sets_gen_20260929_061723",
              "state": "running",
              "mode": "sets",
              "step": null,
              "total_steps": null,
              "current_step": null,
              "images_done": null,
              "total_images": 6
            }
        """.trimIndent()
        val parsed = json.decodeFromString(
            com.acite.axlranko.model.GeneratedSampleJob.serializer(),
            raw,
        )
        assertEquals(0, parsed.totalSteps)
        assertEquals(0, parsed.currentStep)
        assertEquals(0, parsed.imagesDone)
        assertEquals(6, parsed.totalImages)
        assertEquals(null, parsed.step)
    }

    @Test
    fun runsResponseParsesTheHistoryList() {
        val raw = """
            {
              "runs": [
                {
                  "run_id": "Tsukuyomi_20260928_110928",
                  "output_name": "Tsukuyomi",
                  "output_dir": "/out/Tsukuyomi_20260928_110928",
                  "log_dir": "/logs/Tsukuyomi_20260928_110928",
                  "has_output": true,
                  "has_log": false,
                  "last_step": 4500,
                  "samples": 12,
                  "checkpoints": 46,
                  "size_bytes": 11172201792,
                  "modified": 1790587779.5,
                  "current": true,
                  "live": true
                },
                { "run_id": "Kirika_20260927_225224", "output_name": "Kirika" }
              ]
            }
        """.trimIndent()
        val parsed = json.decodeFromString(RunsResponse.serializer(), raw)
        assertEquals(2, parsed.runs.size)
        val live = parsed.runs.first()
        assertEquals("Tsukuyomi_20260928_110928", live.runId)
        assertEquals("Tsukuyomi", live.outputName)
        assertTrue(live.hasOutput)
        assertFalse(live.hasLog)
        assertEquals(4500, live.lastStep)
        assertEquals(12, live.samples)
        assertEquals(46, live.checkpoints)
        assertEquals(11172201792L, live.sizeBytes)
        assertTrue(live.current)
        assertTrue(live.live)
        // The second run omits the optional figures; they default instead of failing the parse.
        assertEquals("Kirika_20260927_225224", parsed.runs[1].runId)
        assertEquals(null, parsed.runs[1].lastStep)
        assertFalse(parsed.runs[1].live)
        assertFalse(parsed.runs[1].current)
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
    fun checkpointPinsResponseParses() {
        val raw = """
            {
              "run_id": "rein_20260911_120000",
              "file": "/logs/rein_20260911_120000/checkpoint_pins.json",
              "pins": [
                {
                  "path": "/out/rein_20260911_120000/rein_s000300/rein.safetensors",
                  "dir": "rein_s000300",
                  "step": 300,
                  "pinned_at": 1757500000.5
                }
              ]
            }
        """.trimIndent()
        val parsed = json.decodeFromString(CheckpointPinsResponse.serializer(), raw)
        assertEquals("rein_20260911_120000", parsed.runId)
        assertEquals("/logs/rein_20260911_120000/checkpoint_pins.json", parsed.file)
        assertEquals(1, parsed.pins.size)
        val pin = parsed.pins.first()
        assertEquals("/out/rein_20260911_120000/rein_s000300/rein.safetensors", pin.path)
        assertEquals("rein_s000300", pin.dir)
        assertEquals(300, pin.step)
        assertEquals(1757500000.5, pin.pinnedAt)
    }

    @Test
    fun emptyPinListAndMissingRunParse() {
        val parsed = json.decodeFromString(
            CheckpointPinsResponse.serializer(),
            """{"run_id": null, "file": null, "pins": []}""",
        )
        assertEquals(null, parsed.runId)
        assertEquals(null, parsed.file)
        assertTrue(parsed.pins.isEmpty())
    }

    @Test
    fun checkpointPinSetRequestRoundTrip() {
        val encoded = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 22,
                method = "checkpoint_pin_set",
                params = buildJsonObject {
                    put("path", "/out/rein_s000300/rein.safetensors")
                    put("pinned", true)
                    put("dir", "rein_s000300")
                    put("step", 300)
                    put("run_id", "rein_20260911_120000")
                },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        assertEquals("checkpoint_pin_set", decoded.method)
        assertEquals("true", decoded.params["pinned"]?.jsonPrimitive?.content)
        assertEquals("300", decoded.params["step"]?.jsonPrimitive?.content)
        assertEquals("rein_20260911_120000", decoded.params["run_id"]?.jsonPrimitive?.content)
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
                "spans": 4,
                "never_reuse": true
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
        assertEquals(true, va?.vaNeverReuse)
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
                params = buildJsonObject { put("name", "rein") },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        assertEquals("train_reset", decoded.method)
        assertEquals("rein", decoded.params["name"]?.jsonPrimitive?.content)
        // Reset clears the state and nothing else: there is no flag that deletes a run's weights.
        assertNull(decoded.params["delete_weights"])
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

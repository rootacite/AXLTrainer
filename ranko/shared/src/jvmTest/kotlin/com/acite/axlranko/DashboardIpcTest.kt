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
import com.acite.axlranko.model.EvaluationPromptsResponse
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.TaggerInfoResult
import com.acite.axlranko.model.TrainStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray

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
        assertNull(bare.requested)
    }

    @Test
    fun trainStatusParsesTheOutstandingRequest() {
        // What `train_settings` answers with while a sample pass is still running: the effective
        // values as published, plus the change that has been accepted and not adopted yet.
        val raw = """
            {
              "status": "sampling",
              "settings": { "save_every_n_steps": 50, "sampling_enabled": true, "next_save_step": 3350 },
              "requested": { "save_every_n_steps": 50, "sampling_enabled": false }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(TrainStatus.serializer(), raw)
        assertTrue(parsed.settings.samplingEnabled)
        assertEquals(50, parsed.requested?.saveEveryNSteps)
        assertFalse(parsed.requested?.samplingEnabled ?: true)
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
    fun anEvaluationJobParsesItsPlanAndScores() {
        val raw = """
            {
              "id": "rein_s003050_evaluate_gen_20261001_120000",
              "state": "done",
              "mode": "evaluate",
              "step": 3050,
              "phase": "done",
              "depth": 20,
              "threshold": 0.35,
              "categories": ["general"],
              "config_source": "/logs/rein_20260911_120000/config.toml",
              "checkpoint": "/out/rein_20260911_120000/rein_s003050/rein.safetensors",
              "files": [
                "/out/rein_20260911_120000/rein_samples/generated/rein_s003050_evaluate_gen_20261001_120000_p0_3.png"
              ],
              "images_done": 21,
              "total_images": 21,
              "scores": {
                "tp": 40, "fp": 12, "fn": 24,
                "precision": 0.769, "recall": 0.625, "f1": 0.689,
                "union_tp": 30, "union_fp": 8, "union_fn": 12,
                "union_precision": 0.789, "union_recall": 0.714, "union_f1": 0.75,
                "images_scored": 21, "images_failed": 1, "images_skipped": 0,
                "groups": [
                  {"prompt": "1girl, solo", "images": 3, "tp": 5, "fp": 2, "fn": 1,
                   "precision": 0.7, "recall": 0.83, "f1": 0.76,
                   "union_precision": 0.8, "union_recall": 1.0, "union_f1": 0.88}
                ],
                "top_false_positives": [{"tag": "solo", "count": 5}],
                "top_false_negatives": [{"tag": "long hair", "count": 7}]
              }
            }
        """.trimIndent()
        val parsed = json.decodeFromString(
            com.acite.axlranko.model.GeneratedSampleJob.serializer(),
            raw,
        )
        assertEquals("evaluate", parsed.mode)
        assertEquals("done", parsed.phase)
        assertEquals(20, parsed.depth)
        assertEquals(0.35f, parsed.threshold)
        assertEquals(listOf("general"), parsed.categories)
        assertEquals("/logs/rein_20260911_120000/config.toml", parsed.configSource)
        assertEquals(1, parsed.files.size)
        val scores = parsed.scores ?: error("scores must parse")
        assertEquals(0.689f, scores.f1)
        assertEquals(0.75f, scores.unionF1)
        assertEquals(21, scores.imagesScored)
        assertEquals(1, scores.imagesFailed)
        assertEquals(1, scores.groups.size)
        assertEquals("1girl, solo", scores.groups.first().prompt)
        assertEquals(3, scores.groups.first().images)
        assertEquals(0.88f, scores.groups.first().unionF1)
        assertEquals("solo", scores.topFalsePositives.first().tag)
        assertEquals(5, scores.topFalsePositives.first().count)
        assertEquals("long hair", scores.topFalseNegatives.first().tag)
        // Its own rendered image carries the set/repeat name a thumbnail reads the Pn badge from.
        assertEquals(0, com.acite.axlranko.pages.components.generatedSampleItems(parsed).first().setIndex)
    }

    @Test
    fun aRunningEvaluationHasNoScores() {
        val raw = """
            {
              "id": "rein_s003050_evaluate_gen_20261001_120000",
              "state": "running",
              "mode": "evaluate",
              "phase": "rendering",
              "depth": 12,
              "threshold": 0.35,
              "images_done": 3,
              "total_images": 8,
              "scores": null
            }
        """.trimIndent()
        val parsed = json.decodeFromString(
            com.acite.axlranko.model.GeneratedSampleJob.serializer(),
            raw,
        )
        assertEquals("rendering", parsed.phase)
        assertEquals(3, parsed.imagesDone)
        assertEquals(8, parsed.totalImages)
        assertEquals(null, parsed.scores)
        assertTrue(parsed.files.isEmpty())
        assertTrue(parsed.categories.isEmpty())
        assertEquals("", parsed.configSource)
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
              "engine": "pixai-tagger-v1.0",
              "categories": ["general", "rating"],
              "threshold": 0.35,
              "thresholds": {"general": 0.35},
              "provider": "pixai-tagger-v1.0 on cuda:0",
              "device": "cuda:0",
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
        assertEquals("pixai-tagger-v1.0 on cuda:0", parsed.provider)
        assertEquals("pixai-tagger-v1.0", parsed.engine)
        assertEquals("cuda:0", parsed.device)
        assertEquals(listOf("general", "rating"), parsed.categories)
        assertEquals(4, parsed.processed)
        assertEquals(0, parsed.failed)
        assertEquals(1.25f, parsed.seconds)
    }

    @Test
    fun anOlderDatasetTagReplyStillParses() {
        // The reply before the tagger grew categories and an engine name carried neither field.
        val raw = """{"directory": "/tmp/alice", "processed": 2, "total": 2}"""
        val parsed = json.decodeFromString(DatasetTagResult.serializer(), raw)
        assertEquals("", parsed.engine)
        assertEquals("", parsed.device)
        assertEquals(emptyList(), parsed.categories)
        assertEquals(2, parsed.processed)
    }

    @Test
    fun taggerInfoParses() {
        val raw = """
            {
              "available": true,
              "engine": "pixai-tagger-v1.0",
              "model": "pixai-labs/pixai-tagger-v1.0",
              "model_path": "/cache/snapshots/9fe10ad",
              "cache_dir": "/repo/tagger2/miopen_cache",
              "categories": [
                {"key": "general", "count": 15043, "calibrated": 0.17},
                {"key": "rating", "count": 4, "calibrated": 0.41}
              ],
              "default_categories": ["general"],
              "reason": ""
            }
        """.trimIndent()
        val parsed = json.decodeFromString(TaggerInfoResult.serializer(), raw)
        assertTrue(parsed.available)
        assertEquals("pixai-tagger-v1.0", parsed.engine)
        assertEquals("/repo/tagger2/miopen_cache", parsed.cacheDir)
        assertEquals(listOf("general"), parsed.defaultCategories)
        assertEquals(2, parsed.categories.size)
        assertEquals(0.17f, parsed.categories[0].calibrated)
        assertEquals(15043, parsed.categories[0].count)
    }

    @Test
    fun anUnavailableTaggerInfoParses() {
        val raw = """{"available": false, "reason": "no model", "categories": []}"""
        val parsed = json.decodeFromString(TaggerInfoResult.serializer(), raw)
        assertFalse(parsed.available)
        assertEquals("no model", parsed.reason)
        assertEquals(emptyList(), parsed.categories)
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
    fun datasetTagRequestCarriesTheCategories() {
        val encoded = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 12,
                method = "dataset_tag",
                params = buildJsonObject {
                    put("directory", "/tmp/alice")
                    put("threshold", 0.4)
                    putJsonArray("categories") { listOf("general", "rating").forEach { add(JsonPrimitive(it)) } }
                },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        val categories = decoded.params["categories"] as JsonArray
        assertEquals(listOf("general", "rating"), categories.map { it.jsonPrimitive.content })
    }

    @Test
    fun datasetTagRequestCarriesThePartialTags() {
        val encoded = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 13,
                method = "dataset_tag",
                params = buildJsonObject {
                    put("directory", "/tmp/alice")
                    put("threshold", 0.6)
                    putJsonArray("only_tags") { listOf("anal", "pussy").forEach { add(JsonPrimitive(it)) } }
                },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), encoded)
        val onlyTags = decoded.params["only_tags"] as JsonArray
        assertEquals(listOf("anal", "pussy"), onlyTags.map { it.jsonPrimitive.content })
    }

    @Test
    fun aPartialTaggingResultParses() {
        val raw = """
            {
              "directory": "/tmp/alice",
              "engine": "pixai-tagger-v1.0",
              "mode": "partial",
              "only_tags": ["anal", "pussy"],
              "categories": [],
              "threshold": 0.6,
              "provider": "pixai-tagger-v1.0 on cuda:0",
              "device": "cuda:0",
              "total": 100,
              "processed": 15,
              "failed": 0,
              "seconds": 12.5,
              "errors": [],
              "added": {"anal": 12, "pussy": 3},
              "unmatched": ["nonexistent tag"]
            }
        """.trimIndent()
        val parsed = json.decodeFromString(DatasetTagResult.serializer(), raw)
        assertEquals("partial", parsed.mode)
        assertEquals(listOf("anal", "pussy"), parsed.onlyTags)
        assertEquals(mapOf("anal" to 12, "pussy" to 3), parsed.added)
        assertEquals(listOf("nonexistent tag"), parsed.unmatched)
        assertEquals(15, parsed.processed)
    }

    @Test
    fun aFullTaggingResultWithoutThePartialFieldsStillParses() {
        val raw = """{"directory": "/tmp/alice", "processed": 2, "total": 2, "categories": ["general"]}"""
        val parsed = json.decodeFromString(DatasetTagResult.serializer(), raw)
        assertEquals("full", parsed.mode)
        assertEquals(emptyMap(), parsed.added)
        assertEquals(emptyList(), parsed.unmatched)
        assertEquals(emptyList(), parsed.onlyTags)
    }

    @Test
    fun errorEnvelopeParses() {
        val raw = """{"id": 3, "ok": false, "error": "unknown method: generate"}"""
        val parsed = json.decodeFromString(IpcResponse.serializer(), raw)
        assertEquals(3, parsed.id)
        assertTrue(!parsed.ok)
        assertEquals("unknown method: generate", parsed.error)
    }

    @Test
    fun anEvaluationWithAScoredTagSelectionParses() {
        val raw =
            """
            {
              "id": "evaluate_gen_1",
              "state": "done",
              "mode": "evaluate",
              "step": 3050,
              "depth": 12,
              "threshold": 0.35,
              "categories": ["general"],
              "tags": ["anal", "pussy"],
              "config_source": "/logs/rein_20260911_120000/config.toml",
              "scores": {
                "tp": 40, "fp": 12, "fn": 24,
                "precision": 0.769, "recall": 0.625, "f1": 0.689,
                "union_tp": 30, "union_fp": 8, "union_fn": 12,
                "union_precision": 0.789, "union_recall": 0.714, "union_f1": 0.75,
                "images_scored": 12, "images_failed": 1, "images_skipped": 0,
                "tags": ["anal", "pussy"]
              }
            }
            """.trimIndent()
        val parsed = json.decodeFromString(GeneratedSampleJob.serializer(), raw)
        assertEquals(listOf("anal", "pussy"), parsed.tags)
        assertEquals(listOf("anal", "pussy"), parsed.scores?.tags)
        assertEquals(0.625f, parsed.scores?.recall)
        // A record from before the selection existed still parses, with an empty list.
        val older = json.decodeFromString(
            GeneratedSampleJob.serializer(),
            """{"id": "evaluate_gen_2", "state": "done", "mode": "evaluate", "scores": {"recall": 0.5}}""",
        )
        assertEquals(emptyList(), older.tags)
        assertEquals(emptyList(), older.scores?.tags)
    }

    @Test
    fun evaluationPromptsPayloadParses() {
        val raw =
            """
            {
              "run_id": "rein_20260911_120000",
              "output_name": "rein",
              "checkpoint": "/out/rein/rein_s000100/rein.safetensors",
              "config_source": "/logs/rein_20260911_120000/config.toml",
              "sample_sets": [{"prompt": "1girl, anal", "repeat": 2}],
              "tags": [
                {"tag": "1girl", "count": 6, "frequency": 100.0},
                {"tag": "anal", "count": 2, "frequency": 33.3333}
              ],
              "reason": ""
            }
            """.trimIndent()
        val parsed = json.decodeFromString(EvaluationPromptsResponse.serializer(), raw)
        assertEquals("rein_20260911_120000", parsed.runId)
        assertEquals(listOf("1girl", "anal"), parsed.tags.map { it.tag })
        assertEquals(2, parsed.tags[1].count)
        assertEquals(33.3333f, parsed.tags[1].frequency)
        assertEquals("", parsed.reason)
    }

    @Test
    fun anUnusablePromptPayloadParsesWithItsReason() {
        val raw = """{"run_id": "rein_x", "tags": [], "reason": "validation.samples is empty"}"""
        val parsed = json.decodeFromString(EvaluationPromptsResponse.serializer(), raw)
        assertEquals(emptyList(), parsed.tags)
        assertEquals("validation.samples is empty", parsed.reason)
    }

    @Test
    fun theEvaluationRequestsCarryTheirSelection() {
        val evaluate = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 21,
                method = "evaluate_checkpoint",
                params = buildJsonObject {
                    put("checkpoint", "/out/rein/rein_s000100/rein.safetensors")
                    put("depth", 12)
                    put("threshold", 0.35)
                    putJsonArray("tags") { listOf("anal", "pussy").forEach { add(JsonPrimitive(it)) } }
                },
            ),
        )
        val decoded = json.decodeFromString(IpcRequest.serializer(), evaluate)
        assertEquals(
            listOf("anal", "pussy"),
            (decoded.params["tags"] as JsonArray).map { it.jsonPrimitive.content },
        )

        val prompts = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 22,
                method = "evaluation_prompts",
                params = buildJsonObject {
                    put("checkpoint", "/out/rein/rein_s000100/rein.safetensors")
                    put("run_id", "rein_20260911_120000")
                },
            ),
        )
        val decodedPrompts = json.decodeFromString(IpcRequest.serializer(), prompts)
        assertEquals("evaluation_prompts", decodedPrompts.method)
        assertEquals("rein_20260911_120000", decodedPrompts.params["run_id"]?.jsonPrimitive?.content)
    }
}

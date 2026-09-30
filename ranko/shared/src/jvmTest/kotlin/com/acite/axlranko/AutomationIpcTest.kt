package com.acite.axlranko

import com.acite.axlranko.data.AutomationConfigResult
import com.acite.axlranko.data.AutomationJobDetail
import com.acite.axlranko.data.AutomationJobListResult
import com.acite.axlranko.data.AutomationWorkflow
import com.acite.axlranko.data.ComfyDiscovery
import com.acite.axlranko.data.PromptMatrixDocument
import com.acite.axlranko.data.PromptProfileDocument
import com.acite.axlranko.data.PromptProfileListResult
import com.acite.axlranko.data.PromptProfileSaveResult
import com.acite.axlranko.data.imageSeedAt
import com.acite.axlranko.data.jobPassProgress
import com.acite.axlranko.model.AutomationSettingsDraft
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.GalleryImageRef
import com.acite.axlranko.model.JobFilter
import com.acite.axlranko.model.PromptExtendDraft
import com.acite.axlranko.model.PromptProfileItem
import com.acite.axlranko.model.jobElapsedSeconds
import com.acite.axlranko.model.jobImagePathFor
import com.acite.axlranko.model.jobProgress
import com.acite.axlranko.model.promptExportFileName
import com.acite.axlranko.model.promptExportText
import com.acite.axlranko.model.promptProfileLabel
import com.acite.axlranko.pages.components.automation.promptEntryFor
import com.acite.axlranko.pages.components.automation.uiText
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.prompt.PromptMode
import com.acite.axlranko.prompt.defaultSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import com.acite.axlranko.data.IpcRequest

/** The Automation page's payloads and its pure UI helpers. */
class AutomationIpcTest {
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
        isLenient = true
        encodeDefaults = true
    }

    @Test
    fun theMatrixDocumentParses() {
        val raw = """{"path": "/repo/input_matrix.txt", "text": "POSES:\nmissionary : both\n"}"""
        val parsed = json.decodeFromString(PromptMatrixDocument.serializer(), raw)
        assertEquals("/repo/input_matrix.txt", parsed.path)
        assertTrue(parsed.text.contains("missionary"))
    }

    @Test
    fun theProfileListParsesVersionsAndErrors() {
        val raw = """
            {
              "profiles": [
                { "name": "Kirika", "version": 3, "modified": 1790516882297, "size": 3489, "error": null },
                { "name": "General2", "version": 2, "modified": 1790507479798, "size": 3075 },
                { "name": "Broken", "version": null, "modified": 1, "size": 2, "error": "invalid JSON (x)" },
                { "name": "Old", "modified": 3, "size": 4 }
              ]
            }
        """.trimIndent()
        val parsed = json.decodeFromString(PromptProfileListResult.serializer(), raw)
        assertEquals(4, parsed.profiles.size)
        assertEquals(3, parsed.profiles[0].version)
        assertEquals(2, parsed.profiles[1].version)
        assertNull(parsed.profiles[1].error)
        assertEquals("invalid JSON (x)", parsed.profiles[2].error)
        assertNull(parsed.profiles[3].version)
    }

    @Test
    fun profileDocumentsAndSaveResultsParse() {
        val document = json.decodeFromString(
            PromptProfileDocument.serializer(),
            """{"name": "Kirika", "text": "{\"version\": 3}"}""",
        )
        assertEquals("Kirika", document.name)
        val saved = json.decodeFromString(
            PromptProfileSaveResult.serializer(),
            """{"name": "Kirika", "path": "/repo/prompt_profiles/Kirika.json"}""",
        )
        assertEquals("/repo/prompt_profiles/Kirika.json", saved.path)
    }

    @Test
    fun unknownFieldsInANewerHelperDoNotBreakTheParse() {
        val raw = """{"path": "/repo/input_matrix.txt", "text": "x", "sha1": "abc"}"""
        val parsed = json.decodeFromString(PromptMatrixDocument.serializer(), raw)
        assertEquals("x", parsed.text)
    }

    @Test
    fun anExportNameCarriesTheSeedAndCount() {
        assertEquals("prompts_seed1234_20.txt", promptExportFileName(1234L, 20))
        assertEquals("prompts_random_3.txt", promptExportFileName(null, 3))
    }

    @Test
    fun exportTextIsOneLinePerPrompt() {
        assertEquals("a\nb\n", promptExportText(listOf("a", "b")))
    }

    @Test
    fun aProfileLabelShowsItsFormatVersion() {
        assertEquals("Kirika  ·  v3", promptProfileLabel(PromptProfileItem("Kirika", version = 3)))
        assertEquals("Old  ·  ?", promptProfileLabel(PromptProfileItem("Old")))
    }

    @Test
    fun thePageChromeFollowsTheLanguageSwitch() {
        val keys = listOf(
            "language",
            "section_prompts",
            "section_comfy",
            "section_gallery",
            "placeholder_comfy_1",
            "placeholder_gallery_1",
            "matrix",
            "reload",
            "refresh",
            "load",
            "delete",
            "wizard",
            "manifest",
            "mode_count",
            "results",
            "results_hint",
            "generate",
            "copy_all",
            "download",
            "send_to_batch",
            "clear",
            "results_empty",
            "batch_queue",
            "profiles_empty",
            "active_profile",
            "done",
            "save",
            "cancel",
            "back",
            "next",
            "exposure_empty",
            "pool_any",
            "no_entries",
            "matrix_missing",
            "seed_hint",
            "face_candidates",
            "ratio_split",
            "copied_one",
            "copied_many",
            "saved_to",
            "sent_to_batch",
            "profile_deleted",
            "need_name",
            "save_failed",
            "load_failed",
            "delete_failed",
            "matrix_failed",
            "profiles_failed",
            "matrix_unavailable",
            "generate_failed",
        )
        keys.forEach { key ->
            val chinese = uiText(PromptLang.Chinese, key)
            val english = uiText(PromptLang.English, key)
            assertTrue(chinese.isNotBlank() && english.isNotBlank(), key)
            assertTrue(chinese != english, "$key: both languages read the same")
            assertTrue(chinese != key, "$key: no Chinese entry")
            assertTrue(english != key, "$key: no English entry")
        }
        assertEquals("missing_key", uiText(PromptLang.English, "missing_key"))
    }

    @Test
    fun thePageOpensInEnglish() {
        assertEquals(PromptLang.English, AutomationUiState().language)
        assertEquals(PromptLang.Chinese, AutomationUiState(language = PromptLang.Chinese).language)
    }

    @Test
    fun theConfigDiscoveryAndWorkflowPayloadsParse() {
        val config = json.decodeFromString(
            AutomationConfigResult.serializer(),
            """
            {
              "settings": {"server": "http://127.0.0.1:8188", "workflow": "/repo/automation/workflows/Kirika.json",
                           "positive_node": "215", "count": 2, "poll": 0.5, "output_dir": "/repo/automation/jobs"},
              "default_output_dir": "/repo/automation/jobs",
              "paths": {"root": "/repo/automation", "workflows": "/repo/automation/workflows",
                        "prompts": "/repo/automation/prompts", "jobs": "/repo/automation/jobs"}
            }
            """.trimIndent(),
        )
        assertEquals(2, config.settings.count)
        assertEquals("215", config.settings.positiveNode)
        assertEquals("/repo/automation/jobs", config.paths.jobs)
        assertEquals(2, AutomationSettingsDraft.of(config.settings).let { it.count.toInt() })

        val discovery = json.decodeFromString(
            ComfyDiscovery.serializer(),
            """{"found": true, "url": "http://127.0.0.1:8188", "version": "0.35.0",
                "queue_running": 0, "queue_pending": 1,
                "checked": [{"url": "http://127.0.0.1:9", "ok": false, "reason": "nope"}], "probed_all": true}""",
        )
        assertTrue(discovery.found)
        assertEquals("0.35.0", discovery.version)
        assertEquals(1, discovery.queuePending)
        assertEquals("nope", discovery.checked.first().reason)

        val workflow = json.decodeFromString(
            AutomationWorkflow.serializer(),
            """
            {
              "name": "Kirika", "path": "/repo/automation/workflows/Kirika.json", "valid": true, "error": null,
              "node_count": 25, "save_image_nodes": ["222"], "batch_size_nodes": ["217", "198:196"],
              "text_nodes": [{"id": "215", "class_type": "CLIPTextEncode", "text": "(kirika_character:1.1), 1girl"},
                             {"id": "198:259", "class_type": "CLIPTextEncode", "text": "best quality"}],
              "positive_node": "215", "positive_node_guessed": true,
              "missing_models": [{"node": "198:41", "class_type": "UpscaleModelLoader", "input": "model_name",
                                  "value": "missing.pth"}]
            }
            """.trimIndent(),
        )
        assertEquals(25, workflow.nodeCount)
        assertEquals(2, workflow.textNodes.size)
        assertEquals("198:259", workflow.textNodes[1].id)
        assertTrue(workflow.positiveNodeGuessed)
        assertEquals("missing.pth", workflow.missingModels.first().value)
    }

    @Test
    fun theJobPayloadsParse() {
        val listed = json.decodeFromString(
            AutomationJobListResult.serializer(),
            """
            {"jobs": [
              {"id": "Kirika_20260928_101500", "state": "running", "created_at": 1.0, "started_at": 2.0,
               "updated_at": 3.0, "finished_at": null, "total": 4, "done": 1, "failed": 0, "images": 2,
               "preview_paths": ["/repo/automation/jobs/Kirika_20260928_101500/images/p0001_01.png"],
               "workflow": "/repo/automation/workflows/Kirika.json", "output_dir": "/repo/automation/jobs",
               "positive_node": "215", "count": 2, "comfy_url": "http://127.0.0.1:8188", "error": null}
            ]}
            """.trimIndent(),
        )
        val job = listed.jobs.single()
        assertEquals(4, job.total)
        assertEquals("/repo/automation/jobs", job.outputDir)
        assertEquals(0.25f, jobProgress(job))
        assertEquals(1.0, jobElapsedSeconds(job, nowMillis = 3_000))

        val detail = json.decodeFromString(
            AutomationJobDetail.serializer(),
            """
            {"id": "Kirika_20260928_101500", "state": "done", "output_dir": "/repo/automation/jobs",
             "comfy_url": "http://127.0.0.1:8188", "error": null, "log_tail": "[automation] done",
             "summary": {"id": "Kirika_20260928_101500", "state": "done", "total": 1, "done": 1, "failed": 0, "images": 2},
             "prompts": [{"index": 0, "text": "alpha", "state": "done", "seed": 1234, "prompt_id": "stub-1",
                          "images": ["p0001_01.png", "p0001_02.png"], "error": null}]}
            """.trimIndent(),
        )
        assertEquals(2, detail.prompts.single().images.size)
        assertEquals(1234L, detail.prompts.single().seed)
        assertEquals("[automation] done", detail.logTail)
        assertEquals(
            "/repo/automation/jobs/Kirika_20260928_101500/images/p0001_01.png",
            jobImagePathFor(detail.id, detail.outputDir, detail.prompts.single().images.first()),
        )

        val state = AutomationUiState(
            jobs = listed.jobs,
            jobDetail = detail,
            jobFilter = JobFilter.Failed,
        )
        assertEquals(emptyList(), state.visibleJobs)
        assertEquals(listOf(detail.prompts.single().images.size.toString()), listOf("2"))
        assertEquals(1, AutomationUiState(jobs = listed.jobs, jobFilter = JobFilter.Running).visibleJobs.size)
        assertEquals(1, AutomationUiState(jobs = listed.jobs, jobSearch = "101500").visibleJobs.size)
        assertEquals(0, AutomationUiState(jobs = listed.jobs, jobSearch = "nope").visibleJobs.size)
    }

    @Test
    fun theGalleryImageActionsCarryTheirTarget() {
        val detail = json.decodeFromString(
            AutomationJobDetail.serializer(),
            """
            {"id": "Kirika_20260928_101500", "state": "done", "output_dir": "/repo/automation/jobs",
             "prompts": [{"index": 0, "text": "alpha", "state": "done", "seed": 99,
                          "images": ["p0001_01.png", "p0001_02.png"], "image_seeds": [11, 12]}]}
            """.trimIndent(),
        )
        val prompt = detail.prompts.single()
        assertEquals(listOf(11L, 12L), prompt.imageSeeds)
        assertEquals(11L, imageSeedAt(prompt, 0))
        assertEquals(12L, imageSeedAt(prompt, 1))
        // A record from before the Gallery could redraw one image has no list: its single seed is
        // what those images were drawn with, and what the page showed for them.
        val older = prompt.copy(imageSeeds = emptyList())
        assertEquals(99L, imageSeedAt(older, 0))
        assertEquals(99L, imageSeedAt(older, 1))
        assertEquals(null, imageSeedAt(prompt, 2), "a longer list has no seed for a name it lacks")

        // Every reply is the job's whole detail, so one action needs one round trip.
        assertEquals("Kirika_20260928_101500", detail.id)
        val ref = GalleryImageRef(detail.id, 0, "p0001_02.png")
        // The position the API takes (its stored `index`), which the row shows as `#1`.
        assertEquals(0, promptEntryFor(detail, ref.image)?.first)
        assertEquals("alpha", promptEntryFor(detail, ref.image)?.second?.text)
        assertEquals(null, promptEntryFor(detail, "p0009_01.png"))
    }

    @Test
    fun theImageActionRequestsRoundTrip() {
        val delete = json.encodeToString(
            IpcRequest.serializer(),
            IpcRequest(
                id = 31,
                method = "automation_image_delete",
                params = buildJsonObject {
                    put("id", "Kirika_20260928_101500")
                    put("image", "p0001_01.png")
                },
            ),
        )
        val decodedDelete = json.decodeFromString(IpcRequest.serializer(), delete)
        assertEquals("automation_image_delete", decodedDelete.method)
        assertEquals("p0001_01.png", decodedDelete.params["image"]?.jsonPrimitive?.content)

        val extend = json.decodeFromString(
            IpcRequest.serializer(),
            json.encodeToString(
                IpcRequest.serializer(),
                IpcRequest(
                    id = 32,
                    method = "automation_prompt_extend",
                    params = buildJsonObject {
                        put("id", "Kirika_20260928_101500")
                        put("prompt_index", 3)
                        put("count", 4)
                    },
                ),
            ),
        )
        assertEquals("4", extend.params["count"]?.jsonPrimitive?.content)
        assertEquals("3", extend.params["prompt_index"]?.jsonPrimitive?.content)

        val edit = json.decodeFromString(
            IpcRequest.serializer(),
            json.encodeToString(
                IpcRequest.serializer(),
                IpcRequest(
                    id = 33,
                    method = "automation_job_prompt_edit",
                    params = buildJsonObject {
                        put("id", "Kirika_20260928_101500")
                        put("prompt_index", 0)
                        put("text", "a new prompt")
                    },
                ),
            ),
        )
        assertEquals("a new prompt", edit.params["text"]?.jsonPrimitive?.content)
    }

    @Test
    fun aRunningPassParsesOnTheJobAndOnTheListRow() {
        val detail = json.decodeFromString(
            AutomationJobDetail.serializer(),
            """
            {"id": "Kirika_20260928_101500", "state": "running", "output_dir": "/repo/automation/jobs",
             "pass": {"mode": "append", "prompt_index": 1, "images_done": 2, "total_images": 4,
                      "image": "p0002_03.png"},
             "prompts": [{"index": 0, "text": "alpha", "state": "done", "seed": 99,
                          "images": ["p0001_01.png"], "image_seeds": [99]}]}
            """.trimIndent(),
        )
        val pass = detail.pass
        assertEquals("append", pass?.mode)
        assertEquals(1, pass?.promptIndex)
        assertEquals("p0002_03.png", pass?.image)
        assertEquals(0.5f, jobPassProgress(pass!!))
        assertEquals(0f, jobPassProgress(pass.copy(imagesDone = 0)))
        assertEquals(1f, jobPassProgress(pass.copy(imagesDone = 9)), "a counter that overshoots stays 0..1")
        assertEquals(0f, jobPassProgress(pass.copy(totalImages = 0)))

        // The job list carries the same block, which is what lets the row say what is running.
        val listed = json.decodeFromString(
            AutomationJobListResult.serializer(),
            """
            {"jobs": [{"id": "Kirika_20260928_101500", "state": "running", "total": 2, "done": 1,
                       "images": 1, "pass": {"mode": "image", "prompt_index": 0, "images_done": 0,
                                             "total_images": 1, "image": "p0001_01.png"}}]}
            """.trimIndent(),
        )
        assertEquals("image", listed.jobs.single().pass?.mode)
        assertEquals("p0001_01.png", listed.jobs.single().pass?.image)
        // A job with no pass in flight (every job until one is started) still parses.
        assertEquals(null, json.decodeFromString(
            AutomationJobListResult.serializer(),
            """{"jobs": [{"id": "old", "state": "done"}]}""",
        ).jobs.single().pass)
    }

    @Test
    fun theExtendDraftOnlySendsAUsableCount() {
        assertEquals(1, PromptExtendDraft("job", 0).images, "a fresh dialog starts at one")
        assertEquals(6, PromptExtendDraft("job", 0, count = "6").images)
        assertEquals(null, PromptExtendDraft("job", 0, count = "").images)
        assertEquals(null, PromptExtendDraft("job", 0, count = "0").images)
        assertEquals(null, PromptExtendDraft("job", 0, count = "17").images)
        assertEquals(null, PromptExtendDraft("job", 0, count = "many").images)
    }

    @Test
    fun theBatchPromptSourcesPickWhatTheUserChose() {
        val settings = AutomationSettingsDraft(
            server = "127.0.0.1:8188",
            workflow = "/w.json",
            positiveNode = "215",
            count = "3",
            poll = "0.25",
            outputDir = "/jobs",
        ).toSettings()
        assertEquals(3, settings.count)
        assertEquals(0.25, settings.poll)
        assertEquals("215", settings.positiveNode)
        assertEquals("1", AutomationSettingsDraft(count = "abc").toSettings().count.toString())
        assertEquals(0.5, AutomationSettingsDraft(poll = "x").toSettings().poll)
    }

    @Test
    fun theWizardRailFollowsWhatApplies() {
        val sfw = AutomationUiState(spec = defaultSpec())
        assertTrue(!sfw.applicablePageKeys.contains("family"))
        assertTrue(sfw.applicablePageKeys.contains("clothing"))
        assertEquals(sfw.applicablePageKeys.first(), sfw.currentPageKey)
        // The figure group applies everywhere; the two pussy groups are SEX only, like the rest of
        // the sex pages.
        assertTrue(sfw.applicablePageKeys.contains("figure"))
        assertTrue(!sfw.applicablePageKeys.contains("pussy_shape"))
        assertTrue(!sfw.applicablePageKeys.contains("pussy_hair"))

        val nsfw = AutomationUiState(spec = defaultSpec().also { it.mode = PromptMode.Nsfw })
        assertTrue(nsfw.applicablePageKeys.contains("figure"))
        assertTrue(!nsfw.applicablePageKeys.contains("pussy_shape"))
        assertTrue(!nsfw.applicablePageKeys.contains("pussy_hair"))

        val sex = AutomationUiState(
            spec = defaultSpec().also { it.mode = PromptMode.Sex },
        )
        assertTrue(sex.applicablePageKeys.contains("stages"))
        assertTrue(sex.applicablePageKeys.contains("figure"))
        assertTrue(sex.applicablePageKeys.contains("pussy_shape"))
        assertTrue(sex.applicablePageKeys.contains("pussy_hair"))

        val nude = AutomationUiState(
            spec = defaultSpec().also { it.exposure = listOf("nude") },
        )
        assertTrue(!nude.applicablePageKeys.contains("clothing"))
    }

    @Test
    fun theVisiblePageIndexClampsWhenPagesDisappear() {
        val state = AutomationUiState(
            spec = defaultSpec().also { it.mode = PromptMode.Sex },
            pageIndex = 12,
        )
        assertTrue(state.visiblePageIndex <= state.applicablePageKeys.size - 1)
        assertEquals(state.applicablePageKeys[state.visiblePageIndex], state.currentPageKey)

        val shrunk = state.copy(spec = defaultSpec())
        assertTrue(shrunk.visiblePageIndex <= shrunk.applicablePageKeys.size - 1)
    }
}

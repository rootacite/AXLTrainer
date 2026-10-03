package com.acite.axlranko.pages

import com.acite.axlranko.changelog.ChangelogEntry
import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.model.DatasetItem
import com.acite.axlranko.model.ImageItem
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.model.TagStat
import com.acite.axlranko.model.HardwareCpu
import com.acite.axlranko.model.HardwareGpu
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.TrainEncoding
import com.acite.axlranko.model.TrainSampling
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.model.TrainTrainingProgress
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.TrainDataDirForm
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class HomeSummariesTest {

    private fun sample(path: String, repeat: Int) =
        SampleItem(filename = path.substringAfterLast('/'), repeatIdx = repeat, path = path)

    @Test
    fun idleHasNoStepAndNoSamples() {
        val home = dashboardHome(TrainStatus(), emptyMap())
        assertEquals("Idle", home.statusLabel)
        assertEquals("Idle", home.detail)
        assertEquals(emptyList(), home.samplePaths)
    }

    @Test
    fun trainingShowsStepEpochAndAvgLoss() {
        val home = dashboardHome(
            TrainStatus(
                status = "training",
                training = TrainTrainingProgress(
                    step = 12,
                    totalSteps = 400,
                    epoch = 1,
                    epochs = 20,
                    avgLoss = 0.25f,
                ),
            ),
            emptyMap(),
        )
        assertEquals("Training", home.statusLabel)
        assertEquals("step 12/400 · epoch 1/20 · avg 0.2500", home.detail)
    }

    @Test
    fun trainingOmitsAvgWhenTheTrainerHasNone() {
        val home = dashboardHome(
            TrainStatus(
                status = "training",
                training = TrainTrainingProgress(step = 1, totalSteps = 10, epoch = 0, epochs = 2),
            ),
            emptyMap(),
        )
        assertEquals("step 1/10 · epoch 0/2", home.detail)
    }

    @Test
    fun encodingShowsImageProgress() {
        val home = dashboardHome(
            TrainStatus(status = "encoding", encoding = TrainEncoding(current = 3, total = 10)),
            emptyMap(),
        )
        assertEquals("Encoding", home.statusLabel)
        assertEquals("3/10 images", home.detail)
    }

    @Test
    fun samplingAddsTheRepeatBesideTheStep() {
        val home = dashboardHome(
            TrainStatus(
                status = "sampling",
                training = TrainTrainingProgress(step = 100, totalSteps = 400, avgLoss = 0.5f),
                sampling = TrainSampling(repeat = 1, repeats = 3),
            ),
            emptyMap(),
        )
        assertEquals("Sampling", home.statusLabel)
        assertEquals("step 100/400 · repeat 1/3 · avg 0.5000", home.detail)
    }

    @Test
    fun samplesStayOnTheNewestStepAndStopAtFour() {
        val samples = mapOf(
            "10" to listOf(sample("/old.png", 0)),
            "12" to (0 until 6).map { sample("/s$it.png", it) },
            "-1" to listOf(sample("/loose.png", 0)),
        )
        assertEquals(
            listOf("/s2.png", "/s3.png", "/s4.png", "/s5.png"),
            latestSamplePaths(samples),
        )
    }

    @Test
    fun aRunningAutomationJobBeatsAnEarlierDoneOne() {
        val home = automationHome(
            listOf(
                AutomationJobSummary(
                    id = "old",
                    state = "done",
                    done = 4,
                    total = 4,
                    workflow = "/wf/old.json",
                ),
                AutomationJobSummary(
                    id = "now",
                    state = "running",
                    done = 1,
                    total = 8,
                    failed = 2,
                    workflow = "/wf/batch.json",
                ),
            ),
        )
        assertEquals("running", home.statusKey)
        assertEquals("Running", home.statusLabel)
        assertEquals("batch.json · 1/8 · 2 failed", home.detail)
    }

    @Test
    fun automationImagesAreTheNewestOnes() {
        val home = automationHome(
            listOf(
                AutomationJobSummary(
                    id = "now",
                    state = "done",
                    previewPaths = listOf("/a.png", "/b.png"),
                    recentPaths = listOf("/w.png", "/x.png", "/y.png", "/z.png", "/too.png"),
                ),
            ),
        )
        assertEquals(listOf("/w.png", "/x.png", "/y.png", "/z.png", "/too.png"), home.imagePaths)
    }

    @Test
    fun noAutomationJobsSaysSo() {
        val home = automationHome(emptyList())
        assertEquals("Idle", home.statusLabel)
        assertEquals("No jobs", home.detail)
    }

    @Test
    fun utilsNamesTheBasicConfig() {
        val lines = utilsHome(
            TrainingConfigForm(
                outputName = "hana",
                baseModelVersion = "sdxl",
                networkDim = "16",
                learningRate = "1e-4",
                epoch = "20",
                trainDataDirs = listOf(
                    TrainDataDirForm(path = "/data/a"),
                    TrainDataDirForm(path = "/data/b"),
                    TrainDataDirForm(),
                ),
            ),
            configLoaded = true,
            helperLine = null,
        )
        assertEquals(
            listOf(
                HomeFact("Output", "hana"),
                HomeFact("Model", "sdxl"),
                HomeFact("Network", "standard · dim 16 · α —"),
                HomeFact("Learning rate", "1e-4"),
                HomeFact("Epochs", "20"),
                HomeFact("Batch", "— × —"),
                HomeFact("Resolution", "—"),
                HomeFact("Save every", "— steps"),
                HomeFact("Sampling", "on"),
                HomeFact("Precision", "bf16"),
                HomeFact("Folders", "2"),
                HomeFact("Seed", "—"),
            ),
            lines,
        )
    }

    @Test
    fun utilsWithoutAConfigShowsTheHelper() {
        assertEquals(
            listOf(HomeFact("Helper", "127.0.0.1:18765 · disconnected")),
            utilsHome(TrainingConfigForm(), configLoaded = false, helperLine = "127.0.0.1:18765 · disconnected"),
        )
        assertEquals(
            listOf(HomeFact("Helper", "not loaded")),
            utilsHome(TrainingConfigForm(), configLoaded = false, helperLine = null),
        )
    }

    @Test
    fun hardwareLineNamesGpuVramTempAndRam() {
        val line = homeHardwareLine(
            HardwareStatus(
                available = true,
                gpus = listOf(
                    HardwareGpu(
                        gpuUtilPct = 41.2,
                        tempEdgeC = 62.4,
                        memUsedBytes = (8.2 * 1024 * 1024 * 1024).toLong(),
                        memTotalBytes = 16L * 1024 * 1024 * 1024,
                    ),
                ),
                cpu = HardwareCpu(memUsedBytes = 54, memTotalBytes = 100),
            ),
        )
        assertEquals("GPU 41% · VRAM 8.2/16.0 GiB · 62 °C · RAM 54%", line)
    }

    @Test
    fun datasetThumbsTakeFromEveryFolder() {
        val picked = pickDatasetThumbs(
            listOf(listOf("a1", "a2", "a3"), listOf("b1"), emptyList()),
            limit = 4,
            random = Random(1),
        )
        assertEquals(4, picked.size)
        assertTrue(picked.any { it.startsWith("a") })
        assertTrue("b1" in picked)
    }

    @Test
    fun statisticsKeepsTheFiveMostCommonTags() {
        val items = listOf(
            DatasetItem("a", "/a.png", "/a.txt", "/a.mask.png", listOf("sky")),
            DatasetItem("b", "/b.png", "/b.txt", "/b.mask.png", emptyList()),
        )
        val tags = (1..6).map { TagStat("t$it", count = it, frequency = it.toFloat()) }
        val lines = statisticsHome("/data/set", imageCount = 2, items = items, tagStats = tags, loading = false, error = null)
        assertEquals("/set".substringAfterLast('/'), "set")
        assertEquals(listOf("set", "2 images · 1 captioned"), lines)
        assertEquals(listOf("t6", "t5", "t4", "t3", "t2", "t1"), homeTagBars(tags).map { it.tag })
    }

    @Test
    fun aFeatSubjectDropsTheKindBox() {
        val entry = ChangelogEntry("abc", "2026-10-03", "[Feat] Home page")
        assertEquals("Feat", entry.kind)
        assertEquals("Home page", entry.title)
    }

    @Test
    fun statisticsReportsAScanErrorAndAScanInFlight() {
        assertEquals(listOf("boom"), statisticsHome("", 0, emptyList(), emptyList(), loading = false, error = "boom"))
        assertEquals(listOf("Scanning…"), statisticsHome("", 0, emptyList(), emptyList(), loading = true, error = null))
    }

    @Test
    fun imagesCountSidecarMasksAndUnsavedDrafts() {
        val clean = ImageItem("/data/set", "a", "/data/set/a.png", "/data/set/a.txt", "sky", maskPath = "")
        val masked = clean.copy(stem = "b", imagePath = "/data/set/b.png", hasSidecarMask = true)
        val dirty = clean.copy(stem = "c", imagePath = "/data/set/c.png", draftTags = "sky, night")
        assertEquals(
            listOf("set", "3 images · 1 mask", "1 unsaved"),
            imagesHome("/data/set", listOf(clean, masked, dirty)),
        )
        assertEquals(
            listOf("No folder", "0 images · 0 masks", "No unsaved captions"),
            imagesHome("", emptyList()),
        )
    }
}

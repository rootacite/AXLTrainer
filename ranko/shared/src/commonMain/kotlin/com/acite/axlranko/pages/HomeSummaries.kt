package com.acite.axlranko.pages

import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.model.DatasetItem
import com.acite.axlranko.model.HardwareCpu
import com.acite.axlranko.model.HardwareGpu
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.ImageItem
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.model.TagStat
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.util.formatFourDecimals
import kotlin.math.roundToInt
import kotlin.random.Random

/** How many of the newest step's samples the Dashboard card draws. */
internal const val HOME_SAMPLE_LIMIT = 4

/** How many tag bars the Statistics card draws. */
internal const val HOME_TAG_BARS = 8

/** Pool of an automation job's newest images. The card draws at most two rows of these. */
internal const val HOME_JOB_IMAGE_LIMIT = 24

/** Pool of dataset thumbs. The Images card draws at most two rows of these. */
internal const val HOME_DATASET_POOL = 24

data class DashboardHome(
    val statusKey: String,
    val statusLabel: String,
    val detail: String,
    val samplePaths: List<String>,
)

data class AutomationHome(
    val statusKey: String,
    val statusLabel: String,
    val detail: String,
    val imagePaths: List<String> = emptyList(),
)

/** Same words as the Dashboard control card. */
internal fun trainStatusLabel(status: String): String = when (status) {
    "starting" -> "Starting"
    "encoding" -> "Encoding"
    "training" -> "Training"
    "sampling" -> "Sampling"
    "pausing" -> "Pausing"
    "paused" -> "Paused"
    "resuming" -> "Resuming"
    "stopping" -> "Stopping"
    "finished" -> "Finished"
    "error" -> "Error"
    "gpu-out" -> "Offloading GPU"
    "gpu-in" -> "Reloading GPU"
    else -> "Idle"
}

/**
 * Newest numeric step, else the only key there is. At most [limit] paths, the later repeats of
 * that step. An older step is not pulled in to fill the row.
 */
internal fun latestSamplePaths(
    samples: Map<String, List<SampleItem>>,
    limit: Int = HOME_SAMPLE_LIMIT,
): List<String> {
    if (samples.isEmpty() || limit <= 0) return emptyList()
    val numeric = samples.keys.mapNotNull { key -> key.toIntOrNull()?.let { it to key } }
    val key = if (numeric.isNotEmpty()) {
        numeric.maxBy { it.first }.second
    } else {
        samples.keys.maxOrNull()
    }
    return key?.let { samples[it] }.orEmpty().takeLast(limit).map { it.path }
}

internal fun dashboardHome(
    status: TrainStatus,
    samples: Map<String, List<SampleItem>>,
): DashboardHome {
    val label = trainStatusLabel(status.status)
    val detail = when (status.status) {
        "encoding" -> encodingLine(status)
        "sampling" -> samplingLine(status)
        else -> trainingLine(status)
    }
    return DashboardHome(
        statusKey = status.status,
        statusLabel = label,
        detail = detail,
        samplePaths = latestSamplePaths(samples),
    )
}

private fun encodingLine(status: TrainStatus): String {
    val enc = status.encoding
    return if (enc.total > 0) "${enc.current}/${enc.total} images" else "Encoding"
}

private fun trainingLine(status: TrainStatus): String {
    val t = status.training
    if (t.totalSteps <= 0 && t.step <= 0) return "Idle"
    val epoch = if (t.epochs > 0) " · epoch ${t.epoch}/${t.epochs}" else ""
    val avg = t.avgLoss?.let { " · avg ${formatFourDecimals(it)}" } ?: ""
    return "step ${t.step}/${t.totalSteps}$epoch$avg"
}

private fun samplingLine(status: TrainStatus): String {
    val t = status.training
    val s = status.sampling
    val step = if (t.totalSteps > 0) "${t.step}/${t.totalSteps}" else "${t.step}"
    val rep = if (s.repeats > 0) " · repeat ${s.repeat}/${s.repeats}" else ""
    val avg = t.avgLoss?.let { " · avg ${formatFourDecimals(it)}" } ?: ""
    return "step $step$rep$avg"
}

/**
 * A running job wins. Otherwise the first entry, which `automation_job_list` returns newest first.
 */
internal fun automationHome(jobs: List<AutomationJobSummary>): AutomationHome {
    val job = jobs.firstOrNull { it.state == "running" } ?: jobs.firstOrNull()
        ?: return AutomationHome(statusKey = "idle", statusLabel = "Idle", detail = "No jobs")
    val name = job.workflow.substringAfterLast('/').ifBlank { job.id }
    val failed = if (job.failed > 0) " · ${job.failed} failed" else ""
    val labeled = trainStatusLabel(job.state)
    val images = job.recentPaths.ifEmpty { job.previewPaths }.takeLast(HOME_JOB_IMAGE_LIMIT)
    return AutomationHome(
        statusKey = job.state.ifBlank { "idle" },
        statusLabel = if (labeled == "Idle" && job.state.isNotBlank()) {
            job.state.replaceFirstChar { it.uppercase() }
        } else {
            labeled
        },
        detail = "$name · ${job.done}/${job.total}$failed",
        imagePaths = images,
    )
}

data class HomeFact(val name: String, val value: String)

/**
 * [configLoaded] is false until `config.toml` has been read. The web build shows the helper
 * endpoint in that gap; [helperLine] is null on desktop.
 */
internal fun utilsHome(
    form: TrainingConfigForm,
    configLoaded: Boolean,
    helperLine: String?,
): List<HomeFact> {
    if (!configLoaded) {
        return listOf(HomeFact("Helper", helperLine ?: "not loaded"))
    }
    val folders = form.trainDataDirs.count { it.path.isNotBlank() }
    val batch = form.trainBatchSize.ifBlank { "—" }
    val grad = form.gradientAccumulationSteps.ifBlank { "—" }
    return listOf(
        HomeFact("Output", form.outputName.ifBlank { "—" }),
        HomeFact("Model", form.baseModelVersion.ifBlank { "—" }),
        HomeFact("Network", networkLine(form)),
        HomeFact("Learning rate", form.learningRate.ifBlank { "—" }),
        HomeFact("Epochs", form.epoch.ifBlank { "—" }),
        HomeFact("Batch", "$batch × $grad"),
        HomeFact("Resolution", form.trainResolution.ifBlank { "—" }),
        HomeFact("Save every", "${form.saveEveryNSteps.ifBlank { "—" }} steps"),
        HomeFact("Sampling", if (form.samplingEnabled) "on" else "off"),
        HomeFact("Precision", form.mixedPrecision.ifBlank { "—" }),
        HomeFact("Folders", folders.toString()),
        HomeFact("Seed", form.seed.ifBlank { "—" }),
    )
}

private fun networkLine(form: TrainingConfigForm): String {
    val type = form.networkType.ifBlank { "—" }
    val dim = form.networkDim.ifBlank { "—" }
    val alpha = form.networkAlpha.ifBlank { "—" }
    return "$type · dim $dim · α $alpha"
}

/**
 * One line for the Dashboard card: GPU use, VRAM, GPU temperature, RAM use.
 * GPU temperature is the edge reading, then the single `temp_c` when edge is absent.
 */
internal fun homeHardwareLine(status: HardwareStatus): String {
    val gpu = status.gpus.firstOrNull()
    if (!status.available && gpu == null) return "Hardware unavailable"
    return listOf(
        "GPU ${pct(gpu?.gpuUtilPct)}",
        "VRAM ${vramLine(gpu)}",
        tempLine(gpu),
        "RAM ${ramPct(status.cpu)}",
    ).joinToString(" · ")
}

private fun pct(value: Double?): String {
    if (value == null || !value.isFinite()) return "—"
    return "${value.roundToInt()}%"
}

private fun tempLine(gpu: HardwareGpu?): String {
    val temp = gpu?.tempEdgeC ?: gpu?.tempC
    if (temp == null || !temp.isFinite()) return "—"
    return "${temp.roundToInt()} °C"
}

private fun vramLine(gpu: HardwareGpu?): String {
    val used = gpu?.memUsedBytes?.toDouble()
    val total = gpu?.memTotalBytes?.toDouble()
    if (used == null && total == null) return "—"
    val usedGiB = used?.div(BYTES_PER_GIB)
    val totalGiB = total?.div(BYTES_PER_GIB)
    return when {
        usedGiB != null && totalGiB != null -> "${oneDecimal(usedGiB)}/${oneDecimal(totalGiB)} GiB"
        usedGiB != null -> "${oneDecimal(usedGiB)} GiB"
        totalGiB != null -> "${oneDecimal(totalGiB)} GiB"
        else -> "—"
    }
}

private fun ramPct(cpu: HardwareCpu): String {
    val used = cpu.memUsedBytes ?: return "—"
    val total = cpu.memTotalBytes ?: return "—"
    if (total <= 0L) return "—"
    return "${((used.toDouble() / total.toDouble()) * 100.0).roundToInt()}%"
}

private fun oneDecimal(value: Double): String {
    val rounded = (value * 10.0).roundToInt() / 10.0
    val text = rounded.toString()
    return if ('.' in text) text else "$text.0"
}

private const val BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0

/**
 * Up to [limit] paths, round-robin across folders so a second dataset is not crowded out.
 * Each folder is shuffled with [random] first. Blank paths and empty folders are skipped.
 */
internal fun pickDatasetThumbs(
    folders: List<List<String>>,
    limit: Int,
    random: Random,
): List<String> {
    if (limit <= 0) return emptyList()
    val pools = folders.map { folder ->
        folder.filter { it.isNotBlank() }.shuffled(random).toMutableList()
    }.filter { it.isNotEmpty() }
    if (pools.isEmpty()) return emptyList()
    val out = ArrayList<String>(limit)
    var turn = 0
    while (out.size < limit && pools.any { it.isNotEmpty() }) {
        val pool = pools[turn % pools.size]
        if (pool.isNotEmpty()) out += pool.removeAt(0)
        turn++
    }
    return out
}

internal fun statisticsHome(
    dir: String,
    imageCount: Int,
    items: List<DatasetItem>,
    tagStats: List<TagStat>,
    loading: Boolean,
    error: String?,
): List<String> {
    if (error != null && imageCount == 0) return listOf(error)
    if (loading && imageCount == 0) return listOf("Scanning…")
    val captioned = items.count { it.tags.isNotEmpty() }
    return listOf(
        folderName(dir).ifBlank { "No folder" },
        "$imageCount images · $captioned captioned",
    )
}

/** The bars on the Statistics card, most common first. Not clickable. */
internal fun homeTagBars(tagStats: List<TagStat>, limit: Int = HOME_TAG_BARS): List<TagStat> =
    tagStats.sortedByDescending { it.count }.take(limit)

internal fun imagesHome(dir: String, items: List<ImageItem>): List<String> {
    val masks = items.count { it.hasSidecarMask }
    val drafts = items.count { it.isDirty }
    val maskWord = if (masks == 1) "mask" else "masks"
    return listOf(
        folderName(dir).ifBlank { "No folder" },
        "${items.size} images · $masks $maskWord",
        if (drafts > 0) "$drafts unsaved" else "No unsaved captions",
    )
}

private fun folderName(path: String): String =
    path.trimEnd('/').substringAfterLast('/').ifBlank { path.trim() }

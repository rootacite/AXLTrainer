package com.acite.axlranko.pages

import com.acite.axlranko.IoDispatcher
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.DpSize
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.ChartPickState
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.HardwareHistory
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.pages.components.JOB_ERROR
import com.acite.axlranko.pages.components.JOB_RUNNING
import com.acite.axlranko.pages.components.checkpointRows
import com.acite.axlranko.pages.components.checkpointsForRun
import com.acite.axlranko.pages.components.displayedRun
import com.acite.axlranko.pages.components.generateFormDefaults
import com.acite.axlranko.pages.components.generateFormError
import com.acite.axlranko.pages.components.nearestCheckpoint
import com.acite.axlranko.pages.components.sectionImages
import com.acite.axlranko.util.PathPicker
import com.acite.axlranko.util.checkpointSaveName
import com.acite.axlranko.util.ensureSafetensorsExtension
import com.acite.axlranko.util.formatBytes
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlin.time.Duration.Companion.milliseconds

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class DashboardScreenViewModel(
    private val ipc: TrainerIpcClient,
    private val pathPicker: PathPicker,
) : ViewModel() {

    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()

    private var pollingJob: Job? = null
    private var hardwareJob: Job? = null
    private var hardwareStep = 0
    private var entered = false

    /** Guards a second checkpoint scan while the first is still reading safetensors headers. */
    private var checkpointScanInFlight = false

    /** Jobs started in this session, which the panel highlights as new. */
    private val sessionJobIds = mutableSetOf<String>()
    private var generatedPollJob: Job? = null

    fun onEnter() {
        if (!entered) {
            entered = true
            startPolling()
            startHardwarePolling()
        } else {
            refreshNow()
        }
    }

    /**
     * Pins the dashboard to one run of the history list; `null` follows the current run.
     *
     * A pinned run brings its own samples, charts and checkpoints, so the previous run's
     * pick panel and preview are dropped rather than left pointing at another run's images.
     */
    fun selectRun(runId: String?) {
        val run = runId?.let { id -> _uiState.value.runs.firstOrNull { it.runId == id } }
        _uiState.update {
            it.copy(
                selectedRun = run,
                // Unpinning clears the id too: the run to follow is the next fetch's answer, and
                // a leftover id would keep the page on the run just unpinned until it lands.
                runId = run?.runId,
                chartPick = null,
                previewIndex = null,
                samples = emptyMap(),
                latestStats = JsonObject(emptyMap()),
                metrics = emptyMap(),
            )
        }
        refreshNow()
    }

    fun toggleAutoRefresh(enabled: Boolean) {
        _uiState.update { it.copy(autoRefresh = enabled) }
        if (enabled) {
            startPolling()
            startHardwarePolling()
        } else {
            pollingJob?.cancel()
            hardwareJob?.cancel()
        }
    }

    fun setSmoothing(value: Float) {
        _uiState.update { it.copy(smoothing = value.coerceIn(0f, 0.99f)) }
    }

    fun setChartStroke(value: Float) {
        _uiState.update { it.copy(chartStroke = value.coerceIn(1f, 8f)) }
    }

    fun setSampleThumbSize(value: Float) {
        _uiState.update { it.copy(sampleThumbSize = value.coerceIn(80f, 360f)) }
    }

    fun openPreview(sample: SampleItem) {
        val index = currentPreviewList().indexOfFirst { it.path == sample.path }
        if (index >= 0) {
            _uiState.update { it.copy(previewIndex = index) }
        }
    }

    fun closePreview() {
        _uiState.update { it.copy(previewIndex = null) }
    }

    /**
     * Ctrl+click on the Avg Loss chart: resolve the checkpoint nearest to [step] from the list the
     * page already polls, then rescan in the background and re-resolve in place.
     */
    fun pickCheckpointAt(step: Float, anchor: Offset) {
        val previous = _uiState.value.chartPick
        val defaults = generateFormDefaults(_uiState.value.config)
        _uiState.update {
            it.copy(
                chartPick = ChartPickState(
                    step = step,
                    anchor = anchor,
                    checkpoint = nearestCheckpoint(it.checkpoints, step),
                    isLoading = it.checkpoints.isEmpty(),
                    // A prompt typed for an earlier pick survives; an untouched form is re-seeded
                    // from config.toml's sample settings.
                    prompt = previous?.prompt?.takeIf { text -> text.isNotBlank() } ?: defaults.prompt,
                    negativePrompt = previous?.negativePrompt?.takeIf { text -> text.isNotBlank() }
                        ?: defaults.negativePrompt,
                    cfg = previous?.cfg?.takeIf { text -> text.isNotBlank() } ?: defaults.cfg,
                    steps = previous?.steps?.takeIf { text -> text.isNotBlank() } ?: defaults.steps,
                    seed = previous?.seed?.takeIf { text -> text.isNotBlank() } ?: defaults.seed,
                    isFormOpen = previous?.isFormOpen ?: false,
                ),
            )
        }
        rescanCheckpoints()
        loadGeneratedSamples()
    }

    private fun rescanCheckpoints() {
        if (checkpointScanInFlight) return
        checkpointScanInFlight = true
        viewModelScope.launch {
            try {
                val shown = displayedRun(
                    _uiState.value.runs,
                    _uiState.value.selectedRun,
                    _uiState.value.runId,
                )
                val name = shown?.outputName ?: _uiState.value.selectedRun?.outputName
                val response = withContext(IoDispatcher) { ipc.listCheckpoints(name = name) }
                _uiState.update { state ->
                    val found = if (shown == null) {
                        emptyList()
                    } else {
                        checkpointsForRun(response.checkpoints, shown.runId)
                    }
                    val pick = state.chartPick ?: return@update state.copy(checkpoints = found)
                    state.copy(
                        checkpoints = found,
                        chartPick = pick.copy(
                            checkpoint = nearestCheckpoint(found, pick.step),
                            isLoading = false,
                            error = null,
                        ),
                    )
                }
            } catch (e: Exception) {
                _uiState.update { state ->
                    val pick = state.chartPick ?: return@update state
                    state.copy(
                        chartPick = pick.copy(isLoading = false, error = e.message ?: e.toString()),
                    )
                }
            } finally {
                checkpointScanInFlight = false
            }
        }
    }

    fun dismissChartPick() {
        generatedPollJob?.cancel()
        _uiState.update { it.copy(chartPick = null) }
    }

    /** Remembers a dragged panel size for the rest of the session; the UI clamps it to the window. */
    fun setChartPanelSize(size: DpSize) {
        _uiState.update { it.copy(chartPanelSize = size) }
    }

    /** Edits the panel's generate form; a change clears the message of the previous attempt. */
    fun updateChartPickForm(transform: ChartPickState.() -> ChartPickState) {
        _uiState.update { state ->
            val pick = state.chartPick ?: return@update state
            state.copy(chartPick = pick.transform().copy(formError = null))
        }
    }

    fun toggleGenerateForm() {
        updateChartPickForm { copy(isFormOpen = !isFormOpen) }
    }

    /**
     * Loads the run's generated samples from disk; a job still running keeps the poll loop alive.
     */
    fun loadGeneratedSamples() {
        val selected = _uiState.value.selectedRun
        val runId = selected?.runId ?: _uiState.value.runId
        if (runId.isNullOrBlank()) return
        viewModelScope.launch {
            val jobs = fetchGeneratedJobs(runId, selected?.outputName)
            _uiState.update { state -> state.copy(generatedJobs = jobs) }
            if (jobs.any { it.state == JOB_RUNNING }) startGeneratedPolling()
        }
    }

    /**
     * "Generate a sample with this checkpoint": validate the form, hand it to api.py (which spawns
     * the generator detached) and follow the job until it finishes.
     *
     * [rowStep] is the sample row the panel is showing — the checkpoint's own step, or the nearest
     * sampled step when that one has no images — so the new image lands in the row the user sees.
     */
    fun generateSample(rowStep: Int? = null) {
        val pick = _uiState.value.chartPick ?: return
        val checkpoint = pick.checkpoint ?: return
        if (pick.isGenerating) return

        val error = generateFormError(pick.prompt, pick.cfg, pick.steps, pick.seed)
        if (error != null) {
            updateChartPickForm { copy(formError = error) }
            return
        }

        updateChartPickForm { copy(isGenerating = true) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.generateSample(
                        checkpoint = checkpoint.path,
                        prompt = pick.prompt,
                        negativePrompt = pick.negativePrompt,
                        cfg = pick.cfg.trim().toFloat(),
                        steps = pick.steps.trim().toInt(),
                        seed = pick.seed.trim().toLong(),
                        step = rowStep ?: checkpoint.step,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                sessionJobIds += response.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        generatedJobs = (listOf(response.job) + state.generatedJobs).distinctBy { it.id },
                        generatedError = null,
                        chartPick = state.chartPick?.copy(isGenerating = false),
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update { state ->
                    state.copy(
                        generatedError = e.message ?: e.toString(),
                        chartPick = state.chartPick?.copy(isGenerating = false),
                    )
                }
            }
        }
    }

    /**
     * "Generate samples" for one checkpoint: the config's whole `[[validation.samples]]` list,
     * rendered detached into the run's `_samples/generated/`. Only offered while the GPU is free
     * (the run is paused, stopped or over) — api.py refuses it otherwise.
     */
    fun generateCheckpointSamples(checkpoint: CheckpointItem) {
        if (_uiState.value.isGeneratingCheckpoint != null) return
        _uiState.update { it.copy(isGeneratingCheckpoint = checkpoint.path, generatedError = null) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.generateCheckpointSamples(
                        checkpoint = checkpoint.path,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                sessionJobIds += response.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        isGeneratingCheckpoint = null,
                        generatedJobs = (listOf(response.job) + state.generatedJobs).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isGeneratingCheckpoint = null,
                        generatedError = e.message ?: e.toString(),
                    )
                }
            }
        }
    }

    /**
     * "Sample range": one detached job that renders the config's sample sets for every checkpoint
     * whose step is inside `fromStep..toStep`, oldest first.
     */
    fun startSampleBatch(fromStep: Int, toStep: Int) {
        if (_uiState.value.isStartingBatch) return
        _uiState.update { it.copy(isStartingBatch = true, batchError = null) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.generateCheckpointSamplesBatch(
                        fromStep = fromStep,
                        toStep = toStep,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                sessionJobIds += response.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        isStartingBatch = false,
                        generatedJobs = (listOf(response.job) + state.generatedJobs).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(isStartingBatch = false, batchError = e.message ?: e.toString())
                }
            }
        }
    }

    /** Ask the running generation (the batch one included) to stop; its images so far stay. */
    fun cancelGeneration(id: String? = null) {
        viewModelScope.launch {
            try {
                val response = withContext(IoDispatcher) { ipc.cancelGeneration(id) }
                _uiState.update { state ->
                    state.copy(
                        batchError = null,
                        generatedJobs = (
                            listOf(response.job) + state.generatedJobs.filterNot { it.id == response.job.id }
                            ).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update { it.copy(batchError = e.message ?: e.toString()) }
            }
        }
    }

    private suspend fun fetchGeneratedJobs(runId: String, name: String? = null): List<GeneratedSampleJob> =
        try {
            withContext(IoDispatcher) { ipc.listGeneratedSamples(name = name, runId = runId) }.jobs
        } catch (_: Exception) {
            emptyList()
        }

    /** 1.5 s poll while a generator works: it runs in its own process and reports through its job file. */
    private fun startGeneratedPolling() {
        if (generatedPollJob?.isActive == true) return
        generatedPollJob = viewModelScope.launch {
            while (isActive) {
                delay(GENERATED_POLL_MILLIS.milliseconds)
                val state = _uiState.value
                val selected = state.selectedRun
                val runId = selected?.runId ?: state.runId ?: return@launch
                val jobs = fetchGeneratedJobs(runId, selected?.outputName)
                if (jobs.isEmpty()) return@launch
                val failed = jobs.firstOrNull { it.state == JOB_ERROR }?.error
                _uiState.update { current ->
                    // A switch to another run under the poll must not inject the old run's jobs.
                    if ((current.selectedRun?.runId ?: current.runId) != runId) return@update current
                    current.copy(
                        generatedJobs = jobs,
                        generatedError = failed ?: current.generatedError,
                    )
                }
                if (jobs.none { it.state == JOB_RUNNING }) return@launch
            }
        }
    }


    /**
     * "Save As" for the picked checkpoint: pick a destination in the OS save dialog, then copy the
     * LoRA off the run directory. The copy is off-thread and reports progress.
     */
    fun saveCheckpointAs() {
        val pick = _uiState.value.chartPick ?: return
        val checkpoint = pick.checkpoint ?: return
        if (pick.isSaving) return

        val source = checkpoint.path
        viewModelScope.launch {
            val parent = source.substringBeforeLast('/', missingDelimiterValue = "")
            val chosen = pathPicker.saveFile(checkpointSaveName(checkpoint), parent) ?: return@launch
            val destName = ensureSafetensorsExtension(chosen.substringAfterLast('/'))
            val destParent = chosen.substringBeforeLast('/', missingDelimiterValue = "")
            val target = if (destParent.isEmpty()) destName else "$destParent/$destName"
            if (target != chosen) pathPicker.deleteEmptyPlaceholder(chosen)

            _uiState.update { state ->
                state.copy(
                    chartPick = state.chartPick?.copy(
                        isSaving = true,
                        saveProgress = 0f,
                        savedPath = null,
                        saveError = null,
                    ),
                )
            }
            try {
                val exported = withContext(IoDispatcher) {
                    ipc.checkpointExport(source, target)
                }
                _uiState.update { state ->
                    state.copy(
                        chartPick = state.chartPick?.copy(
                            isSaving = false,
                            saveProgress = null,
                            savedPath = "$target (${formatBytes(exported.bytes)})",
                        ),
                    )
                }
            } catch (e: Exception) {
                _uiState.update { state ->
                    state.copy(
                        chartPick = state.chartPick?.copy(
                            isSaving = false,
                            saveProgress = null,
                            saveError = e.message ?: e.toString(),
                        ),
                    )
                }
            }
        }
    }

    fun previewNext() = movePreview(1)

    fun previewPrev() = movePreview(-1)

    fun refreshNow() {
        viewModelScope.launch { fetchOnce() }
        viewModelScope.launch { fetchHardwareOnce() }
    }

    /**
     * Retunes the run in progress: the checkpoint cadence, the sampling switch, or both. The
     * trainer adopts the request at its next optimizer step, so the status still reports the old
     * values until then and the card says so.
     */
    fun applyTrainSettings(saveEveryNSteps: Int? = null, samplingEnabled: Boolean? = null) {
        if (_uiState.value.settingsInFlight) return
        _uiState.update { it.copy(settingsInFlight = true, settingsError = null) }
        viewModelScope.launch {
            try {
                val status = withContext(IoDispatcher) {
                    ipc.trainSettings(saveEveryNSteps = saveEveryNSteps, samplingEnabled = samplingEnabled)
                }
                _uiState.update { it.copy(settingsInFlight = false, trainStatus = status) }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(settingsInFlight = false, settingsError = e.message ?: e.toString())
                }
            }
        }
    }

    fun startTraining() = runTrainCommand { ipc.trainStart() }
    fun pauseTraining() = runTrainCommand(pending = "pause") { ipc.trainPause() }

    fun resumeTraining() = runTrainCommand(pending = "resume") { ipc.trainResume() }

    fun stopTraining() = runTrainCommand(pending = "stop") { ipc.trainStop() }

    fun resetTraining(deleteWeights: Boolean) = runTrainCommand(refreshAll = true) {
        ipc.trainReset(deleteWeights = deleteWeights)
    }

    private fun runTrainCommand(
        pending: String? = null,
        refreshAll: Boolean = false,
        block: suspend () -> TrainStatus,
    ) {
        viewModelScope.launch {
            _uiState.update { it.copy(commandInFlight = true, pendingCommand = pending ?: it.pendingCommand) }
            try {
                val status = withContext(IoDispatcher) { block() }
                _uiState.update {
                    it.copy(
                        commandInFlight = false,
                        trainStatus = status,
                        errorMessage = null,
                        pendingCommand = resolvedPending(pending ?: it.pendingCommand, status.status),
                    )
                }
                if (refreshAll) fetchOnce()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        commandInFlight = false,
                        pendingCommand = null,
                        errorMessage = e.message ?: e.toString(),
                    )
                }
            }
        }
    }

    fun retry() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null) }
            try {
                ipc.restart()
                hardwareStep = 0
                _uiState.update { it.copy(hardware = HardwareStatus(), hardwareHistory = HardwareHistory()) }
                fetchOnce()
                fetchHardwareOnce()
                if (_uiState.value.autoRefresh) {
                    startPolling()
                    startHardwarePolling()
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(isLoading = false, connected = false, errorMessage = e.message ?: e.toString())
                }
            }
        }
    }

    private fun startPolling() {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            while (isActive) {
                fetchOnce()
                if (_uiState.value.autoRefresh) {
                    val live = _uiState.value.trainStatus.status in LIVE_TRAIN_STATUSES
                    delay(if (live) 1_000L.milliseconds else 3_000L.milliseconds)
                } else {
                    break
                }
            }
        }
    }

    private fun startHardwarePolling() {
        hardwareJob?.cancel()
        hardwareJob = viewModelScope.launch {
            while (isActive) {
                fetchHardwareOnce()
                if (_uiState.value.autoRefresh) {
                    delay(1_000L.milliseconds)
                } else {
                    break
                }
            }
        }
    }

    private suspend fun fetchHardwareOnce() {
        try {
            val snapshot = withContext(IoDispatcher) { ipc.hardwareStatus() }
            _uiState.update { state ->
                val step = hardwareStep
                hardwareStep += 1
                state.copy(
                    hardware = snapshot,
                    hardwareHistory = appendHardwareHistory(state.hardwareHistory, snapshot, step),
                )
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(
                    hardware = it.hardware.copy(
                        available = false,
                        error = e.message ?: e.toString(),
                    ),
                )
            }
        }
    }

    private suspend fun fetchOnce() {
        val firstLoad = !_uiState.value.connected && _uiState.value.latestStats.isEmpty()
        if (firstLoad) {
            _uiState.update { it.copy(isLoading = true, errorMessage = null) }
        }
        try {
            val pinned = _uiState.value.selectedRun
            val runs = withContext(IoDispatcher) { ipc.listRuns() }.runs
            // Re-read the pinned entry so its badge and figures follow the live list.
            val selected = pinned?.let { run -> runs.firstOrNull { it.runId == run.runId } ?: run }
            val dashboard = withContext(IoDispatcher) {
                ipc.getDashboard(name = selected?.outputName, runId = selected?.runId)
            }
            val samples = withContext(IoDispatcher) {
                ipc.listSamples(name = selected?.outputName, runId = selected?.runId)
            }
            // The Checkpoints section follows the run the page shows, so it needs that run's own
            // name even when the trainer is on a run created under a different `output_name`.
            // Neither list may take the dashboard down: a broken checkpoint file or a job record
            // the generator is rewriting leaves the charts and controls alone.
            val shown = displayedRun(runs, selected, dashboard.runId)
            val shownName = shown?.outputName ?: selected?.outputName
            val checkpoints = runCatching {
                withContext(IoDispatcher) { ipc.listCheckpoints(name = shownName) }.checkpoints
            }.getOrDefault(emptyList())
            val generatedJobs = runCatching {
                withContext(IoDispatcher) {
                    ipc.listGeneratedSamples(name = shownName, runId = shown?.runId)
                }.jobs
            }.getOrDefault(emptyList())
            val trainStatus = withContext(IoDispatcher) { ipc.trainStatus() }
            _uiState.update { state ->
                val generated = generatedJobs
                // The preview list is the section's own images in the section's own order, so the
                // path a click opened is looked up in the very rows the page drew.
                val shownCheckpoints = if (shown == null) {
                    emptyList()
                } else {
                    checkpointsForRun(checkpoints, shown.runId)
                }
                val previewPath = state.previewIndex?.let { index ->
                    previewList(state.checkpoints, state.samples, state.generatedJobs).getOrNull(index)?.path
                }
                val newList = previewList(shownCheckpoints, samples.samples, generated)
                val newPreview = previewPath?.let { path ->
                    newList.indexOfFirst { it.path == path }.takeIf { it >= 0 }
                }
                state.copy(
                    isLoading = false,
                    errorMessage = null,
                    connected = true,
                    config = dashboard.config,
                    runId = dashboard.runId,
                    runs = runs,
                    selectedRun = selected,
                    latestStats = dashboard.latestStats,
                    metrics = dashboard.metrics,
                    samples = samples.samples,
                    checkpoints = shownCheckpoints,
                    generatedJobs = generated,
                    previewIndex = newPreview,
                    trainStatus = trainStatus,
                    pendingCommand = resolvedPending(state.pendingCommand, trainStatus.status),
                )
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(
                    isLoading = false,
                    connected = false,
                    errorMessage = e.message ?: e.toString(),
                )
            }
        }
    }

    private fun movePreview(delta: Int) {
        val list = currentPreviewList()
        if (list.isEmpty()) {
            closePreview()
            return
        }
        val current = _uiState.value.previewIndex ?: return
        val next = (current + delta).mod(list.size)
        _uiState.update { it.copy(previewIndex = next) }
    }

    /** What the fullscreen preview cycles through: every image the Checkpoints section shows. */
    private fun currentPreviewList(state: DashboardUiState = _uiState.value): List<SampleItem> =
        previewList(state.checkpoints, state.samples, state.generatedJobs)
}

/**
 * Start / Pause / Resume / Early Stop / Reset act on the run `state.json` is on. A run the
 * user pinned in the history list is a past run: the page shows it, the controls stay off.
 */
internal fun trainingControlsEnabled(state: DashboardUiState): Boolean {
    val pinned = state.selectedRun ?: return true
    return pinned.runId == state.trainStatus.runId
}

/**
 * The live cadence / sampling controls act on the run the trainer is on, so they follow the same
 * rule as the Start/Pause buttons and additionally need a run that is actually live.
 */
internal fun liveSettingsEnabled(state: DashboardUiState): Boolean =
    trainingControlsEnabled(state) && state.trainStatus.status in LIVE_TRAIN_STATUSES

/**
 * Whether a one-off generation may start: only while no live trainer is using the GPU. A paused
 * run is fine (pause has offloaded every module); api.py enforces the same rule.
 */
internal fun generationAllowed(status: TrainStatus): Boolean =
    !status.alive || status.status !in GPU_BUSY_STATUSES

/** Statuses in which the trainer holds the GPU (a paused one has given it back). */
internal val GPU_BUSY_STATUSES = setOf(
    "starting",
    "encoding",
    "training",
    "sampling",
    "pausing",
    "resuming",
    "stopping",
)

internal fun resolvedPending(pending: String?, status: String): String? {
    if (status in setOf("idle", "finished", "error")) return null
    return when (pending) {
        "pause" -> if (status == "pausing" || status == "paused") null else pending
        "resume" -> if (status in setOf("resuming", "encoding", "training", "sampling")) null else pending
        "stop" -> if (status == "stopping") null else pending
        else -> pending
    }
}

internal val LIVE_TRAIN_STATUSES = setOf(
    "starting",
    "encoding",
    "training",
    "sampling",
    "pausing",
    "paused",
    "resuming",
    "stopping",
)

/**
 * The images a page state shows, in the order the Checkpoints section shows them. `openPreview`
 * resolves a clicked thumbnail in this same list, so anything the section draws can be opened.
 */
internal fun previewList(
    checkpoints: List<CheckpointItem>,
    samples: Map<String, List<SampleItem>>,
    jobs: List<GeneratedSampleJob>,
): List<SampleItem> = sectionImages(checkpointRows(checkpoints, samples, jobs))

private const val HARDWARE_HISTORY_CAP = 360
private const val BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0
private const val GENERATED_POLL_MILLIS = 1_500L

internal fun appendHardwareHistory(
    history: HardwareHistory,
    snapshot: HardwareStatus,
    step: Int,
): HardwareHistory {
    val gpu = snapshot.gpus.firstOrNull()
    val cpu = snapshot.cpu
    val vramGiB = gpu?.memUsedBytes?.let { it.toDouble() / BYTES_PER_GIB }
    val ramGiB = cpu.memUsedBytes?.let { it.toDouble() / BYTES_PER_GIB }
    return HardwareHistory(
        gpuUtil = appendHardwarePoint(history.gpuUtil, step, gpu?.gpuUtilPct),
        vramGiB = appendHardwarePoint(history.vramGiB, step, vramGiB),
        powerW = appendHardwarePoint(history.powerW, step, gpu?.powerW),
        tempEdge = appendHardwarePoint(history.tempEdge, step, gpu?.tempEdgeC ?: gpu?.tempC),
        tempJunction = appendHardwarePoint(history.tempJunction, step, gpu?.tempJunctionC),
        cpuUtil = appendHardwarePoint(history.cpuUtil, step, cpu.utilPct),
        cpuTemp = appendHardwarePoint(history.cpuTemp, step, cpu.tempC),
        ramGiB = appendHardwarePoint(history.ramGiB, step, ramGiB),
    )
}

private fun appendHardwarePoint(points: List<MetricPoint>, step: Int, value: Double?): List<MetricPoint> {
    if (value == null || !value.isFinite()) return points
    return (points + MetricPoint(step = step, value = value.toFloat())).takeLast(HARDWARE_HISTORY_CAP)
}

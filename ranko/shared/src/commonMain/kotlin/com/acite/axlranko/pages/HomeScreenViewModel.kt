package com.acite.axlranko.pages

import com.acite.axlranko.IoDispatcher
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.AutomationJobSummary
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.data.trainDataEntries
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.TrainStatus
import kotlin.random.Random
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.time.Duration.Companion.milliseconds

data class HomeUiState(
    val status: TrainStatus = TrainStatus(),
    val samplePaths: List<String> = emptyList(),
    /** `Train/Avg_Loss` for the run `train_status` names. Empty until `dashboard` answers. */
    val avgLoss: List<MetricPoint> = emptyList(),
    val jobs: List<AutomationJobSummary> = emptyList(),
    val runs: List<RunSummary> = emptyList(),
    /** Null until the first `hardware_status` reply. */
    val hardware: HardwareStatus? = null,
    /** A sample drawn from every train-data folder, round-robin. */
    val datasetImages: List<String> = emptyList(),
    val error: String? = null,
)

/**
 * Light poll for the Home cards. `dashboard()` is read only for `Train/Avg_Loss`, and only when
 * the run or step changes or the run is live — not the Dashboard page's checkpoint and hardware
 * chain. This poll stops when Home leaves the screen.
 */
@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class HomeScreenViewModel(
    private val ipc: TrainerIpcClient,
    private val refreshHub: DatasetRefreshHub,
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    private var entered = false
    private var statusJob: Job? = null
    private var jobsJob: Job? = null
    private var hardwareJob: Job? = null
    private var datasetJob: Job? = null

    fun onEnter() {
        if (entered) return
        entered = true
        statusJob = viewModelScope.launch { statusLoop() }
        jobsJob = viewModelScope.launch { jobsLoop() }
        hardwareJob = viewModelScope.launch { hardwareLoop() }
        datasetJob = viewModelScope.launch {
            refreshDatasets()
            refreshHub.events.collect { refreshDatasets() }
        }
    }

    fun onLeave() {
        entered = false
        statusJob?.cancel()
        jobsJob?.cancel()
        hardwareJob?.cancel()
        datasetJob?.cancel()
        statusJob = null
        jobsJob = null
        hardwareJob = null
        datasetJob = null
    }

    private suspend fun statusLoop() {
        var lastKey: String? = null
        var sinceSamples = SAMPLE_REFRESH_MILLIS
        var sinceRuns = RUN_REFRESH_MILLIS
        var sinceLoss = SAMPLE_REFRESH_MILLIS
        while (currentCoroutineContext().isActive) {
            val delayMs = try {
                val status = withContext(IoDispatcher) { ipc.trainStatus() }
                _uiState.update { it.copy(status = status, error = null) }
                val key = "${status.runId}|${status.status}|${status.training.step}"
                val changed = key != lastKey
                lastKey = key
                val live = status.status in LIVE_TRAIN_STATUSES
                if (changed || (status.status == "sampling" && sinceSamples >= SAMPLE_REFRESH_MILLIS)) {
                    val listed = withContext(IoDispatcher) { ipc.listSamples(runId = status.runId) }
                    sinceSamples = 0
                    _uiState.update { it.copy(samplePaths = latestSamplePaths(listed.samples)) }
                }
                if (changed || sinceRuns >= RUN_REFRESH_MILLIS) {
                    val listed = withContext(IoDispatcher) { ipc.listRuns() }
                    sinceRuns = 0
                    _uiState.update { it.copy(runs = listed.runs) }
                }
                if (changed || (live && sinceLoss >= SAMPLE_REFRESH_MILLIS)) {
                    val dashboard = withContext(IoDispatcher) {
                        ipc.getDashboard(runId = status.runId, name = status.outputName)
                    }
                    sinceLoss = 0
                    _uiState.update {
                        it.copy(avgLoss = dashboard.metrics["Train/Avg_Loss"].orEmpty())
                    }
                }
                if (live) 1_000L else 3_000L
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _uiState.update { it.copy(error = e.message ?: "Status unavailable") }
                3_000L
            }
            sinceSamples += delayMs
            sinceRuns += delayMs
            sinceLoss += delayMs
            delay(delayMs.milliseconds)
        }
    }

    private suspend fun hardwareLoop() {
        while (currentCoroutineContext().isActive) {
            try {
                val snapshot = withContext(IoDispatcher) { ipc.hardwareStatus() }
                _uiState.update { it.copy(hardware = snapshot) }
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                _uiState.update { it.copy(hardware = it.hardware ?: HardwareStatus(available = false)) }
            }
            delay(2_000L.milliseconds)
        }
    }

    private suspend fun refreshDatasets() {
        try {
            val dirs = withContext(IoDispatcher) {
                ipc.parsedConfig().second.environment.trainDataEntries()
                    .map { it.path }
                    .filter { it.isNotBlank() }
            }
            val folders = dirs.map { dir ->
                withContext(IoDispatcher) {
                    runCatching { ipc.datasetList(dir).items.map { it.image } }.getOrDefault(emptyList())
                }
            }
            val picked = pickDatasetThumbs(folders, HOME_DATASET_POOL, Random(Random.Default.nextLong()))
            _uiState.update { it.copy(datasetImages = picked) }
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            // The card keeps the previous sample. A missing config is an empty grid.
        }
    }

    private suspend fun jobsLoop() {
        while (currentCoroutineContext().isActive) {
            val wait = try {
                val listed = withContext(IoDispatcher) { ipc.automationJobList() }
                _uiState.update { it.copy(jobs = listed.jobs, error = null) }
                if (listed.jobs.any { it.state == "running" }) 2_000L else 5_000L
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _uiState.update { it.copy(error = e.message ?: "Jobs unavailable") }
                5_000L
            }
            delay(wait.milliseconds)
        }
    }

    private companion object {
        const val SAMPLE_REFRESH_MILLIS = 5_000L
        const val RUN_REFRESH_MILLIS = 5_000L
    }
}

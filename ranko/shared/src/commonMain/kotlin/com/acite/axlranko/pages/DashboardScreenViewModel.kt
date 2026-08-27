package com.acite.axlranko.pages

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.model.TrainStatus
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
import kotlin.time.Duration.Companion.milliseconds

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class DashboardScreenViewModel(
    private val ipc: TrainerIpcClient,
) : ViewModel() {

    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()

    private var pollingJob: Job? = null
    private var entered = false

    fun onEnter() {
        if (!entered) {
            entered = true
            startPolling()
        } else {
            refreshNow()
        }
    }

    fun toggleAutoRefresh(enabled: Boolean) {
        _uiState.update { it.copy(autoRefresh = enabled) }
        if (enabled) {
            startPolling()
        } else {
            pollingJob?.cancel()
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
        val list = flattenSamples(_uiState.value.samples)
        val index = list.indexOfFirst { it.path == sample.path }
        if (index >= 0) {
            _uiState.update { it.copy(previewIndex = index) }
        }
    }

    fun closePreview() {
        _uiState.update { it.copy(previewIndex = null) }
    }

    fun previewNext() = movePreview(1)

    fun previewPrev() = movePreview(-1)

    fun refreshNow() {
        viewModelScope.launch { fetchOnce() }
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
                val status = withContext(Dispatchers.IO) { block() }
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
                fetchOnce()
                if (_uiState.value.autoRefresh) startPolling()
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

    private suspend fun fetchOnce() {
        val firstLoad = !_uiState.value.connected && _uiState.value.latestStats.isEmpty()
        if (firstLoad) {
            _uiState.update { it.copy(isLoading = true, errorMessage = null) }
        }
        try {
            val dashboard = withContext(Dispatchers.IO) { ipc.getDashboard() }
            val samples = withContext(Dispatchers.IO) { ipc.listSamples() }
            val trainStatus = withContext(Dispatchers.IO) { ipc.trainStatus() }
            _uiState.update { state ->
                val previewPath = state.previewIndex
                    ?.let { flattenSamples(state.samples).getOrNull(it)?.path }
                val newList = flattenSamples(samples.samples)
                val newPreview = previewPath?.let { path ->
                    newList.indexOfFirst { it.path == path }.takeIf { it >= 0 }
                }
                state.copy(
                    isLoading = false,
                    errorMessage = null,
                    connected = true,
                    config = dashboard.config,
                    latestStats = dashboard.latestStats,
                    metrics = dashboard.metrics,
                    samples = samples.samples,
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
        val list = flattenSamples(_uiState.value.samples)
        if (list.isEmpty()) {
            closePreview()
            return
        }
        val current = _uiState.value.previewIndex ?: return
        val next = (current + delta).mod(list.size)
        _uiState.update { it.copy(previewIndex = next) }
    }
}

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

internal fun flattenSamples(samples: Map<String, List<SampleItem>>): List<SampleItem> {
    return samples.entries
        .sortedByDescending { it.key.toIntOrNull() ?: Int.MIN_VALUE }
        .flatMap { it.value }
}

package dashBoard

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dataModel.SampleItem
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import services.LoraApiClient
import kotlin.time.Duration.Companion.milliseconds

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class DashBoardViewModel(
    val loraApiClient: LoraApiClient
) : ViewModel() {
    // UI State
    var autoRefresh by mutableStateOf(true)
    var smoothing by mutableFloatStateOf(0.90f)

    // Data State
    var dashboardData by mutableStateOf<JsonObject?>(null)
    var samplesData by mutableStateOf<Map<String, List<SampleItem>>>(emptyMap())

    // Extracted for convenience
    var configData by mutableStateOf<JsonObject?>(null)
    var latestStats by mutableStateOf<JsonObject?>(null)
    var metricsData by mutableStateOf<JsonObject?>(null)

    private var pollingJob: Job? = null

    init {
        startPolling()
    }

    fun toggleAutoRefresh(enabled: Boolean) {
        autoRefresh = enabled
        if (enabled) {
            startPolling()
        } else {
            pollingJob?.cancel()
        }
    }

    private fun startPolling() {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch(Dispatchers.Default) {
            while (isActive) {
                fetchDashboardData()
                fetchSamples()

                if (autoRefresh) {
                    delay(1_000L.milliseconds) // 10s auto-refresh
                } else {
                    break
                }
            }
        }
    }

    private suspend fun fetchDashboardData() {
        try {
            val data = loraApiClient.getDashboardData()
            dashboardData = data
            configData = data["config"]?.jsonObject
            latestStats = data["latest_stats"]?.jsonObject
            metricsData = data["metrics"]?.jsonObject
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private suspend fun fetchSamples() {
        try {
            val response = loraApiClient.listSamples()
            samplesData = response.samples
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    suspend fun fetchSampleImage(filename: String): ByteArray? {
        return try {
            loraApiClient.getSampleImage(filename)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}
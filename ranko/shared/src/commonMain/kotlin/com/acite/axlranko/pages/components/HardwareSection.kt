package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.HardwareCpu
import com.acite.axlranko.model.HardwareGpu
import com.acite.axlranko.model.HardwareHistory
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.theme.rankoColors
import kotlin.math.roundToInt

private const val BytesPerGiB = 1024.0 * 1024.0 * 1024.0

@Composable
fun HardwareSection(uiState: DashboardUiState) {
    val hardware = uiState.hardware
    val gpu = hardware.gpus.firstOrNull()
    val history = uiState.hardwareHistory
    val stroke = uiState.chartStroke

    if (!hardware.available && gpu == null) {
        PorcelainCard {
            Text(
                text = hardware.error?.let { "Hardware monitor unavailable: $it" }
                    ?: "Waiting for nvtop snapshot…",
                style = MaterialTheme.typography.bodyMedium,
                color = rankoColors.text,
            )
        }
        return
    }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        hardware.error?.let { message ->
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.qualityRed,
            )
        }
        HardwareInfoRow(gpu, hardware.cpu)
        HardwareMetricCards(gpu, hardware.cpu)
        HardwareCharts(history, stroke, gpu)
    }
}

@Composable
private fun HardwareInfoRow(gpu: HardwareGpu?, cpu: HardwareCpu) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            CompactMetric("GPU", gpu?.name?.ifBlank { "—" } ?: "—")
            CompactMetric("Core clock", formatMhz(gpu?.gpuClockMhz))
            CompactMetric("Mem clock", formatMhz(gpu?.memClockMhz))
            CompactMetric("Fan", formatPct(gpu?.fanPct))
            CompactMetric("VRAM", formatVram(gpu))
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            CompactMetric("CPU", cpu.name.ifBlank { "—" })
            CompactMetric("Threads", if (cpu.nLogical > 0) cpu.nLogical.toString() else "—")
            CompactMetric("CPU util", formatPct(cpu.utilPct))
            CompactMetric("CPU temp", formatTemp(cpu.tempC))
        }
    }
}

@Composable
private fun HardwareMetricCards(gpu: HardwareGpu?, cpu: HardwareCpu) {
    val colors = rankoColors
    val gpuUtil = formatPct(gpu?.gpuUtilPct)
    val vram = formatVram(gpu)
    val power = formatWatts(gpu?.powerW)
    val temps = formatTemps(gpu)
    val cpuLine = listOfNotNull(formatPct(cpu.utilPct).takeIf { it != "—" }, formatTemp(cpu.tempC).takeIf { it != "—" })
        .joinToString(" · ")
        .ifBlank { "—" }

    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val isWide = maxWidth > 720.dp
        if (isWide) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                MetricCard("GPU", gpuUtil, colors.accentPink, Modifier.weight(1f))
                MetricCard("VRAM", vram, colors.accentBlue, Modifier.weight(1f))
                MetricCard("Power", power, colors.qualityOrange, Modifier.weight(1f))
                MetricCard("GPU temp", temps, colors.qualityRed, Modifier.weight(1f))
                MetricCard("CPU", cpuLine, colors.accentLilac, Modifier.weight(1f))
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard("GPU", gpuUtil, colors.accentPink, Modifier.weight(1f))
                    MetricCard("VRAM", vram, colors.accentBlue, Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard("Power", power, colors.qualityOrange, Modifier.weight(1f))
                    MetricCard("GPU temp", temps, colors.qualityRed, Modifier.weight(1f))
                }
                MetricCard("CPU", cpuLine, colors.accentLilac, Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun HardwareCharts(
    history: HardwareHistory,
    stroke: Float,
    gpu: HardwareGpu?,
) {
    val colors = rankoColors
    val vramMax = gpu?.memTotalBytes?.toDouble()?.div(BytesPerGiB)?.toFloat()?.takeIf { it > 0f }
        ?: history.vramGiB.maxOfOrNull { it.value }?.takeIf { it > 0f }
        ?: 1f
    val powerMax = (history.powerW.maxOfOrNull { it.value } ?: 1f).coerceAtLeast(1f) * 1.15f
    val cpuTempMax = maxOf(history.cpuTemp.maxOfOrNull { it.value } ?: 0f, 100f)

    Row(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        MultiSeriesChartCard(
            title = "GPU",
            series = listOf(
                ChartSeries("Util", history.gpuUtil, colors.accentPink, domainMin = 0f, domainMax = 100f),
                ChartSeries("VRAM", history.vramGiB, colors.accentBlue, domainMin = 0f, domainMax = vramMax),
                ChartSeries("Power", history.powerW, colors.qualityOrange, domainMin = 0f, domainMax = powerMax),
            ),
            smoothing = 0f,
            modifier = Modifier.weight(1f),
            outlierClip = 0f,
            strokeWidth = stroke,
        )
        MultiSeriesChartCard(
            title = "GPU temp",
            series = listOf(
                ChartSeries("Edge", history.tempEdge, colors.qualityRed),
                ChartSeries("Junction", history.tempJunction, colors.qualityPurple),
            ),
            smoothing = 0f,
            modifier = Modifier.weight(1f),
            outlierClip = 0f,
            strokeWidth = stroke,
        )
        MultiSeriesChartCard(
            title = "CPU",
            series = listOf(
                ChartSeries("Util", history.cpuUtil, colors.accentLilac, domainMin = 0f, domainMax = 100f),
                ChartSeries("Temp", history.cpuTemp, colors.star, domainMin = 0f, domainMax = cpuTempMax),
            ),
            smoothing = 0f,
            modifier = Modifier.weight(1f),
            outlierClip = 0f,
            strokeWidth = stroke,
        )
    }
}

private fun formatPct(value: Double?): String {
    if (value == null || !value.isFinite()) return "—"
    return "${value.roundToInt()}%"
}

private fun formatMhz(value: Double?): String {
    if (value == null || !value.isFinite()) return "—"
    return "${value.roundToInt()} MHz"
}

private fun formatWatts(value: Double?): String {
    if (value == null || !value.isFinite()) return "—"
    return "${value.roundToInt()} W"
}

private fun formatTemp(value: Double?): String {
    if (value == null || !value.isFinite()) return "—"
    return "${value.roundToInt()} °C"
}

private fun formatTemps(gpu: HardwareGpu?): String {
    if (gpu == null) return "—"
    val edge = gpu.tempEdgeC ?: gpu.tempC
    val junction = gpu.tempJunctionC
    val edgeText = formatTemp(edge)
    if (junction == null) return edgeText
    return "$edgeText / ${formatTemp(junction)}"
}

private fun formatVram(gpu: HardwareGpu?): String {
    val used = gpu?.memUsedBytes?.toDouble()
    val total = gpu?.memTotalBytes?.toDouble()
    if (used == null && total == null) return "—"
    val usedGiB = used?.div(BytesPerGiB)
    val totalGiB = total?.div(BytesPerGiB)
    return when {
        usedGiB != null && totalGiB != null -> "${formatGiB(usedGiB)} / ${formatGiB(totalGiB)} GiB"
        usedGiB != null -> "${formatGiB(usedGiB)} GiB"
        totalGiB != null -> "${formatGiB(totalGiB)} GiB"
        else -> "—"
    }
}

private fun formatGiB(value: Double): String = ((value * 100.0).roundToInt() / 100.0).toString()

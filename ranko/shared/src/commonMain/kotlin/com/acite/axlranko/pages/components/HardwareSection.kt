package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.HardwareCpu
import com.acite.axlranko.model.HardwareGpu
import com.acite.axlranko.model.HardwareHistory
import com.acite.axlranko.model.HardwareVmmVa
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.theme.rankoColors
import kotlin.math.roundToInt

private const val BytesPerGiB = 1024.0 * 1024.0 * 1024.0
private const val BytesPerTiB = BytesPerGiB * 1024.0

@Composable
fun HardwareSection(uiState: DashboardUiState) {
    val hardware = uiState.hardware
    val gpu = hardware.gpus.firstOrNull()
    val history = uiState.hardwareHistory
    val stroke = uiState.chartStroke

    val vmmVa = hardware.vmmVa

    if (!hardware.available && gpu == null) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            PorcelainCard {
                Text(
                    text = hardware.error?.let { "Hardware monitor unavailable: $it" }
                        ?: "Waiting for nvtop snapshot…",
                    style = MaterialTheme.typography.bodyMedium,
                    color = rankoColors.text,
                )
            }
            if (vmmVa != null) {
                VmmVaBar(vmmVa)
            }
        }
        return
    }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        if (vmmVa != null) {
            VmmVaBar(vmmVa)
        }
        hardware.error?.let { message ->
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = rankoColors.qualityRed,
            )
        }
        HardwareInfoRow(gpu, hardware.cpu)
        HardwareMetricCards(gpu, hardware.cpu)
        HardwareCharts(history, stroke, gpu, hardware.cpu)
    }
}

@Composable
private fun VmmVaBar(vmmVa: HardwareVmmVa) {
    val colors = rankoColors
    val used = vmmVa.usedBytes.coerceAtLeast(0L)
    val total = vmmVa.totalBytes.coerceAtLeast(0L)
    val fraction = if (total > 0L) {
        (used.toDouble() / total.toDouble()).toFloat().coerceIn(0f, 1f)
    } else {
        0f
    }
    PorcelainCard {
        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                text = if (vmmVa.vaNeverReuse) "GPU VA (not returned)" else "GPU VA (live)",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = colors.text,
            )
            Text(
                text = "${formatVaBytes(used)} / ${formatVaBytes(total)}",
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
                color = colors.text,
            )
            LinearProgressIndicator(
                progress = { fraction },
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = colors.accentBlue,
                trackColor = colors.bgCard.copy(alpha = 0.6f),
            )
            Text(
                text = if (vmmVa.vaNeverReuse) {
                    "VA never reused (legacy workaround): a freed range keeps its address for the process lifetime, so this only grows."
                } else {
                    "VA is given back when a block is freed, so this is what the patch holds right now."
                },
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
        }
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
            CompactMetric("RAM", formatRam(cpu))
        }
    }
}

@Composable
private fun HardwareMetricCards(gpu: HardwareGpu?, cpu: HardwareCpu) {
    val colors = rankoColors
    val gpuUtil = formatPct(gpu?.gpuUtilPct)
    val vram = formatVram(gpu)
    val power = formatWatts(gpu?.powerW)
    val temps = formatTemps(gpu, cpu)
    val cpuLine = listOfNotNull(
        formatPct(cpu.utilPct).takeIf { it != "—" },
        formatRam(cpu).takeIf { it != "—" },
    ).joinToString(" · ").ifBlank { "—" }

    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val isWide = maxWidth > 720.dp
        if (isWide) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                MetricCard("GPU", gpuUtil, colors.accentPink, Modifier.weight(1f))
                MetricCard("VRAM", vram, colors.accentBlue, Modifier.weight(1f))
                MetricCard("Power", power, colors.qualityOrange, Modifier.weight(1f))
                MetricCard("Temp", temps, colors.qualityRed, Modifier.weight(1f))
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
                    MetricCard("Temp", temps, colors.qualityRed, Modifier.weight(1f))
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
    cpu: HardwareCpu,
) {
    val colors = rankoColors
    val vramMax = gpu?.memTotalBytes?.toDouble()?.div(BytesPerGiB)?.toFloat()?.takeIf { it > 0f }
        ?: history.vramGiB.maxOfOrNull { it.value }?.takeIf { it > 0f }
        ?: 1f
    val ramMax = cpu.memTotalBytes?.toDouble()?.div(BytesPerGiB)?.toFloat()?.takeIf { it > 0f }
        ?: history.ramGiB.maxOfOrNull { it.value }?.takeIf { it > 0f }
        ?: 1f

    Column(
        verticalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            MultiSeriesChartCard(
                title = "GPU",
                series = listOf(
                    ChartSeries("Util", history.gpuUtil, colors.accentPink, domainMin = 0f, domainMax = 100f),
                    ChartSeries("VRAM", history.vramGiB, colors.accentBlue, domainMin = 0f, domainMax = vramMax),
                ),
                smoothing = 0f,
                modifier = Modifier.weight(1f),
                outlierClip = 0f,
                strokeWidth = stroke,
            )
            MultiSeriesChartCard(
                title = "Temp",
                series = listOf(
                    ChartSeries("Edge", history.tempEdge, colors.qualityRed),
                    ChartSeries("Junction", history.tempJunction, colors.qualityPurple),
                    ChartSeries("CPU", history.cpuTemp, colors.star),
                ),
                smoothing = 0f,
                modifier = Modifier.weight(1f),
                outlierClip = 0f,
                strokeWidth = stroke,
            )
        }
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            MultiSeriesChartCard(
                title = "Power",
                series = listOf(
                    ChartSeries("GPU", history.powerW, colors.qualityOrange),
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
                    ChartSeries("RAM", history.ramGiB, colors.qualityMint, domainMin = 0f, domainMax = ramMax),
                ),
                smoothing = 0f,
                modifier = Modifier.weight(1f),
                outlierClip = 0f,
                strokeWidth = stroke,
            )
        }
    }
}

private fun formatVaBytes(bytes: Long): String {
    if (bytes >= BytesPerTiB) {
        return "${formatGiB(bytes / BytesPerTiB)} TiB"
    }
    return "${formatGiB(bytes / BytesPerGiB)} GiB"
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

private fun formatTemps(gpu: HardwareGpu?, cpu: HardwareCpu): String {
    val edge = formatTemp(gpu?.tempEdgeC ?: gpu?.tempC)
    val junction = formatTemp(gpu?.tempJunctionC)
    val cpuTemp = formatTemp(cpu.tempC)
    val gpuText = when {
        edge != "—" && junction != "—" -> "$edge / $junction"
        edge != "—" -> edge
        junction != "—" -> junction
        else -> null
    }
    return listOfNotNull(gpuText, cpuTemp.takeIf { it != "—" }).joinToString(" · ").ifBlank { "—" }
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

private fun formatRam(cpu: HardwareCpu): String {
    val used = cpu.memUsedBytes?.toDouble()
    val total = cpu.memTotalBytes?.toDouble()
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

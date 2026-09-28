package com.acite.axlranko.pages

import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.TrainSettings
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.pages.components.nextRunSummary
import com.acite.axlranko.pages.components.settingsSummary
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The live cadence / sampling controls and the rule that gates a one-off generation. */
class TrainSettingsRulesTest {

    private fun status(status: String, alive: Boolean = true) =
        TrainStatus(status = status, alive = alive, pid = if (alive) 4242 else null)

    @Test
    fun theSummaryReadsTheEffectiveValues() {
        assertEquals(
            "Save every 100 steps · next at step 300 · sampling on",
            settingsSummary(TrainSettings(saveEveryNSteps = 100, nextSaveStep = 300, samplingEnabled = true)),
        )
        assertEquals(
            "Save every 50 steps · next at step 50 · sampling off",
            settingsSummary(TrainSettings(saveEveryNSteps = 50, nextSaveStep = 50, samplingEnabled = false)),
        )
    }

    @Test
    fun withNoLiveRunTheRowNamesTheNextRunsValues() {
        // A runtime directory with no run on it carries the placeholder 0, which is not a
        // configuration: the row must read config.toml instead.
        assertEquals(
            "Next run · save every 100 steps · sampling on",
            nextRunSummary(saveEveryNSteps = 100, samplingEnabled = true),
        )
        assertEquals(
            "Next run · save every 50 steps · sampling off",
            nextRunSummary(saveEveryNSteps = 50, samplingEnabled = false),
        )
        assertEquals(
            "Next run · no checkpoints · sampling on",
            nextRunSummary(saveEveryNSteps = 0, samplingEnabled = true),
        )
        assertEquals("Next run · sampling off", nextRunSummary(null, false))
        assertEquals("Next run · save every 100 steps", nextRunSummary(100, null))
        assertNull(nextRunSummary(null, null))
    }

    @Test
    fun aDisabledScheduleSaysSo() {
        assertEquals(
            "Checkpoints off · sampling on",
            settingsSummary(TrainSettings(saveEveryNSteps = 0, nextSaveStep = 0, samplingEnabled = true)),
        )
    }

    @Test
    fun generationIsAllowedOnlyWhileNoLiveTrainerHasTheGpu() {
        for (busy in listOf("starting", "encoding", "training", "sampling", "pausing", "resuming", "stopping")) {
            assertFalse(generationAllowed(status(busy)), "$busy should refuse a generation")
        }
        // A paused run has offloaded every module; a finished or dead trainer is free.
        assertTrue(generationAllowed(status("paused")))
        assertTrue(generationAllowed(status("finished", alive = false)))
        assertTrue(generationAllowed(status("idle", alive = false)))
        assertTrue(generationAllowed(status("error", alive = false)))
    }

    @Test
    fun theLiveControlsFollowTheControlBarAndNeedALiveRun() {
        val live = DashboardUiState(
            runs = listOf(RunSummary(runId = "rein_1", outputName = "rein", live = true)),
            trainStatus = status("training").copy(runId = "rein_1"),
        )
        assertTrue(liveSettingsEnabled(live))

        // Pinned to a past run: the controls act on the trainer's run, so they are off.
        assertFalse(
            liveSettingsEnabled(
                live.copy(selectedRun = RunSummary(runId = "rein_0", outputName = "rein"))
            )
        )
        // No live run: nothing to retune.
        assertFalse(liveSettingsEnabled(live.copy(trainStatus = status("idle", alive = false))))
        assertFalse(liveSettingsEnabled(live.copy(trainStatus = status("finished", alive = false))))
    }

    @Test
    fun aPausedRunCanStillBeRetuned() {
        val state = DashboardUiState(
            runs = listOf(RunSummary(runId = "rein_1", outputName = "rein", live = true)),
            trainStatus = status("paused").copy(runId = "rein_1"),
        )
        assertTrue(liveSettingsEnabled(state))
    }

    @Test
    fun theControlBarRuleIsUnchangedForAFollowedRun() {
        val followed = DashboardUiState(trainStatus = status("training").copy(runId = "rein_1"))
        assertTrue(trainingControlsEnabled(followed))
        assertTrue(liveSettingsEnabled(followed))
    }
}

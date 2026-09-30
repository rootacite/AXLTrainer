package com.acite.axlranko.pages

import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.TrainSettings
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.pages.components.nextRunSummary
import com.acite.axlranko.pages.components.pendingSettingsLabel
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
    fun aPendingRequestNamesOnlyWhatChangedAndWhenItLands() {
        val effective = TrainSettings(saveEveryNSteps = 50, nextSaveStep = 3350, samplingEnabled = true)
        // The switch alone, flipped while a sample pass is running: the pass finishes first.
        assertEquals(
            "Pending (applies from the next sample pass) · sampling off",
            pendingSettingsLabel("sampling", effective, effective.copy(samplingEnabled = false)),
        )
        // During training the next optimizer step adopts it.
        assertEquals(
            "Pending (applies at the next step) · sampling off",
            pendingSettingsLabel("training", effective, effective.copy(samplingEnabled = false)),
        )
        // A paused run adopts it when it resumes.
        assertEquals(
            "Pending (applies when the run resumes) · save every 100 steps",
            pendingSettingsLabel("paused", effective, effective.copy(saveEveryNSteps = 100)),
        )
        // Both at once, one of them turning checkpoints off.
        assertEquals(
            "Pending (applies at the next step) · no checkpoints · sampling off",
            pendingSettingsLabel(
                "training",
                effective,
                effective.copy(saveEveryNSteps = 0, samplingEnabled = false),
            ),
        )
        // A request that matches what is running draws nothing.
        assertEquals("", pendingSettingsLabel("training", effective, effective))
    }

    @Test
    fun aStatusNeverReportsARequestItHasAdopted() {
        // What the wire does: `requested` is null unless the file asks for something else.
        assertNull(TrainStatus(status = "training", alive = true, pid = 4242).requested)
        assertEquals(
            TrainSettings(saveEveryNSteps = 50, samplingEnabled = false),
            TrainStatus(
                status = "sampling",
                alive = true,
                pid = 4242,
                settings = TrainSettings(saveEveryNSteps = 50, nextSaveStep = 3350, samplingEnabled = true),
                requested = TrainSettings(saveEveryNSteps = 50, samplingEnabled = false),
            ).requested,
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
    fun thePanelMayGenerateWheneverTheGpuIsFree() {
        // The Ctrl+click panel's Generate button read `alive` directly: that kept it disabled for
        // a paused run (while the section's own button was enabled), and it is what left it
        // disabled over a run that had finished while the helper still had its PID on record.
        assertTrue(generationAllowed(status("paused")))
        assertTrue(generationAllowed(status("finished", alive = false)))
        assertTrue(generationAllowed(status("finished", alive = true)))
        assertTrue(generationAllowed(status("error", alive = false)))
        assertFalse(generationAllowed(status("training")))
        assertFalse(generationAllowed(status("sampling")))
    }

    @Test
    fun theControlBarRuleIsUnchangedForAFollowedRun() {
        val followed = DashboardUiState(trainStatus = status("training").copy(runId = "rein_1"))
        assertTrue(trainingControlsEnabled(followed))
        assertTrue(liveSettingsEnabled(followed))
    }
}

package com.acite.axlranko.pages.components

import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.pages.trainingControlsEnabled
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RunHistoryTest {

    @Test
    fun stampLabelReadsTheRunIdsOwnStamp() {
        assertEquals("2026-09-28 11:09:28", runStampLabel("Tsukuyomi_20260928_110928"))
        assertEquals("2026-09-11 12:00:00", runStampLabel("rein_20260911_120000_2"))
    }

    @Test
    fun aNameThatContainsAStampKeepsTheTail() {
        assertEquals(
            "2026-09-12 13:00:00",
            runStampLabel("rein_20260911_120000_20260912_130000"),
        )
    }

    @Test
    fun namesWithoutAStampHaveNoLabel() {
        assertNull(runStampLabel("rein"))
        assertNull(runStampLabel("rein_s000100"))
        assertNull(runStampLabel(""))
    }

    @Test
    fun detailLabelListsWhatTheRunLeftOnDisk() {
        val run = RunSummary(
            runId = "konomi_20260912_090000",
            outputName = "konomi",
            lastStep = 300,
            samples = 2,
            checkpoints = 1,
            sizeBytes = 1024,
        )
        assertEquals(
            "2026-09-12 09:00:00 · step 300 · 2 samples · 1 checkpoint · 1.0 KB",
            runDetailLabel(run),
        )
    }

    @Test
    fun detailLabelUsesSingularCounts() {
        val run = RunSummary(runId = "rein_20260911_120000", lastStep = 1, samples = 1, checkpoints = 1)
        assertEquals("2026-09-11 12:00:00 · step 1 · 1 sample · 1 checkpoint", runDetailLabel(run))
    }

    @Test
    fun detailLabelOfARunWithNothingButItsStamp() {
        assertEquals("2026-09-11 12:00:00", runDetailLabel(RunSummary(runId = "rein_20260911_120000")))
        // A step is worth showing on its own, even with no images and no weights.
        assertEquals("2026-09-11 12:00:00 · step 4500", runDetailLabel(RunSummary(runId = "rein_20260911_120000", lastStep = 4500)))
    }

    @Test
    fun anEmptyRunWithoutAStampSaysSo() {
        assertEquals("no artifacts", runDetailLabel(RunSummary(runId = "weird")))
    }

    @Test
    fun controlsAreOnWhileFollowingTheCurrentRun() {
        val following = DashboardUiState(trainStatus = TrainStatus(runId = "rein_20260911_120000"))
        assertTrue(trainingControlsEnabled(following))
    }

    @Test
    fun controlsStayOnWhenThePinnedRunIsTheCurrentOne() {
        val pinned = DashboardUiState(
            selectedRun = RunSummary(runId = "rein_20260911_120000", current = true),
            trainStatus = TrainStatus(runId = "rein_20260911_120000"),
        )
        assertTrue(trainingControlsEnabled(pinned))
    }

    @Test
    fun controlsAreOffForAPastRun() {
        val stopped = DashboardUiState(
            selectedRun = RunSummary(runId = "konomi_20260912_090000"),
            trainStatus = TrainStatus(runId = "rein_20260911_120000"),
        )
        assertFalse(trainingControlsEnabled(stopped))
    }

    @Test
    fun aRunIsLiveStoppedOrNotThereYet() {
        assertNull(runStateLabel(null))
        // The run `state.json` is on is just a stopped run until its process is running.
        assertEquals("Stopped", runStateLabel(RunSummary(runId = "rein_20260911_120000", current = true)))
        assertEquals("Live", runStateLabel(RunSummary(runId = "rein_20260911_120000", current = true, live = true)))
        assertEquals("Stopped", runStateLabel(RunSummary(runId = "konomi_20260912_090000")))
    }
}

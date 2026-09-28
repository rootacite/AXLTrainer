package com.acite.axlranko.pages

import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.util.AppWindow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** The Utils WM tab: who is offered it, and what its two buttons say. */
class UtilsWmSectionTest {

    private class FakeWindow(override var maximized: Boolean = false) : AppWindow {
        override fun toggleMaximized(): Boolean {
            maximized = !maximized
            return maximized
        }

        override fun exit() = Unit
    }

    @Test
    fun theWmTabIsOfferedOnlyWhereThereIsAWindow() {
        assertTrue(ConfigSection.Wm in visibleSections(helperEndpointSettings = false, appWindow = FakeWindow()))
        assertFalse(ConfigSection.Wm in visibleSections(helperEndpointSettings = false, appWindow = null))
    }

    @Test
    fun theOtherTabsAreUnaffectedByTheGapTheWmTabLeaves() {
        val all = ConfigSection.entries.toList()
        assertEquals(all, visibleSections(helperEndpointSettings = true, appWindow = FakeWindow()))
        assertEquals(
            all - ConfigSection.Helper - ConfigSection.Wm,
            visibleSections(helperEndpointSettings = false, appWindow = null),
        )
    }

    @Test
    fun theMaximizeButtonSaysWhatItWillDo() {
        assertEquals("Maximize", maximizeButtonLabel(maximized = false))
        assertEquals("Restore", maximizeButtonLabel(maximized = true))
    }
}

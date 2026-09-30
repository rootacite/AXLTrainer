package com.acite.axlranko.pages

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.pages.components.automation.PromptEditorActions
import com.acite.axlranko.pages.components.automation.PromptPageEditor
import com.acite.axlranko.pages.components.automation.PromptPageHeader
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.prompt.defaultSpec
import com.acite.axlranko.prompt.parseMatrix
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The wizard's choice pills keep their own width whatever the pane is: a fixed row of three used to
 * measure the last pill down to a sliver — the `SEX` mode button disappeared and the `nude` exposure
 * pill shrank to two letters on a narrow window — and a compose-only smoke test cannot see that,
 * because nothing throws. The pane width here is the pane the wizard actually gets (window minus the
 * Automation rail, the step rail and the card padding), so 360-450 dp is an ordinary window.
 *
 * Skipped on a headless JVM.
 */
class PromptChoiceLayoutTest {

    private val matrix by lazy {
        parseMatrix(File(repoRoot(), "input_matrix.txt").readText(Charsets.UTF_8))
    }

    private fun repoRoot(): File {
        var dir: File? = File(System.getProperty("user.dir").orEmpty()).absoluteFile
        while (dir != null) {
            if (File(dir, "input_matrix.txt").isFile && File(dir, "trainer").isDirectory) return dir
            dir = dir.parentFile
        }
        throw AssertionError("could not find the repo root")
    }

    private fun noopActions() = PromptEditorActions(
        setCharacter = {}, setMode = {}, setExposure = { _, _ -> }, setClothingAny = {},
        toggleClothing = {}, setChest = {}, setBelly = {}, setFaceGroup = { _, _ -> },
        toggleFaceTag = { _, _ -> }, setSceneAny = {}, toggleScene = {}, setFamilyAny = {},
        toggleFamily = {}, setVaginalRatio = {}, setStageWeight = { _, _ -> }, setPoseAny = {},
        togglePose = {}, setCount = {}, setSeedText = {},
    )

    /** Every choice pill on one page, as `label -> (width, row)` at the given pane width. */
    private fun pills(pageKey: String, lang: PromptLang, width: Int): Map<String, Pair<Int, Float>> {
        val spec = defaultSpec()
        // Tall enough for the longest page (the face groups) to lay out inside the viewport: a node
        // below it measures to nothing at all, which would look like the bug this test guards.
        val scene = ImageComposeScene(
            width = width,
            height = 2400,
            density = Density(1f),
        ) {
            RankoTheme {
                Box(Modifier.fillMaxSize().background(Color(0xFF15151A))) {
                    Column(
                        modifier = Modifier.width(width.dp).padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        PromptPageHeader(pageKey, lang)
                        PromptPageEditor(pageKey, spec, matrix, lang, "", noopActions())
                    }
                }
            }
        }
        return try {
            scene.render()
            val found = mutableMapOf<String, Pair<Int, Float>>()
            // A pill is the clickable node; its label is the text below it.
            fun labelOf(node: SemanticsNode): String? {
                node.config.getOrNull(SemanticsProperties.Text)?.firstOrNull()?.text?.let { return it }
                return node.children.firstNotNullOfOrNull { labelOf(it) }
            }
            fun walk(node: SemanticsNode) {
                if (node.config.getOrNull(SemanticsActions.OnClick) != null) {
                    labelOf(node)?.let { found.putIfAbsent(it, node.size.width to node.positionInRoot.y) }
                }
                node.children.forEach { walk(it) }
            }
            scene.semanticsOwners.forEach { owner -> owner.rootSemanticsNode?.let { walk(it) } }
            found
        } finally {
            scene.close()
        }
    }

    /** 360 dp pane minus the 14 dp padding on both sides. */
    private val narrowRoom = 360f - 28f

    private fun assertWidthsSurviveShrinking(pageKey: String, lang: PromptLang) {
        if (GraphicsEnvironment.isHeadless()) return
        val wide = pills(pageKey, lang, 830)
        val narrow = pills(pageKey, lang, 360)
        assertTrue(wide.size >= 3, "$pageKey/$lang measured ${wide.size} pills")
        assertEquals(wide.keys, narrow.keys, "$pageKey/$lang lost or gained a pill when the pane narrowed")
        wide.forEach { (label, size) ->
            val narrowed = narrow.getValue(label).first
            assertTrue(narrowed > 0, "$label was measured away at 360 dp: $narrow")
            // A label wider than the pane has to reflow inside its own pill; every other pill keeps
            // the width it was measured with, however narrow the pane is.
            if (size.first <= narrowRoom) {
                assertEquals(size.first, narrowed, "'$label' shrank instead of wrapping to the next line")
            }
        }
    }

    @Test
    fun theModePillsKeepTheirWidthOnANarrowPane() {
        assertWidthsSurviveShrinking("mode", PromptLang.Chinese)
        assertWidthsSurviveShrinking("mode", PromptLang.English)
    }

    @Test
    fun theExposureAndChestPillsKeepTheirWidthOnANarrowPane() {
        assertWidthsSurviveShrinking("exposure", PromptLang.English)
        assertWidthsSurviveShrinking("chest", PromptLang.Chinese)
        assertWidthsSurviveShrinking("belly", PromptLang.English)
        assertWidthsSurviveShrinking("face", PromptLang.Chinese)
    }

    @Test
    fun theThirdModePillWrapsInsteadOfVanishing() {
        if (GraphicsEnvironment.isHeadless()) return
        val narrow = pills("mode", PromptLang.Chinese, 360)
        val rows = narrow.values.map { it.second }.distinct()
        assertEquals(3, rows.size, "the three mode pills: $narrow")
        // A pill narrower than its own label is the failure mode this guards: the old layout gave it
        // a two-character sliver. The shortest shipped mode label is wider than 120 dp.
        narrow.forEach { (label, size) ->
            assertTrue(size.first >= 120, "'$label' measured ${size.first} dp")
        }
    }
}

package com.acite.axlranko.pages.components.automation

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.prompt.BELLY_LEVELS
import com.acite.axlranko.prompt.CHEST_LEVELS
import com.acite.axlranko.prompt.FACE_GROUPS
import com.acite.axlranko.prompt.FacePick
import com.acite.axlranko.prompt.MatrixEntry
import com.acite.axlranko.prompt.PoseFamily
import com.acite.axlranko.prompt.PromptGenerator
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.prompt.PromptMatrix
import com.acite.axlranko.prompt.PromptMode
import com.acite.axlranko.prompt.PromptSpec
import com.acite.axlranko.prompt.SEX_STAGES
import com.acite.axlranko.prompt.SexStage
import com.acite.axlranko.prompt.WizardModel
import com.acite.axlranko.prompt.faceChoice
import com.acite.axlranko.prompt.faceTagsOf
import com.acite.axlranko.prompt.groupTags
import com.acite.axlranko.prompt.stageWeightValue
import com.acite.axlranko.prompt.t
import com.acite.axlranko.prompt.trimNumber
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import kotlin.math.roundToInt

/** Everything a prompt page widget needs to write back into the current spec. */
class PromptEditorActions(
    val setCharacter: (String) -> Unit,
    val setMode: (PromptMode) -> Unit,
    val setExposure: (String, Boolean) -> Unit,
    val setClothingAny: () -> Unit,
    val toggleClothing: (List<String>) -> Unit,
    val setChest: (String) -> Unit,
    val setBelly: (String) -> Unit,
    val setFaceGroup: (String, FacePick) -> Unit,
    val toggleFaceTag: (String, String) -> Unit,
    val setSceneAny: () -> Unit,
    val toggleScene: (List<String>) -> Unit,
    val setFamilyAny: () -> Unit,
    val toggleFamily: (PoseFamily) -> Unit,
    val setVaginalRatio: (Double) -> Unit,
    val setStageWeight: (SexStage, Double) -> Unit,
    val setPoseAny: () -> Unit,
    val togglePose: (List<String>) -> Unit,
    val setCount: (String) -> Unit,
    val setSeedText: (String) -> Unit,
)

/** Short labels for the wizard's page rail (a page's own title is a full sentence). */
private val PAGE_RAIL_LABELS: Map<String, Pair<String, String>> = mapOf(
    "character" to ("角色词" to "Character"),
    "mode" to ("模式" to "Mode"),
    "exposure" to ("暴露" to "Exposure"),
    "clothing" to ("服装" to "Clothing"),
    "chest" to ("胸部" to "Chest"),
    "belly" to ("腹部" to "Belly"),
    "face" to ("表情" to "Face"),
    "scene" to ("场景" to "Scene"),
    "family" to ("家族" to "Family"),
    "ratio" to ("比例" to "Ratio"),
    "stages" to ("阶段" to "Stages"),
    "pose" to ("姿势" to "Pose"),
    "count" to ("数量" to "Count"),
)

fun pageRailLabel(key: String, lang: PromptLang): String {
    val labels = PAGE_RAIL_LABELS[key] ?: return key
    return if (lang == PromptLang.Chinese) labels.first else labels.second
}

private fun pageTitleKey(pageKey: String): String = when (pageKey) {
    "character" -> "character_title"
    "mode" -> "mode_title"
    "exposure" -> "exposure_title"
    "clothing" -> "clothing_title"
    "chest" -> "chest_title"
    "belly" -> "belly_title"
    "face" -> "face_title"
    "scene" -> "scene_title"
    "family" -> "family_title"
    "ratio" -> "ratio_title"
    "stages" -> "stage_title"
    "pose" -> "pose_title"
    "count" -> "count_title"
    else -> "character_title"
}

private fun pageHintKey(pageKey: String): String? = when (pageKey) {
    "character" -> "character_hint"
    "ratio" -> "ratio_hint"
    "stages" -> "stage_hint"
    "face" -> "footer_face"
    else -> null
}

/** The page's heading plus its hint line. */
@Composable
fun PromptPageHeader(pageKey: String, lang: PromptLang) {
    val colors = rankoColors
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
            text = t(lang, pageTitleKey(pageKey)),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = colors.text,
        )
        pageHintKey(pageKey)?.let { key ->
            Text(text = t(lang, key), style = MaterialTheme.typography.labelSmall, color = colors.textDim)
        }
    }
}

/** A yellow strip for the SFW warnings the CLI used to ask about before continuing. */
@Composable
fun PromptWarningBanner(text: String) {
    val colors = rankoColors
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(rankoTokens.panel)
            .background(colors.qualityYellow.copy(alpha = 0.16f))
            .border(1.dp, colors.qualityYellow.copy(alpha = 0.45f), rankoTokens.panel)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Icon(
            Icons.Default.Warning,
            contentDescription = null,
            tint = colors.qualityYellow,
            modifier = Modifier.size(16.dp),
        )
        Text(text, style = MaterialTheme.typography.bodySmall, color = colors.text)
    }
}

@Composable
private fun SingleChoice(
    options: List<Pair<String, String>>,
    selected: String,
    onSelect: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
        options.chunked(3).forEach { row ->
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                row.forEach { (value, label) ->
                    CapsuleChoice(text = label, selected = value == selected, onClick = { onSelect(value) })
                }
            }
        }
    }
}

@Composable
private fun MultiChoice(
    options: List<Pair<String, String>>,
    selected: Set<String>,
    onToggle: (String, Boolean) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
        options.chunked(3).forEach { row ->
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                row.forEach { (value, label) ->
                    CapsuleChoice(
                        text = label,
                        selected = selected.contains(value),
                        onClick = { onToggle(value, !selected.contains(value)) },
                    )
                }
            }
        }
    }
}

@Composable
private fun TickRow(label: String, checked: Boolean, onClick: () -> Unit) {
    val colors = rankoColors
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(rankoTokens.panel)
            .background(if (checked) colors.accentPink.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.04f))
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(
            modifier = Modifier
                .size(14.dp)
                .clip(rankoTokens.capsule)
                .background(if (checked) colors.accentPink else Color.Transparent)
                .border(1.dp, colors.stroke, rankoTokens.capsule),
            contentAlignment = Alignment.Center,
        ) {
            if (checked) {
                Icon(Icons.Default.Check, contentDescription = null, tint = Color.White, modifier = Modifier.size(11.dp))
            }
        }
        Text(text = label, color = colors.text, fontSize = 12.sp, modifier = Modifier.weight(1f))
    }
}

@Composable
private fun WeightSlider(label: String, value: Double, onChange: (Double) -> Unit) {
    val colors = rankoColors
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(text = label, color = colors.text, fontSize = 12.sp, modifier = Modifier.weight(1f))
            Text(
                text = trimNumber((value * 100).roundToInt() / 100.0),
                color = colors.textDim,
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
            )
        }
        Slider(
            value = value.toFloat().coerceIn(0f, 1f),
            onValueChange = { onChange(it.toDouble()) },
            valueRange = 0f..1f,
        )
    }
}

/** One matrix row as a checklist label: the Chinese comment when there is one, else the tags. */
fun optionLabel(entry: MatrixEntry, lang: PromptLang): String {
    val suffix = entry.channel?.let { "  [${it.wire}]" }.orEmpty()
    val body = if (lang == PromptLang.Chinese && entry.comment.isNotEmpty()) entry.comment else entry.blob
    return body + suffix
}

/** The pose pool the current mode and family filter leave, for the pose page. */
fun currentPosePool(matrix: PromptMatrix, spec: PromptSpec): List<MatrixEntry> {
    var poses = PromptGenerator.posePool(matrix, spec.mode)
    if (spec.mode == PromptMode.Sex) poses = PromptGenerator.familyPool(poses, spec)
    return poses
}

/** The widgets of one wizard page; the manifest's row editor renders the same thing. */
@Composable
fun PromptPageEditor(
    pageKey: String,
    spec: PromptSpec,
    matrix: PromptMatrix?,
    lang: PromptLang,
    seedText: String,
    actions: PromptEditorActions,
) {
    val colors = rankoColors
    val dimHint: @Composable (String) -> Unit = { text ->
        Text(text = text, color = colors.textDim, fontSize = 11.sp)
    }

    Column(verticalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
        when (pageKey) {
            "character" -> OutlinedTextField(
                value = spec.character,
                onValueChange = actions.setCharacter,
                singleLine = true,
                colors = rankoFieldColors(),
                modifier = Modifier.fillMaxWidth(),
            )

            "mode" -> SingleChoice(
                options = listOf(
                    PromptMode.Sfw.wire to t(lang, "mode_sfw"),
                    PromptMode.Nsfw.wire to t(lang, "mode_nsfw"),
                    PromptMode.Sex.wire to t(lang, "mode_sex"),
                ),
                selected = spec.mode.wire,
                onSelect = { wire -> PromptMode.ofWire(wire)?.let(actions.setMode) },
            )

            "exposure" -> {
                MultiChoice(
                    options = listOf(
                        "covered" to t(lang, "exp_covered"),
                        "casual" to t(lang, "exp_casual"),
                        "revealing" to t(lang, "exp_revealing"),
                        "open" to t(lang, "exp_open"),
                        "nude" to t(lang, "exp_nude"),
                    ),
                    selected = spec.exposure.toSet(),
                    onToggle = actions.setExposure,
                )
                if (spec.exposure.isEmpty()) {
                    dimHint(uiText(lang, "exposure_empty"))
                }
            }

            "clothing", "scene", "pose" -> {
                val entries = when (pageKey) {
                    "clothing" -> matrix?.let { WizardModel.allowedClothing(it, spec.exposure) }.orEmpty()
                    "scene" -> matrix?.scenes.orEmpty()
                    else -> matrix?.let { currentPosePool(it, spec) }.orEmpty()
                }
                val anyOn = when (pageKey) {
                    "clothing" -> spec.clothingAny
                    "scene" -> spec.sceneAny
                    else -> spec.poseAny
                }
                val keys = when (pageKey) {
                    "clothing" -> spec.clothingKeys
                    "scene" -> spec.sceneKeys
                    else -> spec.poseKeys
                }
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CapsuleChoice(
                        text = t(lang, "any"),
                        selected = anyOn,
                        onClick = {
                            when (pageKey) {
                                "clothing" -> actions.setClothingAny()
                                "scene" -> actions.setSceneAny()
                                else -> actions.setPoseAny()
                            }
                        },
                    )
                    dimHint(
                        if (anyOn) {
                            uiText(lang, "pool_any")
                        } else {
                            t(lang, "manifest_selected").replace("{n}", keys.size.toString())
                        },
                    )
                }
                if (entries.isEmpty()) {
                    dimHint(uiText(lang, "no_entries"))
                }
                Column(
                    modifier = Modifier.fillMaxWidth().heightIn(max = 340.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    entries.forEach { entry ->
                        TickRow(
                            label = optionLabel(entry, lang),
                            checked = keys.contains(entry.key),
                            onClick = {
                                when (pageKey) {
                                    "clothing" -> actions.toggleClothing(entry.key)
                                    "scene" -> actions.toggleScene(entry.key)
                                    else -> actions.togglePose(entry.key)
                                }
                            },
                        )
                    }
                }
            }

            "chest" -> SingleChoice(
                options = CHEST_LEVELS.map { it to t(lang, "chest_" + chestKeySuffix(it)) },
                selected = spec.chest,
                onSelect = actions.setChest,
            )

            "belly" -> SingleChoice(
                options = BELLY_LEVELS.map { it to t(lang, "belly_" + bellyKeySuffix(it)) },
                selected = spec.belly,
                onSelect = actions.setBelly,
            )

            "face" -> FACE_GROUPS.forEach { group ->
                PorcelainCard {
                    Column(
                        modifier = Modifier.fillMaxWidth().padding(10.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text(
                            text = if (lang == PromptLang.Chinese) group.zh else group.en,
                            color = colors.text,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 13.sp,
                        )
                        val value = faceChoice(spec, group)
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            CapsuleChoice(
                                text = t(lang, "any"),
                                selected = value == FacePick.ANY,
                                onClick = { actions.setFaceGroup(group.id, FacePick.ANY) },
                            )
                            CapsuleChoice(
                                text = t(lang, "face_skip"),
                                selected = value == FacePick.None,
                                onClick = { actions.setFaceGroup(group.id, FacePick.None) },
                            )
                            if (value is FacePick.Tags) {
                                dimHint(
                                    uiText(lang, "face_candidates"),
                                )
                            }
                        }
                        val ticked = faceTagsOf(value).toSet()
                        groupTags(group).chunked(3).forEach { row ->
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                row.forEach { tag ->
                                    val option = group.options.first { it.tag == tag }
                                    CapsuleChoice(
                                        text = if (lang == PromptLang.Chinese) option.zh else option.en,
                                        selected = ticked.contains(tag),
                                        onClick = { actions.toggleFaceTag(group.id, tag) },
                                    )
                                }
                            }
                        }
                    }
                }
            }

            "family" -> {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CapsuleChoice(text = t(lang, "any"), selected = spec.familyAny, onClick = actions.setFamilyAny)
                    dimHint(
                        if (spec.familyAny) {
                            uiText(lang, "pool_any")
                        } else {
                            t(lang, "manifest_selected").replace("{n}", spec.families.size.toString())
                        },
                    )
                }
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    PoseFamily.ORDER.chunked(2).forEach { row ->
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            row.forEach { family ->
                                CapsuleChoice(
                                    text = t(lang, "fam_" + family.wire),
                                    selected = spec.families.contains(family),
                                    onClick = { actions.toggleFamily(family) },
                                )
                            }
                        }
                    }
                }
            }

            "ratio" -> {
                WeightSlider(label = t(lang, "ratio_title"), value = spec.vaginalRatio, onChange = actions.setVaginalRatio)
                val percent = (spec.vaginalRatio * 100).roundToInt()
                dimHint(
                    if (lang == PromptLang.Chinese) {
                        uiText(lang, "ratio_split").replace("{v}", percent.toString()).replace("{a}", (100 - percent).toString())
                    } else {
                        "vaginal $percent% · anal ${100 - percent}%"
                    },
                )
            }

            "stages" -> {
                SEX_STAGES.forEach { stage ->
                    WeightSlider(
                        label = t(lang, stage.stringKey),
                        value = spec.stageWeights[stage] ?: 0.0,
                    ) { actions.setStageWeight(stage, it) }
                }
                if (SEX_STAGES.all { stageWeightValue(spec, it) == 0 }) {
                    Text(text = t(lang, "manifest_stage_fallback"), color = colors.qualityYellow, fontSize = 11.sp)
                }
            }

            "count" -> Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.Bottom,
            ) {
                Column(modifier = Modifier.width(110.dp)) {
                    dimHint(t(lang, "count_title"))
                    OutlinedTextField(
                        value = spec.count.toString(),
                        onValueChange = actions.setCount,
                        singleLine = true,
                        colors = rankoFieldColors(),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Column(modifier = Modifier.width(180.dp)) {
                    dimHint(uiText(lang, "seed_hint"))
                    OutlinedTextField(
                        value = seedText,
                        onValueChange = actions.setSeedText,
                        singleLine = true,
                        colors = rankoFieldColors(),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }

            else -> dimHint("$pageKey")
        }
    }
}

private fun chestKeySuffix(level: String): String = when (level) {
    "breasts_out" -> "out"
    "covered", "cleavage", "nipples" -> level
    else -> "auto"
}

private fun bellyKeySuffix(level: String): String = when (level) {
    "covered", "midriff", "navel" -> level
    else -> "auto"
}

/** The stepper beside the wizard page: one row per page that applies to the current spec. */
@Composable
fun PromptStepRail(keys: List<String>, currentIndex: Int, lang: PromptLang, onSelect: (Int) -> Unit) {
    val colors = rankoColors
    // No scroll of its own: the page pane already scrolls, and a nested scroll would be measured
    // against an unbounded height.
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        keys.forEachIndexed { index, key ->
            val selected = index == currentIndex
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(rankoTokens.panel)
                    .background(if (selected) colors.accentPink.copy(alpha = 0.16f) else Color.Transparent)
                    .clickable { onSelect(index) }
                    .padding(horizontal = 8.dp, vertical = 5.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    text = (index + 1).toString(),
                    color = if (selected) colors.accentPink else colors.textDim,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                )
                Text(
                    text = pageRailLabel(key, lang),
                    color = if (selected) colors.text else colors.textDim,
                    fontSize = 12.sp,
                    fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                )
            }
        }
    }
}

/** The wizard's footer: back / next, disabled at either end. */
@Composable
fun PromptWizardFooter(atStart: Boolean, atEnd: Boolean, lang: PromptLang, onBack: () -> Unit, onNext: () -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        CapsuleButton(
            text = uiText(lang, "back"),
            onClick = onBack,
            enabled = !atStart,
            compact = true,
        )
        CapsuleButton(
            text = uiText(lang, "next"),
            onClick = onNext,
            enabled = !atEnd,
            compact = true,
            emphasized = !atEnd,
        )
    }
}

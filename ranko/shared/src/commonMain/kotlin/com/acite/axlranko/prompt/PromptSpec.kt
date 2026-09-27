package com.acite.axlranko.prompt

import kotlin.math.abs
import kotlin.math.floor

/** The wizard's UI text for one key; a key with no entry falls back to itself. */
fun t(lang: PromptLang, key: String): String = PromptStrings.of(lang)[key] ?: key

/** Python's `round`: a tie goes to the even neighbour, not away from zero. */
internal fun pythonRound(value: Double): Long {
    val floorValue = floor(value)
    val diff = value - floorValue
    val base = floorValue.toLong()
    return when {
        diff > 0.5 -> base + 1
        diff < 0.5 -> base
        base % 2 == 0L -> base
        else -> base + 1
    }
}

fun defaultExposure(mode: PromptMode): List<String> = MODE_EXPOSURE_DEFAULTS.getValue(mode)

/** One stage gets all the weight unless the user says otherwise; `during` matches the old behaviour. */
fun defaultStageWeights(): Map<SexStage, Double> =
    SEX_STAGES.associateWith { if (it == SexStage.During) 1.0 else 0.0 }

/**
 * The whole wizard configuration. Mutable on purpose: the ported page functions edit one field
 * each, exactly like `PromptSpec` in `tools/gen_prompts.py`.
 */
data class PromptSpec(
    var character: String = "",
    var mode: PromptMode = PromptMode.Sfw,
    var exposure: List<String> = MODE_EXPOSURE_DEFAULTS.getValue(PromptMode.Sfw),
    var clothingAny: Boolean = true,
    var clothingKeys: Set<List<String>> = emptySet(),
    var sceneAny: Boolean = true,
    var sceneKeys: Set<List<String>> = emptySet(),
    var poseAny: Boolean = true,
    var poseKeys: Set<List<String>> = emptySet(),
    var familyAny: Boolean = true,
    var families: Set<PoseFamily> = emptySet(),
    var vaginalRatio: Double = PromptLimits.VAGINAL_RATIO_DEFAULT,
    var stageWeights: Map<SexStage, Double> = defaultStageWeights(),
    var chest: String = "auto",
    var belly: String = "auto",
    var face: Map<String, FacePick> = defaultFace(),
    var count: Int = PromptLimits.COUNT_DEFAULT,
) {
    /** A copy that shares no mutable collection, so an editor cannot write through to the original. */
    fun deepCopy(): PromptSpec = copy(
        exposure = exposure.toList(),
        clothingKeys = clothingKeys.toSet(),
        sceneKeys = sceneKeys.toSet(),
        poseKeys = poseKeys.toSet(),
        families = families.toSet(),
        stageWeights = stageWeights.toMap(),
        face = LinkedHashMap(face),
    )

    val nudeOnly: Boolean get() = exposure.toSet() == setOf("nude")
}

/** What a fresh wizard starts from; chest/belly defaults follow the mode. */
fun defaultSpec(): PromptSpec = PromptSpec(
    character = "",
    mode = PromptMode.Sfw,
    exposure = defaultExposure(PromptMode.Sfw),
    chest = CHEST_DEFAULT.getValue(PromptMode.Sfw),
    belly = BELLY_DEFAULT.getValue(PromptMode.Sfw),
)

/** A stage weight as an integer in 0..1000, the unit `pick_stage` draws from. */
fun stageWeightValue(spec: PromptSpec, stage: SexStage): Int {
    val value = spec.stageWeights[stage] ?: 0.0
    if (value.isNaN()) return 0
    return pythonRound(value * 1000).coerceIn(0L, 1000L).toInt()
}

/** Parse a ratio the CLI way: a value above 1 is a percentage. */
fun parseVaginalRatio(raw: String, default: Double = PromptLimits.VAGINAL_RATIO_DEFAULT): Double {
    val text = raw.trim()
    if (text.isEmpty()) return default
    var value = text.toDoubleOrNull() ?: return default
    if (value > 1.0) value /= 100.0
    return value.coerceIn(0.0, 1.0)
}

/** The wizard's confirm/manifest ratio line, e.g. `0.5`. */
internal fun trimNumber(value: Double): String =
    if (value == value.toLong().toDouble() && abs(value) < 1e15) value.toLong().toString()
    else value.toString()

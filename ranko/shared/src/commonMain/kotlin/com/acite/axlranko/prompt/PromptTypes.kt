package com.acite.axlranko.prompt

/**
 * Shared vocabulary of the prompt wizard, ported from `tools/gen_prompts.py`.
 *
 * The Python script's string constants become enums here: the same values, compared by identity
 * instead of by spelling. Wire names stay exactly what the CLI writes into a profile.
 */
enum class PromptLang(val wire: String) {
    Chinese("chinese"),
    English("english"),
}

enum class PromptMode(val wire: String) {
    Sfw("sfw"),
    Nsfw("nsfw"),
    Sex("sex");

    companion object {
        val ORDER: List<PromptMode> = entries.toList()

        fun ofWire(wire: String): PromptMode? = entries.firstOrNull { it.wire == wire }
    }
}

/** Which hole a pose can take; the matrix writes these as `both` / `anal only` / `vaginal only` / `none`. */
enum class PromptChannel(val wire: String) {
    Both("both"),
    Anal("anal"),
    Vaginal("vaginal"),
    None("none");

    companion object {
        /** The matrix spellings, e.g. `anal only`. */
        fun ofMatrix(raw: String): PromptChannel? = when (raw) {
            "both" -> Both
            "anal only" -> Anal
            "vaginal only" -> Vaginal
            "none" -> None
            else -> null
        }
    }
}

enum class SexStage(val wire: String) {
    Pose("pose"),
    Before("before"),
    During("during"),
    ObjectInsertion("object_insertion"),
    Fingering("fingering"),
    Ejaculation("ejaculation"),
    After("after"),
    Done("done");

    /** String-table key for this stage's label, e.g. `stage_during`. */
    val stringKey: String get() = "stage_$wire"

    companion object {
        val ORDER: List<SexStage> = entries.toList()
    }
}

enum class PoseFamily(val wire: String) {
    Face("face"),
    Behind("behind"),
    StandBehind("stand_behind"),
    SideBehind("side_behind"),
    GirlOnTop("girl_on_top"),
    Hold("hold");

    companion object {
        val ORDER: List<PoseFamily> = entries.toList()
    }
}

object PromptLimits {
    const val PREFER_WEIGHT = 5
    const val COUNT_MIN = 1
    const val COUNT_MAX = 200
    const val COUNT_DEFAULT = 10
    const val VAGINAL_RATIO_DEFAULT = 0.5
    const val NUDE_SENTINEL = "__nude__"
    const val OPEN_MARK = "(open clothes)"

    /**
     * The anal channel word is written weighted: next to a pose the model reads as vaginal, a bare
     * `anal` is the element that goes missing, so the prompt asks for it at 1.2.
     */
    const val ANAL_CHANNEL_TAG = "(anal:1.2)"
}

val SEX_STAGES: List<SexStage> = SexStage.ORDER
val STAGES_WITH_PARTNER: Set<SexStage> =
    setOf(SexStage.Before, SexStage.During, SexStage.Ejaculation, SexStage.After)
val STAGES_PENETRATING: Set<SexStage> = setOf(SexStage.During, SexStage.Ejaculation)
val STAGES_OBJECT: Set<SexStage> = setOf(SexStage.ObjectInsertion, SexStage.Fingering)
val HOLE_CHANNELS: Set<PromptChannel> = setOf(PromptChannel.Anal, PromptChannel.Vaginal)

/** Exposure levels, in wizard order. */
val EXPOSURE_LEVELS: List<String> = listOf("covered", "casual", "revealing", "open", "nude")
val HIGH_EXPOSURE: Set<String> = setOf("revealing", "open", "nude")
val CLOTHING_GROUPS: List<String> = listOf("covered", "casual", "revealing")

val MODE_EXPOSURE_DEFAULTS: Map<PromptMode, List<String>> = mapOf(
    PromptMode.Sfw to listOf("covered", "casual"),
    PromptMode.Nsfw to listOf("revealing", "open", "nude"),
    PromptMode.Sex to listOf("open"),
)

val CHEST_LEVELS: List<String> = listOf("covered", "cleavage", "breasts_out", "nipples", "auto")
val BELLY_LEVELS: List<String> = listOf("covered", "midriff", "navel", "auto")
val CHEST_DEFAULT: Map<PromptMode, String> =
    mapOf(PromptMode.Sfw to "covered", PromptMode.Nsfw to "auto", PromptMode.Sex to "auto")
val BELLY_DEFAULT: Map<PromptMode, String> =
    mapOf(PromptMode.Sfw to "covered", PromptMode.Nsfw to "auto", PromptMode.Sex to "auto")

val ASS_MARKERS: List<String> = listOf(
    "from behind",
    "doggystyle",
    "prone bone",
    "reverse cowgirl",
    "reverse suspended",
    "top-down bottom-up",
    "full nelson",
    "spooning",
)
val SPREAD_MARKERS: List<String> = listOf(
    "spread legs",
    "legs up",
    "folded",
    "knees to chest",
    "legs over head",
    "leg lift",
    "squatting",
)
val TOP_MARKERS: List<String> = listOf("girl on top", "cowgirl", "sitting on lap")

/**
 * Poses that hold both legs up against the body, so both arms are accounted for: the girl holds her
 * own thighs (`mating press`, `anvil position`) or they are pinned (`full nelson`). A hand action on
 * one of these has no arm left to perform it, and the model fills the gap with a third one, which is
 * why [PromptGenerator.poseHoldsLegs] sends those draws to a partner instead of `solo`.
 *
 * Deliberately narrower than [SPREAD_MARKERS]: `spread legs`, `squatting` and `leg lift` hold
 * nothing, so their hands are free.
 */
val LEGS_HELD_MARKERS: List<String> = listOf(
    "mating press",
    "anvil position",
    "full nelson",
    "legs up",
    "folded",
    "knees to chest",
    "legs over head",
)
val LIE_MARKERS: List<String> = listOf(
    "lying",
    "on back",
    "on stomach",
    "on side",
    "sleeping",
    "missionary",
    "mating press",
    "anvil position",
    "spooning",
    "prone bone",
    "seventh posture",
)

val SECTION_NAMES: Set<String> = setOf("POSES", "CLOTHING", "SCENE", "SUFFIX", "SFW_POSES")
val CLOTHING_GROUP_RE = Regex("^\\[(covered|casual|revealing)]\$")
val CHANNEL_RE = Regex("^(.+?)\\s*:\\s*(both|anal only|vaginal only|none)\\s*\$")

/** Tags the wizard never appends, whatever the matrix says. */
val RATING_TAGS: Set<String> = setOf(
    "sfw",
    "nsfw",
    "explicit",
    "sensitive",
    "questionable",
    "rating:general",
    "rating:sensitive",
    "rating:questionable",
    "rating:explicit",
)
val QUALITY_TAGS: Set<String> = setOf(
    "masterpiece",
    "best quality",
    "amazing quality",
    "newest",
    "absurdres",
    "highres",
    "ultra detailed",
    "extremely detailed",
    "8k",
)
val YEAR_RE = Regex("^year 20\\d\\d\$")
val SCORE_RE = Regex("^score_\\d")

class MatrixException(message: String) : Exception(message)

class ProfileException(message: String) : Exception(message)

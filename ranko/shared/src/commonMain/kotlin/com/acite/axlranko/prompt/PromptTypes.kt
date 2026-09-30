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

/** A place a pose names in its own tags, and the scene words that satisfy it. */
data class PosePlace(val marker: String, val scenes: List<String>)

/**
 * The places [PromptGenerator.posePlaces] looks for. A row that names one pairs only with a scene
 * carrying it, because the pose's own words are more specific than the coarse
 * [PromptGenerator.poseLocus]: `sitting, resting head on desk, sleeping` is read as lying (the
 * `sleeping` marker) and used to be offered a sofa, and `lying, on back, on bed` a beach.
 *
 * `on desk` accepts a cafe table too: the scene vocabulary has no other table, and the rows that
 * name one ask for 教室课桌 or 咖啡靠窗桌. A row naming two places takes either — `standing, leaning on
 * railing, looking outside` is a rooftop or a window. The marker is matched as a substring of the
 * pose's tag blob, so `on bed` also catches `sitting on bed` and the bare `desk` / `railing` /
 * `window` / `steam` entries cover the shorter spellings.
 */
val POSE_PLACES: List<PosePlace> = listOf(
    PosePlace("on bed", listOf("bed")),
    PosePlace("on sofa", listOf("sofa")),
    PosePlace("on bench", listOf("bench")),
    PosePlace("on sand", listOf("sand")),
    PosePlace("on grass", listOf("grass")),
    PosePlace("on desk", listOf("desk", "cafe")),
    PosePlace("at desk", listOf("desk", "cafe")),
    PosePlace("desk", listOf("desk", "cafe")),
    PosePlace("on railing", listOf("rooftop")),
    PosePlace("railing", listOf("rooftop")),
    PosePlace("against tree", listOf("forest", "park")),
    PosePlace("at water", listOf("pool", "ocean", "onsen")),
    PosePlace("in water", listOf("pool", "ocean", "onsen")),
    PosePlace("wading", listOf("pool", "ocean", "onsen")),
    PosePlace("soaking", listOf("pool", "ocean", "onsen")),
    PosePlace("steam", listOf("onsen")),
    PosePlace("window", listOf("window")),
    PosePlace("looking outside", listOf("window")),
    PosePlace("praying", listOf("shrine")),
    PosePlace("shrine", listOf("shrine")),
)

/**
 * What a `QUESTIONABLE_POSES` row says about its chest, and about its bottom: those rows carry
 * their own clothing and exposure, so the sampler adds no outfit, no `nude`, no `open clothes` and
 * no chest/belly level for them. It only fills the half a row leaves open — aggressively, see
 * [PromptGenerator.stateFill]. The words are matched against the row's whole tag blob and are only
 * ever consulted for a row whose [MatrixEntry.selfStated] is set, so a word shared with an
 * `SFW_POSES` or `POSES` row (`arms up`, `stretching`) cannot change those rows.
 */
val POSE_CHEST_WORDS: List<String> = listOf(
    "breasts out",
    "nipples",
    "topless",
    "sideboob",
    "underboob",
    "one breast out",
    "breasts hanging",
    "breasts",
    "open shirt",
    "shirt lift",
    "clothes lift",
    "clothes pull",
    "pulling down clothes",
    "unbuttoning",
    "covering breasts",
    "bra",
    "towel",
    "breast hold",
    "hands on own chest",
    "holding clothes",
    "wet shirt",
    "see-through",
    "cleavage",
    "downblouse",
    "no bra",
    "off shoulder",
    "backless",
    "after bath",
    "pajamas pull",
    "nightgown slip",
)

val POSE_BOTTOM_WORDS: List<String> = listOf(
    "panties",
    "ass",
    "skirt",
    "bikini",
    "swimsuit",
    "no pants",
    "covering crotch",
    "legs together",
    "heels together",
    "knees apart",
    "yukata",
    "pajamas",
    "nightgown",
    "stockings",
)

/** A chest that hangs rather than faces the camera, so [PromptGenerator.stateFill] names its shape. */
val CHEST_HANGING_MARKERS: List<String> = listOf("all fours", "bent over", "on stomach")

val SECTION_NAMES: Set<String> = setOf(
    "POSES",
    "CLOTHING",
    "SCENE",
    "SUFFIX",
    "SFW_POSES",
    "QUESTIONABLE_POSES",
    "FIGURE",
    "PUSSY_SHAPE",
    "PUSSY_HAIR",
)
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

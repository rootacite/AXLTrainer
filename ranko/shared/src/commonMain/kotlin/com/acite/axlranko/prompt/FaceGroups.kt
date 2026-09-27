package com.acite.axlranko.prompt

/** A tag the face page can offer, and the modes it is allowed in. */
data class FaceOption(
    val tag: String,
    val modes: Set<PromptMode>,
    val zh: String,
    val en: String,
)

/**
 * One mutually exclusive axis of the face page: at most one of its tags reaches a prompt.
 *
 * [default] is `any` or `none` (`any` = the group's whole mode pool, rolled per prompt).
 */
data class FaceGroup(
    val id: String,
    val zh: String,
    val en: String,
    val options: List<FaceOption>,
    val default: String = FACE_ANY,
) {
    val label: String get() = zh
}

const val FACE_ANY = "any"
const val FACE_NONE = "none"

/** `any` / `none`, or the explicit candidate list for one group. */
sealed interface FacePick {
    data class Tags(val tags: List<String>) : FacePick
    data object Any : FacePick
    data object None : FacePick

    companion object {
        val ANY: FacePick = Any
        val NONE: FacePick = None
    }
}

private val ALL_MODES = PromptMode.ORDER.toSet()
private val NSFW_MODES = setOf(PromptMode.Nsfw, PromptMode.Sex)
private val SEX_MODES = setOf(PromptMode.Sex)

val FACE_GROUPS: List<FaceGroup> = listOf(
    FaceGroup(
        "expression",
        "总表情",
        "Expression",
        listOf(
            FaceOption("smile", ALL_MODES, "微笑 — smile", "smile"),
            FaceOption("grin", ALL_MODES, "咧嘴笑 — grin", "grin"),
            FaceOption("smirk", ALL_MODES, "坏笑 — smirk", "smirk"),
            FaceOption("shy", ALL_MODES, "害羞 — shy", "shy"),
            FaceOption("embarrassed", ALL_MODES, "尴尬 — embarrassed", "embarrassed"),
            FaceOption("angry", ALL_MODES, "生气 — angry", "angry"),
            FaceOption("surprised", ALL_MODES, "惊讶 — surprised", "surprised"),
            FaceOption("scared", ALL_MODES, "害怕 — scared", "scared"),
            FaceOption("naughty face", NSFW_MODES, "坏脸 — naughty face", "naughty face"),
            FaceOption("pain", NSFW_MODES, "痛苦 — pain", "pain"),
            FaceOption("ahegao", NSFW_MODES, "阿嘿颜 — ahegao", "ahegao"),
            FaceOption("orgasm", NSFW_MODES, "高潮 — orgasm", "orgasm"),
        ),
    ),
    FaceGroup(
        "gaze",
        "视线",
        "Gaze",
        listOf(
            FaceOption("looking at viewer", ALL_MODES, "看镜头 — looking at viewer", "looking at viewer"),
            FaceOption("looking away", ALL_MODES, "视线移开 — looking away", "looking away"),
        ),
        FACE_NONE,
    ),
    FaceGroup(
        "eyes",
        "眼睛状态",
        "Eye state",
        listOf(
            FaceOption("open eyes", ALL_MODES, "睁眼 — open eyes", "open eyes"),
            FaceOption("closed eyes", ALL_MODES, "闭眼 — closed eyes", "closed eyes"),
            FaceOption("half-closed eyes", ALL_MODES, "半闭眼 — half-closed eyes", "half-closed eyes"),
            FaceOption("empty eyes", ALL_MODES, "空洞眼 — empty eyes", "empty eyes"),
            FaceOption("sparkling eyes", ALL_MODES, "亮晶晶 — sparkling eyes", "sparkling eyes"),
            FaceOption("wide-eyed", ALL_MODES, "睁大眼 — wide-eyed", "wide-eyed"),
            FaceOption("one eye closed", ALL_MODES, "闭一只眼 — one eye closed", "one eye closed"),
            FaceOption("squinting", ALL_MODES, "眯眼 — squinting", "squinting"),
            FaceOption("rolling eyes", NSFW_MODES, "翻白眼 — rolling eyes", "rolling eyes"),
            FaceOption("heart-shaped pupils", NSFW_MODES, "爱心瞳 — heart-shaped pupils", "heart-shaped pupils"),
            FaceOption("spiral eyes", SEX_MODES, "螺旋眼 — spiral eyes", "spiral eyes"),
        ),
    ),
    FaceGroup(
        "mouth",
        "嘴状态",
        "Mouth",
        listOf(
            FaceOption("open mouth", ALL_MODES, "张嘴 — open mouth", "open mouth"),
            FaceOption("closed mouth", ALL_MODES, "闭嘴 — closed mouth", "closed mouth"),
            FaceOption("tongue out", NSFW_MODES, "吐舌 — tongue out", "tongue out"),
            FaceOption("clenched teeth", NSFW_MODES, "咬牙 — clenched teeth", "clenched teeth"),
            FaceOption("drooling", NSFW_MODES, "流口水 — drooling", "drooling"),
            FaceOption("heavy breathing", NSFW_MODES, "喘息 — heavy breathing", "heavy breathing"),
        ),
        FACE_NONE,
    ),
    FaceGroup(
        "blush",
        "脸红",
        "Blush",
        listOf(FaceOption("blush", ALL_MODES, "脸红 — blush", "blush")),
        FACE_NONE,
    ),
    FaceGroup(
        "tears",
        "眼泪",
        "Tears",
        listOf(
            FaceOption("tears", ALL_MODES, "含泪 — tears", "tears"),
            FaceOption("crying", ALL_MODES, "哭泣 — crying", "crying"),
        ),
        FACE_NONE,
    ),
)

val FACE_GROUPS_BY_ID: Map<String, FaceGroup> = FACE_GROUPS.associateBy { it.id }

val FACE_TAG_GROUP: Map<String, String> =
    FACE_GROUPS.flatMap { group -> group.options.map { it.tag to group.id } }.toMap()

/** Cross-group contradictions; pairs inside one group are already exclusive there. */
val FACE_CONFLICTS: Map<String, Set<String>> = mapOf(
    "closed eyes" to setOf("looking at viewer", "looking away"),
    "looking at viewer" to setOf("closed eyes"),
    "looking away" to setOf("closed eyes"),
)

val SEX_FACE_TAGS: Set<String> = setOf(
    "ahegao",
    "orgasm",
    "pain",
    "clenched teeth",
    "rolling eyes",
    "spiral eyes",
    "naughty face",
    "drooling",
    "heavy breathing",
    "tongue out",
    "heart-shaped pupils",
)

/** The stored value a fresh wizard starts from: each group's own default. */
fun defaultFace(): MutableMap<String, FacePick> =
    FACE_GROUPS.associateTo(linkedMapOf()) { it.id to facePickOfString(it.default) }

fun groupTags(group: FaceGroup): List<String> = group.options.map { it.tag }

fun groupPool(group: FaceGroup, mode: PromptMode): List<String> =
    group.options.filter { mode in it.modes }.map { it.tag }

/** `any` / `none` as a [FacePick]; anything else is an error, like the CLI's profile loader. */
fun facePickOfString(value: String): FacePick = when (value) {
    FACE_ANY -> FacePick.ANY
    FACE_NONE -> FacePick.NONE
    else -> throw ProfileException("face value must be '$FACE_ANY', '$FACE_NONE' or a list of tags")
}

fun faceChoiceOf(face: Map<String, FacePick>, group: FaceGroup): FacePick =
    face[group.id] ?: facePickOfString(group.default)

fun faceChoice(spec: PromptSpec, group: FaceGroup): FacePick = faceChoiceOf(spec.face, group)

/** The explicitly chosen tags; empty for the `any` and `none` sentinels. */
fun faceTagsOf(value: FacePick): List<String> = when (value) {
    is FacePick.Tags -> value.tags
    else -> emptyList()
}

fun selectedFaceTags(face: Map<String, FacePick>): Set<String> =
    face.values.flatMap { faceTagsOf(it) }.toSet()

fun faceConflict(tag: String, other: Collection<String>): Boolean =
    FACE_CONFLICTS[tag].orEmpty().any { other.contains(it) }

fun faceLabel(item: FaceOption, lang: PromptLang): String =
    if (lang == PromptLang.Chinese) item.zh else item.en

/** (any ticked?, per-tag flags) for the wizard page, from a stored group value. */
fun faceSelection(group: FaceGroup, value: FacePick): Pair<Boolean, List<Boolean>> {
    val flags = MutableList(group.options.size) { false }
    when (value) {
        FacePick.Any -> return true to flags
        FacePick.None -> return false to flags
        is FacePick.Tags -> {
            val tags = groupTags(group)
            value.tags.forEach { tag ->
                val index = tags.indexOf(tag)
                if (index >= 0) flags[index] = true
            }
        }
    }
    return false to flags
}

/** The stored group value for a page answer: any > tags > nothing. */
fun faceValue(group: FaceGroup, anyOn: Boolean, flags: List<Boolean>): FacePick {
    if (anyOn) return FacePick.ANY
    val tags = groupTags(group).filterIndexed { index, _ -> flags.getOrElse(index) { false } }
    return if (tags.isEmpty()) FacePick.NONE else FacePick.Tags(tags)
}

fun needsSfwFaceWarning(mode: PromptMode, keys: Collection<String>): Boolean =
    mode == PromptMode.Sfw && keys.any { SEX_FACE_TAGS.contains(it) }

fun faceValueLabel(value: FacePick, lang: PromptLang): String = when (value) {
    FacePick.Any -> t(lang, "any")
    FacePick.None -> t(lang, "face_skip")
    is FacePick.Tags -> value.tags.joinToString("/")
}

fun faceSummary(face: Map<String, FacePick>, lang: PromptLang): String =
    FACE_GROUPS.joinToString(", ") { group ->
        val name = if (lang == PromptLang.Chinese) group.zh else group.en
        "$name=${faceValueLabel(faceChoiceOf(face, group), lang)}"
    }

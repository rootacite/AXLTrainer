package com.acite.axlranko.prompt

import kotlin.math.floor
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive

/**
 * The named wizard profiles under repo-root `prompt_profiles/`, ported from `tools/gen_prompts.py`.
 *
 * The file format is version 3; profiles written by the older script versions (v1 without the
 * grouped face page, v2 with a single tag per group) are upgraded on load and never rewritten
 * until the user saves again.
 */
@OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)
object ProfileCodec {

    const val PROFILE_VERSION = 3

    private val json = Json { prettyPrint = true; prettyPrintIndent = "  " }

    private val UNSAFE_NAME_RE = Regex("[\\\\/:*?\"<>|\\x00-\\x1f]")

    /** Filesystem-safe stem for a profile name (unicode is kept), like `safe_profile_name`. */
    fun safeProfileName(name: String): String =
        UNSAFE_NAME_RE.replace(name, "_").trim().trim('.')

    fun requireProfileName(name: String): String {
        val safe = safeProfileName(name)
        if (safe.isEmpty()) throw ProfileException("invalid profile name: '$name'")
        return safe
    }

    /** One profile as it is stored: the pretty JSON text the filesystem store writes verbatim. */
    fun profileText(name: String, spec: PromptSpec): String {
        val payload = JsonObject(
            linkedMapOf(
                "version" to JsonPrimitive(PROFILE_VERSION),
                "name" to JsonPrimitive(name.trim()),
                "spec" to specToJson(spec),
            ),
        )
        return json.encodeToString(JsonObject.serializer(), payload) + "\n"
    }

    /** A profile read back: its display name plus the spec, after any version upgrade. */
    data class LoadedProfile(val name: String, val spec: PromptSpec, val notes: List<String>)

    fun loadProfile(text: String, fallbackName: String): LoadedProfile {
        val payload = parseObject(text, fallbackName)
        val rawSpec = payload["spec"] ?: throw ProfileException("$fallbackName: missing spec")
        val specObject = rawSpec as? JsonObject
            ?: throw ProfileException("$fallbackName: profile spec must be a JSON object")
        val version = versionOf(payload, fallbackName)
        val notes = mutableListOf<String>()
        val upgraded = when (version) {
            1 -> upgradeV1(specObject, notes)
            2 -> upgradeV2(specObject)
            PROFILE_VERSION -> specObject
            else -> throw ProfileException("$fallbackName: unsupported profile version $version")
        }
        if (upgraded !== specObject) {
            notes.add(0, "upgraded from v$version to v$PROFILE_VERSION")
        }
        val name = payload["name"]?.jsonPrimitive?.contentOrNull?.trim().orEmpty().ifEmpty { fallbackName }
        return LoadedProfile(name = name, spec = specFromJson(upgraded), notes = notes)
    }

    /** The profile's declared format version; a missing key means the current format, like the CLI. */
    private fun versionOf(payload: JsonObject, label: String): Int {
        val element = payload["version"] ?: return PROFILE_VERSION
        if (element is JsonNull) return PROFILE_VERSION
        val number = asNumber(element)
        val version = number?.takeIf { it == floor(it) }?.toInt()
        return version ?: throw ProfileException("$label: unsupported profile version $element")
    }

    /** A JSON number, or null for a string, boolean, array or object. */
    private fun asNumber(element: JsonElement?): Double? {
        val primitive = element as? JsonPrimitive ?: return null
        if (primitive.isString) return null
        return primitive.doubleOrNull
    }

    private fun parseObject(text: String, label: String): JsonObject {
        val parsed = try {
            Json.parseToJsonElement(text)
        } catch (exc: Exception) {
            throw ProfileException("$label: invalid JSON (${exc.message})")
        }
        return parsed as? JsonObject ?: throw ProfileException("$label: profile must be a JSON object")
    }

    // --- spec <-> JSON ---

    fun specToJson(spec: PromptSpec): JsonObject = JsonObject(
        linkedMapOf(
            "character" to JsonPrimitive(spec.character),
            "mode" to JsonPrimitive(spec.mode.wire),
            "exposure" to JsonArray(spec.exposure.map { JsonPrimitive(it) }),
            "clothing_any" to JsonPrimitive(spec.clothingAny),
            "clothing_keys" to keysToJson(spec.clothingKeys),
            "scene_any" to JsonPrimitive(spec.sceneAny),
            "scene_keys" to keysToJson(spec.sceneKeys),
            "pose_any" to JsonPrimitive(spec.poseAny),
            "pose_keys" to keysToJson(spec.poseKeys),
            "family_any" to JsonPrimitive(spec.familyAny),
            "families" to JsonArray(spec.families.map { JsonPrimitive(it.wire) }.sortedBy { it.content }),
            "vaginal_ratio" to JsonPrimitive(spec.vaginalRatio),
            "stage_weights" to JsonObject(
                SEX_STAGES.associate { it.wire to JsonPrimitive(spec.stageWeights[it] ?: 0.0) },
            ),
            "chest" to JsonPrimitive(spec.chest),
            "belly" to JsonPrimitive(spec.belly),
            "figure" to tagsToJson(spec.figure),
            "pussy_shape" to tagsToJson(spec.pussyShape),
            "pussy_hair" to tagsToJson(spec.pussyHair),
            "face" to JsonObject(
                FACE_GROUPS.associate { group -> group.id to faceToJson(faceChoice(spec, group)) },
            ),
            "count" to JsonPrimitive(spec.count),
        ),
    )

    private fun keysToJson(keys: Set<List<String>>): JsonArray =
        JsonArray(
            keys.sortedWith(compareBy({ it.size }, { it.joinToString("\u0000") }))
                .map { key -> JsonArray(key.map { JsonPrimitive(it) }) },
        )

    /** One row's tags, for a single-pick group; an empty array is its "off". */
    private fun tagsToJson(tags: List<String>): JsonArray = JsonArray(tags.map { JsonPrimitive(it) })

    private fun faceToJson(value: FacePick): JsonElement = when (value) {
        is FacePick.Tags -> JsonArray(value.tags.map { JsonPrimitive(it) })
        FacePick.Any -> JsonPrimitive(FACE_ANY)
        FacePick.None -> JsonPrimitive(FACE_NONE)
    }

    fun specFromJson(data: JsonObject): PromptSpec {
        val character = data["character"]?.jsonPrimitive?.contentOrNull
        if (character.isNullOrBlank()) throw ProfileException("character must be a non-empty string")
        val mode = data["mode"]?.jsonPrimitive?.contentOrNull?.let { PromptMode.ofWire(it) }
            ?: throw ProfileException("mode must be one of ${PromptMode.ORDER.joinToString(", ") { it.wire }}")
        val rawExposure = data["exposure"] as? JsonArray
            ?: throw ProfileException("exposure must be a non-empty list")
        if (rawExposure.isEmpty()) throw ProfileException("exposure must be a non-empty list")
        val exposure = rawExposure.map {
            it.jsonPrimitive.contentOrNull
                ?: throw ProfileException("exposure must be a list of strings")
        }
        val unknownExposure = exposure.filter { it !in EXPOSURE_LEVELS }
        if (unknownExposure.isNotEmpty()) {
            throw ProfileException("unknown exposure level: ${unknownExposure.joinToString(", ")}")
        }
        val ratio = if (data.containsKey("vaginal_ratio")) {
            asNumber(data["vaginal_ratio"])
                ?: throw ProfileException("vaginal_ratio must be a number in 0..1")
        } else {
            PromptLimits.VAGINAL_RATIO_DEFAULT
        }
        if (ratio < 0.0 || ratio > 1.0) throw ProfileException("vaginal_ratio must be a number in 0..1")
        val chest = data["chest"]?.jsonPrimitive?.contentOrNull ?: "auto"
        if (chest !in CHEST_LEVELS) throw ProfileException("chest must be one of ${CHEST_LEVELS.joinToString(", ")}")
        val belly = data["belly"]?.jsonPrimitive?.contentOrNull ?: "auto"
        if (belly !in BELLY_LEVELS) throw ProfileException("belly must be one of ${BELLY_LEVELS.joinToString(", ")}")
        val count = if (data.containsKey("count")) {
            val number = asNumber(data["count"]) ?: throw ProfileException("count must be an integer")
            if (number != floor(number)) throw ProfileException("count must be an integer")
            number.toInt()
        } else {
            PromptLimits.COUNT_DEFAULT
        }
        if (count < PromptLimits.COUNT_MIN || count > PromptLimits.COUNT_MAX) {
            throw ProfileException("count must be in ${PromptLimits.COUNT_MIN}..${PromptLimits.COUNT_MAX}")
        }
        return PromptSpec(
            character = character,
            mode = mode,
            exposure = exposure,
            clothingAny = boolAt(data, "clothing_any", true),
            clothingKeys = keySetAt(data, "clothing_keys"),
            sceneAny = boolAt(data, "scene_any", true),
            sceneKeys = keySetAt(data, "scene_keys"),
            poseAny = boolAt(data, "pose_any", true),
            poseKeys = keySetAt(data, "pose_keys"),
            familyAny = boolAt(data, "family_any", true),
            families = stringSetAt(data, "families", PoseFamily.entries.map { it.wire })
                .mapNotNull { wire -> PoseFamily.entries.firstOrNull { it.wire == wire } }
                .toSet(),
            vaginalRatio = ratio,
            stageWeights = stageWeightsAt(data),
            chest = chest,
            belly = belly,
            figure = pickAt(data, "figure"),
            pussyShape = pickAt(data, "pussy_shape"),
            pussyHair = pickAt(data, "pussy_hair"),
            face = faceAt(data),
            count = count,
        )
    }

    /**
     * A single-pick group's stored row tags. A profile written before the group existed has no key,
     * and `[]` is the wizard's own "off", so both read as no pick. The tags are not checked against
     * the matrix: like `clothing_keys`, the codec does not depend on it.
     */
    private fun pickAt(data: JsonObject, key: String): List<String> {
        val raw = data[key] ?: return emptyList()
        if (raw is JsonNull) return emptyList()
        val array = raw as? JsonArray ?: throw ProfileException("$key must be a list of tag strings")
        return array.map { element ->
            val text = (element as? JsonPrimitive)?.contentOrNull
            if (text.isNullOrEmpty()) throw ProfileException("$key must be a list of tag strings")
            text
        }
    }

    private fun numberOrNull(element: JsonElement?): Double? = asNumber(element)

    private fun boolAt(data: JsonObject, key: String, default: Boolean): Boolean {
        val element = data[key] ?: return default
        val primitive = element as? JsonPrimitive ?: throw ProfileException("$key must be true or false")
        if (primitive.isString) throw ProfileException("$key must be true or false")
        return primitive.booleanOrNull ?: throw ProfileException("$key must be true or false")
    }

    private fun keySetAt(data: JsonObject, key: String): Set<List<String>> {
        val raw = data[key] ?: return emptySet()
        val array = raw as? JsonArray ?: throw ProfileException("$key must be a list")
        val keys = linkedSetOf<List<String>>()
        array.forEach { item ->
            when (item) {
                is JsonPrimitive -> {
                    val text = item.contentOrNull
                    if (text.isNullOrEmpty()) {
                        throw ProfileException("$key entries must be tag strings or lists of tag strings")
                    }
                    keys.add(listOf(text))
                }
                is JsonArray -> {
                    val tags = item.map { it.jsonPrimitive.contentOrNull }
                    if (tags.isEmpty() || tags.any { it.isNullOrEmpty() }) {
                        throw ProfileException("$key entries must be tag strings or lists of tag strings")
                    }
                    keys.add(tags.map { it!! })
                }
                else -> throw ProfileException("$key entries must be tag strings or lists of tag strings")
            }
        }
        return keys
    }

    private fun stringSetAt(data: JsonObject, key: String, allowed: List<String>?): Set<String> {
        val raw = data[key] ?: return emptySet()
        val array = raw as? JsonArray ?: throw ProfileException("$key must be a list of strings")
        val values = array.map {
            it.jsonPrimitive.contentOrNull?.takeIf { text -> text.isNotEmpty() }
                ?: throw ProfileException("$key must be a list of strings")
        }.toSet()
        if (allowed != null) {
            val unknown = values.filter { it !in allowed }
            if (unknown.isNotEmpty()) throw ProfileException("unknown $key: ${unknown.sorted().joinToString(", ")}")
        }
        return values
    }

    private fun stageWeightsAt(data: JsonObject): Map<SexStage, Double> {
        val raw = data["stage_weights"] ?: return defaultStageWeights()
        val obj = raw as? JsonObject ?: throw ProfileException("stage_weights must be a JSON object")
        val weights = SEX_STAGES.associateWith { 0.0 }.toMutableMap()
        obj.forEach { (key, value) ->
            val stage = SEX_STAGES.firstOrNull { it.wire == key } ?: throw ProfileException("unknown stage: $key")
            val number = numberOrNull(value)
                ?: throw ProfileException("stage_weights.$key must be a number in 0..1")
            if (number < 0.0 || number > 1.0) {
                throw ProfileException("stage_weights.$key must be a number in 0..1")
            }
            weights[stage] = number
        }
        return weights
    }

    private fun faceAt(data: JsonObject): Map<String, FacePick> {
        val raw = data["face"]
        if (raw == null || raw is JsonNull) return defaultFace()
        val obj = raw as? JsonObject ?: throw ProfileException("face must be a JSON object")
        val face = LinkedHashMap<String, FacePick>()
        obj.forEach { (key, value) ->
            val group = FACE_GROUPS_BY_ID[key] ?: throw ProfileException("unknown face group: '$key'")
            if (value is JsonPrimitive) {
                val text = value.contentOrNull ?: throw ProfileException("face.$key must be a string or a list")
                face[key] = facePickOfString(text)
                return@forEach
            }
            val array = value as? JsonArray
                ?: throw ProfileException("face.$key must be '$FACE_ANY', '$FACE_NONE' or a non-empty list")
            if (array.isEmpty()) {
                throw ProfileException("face.$key must be '$FACE_ANY', '$FACE_NONE' or a non-empty list")
            }
            val tags = array.map { element ->
                val tag = element.jsonPrimitive.contentOrNull
                if (tag == null || tag !in groupTags(group)) {
                    throw ProfileException("'$tag' is not an option of face group '$key'")
                }
                tag
            }
            face[key] = FacePick.Tags(tags.distinct())
        }
        return face
    }

    // --- version upgrades ---

    /** Which new groups each v1 face axis covered, so a v1 profile's locked axis can be migrated. */
    private data class V1FaceAxis(
        val anyKey: String,
        val noneKey: String,
        val keysKey: String,
        val primary: String,
        val scope: List<String>,
    )

    private val V1_FACE_AXES: List<V1FaceAxis> = listOf(
        V1FaceAxis(
            "expression_any",
            "expression_none",
            "expression_keys",
            "expression",
            listOf("expression", "mouth", "blush", "tears"),
        ),
        V1FaceAxis("eye_any", "eye_none", "eye_keys", "eyes", listOf("gaze", "eyes")),
    )

    /**
     * Map a v1 spec's expression/eye fields onto the grouped, multi-select face model: each locked
     * tag goes to whichever group now holds it, and an axis left `any`/`none` keeps that meaning
     * for every group it covered.
     */
    private fun upgradeV1(specData: JsonObject, notes: MutableList<String>): JsonObject {
        val upgraded = specData.toMutableMap()
        val face = linkedMapOf<String, JsonElement>()
        val buckets = linkedMapOf<String, MutableList<String>>()
        V1_FACE_AXES.forEach { axis ->
            val anyValue = (upgraded.remove(axis.anyKey) as? JsonPrimitive)?.booleanOrNull ?: true
            val noneValue = (upgraded.remove(axis.noneKey) as? JsonPrimitive)?.booleanOrNull ?: false
            axis.scope.forEach { face[it] = JsonPrimitive(FACE_NONE) }
            if (anyValue && !noneValue) face[axis.primary] = JsonPrimitive(FACE_ANY)
            val rawKeys = upgraded.remove(axis.keysKey) as? JsonArray
            rawKeys?.forEach { element ->
                val tag = (element as? JsonPrimitive)?.contentOrNull
                val groupId = if (tag == null) null else FACE_TAG_GROUP[tag]
                if (groupId == null) {
                    notes.add("warning: profile v1 face tag '$tag' is unknown; dropped")
                    return@forEach
                }
                buckets.getOrPut(groupId) { mutableListOf() }.add(tag!!)
            }
        }
        buckets.forEach { (groupId, tags) -> face[groupId] = JsonArray(tags.distinct().map { JsonPrimitive(it) }) }
        upgraded["face"] = JsonObject(face)
        return JsonObject(upgraded)
    }

    /** v2 stored one tag per group; v3 stores a candidate list, so wrap bare tags. */
    private fun upgradeV2(specData: JsonObject): JsonObject {
        val upgraded = specData.toMutableMap()
        val raw = upgraded["face"] as? JsonObject ?: return JsonObject(upgraded)
        upgraded["face"] = JsonObject(
            raw.mapValues { (_, value) ->
                val text = (value as? JsonPrimitive)?.contentOrNull
                if (text != null && text != FACE_ANY && text != FACE_NONE) {
                    JsonArray(listOf(JsonPrimitive(text)))
                } else {
                    value
                }
            },
        )
        return JsonObject(upgraded)
    }
}

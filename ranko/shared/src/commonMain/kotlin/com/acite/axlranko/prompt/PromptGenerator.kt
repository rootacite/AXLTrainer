package com.acite.axlranko.prompt

import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * Prompt assembly, ported from `tools/gen_prompts.py`.
 *
 * The draws use Kotlin's [Random], so a seed reproduces within Ranko; the script's own sequence is
 * not reproduced (that was a deliberate choice, see the plan).
 */
object PromptGenerator {

    fun posePool(matrix: PromptMatrix, mode: PromptMode): List<MatrixEntry> =
        if (mode == PromptMode.Sex) matrix.poses else matrix.sfwPoses

    fun poseFamilies(entry: MatrixEntry): Set<PoseFamily> {
        val blob = entry.blob
        val tags = entry.tags.toSet()
        val found = mutableSetOf<PoseFamily>()
        if ("cowgirl" in blob || "girl on top" in blob || "upright straddle" in blob ||
            "sitting on lap" in blob
        ) {
            found.add(PoseFamily.GirlOnTop)
        }
        if ("suspended congress" in blob || "held up" in tags || "full nelson" in blob ||
            "sitting on lap" in blob || "upright straddle" in blob
        ) {
            found.add(PoseFamily.Hold)
        }
        if ("standing doggystyle" in blob) {
            found.add(PoseFamily.StandBehind)
        } else if ("doggystyle" in blob || "prone bone" in blob) {
            found.add(PoseFamily.Behind)
        }
        if ("reverse suspended" in blob || "full nelson" in blob) {
            found.add(PoseFamily.StandBehind)
        }
        if ("spooning" in blob || "seventh posture" in blob) {
            found.add(PoseFamily.SideBehind)
        } else if ("on side" in tags && "lying" in tags) {
            found.add(PoseFamily.SideBehind)
        }
        if ("reverse cowgirl" in blob) {
            found.add(PoseFamily.Behind)
        }
        if ("mating press" in blob || "anvil position" in blob || "standing missionary" in blob ||
            ("missionary" in blob && "doggystyle" !in blob)
        ) {
            found.add(PoseFamily.Face)
        }
        if ("suspended congress" in blob && "reverse" !in blob) {
            found.add(PoseFamily.Face)
        }
        return found
    }

    /** How visible the chest is: `front`, `side`, `back`, or `optional` (kneeling/prone). */
    fun chestView(entry: MatrixEntry): String {
        val blob = entry.blob
        val tags = entry.tags.toSet()
        if ("prone bone" in blob || "on stomach" in tags) return "optional"
        if ("doggystyle" in blob && "standing doggystyle" !in blob) return "optional"
        if ("standing doggystyle" in blob || "reverse cowgirl" in blob || "reverse suspended" in blob) {
            return "back"
        }
        if ("spooning" in blob || "seventh posture" in blob || "on side" in tags) return "side"
        if ("full nelson" in blob) return "side"
        val families = poseFamilies(entry)
        if (PoseFamily.Face in families || PoseFamily.GirlOnTop in families) return "front"
        if (PoseFamily.Hold in families && PoseFamily.StandBehind !in families) return "front"
        if ("looking at viewer" in blob && "from behind" !in blob) return "front"
        if ("from behind" in blob || "looking back" in blob) return "back"
        return "front"
    }

    /** Chest/belly tags. Front + open clothes in nsfw/sex always pulls breasts out. */
    fun torsoTags(
        spec: PromptSpec,
        pose: MatrixEntry,
        clothing: MatrixEntry?,
        addOpen: Boolean,
        rng: Random,
    ): List<String> {
        val nude = clothing == null
        val view = chestView(pose)
        var chestPref = if (spec.chest in CHEST_LEVELS) spec.chest else "auto"
        var bellyPref = if (spec.belly in BELLY_LEVELS) spec.belly else "auto"
        if (spec.mode == PromptMode.Sfw) {
            if (chestPref == "auto") chestPref = "covered"
            if (bellyPref == "auto") bellyPref = "covered"
        }

        val tags = mutableListOf<String>()
        if (nude) {
            tags.addAll(listOf("breasts", "nipples"))
        } else {
            val forceFrontOut = spec.mode != PromptMode.Sfw && addOpen && view == "front"
            val optionalView = view == "optional" || view == "side" || view == "back"
            var showOut = false
            var showNipples = false
            var showCleavage = false
            if (forceFrontOut) {
                showOut = true
                showNipples = true
            } else if (chestPref == "cleavage") {
                showCleavage = !addOpen
            } else if (chestPref == "breasts_out") {
                showOut = addOpen
            } else if (chestPref == "nipples") {
                showOut = addOpen
                showNipples = addOpen
            } else if (chestPref == "auto" && spec.mode != PromptMode.Sfw) {
                if (addOpen && optionalView) {
                    showOut = rng.nextDouble() < 0.5
                    showNipples = showOut
                } else if (!addOpen && view == "front" && spec.mode == PromptMode.Nsfw) {
                    showCleavage = true
                }
            }
            if (showOut) {
                tags.addAll(listOf("breasts", "breasts out"))
                if (showNipples) tags.add("nipples")
            } else if (showNipples) {
                tags.addAll(listOf("breasts", "nipples"))
            } else if (showCleavage) {
                tags.add("cleavage")
            }
        }

        if (bellyPref == "midriff") {
            tags.add("midriff")
        } else if (bellyPref == "navel") {
            tags.addAll(listOf("midriff", "navel"))
        } else if (bellyPref == "auto" && spec.mode != PromptMode.Sfw) {
            if ((nude || addOpen) && view == "front") tags.add("navel")
        }
        return tags.distinct()
    }

    fun familyPool(poses: List<MatrixEntry>, spec: PromptSpec): List<MatrixEntry> {
        if (spec.familyAny || spec.families.isEmpty()) return poses
        val chosen = poses.filter { poseFamilies(it).any { family -> family in spec.families } }
        return chosen.ifEmpty { poses }
    }

    fun sceneFlags(tags: List<String>): Set<String> {
        val present = tags.toSet()
        val flags = mutableSetOf<String>()
        if ("on bed" in present) flags.add("on_bed")
        if ("bed" in present || "on bed" in present) flags.add("bed")
        if ("desk" in present) flags.add("desk")
        if ("window" in present) flags.add("window")
        if ("indoors" in present) flags.add("indoor")
        if ("outdoors" in present) flags.add("outdoor")
        if ("grass" in present || "sand" in present) flags.add("ground")
        if ("sofa" in present || "bench" in present) flags.add("seat")
        if ("ocean" in present || "pool" in present || "onsen" in present) flags.add("water")
        if ("train interior" in present) flags.add("cramped")
        if (present.any {
                it in setOf("hallway", "classroom", "bedroom", "cafe", "living room", "street")
            }
        ) {
            flags.add("wall")
        }
        return flags
    }

    fun poseLocus(entry: MatrixEntry): String {
        val tags = entry.tags.toSet()
        val blob = entry.blob
        if ("sitting on stairs" in blob) return "sit_stairs"
        if ("sitting" in tags && "desk" in tags) return "sit_desk"
        if ("looking outside" in tags && "window" in tags) return "window"
        if ("walking" in tags) return "walk"
        if ("holding bouquet" in tags || "twirl" in tags) return "twirl"
        if ("held up" in tags || "suspended congress" in blob || "reverse suspended congress" in blob ||
            "full nelson" in blob
        ) {
            return "lift"
        }
        if ("standing doggystyle" in blob || "standing missionary" in blob || "standing sex" in blob) {
            return "sex_stand"
        }
        if ("doggystyle" in blob) return "sex_kneel"
        if ("cowgirl" in blob || "girl on top" in blob || "sitting on lap" in blob ||
            "upright straddle" in blob
        ) {
            return "cowgirl"
        }
        if ("against wall" in tags) return "wall"
        if (containsMarker(blob, LIE_MARKERS)) return "lie"
        if ("standing" in tags) return "stand"
        if ("sitting" in tags) return "sit"
        return "other"
    }

    fun sceneCompatible(pose: MatrixEntry, scene: MatrixEntry): Boolean {
        val locus = poseLocus(pose)
        val tags = scene.tags.toSet()
        val flags = sceneFlags(scene.tags)
        fun has(vararg names: String): Boolean = names.any { it in tags }

        when (locus) {
            "lie" -> {
                if (has("hallway", "rooftop", "train interior", "street", "cafe")) return false
                return flags.any { it in setOf("on_bed", "bed", "ground", "seat") }
            }
            "sit_desk" -> {
                if ("on bed" in tags || has(
                        "rooftop", "onsen", "hallway", "street", "grass", "sand", "forest", "pool",
                    )
                ) {
                    return false
                }
                return "desk" in tags || ("window" in tags && "indoors" in tags)
            }
            "sit_stairs" -> {
                if (has(
                        "on bed", "beach", "train interior", "cafe", "poolside", "forest", "classroom",
                    )
                ) {
                    return false
                }
                return has("hallway", "shrine", "street", "school courtyard")
            }
            "window" -> return "window" in tags && "outdoors" !in tags
            "wall" -> {
                if (has("grass", "beach", "forest", "poolside", "rooftop", "onsen")) return false
                return "indoors" in tags || "street" in tags
            }
            "lift" -> {
                if ("on bed" in tags || has("train interior", "cafe", "desk")) return false
                return true
            }
            "walk" -> {
                if ("on bed" in tags || has("train interior", "cafe", "onsen", "desk")) return false
                return has("hallway", "street", "park", "school courtyard", "beach", "shrine", "rooftop")
            }
            "twirl" -> {
                if ("on bed" in tags || has("train interior", "desk")) return false
                return has("shrine", "school courtyard", "park", "rooftop", "street", "beach")
            }
            "stand" -> return "on bed" !in tags
            "cowgirl" -> {
                if (has("hallway", "street", "rooftop")) return false
                return flags.any { it in setOf("bed", "on_bed", "seat", "ground") }
            }
            "sex_kneel" -> {
                if (has("rooftop", "street", "train interior", "cafe")) return false
                return flags.any { it in setOf("bed", "on_bed", "ground") } ||
                    has("bedroom", "living room", "classroom")
            }
            "sex_stand" -> return "on bed" !in tags && "train interior" !in tags
        }
        return true
    }

    fun poseAllowsHole(pose: MatrixEntry): Boolean =
        pose.channel == PromptChannel.Both || (pose.channel != null && pose.channel in HOLE_CHANNELS)

    fun anatomyTags(pose: MatrixEntry, channel: PromptChannel, includePenis: Boolean = true): List<String> {
        val blob = pose.blob
        val out = mutableListOf<String>()
        if (includePenis) out.add("penis")
        if (channel == PromptChannel.Vaginal) out.add("pussy")
        else if (channel == PromptChannel.Anal) out.add("anus")
        if (containsMarker(blob, ASS_MARKERS) && "ass" !in out) out.add("ass")
        if (containsMarker(blob, SPREAD_MARKERS)) {
            if ("pussy" !in out) out.add("pussy")
            if (channel == PromptChannel.Anal && "ass" !in out) out.add("ass")
        }
        if (containsMarker(blob, TOP_MARKERS)) {
            if (channel == PromptChannel.Vaginal && "pussy" !in out) out.add("pussy")
            if (channel == PromptChannel.Anal && "ass" !in out) out.add("ass")
        }
        return out
    }

    fun stageTags(stage: SexStage, channel: PromptChannel): List<String> = when (stage) {
        SexStage.ObjectInsertion -> when (channel) {
            PromptChannel.Anal -> listOf("anal object insertion")
            PromptChannel.Vaginal -> listOf("vaginal object insertion")
            else -> emptyList()
        }
        SexStage.Fingering -> when (channel) {
            PromptChannel.Anal -> listOf("anal fingering")
            PromptChannel.Vaginal -> listOf("fingering")
            else -> emptyList()
        }
        else -> if (channel !in HOLE_CHANNELS) {
            when (stage) {
                SexStage.Pose -> listOf("presenting")
                SexStage.Ejaculation -> listOf("ejaculation")
                SexStage.Before, SexStage.During -> emptyList()
                else -> listOf("after sex", "cumdrip")
            }
        } else {
            when (stage) {
                SexStage.Pose -> listOf("presenting")
                SexStage.Before ->
                    if (channel == PromptChannel.Anal) listOf("imminent anal", "penis on ass")
                    else listOf("imminent vaginal", "penis on pussy")
                SexStage.During -> emptyList()
                SexStage.Ejaculation ->
                    if (channel == PromptChannel.Anal) {
                        listOf("ejaculation", "cum in ass", "cum overflow")
                    } else {
                        listOf("ejaculation", "cum in pussy", "cum overflow")
                    }
                else -> {
                    val extra = mutableListOf("after sex", "cumdrip", "gaping")
                    if (channel == PromptChannel.Anal) {
                        extra.addAll(listOf("after anal", "cum in ass"))
                    } else {
                        extra.addAll(listOf("after vaginal", "cum in pussy"))
                    }
                    extra
                }
            }
        }
    }

    fun pickStage(spec: PromptSpec, rng: Random): SexStage {
        val weights = SEX_STAGES.map { stageWeightValue(spec, it) }
        val total = weights.sum()
        if (total <= 0) return SexStage.During
        val pick = rng.nextInt(total)
        var acc = 0
        SEX_STAGES.forEachIndexed { index, stage ->
            acc += weights[index]
            if (pick < acc) return stage
        }
        return SexStage.During
    }

    /** Object-insertion / fingering only ride poses that can take a hole. */
    fun bindStageToPoses(poses: List<MatrixEntry>, stage: SexStage?): Pair<SexStage?, List<MatrixEntry>> {
        if (stage !in STAGES_OBJECT) return stage to poses
        val holePoses = poses.filter { poseAllowsHole(it) }
        if (holePoses.isNotEmpty()) return stage to holePoses
        return SexStage.During to poses
    }

    fun pickChannel(pose: MatrixEntry, spec: PromptSpec, rng: Random): PromptChannel =
        when (pose.channel) {
            PromptChannel.Anal -> PromptChannel.Anal
            PromptChannel.Vaginal -> PromptChannel.Vaginal
            PromptChannel.None -> PromptChannel.None
            else -> if (rng.nextDouble() < spec.vaginalRatio) PromptChannel.Vaginal else PromptChannel.Anal
        }

    fun weightFor(entry: MatrixEntry, anySelected: Boolean, selected: Set<List<String>>): Int =
        if (anySelected || selected.isEmpty()) 1
        else if (selected.contains(entry.key)) PromptLimits.PREFER_WEIGHT else 1

    fun preferredPool(
        items: List<MatrixEntry>,
        anySelected: Boolean,
        selected: Set<List<String>>,
    ): List<MatrixEntry> {
        if (anySelected || selected.isEmpty()) return items
        val chosen = items.filter { selected.contains(it.key) }
        return chosen.ifEmpty { items }
    }

    fun poseWeight(entry: MatrixEntry, spec: PromptSpec): Int {
        var weight = weightFor(entry, spec.poseAny, spec.poseKeys)
        if (spec.mode != PromptMode.Sex) return weight
        val matched = poseFamilies(entry)
        if (!spec.familyAny && spec.families.isNotEmpty()) {
            val hits = matched.count { it in spec.families }
            if (hits == 0) return 0
            weight *= max(1, hits)
        }
        val vaginal = min(1000, max(0, pythonRound(spec.vaginalRatio * 1000).toInt()))
        val anal = 1000 - vaginal
        weight *= when (entry.channel) {
            PromptChannel.Vaginal -> vaginal
            PromptChannel.Anal -> anal
            else -> 1000
        }
        return weight
    }

    fun weightedChoice(rng: Random, items: List<MatrixEntry>, weights: List<Int>): MatrixEntry {
        val total = weights.sum()
        if (total <= 0) return items[rng.nextInt(items.size)]
        val pick = rng.nextInt(total)
        var acc = 0
        items.forEachIndexed { index, item ->
            acc += weights[index]
            if (pick < acc) return item
        }
        return items.last()
    }

    fun clothingBuckets(spec: PromptSpec): List<String> {
        val buckets = EXPOSURE_LEVELS.filter { spec.exposure.contains(it) }
        return buckets.ifEmpty { defaultExposure(spec.mode) }
    }

    fun clothingPoolForBucket(matrix: PromptMatrix, bucket: String): List<MatrixEntry> = when (bucket) {
        "nude" -> emptyList()
        "open" -> matrix.clothing.filter { it.openClothes }
        else -> matrix.clothing.filter { it.group == bucket }
    }

    fun pickClothing(matrix: PromptMatrix, spec: PromptSpec, rng: Random): Pair<MatrixEntry?, Boolean> {
        val buckets = clothingBuckets(spec).ifEmpty { defaultExposure(spec.mode) }
        val bucket = buckets[rng.nextInt(buckets.size)]
        if (bucket == "nude") return null to false
        var pool = preferredPool(clothingPoolForBucket(matrix, bucket), spec.clothingAny, spec.clothingKeys)
        if (pool.isEmpty()) pool = matrix.clothing
        val weights = pool.map { weightFor(it, spec.clothingAny, spec.clothingKeys) }
        val chosen = weightedChoice(rng, pool, weights)
        return chosen to (bucket == "open")
    }

    fun pickScene(
        matrix: PromptMatrix,
        spec: PromptSpec,
        pose: MatrixEntry,
        rng: Random,
        warnings: MutableList<String>,
    ): MatrixEntry {
        val pool = preferredPool(matrix.scenes, spec.sceneAny, spec.sceneKeys)
        var compatible = pool.filter { sceneCompatible(pose, it) }
        if (compatible.isEmpty()) {
            warnings.add("no compatible scene for pose '${pose.blob}'; using full list")
            compatible = pool.ifEmpty { matrix.scenes }
        }
        val weights = compatible.map { weightFor(it, spec.sceneAny, spec.sceneKeys) }
        return weightedChoice(rng, compatible, weights)
    }

    /** At most one tag per group, none of them contradicting the pose or an earlier group. */
    fun faceTags(spec: PromptSpec, pose: MatrixEntry, rng: Random): List<String> {
        val poseTags = pose.tags.toSet()
        val out = mutableListOf<String>()
        for (group in FACE_GROUPS) {
            if (poseTags.intersect(groupTags(group).toSet()).isNotEmpty()) continue
            val choice = faceChoice(spec, group)
            if (choice == FacePick.None) continue
            val taken = poseTags + out
            val pool = if (choice == FacePick.Any) groupPool(group, spec.mode) else faceTagsOf(choice)
            val candidates = pool.filter { it !in taken && !faceConflict(it, taken) }
            if (candidates.isEmpty()) continue
            out.add(candidates[rng.nextInt(candidates.size)])
        }
        return out
    }

    /**
     * True when the pose itself holds the legs up, leaving no free arm for a hand action: `mating
     * press` and `anvil position` fold them against the body, `full nelson` pins them. See
     * [LEGS_HELD_MARKERS].
     */
    fun poseHoldsLegs(pose: MatrixEntry): Boolean = containsMarker(pose.blob, LEGS_HELD_MARKERS)

    /**
     * The channel word as the prompt writes it. The anal one carries its weight — a bare `anal`
     * beside a pose the model reads as vaginal is what gets dropped — while the vaginal one is a
     * plain tag.
     */
    fun channelTag(channel: PromptChannel): String =
        if (channel == PromptChannel.Anal) PromptLimits.ANAL_CHANNEL_TAG else channel.wire

    /**
     * The stages that put a hand on the body with no penis in play. On a pose that holds its own
     * legs one of these has no arm to use, so [assemble] gives those draws a partner.
     *
     * `ObjectInsertion` is not here: the same geometry argues for it, but the maintainer asked for
     * the fingering words specifically. Add it to this set if the artifact shows up there too.
     */
    private val HAND_ONLY_STAGES: Set<SexStage> = setOf(SexStage.Fingering)

    fun assemble(
        spec: PromptSpec,
        pose: MatrixEntry,
        clothing: MatrixEntry?,
        addOpen: Boolean,
        scene: MatrixEntry,
        suffix: MatrixEntry,
        channel: PromptChannel?,
        torso: List<String> = emptyList(),
        face: List<String> = emptyList(),
        stage: SexStage? = null,
    ): String {
        val prefix = splitTags(spec.character)
        val extra = mutableListOf<String>()
        val resolvedStage = if (spec.mode == PromptMode.Sex) {
            if (stage != null && SEX_STAGES.contains(stage)) stage else SexStage.During
        } else {
            null
        }
        if (resolvedStage != null) {
            // A hand action on a pose that already holds both legs has no arm left to do it, and the
            // model answers with a third hand: those draws get a partner — his hand, her legs —
            // instead of `solo`, which is the tag that claimed she was alone to begin with.
            val withPartner = resolvedStage in STAGES_WITH_PARTNER ||
                (resolvedStage in HAND_ONLY_STAGES && poseHoldsLegs(pose))
            if (withPartner) extra.addAll(listOf("1boy", "hetero")) else extra.add("solo")
        } else {
            extra.add("solo")
        }
        if (clothing == null) {
            extra.add("nude")
        } else {
            extra.addAll(clothing.tags)
            if (addOpen) extra.add("open clothes")
        }
        extra.addAll(torso)
        extra.addAll(face)
        extra.addAll(pose.tags)
        if (resolvedStage != null) {
            val hole = channel ?: PromptChannel.Vaginal
            if (resolvedStage in STAGES_PENETRATING && hole in HOLE_CHANNELS) {
                extra.add("sex")
                extra.add(channelTag(hole))
            }
            extra.addAll(anatomyTags(pose, hole, includePenis = resolvedStage in STAGES_WITH_PARTNER))
            extra.addAll(stageTags(resolvedStage, hole))
        }
        extra.addAll(scene.tags)
        extra.addAll(suffix.tags)

        val seen = prefix.mapTo(mutableSetOf()) { it.lowercase() }
        val out = prefix.toMutableList()
        for (tag in extra) {
            if (isForbiddenTag(tag)) continue
            val key = tag.lowercase()
            if (!seen.add(key)) continue
            out.add(tag)
        }
        return out.joinToString(", ")
    }

    private data class Combo(
        val pose: List<String>,
        val clothing: List<String>,
        val addOpen: Boolean,
        val torso: List<String>,
        val face: List<String>,
        val scene: List<String>,
        val suffix: List<String>,
        val channel: PromptChannel?,
        val stage: SexStage?,
    )

    /**
     * [count] prompts from [matrix]. A draw that repeats an earlier combination is retried, and the
     * last few are filled without the uniqueness check so a small pool still yields `count` lines.
     */
    fun generate(
        spec: PromptSpec,
        matrix: PromptMatrix,
        seed: Long? = null,
        warnings: MutableList<String> = mutableListOf(),
    ): List<String> {
        if (spec.count < PromptLimits.COUNT_MIN || spec.count > PromptLimits.COUNT_MAX) {
            throw IllegalArgumentException(
                "count must be ${PromptLimits.COUNT_MIN}..${PromptLimits.COUNT_MAX}",
            )
        }
        val rng = if (seed == null) Random.Default else Random(seed)
        var poses = posePool(matrix, spec.mode)
        if (spec.mode == PromptMode.Sex) poses = familyPool(poses, spec)
        poses = preferredPool(poses, spec.poseAny, spec.poseKeys)
        if (poses.isEmpty()) throw IllegalStateException("empty pose pool")

        val results = mutableListOf<String>()
        val used = mutableSetOf<Combo>()
        var attempts = 0
        val limit = max(spec.count * 40, 80)
        while (results.size < spec.count && attempts < limit) {
            attempts += 1
            var stage = if (spec.mode == PromptMode.Sex) pickStage(spec, rng) else null
            val bound = bindStageToPoses(poses, stage)
            stage = bound.first
            val stagePoses = bound.second
            val pose = weightedChoice(rng, stagePoses, stagePoses.map { poseWeight(it, spec) })
            val (clothing, addOpen) = pickClothing(matrix, spec, rng)
            val scene = pickScene(matrix, spec, pose, rng, warnings)
            val suffix = weightedChoice(rng, matrix.suffixes, List(matrix.suffixes.size) { 1 })
            val channel = if (spec.mode == PromptMode.Sex) pickChannel(pose, spec, rng) else null
            val torso = torsoTags(spec, pose, clothing, addOpen, rng)
            val face = faceTags(spec, pose, rng)
            val combo = Combo(
                pose = pose.key,
                clothing = clothing?.key ?: listOf(PromptLimits.NUDE_SENTINEL),
                addOpen = addOpen,
                torso = torso,
                face = face,
                scene = scene.key,
                suffix = suffix.key,
                channel = channel,
                stage = stage,
            )
            if (!used.add(combo)) continue
            results.add(assemble(spec, pose, clothing, addOpen, scene, suffix, channel, torso, face, stage))
        }
        while (results.size < spec.count) {
            var stage = if (spec.mode == PromptMode.Sex) pickStage(spec, rng) else null
            val bound = bindStageToPoses(poses, stage)
            stage = bound.first
            val stagePoses = bound.second
            val pose = weightedChoice(rng, stagePoses, stagePoses.map { poseWeight(it, spec) })
            val (clothing, addOpen) = pickClothing(matrix, spec, rng)
            val scene = pickScene(matrix, spec, pose, rng, warnings)
            val suffix = matrix.suffixes[rng.nextInt(matrix.suffixes.size)]
            val channel = if (spec.mode == PromptMode.Sex) pickChannel(pose, spec, rng) else null
            val torso = torsoTags(spec, pose, clothing, addOpen, rng)
            val face = faceTags(spec, pose, rng)
            results.add(assemble(spec, pose, clothing, addOpen, scene, suffix, channel, torso, face, stage))
        }
        return results
    }
}

package com.acite.axlranko.prompt

/** One matrix row: its tags, the comment line under it, and, for poses, the channel it accepts. */
data class MatrixEntry(
    val tags: List<String>,
    val comment: String = "",
    val channel: PromptChannel? = null,
    val openClothes: Boolean = false,
    val group: String? = null,
    /**
     * A `QUESTIONABLE_POSES` row. Those rows state their own clothing and body exposure, so the
     * clothing and torso layers leave them alone — see `PromptGenerator.stateFill`.
     */
    val selfStated: Boolean = false,
) {
    val blob: String get() = tags.joinToString(", ")
    val key: List<String> get() = tags
}

class PromptMatrix(
    val poses: List<MatrixEntry>,
    val clothing: List<MatrixEntry>,
    val scenes: List<MatrixEntry>,
    val suffixes: List<MatrixEntry>,
    val sfwPoses: List<MatrixEntry>,
    val questionablePoses: List<MatrixEntry>,
)

fun splitTags(text: String): List<String> =
    text.split(",").map { it.trim() }.filter { it.isNotEmpty() }

/** Parse repo-root `input_matrix.txt` into its five sections. Ported from `parse_matrix`. */
fun parseMatrix(text: String): PromptMatrix {
    val buckets = linkedMapOf(
        "POSES" to mutableListOf<MatrixEntry>(),
        "CLOTHING" to mutableListOf<MatrixEntry>(),
        "SCENE" to mutableListOf<MatrixEntry>(),
        "SUFFIX" to mutableListOf<MatrixEntry>(),
        "SFW_POSES" to mutableListOf<MatrixEntry>(),
        "QUESTIONABLE_POSES" to mutableListOf<MatrixEntry>(),
    )
    var section: String? = null
    var group: String? = null
    var pending: MatrixEntry? = null

    fun commit() {
        val entry = pending ?: return
        val name = section ?: throw MatrixException("entry before a section header")
        buckets.getValue(name).add(entry)
        pending = null
    }

    text.lines().forEachIndexed { index, raw ->
        val line = raw.trim()
        val lineno = index + 1
        if (line.isEmpty()) return@forEachIndexed
        if (line.endsWith(":") && SECTION_NAMES.contains(line.dropLast(1))) {
            commit()
            section = line.dropLast(1)
            group = null
            return@forEachIndexed
        }
        // A header this parser does not know would otherwise land in the section above it as a tag
        // row, taking every row under it along — that is how `QUESTIONABLE_POSES` first leaked into
        // the SFW pool. Spell it out instead.
        if (SECTION_HEADER_RE.matches(line)) {
            throw MatrixException("line $lineno: unknown section ${line.dropLast(1)}")
        }
        val groupMatch = CLOTHING_GROUP_RE.matchEntire(line)
        if (section == "CLOTHING" && groupMatch != null) {
            commit()
            group = groupMatch.groupValues[1]
            return@forEachIndexed
        }
        if (line.startsWith("#")) {
            val current = pending ?: return@forEachIndexed
            pending = current.copy(comment = line.drop(1).trim())
            return@forEachIndexed
        }
        commit()
        if (section == null) throw MatrixException("line $lineno: tags before a section header")
        if (section == "POSES") {
            val match = CHANNEL_RE.matchEntire(line)
                ?: throw MatrixException(
                    "line $lineno: pose must end with ': both|anal only|vaginal only|none'",
                )
            val channel = PromptChannel.ofMatrix(match.groupValues[2])
                ?: throw MatrixException("unknown channel: ${match.groupValues[2]}")
            pending = MatrixEntry(tags = splitTags(match.groupValues[1]), channel = channel)
            return@forEachIndexed
        }
        var body = line
        var openClothes = false
        if (body.endsWith(PromptLimits.OPEN_MARK)) {
            openClothes = true
            body = body.dropLast(PromptLimits.OPEN_MARK.length).trimEnd()
        }
        if (section == "CLOTHING") {
            val currentGroup = group
                ?: throw MatrixException("line $lineno: clothing row before a [group] header")
            pending = MatrixEntry(
                tags = splitTags(body),
                openClothes = openClothes,
                group = currentGroup,
            )
            return@forEachIndexed
        }
        pending = MatrixEntry(
            tags = splitTags(body),
            openClothes = openClothes,
            selfStated = section == "QUESTIONABLE_POSES",
        )
    }
    commit()

    if (buckets.values.any { it.isEmpty() }) {
        throw MatrixException("matrix is missing a required section")
    }
    val clothing = buckets.getValue("CLOTHING")
    if (clothing.any { it.group == null }) {
        throw MatrixException("clothing entry without an exposure group")
    }
    return PromptMatrix(
        poses = buckets.getValue("POSES").toList(),
        clothing = clothing.toList(),
        scenes = buckets.getValue("SCENE").toList(),
        suffixes = buckets.getValue("SUFFIX").toList(),
        sfwPoses = buckets.getValue("SFW_POSES").toList(),
        questionablePoses = buckets.getValue("QUESTIONABLE_POSES").toList(),
    )
}

/** A section header shape (`POSES:`, `QUESTIONABLE_POSES:`), as opposed to a tag row or `[covered]`. */
private val SECTION_HEADER_RE = Regex("^[A-Z][A-Z_]*:$")

/** `re.match` semantics: the pattern only has to match from the first character. */
private fun Regex.matchesFromStart(text: String): Boolean = find(text)?.range?.first == 0

fun isForbiddenTag(tag: String): Boolean {
    val lowered = tag.lowercase().trim()
    if (RATING_TAGS.contains(lowered) || QUALITY_TAGS.contains(lowered)) return true
    return YEAR_RE.matchesFromStart(lowered) || SCORE_RE.matchesFromStart(lowered)
}

fun containsMarker(blob: String, markers: List<String>): Boolean = markers.any { blob.contains(it) }

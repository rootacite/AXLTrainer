package com.acite.axlranko.data

/**
 * What [TomlDocumentPatcher.parse] found in a document: the uncommented `key = value` pairs of
 * every table, and the blocks of every `[[array of tables]]`, with the value text taken verbatim.
 */
data class ParsedToml(
    val sections: Map<String, Map<String, String>>,
    val arrayBlocks: Map<String, List<Map<String, String>>>,
)

/**
 * Updates uncommented `key = value` pairs inside named TOML tables
 * without rewriting the rest of the document (comments, blank lines,
 * unknown tables such as [bookkeeping] stay intact).
 */
object TomlDocumentPatcher {

    /**
     * Write a standalone document from the same maps [apply] / [replaceArrayOfTables] consume: each
     * table in map order, with every array block that belongs to it right after it
     * (`[[environment.train_data]]` after `[environment]`). A profile is written this way, and the
     * result has to decode as a whole config again.
     */
    fun render(
        sections: Map<String, Map<String, String>>,
        arrayBlocks: Map<String, List<Map<String, String>>> = emptyMap(),
        header: String? = null,
    ): String {
        val lines = mutableListOf<String>()
        if (header != null) lines.add("# $header")
        for ((section, values) in sections) {
            if (lines.isNotEmpty()) lines.add("")
            lines.add("[$section]")
            for ((key, value) in values) lines.add("$key = $value")
            for ((arraySection, blocks) in arrayBlocks) {
                if (arraySection.substringBeforeLast('.') != section) continue
                for (block in blocks) {
                    lines.add("")
                    lines.add("[[$arraySection]]")
                    for ((key, value) in block) lines.add("$key = $value")
                }
            }
        }
        return lines.joinToString("\n") + "\n"
    }

    /**
     * [render]'s reverse: table headers, array headers, and the raw value text of their
     * uncommented keys. Comments, blank lines and anything before the first header are skipped, and
     * a value is never re-encoded - an apply hands these strings straight to [apply].
     */
    fun parse(text: String): ParsedToml {
        val sections = linkedMapOf<String, MutableMap<String, String>>()
        val arrayBlocks = linkedMapOf<String, MutableList<MutableMap<String, String>>>()
        var openTable: MutableMap<String, String>? = null

        for (rawLine in text.split("\r\n", "\n")) {
            val line = rawLine.trim()
            if (line.isEmpty() || line.startsWith("#")) continue
            if (line.startsWith("[[") && line.endsWith("]]")) {
                val block = linkedMapOf<String, String>()
                arrayBlocks.getOrPut(line.substring(2, line.length - 2).trim()) { mutableListOf() }
                    .add(block)
                openTable = block
                continue
            }
            if (line.startsWith("[") && line.endsWith("]")) {
                openTable = sections.getOrPut(line.substring(1, line.length - 1).trim()) { linkedMapOf() }
                continue
            }
            val equals = line.indexOf('=')
            if (equals <= 0) continue
            val value = line.substring(equals + 1).trim()
            if (value.isEmpty()) continue
            openTable?.put(line.substring(0, equals).trim(), value)
        }
        return ParsedToml(sections, arrayBlocks)
    }

    /** Whether the document carries a `[section]` header (a dotted name needs its full header). */
    fun hasTable(text: String, section: String): Boolean {
        val header = "[$section]"
        return text.split("\r\n", "\n").any { it.trim() == header }
    }

    fun apply(original: String, sectionValues: Map<String, Map<String, String>>): String {
        val newline = if (original.contains("\r\n")) "\r\n" else "\n"
        val lines = original.split("\r\n", "\n").toMutableList()
        val hadTrailingNewline = original.endsWith("\n") || original.endsWith("\r\n")

        for ((section, values) in sectionValues) {
            val header = "[$section]"
            val start = lines.indexOfFirst { it.trim() == header }
            if (start < 0) continue

            var end = lines.size
            for (i in (start + 1) until lines.size) {
                val trimmed = lines[i].trim()
                if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
                    end = i
                    break
                }
            }

            for ((key, encodedValue) in values) {
                var found = false
                for (i in (start + 1) until end) {
                    val line = lines[i]
                    val leading = line.takeWhile { it == ' ' || it == '\t' }
                    val trimmed = line.trimStart()
                    if (trimmed.startsWith("#") || trimmed.isEmpty()) continue
                    val eq = trimmed.indexOf('=')
                    if (eq <= 0) continue
                    val lineKey = trimmed.substring(0, eq).trim()
                    if (lineKey == key) {
                        lines[i] = "$leading$key = $encodedValue"
                        found = true
                        break
                    }
                }
                if (!found) {
                    var insertAt = end
                    while (insertAt > start + 1 && lines[insertAt - 1].isBlank()) {
                        insertAt--
                    }
                    lines.add(insertAt, "$key = $encodedValue")
                    end++
                }
            }
        }

        val joined = lines.joinToString(newline)
        return if (hadTrailingNewline && !joined.endsWith(newline)) joined + newline else joined
    }

    /**
     * Replace the `[[section]]` array of tables with [blocks], one block per entry.
     *
     * Each block's values are written in the map's iteration order. An existing array is
     * replaced as a whole (comments inside it are lost, as in [apply]); without one the
     * blocks are appended to the parent table. [blocks] empty leaves the document alone.
     */
    fun replaceArrayOfTables(
        original: String,
        section: String,
        blocks: List<Map<String, String>>,
    ): String {
        if (blocks.isEmpty()) return original

        val newline = if (original.contains("\r\n")) "\r\n" else "\n"
        val lines = original.split("\r\n", "\n").toMutableList()
        val hadTrailingNewline = original.endsWith("\n") || original.endsWith("\r\n")
        val header = "[[$section]]"
        val parentHeader = "[${section.substringBeforeLast('.')}]"

        val emitted = mutableListOf<String>()
        for ((index, values) in blocks.withIndex()) {
            if (index > 0) emitted.add("")
            emitted.add(header)
            for ((key, value) in values) emitted.add("$key = $value")
        }

        val start = lines.indexOfFirst { it.trim() == header }
        val regionStart: Int
        val regionEnd: Int
        if (start >= 0) {
            // The whole array, i.e. every repeated header and the keys under it.
            var end = start + 1
            while (end < lines.size) {
                val trimmed = lines[end].trim()
                if (trimmed == header || !trimmed.startsWith("[")) end++ else break
            }
            regionStart = start
            regionEnd = end
        } else {
            val parent = lines.indexOfFirst { it.trim() == parentHeader }
            require(parent >= 0) { "missing table $parentHeader" }
            var end = parent + 1
            while (end < lines.size && !lines[end].trim().startsWith("[")) end++
            var from = end
            while (from > parent + 1 && lines[from - 1].isBlank()) from--
            regionStart = from
            regionEnd = from
        }

        // The blank lines around the array are re-emitted, so drop them with it.
        var from = regionStart
        var to = regionEnd
        while (to > from && lines[to - 1].isBlank()) to--
        while (to < lines.size && lines[to].isBlank()) to++
        lines.subList(from, to).clear()

        val payload = mutableListOf<String>()
        if (from > 0 && lines[from - 1].isNotBlank()) payload.add("")
        payload.addAll(emitted)
        if (lines.getOrNull(from) != null) payload.add("")
        lines.addAll(from, payload)

        val joined = lines.joinToString(newline)
        return if (hadTrailingNewline && !joined.endsWith(newline)) joined + newline else joined
    }

    fun quote(value: String): String {
        val escaped = buildString(value.length + 2) {
            for (ch in value) {
                when (ch) {
                    '\\' -> append("\\\\")
                    '"' -> append("\\\"")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> append(ch)
                }
            }
        }
        return "\"$escaped\""
    }

    /**
     * Encode a numeric field as a TOML float.
     *
     * ktoml refuses to decode an integer literal (`5`) into a Kotlin `Double`.
     * Whole-valued floats such as `5.0` / `1` must therefore be written with a
     * fractional or exponent part (`5.0`), not as a bare integer.
     */
    fun float(raw: String): String = encodeFloat(raw.trim().toDouble())

    fun encodeFloat(value: Double): String {
        require(value.isFinite()) { "TOML float must be finite" }
        if (value == value.toLong().toDouble() &&
            value in Long.MIN_VALUE.toDouble()..Long.MAX_VALUE.toDouble()
        ) {
            return "${value.toLong()}.0"
        }
        val text = value.toString()
        if ('.' !in text && text.none { it == 'e' || it == 'E' }) {
            return "$text.0"
        }
        return text
    }
}

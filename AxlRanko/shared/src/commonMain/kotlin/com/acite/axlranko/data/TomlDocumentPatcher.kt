package com.acite.axlranko.data

/**
 * Updates uncommented `key = value` pairs inside named TOML tables
 * without rewriting the rest of the document (comments, blank lines,
 * unknown tables such as [bookkeeping] stay intact).
 */
object TomlDocumentPatcher {

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
}

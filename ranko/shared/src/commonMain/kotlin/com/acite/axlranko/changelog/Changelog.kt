package com.acite.axlranko.changelog

data class ChangelogEntry(
    val hash: String,
    val date: String,
    val subject: String,
    val gitTags: List<String> = emptyList(),
) {
    val kind: String?
    val title: String

    init {
        val parsed = parseSubject(subject)
        kind = parsed.kind
        title = parsed.title
    }
}

internal data class ParsedSubject(
    val kind: String?,
    val title: String,
)

/** `(tag) [Kind] title` — leading parenthetical tags are dropped from the title, then an optional kind. */
internal fun parseSubject(subject: String): ParsedSubject {
    var rest = subject.trim()
    while (rest.startsWith("(")) {
        val close = rest.indexOf(')')
        if (close <= 1) break
        rest = rest.substring(close + 1).trimStart()
    }
    val kindMatch = KIND_PREFIX.matchEntire(rest)
    return if (kindMatch != null) {
        val leftover = kindMatch.groupValues[2].trim()
        ParsedSubject(kindMatch.groupValues[1], leftover.ifBlank { rest })
    } else {
        ParsedSubject(null, rest.ifBlank { subject.trim() })
    }
}

private val KIND_PREFIX = Regex("^\\[([^\\]]+)\\]\\s*(.*)$")

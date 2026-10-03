package com.acite.axlranko.data

import kotlinx.serialization.descriptors.SerialDescriptor

/**
 * ktoml reads `0` as a Long and refuses to decode an integer literal into a Kotlin `Double`, so a
 * `key = 0` someone typed by hand makes the whole config.toml unreadable and Chromatrix fails at
 * startup. The trainer's Python side has no such rule: `float(0)` is 0.0 there.
 *
 * [normalize] rewrites those literals as floats before decoding. The keys come from
 * [AxlTrainerConfig]'s own descriptors, so a new `Double` field is covered without a list to keep
 * in sync. The rule ignores a key's table, which holds because no name is a `Double` in one table
 * and an integer in another.
 */
object TomlIntegerLiterals {

    /** `indent key = <integer> [comment]`; a value with a dot or an exponent does not match. */
    private val INTEGER_LITERAL = Regex("""^(\s*)([A-Za-z0-9_-]+)(\s*=\s*)([+-]?\d+)(\s*(?:#.*)?)""")

    /** TOML keys the config model declares as `Double` (or `Double?`). */
    fun doubleKeys(): Set<String> {
        val keys = mutableSetOf<String>()
        collectDoubleKeys(AxlTrainerConfig.serializer().descriptor, keys)
        return keys
    }

    /** `key = 0` → `key = 0.0` for those keys; every other line is returned unchanged. */
    fun normalize(text: String): String {
        val keys = doubleKeys()
        return text.split("\n").joinToString("\n") { line ->
            val match = INTEGER_LITERAL.matchEntire(line) ?: return@joinToString line
            val key = match.groupValues[2]
            if (key !in keys) return@joinToString line
            "${match.groupValues[1]}$key${match.groupValues[3]}${match.groupValues[4]}.0${match.groupValues[5]}"
        }
    }

    // A primitive descriptor has no elements, so recursing into one stops by itself; a list's only
    // element is the entry type, which is what reaches `[[validation.samples]]`'s own keys.
    private fun collectDoubleKeys(descriptor: SerialDescriptor, into: MutableSet<String>) {
        for (index in 0 until descriptor.elementsCount) {
            val element = descriptor.getElementDescriptor(index)
            if (isDouble(element)) into += descriptor.getElementName(index) else collectDoubleKeys(element, into)
        }
    }

    private fun isDouble(descriptor: SerialDescriptor): Boolean =
        descriptor.serialName.removeSuffix("?") == "kotlin.Double"
}

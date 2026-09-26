package com.acite.axlranko.data

/**
 * One saved `config.toml` preset under `<repo root>/configs/`.
 */
data class ConfigProfile(
    val name: String,
    val modified: Long = 0,
    val size: Long = 0,
)

/**
 * The merge an apply produced: the patched document, and the profile sections `config.toml` has no
 * header for - `TomlDocumentPatcher.apply` writes into tables that exist, so those keys are skipped
 * rather than created, and the panel says which ones instead of dropping them silently.
 */
data class ProfileApply(
    val text: String,
    val skippedSections: List<String>,
    val appliedKeys: Int,
)

/**
 * Name rules and in-memory merge for Utils -> Profiles. Disk IO lives in the helper (`profile_*`).
 */
object ConfigProfileStore {
    const val DIRECTORY_NAME = "configs"
    private const val MAX_NAME_LENGTH = 64
    private val ILLEGAL_NAME_CHARS = charArrayOf('/', '\\', ':', '*', '?', '"', '<', '>', '|')
    private const val PROFILE_HEADER =
        "Ranko config profile (Utils -> Profiles). Applying it patches config.toml in place."

    /** Null when the name is usable, otherwise the message to show next to the field. */
    fun validateName(name: String): String? {
        val trimmed = name.trim()
        return when {
            trimmed.isEmpty() -> "Enter a profile name"
            trimmed.length > MAX_NAME_LENGTH -> "Use at most $MAX_NAME_LENGTH characters"
            trimmed.startsWith(".") -> "A profile name cannot start with a dot"
            trimmed.any { it in ILLEGAL_NAME_CHARS || it.isISOControl() } ->
                "A profile name cannot contain / \\ : * ? \" < > |"
            else -> null
        }
    }

    /** The document a profile file holds: the header comment plus the rendered sections. */
    fun document(
        sections: Map<String, Map<String, String>>,
        arrayBlocks: Map<String, List<Map<String, String>>>,
    ): String = TomlDocumentPatcher.render(sections, arrayBlocks, PROFILE_HEADER)

    /**
     * Patch [configText] with the keys [profileText] holds, and nothing else: comments, blank lines,
     * unknown tables and every key the profile leaves out stay as they are.
     */
    fun applyToConfig(configText: String, profileText: String): Result<ProfileApply> {
        val parsed = TomlDocumentPatcher.parse(profileText)
        if (parsed.sections.isEmpty() && parsed.arrayBlocks.isEmpty()) {
            return Result.failure(IllegalArgumentException("The profile holds no settings"))
        }
        val merged = try {
            var text = configText
            for ((section, blocks) in parsed.arrayBlocks) {
                text = TomlDocumentPatcher.replaceArrayOfTables(text, section, blocks)
            }
            TomlDocumentPatcher.apply(text, parsed.sections)
        } catch (e: Exception) {
            return Result.failure(e)
        }
        if (ConfigImporter.parseConfig(merged) == null) {
            return Result.failure(
                IllegalStateException("Applying this profile would leave config.toml unreadable")
            )
        }
        val skipped = parsed.sections.keys
            .filterNot { TomlDocumentPatcher.hasTable(configText, it) }
            .sorted()
        val keys = parsed.sections.values.sumOf { it.size } +
            parsed.arrayBlocks.values.sumOf { blocks -> blocks.sumOf { it.size } }
        return Result.success(ProfileApply(merged, skipped, keys))
    }
}

package com.acite.axlranko.data

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * One saved `config.toml` preset: a file under `<repo root>/configs/`.
 *
 * `name` is the file name without its extension, `path` is absolute so a row can act on it without
 * resolving the repository again.
 */
data class ConfigProfile(
    val name: String,
    val path: String,
    val modified: Long,
    val size: Long,
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
 * The `configs/` directory: named `config.toml` presets, written by Ranko Utils -> Profiles.
 *
 * The directory sits next to `config.toml` in the repository root and is git-ignored apart from its
 * `.gitkeep` placeholder; the trainer never reads it. Every function takes the directory, so tests
 * work on a temp folder and never write into the repository's own `configs/`.
 */
object ConfigProfileStore {
    const val DIRECTORY_NAME = "configs"
    private const val EXTENSION = "toml"
    private const val MAX_NAME_LENGTH = 64
    private val ILLEGAL_NAME_CHARS = charArrayOf('/', '\\', ':', '*', '?', '"', '<', '>', '|')
    private const val PROFILE_HEADER =
        "Ranko config profile (Utils -> Profiles). Applying it patches config.toml in place."

    fun dir(root: File): File = File(root, DIRECTORY_NAME)

    fun resolveDir(): File? = TrainerRepo.findRoot()?.let { dir(it) }

    /** The saved profiles, by name. `*.toml` only, so the `.gitkeep` placeholder never shows up. */
    fun list(dir: File): List<ConfigProfile> =
        dir.listFiles()
            ?.filter { it.isFile && it.extension.equals(EXTENSION, ignoreCase = true) }
            ?.map {
                ConfigProfile(
                    name = it.nameWithoutExtension,
                    path = it.absolutePath,
                    modified = it.lastModified(),
                    size = it.length(),
                )
            }
            ?.sortedBy { it.name.lowercase() }
            ?: emptyList()

    fun findByName(dir: File, name: String): ConfigProfile? =
        list(dir).firstOrNull { it.name.equals(name.trim(), ignoreCase = true) }

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
     * Write `text` as `dir/<name>.toml`, creating the directory if needed. A name that is already
     * taken fails unless [overwrite] is set, and the file itself is replaced through a temp name in
     * the same directory so a killed write cannot leave a half-written preset behind.
     */
    fun save(dir: File, name: String, text: String, overwrite: Boolean): Result<File> {
        val trimmed = name.trim()
        validateName(trimmed)?.let { return Result.failure(IllegalArgumentException(it)) }
        val existing = findByName(dir, trimmed)
        if (existing != null && !overwrite) {
            return Result.failure(IllegalStateException("A profile named \"$trimmed\" already exists"))
        }
        return try {
            dir.mkdirs()
            // An existing name is overwritten in place: on a case-insensitive name check, "A" has to
            // land on the "a.toml" it matched, not sit next to it as a second file.
            val target = existing?.let { File(it.path) } ?: File(dir, "$trimmed.$EXTENSION")
            val temporary = File(dir, "${target.name}.tmp")
            temporary.writeText(text, Charsets.UTF_8)
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
            Result.success(target)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Delete one profile. A path outside [dir] is refused: a stale row must not delete anything else. */
    fun delete(dir: File, path: String): Result<Unit> {
        return try {
            val file = File(path)
            if (!file.exists()) {
                Result.failure(IllegalStateException("The profile is already gone: $path"))
            } else if (file.absoluteFile.parentFile?.canonicalFile != dir.canonicalFile) {
                Result.failure(IllegalArgumentException("Not a profile file: $path"))
            } else if (file.delete()) {
                Result.success(Unit)
            } else {
                Result.failure(IllegalStateException("Could not delete $path"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Patch [configText] with the keys [profileText] holds, and nothing else: comments, blank lines,
     * unknown tables and every key the profile leaves out stay as they are. A profile produced by
     * [document] carries the whole config; a hand-trimmed one carries part of it.
     *
     * The merged document is decoded before it is handed back, so an apply that would leave
     * `config.toml` unreadable fails here and the file is never touched.
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

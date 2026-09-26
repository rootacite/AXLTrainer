package com.acite.axlranko.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ConfigDocument(
    val path: String,
    val text: String,
)

@Serializable
data class ConfigSaveResult(
    val path: String,
)

@Serializable
data class ProfileInfo(
    val name: String,
    val modified: Long = 0,
    val size: Long = 0,
)

@Serializable
data class ProfileListResult(
    val profiles: List<ProfileInfo> = emptyList(),
)

@Serializable
data class ProfileDocument(
    val name: String,
    val text: String,
)

@Serializable
data class ProfileSaveResult(
    val name: String,
)

@Serializable
data class TagLexiconResult(
    val text: String = "",
)

@Serializable
data class DatasetRecord(
    val stem: String,
    val image: String,
    val txt: String,
    val mask: String = "",
    val width: Int = 0,
    val height: Int = 0,
    val tags: List<String> = emptyList(),
    @SerialName("has_sidecar_mask") val hasSidecarMask: Boolean = false,
    @SerialName("has_alpha") val hasAlpha: Boolean = false,
)

@Serializable
data class DatasetListResult(
    val items: List<DatasetRecord> = emptyList(),
    val orphans: List<String> = emptyList(),
)

@Serializable
data class DatasetDropResult(
    val moved: Int = 0,
)

@Serializable
data class DatasetShuffleResult(
    val groups: Int = 0,
    @SerialName("renamed_files") val renamedFiles: Int = 0,
    @SerialName("first_stem") val firstStem: String = "",
    @SerialName("last_stem") val lastStem: String = "",
)

@Serializable
data class MaskGetResult(
    @SerialName("png_base64") val pngBase64: String = "",
)

@Serializable
data class BlobItem(
    val path: String,
    val hash: String? = null,
    val width: Int? = null,
    val height: Int? = null,
    val bytes: Int? = null,
    val mime: String? = null,
    val cache: String? = null,
    val base64: String? = null,
    val error: String? = null,
)

@Serializable
data class BlobListResult(
    val items: List<BlobItem> = emptyList(),
)

@Serializable
data class CheckpointExportResult(
    val bytes: Long = 0,
)

@Serializable
data class FsEntry(
    val name: String,
    val path: String,
    @SerialName("is_dir") val isDir: Boolean = false,
    val size: Long = 0,
    @SerialName("mtime_ms") val mtimeMs: Long = 0,
)

@Serializable
data class FsListResult(
    val path: String = "",
    val parent: String? = null,
    val entries: List<FsEntry> = emptyList(),
)

@Serializable
data class FsRoot(
    val name: String,
    val path: String,
)

@Serializable
data class FsRootsResult(
    val roots: List<FsRoot> = emptyList(),
)

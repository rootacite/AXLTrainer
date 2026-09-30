package com.acite.axlranko.prompt

/** One row of the one-page configuration list (总清单): its label, its current value, and what it edits. */
data class ManifestRow(
    val pageKey: String,
    val label: String,
    val value: String,
)

/**
 * The configuration list that reviews a whole profile, ported from `manifest_items`.
 *
 * Rows follow the wizard's order; the sex-only rows and the clothing row are dropped exactly like
 * the CLI drops them.
 */
object ManifestModel {

    fun faceValue(spec: PromptSpec, lang: PromptLang): String {
        val parts = FACE_GROUPS.mapNotNull { group ->
            val value = faceChoice(spec, group)
            if (value == FacePick.None) return@mapNotNull null
            val name = if (lang == PromptLang.Chinese) group.zh else group.en
            "$name=${faceValueLabel(value, lang)}"
        }
        return parts.joinToString(", ").ifEmpty { t(lang, "face_skip") }
    }

    fun stageValue(spec: PromptSpec, lang: PromptLang): String {
        val parts = SEX_STAGES.filter { stageWeightValue(spec, it) > 0 }
            .map { "${it.wire} ${trimNumber(spec.stageWeights[it] ?: 0.0)}" }
        return parts.joinToString(", ").ifEmpty { t(lang, "manifest_stage_fallback") }
    }

    /** A single-pick group's value: the chosen row's tags, or the "off" label. */
    fun pickValue(tags: List<String>, lang: PromptLang): String =
        tags.joinToString(", ").ifEmpty { t(lang, "pick_off") }

    fun items(spec: PromptSpec, lang: PromptLang): List<ManifestRow> {
        fun label(key: String) = t(lang, "item_$key")
        fun picked(anyFlag: Boolean, count: Int) =
            if (anyFlag) t(lang, "any") else t(lang, "manifest_selected").replace("{n}", count.toString())

        val rows = mutableListOf(
            ManifestRow("character", label("character"), spec.character),
            ManifestRow("mode", label("mode"), spec.mode.wire),
            ManifestRow("exposure", label("exposure"), spec.exposure.joinToString(", ")),
        )
        if (!spec.nudeOnly) {
            rows.add(ManifestRow("clothing", label("clothing"), picked(spec.clothingAny, spec.clothingKeys.size)))
        }
        rows.add(ManifestRow("chest", label("chest"), spec.chest))
        rows.add(ManifestRow("belly", label("belly"), spec.belly))
        rows.add(ManifestRow("figure", label("figure"), pickValue(spec.figure, lang)))
        rows.add(ManifestRow("face", label("face"), faceValue(spec, lang)))
        rows.add(ManifestRow("scene", label("scene"), picked(spec.sceneAny, spec.sceneKeys.size)))
        if (spec.mode == PromptMode.Sex) {
            rows.add(ManifestRow("family", label("family"), picked(spec.familyAny, spec.families.size)))
            rows.add(ManifestRow("ratio", label("ratio"), trimNumber(spec.vaginalRatio)))
            rows.add(ManifestRow("stages", label("stages"), stageValue(spec, lang)))
            rows.add(ManifestRow("pussy_shape", label("pussy_shape"), pickValue(spec.pussyShape, lang)))
            rows.add(ManifestRow("pussy_hair", label("pussy_hair"), pickValue(spec.pussyHair, lang)))
        }
        rows.add(ManifestRow("pose", label("pose"), picked(spec.poseAny, spec.poseKeys.size)))
        rows.add(ManifestRow("count", label("count"), spec.count.toString()))
        return rows
    }
}

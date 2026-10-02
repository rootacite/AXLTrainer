package com.acite.axlranko.model

/**
 * The pure rules behind the Dashboard's Sampling Prompts section, its editor and a checkpoint
 * card's "clear samples" action. Nothing here touches Compose or IPC, so all of it is unit-tested
 * without a window (the panel and the card only render what these return).
 */

/** `1280×720 · 55 steps · CFG 6.0 · seed 0 · ×3`: one set of a run's prompts on one line. */
internal fun sampleSetSummaryLine(set: SampleSetInfo): String {
    val parts = mutableListOf("${set.width}×${set.height}", "${set.steps} steps")
    parts += "CFG ${sampleValueLabel(set.guidanceScale)}"
    if (set.guidanceRescale > 0f) parts += "rescale ${sampleValueLabel(set.guidanceRescale)}"
    parts += "seed ${set.seed}"
    parts += "×${set.repeat}"
    return parts.joinToString(" · ")
}

/** `3 sets · 9 images per pass`: what one sample point of this run writes. */
internal fun samplePromptsSummaryLine(response: SamplePromptsResponse): String {
    val sets = response.sets.size
    val images = response.sets.sumOf { it.repeat }
    val setWord = if (sets == 1) "set" else "sets"
    val imageWord = if (images == 1) "image" else "images"
    return "$sets $setWord · $images $imageWord per pass"
}

/**
 * Where the prompts came from. An edited run names the file they were saved in, which is what
 * tells it apart from the config snapshot that is never rewritten.
 */
internal fun samplePromptsSourceLabel(response: SamplePromptsResponse): String {
    response.file?.takeIf { response.edited }?.let { file ->
        return "Edited for this run · ${file.substringAfterLast('/')}"
    }
    val source = response.configSource.substringAfterLast('/')
    return if (source.isBlank()) "No config resolved" else "From $source"
}

/**
 * The line under the sets while the run is live: an edit made now lands on the next checkpoint's
 * sample point, because the trainer reads the run's own file before every pass.
 */
internal fun samplePromptsLiveLabel(response: SamplePromptsResponse): String? =
    if (response.live && response.sets.isNotEmpty()) {
        "This run is live — the next sample point uses these prompts."
    } else {
        null
    }

/** The card's confirmation title; the body is [clearSamplesConfirmText]. */
internal const val CLEAR_SAMPLES_CONFIRM_TITLE = "Clear this checkpoint's sample images?"

/**
 * What the button removes: the run's own samples of that step, the generated and evaluated images
 * of the passes that belong to the card, and the job records that produced them.
 */
internal fun clearSamplesConfirmText(images: Int, jobs: Int, step: Int?): String {
    val where = step?.let { " at step $it" } ?: ""
    val imageWord = if (images == 1) "image" else "images"
    val records = when {
        jobs <= 0 -> ""
        jobs == 1 -> " and the pass record that produced it"
        else -> " and the $jobs pass records that produced them"
    }
    return "Removes $images $imageWord$where$records — the training samples and every pass this " +
        "card shows. The checkpoint itself is kept; this cannot be undone."
}

/** The line a card shows once a clear has landed. */
internal fun clearedSamplesLabel(result: SampleClearResult): String {
    result.error?.let { return "Could not clear samples: $it" }
    val imageWord = if (result.images == 1) "image" else "images"
    return "Cleared ${result.images} $imageWord"
}

/** The editor's own heading: `Sampling prompts · 3 sets`. */
internal fun samplePromptsEditorTitle(response: SamplePromptsResponse?): String {
    val sets = response?.sets?.size ?: 0
    val setWord = if (sets == 1) "set" else "sets"
    return "Sampling prompts · $sets $setWord"
}

/**
 * What the editor would send, or null when a field does not parse. The form's validation
 * (`sampleSetFormErrors`) rejects exactly the values that fail here, so null means a bug rather
 * than a user error.
 */
internal fun sampleSetInfos(forms: List<SampleSetForm>): List<SampleSetInfo>? {
    val sets = ArrayList<SampleSetInfo>(forms.size)
    for (form in forms) {
        val width = form.width.trim().toIntOrNull() ?: return null
        val height = form.height.trim().toIntOrNull() ?: return null
        val steps = form.steps.trim().toIntOrNull() ?: return null
        val repeat = form.repeat.trim().toIntOrNull() ?: return null
        val guidance = form.guidanceScale.trim().toFloatOrNull() ?: return null
        val rescale = form.guidanceRescale.trim().toFloatOrNull() ?: return null
        val seed = form.seed.trim().toLongOrNull() ?: return null
        sets += SampleSetInfo(
            name = form.name.trim(),
            prompt = form.prompt.trim(),
            negative = form.negative.trim(),
            width = width,
            height = height,
            steps = steps,
            guidanceScale = guidance,
            guidanceRescale = rescale,
            seed = seed,
            repeat = repeat,
        )
    }
    return sets
}

/** A resolved set as the editor holds it (every field a text box, as the Utils form does). */
internal fun SampleSetInfo.toForm(): SampleSetForm = SampleSetForm(
    name = name,
    prompt = prompt,
    negative = negative,
    width = width.toString(),
    height = height.toString(),
    steps = steps.toString(),
    guidanceScale = sampleValueLabel(guidanceScale),
    guidanceRescale = sampleValueLabel(guidanceRescale),
    seed = seed.toString(),
    repeat = repeat.toString(),
)

/** `6.0` rather than `6`, `0.6` kept as it is: what the field shows and what a request carries. */
internal fun sampleValueLabel(value: Float): String =
    if (value == value.toInt().toFloat()) value.toInt().toString() else value.toString()

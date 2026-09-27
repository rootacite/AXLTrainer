package com.acite.axlranko.prompt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** A whole-config profile, the way the Python suite's `_full_spec` builds one. */
private fun fullSpec(): PromptSpec = testSpec(
    mode = PromptMode.Sex,
    exposure = listOf("revealing", "open"),
    clothingAny = false,
    clothingKeys = setOf(listOf("bikini", "side-tie bikini bottom")),
    sceneAny = false,
    sceneKeys = setOf(listOf("bedroom", "indoors", "bed")),
    poseAny = false,
    poseKeys = setOf(listOf("cowgirl position", "girl on top", "sitting", "straddling", "looking at viewer")),
    familyAny = false,
    families = setOf(PoseFamily.GirlOnTop),
    vaginalRatio = 0.3,
    stageWeights = defaultStageWeights() + mapOf(
        SexStage.Pose to 0.1,
        SexStage.Before to 0.1,
        SexStage.During to 0.4,
        SexStage.Ejaculation to 0.2,
        SexStage.After to 0.1,
        SexStage.Done to 0.1,
    ),
    chest = "nipples",
    belly = "navel",
    face = testFace(
        "expression" to FacePick.Tags(listOf("orgasm")),
        "gaze" to FacePick.None,
        "eyes" to FacePick.None,
    ),
    count = 7,
)

private fun v1Spec(
    expressionAny: Boolean = true,
    expressionNone: Boolean = false,
    expressionKeys: List<String> = emptyList(),
    eyeAny: Boolean = true,
    eyeNone: Boolean = false,
    eyeKeys: List<String> = emptyList(),
): String {
    fun flags(value: Boolean) = if (value) "true" else "false"
    fun keys(values: List<String>) = values.joinToString(", ") { "\"$it\"" }
    return """
        {
          "version": 1,
          "name": "v1",
          "spec": {
            "character": "(sena_character:1.1), 1girl",
            "mode": "nsfw",
            "exposure": ["revealing"],
            "count": 4,
            "expression_any": ${flags(expressionAny)},
            "expression_none": ${flags(expressionNone)},
            "expression_keys": [${keys(expressionKeys)}],
            "eye_any": ${flags(eyeAny)},
            "eye_none": ${flags(eyeNone)},
            "eye_keys": [${keys(eyeKeys)}]
          }
        }
    """.trimIndent()
}

class ProfileCodecTest {
    private val matrix = miniMatrix()

    @Test
    fun aRoundTripKeepsEveryField() {
        val spec = fullSpec()
        val text = ProfileCodec.profileText("sena test", spec)
        val loaded = ProfileCodec.loadProfile(text, "sena test")
        assertEquals(spec, loaded.spec)
        assertEquals("sena test", loaded.name)
        assertTrue(text.contains(""""version": 3"""), text)
        assertTrue(loaded.notes.isEmpty(), loaded.notes.toString())
    }

    @Test
    fun aReloadedProfileGeneratesTheSamePrompts() {
        val spec = fullSpec()
        val loaded = ProfileCodec.loadProfile(ProfileCodec.profileText("sena test", spec), "sena test")
        assertEquals(
            generatePrompts(spec, matrix, 5),
            generatePrompts(loaded.spec, matrix, 5),
        )
    }

    @Test
    fun unsafeNamesAreSanitized() {
        assertEquals("sena_v2", ProfileCodec.safeProfileName("sena/v2"))
        assertEquals("", ProfileCodec.safeProfileName(".."))
        assertFailsWith<ProfileException> { ProfileCodec.requireProfileName("..") }
        assertEquals("Kirika", ProfileCodec.requireProfileName(" Kirika "))
    }

    @Test
    fun anInvalidSpecIsRejected() {
        val text = """{"version": 2, "name": "bad", "spec": {"character": "", "mode": "sfw"}}"""
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(text, "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile("not json", "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile("""{"name": "x"}""", "x") }
    }

    @Test
    fun missingStageWeightsDefaultToDuring() {
        val spec = fullSpec()
        val text = ProfileCodec.profileText("old", spec)
            .replace(Regex("\"stage_weights\": \\{[^}]*},?"), "")
        val loaded = ProfileCodec.loadProfile(text, "old")
        assertEquals(defaultStageWeights(), loaded.spec.stageWeights)
    }

    @Test
    fun anUnknownStageIsRejected() {
        val text = ProfileCodec.profileText("bad", fullSpec())
            .replace(Regex("\"stage_weights\": \\{[^}]*}"), """"stage_weights": {"pose": 1.0, "foreplay": 0.5}""")
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(text, "bad") }
    }

    @Test
    fun oldSixStageWeightsZeroTheNewStages() {
        val text = ProfileCodec.profileText("oldstages", fullSpec()).replace(
            Regex("\"stage_weights\": \\{[^}]*}"),
            """"stage_weights": {"pose": 0.0, "before": 0.0, "during": 1.0, "ejaculation": 0.0, "after": 0.0, "done": 0.0}""",
        )
        val weights = ProfileCodec.loadProfile(text, "oldstages").spec.stageWeights
        assertEquals(1.0, weights.getValue(SexStage.During))
        assertEquals(0.0, weights.getValue(SexStage.ObjectInsertion))
        assertEquals(0.0, weights.getValue(SexStage.Fingering))
    }

    @Test
    fun anUnknownFaceOptionIsRejected() {
        val text =
            """{"version": 2, "name": "bad face", "spec": {"character": "c", "mode": "sfw", "exposure": ["covered"], "face": {"expression": "not a tag"}}}"""
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(text, "bad face") }
    }

    @Test
    fun aFaceOptionFromAnotherGroupIsRejected() {
        listOf(""""mouth": "smile"""", """"mouth": ["smile"]"""").forEach { face ->
            val text =
                """{"version": 3, "name": "misgrouped", "spec": {"character": "c", "mode": "sfw", "exposure": ["covered"], "face": {$face}}}"""
            assertFailsWith<ProfileException> { ProfileCodec.loadProfile(text, "misgrouped") }
        }
    }

    @Test
    fun aCandidateListRoundTripsThroughAProfile() {
        val spec = fullSpec()
        spec.face = spec.face + ("expression" to FacePick.Tags(listOf("smile", "shy")))
        val text = ProfileCodec.profileText("multi", spec)
        assertTrue(text.contains(""""version": 3"""))
        val loaded = ProfileCodec.loadProfile(text, "multi")
        assertEquals(FacePick.Tags(listOf("smile", "shy")), loaded.spec.face["expression"])
    }

    @Test
    fun aV1AnyAxisBecomesTheNewDefaults() {
        val loaded = ProfileCodec.loadProfile(v1Spec(), "v1any")
        assertEquals(defaultFace(), loaded.spec.face)
        assertTrue(loaded.notes.any { it.contains("upgraded from v1") })
    }

    @Test
    fun aV1ProfileRoutesLockedTagsIntoTheirGroups() {
        val loaded = ProfileCodec.loadProfile(
            v1Spec(
                expressionAny = false,
                expressionKeys = listOf("blush"),
                eyeAny = false,
                eyeKeys = listOf("looking at viewer"),
            ),
            "v1route",
        )
        assertEquals(FacePick.Tags(listOf("blush")), loaded.spec.face["blush"])
        assertEquals(FacePick.Tags(listOf("looking at viewer")), loaded.spec.face["gaze"])
        assertEquals(FacePick.None, loaded.spec.face["expression"])
        assertEquals(FacePick.None, loaded.spec.face["mouth"])
        assertEquals(FacePick.None, loaded.spec.face["tears"])
        assertEquals(FacePick.None, loaded.spec.face["eyes"])
    }

    @Test
    fun aV1MultiSelectBecomesACandidateList() {
        val loaded = ProfileCodec.loadProfile(
            v1Spec(
                expressionAny = false,
                expressionKeys = listOf("smile", "shy"),
                eyeAny = false,
                eyeNone = true,
            ),
            "v1multi",
        )
        assertEquals(FacePick.Tags(listOf("smile", "shy")), loaded.spec.face["expression"])
    }

    @Test
    fun aV1AxisPinnedToNothingStaysOff() {
        val loaded = ProfileCodec.loadProfile(
            v1Spec(expressionAny = false, expressionNone = true, eyeAny = false, eyeNone = true),
            "v1none",
        )
        listOf("expression", "mouth", "blush", "tears", "gaze", "eyes").forEach { group ->
            assertEquals(FacePick.None, loaded.spec.face[group], group)
        }
    }

    @Test
    fun aV1UnknownFaceTagIsDroppedWithANote() {
        val loaded = ProfileCodec.loadProfile(
            v1Spec(expressionAny = false, expressionKeys = listOf("not a tag")),
            "v1quiet",
        )
        assertTrue(loaded.notes.any { it.contains("unknown") }, loaded.notes.toString())
        assertEquals(FacePick.None, loaded.spec.face["expression"])
    }

    @Test
    fun aV2TagBecomesACandidateList() {
        val text = ProfileCodec.profileText("v2tag", fullSpec())
            .replace(""""version": 3""", """"version": 2""")
            .replace(
                Regex("\"face\": \\{[^}]*}"),
                """"face": {"expression": "smile", "gaze": "none", "eyes": "any"}""",
            )
        val loaded = ProfileCodec.loadProfile(text, "v2tag")
        assertEquals(FacePick.Tags(listOf("smile")), loaded.spec.face["expression"])
        assertEquals(FacePick.None, loaded.spec.face["gaze"])
        assertEquals(FacePick.ANY, loaded.spec.face["eyes"])
        assertTrue(loaded.notes.any { it.contains("upgraded from v2") })
    }

    @Test
    fun anUnsupportedVersionIsRejected() {
        val text = ProfileCodec.profileText("p", fullSpec()).replace(""""version": 3""", """"version": 99""")
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(text, "p") }
        val stringVersion = ProfileCodec.profileText("p", fullSpec()).replace(""""version": 3""", """"version": "3"""")
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(stringVersion, "p") }
    }

    @Test
    fun badFieldValuesAreRejected() {
        fun specWith(body: String) = """{"version": 3, "name": "bad", "spec": {$body}}"""

        val base = """"character": "c", "mode": "sfw", "exposure": ["covered"]"""
        assertFailsWith<ProfileException> {
            ProfileCodec.loadProfile(specWith(""""character": "", "mode": "sfw", "exposure": ["covered"]"""), "bad")
        }
        assertFailsWith<ProfileException> {
            ProfileCodec.loadProfile(specWith(""""character": "c", "mode": "weird", "exposure": ["covered"]"""), "bad")
        }
        assertFailsWith<ProfileException> {
            ProfileCodec.loadProfile(specWith(""""character": "c", "mode": "sfw", "exposure": []"""), "bad")
        }
        assertFailsWith<ProfileException> {
            ProfileCodec.loadProfile(specWith(""""character": "c", "mode": "sfw", "exposure": ["naked"]"""), "bad")
        }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"chest\": \"big\""), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"belly\": \"big\""), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"count\": 0"), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"count\": 1.5"), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"vaginal_ratio\": 2"), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"vaginal_ratio\": true"), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"clothing_any\": \"yes\""), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"families\": [\"nope\"]"), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"face\": []"), "bad") }
        assertFailsWith<ProfileException> { ProfileCodec.loadProfile(specWith("$base, \"stage_weights\": []"), "bad") }
    }
}

class ManifestModelTest {
    private fun keys(spec: PromptSpec, lang: PromptLang = PromptLang.Chinese): List<String> =
        ManifestModel.items(spec, lang).map { it.pageKey }

    @Test
    fun sfwDropsTheSexOnlyRows() {
        val rows = keys(testSpec(mode = PromptMode.Sfw))
        assertFalse(rows.contains("family"))
        assertFalse(rows.contains("ratio"))
        assertFalse(rows.contains("stages"))
        assertTrue(rows.contains("pose"))
        assertTrue(rows.contains("count"))
    }

    @Test
    fun sexCarriesTheSexOnlyRows() {
        val rows = keys(testSpec(mode = PromptMode.Sex))
        assertTrue(rows.contains("family"))
        assertTrue(rows.contains("ratio"))
        assertTrue(rows.contains("stages"))
    }

    @Test
    fun nudityDropsTheClothingRow() {
        assertFalse(keys(testSpec(mode = PromptMode.Sex, exposure = listOf("nude"))).contains("clothing"))
        assertTrue(keys(testSpec(mode = PromptMode.Sfw)).contains("clothing"))
    }

    @Test
    fun everyRowNamesAPageTheWizardHas() {
        ManifestModel.items(fullSpec(), PromptLang.English).forEach { row ->
            assertTrue(WizardModel.PAGE_BY_KEY.containsKey(row.pageKey), row.pageKey)
        }
    }

    @Test
    fun rowsShowTheCurrentValues() {
        val spec = fullSpec()
        val rows = ManifestModel.items(spec, PromptLang.English).associate { it.pageKey to it.value }
        assertEquals("sex", rows["mode"])
        assertEquals("revealing, open", rows["exposure"])
        assertEquals("0.3", rows["ratio"])
        assertEquals("7", rows["count"])
        assertEquals("nipples", rows["chest"])
        assertEquals("navel", rows["belly"])
        assertEquals("Expression=orgasm", rows["face"])
        assertTrue(rows["stages"]!!.contains("during 0.4"), rows["stages"]!!)
        assertFalse(rows["stages"]!!.contains("object_insertion"), rows["stages"]!!)
    }

    @Test
    fun theClothingRowCountsThePicks() {
        val spec = fullSpec()
        val rows = ManifestModel.items(spec, PromptLang.English).associate { it.pageKey to it.value }
        assertEquals("1 selected", rows["clothing"])
        assertEquals("1 selected", rows["pose"])
        assertEquals("1 selected", rows["scene"])
        assertEquals("1 selected", rows["family"])
        val anySpec = testSpec(mode = PromptMode.Sex)
        val anyRows = ManifestModel.items(anySpec, PromptLang.English).associate { it.pageKey to it.value }
        assertEquals("Any", anyRows["clothing"])
        assertEquals("Any", anyRows["family"])
    }

    @Test
    fun allZeroStagesFallBackToTheHint() {
        val spec = testSpec(mode = PromptMode.Sex, stageWeights = SEX_STAGES.associateWith { 0.0 })
        val rows = ManifestModel.items(spec, PromptLang.Chinese).associate { it.pageKey to it.value }
        assertEquals(t(PromptLang.Chinese, "manifest_stage_fallback"), rows["stages"])
    }

    @Test
    fun groupsSwitchedOffAreHiddenFromTheFaceRow() {
        val spec = testSpec(
            face = testFace(
                "expression" to FacePick.Tags(listOf("smile")),
                "gaze" to FacePick.None,
                "eyes" to FacePick.None,
                "mouth" to FacePick.None,
                "blush" to FacePick.None,
                "tears" to FacePick.None,
            ),
        )
        val value = ManifestModel.faceValue(spec, PromptLang.Chinese)
        assertTrue(value.contains("总表情=smile"), value)
        assertFalse(value.contains("视线"), value)
    }
}

class WizardNavigationTest {
    private fun indexOf(key: String): Int = WizardModel.PAGES.indexOfFirst { it.key == key }

    @Test
    fun backFromPoseLandsOnSceneOutsideSex() {
        val spec = testSpec(mode = PromptMode.Sfw)
        assertEquals(indexOf("scene"), WizardModel.previousPage(indexOf("pose"), spec))
    }

    @Test
    fun backFromPoseLandsOnStagesInSexMode() {
        val spec = testSpec(mode = PromptMode.Sex)
        assertEquals(indexOf("stages"), WizardModel.previousPage(indexOf("pose"), spec))
    }

    @Test
    fun backFromChestLandsOnExposureWhenNude() {
        val spec = testSpec(mode = PromptMode.Sfw, exposure = listOf("nude"))
        assertEquals(indexOf("exposure"), WizardModel.previousPage(indexOf("chest"), spec))
    }

    @Test
    fun forwardSkipsThePagesTheSpecDoesNotUse() {
        val spec = testSpec(mode = PromptMode.Sfw)
        assertEquals(indexOf("scene"), WizardModel.nextPage(indexOf("scene") - 1, spec))
        val sex = testSpec(mode = PromptMode.Sex)
        assertEquals(indexOf("family"), WizardModel.nextPage(indexOf("scene"), sex))
        assertEquals(WizardModel.PAGES.size, WizardModel.nextPage(indexOf("count"), spec))
    }

    @Test
    fun theClothingPageIsSkippedForNudity() {
        assertFalse(WizardModel.PAGE_BY_KEY.getValue("clothing").applies(testSpec(exposure = listOf("nude"))))
        assertTrue(WizardModel.PAGE_BY_KEY.getValue("clothing").applies(testSpec(exposure = listOf("covered"))))
        assertFalse(WizardModel.PAGE_BY_KEY.getValue("ratio").applies(testSpec(mode = PromptMode.Nsfw)))
        assertTrue(WizardModel.PAGE_BY_KEY.getValue("ratio").applies(testSpec(mode = PromptMode.Sex)))
    }

    @Test
    fun switchingModeResetsTheModeSpecificFields() {
        val spec = testSpec(
            mode = PromptMode.Sex,
            exposure = listOf("open"),
            chest = "nipples",
            belly = "navel",
            face = testFace("expression" to FacePick.Tags(listOf("orgasm"))),
        )
        WizardModel.applyModeChange(spec, PromptMode.Sfw)
        assertEquals(PromptMode.Sfw, spec.mode)
        assertEquals(listOf("covered", "casual"), spec.exposure)
        assertEquals("covered", spec.chest)
        assertEquals("covered", spec.belly)
        assertEquals(defaultFace(), spec.face)
    }

    @Test
    fun switchingToTheSameModeKeepsTheEdits() {
        val spec = testSpec(mode = PromptMode.Sfw, chest = "cleavage")
        WizardModel.applyModeChange(spec, PromptMode.Sfw)
        assertEquals("cleavage", spec.chest)
    }

    @Test
    fun allowedClothingFollowsTheExposureSetting() {
        val matrix = miniMatrix()
        val covered = WizardModel.allowedClothing(matrix, listOf("covered"))
        assertTrue(covered.any { it.blob.contains("serafuku") })
        assertTrue(covered.any { it.blob.contains("wedding dress") })
        assertFalse(covered.any { it.blob.contains("white sundress") })
        assertFalse(covered.any { it.blob.contains("bikini") })

        // "open" adds every row the matrix marks as openable, whatever its own bucket.
        val open = WizardModel.allowedClothing(matrix, listOf("covered", "open"))
        assertTrue(open.any { it.blob.contains("white sundress") })
        assertTrue(open.any { it.blob.contains("bikini") })
        assertEquals(open.size, open.map { it.key }.distinct().size)
        assertTrue(WizardModel.allowedClothing(matrix, listOf("nude")).isEmpty())
    }

    @Test
    fun bodyAndFaceWarningsFollowTheMode() {
        assertTrue(WizardModel.needsSfwBodyWarning(testSpec(mode = PromptMode.Sfw, chest = "breasts_out")))
        assertFalse(WizardModel.needsSfwBodyWarning(testSpec(mode = PromptMode.Nsfw, chest = "breasts_out")))
        assertTrue(
            WizardModel.needsSfwFaceWarning(
                testSpec(mode = PromptMode.Sfw, face = testFace("expression" to FacePick.Tags(listOf("orgasm")))),
            ),
        )
        assertFalse(
            WizardModel.needsSfwFaceWarning(
                testSpec(mode = PromptMode.Sfw, face = testFace("expression" to FacePick.Tags(listOf("smile")))),
            ),
        )
    }
}

class RatioAndCountTest {
    @Test
    fun aRatioAboveOneIsAPercentage() {
        assertEquals(0.5, parseVaginalRatio("0.5"))
        assertEquals(0.5, parseVaginalRatio("50"))
        assertEquals(1.0, parseVaginalRatio("1"))
        assertEquals(0.0, parseVaginalRatio("0"))
        assertEquals(PromptLimits.VAGINAL_RATIO_DEFAULT, parseVaginalRatio(""))
        assertEquals(PromptLimits.VAGINAL_RATIO_DEFAULT, parseVaginalRatio("  "))
    }

    @Test
    fun anUnparsableRatioKeepsTheCurrentValue() {
        assertEquals(0.75, parseVaginalRatio("abc", default = 0.75))
    }

    @Test
    fun stageWeightsAreStoredAsThousandths() {
        assertEquals(1000, stageWeightValue(testSpec(stageWeights = mapOf(SexStage.During to 1.0)), SexStage.During))
        assertEquals(0, stageWeightValue(testSpec(stageWeights = emptyMap()), SexStage.During))
        assertEquals(500, stageWeightValue(testSpec(stageWeights = mapOf(SexStage.During to 0.5)), SexStage.During))
        assertEquals(1000, stageWeightValue(testSpec(stageWeights = mapOf(SexStage.During to 3.0)), SexStage.During))
    }

    @Test
    fun thePoseWeightPrefersThePickedPoses() {
        val dog = find(miniMatrix().poses, "doggystyle")
        val prefer = testSpec(poseAny = false, poseKeys = setOf(dog.key))
        assertEquals(PromptLimits.PREFER_WEIGHT, PromptGenerator.poseWeight(dog, prefer))
        val any = testSpec(poseAny = true, poseKeys = setOf(dog.key))
        assertEquals(1, PromptGenerator.poseWeight(dog, any))
        val other = testSpec(poseAny = false, poseKeys = setOf(listOf("not this pose")))
        assertEquals(1, PromptGenerator.poseWeight(dog, other))
    }

    @Test
    fun theVaginalRatioWeightsBothChannelPoses() {
        val matrix = miniMatrix()
        val both = find(matrix.poses, "cowgirl position")
        val vaginalSpec = testSpec(mode = PromptMode.Sex, vaginalRatio = 1.0)
        val analSpec = testSpec(mode = PromptMode.Sex, vaginalRatio = 0.0)
        assertEquals(1000, PromptGenerator.poseWeight(both, vaginalSpec))
        assertEquals(1000, PromptGenerator.poseWeight(both, analSpec))
        val analOnly = find(matrix.poses, "full nelson")
        assertEquals(0, PromptGenerator.poseWeight(analOnly, vaginalSpec))
        assertEquals(1000, PromptGenerator.poseWeight(analOnly, analSpec))
    }

    @Test
    fun theFamilyFilterZeroesOtherFamilies() {
        val matrix = miniMatrix()
        val dog = find(matrix.poses, "doggystyle")
        val spec = testSpec(
            mode = PromptMode.Sex,
            familyAny = false,
            families = setOf(PoseFamily.GirlOnTop),
        )
        assertEquals(0, PromptGenerator.poseWeight(dog, spec))
        val cowgirl = find(matrix.poses, "cowgirl position")
        assertTrue(PromptGenerator.poseWeight(cowgirl, spec) > 0)
    }
}

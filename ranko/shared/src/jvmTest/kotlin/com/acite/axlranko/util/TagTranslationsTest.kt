package com.acite.axlranko.util

import com.acite.axlranko.data.TrainerRepo
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TagTranslationsTest {

    private val sample = """
        tag_id,name,category,count,name,Translated
        1女,1girl,0,1,1girl,1女
        长发,long hair,0,1,long hair,长发
        普遍,general,9,1,general,普遍
        christmas,christmas,0,1,christmas,
    """.trimIndent()

    @Test
    fun displayUsesEnglishThenChinese() {
        val lex = TagLexicon.parse(sample)
        assertEquals("1girl [1女]", lex.display("1girl"))
        assertEquals("long hair [长发]", lex.display("long hair"))
        assertEquals("long_hair [长发]", lex.display("long_hair"))
        assertEquals("1Girl [1女]", lex.display("1Girl"))
    }

    @Test
    fun missingOrBlankTranslationStaysEnglish() {
        val lex = TagLexicon.parse(sample)
        assertEquals("unknown tag", lex.display("unknown tag"))
        assertEquals("christmas", lex.display("christmas"))
        assertNull(lex.chinese("christmas"))
    }

    @Test
    fun identicalTranslationIsNotWrapped() {
        val lex = TagLexicon.parse(
            """
            tag_id,name,category,count,name,Translated
            solo,solo,0,1,solo,solo
            """.trimIndent(),
        )
        assertEquals("solo", lex.display("solo"))
    }

    @Test
    fun quotedCsvFieldKeepsInnerQuotes() {
        val quoted = "tag_id,name,category,count,name,Translated\n" +
            "zh,\"don't say \"\"lazy\"\"\",0,1,\"don't say \"\"lazy\"\"\",切勿说“懒惰”\n"
        val lex = TagLexicon.parse(quoted)
        val english = "don't say \"lazy\""
        assertEquals("切勿说“懒惰”", lex.chinese(english))
        assertEquals("$english [切勿说“懒惰”]", lex.display(english))
    }

    @Test
    fun searchMatchesEnglishOrChinese() {
        val lex = TagLexicon.parse(sample)
        assertTrue(lex.matchesQuery("1girl", "1g"))
        assertTrue(lex.matchesQuery("1girl", "1女"))
        assertTrue(lex.matchesQuery("long hair", "长发"))
        assertFalse(lex.matchesQuery("1girl", "长发"))
        assertTrue(lex.matchesQuery("1girl", ""))
    }

    @Test
    fun parseCsvLineSplitsQuotedCommasAndDoubledQuotes() {
        assertEquals(
            listOf("a", "b,c", "say \"hi\"", "d"),
            parseCsvLine("a,\"b,c\",\"say \"\"hi\"\"\",d"),
        )
    }

    @Test
    fun repoCsvTranslatesShippedTags() {
        val root = TrainerRepo.findRoot() ?: return
        val csv = File(root, "tagger/selected_tags.csv")
        if (!csv.isFile) return
        val lex = TagLexicon.load(csv)
        assertEquals("1女", lex.chinese("1girl"))
        assertEquals("1girl [1女]", lex.display("1girl"))
        assertEquals("solo [单人]", lex.display("solo"))
        assertEquals("long hair [长发]", lex.display("long hair"))
    }
}

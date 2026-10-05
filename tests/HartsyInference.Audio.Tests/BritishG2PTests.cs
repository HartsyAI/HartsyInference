using System.IO;
using System.Text;
using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>misaki's British mode, behind Kokoro's <c>b</c> voices, over real <c>gb_gold.json</c> entries. Expected
/// strings are what misaki's <c>en.G2P(british=True)</c> and <c>EspeakFallback(british=True)</c> print.</summary>
public sealed class BritishG2PTests
{
    private const string Gold = """
        {"the": "ðə", "dog": "dˈɒɡ", "watch": "wˈɒʧ", "we": "wiː", "wait": "wˈAt", "and": "and", "need": "nˈiːd",
         "it": "ɪt", "sit": "sˈɪt", "by": {"DEFAULT": "bI", "None": "bˈI"}, "water": "wˈɔːtə", "colour": "kˈʌlə",
         "of": "ɒv", "car": "kˈɑː"}
        """;

    private static EnglishG2P G2P()
    {
        using MemoryStream gold = new(Encoding.UTF8.GetBytes(Gold));
        using MemoryStream silver = new("{}"u8.ToArray());
        return new EnglishG2P(MisakiLexicon.FromStreams(gold, silver, british: true));
    }

    [Theory]
    // -s and -ed after a sibilant or d/t take ɪ, not American ᵻ, and t is never flapped.
    [InlineData("The dog watches.", "ðə dˈɒɡ wˈɒʧɪz.")]
    [InlineData("We waited and needed it.", "wˌiː wˈAtɪd and nˈiːdɪd ɪt.")]
    [InlineData("Sitting by the water.", "sˈɪtɪŋ bI ðə wˈɔːtə.")]
    [InlineData("The colour of the car.", "ðə kˈʌlə ɒv ðə kˈɑː.")]
    public void ToIpa_MatchesMisakiBritish(string text, string expected) => Assert.Equal(expected, G2P().ToIpa(text));

    [Theory]
    // espeak's tied IPA for words misaki lacks: British keeps its lengths and spells ə^ʊ as Q; American drops the
    // lengths and spells o^ʊ as O.
    [InlineData("zˈɛbɹə", true, "zˈɛbɹə")]
    [InlineData("zˈiːbɹə", false, "zˈibɹə")]
    [InlineData("ɡəndˈə^ʊlə", true, "ɡəndˈQlə")]
    [InlineData("ɡəndˈo^ʊlə", false, "ɡəndˈOlə")]
    [InlineData("təmˈɑːtə^ʊ", true, "təmˈɑːtQ")]
    [InlineData("təmˈe^ɪɾo^ʊ", false, "təmˈAɾO")]
    public void EspeakFallback_SpellsAsMisaki(string espeak, bool british, string expected)
        => Assert.Equal(expected, EspeakToMisaki.Convert(espeak, british));
}

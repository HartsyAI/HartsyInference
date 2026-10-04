using System.IO;
using System.Text;
using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The misaki port behind Kokoro's G2P, over a small lexicon of real <c>us_gold.json</c> entries. Every
/// expected string is what misaki itself (KPipeline's <c>en.G2P</c>) prints for the same input and dictionary words,
/// so a regression here is a divergence from the phonemes Kokoro was trained on.</summary>
public sealed class EnglishG2PTests
{
    private const string Gold = """
        {"hello": "həlˈO", "world": "wˈɜɹld", "to": "tu", "apple": "ˈæpᵊl", "go": "ɡˌO", "don't": "dˈOnt",
         "water": "wˈɔɾəɹ", "dog": "dˈɔɡ", "A": "ˈA", "G": "ʤˈi", "P": "pˈi", "U": "jˈu", "one": "wˈʌn",
         "three": "θɹˈi", "five": "fˈIv", "thousand": "θˈWzᵊnd", "dollar": "dˈɑləɹ", "cent": "sˈɛnt",
         "point": "pˈYnt", "nineteen": "nˌIntˈin", "ninety": "nˈIndi", "percent": "pəɹsˈɛnt", "first": "fˈɜɹst",
         "twenty": "twˈɛnti", "four": "fˈɔɹ", "walk": "wˈɔk", "that": {"DEFAULT": "ðæt", "DT": "ðˈæt"},
         "used": {"DEFAULT": "jˈuzd", "VBD": "jˈust"}, "and": "ænd", "fifty": "fˈɪfti", "minus": "mˈInəs",
         "at": "æt", "end": "ˈɛnd", "say": "sˈA", "is": "ɪz", "of": "ʌv", "he": "hi", "it": "ɪt", "am": "æm",
         "I": "ˈI", "in": "ɪn", "O": "ˈO", "million": "mˈɪljᵊn"}
        """;

    private static EnglishG2P G2P(string cmudict = "")
    {
        using MemoryStream gold = new(Encoding.UTF8.GetBytes(Gold));
        using MemoryStream silver = new("{}"u8.ToArray());
        return new EnglishG2P(MisakiLexicon.FromStreams(gold, silver), new MemoryStream(Encoding.UTF8.GetBytes(cmudict)));
    }

    [Theory]
    [InlineData("Hello, world!", "həlˈO, wˈɜɹld!")]
    // "the" and "to" read by the next word's first sound; function words keep the dictionary's missing stress.
    [InlineData("the apple and the dog", "ði ˈæpᵊl ænd ðə dˈɔɡ")]
    [InlineData("to go, to an apple", "tə ɡˌO, tə ɐn ˈæpᵊl")]
    [InlineData("A dog", "ɐ dˈɔɡ")]
    [InlineData("Is it?", "ˌɪz ɪt?")]
    // Curly apostrophes are apostrophes: "don’t" is one word, not "don" + "t".
    [InlineData("I don’t walk", "ˌI dˈOnt wˈɔk")]
    // misaki v1's flap spelling, the symbol Kokoro-82M was trained with.
    [InlineData("water", "wˈɔTəɹ")]
    [InlineData("GPU", "ʤˌipˌijˈu")]
    [InlineData("Walked, walks, walking.", "wˈɔkt, wˈɔks, wˈɔkɪŋ.")]
    [InlineData("He used to say that.", "hˌi jˈust tə sˈA ðˈæt.")]
    [InlineData("The end.", "ði ˈɛnd.")]
    [InlineData("\"Hello\" (world)…", "“həlˈO” (wˈɜɹld)…")]
    public void ToIpa_MatchesMisaki(string text, string expected) => Assert.Equal(expected, G2P().ToIpa(text));

    [Theory]
    [InlineData("The 1st of 1990", "ðə fˈɜɹst ʌv nˌIntˈin nˈIndi")]
    [InlineData("$5.50", "fˈIv dˈɑləɹz ænd fˈɪfti sˈɛnts")]
    [InlineData("-3.5", "mˈInəs θɹˈi pYnt fˈIv")]
    [InlineData("50%", "fˈɪfti pəɹsˈɛnt")]
    [InlineData("1,000", "wˈʌn θˈWzᵊnd")]
    // The currency follows a spaced magnitude, as spaCy's CD tag carries it in misaki.
    [InlineData("$5 million", "fˈIv mˈɪljᵊn dˈɑləɹz")]
    // Unicode decimal digits read as numbers (misaki numeric_if_needed).
    [InlineData("٣", "θɹˈi")]
    public void ToIpa_ReadsNumbersAsMisaki(string text, string expected) => Assert.Equal(expected, G2P().ToIpa(text));

    [Fact]
    public void ToIpa_OutOfLexiconWord_FallsBackToCmudict_ThenLetterRules()
    {
        EnglishG2P g2p = G2P("zylophone  Z AY1 L AH0 F OW2 N\n");
        Assert.Equal("zˈIləfˌOn", g2p.ToIpa("zylophone"));
        Assert.Equal(EnglishG2P.LetterToSound("blorft"), g2p.ToIpa("blorft"));
        // A vowel-less piece is an abbreviation and is spelled ("kg" → K G).
        Assert.Equal(G2P().ToIpa("GP"), G2P().ToIpa("gp"));
    }

    [Fact]
    public void ToIpa_ANumberTooLongForALong_IsReadDigitByDigit()
    {
        Assert.Equal(string.Join(' ', Enumerable.Repeat("wˈʌn", 23)), G2P().ToIpa(new string('1', 23)));
    }

    [Fact]
    public void ToIpa_AWordBeforeAFallbackCompound_SeesTheCompoundsFirstSound()
    {
        // "blorft-zz" is not in the lexicon, so it falls back whole; "the" must hear its consonant, not "apple".
        Assert.StartsWith("ðə ", G2P().ToIpa("the blorft-zz apple"));
    }

    [Fact]
    public void ObsoleteCmudictOnlyConstructor_StillReadsWordsAndNumbers()
    {
#pragma warning disable CS0618
        EnglishG2P g2p = new(new MemoryStream(Encoding.UTF8.GetBytes("hello  HH AH0 L OW1\ntwo  T UW1\n")));
#pragma warning restore CS0618
        Assert.Equal(2, g2p.WordCount);
        Assert.Equal("həlˈO tˈu", g2p.ToIpa("hello 2"));
    }

    [Fact]
    public void LetterToSound_UsesMisakiSingleSymbols_AndStressesTheFirstVowel()
    {
        Assert.Equal("ʧˈɑp", EnglishG2P.LetterToSound("chop"));
        Assert.Equal("bˈA", EnglishG2P.LetterToSound("bay"));
        Assert.Equal("hˈæpi", EnglishG2P.LetterToSound("happy"));
    }

    [Fact]
    public void ToIpa_ReadsANewlineAsASpace()
    {
        Assert.Equal("həlˈO wˈɜɹld", G2P().ToIpa("hello\nworld"));
    }

    [Theory]
    [InlineData(105, "one hundred and five", "one hundred and fifth")]
    [InlineData(1234, "one thousand, two hundred and thirty-four", "one thousand, two hundred and thirty-fourth")]
    [InlineData(100001, "one hundred thousand and one", "one hundred thousand and first")]
    [InlineData(21, "twenty-one", "twenty-first")]
    public void NumberWords_MatchNum2words(long n, string cardinal, string ordinal)
    {
        Assert.Equal(cardinal, EnglishNumberWords.Cardinal(n));
        Assert.Equal(ordinal, EnglishNumberWords.Ordinal(n));
    }

    [Theory]
    [InlineData(1905, "nineteen oh-five")]
    [InlineData(2005, "two thousand and five")]
    [InlineData(2024, "twenty twenty-four")]
    [InlineData(1800, "eighteen hundred")]
    public void NumberWords_Years_MatchNum2words(long year, string expected) => Assert.Equal(expected, EnglishNumberWords.Year(year));

    [Theory]
    [InlineData("3.14", "three point one four")]
    [InlineData("3.50", "three point five")]
    [InlineData("1.0", "one")]
    public void NumberWords_Decimals_MatchNum2words(string text, string expected) => Assert.Equal(expected, EnglishNumberWords.Decimal(text));
}

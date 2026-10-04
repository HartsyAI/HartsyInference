using HartsyInference.Audio.Models.Kokoro;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro's port of KPipeline's <c>waterfall_last</c>: no chunk may exceed PLBERT's 510 phonemes, and a cut
/// lands on the strongest pause available so long text is not split mid-phrase.</summary>
public sealed class KokoroPhonemeChunkerTests
{
    [Fact]
    public void ShortInput_IsOneTrimmedChunk()
    {
        Assert.Equal(["həlˈO wˈɜɹld."], KokoroPhonemeChunker.Split("  həlˈO wˈɜɹld.  "));
        Assert.Empty(KokoroPhonemeChunker.Split("   "));
    }

    [Fact]
    public void CutsAfterTheLastSentenceEnd_BeforeWeakerPauses()
    {
        // "aa, bb. cc, dd" with a cap that cannot hold it all: the period wins over the later comma.
        Assert.Equal(["aa, bb.", "cc, dd"], KokoroPhonemeChunker.Split("aa, bb. cc, dd", maxLength: 12));
    }

    [Fact]
    public void FallsBackToColonThenComma_ThenSpace_ThenHardCut()
    {
        Assert.Equal(["aa;", "bb cc, dd"], KokoroPhonemeChunker.Split("aa; bb cc, dd", maxLength: 10));
        Assert.Equal(["aa,", "bb cc dd"], KokoroPhonemeChunker.Split("aa, bb cc dd", maxLength: 8));
        Assert.Equal(["aa bb", "cc dd"], KokoroPhonemeChunker.Split("aa bb cc dd", maxLength: 6));
        Assert.Equal(["abcd", "efgh"], KokoroPhonemeChunker.Split("abcdefgh", maxLength: 4));
    }

    [Fact]
    public void StepsPastAClosingQuoteOrBracket()
    {
        Assert.Equal(["“aa.”", "bb cc"], KokoroPhonemeChunker.Split("“aa.” bb cc", maxLength: 7));
    }

    [Fact]
    public void ANewline_IsAlwaysABoundary()
    {
        Assert.Equal(["aa", "bb"], KokoroPhonemeChunker.Split("aa\nbb"));
        Assert.Equal(["aa", "bb"], KokoroPhonemeChunker.Split("aa\r\n\n bb"));
    }

    [Fact]
    public void EveryChunkFits_AndNoPhonemeIsLost()
    {
        string word = "ðə kwˈɪk bɹˈWn fˈɑks, ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ. ";
        string text = string.Concat(Enumerable.Repeat(word, 40));
        IReadOnlyList<string> chunks = KokoroPhonemeChunker.Split(text);
        Assert.True(chunks.Count > 1);
        Assert.All(chunks, c => Assert.InRange(c.Length, 1, KokoroPhonemeChunker.MaxPhonemes));
        Assert.All(chunks, c => Assert.EndsWith(".", c));
        Assert.Equal(text.Replace(" ", ""), string.Concat(chunks).Replace(" ", ""));
    }
}

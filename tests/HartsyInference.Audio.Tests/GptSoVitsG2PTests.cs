using System.Collections.Generic;
using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Validates the GPT-SoVITS English G2P (CMUdict → ARPAbet) and the language-routing front-end
/// (checkpoint-free; golden phones from <c>cmudict.rep</c>).</summary>
public sealed class GptSoVitsG2PTests
{
    [Fact]
    public void English_CmudictWords_MatchArpabet()
    {
        List<string> phones = GptSoVitsEnglishG2P.G2P("hello world");
        Assert.Equal(new[] { "HH", "AH0", "L", "OW1", "W", "ER1", "L", "D" }, phones);
        foreach (string p in phones) Assert.True(GptSoVitsSymbols.IdOf(p) >= 0, $"'{p}' not a symbol");
    }

    [Fact]
    public void English_ShortUtterance_IsCommaPadded()
    {
        // "the" → [DH, AH0] is < 4 phones, so a leading "," is inserted (upstream behavior).
        List<string> phones = GptSoVitsEnglishG2P.G2P("the");
        Assert.Equal(",", phones[0]);
        Assert.Contains("DH", phones);
        Assert.Contains("AH0", phones);
    }

    [Fact]
    public void Frontend_RoutesChineseAndEnglish_AndMapsToIds()
    {
        (List<string> zhPh, List<int>? zhW2p) = GptSoVitsFrontend.CleanText("你好", "zh");
        Assert.NotNull(zhW2p);                                  // character language → word2ph present
        Assert.Equal(new[] { "n", "i3", "h", "ao3" }, zhPh);
        int[] ids = GptSoVitsSymbols.ToSequence(zhPh);          // all map cleanly to symbol ids
        Assert.All(ids, id => Assert.InRange(id, 0, GptSoVitsSymbols.Count - 1));

        (List<string> enPh, List<int>? enW2p) = GptSoVitsFrontend.CleanText("speech", "en");
        Assert.Null(enW2p);                                     // english → no word2ph
        Assert.Contains("S", enPh);
        Assert.Contains("IY1", enPh);
    }

    [Fact]
    public void Frontend_UnportedLanguage_Throws()
        => Assert.Throws<System.NotSupportedException>(() => GptSoVitsFrontend.CleanText("こんにちは", "ja"));
}

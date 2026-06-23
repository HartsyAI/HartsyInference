using System.Collections.Generic;
using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Validates the Chinese (zh) per-character default-pinyin G2P against upstream pypinyin →
/// <c>chinese2</c> assembly (checkpoint-free; g2pW/tone-sandhi refinements not exercised here).</summary>
public sealed class ChineseG2PTests
{
    [Fact]
    public void G2P_CharDefault_MatchesUpstream()
    {
        (List<string> phones, List<int> word2ph) = ChineseG2P.G2P("你好世界");
        Assert.Equal(new[] { "n", "i3", "h", "ao3", "sh", "ir4", "j", "ie4" }, phones);
        Assert.Equal(new[] { 2, 2, 2, 2 }, word2ph);
        // sum(word2ph) == phones.Count, and every phone is a valid symbol (the model invariant).
        Assert.Equal(phones.Count, Sum(word2ph));
        foreach (string p in phones) Assert.True(GptSoVitsSymbols.IdOf(p) >= 0, $"'{p}' not a symbol");
    }

    [Fact]
    public void G2P_NormalizesPunctuation()
    {
        (List<string> phones, List<int> word2ph) = ChineseG2P.G2P("你，好。");
        // ，→ "," and 。→ "." as single-symbol tokens (word2ph 1).
        Assert.Contains(",", phones);
        Assert.Contains(".", phones);
        Assert.Equal(phones.Count, Sum(word2ph));
        Assert.Equal(4, word2ph.Count);     // 你 + , + 好 + .
    }

    [Fact]
    public void CharPinyin_KnownChar_ReturnsInitialFinal()
    {
        (string Initial, string Final)? py = ChineseG2P.CharPinyin('中');
        Assert.NotNull(py);
        Assert.Equal("zh", py!.Value.Initial);
        Assert.Equal("ong1", py.Value.Final);
    }

    private static int Sum(List<int> xs) { int s = 0; foreach (int x in xs) s += x; return s; }
}

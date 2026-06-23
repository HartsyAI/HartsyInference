using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Validates the deterministic pinyin→phoneme assembly (opencpop-strict + rep-maps) against the
/// upstream GPT-SoVITS <c>chinese2._g2p</c> output for known pinyin (checkpoint-free). Goldens from pypinyin
/// initials/finals → the upstream syllable→phoneme assembly.</summary>
public sealed class PinyinPhonemizerTests
{
    [Theory]
    // 你好 / 中国 / 拼音: (initials, finals_tone3) → phones, word2ph.
    [InlineData(new[] { "n", "h" }, new[] { "i3", "ao3" }, new[] { "n", "i3", "h", "ao3" }, new[] { 2, 2 })]
    [InlineData(new[] { "zh", "g" }, new[] { "ong1", "uo2" }, new[] { "zh", "ong1", "g", "uo2" }, new[] { 2, 2 })]
    [InlineData(new[] { "p", "" }, new[] { "in1", "in1" }, new[] { "p", "in1", "y", "in1" }, new[] { 2, 2 })]
    public void Assemble_MatchesUpstream(string[] initials, string[] finals, string[] expPhones, int[] expWord2Ph)
    {
        (System.Collections.Generic.List<string> phones, System.Collections.Generic.List<int> word2ph) =
            PinyinPhonemizer.Assemble(initials, finals);
        Assert.Equal(expPhones, phones);
        Assert.Equal(expWord2Ph, word2ph);
        // Every produced phone must be in the model's symbol set (UNK would break the embedding).
        foreach (string p in phones) Assert.True(GptSoVitsSymbols.IdOf(p) >= 0, $"phone '{p}' not in symbol set");
    }

    [Fact]
    public void Assemble_Punctuation_IsSingleSymbolWord2Ph1()
    {
        (System.Collections.Generic.List<string> phones, System.Collections.Generic.List<int> word2ph) =
            PinyinPhonemizer.Assemble([","], [","]);
        Assert.Equal([","], phones);
        Assert.Equal([1], word2ph);
    }
}

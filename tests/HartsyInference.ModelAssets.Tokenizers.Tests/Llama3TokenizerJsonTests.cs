using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Numeric parity for the Llama-3 byte-level BPE built from the embedded <c>tokenizer.json</c> via
/// <see cref="HfTokenizerJson"/>. Golden ids are from the HuggingFace <c>tokenizers</c> library
/// (Llama-3.2 tokenizer, <c>add_special_tokens=False</c>) — they exercise the family-specific split regex
/// (digit grouping, contraction casing) and <c>ignore_merges</c> that the Orpheus / CSM front-ends rely on.</summary>
public sealed class Llama3TokenizerJsonTests
{
    private static GgufTokenizer Load()
    {
        using Stream json = EmbeddedTokenizerResources.OpenLlama3TokenizerJson();
        return HfTokenizerJson.LoadByteLevelBpe(json);
    }

    [Theory]
    [InlineData("tara: Hello, world!", new[] { 83, 5169, 25, 22691, 11, 1917, 0 })]
    [InlineData("plain text without specials", new[] { 21435, 1495, 2085, 60874 })]
    [InlineData("The year was 2024 and there were 12345 reasons.",
        new[] { 791, 1060, 574, 220, 2366, 19, 323, 1070, 1051, 220, 4513, 1774, 8125, 13 })]
    [InlineData("Café déjà vu — naïve", new[] { 34, 2642, 978, 46939, 33614, 2001, 95980, 588 })]
    public void EncodeOrdinary_MatchesHuggingFaceIds(string text, int[] expected)
    {
        GgufTokenizer tok = Load();
        Assert.Equal(expected, tok.EncodeOrdinary(text));
    }

    [Fact]
    public void DigitGrouping_SplitsRunsIntoGroupsOfThree()
    {
        // The Llama-3 split regex groups digits in runs of <=3 (GPT-2's regex would not), so "2024" -> "202","4".
        GgufTokenizer tok = Load();
        Assert.Equal(new[] { 2366, 19 }, tok.EncodeOrdinary("2024"));
    }

    [Fact]
    public void Vocab_HasFullLlama3Range_AndSpecialsAreControlTokens()
    {
        GgufTokenizer tok = Load();
        Assert.Equal(128_000, tok.BosId);
        Assert.Equal(128_001, tok.EosId);
        // Begin-of-text is a control token: it must NOT appear when encoding its literal as ordinary text.
        Assert.DoesNotContain(128_000, tok.EncodeOrdinary("<|begin_of_text|>"));
        // ...but special-aware encoding maps the literal to its id.
        Assert.Contains(128_000, tok.Encode("<|begin_of_text|>hello", addSpecial: true));
    }
}

using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>SentencePiece <c>add_space_prefix</c> with and without <c>prepend_scheme = first</c> (<c>tokenizer.ggml.metaspace.prepend_scheme</c>): the scheme decides whether text after a special token takes the leading ▁.</summary>
public sealed class SpmPrependSchemeTests
{
    // <unk>=0, <s>=1, <|x|>=2, ▁=3, a=4, ▁a=5, b=6
    private static readonly string[] Tokens = ["<unk>", "<s>", "<|x|>", "▁", "a", "▁a", "b"];
    private static readonly float[] Scores = [0f, 0f, 0f, -1f, -2f, -0.5f, -3f];
    private static readonly int[] Types = [2, 3, 3, 1, 1, 1, 1];

    private static SpmGgufTokenizer Build(bool onlyAtStart) =>
        new(Tokens, Scores, Types, bosId: 1, eosId: null, addSpacePrefix: true, prefixOnlyAtTextStart: onlyAtStart);

    [Fact]
    public void EveryOrdinarySegmentTakesThePrefixByDefault() => Assert.Equal([5, 2, 5], Build(onlyAtStart: false).Encode("a<|x|>a", addSpecial: true));

    [Fact]
    public void OnlyAnOpeningSegmentTakesThePrefixUnderTheFirstScheme() => Assert.Equal([5, 2, 4], Build(onlyAtStart: true).Encode("a<|x|>a", addSpecial: true));

    [Fact]
    public void ATextThatOpensWithASpecialTokenTakesNoPrefixUnderTheFirstScheme() =>
        Assert.Equal([2, 4, 6], Build(onlyAtStart: true).Encode("<|x|>ab", addSpecial: true));

    [Fact]
    public void PlainTextStillTakesThePrefixUnderTheFirstScheme() => Assert.Equal([5], Build(onlyAtStart: true).Encode("a", addSpecial: true));

    [Fact]
    public void TextAfterASpecialTokenFollowsTheScheme()
    {
        Assert.Equal([4], Build(onlyAtStart: true).EncodeAfterSpecial("a"));
        Assert.Equal([5], Build(onlyAtStart: false).EncodeAfterSpecial("a"));
    }
}

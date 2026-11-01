using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The rendered-prompt encoder with a SentencePiece tokenizer: under <c>prepend_scheme = first</c> the text after a special token takes no leading ▁
/// (the <c>&lt;|im_start|&gt;system</c> case), and by default it does.</summary>
public sealed class SpecialLiteralEscaperSpmTests
{
    // <unk>=0, <s>=1, <|x|>=2, ▁=3, a=4, ▁a=5, b=6
    private static readonly string[] Tokens = ["<unk>", "<s>", "<|x|>", "▁", "a", "▁a", "b"];
    private static readonly float[] Scores = [0f, 0f, 0f, -1f, -2f, -0.5f, -3f];
    private static readonly int[] Types = [2, 3, 3, 1, 1, 1, 1];

    private static SpmGgufTokenizer Build(bool onlyAtStart) =>
        new(Tokens, Scores, Types, bosId: 1, eosId: null, addSpacePrefix: true, prefixOnlyAtTextStart: onlyAtStart);

    [Fact]
    public void FirstScheme_GivesNoPrefixAfterASpecialToken() =>
        Assert.Equal([2, 4, 2, 4], SpecialLiteralEscaper.EncodeRendered("<|x|>a<|x|>a", Build(onlyAtStart: true)));

    [Fact]
    public void FirstScheme_StillPrefixesATextThatOpensTheRender() =>
        Assert.Equal([5, 2, 4], SpecialLiteralEscaper.EncodeRendered("a<|x|>a", Build(onlyAtStart: true)));

    [Fact]
    public void DefaultScheme_PrefixesEverySegment() =>
        Assert.Equal([2, 5, 2, 5], SpecialLiteralEscaper.EncodeRendered("<|x|>a<|x|>a", Build(onlyAtStart: false)));
}

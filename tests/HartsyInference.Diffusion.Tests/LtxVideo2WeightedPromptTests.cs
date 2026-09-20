using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>LTX-2 assembles its own conditioning sequence, so the weight array has to line up with ids the
/// recipe built rather than ids a tokenizer returned. These use a stand-in tokenizer: the real Gemma vocabs are
/// side models rather than embedded resources, and what is worth pinning here is the assembly, not the BPE.</summary>
public sealed class LtxVideo2WeightedPromptTests
{
    /// <summary>One id per character, so a span's token count is its length — enough to check alignment
    /// exactly, and it keeps the test independent of any real vocabulary.</summary>
    private sealed class StubTokenizer : ILtx2PromptTokenizer
    {
        public const int StartId = 2;

        public int[] EncodeForConditioning(string text) => [StartId, .. text.Select(c => (int)c)];

        public int MinimumConditioningLength => 0;

        public IReadOnlyList<int> EncodeSpan(string text) => [.. text.Select(c => (int)c)];

        public int ConditioningStartId => StartId;
    }

    /// <summary>An unweighted prompt keeps the tokenizer's own whole-string encode, so wiring weighting moves no
    /// existing generation. `(fox:1.0)` is in the list because a weight of exactly one does nothing, but the
    /// grammar that expressed it must still come off the text.</summary>
    [Theory]
    [InlineData("a red fox in snow")]
    [InlineData(" leading space")]
    [InlineData("a red (fox:1.0) in snow")]
    public void AnUnweightedPromptKeepsTheWholeStringEncode(string prompt)
    {
        StubTokenizer tokenizer = new StubTokenizer();
        (int[] ids, float[]? weights) = LtxVideo2RecipePipeline.EncodeWeighted(tokenizer, prompt);
        Assert.Null(weights);
        Assert.Equal(tokenizer.EncodeForConditioning(prompt.Replace("(fox:1.0)", "fox")), ids);
    }

    /// <summary>A weighted prompt carries exactly one weight per id, including the sequence-start token, because
    /// the connector scales rows positionally from the front — a weight array off by one would shift every
    /// emphasis by a token and still render.</summary>
    [Fact]
    public void EveryIdHasAWeightAndTheStartTokenWeighsOne()
    {
        (int[] ids, float[]? weights) = LtxVideo2RecipePipeline.EncodeWeighted(new StubTokenizer(), "ab (cd:1.5) ef");
        Assert.NotNull(weights);
        Assert.Equal(ids.Length, weights!.Length);
        Assert.Equal(StubTokenizer.StartId, ids[0]);
        Assert.Equal(1f, weights[0]);
    }

    /// <summary>The weighted span's ids land where its weights do. With one id per character, "cd" at 1.5 must
    /// be exactly the two rows carrying 1.5 — which is the property the whole mechanism rests on.</summary>
    [Fact]
    public void TheWeightedSpansIdsAreTheOnesCarryingItsWeight()
    {
        (int[] ids, float[]? weights) = LtxVideo2RecipePipeline.EncodeWeighted(new StubTokenizer(), "ab (cd:1.5) ef");
        Assert.NotNull(weights);
        List<int> weighted = [];
        for (int i = 0; i < weights!.Length; i++)
        {
            if (weights[i] == 1.5f)
            {
                weighted.Add(ids[i]);
            }
        }
        Assert.Equal([(int)'c', (int)'d'], weighted);
    }

    /// <summary>The start token is added exactly once, not once per span — a per-span start would put a sentence
    /// beginning in the middle of the caption.</summary>
    [Fact]
    public void TheStartTokenAppearsOnlyAtTheFront()
    {
        (int[] ids, _) = LtxVideo2RecipePipeline.EncodeWeighted(new StubTokenizer(), "(a:1.2) (b:1.3) (c:1.4)");
        Assert.Equal(StubTokenizer.StartId, ids[0]);
        Assert.DoesNotContain(StubTokenizer.StartId, ids[1..]);
    }
}

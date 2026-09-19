using System.Linq;
using System.Collections.Generic;
using HartsyInference.Diffusion.Prompting;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins <see cref="WeightedTokenBuilder"/> against SwarmUI's <c>calc_leaf</c>/<c>uniform_weight</c>
/// (<c>SwarmText.py:401-411</c>, <c>:464-472</c>). The property that matters most is the negative one: a prompt with
/// no weighting syntax must produce exactly the ids a plain <c>EncodeRaw</c> would, because every downstream
/// mechanism gates on "did anything change" rather than multiplying by one.</summary>
public sealed class WeightedTokenBuilderTests
{
    /// <summary>A toy word-level tokenizer standing in for a real BPE: one id per whitespace-separated word.</summary>
    private static IReadOnlyList<int> Encode(string text)
    {
        List<int> ids = new List<int>();
        foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            ids.Add(word.GetHashCode(StringComparison.Ordinal) & 0xFFFF);
        }
        return ids;
    }

    [Fact]
    public void AnUnweightedPromptTokenizesExactlyLikeAPlainEncode()
    {
        WeightedTokenSequence built = WeightedTokenBuilder.Build("a red cat", Encode, [], []);
        int[] plain = [.. Encode("a red cat")];
        Assert.Equal(plain, built.Tokens);
        Assert.All(built.Weights, weight => Assert.Equal(1f, weight));
        Assert.True(built.IsUniformlyUnweighted);
    }

    [Fact]
    public void EveryIdOfASpanCarriesThatSpansWeight()
    {
        WeightedTokenSequence built = WeightedTokenBuilder.Build("a (red cat:1.5) sat", Encode, [], []);
        Assert.Equal(built.Tokens.Length, built.Weights.Length);
        Assert.False(built.IsUniformlyUnweighted);
        // "a " -> 1 id at 1.0, "red cat" -> 2 ids at 1.5, " sat" -> 1 id at 1.0.
        float[] expected = [1f, 1.5f, 1.5f, 1f];
        Assert.Equal(expected, built.Weights);
    }

    [Fact]
    public void TemplateIdsAreAppendedAtWeightOne()
    {
        int[] prefix = [101, 102];
        int[] suffix = [201];
        WeightedTokenSequence built = WeightedTokenBuilder.Build("(cat:2)", Encode, prefix, suffix);
        Assert.Equal(101, built.Tokens[0]);
        Assert.Equal(102, built.Tokens[1]);
        Assert.Equal(201, built.Tokens[^1]);
        Assert.Equal(1f, built.Weights[0]);
        Assert.Equal(1f, built.Weights[1]);
        Assert.Equal(1f, built.Weights[^1]);
        Assert.Equal(2f, built.Weights[2]);
    }

    /// <summary>A template's ids must not veto the uniform fallback: SwarmUI computes the uniform weight over the
    /// prompt's leaves, long before any tokenizer shell is wrapped around them.</summary>
    [Fact]
    public void UniformWeightIgnoresTemplateIdsAndWhitespaceSpans()
    {
        WeightedTokenSequence built = WeightedTokenBuilder.Build("(a:1.5) (b:1.5)", Encode, [101], [201]);
        Assert.Equal(1.5f, built.UniformWeight);
    }

    [Fact]
    public void DisagreeingSpanWeightsHaveNoUniformWeight()
    {
        WeightedTokenSequence built = WeightedTokenBuilder.Build("(a:1.5) (b:0.5)", Encode, [], []);
        Assert.Null(built.UniformWeight);
    }

    [Fact]
    public void AnUnweightedPromptIsUniformAtOne()
    {
        WeightedTokenSequence built = WeightedTokenBuilder.Build("a cat", Encode, [], []);
        Assert.Equal(1f, built.UniformWeight);
    }

    [Fact]
    public void AnEmptyPromptHasNoUniformWeightAndNoContentIds()
    {
        WeightedTokenSequence built = WeightedTokenBuilder.Build("", Encode, [101], [201]);
        int[] expected = [101, 201];
        Assert.Equal(expected, built.Tokens);
        Assert.Null(built.UniformWeight);
    }

    /// <summary>Nested emphasis compounds multiplicatively, the same as ComfyUI's <c>token_weights</c>; the builder
    /// must carry whatever <see cref="PromptWeighting"/> resolved rather than re-deriving it.</summary>
    [Fact]
    public void NestedEmphasisCompounds()
    {
        WeightedTokenSequence built = WeightedTokenBuilder.Build("((cat:1.5):2)", Encode, [], []);
        Assert.Equal(3.0, built.Weights[0], 5);
    }

    [Fact]
    public void SpansAreTokenizedIndividuallySoAMergeCannotCrossAWeightBoundary()
    {
        IReadOnlyList<WeightedSpan> spans = [new WeightedSpan("red", 1.5f), new WeightedSpan(" cat", 1f)];
        WeightedTokenSequence built = WeightedTokenBuilder.Build(spans, Encode, [], []);
        int[] expected = [.. Encode("red"), .. Encode(" cat")];
        Assert.Equal(expected, built.Tokens);
    }

    /// <summary>The acceptance criterion the plan states for every weighted family: a weight of 1.0 must be
    /// byte-identical to the plain prompt. Per-span tokenization breaks that on its own, because a byte-level BPE
    /// merges a leading space into the following word — so `a `/`red`/` cat` encoded separately is not `a red cat`.
    /// The scaling would then be a correct no-op applied to conditioning that had already changed.</summary>
    [Theory]
    [InlineData("a (red:1.0) cat")]
    [InlineData("(a:1.0) (red:1.0) (cat:1.0)")]
    public void EmphasisThatWeighsOneTokenizesExactlyLikeThePlainPrompt(string prompt)
    {
        List<string> calls = [];
        IReadOnlyList<int> Encode(string text)
        {
            calls.Add(text);
            return [.. text.Select(c => (int)c)];
        }

        WeightedTokenSequence weighted = WeightedTokenBuilder.Build(prompt, Encode, [], []);
        Assert.Equal(["a red cat"], calls);
        Assert.Equal("a red cat".Select(c => (int)c), weighted.Tokens);
        Assert.All(weighted.Weights, w => Assert.Equal(1f, w));
    }

    /// <summary>The control: a weight that does something still splits, which is the parity behaviour.</summary>
    [Fact]
    public void ARealWeightStillTokenizesItsSpanAlone()
    {
        List<string> calls = [];
        IReadOnlyList<int> Encode(string text)
        {
            calls.Add(text);
            return [.. text.Select(c => (int)c)];
        }

        WeightedTokenBuilder.Build("a (red:1.5) cat", Encode, [], []);
        Assert.Equal(["a ", "red", " cat"], calls);
    }
}

using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests.Sampling;

/// <summary>The repetition penalty applies once per DISTINCT token, as Hugging Face and llama.cpp apply it. Applying it per occurrence
/// divided a token's logit by <c>penalty^n</c> after n occurrences, which wrecked long greedy outputs (misspelled identifiers in a
/// code review after a few thousand tokens). Nothing fails loudly when it regresses, so the arithmetic is pinned here.</summary>
public sealed class RepetitionPenaltyOncePerTokenTests
{
    [Fact]
    public void ARepeatedToken_IsPenalizedOnce()
    {
        float[] logits = [0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 2.0f, 0.5f, -2.0f];
        new RepetitionPenaltyStep(2.0f).Apply(logits, [5, 5, 5, 7, 7, 5]);
        Assert.Equal(1.0f, logits[5]);    // 2 / 2, not 2 / 2^4
        Assert.Equal(-4.0f, logits[7]);   // -2 * 2, not -2 * 2^2
        Assert.Equal(0.5f, logits[0]);    // never generated: untouched
    }

    [Fact]
    public void EachCallStartsFresh_SoTheSameStepServesEveryDecodeStep()
    {
        RepetitionPenaltyStep step = new(2.0f);
        float[] first = [4.0f, 4.0f];
        step.Apply(first, [0, 0]);
        float[] second = [4.0f, 4.0f];
        step.Apply(second, [0, 1, 0]);
        Assert.Equal([2.0f, 4.0f], first);
        Assert.Equal([2.0f, 2.0f], second);
    }

    [Fact]
    public void ALargerVocabularyOnALaterCall_GrowsTheStampsAndStillPenalizesOnce()
    {
        RepetitionPenaltyStep step = new(2.0f);
        float[] small = [4.0f, 4.0f];
        step.Apply(small, [1, 1]);
        float[] large = [4.0f, 4.0f, 4.0f, 4.0f, 4.0f];
        step.Apply(large, [4, 1, 4, 4]);
        Assert.Equal([4.0f, 2.0f], small);
        Assert.Equal([4.0f, 2.0f, 4.0f, 4.0f, 2.0f], large);
    }

    [Fact]
    public void OutOfRangeTokens_AreIgnored()
    {
        float[] logits = [3.0f];
        new RepetitionPenaltyStep(3.0f).Apply(logits, [-1, 7, 0]);
        Assert.Equal(1.0f, logits[0]);
    }
}

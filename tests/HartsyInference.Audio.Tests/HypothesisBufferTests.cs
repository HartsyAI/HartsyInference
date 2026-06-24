using System.Collections.Generic;
using System.Linq;
using HartsyInference.Audio.Streaming;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests the LocalAgreement-2 stabilizer: a word is confirmed only when two consecutive hypotheses agree
/// on it, output never rewrites earlier confirmations, and a growing hypothesis confirms its stable prefix.</summary>
public sealed class HypothesisBufferTests
{
    private static List<string> W(params string[] words) => words.ToList();

    [Fact]
    public void FirstHypothesis_ConfirmsNothing()
    {
        HypothesisBuffer buf = new();
        IReadOnlyList<string> confirmed = buf.Insert(W("the", "cat", "sat"));
        Assert.Empty(confirmed);
        Assert.Equal(W("the", "cat", "sat"), buf.PendingTail);
    }

    [Fact]
    public void AgreedPrefix_IsConfirmedOnSecondHypothesis()
    {
        HypothesisBuffer buf = new();
        buf.Insert(W("the", "cat", "sat"));
        IReadOnlyList<string> confirmed = buf.Insert(W("the", "cat", "sat", "on"));
        Assert.Equal(W("the", "cat", "sat"), confirmed);
        Assert.Equal(W("the", "cat", "sat"), buf.Committed);
        Assert.Equal(W("on"), buf.PendingTail);
    }

    [Fact]
    public void GrowingStream_ConfirmsIncrementally_WithoutRewrite()
    {
        HypothesisBuffer buf = new();
        buf.Insert(W("ask", "not"));
        Assert.Equal(W("ask", "not"), buf.Insert(W("ask", "not", "what")));
        Assert.Equal(W("what"), buf.Insert(W("ask", "not", "what", "your")));
        Assert.Equal(W("your"), buf.Insert(W("ask", "not", "what", "your", "country")));
        Assert.Equal(W("ask", "not", "what", "your"), buf.Committed);
        Assert.Equal(W("country"), buf.PendingTail);
    }

    [Fact]
    public void Disagreement_HoldsTheWord_UntilCorroborated()
    {
        HypothesisBuffer buf = new();
        buf.Insert(W("the", "quick"));
        // Second hypothesis revises the tail: only "the" agrees with the previous pending tail.
        IReadOnlyList<string> confirmed = buf.Insert(W("the", "brown"));
        Assert.Equal(W("the"), confirmed);
        Assert.Equal(W("brown"), buf.PendingTail);
        // Third agrees on "brown".
        Assert.Equal(W("brown"), buf.Insert(W("the", "brown", "fox")));
    }

    [Fact]
    public void Flush_ConfirmsRemainingTail()
    {
        HypothesisBuffer buf = new();
        buf.Insert(W("hello", "world"));
        IReadOnlyList<string> flushed = buf.Flush();
        Assert.Equal(W("hello", "world"), flushed);
        Assert.Equal(W("hello", "world"), buf.Committed);
        Assert.Empty(buf.PendingTail);
    }

    [Fact]
    public void Reset_ClearsState()
    {
        HypothesisBuffer buf = new();
        buf.Insert(W("a", "b"));
        buf.Insert(W("a", "b", "c"));
        buf.Reset();
        Assert.Empty(buf.Committed);
        Assert.Empty(buf.PendingTail);
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>A repeated prompt must reuse its encoder states exactly, return copies the caller may dispose, and keep only the
/// newest prompts. Without the cache every request re-runs the text encoder.</summary>
public sealed class PromptEncodingCacheTests
{
    private static MiniMaxH3TextEncoder.Result Encoded(float seed)
    {
        Tensor states = new Tensor(new TensorShape(3, 4), DType.F32);
        for (int i = 0; i < 12; i++)
        {
            states.AsSpan<float>()[i] = seed + i;
        }
        return new MiniMaxH3TextEncoder.Result { HiddenStates = states, TagRuns = [(0, 3, 7)] };
    }

    [Fact]
    public void RepeatedPromptReturnsTheEncodedStates()
    {
        PromptEncodingCache cache = new(4);
        Assert.Null(cache.Get("a red fox"));

        MiniMaxH3TextEncoder.Result fresh = Encoded(1f);
        cache.Put("a red fox", fresh);
        MiniMaxH3TextEncoder.Result? again = cache.Get("a red fox");

        Assert.NotNull(again);
        Assert.Equal(fresh.HiddenStates.AsSpan<float>().ToArray(), again!.HiddenStates.AsSpan<float>().ToArray());
        Assert.Equal(fresh.HiddenStates.Shape, again.HiddenStates.Shape);
        Assert.Equal(fresh.TagRuns, again.TagRuns);
    }

    [Fact]
    public void EveryHitIsAnIndependentCopy()
    {
        PromptEncodingCache cache = new(4);
        cache.Put("p", Encoded(2f));

        MiniMaxH3TextEncoder.Result first = cache.Get("p")!;
        first.HiddenStates.Dispose();
        MiniMaxH3TextEncoder.Result second = cache.Get("p")!;

        Assert.Equal(2f, second.HiddenStates.AsSpan<float>()[0]);
        second.HiddenStates.Dispose();
    }

    [Fact]
    public void EvictsTheOldestPromptBeyondCapacity()
    {
        PromptEncodingCache cache = new(2);
        cache.Put("a", Encoded(1f));
        cache.Put("b", Encoded(2f));
        cache.Put("c", Encoded(3f));

        Assert.Equal(2, cache.Count);
        Assert.Null(cache.Get("a"));
        Assert.NotNull(cache.Get("b"));
        Assert.NotNull(cache.Get("c"));
    }
}

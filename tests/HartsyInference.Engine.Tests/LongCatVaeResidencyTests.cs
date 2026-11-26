using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The VAE's weights are placed around each decode or encode and released after it, also when it throws (see
/// <see cref="LongCatRecipeSupport.WithWeightsPlaced{T}"/>). The device is not available to a tiny-config test, so the seam records the order of the
/// preload, the body and the free.</summary>
public sealed class LongCatVaeResidencyTests
{
    [Fact]
    public void TheWeightsArePlacedBeforeTheBodyAndFreedAfterIt()
    {
        using Tensor a = new(new TensorShape([2]), DType.F32);
        using Tensor b = new(new TensorShape([3]), DType.F32);
        List<string> log = [];
        List<Tensor> preloaded = [], freed = [];

        int result = LongCatRecipeSupport.WithWeightsPlaced(new[] { a, b },
            weights => { log.Add("preload"); preloaded.AddRange(weights); },
            weights => { log.Add("free"); freed.AddRange(weights); },
            () => { log.Add("body"); return 7; });

        Assert.Equal(7, result);
        Assert.Equal(["preload", "body", "free"], log);
        Assert.Equal([a, b], preloaded);
        Assert.Equal([a, b], freed);
    }

    [Fact]
    public void ThePoolIsTrimmedAfterTheWeightsAreFreed()
    {
        using Tensor a = new(new TensorShape([2]), DType.F32);
        List<string> log = [];

        LongCatRecipeSupport.WithWeightsPlaced(new[] { a },
            _ => log.Add("preload"),
            _ => log.Add("free"),
            () => { log.Add("body"); return 0; },
            trim: () => log.Add("trim"));

        Assert.Equal(["preload", "body", "free", "trim"], log);
    }

    [Fact]
    public void TheWeightsAreFreedWhenTheBodyThrows()
    {
        using Tensor a = new(new TensorShape([2]), DType.F32);
        bool freed = false;

        Assert.Throws<InvalidOperationException>(() => LongCatRecipeSupport.WithWeightsPlaced<int>([a],
            _ => { },
            _ => freed = true,
            () => throw new InvalidOperationException("decode failed")));

        Assert.True(freed);
    }

    [Fact]
    public void AnEmptyWeightSetStillRunsTheBody()
    {
        int calls = 0;
        string value = LongCatRecipeSupport.WithWeightsPlaced(Array.Empty<Tensor>(),
            _ => calls++,
            _ => calls++,
            () => "ok");

        Assert.Equal("ok", value);
        Assert.Equal(2, calls);
    }
}

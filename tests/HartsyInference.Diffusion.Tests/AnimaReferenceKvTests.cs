using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;
using Xunit;

using static HartsyInference.Tests.Common.AnimaSyntheticWeights;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The reference K/V path: the rewrite that joins a cached reference to a block's live tokens, the placement of a reference on
/// the RoPE axes, the prepass that caches a reference's keys and values, and a tiny Anima transformer that joins a reference into every
/// block. These run on the CPU; the parity against ComfyUI's reference adapter is a GPU check.</summary>
public sealed class AnimaReferenceKvTests
{
    private const int TargetH = 6, TargetW = 10, RefH = 4, RefW = 6, Text = 7, TextWidth = AnimaSyntheticWeights.TextWidth;

    [Fact]
    public void Rewrite_Append_AndPrepend_KeepTheStatedOrder()
    {
        using CpuBackend cpu = new();
        using Tensor cachedK = Values(new TensorShape(1, 1, 1, 2), 101f);
        using Tensor cachedV = Values(new TensorShape(1, 1, 1, 2), 201f);

        (Tensor appendK, Tensor appendV) = new ReferenceKvRewrite(cachedK, cachedV, ReferenceKvOrder.Append)
            .Rewrite(cpu, Values(new TensorShape(1, 2, 1, 2), 1f), Values(new TensorShape(1, 2, 1, 2), 11f));
        Assert.Equal(new TensorShape(1, 3, 1, 2), appendK.Shape);
        Assert.Equal(new float[] { 1, 2, 3, 4, 101, 102 }, appendK.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(new float[] { 11, 12, 13, 14, 201, 202 }, appendV.AsReadOnlySpan<float>().ToArray());
        appendK.Dispose();
        appendV.Dispose();

        (Tensor prependK, Tensor prependV) = new ReferenceKvRewrite(cachedK, cachedV, ReferenceKvOrder.Prepend)
            .Rewrite(cpu, Values(new TensorShape(1, 2, 1, 2), 1f), Values(new TensorShape(1, 2, 1, 2), 11f));
        Assert.Equal(new float[] { 101, 102, 1, 2, 3, 4 }, prependK.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(new float[] { 201, 202, 11, 12, 13, 14 }, prependV.AsReadOnlySpan<float>().ToArray());
        prependK.Dispose();
        prependV.Dispose();
    }

    [Theory]
    [InlineData(0.5f, 3, 2)]
    [InlineData(0.5f, 5, 2)]
    [InlineData(0.5f, 7, 4)]
    [InlineData(1.0f, 4, 4)]
    [InlineData(0f, 9, 0)]
    public void Placement_RowOffset_RoundsHalfToEven(float heights, int targetHeight, int expected)
    {
        Assert.Equal(expected, new ReferencePlacement(1, heights, 0).RowOffset(targetHeight));
    }

    [Fact]
    public void Placement_RefusesANegativeFrameOrOffset()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferencePlacement(-1, 1f, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferencePlacement(1, -0.5f, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferencePlacement(1, 1f, -1));
    }

    [Fact]
    public void Prepass_SkipsTheListedLayers_AndCachesTheRest()
    {
        using CpuBackend cpu = new();
        using Tensor embedded = Ramp(new TensorShape(1, 3, 4), 0f);
        int calls = 0;
        ReferenceKvCache cache = ReferenceKvPrepass.Build(cpu, 3, [1], embedded,
            (int layer, Tensor stream, out Tensor keys, out Tensor values) =>
            {
                calls++;
                keys = Values(new TensorShape(1, 3, 1, 2), layer + 1f);
                values = Values(new TensorShape(1, 3, 1, 2), 10f + layer);
                return Ramp(stream.Shape, 0f);
            },
            fingerprint: 9, family: "test");
        using ReferenceKvCache owned = cache;

        Assert.Equal(2, calls);
        Assert.Equal(3, owned.Layers);
        Assert.Equal(3, owned.Tokens);
        Assert.Equal(9, owned.Fingerprint);
        Assert.Null(owned.Keys(1));
        Assert.Equal(3f, owned.Keys(2)!.AsReadOnlySpan<float>()[0]);
        Assert.Equal(10f, owned.Values(0)!.AsReadOnlySpan<float>()[0]);
    }

    [Fact]
    public void Fingerprint_TracksEachInputOfTheCache()
    {
        ReferencePlacement placement = new(1, 1.0f, 0);
        long baseline = AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, placement);

        Assert.Equal(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(43, 2, 3, 3, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 4, 3, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 5, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, new ReferencePlacement(2, 1.0f, 0)));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, new ReferencePlacement(1, 1.0f, 1)));
    }

    [Fact]
    public void Transformer_ReferenceJoin_ChangesTheVelocity_AndTheCacheIsReusable()
    {
        using CpuBackend cpu = new();
        AnimaConfig config = TinyConfig();
        using AnimaTransformer transformer = new(config);
        transformer.LoadWeights(Weights(config));
        using Tensor latent = Ramp(new TensorShape(1, 2, TargetH, TargetW), 0.3f);
        using Tensor text = Ramp(new TensorShape(1, Text, TextWidth), 0.7f);
        using Tensor reference = Ramp(new TensorShape(1, 2, RefH, RefW), 1.1f);

        using Tensor baseline = transformer.Forward(cpu, latent, 0.625f, text);
        using ReferenceKvCache cache = transformer.BuildReferenceCache(cpu, reference, TargetH / 2,
            new ReferencePlacement(1, 1.0f, 0), identity: 42);
        using Tensor first = transformer.Forward(cpu, latent, 0.625f, text, null, cache);
        using Tensor second = transformer.Forward(cpu, latent, 0.625f, text, null, cache);

        Assert.Equal(baseline.Shape, first.Shape);
        Assert.Equal(first.AsReadOnlySpan<float>().ToArray(), second.AsReadOnlySpan<float>().ToArray());
        Assert.True(MaxAbsDifference(baseline, first) > 1e-4f, "The reference left the velocity unchanged.");
    }

    [Fact]
    public void Transformer_ReferenceForABatch_IsRefused()
    {
        using CpuBackend cpu = new();
        AnimaConfig config = TinyConfig();
        using AnimaTransformer transformer = new(config);
        transformer.LoadWeights(Weights(config));
        using Tensor reference = Ramp(new TensorShape(1, 2, RefH, RefW), 1.1f);
        using ReferenceKvCache cache = transformer.BuildReferenceCache(cpu, reference, TargetH / 2,
            new ReferencePlacement(1, 1.0f, 0), identity: 1);
        using Tensor latent = Ramp(new TensorShape(2, 2, TargetH, TargetW), 0.3f);
        using Tensor text = Ramp(new TensorShape(2, Text, TextWidth), 0.7f);

        Assert.Throws<NotSupportedException>(() => transformer.Forward(cpu, latent, 0.625f, text, null, cache));
    }

    [Fact]
    public void Transformer_ReferenceBuiltForAnotherTargetGrid_IsRefused()
    {
        // The reference's RoPE rows depend on the target grid, so a cache built for 3 rows must not join a 2-row target.
        using CpuBackend cpu = new();
        AnimaConfig config = TinyConfig();
        using AnimaTransformer transformer = new(config);
        transformer.LoadWeights(Weights(config));
        using Tensor reference = Ramp(new TensorShape(1, 2, RefH, RefW), 1.1f);
        using ReferenceKvCache cache = transformer.BuildReferenceCache(cpu, reference, TargetH / 2,
            new ReferencePlacement(1, 1.0f, 0), identity: 1);
        using Tensor latent = Ramp(new TensorShape(1, 2, TargetH - 2, TargetW), 0.3f);
        using Tensor text = Ramp(new TensorShape(1, Text, TextWidth), 0.7f);

        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, latent, 0.625f, text, null, cache));
    }

    private static float MaxAbsDifference(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        float max = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            max = MathF.Max(max, MathF.Abs(x[i] - y[i]));
        }
        return max;
    }


    private static Tensor Values(TensorShape shape, float start)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = start + i;
        }
        return t;
    }
}

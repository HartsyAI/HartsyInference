using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The engine hands the ACE-Step v1 DiT its UMT5 features as <c>[1, T, 768]</c>; counting rows from the
/// leading dim kept one tag token and let the projection write T rows into a one-row buffer.</summary>
public sealed class AceStepDitContextTests
{
    [Fact]
    public void BuildContext_BatchOfOneTextFeatures_MatchesRankTwo()
    {
        AceStepConfig config = AceStepSyntheticWeights.TinyConfig;
        using AceStepDit dit = new AceStepDit(config);
        dit.LoadWeights(AceStepSyntheticWeights.BuildDit(config));
        using CpuBackend backend = new CpuBackend();
        const int tokens = 3;
        using Tensor rows = new Tensor(new TensorShape(tokens, config.TextDim), DType.F32);
        using Tensor batched = new Tensor(new TensorShape(1, tokens, config.TextDim), DType.F32);
        Span<float> values = rows.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.37f);
        }
        values.CopyTo(batched.AsSpan<float>());

        using Tensor expected = dit.BuildContext(backend, rows, [1, 2], null);
        using Tensor actual = dit.BuildContext(backend, batched, [1, 2], null);

        Assert.Equal(new TensorShape(1 + tokens + 2, config.InnerDim), actual.Shape);
        Assert.Equal(expected.AsSpan<float>().ToArray(), actual.AsSpan<float>().ToArray());
    }

    [Fact]
    public void BuildContext_RefusesAnythingButRowsOrABatchOfOne()
    {
        AceStepConfig config = AceStepSyntheticWeights.TinyConfig;
        using AceStepDit dit = new AceStepDit(config);
        dit.LoadWeights(AceStepSyntheticWeights.BuildDit(config));
        using CpuBackend backend = new CpuBackend();
        using Tensor batchOfTwo = new Tensor(new TensorShape(2, 3, config.TextDim), DType.F32);
        using Tensor rankFour = new Tensor(new TensorShape([1L, 1, 3, config.TextDim]), DType.F32);

        Assert.Throws<ArgumentException>(() => dit.BuildContext(backend, batchOfTwo, [], null));
        Assert.Throws<ArgumentException>(() => dit.BuildContext(backend, rankFour, [], null));
    }
}

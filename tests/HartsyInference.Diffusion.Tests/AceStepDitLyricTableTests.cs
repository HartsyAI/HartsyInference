using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The ACE-Step v1 lyric embedding table is looked up on the host; the Comfy all-in-one checkpoint ships it as
/// bf16, which a <c>float*</c> lookup misreads.</summary>
public sealed class AceStepDitLyricTableTests
{
    [Fact]
    public void BuildContext_Bf16LyricTable_MatchesTheSameTableInF32()
    {
        AceStepConfig config = AceStepSyntheticWeights.TinyConfig;
        Dictionary<string, Tensor> weights = AceStepSyntheticWeights.BuildDit(config);
        using Tensor bf16 = weights["lyric_embs.weight"].CastTo(DType.BF16);
        using Tensor f32 = bf16.CastTo(DType.F32);
        weights["lyric_embs.weight"] = f32;
        using AceStepDit reference = new AceStepDit(config);
        reference.LoadWeights(weights);
        weights["lyric_embs.weight"] = bf16;
        using AceStepDit dit = new AceStepDit(config);
        dit.LoadWeights(weights);
        using CpuBackend backend = new CpuBackend();
        using Tensor text = new Tensor(new TensorShape(2, config.TextDim), DType.F32);

        using Tensor expected = reference.BuildContext(backend, text, [1, 2, 3], null);
        using Tensor actual = dit.BuildContext(backend, text, [1, 2, 3], null);

        Assert.Equal(expected.AsSpan<float>().ToArray(), actual.AsSpan<float>().ToArray());
    }
}

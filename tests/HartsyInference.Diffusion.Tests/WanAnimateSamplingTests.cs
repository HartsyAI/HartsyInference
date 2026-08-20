using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the two Wan-Animate sampling decisions that fail silently: whether the denoise runs the negative
/// branch at all (upstream <c>wan/animate.py</c> folds CFG only above guidance 1.0), and that skipping it at or below
/// 1.0 changes nothing numerically — a wrong gate here still renders a video, just at twice the cost or with the
/// guidance quietly disabled.</summary>
public sealed class WanAnimateSamplingTests
{
    [Theory]
    [InlineData(0f, false)]
    [InlineData(0.5f, false)]
    [InlineData(1f, false)]         // upstream's `guide_scale > 1` and the recipe's own default
    [InlineData(1.0001f, true)]
    [InlineData(5f, true)]
    public void CfgBranchRunsOnlyAboveGuidanceOne(float guidance, bool expected)
    {
        Assert.Equal(expected, WanAnimatePipeline.UsesCfgBranch(guidance));
    }

    [Theory]
    [InlineData(0f)]        // plain CFG
    [InlineData(0.7f)]      // the fp8 renorm strength WanConfigDetector sets
    public void FoldingCfgAtGuidanceOneLeavesTheConditionalPrediction(float rescale)
    {
        const int n = 4096;
        Tensor cond = new Tensor(new TensorShape([1L, n]), DType.F32);
        Tensor uncond = new Tensor(new TensorShape([1L, n]), DType.F32);
        Span<float> c = cond.AsSpan<float>();
        Span<float> u = uncond.AsSpan<float>();
        Random rng = new Random(1234);
        float[] expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            c[i] = expected[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            u[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
        LancePipelineCommon.CfgCombineRenormInPlace(cond, uncond, 1f, rescale);
        // Not bit-identical: the fold recomputes c as u + 1·(c − u), so it lands within float rounding of c.
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(expected[i] - c[i]) <= 1e-6f, $"element {i}: {expected[i]} vs {c[i]}");
        }
        cond.Dispose();
        uncond.Dispose();
    }
}

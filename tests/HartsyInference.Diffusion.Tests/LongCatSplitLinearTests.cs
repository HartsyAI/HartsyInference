using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Per-site checks of the BF16 route's projections (CPU): the LoRA branch rounded apart from the base product, the two-step head norm, and the
/// output of each piece on the BF16 grid. Each expected value is the upstream formula on the host, rounded where upstream rounds.</summary>
public sealed class LongCatSplitLinearTests
{
    private const int Rows = 5, In = 8, Out = 4, Rank = 2;

    [Fact]
    public void TheSplitLinearIsBaseAndBranchEachRoundedThenSummed()
    {
        using CpuBackend backend = new();
        float[] w = Values(Out * In, 1), b = Values(Out, 2), x = Values(Rows * In, 3);
        float[] down = Values(Rank * In, 4), up = Values(Out * Rank, 5);
        const float scale = 0.5f;

        using Tensor weight = Make(new long[] { Out, In }, w);
        using Tensor bias = Make(new long[] { Out }, b);
        using Tensor input = Make(new long[] { Rows, In }, x);
        using Tensor downW = Make(new long[] { Rank, In }, down);
        using Tensor upW = Make(new long[] { Out, Rank }, up);
        LowRankAdjunctTerm term = new() { Down = downW, Up = upW, Scale = scale, Gate = null };
        Tensor adjunct = weight.WithLowRankAdjunct(new LowRankAdjunct { Terms = [term] });

        using Tensor actual = LongCatSplitLinear.Linear(backend, input, adjunct, bias, bf16: true);
        float[] got = actual.AsReadOnlySpan<float>().ToArray();

        float[] expected = new float[Rows * Out];
        for (int r = 0; r < Rows; r++)
        {
            float[] dRow = new float[Rank];
            for (int j = 0; j < Rank; j++)
                dRow[j] = Bf16(Dot(x, r * In, down, j * In, In));
            for (int o = 0; o < Out; o++)
            {
                float baseValue = Bf16(Dot(x, r * In, w, o * In, In) + b[o]);
                float branchSum = 0;
                float u = 0;
                for (int j = 0; j < Rank; j++)
                    u += dRow[j] * up[o * Rank + j];
                branchSum = Bf16(Bf16(Bf16(u) * scale));
                expected[r * Out + o] = Bf16(baseValue + Bf16(branchSum));
            }
        }

        Assert.All(got, v => Assert.Equal(v, Bf16(v)));
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-2f * Math.Max(1f, Math.Abs(expected[i])), $"element {i}: {got[i]} vs {expected[i]}");
    }

    [Fact]
    public void TheHeadNormRoundsTheUnitVectorThenTheWeightMultiply()
    {
        const int heads = 2, headDim = 4, rows = 3;
        const float eps = 1e-6f;
        float[] x = Values(rows * heads * headDim, 7), w = Values(headDim, 8);
        using Tensor input = Make(new long[] { rows, heads * headDim }, x);
        using Tensor weight = Make(new long[] { headDim }, w);

        using Tensor normed = LongCatSplitLinear.HeadNorm(input, weight, heads, eps);
        float[] got = normed.AsReadOnlySpan<float>().ToArray();

        Assert.All(got, v => Assert.Equal(v, Bf16(v)));
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < heads; h++)
            {
                int at = r * heads * headDim + h * headDim;
                float sum = 0f;
                for (int d = 0; d < headDim; d++) sum += x[at + d] * x[at + d];
                float inv = 1f / MathF.Sqrt(sum / headDim + eps);
                for (int d = 0; d < headDim; d++)
                {
                    float expected = Bf16(Bf16(x[at + d] * inv) * w[d]);
                    Assert.Equal(expected, got[at + d]);
                }
            }
    }

    [Fact]
    public void WithoutAnActiveAdjunctTheSplitLinearIsTheSharedLinearRounded()
    {
        using CpuBackend backend = new();
        float[] w = Values(Out * In, 11), x = Values(Rows * In, 12);
        using Tensor weight = Make(new long[] { Out, In }, w);
        using Tensor input = Make(new long[] { Rows, In }, x);
        using Tensor plain = new(new TensorShape(Rows, Out), DType.F32);
        backend.Linear(plain, input, weight, null);
        float[] shared = plain.AsReadOnlySpan<float>().ToArray();

        using Tensor split = LongCatSplitLinear.Linear(backend, input, weight, null, bf16: true);
        float[] got = split.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < got.Length; i++)
            Assert.Equal(Bf16(shared[i]), got[i]);
    }

    [Fact]
    public void TheBaseViewKeepsTheQuantizationMetadataAndDropsTheAdjunct()
    {
        using Tensor weight = Make(new long[] { Out, In }, Values(Out * In, 21));
        weight.Fp8ScaleFactor = 2f;
        LowRankAdjunctTerm term = new() { Down = Make(new long[] { Rank, In }, Values(Rank * In, 22)), Up = Make(new long[] { Out, Rank }, Values(Out * Rank, 23)), Scale = 1f, Gate = null };
        Tensor adjunct = weight.WithLowRankAdjunct(new LowRankAdjunct { Terms = [term] });

        using Tensor baseWhole = adjunct.WithoutLowRankAdjunct();
        Assert.Null(baseWhole.LowRankAdjunct);
        Assert.Equal(2f, baseWhole.Fp8ScaleFactor);
        Assert.Equal(Out, baseWhole.Shape[0]);

        using Tensor baseWindow = adjunct.WithoutLowRankAdjunct(1, 2);
        Assert.Null(baseWindow.LowRankAdjunct);
        Assert.Equal(2f, baseWindow.Fp8ScaleFactor);
        Assert.Equal(2, baseWindow.Shape[0]);
    }

    private static float Bf16(float value) => BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(value));

    private static float Dot(float[] a, int aAt, float[] b, int bAt, int length)
    {
        double sum = 0;
        for (int i = 0; i < length; i++)
            sum += (double)a[aAt + i] * b[bAt + i];
        return (float)sum;
    }

    private static float[] Values(int count, int seed)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return values;
    }

    private static Tensor Make(long[] shape, float[] values)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }
}

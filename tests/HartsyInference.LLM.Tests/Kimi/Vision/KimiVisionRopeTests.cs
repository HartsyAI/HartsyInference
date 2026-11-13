using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi.Vision;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi.Vision;

/// <summary>The MoonViT-V2 2D rotary angles and their interleaved application, checked against values worked out by hand.</summary>
public sealed class KimiVisionRopeTests
{
    private const float Tol = 1e-6f;

    [Fact]
    public void Build_EvenPairsTurnWithColumnAndOddPairsWithRow()
    {
        // Merge 2 over a 2x2 grid is one block, so token k is row k/2, column k%2 in row-major order.
        (float[] cos, float[] sin) = KimiVisionRope.Build(2, 2, 8, 2, 10000f);

        // Token 2 is row 1, column 0. Pair 1 rotates by the row at frequency 1; pair 3 by the row at frequency 0.01.
        Assert.Equal(MathF.Cos(0f), cos[2 * 4 + 0], Tol);
        Assert.Equal(MathF.Cos(1f), cos[2 * 4 + 1], Tol);
        Assert.Equal(MathF.Sin(1f), sin[2 * 4 + 1], Tol);
        Assert.Equal(MathF.Cos(0.01f), cos[2 * 4 + 3], Tol);
        Assert.Equal(MathF.Sin(0.01f), sin[2 * 4 + 3], Tol);
    }

    [Fact]
    public void Build_TokenOneTakesItsColumnOnEvenPairs()
    {
        (float[] cos, float[] sin) = KimiVisionRope.Build(2, 2, 8, 2, 10000f);

        // Token 1 is row 0, column 1: pair 0 turns by the column (1), pair 1 by the row (0).
        Assert.Equal(MathF.Cos(1f), cos[1 * 4 + 0], Tol);
        Assert.Equal(MathF.Sin(1f), sin[1 * 4 + 0], Tol);
        Assert.Equal(1f, cos[1 * 4 + 1], Tol);
        Assert.Equal(0f, sin[1 * 4 + 1], Tol);
    }

    [Fact]
    public void Rotary_RotatesAdjacentElementsWithTheTable()
    {
        float[] x = [1f, 2f, 3f, 4f, 0.5f, -1f, 2f, 0.25f];
        float[] cos = [MathF.Cos(0.3f), MathF.Cos(0.7f), MathF.Cos(-0.4f), MathF.Cos(1.1f)];
        float[] sin = [MathF.Sin(0.3f), MathF.Sin(0.7f), MathF.Sin(-0.4f), MathF.Sin(1.1f)];
        using Tensor xt = Make(x, 1, 2, 1, 4);
        using Tensor ct = Make(cos, 2, 2);
        using Tensor st = Make(sin, 2, 2);

        using IBackend backend = new CpuBackend();
        backend.Rotary(RotaryCall.TokenMajor(xt, ct, st, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix));

        float[] actual = xt.AsReadOnlySpan<float>().ToArray();
        for (int token = 0; token < 2; token++)
            for (int p = 0; p < 2; p++)
            {
                int e = token * 4 + 2 * p;
                float c = cos[token * 2 + p], s = sin[token * 2 + p];
                float x0 = x[e], x1 = x[e + 1];
                Assert.Equal(x0 * c - x1 * s, actual[e], 1e-5f);
                Assert.Equal(x1 * c + x0 * s, actual[e + 1], 1e-5f);
            }
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}

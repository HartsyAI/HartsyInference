using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Transformer;

public sealed class HostMathLinearTests
{
    // Widths below one vector step, across the four-accumulator loop with a scalar tail, and ERNIE-5.5's flattened stream (4 x 4096).
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(33)]
    [InlineData(259)]
    [InlineData(16384)]
    public void Linear_MatchesADoublePrecisionProduct(int inDim)
    {
        const int rows = 3, outDim = 5;
        (float[] x, float[] w) = Operands(rows, inDim, outDim, seed: inDim);

        float[] got = HostMath.Linear(x, w, rows, inDim, outDim);

        for (int t = 0; t < rows; t++)
        {
            for (int o = 0; o < outDim; o++)
            {
                double want = 0, magnitude = 0;
                for (int i = 0; i < inDim; i++)
                {
                    double term = (double)x[t * inDim + i] * w[o * inDim + i];
                    want += term;
                    magnitude += Math.Abs(term);
                }
                // An F32 sum of inDim terms, each rounded once: a loose bound on any summation order.
                Assert.InRange(got[t * outDim + o], want - 1e-6 * magnitude - 1e-30, want + 1e-6 * magnitude + 1e-30);
            }
        }
    }

    // A weight streamed through row windows (DeepSeekV41Weight) must give the whole-matrix product's exact bits.
    [Fact]
    public void LinearInto_OverRowWindows_EqualsTheWholeProductBitForBit()
    {
        const int rows = 4, inDim = 1031, outDim = 70;
        (float[] x, float[] w) = Operands(rows, inDim, outDim, seed: 11);
        float[] whole = HostMath.Linear(x, w, rows, inDim, outDim);

        float[] windowed = new float[rows * outDim];
        int[] windows = [1, 33, 36];
        int first = 0;
        foreach (int count in windows)
        {
            HostMath.LinearInto(x, w.AsSpan(first * inDim, count * inDim), rows, inDim, count, windowed, outDim, first);
            first += count;
        }

        Assert.Equal(outDim, first);
        Assert.Equal(whole, windowed);
    }

    private static (float[] X, float[] W) Operands(int rows, int inDim, int outDim, int seed)
    {
        Random rng = new(seed);
        float[] x = [.. Enumerable.Range(0, rows * inDim).Select(_ => (float)(rng.NextDouble() * 2 - 1))];
        float[] w = [.. Enumerable.Range(0, outDim * inDim).Select(_ => (float)(rng.NextDouble() * 2 - 1))];
        return (x, w);
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Music;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The DCAE's Sana linear attention cuts its heads from the channel-concatenated <c>[q|k|v]</c> stream in blocks of
/// 3·d, as diffusers' <c>SanaMultiscaleLinearAttention</c> reshapes it; taking the h-th d columns of each projection
/// instead decodes a different mel from the same latent.</summary>
public sealed class AceStepEfficientVitHeadGroupingTests
{
    [Theory]
    [InlineData(0)]   // base branch: the first half of the attention output
    [InlineData(1)]   // multiscale branch: the second half
    public void LinearAttentionHeads_CutHeadsFromTheConcatenatedQkvStream(int branch)
    {
        const int Channels = 8, HeadDim = 4, Tokens = 5, Width = 2 * Channels;
        using Tensor q = Values(Tokens, Channels, 1);
        using Tensor k = Values(Tokens, Channels, 2);
        using Tensor v = Values(Tokens, Channels, 3);
        using Tensor actual = new Tensor(new TensorShape(Tokens, Width), DType.F32);

        EfficientVitBlock.LinearAttentionHeads(q, k, v, actual, Channels, Tokens, headOffset: branch * Channels, HeadDim);

        float[] expected = Reference(q.AsSpan<float>().ToArray(), k.AsSpan<float>().ToArray(), v.AsSpan<float>().ToArray(),
            Channels, Tokens, HeadDim);
        float[] got = actual.AsSpan<float>().ToArray();
        for (int i = 0; i < Tokens; i++)
        {
            for (int c = 0; c < Width; c++)
            {
                int column = c - branch * Channels;
                float want = column is >= 0 and < Channels ? expected[i * Channels + column] : 0f;
                Assert.Equal(want, got[i * Width + c], 5);
            }
        }
    }

    /// <summary>The reference: reshape <c>[q|k|v]</c> to heads of 3·d, split each into query, key and value, ReLU the
    /// first two, and divide by the attention mass (the padded ones row).</summary>
    private static float[] Reference(float[] q, float[] k, float[] v, int channels, int tokens, int headDim)
    {
        float Stream(int token, int column) =>
            ((column / channels) switch { 0 => q, 1 => k, _ => v })[token * channels + column % channels];
        float[] output = new float[tokens * channels];
        for (int h = 0; h < channels / headDim; h++)
        {
            int start = h * 3 * headDim;
            for (int i = 0; i < tokens; i++)
            {
                double mass = 0;
                double[] acc = new double[headDim];
                for (int j = 0; j < tokens; j++)
                {
                    double score = 0;
                    for (int t = 0; t < headDim; t++)
                    {
                        score += Math.Max(Stream(j, start + headDim + t), 0f) * Math.Max(Stream(i, start + t), 0f);
                    }
                    for (int t = 0; t < headDim; t++)
                    {
                        acc[t] += score * Stream(j, start + 2 * headDim + t);
                    }
                    mass += score;
                }
                for (int t = 0; t < headDim; t++)
                {
                    output[i * channels + h * headDim + t] = (float)(acc[t] / (mass + 1e-15));
                }
            }
        }
        return output;
    }

    private static Tensor Values(int rows, int columns, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, columns), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = 0.1f + 0.6f * MathF.Sin(seed * 7.31f + i * 1.17f);   // about 40% negative, so the ReLU bites
        }
        return t;
    }
}

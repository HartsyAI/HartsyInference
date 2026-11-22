using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Vision.Detection.Blocks;
using Xunit;

namespace HartsyInference.Vision.Tests.Detection;

/// <summary>The PSA block's attention through the shared attention layer, against the host loop the block ran before:
/// per head, scores <c>scale · q · k</c> over <c>keyDim</c> channels and a mix of values that are <c>headDim</c> wide.
/// The loop is kept here, verbatim in its arithmetic, as the oracle. The layer sees Q and K zero-padded to the value's
/// width, which must not change a score.
/// <para>The tolerance was fixed before the first run: 1e-4 relative to the larger of 1 and the value.</para></summary>
public sealed class PsaAttentionTests
{
    private const float Tolerance = 1e-4f;

    /// <summary>The former loop over channel-major planes: for each batch item and head its query, key and value planes.</summary>
    private static float[] FormerLoop(float[] qkv, int batch, int heads, int keyDim, int headDim, int n, float scale)
    {
        int slots = 2 * keyDim + headDim;
        float[] output = new float[batch * heads * headDim * n], attn = new float[n * n];
        for (int h = 0; h < batch * heads; h++)
        {
            int qPlane = h * slots * n, kPlane = qPlane + keyDim * n, vPlane = kPlane + keyDim * n;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int d = 0; d < keyDim; d++) sum += qkv[qPlane + d * n + i] * qkv[kPlane + d * n + j];
                    attn[i * n + j] = sum * scale;
                }
                float max = float.NegativeInfinity;
                for (int j = 0; j < n; j++)
                    if (attn[i * n + j] > max) max = attn[i * n + j];
                float sumExp = 0f;
                for (int j = 0; j < n; j++)
                {
                    float e = MathF.Exp(attn[i * n + j] - max);
                    attn[i * n + j] = e;
                    sumExp += e;
                }
                float inv = sumExp > 0f ? 1f / sumExp : 0f;
                for (int j = 0; j < n; j++) attn[i * n + j] *= inv;
            }
            for (int d = 0; d < headDim; d++)
                for (int i = 0; i < n; i++)
                {
                    float sum = 0f;
                    for (int j = 0; j < n; j++) sum += qkv[vPlane + d * n + j] * attn[i * n + j];
                    output[(h * headDim + d) * n + i] = sum;
                }
        }
        return output;
    }

    /// <summary>The detector's geometry (heads of 64 with a key width of 32, over a 20 by 20 grid), a wider model, a
    /// key ratio that is not a half, a small odd shape, and two batch items in one call.</summary>
    [Theory]
    [InlineData(1, 4, 32, 64, 400)]
    [InlineData(1, 8, 32, 64, 400)]
    [InlineData(1, 2, 16, 64, 49)]
    [InlineData(1, 3, 4, 8, 15)]
    [InlineData(2, 3, 4, 8, 15)]
    public void OnTheCpuBackendItMatchesTheFormerLoop(int batch, int heads, int keyDim, int headDim, int n)
    {
        Random random = new(batch * 7919 + heads * 1000 + n);
        float[] qkv = new float[batch * heads * (2 * keyDim + headDim) * n];
        for (int i = 0; i < qkv.Length; i++) qkv[i] = random.NextSingle() * 2f - 1f;
        float scale = 1f / MathF.Sqrt(keyDim);
        float[] expected = FormerLoop(qkv, batch, heads, keyDim, headDim, n, scale);

        using CpuBackend backend = new();
        float[] actual = new float[expected.Length];
        PsaAttention.Attend(backend, qkv, actual, batch, heads, keyDim, headDim, n, scale);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance * Math.Max(1f, Math.Abs(expected[i])),
                $"batch {batch}, {heads} heads, key {keyDim}, value {headDim}, {n} tokens: element {i} is {actual[i]}, expected {expected[i]}");
    }
}

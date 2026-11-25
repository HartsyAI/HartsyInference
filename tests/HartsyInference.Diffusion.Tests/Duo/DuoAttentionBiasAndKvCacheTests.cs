using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The additive bias from the visibility rectangles, and the KV cache under append and sliding-window eviction, checked against a
/// scalar reference that attends over the full key sequence. These are self-consistency checks, not upstream parity: the upstream DiT
/// attention is not yet part of the L0 oracle.</summary>
public sealed class DuoAttentionBiasAndKvCacheTests
{
    private const int Heads = 2;
    private const int HeadDim = 4;
    private const int Width = Heads * HeadDim;

    [Fact]
    public void BiasIsZeroInsideRectanglesAndNegativeInfinityOutside()
    {
        float[] bias = DuoAttentionBias.Build([new DuoMaskRect(0, 2, 0, 1), new DuoMaskRect(0, 2, 3, 4), new DuoMaskRect(2, 3, 0, 4)], 3, 4);
        float[] expected =
        [
            0f, float.NegativeInfinity, float.NegativeInfinity, 0f,
            0f, float.NegativeInfinity, float.NegativeInfinity, 0f,
            0f, 0f, 0f, 0f,
        ];
        Assert.Equal(expected, bias);
    }

    [Fact]
    public void BiasMatchesDenseMaskFromLayoutGroups()
    {
        DuoGroup[] groups =
        [
            new(0, -1, DuoGroupKind.Shared, DuoModality.None, 0, 2),
            new(0, 0, DuoGroupKind.Text, DuoModality.None, 2, 3),
            new(0, 0, DuoGroupKind.Noisy, DuoModality.Video, 3, 5),
            new(0, 1, DuoGroupKind.Noisy, DuoModality.Video, 5, 7),
        ];
        IReadOnlyList<DuoMaskRect> rects = DuoMask.Build(groups, groups, DuoAttentionMode.Causal);
        bool[,] dense = DuoMask.Dense(rects, 7, 7);
        float[] bias = DuoAttentionBias.Build(rects, 7, 7);
        for (int q = 0; q < 7; q++)
        {
            for (int k = 0; k < 7; k++)
            {
                Assert.Equal(dense[q, k], bias[q * 7 + k] == 0f);
            }
        }
    }

    [Fact]
    public void RectangleOutsideMatrixIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DuoAttentionBias.Build([new DuoMaskRect(0, 3, 0, 1)], 2, 2));
    }

    [Fact]
    public void QueryRowWithNoKeyIsRefused()
    {
        Assert.Throws<InvalidOperationException>(() => DuoAttentionBias.Build([new DuoMaskRect(0, 1, 0, 1)], 2, 2));
    }

    [Fact]
    public void CachedAttentionEqualsFullAttentionOverTheSameKeys()
    {
        const int total = 6, queries = 2, past = total - queries;
        float[] keys = Sequence(total, 1), values = Sequence(total, 2), q = Sequence(queries, 3);
        float[] bias = CausalBias(past, queries);

        float[] full = Attend(q, queries, keys, values, total, bias);

        DuoKvCache cache = new DuoKvCache(total, Width);
        cache.Append(keys.AsSpan(0, past * Width), values.AsSpan(0, past * Width));
        cache.Append(keys.AsSpan(past * Width), values.AsSpan(past * Width));
        float[] cached = Attend(q, queries, cache.Keys.ToArray(), cache.Values.ToArray(), cache.Rows, bias);

        AssertClose(full, cached);
    }

    [Fact]
    public void SlidingWindowAfterEvictionEqualsAttentionOverTheWindow()
    {
        const int window = 4, chunk = 2;
        float[] keys = Sequence(6, 4), values = Sequence(6, 5);
        DuoKvCache cache = new DuoKvCache(window, Width);
        cache.Append(keys.AsSpan(0, 4 * Width), values.AsSpan(0, 4 * Width));

        // Commit the next two rows: drop the two oldest, then append.
        cache.EvictOldest(chunk);
        cache.Append(keys.AsSpan(4 * Width, chunk * Width), values.AsSpan(4 * Width, chunk * Width));
        Assert.Equal(window, cache.Rows);

        float[] q = Sequence(1, 6);
        float[] windowKeys = keys.AsSpan(2 * Width, window * Width).ToArray();
        float[] windowValues = values.AsSpan(2 * Width, window * Width).ToArray();
        AssertClose(Attend(q, 1, windowKeys, windowValues, window, new float[window]),
            Attend(q, 1, cache.Keys.ToArray(), cache.Values.ToArray(), cache.Rows, new float[window]));
        Assert.True(cache.Keys.SequenceEqual(windowKeys));
        Assert.True(cache.Values.SequenceEqual(windowValues));
    }

    [Fact]
    public void AppendBeyondCapacityIsRefused()
    {
        DuoKvCache cache = new DuoKvCache(2, Width);
        float[] row = new float[Width];
        cache.Append(row, row);
        cache.Append(row, row);
        Assert.Throws<InvalidOperationException>(() => cache.Append(row, row));
    }

    [Fact]
    public void ClearKeepsCapacity()
    {
        DuoKvCache cache = new DuoKvCache(2, Width);
        float[] row = new float[Width];
        cache.Append(row, row);
        cache.Clear();
        Assert.Equal(0, cache.Rows);
        Assert.Equal(2, cache.CapacityRows);
    }

    /// <summary>Scalar reference: softmax over (q·k / sqrt(d) + bias) per head, then the weighted values.</summary>
    private static float[] Attend(float[] q, int queryRows, float[] keys, float[] values, int keyRows, float[] bias)
    {
        float[] output = new float[queryRows * Width];
        float scale = 1f / MathF.Sqrt(HeadDim);
        for (int qi = 0; qi < queryRows; qi++)
        {
            for (int h = 0; h < Heads; h++)
            {
                float[] score = new float[keyRows];
                float max = float.NegativeInfinity;
                for (int k = 0; k < keyRows; k++)
                {
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++)
                    {
                        dot += q[qi * Width + h * HeadDim + d] * keys[k * Width + h * HeadDim + d];
                    }
                    score[k] = dot * scale + bias[qi * keyRows + k];
                    max = MathF.Max(max, score[k]);
                }
                float sum = 0f;
                for (int k = 0; k < keyRows; k++)
                {
                    score[k] = MathF.Exp(score[k] - max);
                    sum += score[k];
                }
                for (int d = 0; d < HeadDim; d++)
                {
                    float acc = 0f;
                    for (int k = 0; k < keyRows; k++)
                    {
                        acc += score[k] / sum * values[k * Width + h * HeadDim + d];
                    }
                    output[qi * Width + h * HeadDim + d] = acc;
                }
            }
        }
        return output;
    }

    /// <summary>Queries are the last <paramref name="queries"/> rows of a causal sequence of <paramref name="past"/> + queries keys.</summary>
    private static float[] CausalBias(int past, int queries)
    {
        int total = past + queries;
        float[] bias = new float[queries * total];
        for (int qi = 0; qi < queries; qi++)
        {
            for (int k = 0; k < total; k++)
            {
                bias[qi * total + k] = k <= past + qi ? 0f : float.NegativeInfinity;
            }
        }
        return bias;
    }

    private static float[] Sequence(int rows, int seed)
    {
        float[] data = new float[rows * Width];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = MathF.Sin(seed * 1.7f + i * 0.37f);
        }
        return data;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(actual[i], expected[i] - 1e-5f, expected[i] + 1e-5f);
        }
    }
}

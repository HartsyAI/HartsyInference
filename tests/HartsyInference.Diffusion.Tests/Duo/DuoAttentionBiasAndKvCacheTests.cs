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
    public void EvictingThenMaskingEqualsFullSequenceWindowedBias()
    {
        // Blocks of 2 rows over 8 rows. A block sees the last 4 committed rows and causal rows of its own block.
        // The cached path physically evicts old rows and builds its bias over [cache | block] indices. The full-sequence path keeps every
        // row and builds a window bias over absolute indices. Both must agree at every block.
        const int blockRows = 2, blocks = 4, window = 4, total = blockRows * blocks;
        float[] keys = Sequence(total, 1), values = Sequence(total, 2), q = Sequence(total, 3);
        DuoKvCache cache = new DuoKvCache(window, Width);

        for (int b = 0; b < blocks; b++)
        {
            int start = b * blockRows;
            int cacheRows = cache.Rows;
            Assert.Equal(Math.Min(start, window), cacheRows);

            // Cached path: history rows from the cache, then the block's own rows.
            List<DuoMaskRect> cachedRects = [];
            for (int i = 0; i < blockRows; i++)
            {
                if (cacheRows > 0)
                {
                    cachedRects.Add(new DuoMaskRect(i, i + 1, 0, cacheRows));
                }
                cachedRects.Add(new DuoMaskRect(i, i + 1, cacheRows, cacheRows + i + 1));
            }
            float[] cachedKeys = new float[(cacheRows + blockRows) * Width];
            float[] cachedValues = new float[(cacheRows + blockRows) * Width];
            cache.Keys.CopyTo(cachedKeys);
            cache.Values.CopyTo(cachedValues);
            keys.AsSpan(start * Width, blockRows * Width).CopyTo(cachedKeys.AsSpan(cacheRows * Width));
            values.AsSpan(start * Width, blockRows * Width).CopyTo(cachedValues.AsSpan(cacheRows * Width));
            float[] cachedOut = Attend(q.AsSpan(start * Width, blockRows * Width).ToArray(), blockRows, cachedKeys, cachedValues,
                cacheRows + blockRows, DuoAttentionBias.Build(cachedRects, blockRows, cacheRows + blockRows));

            // Full-sequence path: every row kept; the window is a mask over absolute positions.
            List<DuoMaskRect> fullRects = [];
            for (int i = 0; i < blockRows; i++)
            {
                int position = start + i;
                fullRects.Add(new DuoMaskRect(i, i + 1, Math.Max(0, start - window), position + 1));
            }
            float[] fullOut = Attend(q.AsSpan(start * Width, blockRows * Width).ToArray(), blockRows,
                keys.AsSpan(0, (start + blockRows) * Width).ToArray(), values.AsSpan(0, (start + blockRows) * Width).ToArray(),
                start + blockRows, DuoAttentionBias.Build(fullRects, blockRows, start + blockRows));
            AssertClose(fullOut, cachedOut);

            // Commit the block: evict to make room, then append.
            int excess = cache.Rows + blockRows - window;
            if (excess > 0)
            {
                cache.EvictRange(0, excess);
            }
            cache.Append(keys.AsSpan(start * Width, blockRows * Width), values.AsSpan(start * Width, blockRows * Width));
        }
    }

    [Fact]
    public void SlidingWindowAfterEvictionEqualsAttentionOverTheWindow()
    {
        const int window = 4, chunk = 2;
        float[] keys = Sequence(6, 4), values = Sequence(6, 5);
        DuoKvCache cache = new DuoKvCache(window, Width);
        cache.Append(keys.AsSpan(0, 4 * Width), values.AsSpan(0, 4 * Width));

        // Commit the next two rows: drop the two oldest, then append.
        cache.EvictRange(0, chunk);
        cache.Append(keys.AsSpan(4 * Width, chunk * Width), values.AsSpan(4 * Width, chunk * Width));
        Assert.Equal(window, cache.Rows);

        float[] windowKeys = keys.AsSpan(2 * Width, window * Width).ToArray();
        float[] windowValues = values.AsSpan(2 * Width, window * Width).ToArray();
        Assert.True(cache.Keys.SequenceEqual(windowKeys));
        Assert.True(cache.Values.SequenceEqual(windowValues));
    }

    [Fact]
    public void AppendRefusesMismatchedAndPartialRows()
    {
        DuoKvCache cache = new DuoKvCache(2, Width);
        Assert.Throws<ArgumentException>(() => cache.Append(new float[Width], new float[Width + 1]));
        Assert.Throws<ArgumentException>(() => cache.Append(new float[Width + 1], new float[Width + 1]));
        Assert.Equal(0, cache.Rows);
    }

    [Fact]
    public void EvictRangeInTheMiddleKeepsTheRowsOnEitherSideInOrder()
    {
        DuoKvCache cache = new DuoKvCache(5, 2);
        cache.Append([0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f], [10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f]);

        cache.EvictRange(1, 2);

        Assert.Equal(2, cache.Rows);
        Assert.True(cache.Keys.SequenceEqual(new float[] { 0f, 1f, 6f, 7f }));
        Assert.True(cache.Values.SequenceEqual(new float[] { 10f, 11f, 16f, 17f }));
    }

    [Fact]
    public void EvictOldestChecksBoundsAndZeroIsANoOp()
    {
        DuoKvCache cache = new DuoKvCache(3, Width);
        float[] keys = Sequence(2, 7), values = Sequence(2, 8);
        cache.Append(keys, values);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.EvictRange(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.EvictRange(0, -1));
        cache.EvictRange(0, 0);
        Assert.Equal(2, cache.Rows);
        Assert.True(cache.Keys.SequenceEqual(keys));
        Assert.True(cache.Values.SequenceEqual(values));
    }

    [Fact]
    public void DegenerateRectangleIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DuoAttentionBias.Build([new DuoMaskRect(1, 1, 0, 1)], 2, 2));
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

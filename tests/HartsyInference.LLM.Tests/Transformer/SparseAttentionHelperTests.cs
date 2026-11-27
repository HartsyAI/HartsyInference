using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Compression;
using HartsyInference.LLM.Transformer.Sparse;

namespace HartsyInference.LLM.Tests.Transformer;

public sealed class SparseAttentionHelperTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(126)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(2175)]
    [InlineData(2176)]
    [InlineData(2177)]
    [InlineData(100000)]
    public void WindowMerge_MatchesMergeSwaTopkFormula(int position)
    {
        const int topkWidth = 2048, window = 128;
        int limit = SparseWindowMerge.IndexerLimit(position, window);
        int n = Math.Min(limit, topkWidth);
        // An ascending top-k: n distinct reachable keys picked at random.
        Random rng = new(position);
        int[] topk = Enumerable.Range(0, limit).OrderBy(_ => rng.Next()).Take(n).Order().ToArray();
        int[] row = new int[topkWidth + window];

        int count = SparseWindowMerge.Merge(topk, topkWidth, position, window, row);

        // merge_swa_topk: seq = pos + 1, n = min(max(seq - window, 0), topk), w = min(seq, window), row = [topk[:n] | seq-w .. seq-1 | -1 ...]
        int seq = position + 1, head = Math.Min(Math.Max(seq - window, 0), topkWidth), w = Math.Min(seq, window);
        int[] expected = new int[topkWidth + window];
        Array.Fill(expected, -1);
        topk.AsSpan(0, head).CopyTo(expected);
        for (int i = 0; i < w; i++) expected[head + i] = seq - w + i;
        Assert.Equal(head + w, count);
        Assert.Equal(expected, row);
    }

    [Fact]
    public void WindowMerge_RejectsTopkEntryInsideTheWindow()
    {
        int[] row = new int[4 + 8];
        Assert.Throws<ArgumentException>(() => SparseWindowMerge.Merge([3], 4, position: 11, window: 8, row));
    }

    [Fact]
    public void TopK_TiesGoToLowerIndexAndUnreachableBecomeMinusOne()
    {
        float[] scores = [1f, 5f, 5f, 2f, float.NegativeInfinity, float.NegativeInfinity];
        int[] dest = new int[4];

        int written = SparseTopK.SelectSorted(scores, limit: 4, topk: 4, offset: 0, dest);

        Assert.Equal(4, written);
        Assert.Equal([0, 1, 2, 3], dest);
        Assert.Equal([1, 2], SparseTopK.Best([0f, 3f, 3f, 1f], 2));

        written = SparseTopK.SelectSorted([4f, 1f, float.NegativeInfinity], limit: 2, topk: 3, offset: 10, dest);
        Assert.Equal(3, written);
        Assert.Equal([10, 11, -1], dest[..3]);
    }

    [Theory]
    [InlineData(0, true, false, false)]
    [InlineData(-2, false, false, true)]
    [InlineData(1, false, true, false)]
    [InlineData(4, false, true, false)]
    [InlineData(128, false, true, false)]
    public void AttentionKind_MapsCompressRatios(int ratio, bool standard, bool compressed, bool dsa)
    {
        LayerAttentionKind kind = LayerAttentionKind.FromCompressRatio(ratio);
        Assert.Equal(standard, kind.IsStandard);
        Assert.Equal(compressed, kind.IsCompressed);
        Assert.Equal(dsa, kind.IsDsa);
        Assert.Equal(ratio, kind.CompressRatio);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-3)]
    public void AttentionKind_RefusesUnknownNegativeRatiosByName(int ratio)
    {
        NotSupportedException e = Assert.Throws<NotSupportedException>(() => LayerAttentionKind.FromCompressRatio(ratio));
        Assert.Contains($"compress_ratio {ratio}", e.Message);
    }

    [Fact]
    public void HeadMix_FoldsIPlusVUTransposeAndMixesOverHeads()
    {
        const int heads = 3, rank = 2, n = 4;
        // U and V differ, so U V^T and V U^T differ and a transposed fold fails the check.
        float[] u = [1f, 0f, 0.5f, -1f, 2f, 0.25f];
        float[] v = [0f, 1f, 3f, 0.5f, -0.5f, 1f];

        float[] m = HeadMix.FoldLowRank(u, v, heads, rank, roundToBf16: false);

        for (int h = 0; h < heads; h++)
            for (int g = 0; g < heads; g++)
            {
                float expected = (h == g ? 1f : 0f) + v[h * rank] * u[g * rank] + v[h * rank + 1] * u[g * rank + 1];
                Assert.Equal(expected, m[h * heads + g], 6);
            }
        Assert.NotEqual(m[0 * heads + 1], m[1 * heads + 0]);

        float[] o = new float[heads * n];
        for (int i = 0; i < o.Length; i++) o[i] = i * 0.5f - 2f;
        using Tensor mix = Make(m, heads, heads);
        using Tensor input = Make(o, heads, n);
        using Tensor output = new(new TensorShape(heads, n), DType.F32);
        HeadMix.Apply(new CpuBackend(), mix, input, output);

        float[] y = output.AsReadOnlySpan<float>().ToArray();
        for (int h = 0; h < heads; h++)
            for (int j = 0; j < n; j++)
            {
                float expected = 0f;
                for (int g = 0; g < heads; g++) expected += m[h * heads + g] * o[g * n + j];
                Assert.Equal(expected, y[h * n + j], 5);
            }
    }

    [Theory]
    [InlineData(true, RotaryPairing.RotateHalf)]
    [InlineData(false, RotaryPairing.Interleaved)]
    [InlineData(false, RotaryPairing.RotateHalf)]
    [InlineData(true, RotaryPairing.Interleaved)]
    public void PartialRotary_RotatesTheChosenSliceWithTheChosenPairing(bool leading, RotaryPairing pairing)
    {
        const int tokens = 3, heads = 2, headDim = 12, rd = 8, half = rd / 2;
        DeepSeekV41RopeTable rope = DeepSeekV41RopeTable.Build(rd, 16, 10000, null);
        int[] positions = [0, 5, 13];
        float[] data = new float[tokens * heads * headDim];
        Random rng = new(21);
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] expected = (float[])data.Clone();
        int offset = leading ? 0 : headDim - rd;
        for (int t = 0; t < tokens; t++)
            for (int h = 0; h < heads; h++)
            {
                int b = (t * heads + h) * headDim + offset;
                for (int i = 0; i < half; i++)
                {
                    float c = rope.Cos[positions[t] * half + i], s = rope.Sin[positions[t] * half + i];
                    (int a, int bb) = pairing == RotaryPairing.Interleaved ? (b + 2 * i, b + 2 * i + 1) : (b + i, b + i + half);
                    float x0 = data[a], x1 = data[bb];
                    expected[a] = x0 * c - x1 * s;
                    expected[bb] = x0 * s + x1 * c;
                }
            }

        long[] shape = [1, tokens, heads, headDim];
        if (leading) PartialRotary.RotateLeading(new CpuBackend(), rope, rd, data, shape, positions, inverse: false, pairing);
        else PartialRotary.Rotate(new CpuBackend(), rope, rd, data, shape, positions, inverse: false, pairing);

        for (int i = 0; i < data.Length; i++) Assert.Equal(expected[i], data[i], 5);
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}

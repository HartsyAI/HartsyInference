using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests;

/// <summary>The contiguous-range plan of a multi-image cross-attention group, and that the ranged layer path equals the row-masked one.</summary>
public sealed class CrossAttentionRangedGroupTests
{
    private const int Rows = 8;
    private const int VisionLength = 6;
    private const int Inter = 12;

    [Fact]
    public void ContiguousRangeFindsTheRunOfSelectedRows()
    {
        Assert.Equal((1, 2), CrossImageGroup.ContiguousRange([0f, 1f, 1f, 0f]));
        Assert.Equal((0, 3), CrossImageGroup.ContiguousRange([1f, 1f, 1f]));
        Assert.Equal((2, 2), CrossImageGroup.ContiguousRange([0f, 0f, 1f, 1f]));
        Assert.Equal((0, 2), CrossImageGroup.ContiguousRange([1f, 1f, 0f, 0f]));
    }

    [Fact]
    public void ContiguousRangeOfASingleRowHasCountOne()
    {
        Assert.Equal((2, 1), CrossImageGroup.ContiguousRange([0f, 0f, 1f, 0f]));
        Assert.Equal((0, 1), CrossImageGroup.ContiguousRange([1f]));
    }

    [Fact]
    public void ContiguousRangeIsUnknownForAGapOrNoRows()
    {
        Assert.Equal((-1, 0), CrossImageGroup.ContiguousRange([1f, 0f, 1f]));
        Assert.Equal((-1, 0), CrossImageGroup.ContiguousRange([0f, 1f, 1f, 0f, 0f, 1f]));
        Assert.Equal((-1, 0), CrossImageGroup.ContiguousRange([0f, 0f, 0f]));
        Assert.Equal((-1, 0), CrossImageGroup.ContiguousRange([]));
    }

    [Fact]
    public void ContiguousRangeTreatsAnyNonZeroValueAsSelected()
    {
        Assert.Equal((1, 3), CrossImageGroup.ContiguousRange([0f, 2f, 0.5f, 1f, 0f]));
    }

    [Theory]
    [InlineData(8, 2, 4, 1)]   // merged heads equal the hidden width
    [InlineData(6, 2, 4, 1)]   // merged heads (8) wider than the hidden width (6)
    [InlineData(8, 4, 2, 2)]
    public void RangedGroupsMatchTheRowMaskedPathWithAGapAndUncoveredRows(int hidden, int heads, int headDim, int kvHeads)
    {
        // Rows 1-3 see the first 2 vision keys, rows 4-5 all 6; row 0 (leading gap) and rows 6-7 (trailing) see nothing.
        (int Start, int Count, int Kv)[] spans = [(1, 3, 2), (4, 2, VisionLength)];

        float[] masked = Run(hidden, heads, headDim, kvHeads, spans, ranged: false);
        AssertClose(masked, Run(hidden, heads, headDim, kvHeads, spans, ranged: true), 1e-5f);
    }

    [Fact]
    public void RangedGroupsMatchWhenListedOutOfOrderWithAGapBetweenThem()
    {
        (int Start, int Count, int Kv)[] spans = [(5, 2, VisionLength), (1, 1, 3), (3, 1, 4)];

        AssertClose(Run(8, 2, 4, 1, spans, ranged: false), Run(8, 2, 4, 1, spans, ranged: true), 1e-5f);
    }

    [Fact]
    public void RangedGroupsCoveringEveryRowMatchTheMaskedPath()
    {
        (int Start, int Count, int Kv)[] spans = [(0, 4, 3), (4, 4, VisionLength)];

        AssertClose(Run(8, 2, 4, 1, spans, ranged: false), Run(8, 2, 4, 1, spans, ranged: true), 1e-5f);
    }

    [Fact]
    public void OverlappingRangedGroupsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => Run(8, 2, 4, 1, [(0, 4, 2), (3, 3, VisionLength)], ranged: true));
    }

    /// <summary>Runs one layer over seeded weights, text and vision; the groups carry only their row range when <paramref name="ranged"/>,
    /// otherwise only their row-selection mask.</summary>
    private static float[] Run(int hidden, int heads, int headDim, int kvHeads, (int Start, int Count, int Kv)[] spans, bool ranged)
    {
        using CpuBackend backend = new();
        Random rng = new(7);
        List<Tensor> owned = [];
        try
        {
            Tensor Own(Tensor t)
            {
                owned.Add(t);
                return t;
            }

            Dictionary<string, Tensor> w = new()
            {
                ["l.input_layernorm.weight"] = Own(Norm(rng, hidden)),
                ["l.post_attention_layernorm.weight"] = Own(Norm(rng, hidden)),
                ["l.cross_attn.q_proj.weight"] = Own(Filled(rng, 0.4f, heads * headDim, hidden)),
                ["l.cross_attn.k_proj.weight"] = Own(Filled(rng, 0.4f, kvHeads * headDim, hidden)),
                ["l.cross_attn.v_proj.weight"] = Own(Filled(rng, 0.4f, kvHeads * headDim, hidden)),
                ["l.cross_attn.o_proj.weight"] = Own(Filled(rng, 0.4f, hidden, heads * headDim)),
                ["l.cross_attn.q_norm.weight"] = Own(Norm(rng, headDim)),
                ["l.cross_attn.k_norm.weight"] = Own(Norm(rng, headDim)),
                ["l.mlp.gate_proj.weight"] = Own(Filled(rng, 0.4f, Inter, hidden)),
                ["l.mlp.up_proj.weight"] = Own(Filled(rng, 0.4f, Inter, hidden)),
                ["l.mlp.down_proj.weight"] = Own(Filled(rng, 0.4f, hidden, Inter)),
                ["l.cross_attn_attn_gate"] = Own(Scalar(0.8f)),
                ["l.cross_attn_mlp_gate"] = Own(Scalar(-0.6f)),
            };
            MllamaCrossAttentionLayer layer = new(hidden, heads, kvHeads, headDim, Inter, 1e-5f);
            layer.LoadWeights(w, "l");

            Tensor text = Own(Filled(rng, 1.0f, 1, Rows, hidden));
            Tensor vision = Own(Filled(rng, 1.0f, 1, VisionLength, hidden));
            Tensor textCos = Own(Table(Rows, headDim, Math.Cos));
            Tensor textSin = Own(Table(Rows, headDim, Math.Sin));
            Tensor visionCos = Own(Table(VisionLength, headDim, Math.Cos));
            Tensor visionSin = Own(Table(VisionLength, headDim, Math.Sin));

            float[] rowMask = new float[Rows];
            List<CrossImageGroup> groups = [];
            foreach ((int start, int count, int kv) in spans)
            {
                float[] select = new float[Rows];
                for (int r = start; r < start + count; r++)
                    rowMask[r] = select[r] = 1f;
                groups.Add(ranged
                    ? new CrossImageGroup(kv, null, start, count)
                    : new CrossImageGroup(kv, Own(RowTensor(select, hidden))));
            }
            Tensor mask = Own(RowTensor(rowMask, hidden));
            MultimodalRope rope = new(textCos, textSin, visionCos, visionSin, mask, groups);

            using Tensor result = layer.Forward(backend, text, Rows, vision, VisionLength, rope);
            return result.AsSpan<float>().ToArray();
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    private static Tensor Filled(Random rng, float scale, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5) * 2f * scale;
        return t;
    }

    private static Tensor Norm(Random rng, int width)
    {
        Tensor t = new(new TensorShape(width), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = 1f + (float)(rng.NextDouble() - 0.5) * 0.2f;
        return t;
    }

    private static Tensor Scalar(float value)
    {
        Tensor t = new(new TensorShape(1), DType.F32);
        t.AsSpan<float>()[0] = value;
        return t;
    }

    /// <summary>A <c>[1, rows, headDim]</c> rotary table whose halves repeat, as the rotate-half layout reads it.</summary>
    private static Tensor Table(int rows, int headDim, Func<double, double> fn)
    {
        Tensor t = new(new TensorShape(1, rows, headDim), DType.F32);
        Span<float> s = t.AsSpan<float>();
        int half = headDim / 2;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < headDim; c++)
                s[r * headDim + c] = (float)fn(0.3 * (r + 1) * (1 + c % half));
        return t;
    }

    /// <summary>A <c>[1, rows, width]</c> tensor whose row r is filled with <c>values[r]</c>.</summary>
    private static Tensor RowTensor(float[] values, int width)
    {
        Tensor t = new(new TensorShape(1, values.Length, width), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int r = 0; r < values.Length; r++) s.Slice(r * width, width).Fill(values[r]);
        return t;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tolerance, $"Element {i}: masked {expected[i]}, ranged {actual[i]}.");
    }
}

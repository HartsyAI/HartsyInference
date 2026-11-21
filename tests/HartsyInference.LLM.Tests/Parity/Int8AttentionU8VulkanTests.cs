using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>INT8 attention with u8 probabilities on Vulkan, from F32 operands to the output, against
/// <see cref="Int8AttentionV2Reference"/> run with <see cref="Int8AttentionExp2.Evaluate"/> as its exp2.
/// <para>The bound was fixed before the first run: every tile maximum bit-equal, every P byte and tile sum equal to the
/// reference's tile trace, and every output value bit-equal (a NaN for a NaN).</para>
/// <para>That bound was missed once on the first run, and the allowance below was set after seeing it: with scores
/// spread over thousands of binary orders, one output value of 4,352 was 0 where the reference has -2.8e-37. The
/// reference had carried a drain scale below 2^-126, a subnormal, and the RTX 3060 reads subnormals as zero. An output
/// value may therefore differ where both it and the reference's are below 2^-100 in magnitude; such values are
/// counted, and every case but that one still requires none.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class Int8AttentionU8VulkanTests(ITestOutputHelper output)
{
    private sealed record Case(int Batch, int KvHeads, int Group, int HeadDim, Int8Rotation Rotation, bool PerTile, bool Smooth,
        int BlockKeys, int Keys, int Rows, float Scale)
    {
        public int Heads => KvHeads * Group;
        public int Padded => HeadDim <= 64 ? 64 : HeadDim <= 128 ? 128 : 256;
    }

    private static void FillRandom(Tensor tensor, int seed, float shared)
    {
        Random random = new(seed);
        TensorShape shape = tensor.Shape;
        int width = (int)shape[3], perHead = (int)shape[2] * width;
        float[] common = new float[shape[0] * shape[1] * width];
        for (int i = 0; i < common.Length; i++) common[i] = (random.NextSingle() * 2f - 1f) * shared;
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = random.NextSingle() * 2f - 1f + common[i / perHead * width + i % width];
    }

    /// <summary>A head-major <c>[batch · heads, tokens, padded]</c> copy of <c>[batch, heads, tokens, dim]</c>, the
    /// channels past <c>dim</c> zero.</summary>
    private static Tensor Padded(Tensor source, int padded)
    {
        TensorShape s = source.Shape;
        int rowsTotal = (int)(s[0] * s[1] * s[2]), dim = (int)s[3];
        Tensor result = new(new TensorShape(s[0] * s[1], s[2], padded), DType.F32);
        ReadOnlySpan<float> from = source.AsReadOnlySpan<float>();
        Span<float> to = result.AsSpan<float>();
        to.Clear();
        for (int r = 0; r < rowsTotal; r++) from.Slice(r * dim, dim).CopyTo(to.Slice(r * padded, dim));
        return result;
    }

    private static bool SameBits(float expected, float actual) => float.IsNaN(expected) ? float.IsNaN(actual)
        : BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual);

    /// <summary>Runs one call through the reference and through Vulkan and compares the trace and the output. Returns
    /// how many (row, tile) cells the reference traced, how many output rows are NaN, and how many output values
    /// differ with both sides below 2^-100 (the subnormal allowance); any other difference fails.</summary>
    private static (int Traced, int NanRows, int Tiny) Run(Case c, Action<Tensor, Tensor, Tensor>? shape = null)
    {
        int heads = c.Heads, dp = c.Padded, bh = c.Batch * heads, bkv = c.Batch * c.KvHeads;
        int tiles = (c.Keys + c.BlockKeys - 1) / c.BlockKeys;
        using Tensor query = new(new TensorShape(c.Batch, heads, c.Rows, c.HeadDim), DType.F32);
        using Tensor key = new(new TensorShape(c.Batch, c.KvHeads, c.Keys, c.HeadDim), DType.F32);
        using Tensor value = new(key.Shape, DType.F32);
        FillRandom(query, 101 + c.HeadDim + c.Group, 0.5f);
        FillRandom(key, 102 + c.HeadDim + c.Keys, 3f);
        FillRandom(value, 103 + c.HeadDim, 1f);
        shape?.Invoke(query, key, value);
        using Tensor expected = new(query.Shape, DType.F32);

        Dictionary<(int Row, int Tile), (float Max, byte[] Bytes, int Sum)> trace = [];
        AttentionCall call = AttentionCall.FromFlashAttention(expected, query, key, value, c.Keys, false, 0, c.Scale, 0f, null, 0, null);
        Int8AttentionV2Options options = new()
        {
            SmoothKeys = c.Smooth, KeyCenter = Int8KeyCenter.SegmentMean, Rotation = c.Rotation,
            KeyScales = Int8KeyScaleGrouping.PerKey, ValueScalesPerTile = c.PerTile, BlockCols = c.BlockKeys,
            Probabilities = Int8ProbabilityFormat.U8, Exp2 = Int8AttentionExp2.Evaluate,
            TileTrace = (batchHead, row, tile, tileMax, bytes, _, sum) =>
                trace[(batchHead * c.Rows + row, tile)] = (tileMax, bytes.ToArray(), sum),
        };
        Int8AttentionV2Operands operands = Int8AttentionV2Reference.Quantize(in call, options);
        Int8AttentionV2Reference.Attend(in call, operands);

        string what = $"{c}";
        using VulkanBackend backend = new();
        using Tensor keyRows = Padded(key, dp), valueRows = Padded(value, dp), queryRows = Padded(query, dp);
        using Tensor keyBytes = new(new TensorShape(bkv * c.Keys, dp), DType.I8), keyScales = new(new TensorShape(bkv * c.Keys), DType.F32);
        using Tensor means = new(new TensorShape(bkv, dp), DType.F32);
        if (c.Smooth) backend.QuantizeInt8AttentionKeys(keyBytes, keyScales, means, keyRows, c.Rotation);
        else
        {
            using Tensor flat = new(new TensorShape(bkv * c.Keys, dp), DType.F32);
            keyRows.AsReadOnlySpan<float>().CopyTo(flat.AsSpan<float>());
            backend.QuantizeInt8AttentionRows(keyBytes, keyScales, flat, c.Rotation);
        }
        using Tensor valueBytes = new(keyBytes.Shape, DType.I8);
        using Tensor valueScales = new(new TensorShape(bkv, c.PerTile ? tiles : 1, dp), DType.F32);
        backend.QuantizeInt8AttentionValues(valueBytes, valueScales, valueRows, c.PerTile ? c.BlockKeys : c.Keys);
        using Tensor queryBytes = new(new TensorShape(bh * c.Rows, dp), DType.I8), queryScales = new(new TensorShape(bh * c.Rows), DType.F32);
        using Tensor dots = new(new TensorShape(bh, c.Rows), DType.F32);
        backend.QuantizeInt8AttentionQueries(queryBytes, queryScales, c.Smooth ? dots : null, queryRows, c.Smooth ? means : null, c.Rotation);

        long cells = (long)bh * c.Rows * tiles;
        using Tensor tileMax = new(new TensorShape(cells), DType.F32), tileSums = new(new TensorShape(cells), DType.U32);
        using Tensor probabilities = new(new TensorShape(cells * c.BlockKeys), DType.U8);
        backend.Int8AttentionU8Tiles(tileMax, tileSums, probabilities, queryBytes, queryScales, c.Smooth ? dots : null, keyBytes,
            keyScales, bh, bkv, c.BlockKeys, c.Scale);
        using Tensor actual = new(new TensorShape(bh, c.Rows, c.HeadDim), DType.F32);
        backend.Int8AttentionU8Drain(actual, tileMax, tileSums, probabilities, valueBytes, valueScales, bh, bkv, c.BlockKeys, c.PerTile);
        // The entry that runs both passes, which has to give the same bits.
        using Tensor composed = new(actual.Shape, DType.F32);
        backend.Int8AttentionU8(composed, queryBytes, queryScales, c.Smooth ? dots : null, keyBytes, keyScales, valueBytes,
            valueScales, bh, bkv, c.BlockKeys, c.PerTile, c.Scale);
        backend.Sync();
        ReadOnlySpan<float> two = actual.AsReadOnlySpan<float>(), one = composed.AsReadOnlySpan<float>();
        for (int i = 0; i < two.Length; i++)
            Assert.True(SameBits(two[i], one[i]), $"{c}: the composed entry gives {one[i]:R} at {i}, the two passes {two[i]:R}");

        // The first pass against the trace, so that a miss in the output can be placed.
        ReadOnlySpan<float> tm = tileMax.AsReadOnlySpan<float>();
        ReadOnlySpan<uint> sums = tileSums.AsReadOnlySpan<uint>();
        ReadOnlySpan<byte> p = probabilities.AsReadOnlySpan<byte>();
        int maxMisses = 0, sumMisses = 0, byteMisses = 0;
        string firstMiss = "";
        foreach (((int row, int tile), (float max, byte[] bytes, int sum)) in trace)
        {
            int cell = row * tiles + tile;
            if (!SameBits(max, tm[cell]))
            {
                if (maxMisses++ == 0) firstMiss = $"row {row} tile {tile}: tile max {tm[cell]:R}, expected {max:R}";
                continue;
            }
            if (sum != (int)sums[cell]) sumMisses++;
            for (int j = 0; j < bytes.Length; j++)
                if (bytes[j] != p[cell * c.BlockKeys + j])
                {
                    if (byteMisses++ == 0 && firstMiss.Length == 0)
                        firstMiss = $"row {row} tile {tile} key slot {j}: P {p[cell * c.BlockKeys + j]}, expected {bytes[j]}";
                }
        }
        Assert.True(maxMisses == 0 && sumMisses == 0 && byteMisses == 0,
            $"{what}: {maxMisses} tile maxima, {sumMisses} sums and {byteMisses} P bytes differ over {trace.Count} traced tiles; {firstMiss}");

        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>(), a = actual.AsReadOnlySpan<float>();
        Assert.Equal(e.Length, a.Length);
        int misses = 0, first = -1, nanRows = 0, tiny = 0;
        float allowance = MathF.ScaleB(1f, -100);
        for (int i = 0; i < e.Length; i++)
        {
            if (i % c.HeadDim == 0 && float.IsNaN(e[i])) nanRows++;
            if (SameBits(e[i], a[i])) continue;
            if (MathF.Abs(e[i]) < allowance && MathF.Abs(a[i]) < allowance)
            {
                // The allowance is for one cause, so that cause has to be there: for this row and channel the
                // reference formed a drain scale (tile scale times V scale) that is a subnormal float.
                int row = i / c.HeadDim, channel = i % c.HeadDim, kvHead = row / c.Rows / c.Group;
                bool subnormalScale = false, valid = false;
                float m = 0f;
                for (int t = 0; t < tiles && c.PerTile; t++)
                {
                    float tileM = trace[(row, t)].Max - Int8AttentionV2Reference.U8Offset, tileScale = 1f;
                    if (!valid) (m, valid) = (tileM, true);
                    else
                    {
                        if (tileM < m) tileScale = Int8AttentionExp2.Evaluate(-MathF.Abs(m - tileM));
                        m = MathF.Max(m, tileM);
                    }
                    float scaled = tileScale * operands.ValueScales[(kvHead * tiles + t) * dp + channel];
                    if (scaled != 0f && MathF.Abs(scaled) < 1.17549435e-38f) subnormalScale = true;
                }
                Assert.True(subnormalScale, $"{what}: output {i} differs below 2^-100 ({a[i]:R}, expected {e[i]:R}) with no subnormal drain scale in the reference");
                tiny++;
                continue;
            }
            if (first < 0) first = i;
            misses++;
        }
        Assert.True(misses == 0, $"{what}: {misses} of {e.Length} output values differ; element {first} is "
            + $"{(first < 0 ? 0f : a[first]):R}, expected {(first < 0 ? 0f : e[first]):R}");
        return (trace.Count, nanRows, tiny);
    }

    [VulkanTheory]
    [InlineData(1, 1, 64, Int8Rotation.None, false, true, 128, 300)]
    [InlineData(2, 2, 64, Int8Rotation.H64, true, true, 64, 300)]
    [InlineData(1, 4, 128, Int8Rotation.SignedH128, true, true, 128, 300)]
    [InlineData(2, 1, 128, Int8Rotation.H4, false, false, 128, 257)]
    [InlineData(1, 2, 256, Int8Rotation.PaddedH128, false, true, 64, 130)]
    [InlineData(1, 3, 256, Int8Rotation.H4, true, false, 128, 128)]
    [InlineData(1, 2, 96, Int8Rotation.H4, true, true, 128, 300)]
    public void TheOutputEqualsTheReference(int batch, int group, int headDim, Int8Rotation rotation, bool perTile, bool smooth,
        int blockKeys, int keys)
    {
        Case c = new(batch, 2, group, headDim, rotation, perTile, smooth, blockKeys, keys, 21, 1f / MathF.Sqrt(headDim));
        (int traced, int nanRows, int tiny) = Run(c);
        Assert.Equal(c.Batch * c.Heads * c.Rows * ((keys + blockKeys - 1) / blockKeys), traced);
        Assert.Equal(0, nanRows);
        Assert.Equal(0, tiny);
    }

    /// <summary>Scores spread over hundreds of binary orders: tiles far below the running maximum drain with scales down to
    /// the fused multiply-add's small-product range, and probabilities past the point where 2^x is taken as zero. At a
    /// scale of 30 the reference carries subnormal drain scales, and the allowance of the class summary applies: a few
    /// output values below 2^-100 may differ, and no more than one in a hundred.</summary>
    [VulkanTheory]
    [InlineData(false, 6f, false)]
    [InlineData(true, 6f, false)]
    [InlineData(false, 30f, true)]
    [InlineData(true, 30f, true)]
    public void WidelySpreadScoresEqualTheReference(bool perTile, float scale, bool subnormalsExpected)
    {
        Case c = new(1, 2, 2, 64, Int8Rotation.H64, perTile, true, 64, 400, 17, scale);
        (_, int nanRows, int tiny) = Run(c);
        Assert.Equal(0, nanRows);
        output.WriteLine($"scale {scale}, per-tile V scales {perTile}: {tiny} of {c.Heads * c.Rows * c.HeadDim} output values differ below 2^-100");
        Assert.True(subnormalsExpected ? tiny <= c.Heads * c.Rows * c.HeadDim / 100 : tiny == 0, $"{tiny} values differ below 2^-100");
    }

    /// <summary>A later tile equal to, above and below the running maximum: the second tile repeats the first key for key,
    /// the third is the first scaled up and the fourth the first scaled down.</summary>
    [VulkanFact]
    public void TilesEqualToAboveAndBelowTheRunningMaximum()
    {
        Case c = new(1, 1, 1, 64, Int8Rotation.None, true, false, 64, 256, 9, 0.5f);
        (_, int nanRows, int tiny) = Run(c, (_, key, _) =>
        {
            Span<float> k = key.AsSpan<float>();
            int tile = 64 * 64;
            for (int i = 0; i < tile; i++)
            {
                k[tile + i] = k[i];
                k[2 * tile + i] = k[i] * 1.5f;
                k[3 * tile + i] = k[i] * 0.25f;
            }
        });
        Assert.Equal(0, nanRows);
        Assert.Equal(0, tiny);
    }

    /// <summary>Rows the reference turns into NaNs, and only those: a query row holding a NaN, one holding an infinity,
    /// and (keys not centered) every row of a head one of whose keys holds a NaN.</summary>
    [VulkanFact]
    public void RowsThatAreNotFiniteAreNaNAsInTheReference()
    {
        Case c = new(1, 2, 1, 64, Int8Rotation.H4, false, false, 64, 150, 8, 0.125f);
        (_, int nanRows, int tiny) = Run(c, (query, key, _) =>
        {
            Span<float> q = query.AsSpan<float>(), k = key.AsSpan<float>();
            q[(0 * 8 + 2) * 64 + 5] = float.NaN;                    // head 0, row 2
            q[(0 * 8 + 6) * 64 + 9] = float.PositiveInfinity;       // head 0, row 6
            k[(1 * 150 + 100) * 64 + 3] = float.NaN;                // head 1, key 100: every row of head 1
        });
        Assert.Equal(2 + 8, nanRows);
        Assert.Equal(0, tiny);
    }

    /// <summary>No keys: every output value is zero, in place of whatever the tensor held.</summary>
    [VulkanFact]
    public void NoKeysGiveZeros()
    {
        const int heads = 2, rows = 3, dp = 64;
        using VulkanBackend backend = new();
        using Tensor queryBytes = new(new TensorShape(heads * rows, dp), DType.I8), queryScales = new(new TensorShape(heads * rows), DType.F32);
        using Tensor keyBytes = new(new TensorShape(0, dp), DType.I8), keyScales = new(new TensorShape(0), DType.F32);
        using Tensor valueBytes = new(new TensorShape(0, dp), DType.I8), valueScales = new(new TensorShape(heads, 1, dp), DType.F32);
        using Tensor output = new(new TensorShape(heads, rows, dp), DType.F32);
        output.AsSpan<float>().Fill(7f);
        backend.Int8AttentionU8(output, queryBytes, queryScales, null, keyBytes, keyScales, valueBytes, valueScales, heads, heads,
            128, false, 1f);
        backend.Sync();
        foreach (float x in output.AsReadOnlySpan<float>()) Assert.Equal(0f, x);
    }
}

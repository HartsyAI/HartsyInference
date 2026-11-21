using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The Vulkan key centering and value quantization of INT8 attention against
/// <see cref="Int8AttentionV2Reference.Quantize"/>: keys smoothed by their mean over one segment with per-key scales,
/// and values with one scale a channel for the whole head or for each key tile.
/// <para>The bound was fixed before the first run: every mean and scale bit-equal and every byte equal to the
/// reference's, on finite operands of ordinary magnitude.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class Int8AttentionCenteringVulkanTests
{
    /// <summary>Random values with a component every key of a head shares, which is what centering removes.</summary>
    private static Tensor Random(TensorShape shape, int seed, float shared)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        int heads = (int)shape[1], keys = (int)shape[2], width = (int)shape[3];
        float[] common = new float[heads * width];
        for (int i = 0; i < common.Length; i++) common[i] = (random.NextSingle() * 2f - 1f) * shared;
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            int head = i / (keys * width) % heads, channel = i % width;
            values[i] = random.NextSingle() * 2f - 1f + common[head * width + channel];
        }
        return tensor;
    }

    private static void AssertBits(float[] expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int misses = 0, first = -1;
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                if (first < 0) first = i;
                misses++;
            }
        Assert.True(misses == 0, $"{what}: {misses} of {expected.Length} differ; element {first} is "
            + $"{(first < 0 ? 0f : actual[first]):R}, expected {(first < 0 ? 0f : expected[first]):R}");
    }

    private static void AssertBytes(sbyte[] expected, ReadOnlySpan<sbyte> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int misses = 0, first = -1;
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i])
            {
                if (first < 0) first = i;
                misses++;
            }
        Assert.True(misses == 0, $"{what}: {misses} of {expected.Length} differ; element {first} is "
            + $"{(first < 0 ? 0 : actual[first])}, expected {(first < 0 ? 0 : expected[first])}");
    }

    [VulkanTheory]
    [InlineData(64, Int8Rotation.None, false)]
    [InlineData(64, Int8Rotation.H64, true)]
    [InlineData(128, Int8Rotation.H4, false)]
    [InlineData(128, Int8Rotation.SignedH128, true)]
    [InlineData(256, Int8Rotation.PaddedH128, false)]
    [InlineData(256, Int8Rotation.H4, true)]
    public void CenteredKeysAndValuesEqualTheReference(int headDim, Int8Rotation rotation, bool valueScalesPerTile)
    {
        // 300 keys: four whole runs of 64 and a short one for the mean, and two whole key tiles of 128 and a short one.
        const int heads = 2, queryRows = 4, keys = 300, tile = 128;
        using Tensor query = Random(new TensorShape(1, heads, queryRows, headDim), 41 + headDim, 0f);
        using Tensor key = Random(new TensorShape(1, heads, keys, headDim), 42 + headDim, 3f);
        using Tensor value = Random(key.Shape, 43 + headDim, 1f);
        using Tensor output = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, query, key, value, null, 1f / MathF.Sqrt(headDim), false);
        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call, new Int8AttentionV2Options
        {
            SmoothKeys = true, KeyCenter = Int8KeyCenter.SegmentMean, Rotation = rotation, KeyScales = Int8KeyScaleGrouping.PerKey,
            ValueScalesPerTile = valueScalesPerTile, BlockCols = tile, Probabilities = Int8ProbabilityFormat.U8,
        });
        string what = $"D {headDim}, {rotation}, per-tile V scales {valueScalesPerTile}";

        using VulkanBackend backend = new();
        using Tensor keyRows = new(new TensorShape(heads, keys, headDim), DType.F32);
        key.AsReadOnlySpan<float>().CopyTo(keyRows.AsSpan<float>());
        using Tensor keyBytes = new(new TensorShape(heads * keys, headDim), DType.I8);
        using Tensor keyScales = new(new TensorShape(heads * keys), DType.F32);
        using Tensor means = new(new TensorShape(heads, headDim), DType.F32);
        backend.QuantizeInt8AttentionKeys(keyBytes, keyScales, means, keyRows, rotation);
        backend.Sync();
        AssertBits(expected.KeyMeans, means.AsReadOnlySpan<float>(), $"key means, {what}");
        AssertBits(expected.KeyScales, keyScales.AsReadOnlySpan<float>(), $"key scales, {what}");
        AssertBytes(expected.Keys, keyBytes.AsReadOnlySpan<sbyte>(), $"key bytes, {what}");

        using Tensor valueRows = new(keyRows.Shape, DType.F32);
        value.AsReadOnlySpan<float>().CopyTo(valueRows.AsSpan<float>());
        int blockKeys = valueScalesPerTile ? tile : keys, blocks = valueScalesPerTile ? (keys + tile - 1) / tile : 1;
        using Tensor valueBytes = new(new TensorShape(heads * keys, headDim), DType.I8);
        using Tensor valueScales = new(new TensorShape(heads, blocks, headDim), DType.F32);
        backend.QuantizeInt8AttentionValues(valueBytes, valueScales, valueRows, blockKeys);
        backend.Sync();
        AssertBits(expected.ValueScales, valueScales.AsReadOnlySpan<float>(), $"value scales, {what}");
        AssertBytes(expected.Values, valueBytes.AsReadOnlySpan<sbyte>(), $"value bytes, {what}");
    }

    /// <summary>The value rules at the edges: a channel holding a NaN in one tile (a NaN scale and zero bytes for that
    /// channel in that tile only), an all-zero channel (the scale floor of 1) and a channel holding an infinity.</summary>
    [VulkanFact]
    public void ValueEdgeChannelsFollowTheReference()
    {
        const int heads = 1, keys = 200, headDim = 64, tile = 128;
        using Tensor query = Random(new TensorShape(1, heads, 2, headDim), 51, 0f);
        using Tensor key = Random(new TensorShape(1, heads, keys, headDim), 52, 0f);
        using Tensor value = Random(key.Shape, 53, 0f);
        Span<float> v = value.AsSpan<float>();
        v[10 * headDim + 5] = float.NaN;                                  // channel 5, first tile
        for (int j = 0; j < keys; j++) v[j * headDim + 9] = 0f;           // channel 9: zeros
        v[150 * headDim + 12] = float.PositiveInfinity;                   // channel 12, second tile
        using Tensor output = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, query, key, value, null, 0.125f, false);
        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call, new Int8AttentionV2Options
        {
            SmoothKeys = false, KeyScales = Int8KeyScaleGrouping.PerKey, ValueScalesPerTile = true, BlockCols = tile,
        });
        Assert.True(float.IsNaN(expected.ValueScales[5]));
        Assert.Equal(1f, expected.ValueScales[9]);
        Assert.True(float.IsPositiveInfinity(expected.ValueScales[headDim + 12]));

        using VulkanBackend backend = new();
        using Tensor rows = new(new TensorShape(heads, keys, headDim), DType.F32);
        value.AsReadOnlySpan<float>().CopyTo(rows.AsSpan<float>());
        using Tensor bytes = new(new TensorShape(heads * keys, headDim), DType.I8);
        using Tensor scales = new(new TensorShape(heads, 2, headDim), DType.F32);
        backend.QuantizeInt8AttentionValues(bytes, scales, rows, tile);
        backend.Sync();
        ReadOnlySpan<float> actual = scales.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.ValueScales.Length; i++)
            Assert.True(float.IsNaN(expected.ValueScales[i]) ? float.IsNaN(actual[i])
                : BitConverter.SingleToInt32Bits(expected.ValueScales[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"value scale {i} is {actual[i]:R}, expected {expected.ValueScales[i]:R}");
        AssertBytes(expected.Values, bytes.AsReadOnlySpan<sbyte>(), "value bytes at the edges");
    }
}

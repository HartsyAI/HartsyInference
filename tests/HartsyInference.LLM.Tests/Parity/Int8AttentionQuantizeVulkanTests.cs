using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The Vulkan Q and K prequantization of INT8 attention against <see cref="Int8AttentionV2Reference.Quantize"/>
/// with per-key scales and no key smoothing: every rotation at every padded head dim it runs at.
/// <para>The bound was fixed before the first run: every byte equal and every scale bit-equal to the reference's, on
/// finite rows of ordinary magnitude. That needs the device to round a divide correctly, which the Vulkan specification
/// does not require; where a device does not, a mismatch here is a finding about that device.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class Int8AttentionQuantizeVulkanTests
{
    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        // One channel far larger than the rest, the case a rotation exists for.
        long width = shape[shape.Rank - 1];
        for (int i = 3; i < values.Length; i += (int)width) values[i] *= 40f;
        return tensor;
    }

    [VulkanTheory]
    [InlineData(64, Int8Rotation.None)]
    [InlineData(64, Int8Rotation.H4)]
    [InlineData(64, Int8Rotation.H64)]
    [InlineData(128, Int8Rotation.None)]
    [InlineData(128, Int8Rotation.H4)]
    [InlineData(128, Int8Rotation.SignedH128)]
    [InlineData(256, Int8Rotation.None)]
    [InlineData(256, Int8Rotation.H4)]
    [InlineData(256, Int8Rotation.PaddedH128)]
    public void QueryAndKeyBytesAndScalesEqualTheReference(int headDim, Int8Rotation rotation)
    {
        const int batch = 1, heads = 2, queryRows = 40, keys = 300;
        using Tensor query = Random(new TensorShape(batch, heads, queryRows, headDim), 11 + headDim);
        using Tensor key = Random(new TensorShape(batch, heads, keys, headDim), 12 + headDim);
        using Tensor value = Random(key.Shape, 13 + headDim);
        using Tensor output = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, query, key, value, null, 1f / MathF.Sqrt(headDim), false);
        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call,
            new Int8AttentionV2Options { SmoothKeys = false, Rotation = rotation, KeyScales = Int8KeyScaleGrouping.PerKey });
        Assert.Equal(rotation, expected.Rotation);

        using VulkanBackend backend = new();
        Compare(backend, query, heads * queryRows, headDim, rotation, expected.Queries, expected.QueryScales, "query");
        Compare(backend, key, heads * keys, headDim, rotation, expected.Keys, expected.KeyScales, "key");
    }

    private static void Compare(VulkanBackend backend, Tensor operand, int rowCount, int headDim, Int8Rotation rotation, sbyte[] expectedBytes,
        float[] expectedScales, string what)
    {
        // Head-major [B, H, S, D] is already row after row; the backend takes it as [rows, D].
        using Tensor rows = new(new TensorShape(rowCount, headDim), DType.F32);
        operand.AsReadOnlySpan<float>().CopyTo(rows.AsSpan<float>());
        using Tensor bytes = new(rows.Shape, DType.I8);
        using Tensor scales = new(new TensorShape(rowCount), DType.F32);
        backend.QuantizeInt8AttentionRows(bytes, scales, rows, rotation);
        backend.Sync();

        ReadOnlySpan<float> actualScales = scales.AsReadOnlySpan<float>();
        Assert.Equal(expectedScales.Length, actualScales.Length);
        int scaleMisses = 0, firstScale = -1;
        for (int i = 0; i < expectedScales.Length; i++)
            if (BitConverter.SingleToInt32Bits(expectedScales[i]) != BitConverter.SingleToInt32Bits(actualScales[i]))
            {
                if (firstScale < 0) firstScale = i;
                scaleMisses++;
            }
        Assert.True(scaleMisses == 0,
            $"{what}, D {headDim}, {rotation}: {scaleMisses} of {expectedScales.Length} scales differ; row {firstScale} is "
            + $"{(firstScale < 0 ? 0f : actualScales[firstScale]):R}, expected {(firstScale < 0 ? 0f : expectedScales[firstScale]):R}");

        ReadOnlySpan<sbyte> actualBytes = bytes.AsReadOnlySpan<sbyte>();
        Assert.Equal(expectedBytes.Length, actualBytes.Length);
        int byteMisses = 0, firstByte = -1;
        for (int i = 0; i < expectedBytes.Length; i++)
            if (expectedBytes[i] != actualBytes[i])
            {
                if (firstByte < 0) firstByte = i;
                byteMisses++;
            }
        Assert.True(byteMisses == 0,
            $"{what}, D {headDim}, {rotation}: {byteMisses} of {expectedBytes.Length} bytes differ; element {firstByte} is "
            + $"{(firstByte < 0 ? 0 : actualBytes[firstByte])}, expected {(firstByte < 0 ? 0 : expectedBytes[firstByte])}");
    }

    /// <summary>The rows the reference treats specially: one holding a NaN (a NaN scale and zero bytes), one whose largest
    /// magnitude is subnormal (the scale floor of 1, so every byte rounds to zero), one of zeros, one holding an infinity
    /// (an infinite scale and zero bytes), and one whose scale is exactly the smallest normal float, where the floor
    /// does not apply.</summary>
    [VulkanFact]
    public void EdgeRowsFollowTheReference()
    {
        const int heads = 1, queryRows = 8, keys = 8, headDim = 64;
        using Tensor query = Random(new TensorShape(1, heads, queryRows, headDim), 21);
        using Tensor key = Random(new TensorShape(1, heads, keys, headDim), 22);
        using Tensor value = Random(key.Shape, 23);
        using Tensor output = new(query.Shape, DType.F32);
        Span<float> q = query.AsSpan<float>();
        q[5] = float.NaN;                                   // row 0
        q.Slice(headDim, headDim).Fill(1e-40f);             // row 1: subnormal
        q.Slice(2 * headDim, headDim).Clear();              // row 2: zeros
        q[3 * headDim + 7] = float.PositiveInfinity;        // row 3
        q.Slice(4 * headDim, headDim).Fill(127f * 1.17549435e-38f);   // row 4: the scale is the smallest normal float
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, query, key, value, null, 0.125f, false);
        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call,
            new Int8AttentionV2Options { SmoothKeys = false, Rotation = Int8Rotation.None, KeyScales = Int8KeyScaleGrouping.PerKey });
        Assert.True(float.IsNaN(expected.QueryScales[0]));
        Assert.Equal(1f, expected.QueryScales[1]);
        Assert.Equal(1f, expected.QueryScales[2]);
        Assert.True(float.IsPositiveInfinity(expected.QueryScales[3]));
        Assert.Equal(1.17549435e-38f, expected.QueryScales[4]);

        using VulkanBackend backend = new();
        using Tensor rows = new(new TensorShape(queryRows, headDim), DType.F32);
        query.AsReadOnlySpan<float>().CopyTo(rows.AsSpan<float>());
        using Tensor bytes = new(rows.Shape, DType.I8);
        using Tensor scales = new(new TensorShape(queryRows), DType.F32);
        backend.QuantizeInt8AttentionRows(bytes, scales, rows, Int8Rotation.None);
        backend.Sync();
        ReadOnlySpan<float> actualScales = scales.AsReadOnlySpan<float>();
        Assert.True(float.IsNaN(actualScales[0]), $"the NaN row's scale is {actualScales[0]}");
        for (int i = 1; i < queryRows; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected.QueryScales[i]), BitConverter.SingleToInt32Bits(actualScales[i]));
        Assert.True(expected.Queries.AsSpan().SequenceEqual(bytes.AsReadOnlySpan<sbyte>()), "the bytes differ from the reference");
        Assert.Equal(-1, bytes.AsReadOnlySpan<sbyte>().Slice(0, 4 * headDim).IndexOfAnyExcept((sbyte)0));
        Assert.Equal(-1, bytes.AsReadOnlySpan<sbyte>().Slice(4 * headDim, headDim).IndexOfAnyExcept((sbyte)127));
    }

    /// <summary>More rows than one grid dimension is guaranteed to hold: the entry dispatches in pieces, and the rows either
    /// side of each seam are the reference's.</summary>
    [VulkanFact]
    public void MoreRowsThanOneDispatchHolds()
    {
        const int heads = 1, queryRows = 2, keys = 140_000, headDim = 64;
        using Tensor query = Random(new TensorShape(1, heads, queryRows, headDim), 31);
        using Tensor key = Random(new TensorShape(1, heads, keys, headDim), 32);
        using Tensor value = new(key.Shape, DType.F32);
        using Tensor output = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, query, key, value, null, 0.125f, false);
        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call,
            new Int8AttentionV2Options { SmoothKeys = false, Rotation = Int8Rotation.H64, KeyScales = Int8KeyScaleGrouping.PerKey });
        using VulkanBackend backend = new();
        Compare(backend, key, keys, headDim, Int8Rotation.H64, expected.Keys, expected.KeyScales, "key");
    }

    /// <summary>What the entry refuses: the unresolved rotation, a transform that does not run at the width, and outputs of
    /// the wrong type or size.</summary>
    [VulkanFact]
    public void MalformedCallsAreRefused()
    {
        using VulkanBackend backend = new();
        using Tensor rows = Random(new TensorShape(4, 64), 1);
        using Tensor bytes = new(rows.Shape, DType.I8);
        using Tensor scales = new(new TensorShape(4), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.QuantizeInt8AttentionRows(bytes, scales, rows, Int8Rotation.Pinned));
        Assert.Throws<ArgumentException>(() => backend.QuantizeInt8AttentionRows(bytes, scales, rows, Int8Rotation.SignedH128));
        using Tensor floats = new(rows.Shape, DType.F32);
        Assert.Throws<ArgumentException>(() => backend.QuantizeInt8AttentionRows(floats, scales, rows, Int8Rotation.H4));
        using Tensor tooFew = new(new TensorShape(3), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.QuantizeInt8AttentionRows(bytes, tooFew, rows, Int8Rotation.H4));
        using Tensor odd = Random(new TensorShape(4, 96), 2);
        using Tensor oddBytes = new(odd.Shape, DType.I8);
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.QuantizeInt8AttentionRows(oddBytes, scales, odd, Int8Rotation.None));
    }
}

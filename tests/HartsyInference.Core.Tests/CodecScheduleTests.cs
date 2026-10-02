using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tests.MemoryManagement;
using HartsyInference.Tests.Common;
using Xunit;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.Core.Tests;

/// <summary>The host weight codecs in Core fan their rows out through <see cref="CpuParallel"/>. Their output is the
/// same bytes whether the rows ran over every core, under a <c>numerics.cpuThreads</c> cap of 1, or inside an
/// <see cref="CpuParallel.InlineScope"/>, and it matches a reference that shares nothing with the parallel loop, so a
/// row scratch buffer that came back from the pool dirty, or a row written to the wrong place, cannot hide behind
/// every schedule agreeing. Each case is big enough that the default schedule really fans out. Serialized with the
/// other classes that change process-wide knobs.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class CodecScheduleTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(64)]
    public void Int8ConvRot_DequantToBf16_MatchesARowByRowReference_UnderEverySchedule(int groupSize)
    {
        const int rows = 96, columns = 1024;
        Random rng = new(groupSize + 3);
        using Tensor weight = new(new TensorShape(rows, columns), DType.I8);
        rng.NextBytes(weight.AsSpan<byte>());
        using Tensor rowScale = new(new TensorShape(rows), DType.F32);
        Span<float> scales = rowScale.AsSpan<float>();
        for (int row = 0; row < rows; row++) scales[row] = (float)(rng.NextDouble() * 0.02 + 1e-4);
        ushort[] expected = DequantReference(weight, rowScale, groupSize);

        (ushort[] parallel, ushort[] capped, ushort[] inline) =
            UnderEverySchedule(() => Words(Int8ConvRotCodec.DequantToBf16(weight, rowScale, groupSize)));

        Assert.Equal(expected, parallel);
        Assert.Equal(expected, capped);
        Assert.Equal(expected, inline);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(64)]
    public void Int8ConvRot_QuantizeFromF32_GivesTheSameBytesAndScales_UnderEverySchedule(int groupSize)
    {
        const int rows = 96, columns = 1024;
        Random rng = new(groupSize + 5);
        float[] values = new float[rows * columns];
        for (int i = 0; i < values.Length; i++) values[i] = (float)((rng.NextDouble() - 0.5) * 4.0);

        byte[] Quantize()
        {
            // The source is rotated in place, so every run starts from its own copy.
            using Tensor source = new(new TensorShape(rows, columns), DType.F32);
            values.CopyTo(source.AsSpan<float>());
            (Tensor packed, Tensor rowScale) = Int8ConvRotCodec.QuantizeFromF32(source, groupSize);
            using (packed)
            using (rowScale)
            {
                return [.. packed.AsReadOnlySpan<byte>(), .. rowScale.AsReadOnlySpan<byte>()];
            }
        }

        (byte[] parallel, byte[] capped, byte[] inline) = UnderEverySchedule(Quantize);

        Assert.Equal(parallel, capped);
        Assert.Equal(parallel, inline);
    }

    [Fact]
    public void Nvfp4Resident_DequantToBf16_MatchesTheHostReference_UnderEverySchedule()
    {
        const int n = 256, k = 512;
        int paddedCols = k / Nvfp4ResidentCodec.GroupSize;
        byte[] packedBytes = new byte[n * (k / 2)];
        new Random(41).NextBytes(packedBytes);
        byte[] scaleBytes = new byte[n * paddedCols];
        // Every E4M3 byte, as the parity tests do, so subnormals, the 480 maximum and the negative half all show up.
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)(i & 0xFF);
        using Tensor packed = FromBytes(packedBytes, new TensorShape(n, k / 2), DType.U8);
        using Tensor blockScale = FromBytes(scaleBytes, new TensorShape(n, paddedCols), DType.F8E4M3);
        // A power of two, so the reference and the codec agree whichever order they multiply the scales in.
        blockScale.Fp8ScaleFactor = 0.5f;
        using Tensor globalScale = new(new TensorShape(1), DType.F32);
        globalScale.AsSpan<float>()[0] = 0.37f;
        ushort[] expected = Nvfp4HostReference.Bf16Words(packed, blockScale, globalScale);
        using Tensor relabelled = packed.ReinterpretAs(DType.F4E2M1, new TensorShape(n, k));

        (ushort[] parallel, ushort[] capped, ushort[] inline) =
            UnderEverySchedule(() => Words(Nvfp4ResidentCodec.DequantToBf16(relabelled, blockScale, globalScale)));

        Assert.Equal(expected, parallel);
        Assert.Equal(expected, capped);
        Assert.Equal(expected, inline);
    }

    /// <summary>The dequant spelled out a row at a time with a fresh buffer per row: scale, un-rotate, round to BF16.</summary>
    private static ushort[] DequantReference(Tensor weight, Tensor rowScale, int groupSize)
    {
        int rows = (int)weight.Shape[0], columns = (int)weight.Shape[1];
        ReadOnlySpan<sbyte> source = weight.AsReadOnlySpan<sbyte>();
        ReadOnlySpan<float> scales = rowScale.AsReadOnlySpan<float>();
        ushort[] words = new ushort[rows * columns];
        for (int row = 0; row < rows; row++)
        {
            float[] values = new float[columns];
            for (int column = 0; column < columns; column++) values[column] = source[row * columns + column] * scales[row];
            if (groupSize > 0) Int8ConvRotCodec.ApplyRotationInPlace(values, groupSize);
            for (int column = 0; column < columns; column++) words[row * columns + column] = TensorCasts.F32ToBf16Bits(values[column]);
        }
        return words;
    }

    private static Tensor FromBytes(byte[] bytes, TensorShape shape, DType dtype)
    {
        Tensor tensor = new(shape, dtype);
        bytes.CopyTo(tensor.AsSpan<byte>());
        return tensor;
    }

    private static ushort[] Words(Tensor tensor)
    {
        using (tensor)
        {
            return tensor.AsReadOnlySpan<ushort>().ToArray();
        }
    }
}

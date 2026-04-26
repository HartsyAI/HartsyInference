using SharpInference.Core.Backends;
using SharpInference.Core.Exceptions;
using SharpInference.Core.Tensors;
using Xunit;

namespace SharpInference.Core.Tests;

/// <summary>Tests for <see cref="Tensor"/>.</summary>
public sealed unsafe class TensorTests
{
    [Fact]
    public void Create_F32_HasCorrectProperties()
    {
        TensorShape shape = new TensorShape(4, 8);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Assert.Equal(2, tensor.Shape.Rank);
        Assert.Equal(4, tensor.Shape[0]);
        Assert.Equal(8, tensor.Shape[1]);
        Assert.Equal(DType.F32, tensor.DType);
        Assert.Equal(DeviceKind.Cpu, tensor.Device);
        Assert.Equal(32, tensor.ElementCount);
        Assert.True(tensor.OwnsMemory);
    }

    [Fact]
    public void Create_ZeroesMemory()
    {
        TensorShape shape = new TensorShape(16);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            Assert.Equal(0.0f, span[i]);
        }
    }

    [Fact]
    public void AsSpan_ReturnsCorrectLength()
    {
        TensorShape shape = new TensorShape(3, 4);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> span = tensor.AsSpan<float>();

        Assert.Equal(12, span.Length);
    }

    [Fact]
    public void AsSpan_WriteAndRead()
    {
        TensorShape shape = new TensorShape(4);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> writeSpan = tensor.AsSpan<float>();
        writeSpan[0] = 1.0f;
        writeSpan[1] = 2.0f;
        writeSpan[2] = 3.0f;
        writeSpan[3] = 4.0f;

        ReadOnlySpan<float> readSpan = tensor.AsReadOnlySpan<float>();
        Assert.Equal(1.0f, readSpan[0]);
        Assert.Equal(2.0f, readSpan[1]);
        Assert.Equal(3.0f, readSpan[2]);
        Assert.Equal(4.0f, readSpan[3]);
    }

    [Fact]
    public void Reshape_SameElementCount_Succeeds()
    {
        TensorShape shape = new TensorShape(2, 6);
        using Tensor original = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        TensorShape newShape = new TensorShape(3, 4);
        using Tensor reshaped = original.Reshape(newShape);

        Assert.Equal(3, reshaped.Shape[0]);
        Assert.Equal(4, reshaped.Shape[1]);
        Assert.Equal(2, reshaped.Shape.Rank);
        Assert.Equal(12, reshaped.ElementCount);
    }

    [Fact]
    public void Reshape_DifferentElementCount_Throws()
    {
        TensorShape shape = new TensorShape(2, 3);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        TensorShape badShape = new TensorShape(2, 4);
        SharpInferenceException exception = Assert.Throws<SharpInferenceException>(
            () => tensor.Reshape(badShape));

        Assert.Contains("Cannot reshape", exception.Message);
    }

    [Fact]
    public void To_SameDevice_CreatesDeepCopy()
    {
        TensorShape shape = new TensorShape(4);
        using Tensor original = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = original.AsSpan<float>();
        data[0] = 42.0f;
        data[1] = 99.0f;

        using Tensor copy = original.To(DeviceKind.Cpu);

        ReadOnlySpan<float> copyData = copy.AsReadOnlySpan<float>();
        Assert.Equal(42.0f, copyData[0]);
        Assert.Equal(99.0f, copyData[1]);
        Assert.NotEqual((nint)original.DataPointer, (nint)copy.DataPointer);
    }

    [Fact]
    public void CastTo_F32ToF16_Roundtrips()
    {
        TensorShape shape = new TensorShape(4);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 1.0f;
        data[1] = 0.5f;
        data[2] = -3.25f;
        data[3] = 0.0f;

        using Tensor f16 = f32.CastTo(DType.F16);
        Assert.Equal(DType.F16, f16.DType);

        using Tensor roundtripped = f16.CastTo(DType.F32);
        Assert.Equal(DType.F32, roundtripped.DType);

        ReadOnlySpan<float> result = roundtripped.AsReadOnlySpan<float>();
        Assert.Equal(1.0f, result[0], 0.01f);
        Assert.Equal(0.5f, result[1], 0.01f);
        Assert.Equal(-3.25f, result[2], 0.01f);
        Assert.Equal(0.0f, result[3], 0.01f);
    }

    // ── FP8 E4M3 Tests ──────────────────────────────────────────────────

    [Fact]
    public void CastTo_F32ToF8E4M3_Roundtrips()
    {
        TensorShape shape = new TensorShape(6);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 1.0f;
        data[1] = 0.5f;
        data[2] = -3.0f;
        data[3] = 0.0f;
        data[4] = 448.0f;  // max E4M3 value
        data[5] = -0.125f;

        using Tensor f8 = f32.CastTo(DType.F8E4M3);
        Assert.Equal(DType.F8E4M3, f8.DType);
        Assert.Equal(6, f8.ElementCount);

        using Tensor roundtripped = f8.CastTo(DType.F32);
        Assert.Equal(DType.F32, roundtripped.DType);

        ReadOnlySpan<float> result = roundtripped.AsReadOnlySpan<float>();
        Assert.Equal(1.0f, result[0], 0.1f);
        Assert.Equal(0.5f, result[1], 0.1f);
        Assert.Equal(-3.0f, result[2], 0.5f);
        Assert.Equal(0.0f, result[3]);
        Assert.Equal(448.0f, result[4], 1.0f);
        Assert.Equal(-0.125f, result[5], 0.05f);
    }

    [Fact]
    public void CastTo_F8E4M3ToF16_Roundtrips()
    {
        TensorShape shape = new TensorShape(4);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 2.0f;
        data[1] = -1.5f;
        data[2] = 0.25f;
        data[3] = 64.0f;

        using Tensor f8 = f32.CastTo(DType.F8E4M3);
        using Tensor f16 = f8.CastTo(DType.F16);
        Assert.Equal(DType.F16, f16.DType);

        using Tensor backToF32 = f16.CastTo(DType.F32);
        ReadOnlySpan<float> result = backToF32.AsReadOnlySpan<float>();
        Assert.Equal(2.0f, result[0], 0.1f);
        Assert.Equal(-1.5f, result[1], 0.1f);
        Assert.Equal(0.25f, result[2], 0.05f);
        Assert.Equal(64.0f, result[3], 1.0f);
    }

    [Fact]
    public void CastTo_F16ToF8E4M3_Roundtrips()
    {
        TensorShape shape = new TensorShape(3);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 1.0f;
        data[1] = -4.0f;
        data[2] = 0.0f;

        using Tensor f16 = f32.CastTo(DType.F16);
        using Tensor f8 = f16.CastTo(DType.F8E4M3);
        Assert.Equal(DType.F8E4M3, f8.DType);

        using Tensor backToF32 = f8.CastTo(DType.F32);
        ReadOnlySpan<float> result = backToF32.AsReadOnlySpan<float>();
        Assert.Equal(1.0f, result[0], 0.1f);
        Assert.Equal(-4.0f, result[1], 0.5f);
        Assert.Equal(0.0f, result[2]);
    }

    [Fact]
    public void CastTo_F8E4M3_SaturatesAboveMax()
    {
        TensorShape shape = new TensorShape(2);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 1000.0f;   // above max (448)
        data[1] = -999.0f;   // below min (-448)

        using Tensor f8 = f32.CastTo(DType.F8E4M3);
        using Tensor backToF32 = f8.CastTo(DType.F32);
        ReadOnlySpan<float> result = backToF32.AsReadOnlySpan<float>();

        Assert.True(result[0] <= 448.0f);
        Assert.True(result[1] >= -448.0f);
    }

    [Fact]
    public void CastTo_F8E4M3_SubnormalsPreserved()
    {
        // E4M3 smallest subnormal = 2^-9 = 0.001953125
        TensorShape shape = new TensorShape(3);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 0.001953125f;  // 2^-9 (smallest subnormal, mant=1)
        data[1] = 0.013671875f;  // 2^-9 * 7 (largest subnormal, mant=7)
        data[2] = 0.00390625f;   // 2^-8 (subnormal, mant=2)

        using Tensor f8 = f32.CastTo(DType.F8E4M3);
        using Tensor backToF32 = f8.CastTo(DType.F32);
        ReadOnlySpan<float> result = backToF32.AsReadOnlySpan<float>();

        Assert.Equal(0.001953125f, result[0], 0.001f);
        Assert.Equal(0.013671875f, result[1], 0.002f);
        Assert.Equal(0.00390625f, result[2], 0.001f);
    }

    [Fact]
    public void CastTo_BF16ToF8E4M3_Roundtrips()
    {
        TensorShape shape = new TensorShape(3);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 1.0f;
        data[1] = -2.0f;
        data[2] = 0.5f;

        using Tensor bf16 = f32.CastTo(DType.BF16);
        using Tensor f8 = bf16.CastTo(DType.F8E4M3);
        Assert.Equal(DType.F8E4M3, f8.DType);

        using Tensor backToF32 = f8.CastTo(DType.F32);
        ReadOnlySpan<float> result = backToF32.AsReadOnlySpan<float>();
        Assert.Equal(1.0f, result[0], 0.1f);
        Assert.Equal(-2.0f, result[1], 0.1f);
        Assert.Equal(0.5f, result[2], 0.1f);
    }

    [Fact]
    public void CastTo_F8E4M3ToBF16_Roundtrips()
    {
        TensorShape shape = new TensorShape(3);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 8.0f;
        data[1] = -0.5f;
        data[2] = 0.0f;

        using Tensor f8 = f32.CastTo(DType.F8E4M3);
        using Tensor bf16 = f8.CastTo(DType.BF16);
        Assert.Equal(DType.BF16, bf16.DType);

        using Tensor backToF32 = bf16.CastTo(DType.F32);
        ReadOnlySpan<float> result = backToF32.AsReadOnlySpan<float>();
        Assert.Equal(8.0f, result[0], 0.1f);
        Assert.Equal(-0.5f, result[1], 0.1f);
        Assert.Equal(0.0f, result[2]);
    }

    // ── FP8 E5M2 Tests ──────────────────────────────────────────────────

    [Fact]
    public void CastTo_F32ToF8E5M2_Roundtrips()
    {
        TensorShape shape = new TensorShape(4);
        using Tensor f32 = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = f32.AsSpan<float>();
        data[0] = 1.0f;
        data[1] = -2.0f;
        data[2] = 0.0f;
        data[3] = 16.0f;

        using Tensor f8 = f32.CastTo(DType.F8E5M2);
        Assert.Equal(DType.F8E5M2, f8.DType);

        using Tensor roundtripped = f8.CastTo(DType.F32);
        ReadOnlySpan<float> result = roundtripped.AsReadOnlySpan<float>();
        Assert.Equal(1.0f, result[0], 0.5f);
        Assert.Equal(-2.0f, result[1], 0.5f);
        Assert.Equal(0.0f, result[2]);
        Assert.Equal(16.0f, result[3], 2.0f);
    }

    // ── FP8 DType Properties ─────────────────────────────────────────────

    [Fact]
    public void DType_F8E4M3_Properties()
    {
        Assert.Equal(1, DType.F8E4M3.SizeInBytes);
        Assert.False(DType.F8E4M3.IsQuantized);
        Assert.True(DType.F8E4M3.IsFloatingPoint);
        Assert.True(DType.F8E4M3.IsFp8);
        Assert.Equal("F8_E4M3", DType.F8E4M3.Name);
    }

    [Fact]
    public void DType_F8E5M2_Properties()
    {
        Assert.Equal(1, DType.F8E5M2.SizeInBytes);
        Assert.False(DType.F8E5M2.IsQuantized);
        Assert.True(DType.F8E5M2.IsFloatingPoint);
        Assert.True(DType.F8E5M2.IsFp8);
        Assert.Equal("F8_E5M2", DType.F8E5M2.Name);
    }

    [Fact]
    public void DType_F8E4M3_ByteCount()
    {
        long byteCount = DType.F8E4M3.ComputeByteCount(1024);
        Assert.Equal(1024, byteCount);  // 1 byte per element
    }

    [Fact]
    public void CastTo_F8E4M3_MemorySize()
    {
        TensorShape shape = new TensorShape(256);
        using Tensor f8 = new Tensor(shape, DType.F8E4M3, DeviceKind.Cpu);
        long expected = 256 * 1;  // 1 byte per element
        Assert.Equal(expected, Tensor.ComputeByteSize(shape, DType.F8E4M3));
    }

    [Fact]
    public void CastTo_QuantizedType_Throws()
    {
        TensorShape shape = new TensorShape(32);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Assert.Throws<SharpInferenceException>(() => tensor.CastTo(DType.Q8_0));
    }

    [Fact]
    public void Dispose_SetsPointerToZero()
    {
        Tensor tensor = new Tensor(new TensorShape(4), DType.F32, DeviceKind.Cpu);
        tensor.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            void* _ = tensor.DataPointer;
        });
    }

    [Fact]
    public void Dispose_Idempotent()
    {
        Tensor tensor = new Tensor(new TensorShape(4), DType.F32, DeviceKind.Cpu);

        Exception? exception = Record.Exception(() =>
        {
            tensor.Dispose();
            tensor.Dispose();
        });

        Assert.Null(exception);
    }

    [Fact]
    public void BorrowedTensor_DoesNotFreeMemory()
    {
        TensorShape shape = new TensorShape(4);
        using Tensor owned = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Span<float> data = owned.AsSpan<float>();
        data[0] = 123.0f;
        data[1] = 456.0f;

        Tensor borrowed = new Tensor(owned.DataPointer, shape, DType.F32, DeviceKind.Cpu);
        Assert.False(borrowed.OwnsMemory);

        borrowed.Dispose();

        ReadOnlySpan<float> ownerData = owned.AsReadOnlySpan<float>();
        Assert.Equal(123.0f, ownerData[0]);
        Assert.Equal(456.0f, ownerData[1]);
    }
}

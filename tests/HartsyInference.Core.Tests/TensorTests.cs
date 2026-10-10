using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Tests for <see cref="Tensor"/>.</summary>
public sealed unsafe class TensorTests
{
    [Fact]
    public void Create_F32_HasCorrectPropertiesAndZeroFills()
    {
        TensorShape shape = new TensorShape(4, 8);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Assert.Equal(2, tensor.Shape.Rank);
        Assert.Equal(DType.F32, tensor.DType);
        Assert.Equal(DeviceKind.Cpu, tensor.Device);
        Assert.Equal(32, tensor.ElementCount);
        Assert.True(tensor.OwnsMemory);

        foreach (float value in tensor.AsReadOnlySpan<float>())
        {
            Assert.Equal(0.0f, value);
        }
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
        HartsyInferenceException exception = Assert.Throws<HartsyInferenceException>(
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

    // ── FP8 E4M3 (representative FP8 cast semantics) ────────────────────

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
    public void CastTo_QuantizedType_Throws()
    {
        TensorShape shape = new TensorShape(32);
        using Tensor tensor = new Tensor(shape, DType.F32, DeviceKind.Cpu);

        Assert.Throws<HartsyInferenceException>(() => tensor.CastTo(DType.Q8_0));
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

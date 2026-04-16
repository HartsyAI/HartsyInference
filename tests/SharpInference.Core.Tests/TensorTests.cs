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

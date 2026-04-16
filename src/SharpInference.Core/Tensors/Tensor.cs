using SharpInference.Core.Backends;
using SharpInference.Core.Memory;

namespace SharpInference.Core.Tensors;

/// <summary>Core tensor type holding a pointer to unmanaged memory, shape, dtype, and device.</summary>
public sealed unsafe class Tensor : IDisposable
{
    private NativeBuffer? _ownedBuffer;
    private nint _dataPointer;

    /// <summary>Shape and strides of this tensor.</summary>
    public TensorShape Shape { get; }

    /// <summary>Data type of the tensor elements.</summary>
    public DType DType { get; }

    /// <summary>Device this tensor resides on.</summary>
    public DeviceKind Device { get; }

    /// <summary>Total number of elements across all dimensions.</summary>
    public long ElementCount => Shape.ElementCount;

    /// <summary>Whether this tensor owns its memory or borrows it from a mmap/view.</summary>
    public bool OwnsMemory => _ownedBuffer is not null;

    /// <summary>Pointer to the raw tensor data.</summary>
    public void* DataPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            nint ptr = _dataPointer;
            if (ptr == 0)
                throw new ObjectDisposedException(nameof(Tensor));
            return (void*)ptr;
        }
    }

    /// <summary>Creates a new tensor with freshly allocated unmanaged memory, zeroed.</summary>
    public Tensor(TensorShape shape, DType dtype, DeviceKind device = default)
    {
        Shape = shape;
        DType = dtype;
        Device = device;

        long byteSize = ComputeByteSize(shape, dtype);
        _ownedBuffer = new NativeBuffer((nuint)byteSize);
        _dataPointer = (nint)_ownedBuffer.Pointer;
    }

    /// <summary>Creates a tensor that borrows memory from an external pointer (e.g., mmap'd weights).</summary>
    public Tensor(void* dataPointer, TensorShape shape, DType dtype, DeviceKind device = default)
    {
        _dataPointer = (nint)dataPointer;
        Shape = shape;
        DType = dtype;
        Device = device;
        _ownedBuffer = null;
    }

    /// <summary>Returns a span over the tensor data interpreted as <typeparamref name="T"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        nint ptr = _dataPointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(Tensor));
        int count = (int)(ComputeByteSize(Shape, DType) / sizeof(T));
        return new Span<T>((void*)ptr, count);
    }

    /// <summary>Returns a read-only span over the tensor data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        nint ptr = _dataPointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(Tensor));
        int count = (int)(ComputeByteSize(Shape, DType) / sizeof(T));
        return new ReadOnlySpan<T>((void*)ptr, count);
    }

    /// <summary>Creates a view with a different shape but same underlying data. No copy.</summary>
    public Tensor Reshape(TensorShape newShape)
    {
        void* ptr = DataPointer;

        if (newShape.ElementCount != Shape.ElementCount)
            throw new SharpInference.Core.Exceptions.SharpInferenceException(
                $"Cannot reshape {Shape} ({Shape.ElementCount} elements) to {newShape} ({newShape.ElementCount} elements).");

        return new Tensor(ptr, newShape, DType, Device);
    }

    /// <summary>Creates a contiguous copy on the specified device. Cross-device requires IBackend.CopyTo.</summary>
    public Tensor To(DeviceKind targetDevice)
    {
        void* ptr = DataPointer;

        if (targetDevice == Device)
        {
            Tensor copy = new Tensor(Shape, DType, Device);
            long byteSize = ComputeByteSize(Shape, DType);
            Buffer.MemoryCopy(ptr, copy.DataPointer, byteSize, byteSize);
            return copy;
        }

        throw new SharpInference.Core.Exceptions.SharpInferenceException(
            $"Direct tensor copy from {Device} to {targetDevice} is not supported. Use IBackend.CopyTo for cross-device transfers.");
    }

    /// <summary>Creates a copy cast to the specified dtype. Quantized types require GgufDequantizer.</summary>
    public Tensor CastTo(DType targetDtype)
    {
        void* ptr = DataPointer;

        if (targetDtype == DType)
            return To(Device);

        if (DType.IsQuantized() || targetDtype.IsQuantized())
            throw new SharpInference.Core.Exceptions.SharpInferenceException(
                $"Quantized dtype conversion ({DType} → {targetDtype}) requires a dedicated dequantizer. Use GgufDequantizer instead.");

        Tensor result = new Tensor(Shape, targetDtype, Device);
        try
        {
            long count = Shape.ElementCount;

            if (DType == DType.F32 && targetDtype == DType.F16)
            {
                ReadOnlySpan<float> src = new ReadOnlySpan<float>(ptr, (int)count);
                Span<Half> dst = new Span<Half>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = (Half)src[i];
            }
            else if (DType == DType.F16 && targetDtype == DType.F32)
            {
                ReadOnlySpan<Half> src = new ReadOnlySpan<Half>(ptr, (int)count);
                Span<float> dst = new Span<float>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = (float)src[i];
            }
            else if (DType == DType.F32 && targetDtype == DType.BF16)
            {
                ReadOnlySpan<float> src = new ReadOnlySpan<float>(ptr, (int)count);
                Span<ushort> dst = new Span<ushort>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                {
                    // BF16: truncate lower 16 bits of F32
                    uint bits = BitConverter.SingleToUInt32Bits(src[i]);
                    dst[i] = (ushort)(bits >> 16);
                }
            }
            else if (DType == DType.BF16 && targetDtype == DType.F32)
            {
                ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(ptr, (int)count);
                Span<float> dst = new Span<float>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
            }
            else
            {
                throw new SharpInference.Core.Exceptions.SharpInferenceException(
                    $"Unsupported dtype conversion: {DType} → {targetDtype}.");
            }

            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    /// <summary>Computes the total byte size for a tensor of the given shape and dtype.</summary>
    public static long ComputeByteSize(TensorShape shape, DType dtype)
    {
        if (dtype.IsQuantized())
        {
            int blockSize = dtype.BlockSize();
            int blockByteSize = dtype.BlockByteSize();
            long numBlocks = (shape.ElementCount + blockSize - 1) / blockSize;
            return numBlocks * blockByteSize;
        }

        return shape.ElementCount * dtype.ElementSize();
    }

    /// <summary>Frees owned memory via atomic pointer exchange. Borrowed tensors are no-ops.</summary>
    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _dataPointer, 0);
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (ptr != 0 && buffer is not null)
        {
            buffer.Dispose();
        }
        GC.SuppressFinalize(this);
    }

    ~Tensor()
    {
        nint ptr = Interlocked.Exchange(ref _dataPointer, 0);
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (ptr != 0 && buffer is not null)
        {
            buffer.Dispose();
        }
    }
}

using SharpInference.Core.Backends;

namespace SharpInference.Core.Tensors;

/// <summary>Non-owning view into a Tensor's data. Dispose() is a no-op — does not free underlying memory. Safe to pass through IDisposable-expecting APIs.</summary>
public sealed unsafe class TensorView : IDisposable
{
    private readonly nint _dataPointer;

    /// <summary>Pointer to the start of this view's data (offset from the source tensor).</summary>
    public void* DataPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => (void*)_dataPointer;
    }

    /// <summary>Shape of this view (may differ from the source tensor).</summary>
    public TensorShape Shape { get; }

    /// <summary>Data type (same as source tensor).</summary>
    public DType DType { get; }

    /// <summary>Device this view's data resides on.</summary>
    public DeviceKind Device { get; }

    /// <summary>Creates a view into existing tensor data.</summary>
    public TensorView(void* dataPointer, TensorShape shape, DType dtype, DeviceKind device = default)
    {
        _dataPointer = (nint)dataPointer;
        Shape = shape;
        DType = dtype;
        Device = device;
    }

    /// <summary>Returns a span over the view's data interpreted as <typeparamref name="T"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new Span<T>((void*)_dataPointer, count);
    }

    /// <summary>Returns a read-only span over the view's data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new ReadOnlySpan<T>((void*)_dataPointer, count);
    }

    /// <summary>Creates a zero-alloc TensorRef from this view.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef AsRef() => new(_dataPointer, Shape, DType, Device);

    /// <summary>Creates a sub-view by slicing along the first dimension.</summary>
    public TensorView Slice(long start, long length)
    {
        if (Shape.Rank == 0)
            throw new InvalidOperationException("Cannot slice a scalar tensor.");

        if (start < 0 || start + length > Shape[0])
            throw new ArgumentOutOfRangeException(nameof(start), $"Slice [{start}..{start + length}) out of range for dimension 0 with size {Shape[0]}.");

        long stride0 = Shape.Stride(0);
        long byteOffset;
        if (DType.IsQuantized)
        {
            long elemOffset = start * stride0;
            byteOffset = (elemOffset / DType.BlockElementCount) * DType.BlockByteSize;
        }
        else
        {
            byteOffset = start * stride0 * DType.SizeInBytes;
        }

        // Build sliced shape: replace dim 0 with length, keep remaining dims
        Span<long> newDims = stackalloc long[Shape.Rank];
        Shape.CopyDimsTo(newDims);
        newDims[0] = length;
        TensorShape newShape = new TensorShape(newDims);

        return new TensorView((byte*)_dataPointer + byteOffset, newShape, DType, Device);
    }

    /// <summary>No-op disposal — TensorView does not own its memory.</summary>
    public void Dispose()
    {
        // Intentionally empty — non-owning view
    }
}

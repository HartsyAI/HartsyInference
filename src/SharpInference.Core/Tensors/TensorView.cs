namespace SharpInference.Core.Tensors;

/// <summary>Non-owning, stack-only view into a Tensor's data with adjusted offset and shape. Zero allocation.</summary>
public readonly unsafe ref struct TensorView
{
    /// <summary>Pointer to the start of this view's data (offset from the source tensor).</summary>
    public void* DataPointer { get; }

    /// <summary>Shape of this view (may differ from the source tensor).</summary>
    public TensorShape Shape { get; }

    /// <summary>Data type (same as source tensor).</summary>
    public DType DType { get; }

    /// <summary>Creates a view into existing tensor data.</summary>
    public TensorView(void* dataPointer, TensorShape shape, DType dtype)
    {
        DataPointer = dataPointer;
        Shape = shape;
        DType = dtype;
    }

    /// <summary>Returns a span over the view's data interpreted as <typeparamref name="T"/>.</summary>
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        int count = (int)(Tensor.ComputeByteSize(Shape, DType) / sizeof(T));
        return new Span<T>(DataPointer, count);
    }

    /// <summary>Returns a read-only span over the view's data.</summary>
    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        int count = (int)(Tensor.ComputeByteSize(Shape, DType) / sizeof(T));
        return new ReadOnlySpan<T>(DataPointer, count);
    }

    /// <summary>Creates a sub-view by slicing along the first dimension.</summary>
    public TensorView Slice(long start, long length)
    {
        if (Shape.Rank == 0)
            throw new InvalidOperationException("Cannot slice a scalar tensor.");

        if (start < 0 || start + length > Shape[0])
            throw new ArgumentOutOfRangeException(nameof(start), $"Slice [{start}..{start + length}) out of range for dimension 0 with size {Shape[0]}.");

        long stride0 = Shape.Stride(0);
        long byteOffset;
        if (DType.IsQuantized())
        {
            long elemOffset = start * stride0;
            int blockSize = DType.BlockSize();
            int blockByteSize = DType.BlockByteSize();
            byteOffset = (elemOffset / blockSize) * blockByteSize;
        }
        else
        {
            byteOffset = start * stride0 * DType.ElementSize();
        }

        // Build sliced shape: replace dim 0 with length, keep remaining dims
        Span<long> newDims = stackalloc long[Shape.Rank];
        Shape.CopyDimsTo(newDims);
        newDims[0] = length;
        TensorShape newShape = new TensorShape(newDims);

        return new TensorView((byte*)DataPointer + byteOffset, newShape, DType);
    }
}

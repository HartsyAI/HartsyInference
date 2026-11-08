using HartsyInference.Core.Backends.Layout;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>A validated <see cref="StridedCopySpec"/> with its axes collapsed: adjacent axes that both views walk
/// contiguously are merged and axes of extent 1 dropped, so a kernel decomposes as few indices per element as the
/// geometry allows. Strides and offsets are in elements of <see cref="ElementSize"/> bytes.</summary>
public readonly struct StridedCopyGeometry
{
    private readonly StridedExtents _dims;
    private readonly StridedExtents _sourceStrides;
    private readonly StridedExtents _destinationStrides;

    internal StridedCopyGeometry(ReadOnlySpan<long> dims, ReadOnlySpan<long> sourceStrides, ReadOnlySpan<long> destinationStrides,
        long sourceOffset, long destinationOffset, int elementSize, bool destinationCoversOutput)
    {
        Rank = dims.Length;
        dims.CopyTo(_dims);
        sourceStrides.CopyTo(_sourceStrides);
        destinationStrides.CopyTo(_destinationStrides);
        SourceOffset = sourceOffset;
        DestinationOffset = destinationOffset;
        ElementSize = elementSize;
        DestinationCoversOutput = destinationCoversOutput;
        long count = 1;
        foreach (long dim in dims) count *= dim;
        ElementCount = count;
    }

    /// <summary>Axes after collapsing, at least 1.</summary>
    public int Rank { get; }

    /// <summary>Elements moved.</summary>
    public long ElementCount { get; }

    /// <summary>Bytes per element: 1, 2, 4 or 8.</summary>
    public int ElementSize { get; }

    /// <summary>True when the destination is the whole output in row-major order, so a backend may allocate a fresh
    /// output buffer instead of preserving the output's other elements.</summary>
    public bool DestinationCoversOutput { get; }

    /// <summary>Element offset of the first source element.</summary>
    public long SourceOffset { get; }

    /// <summary>Element offset of the first destination element.</summary>
    public long DestinationOffset { get; }

    /// <summary>Collapsed extent of one axis, outermost first.</summary>
    public long Dim(int axis) => _dims[axis];

    /// <summary>Source stride of one collapsed axis.</summary>
    public long SourceStride(int axis) => _sourceStrides[axis];

    /// <summary>Destination stride of one collapsed axis.</summary>
    public long DestinationStride(int axis) => _destinationStrides[axis];
}

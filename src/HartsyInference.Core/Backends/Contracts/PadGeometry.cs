using HartsyInference.Core.Backends.Layout;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>A validated <see cref="PadSpec"/> against its tensors: output and input extents per axis, the leading pad
/// per axis, the fill mode and the element width.</summary>
public readonly struct PadGeometry
{
    private readonly StridedExtents _outputDims;
    private readonly StridedExtents _inputDims;
    private readonly StridedExtents _before;

    internal PadGeometry(ReadOnlySpan<long> outputDims, ReadOnlySpan<long> inputDims, ReadOnlySpan<long> before, PadMode mode, int elementSize)
    {
        Rank = outputDims.Length;
        outputDims.CopyTo(_outputDims);
        inputDims.CopyTo(_inputDims);
        before.CopyTo(_before);
        Mode = mode;
        ElementSize = elementSize;
        long count = 1;
        foreach (long dim in outputDims) count *= dim;
        ElementCount = count;
    }

    /// <summary>Number of axes.</summary>
    public int Rank { get; }

    /// <summary>Elements of the output.</summary>
    public long ElementCount { get; }

    /// <summary>Bytes per element: 1, 2, 4 or 8.</summary>
    public int ElementSize { get; }

    /// <summary>How the padded positions are filled.</summary>
    public PadMode Mode { get; }

    /// <summary>Output extent of one axis.</summary>
    public long OutputDim(int axis) => _outputDims[axis];

    /// <summary>Input extent of one axis.</summary>
    public long InputDim(int axis) => _inputDims[axis];

    /// <summary>Positions before the input on one axis.</summary>
    public long Before(int axis) => _before[axis];
}

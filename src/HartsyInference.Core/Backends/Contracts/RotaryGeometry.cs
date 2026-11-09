using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>A validated <see cref="RotarySpec"/> as a kernel runs it: the six index axes <c>[B, H, S1, S2, S3, P]</c>
/// with each view's strides, the stride from a pair's first element to its second, the offsets and the device
/// position terms. Strides and offsets are in elements.</summary>
public readonly struct RotaryGeometry
{
    private readonly StridedExtents _dims;
    private readonly StridedExtents _x;
    private readonly StridedExtents _cos;
    private readonly StridedExtents _sin;

    internal RotaryGeometry(in RotarySpec spec, int elementSize)
    {
        for (int axis = 0; axis < IndexAxes; axis++)
        {
            _dims[axis] = spec.X[axis];
            _x[axis] = spec.X.Stride(axis);
            _cos[axis] = spec.Cos.Stride(axis);
            _sin[axis] = spec.Sin.Stride(axis);
        }
        XSecond = spec.X.Stride(IndexAxes);
        CosSecond = spec.Cos.Stride(IndexAxes);
        SinSecond = spec.Sin.Stride(IndexAxes);
        XOffset = spec.X.Offset;
        CosOffset = spec.Cos.Offset;
        SinOffset = spec.Sin.Offset;
        DevicePosition = spec.DevicePosition;
        PositionSlot = spec.PositionSlot;
        PositionRowStride = spec.PositionRowStride;
        ElementSize = elementSize;
        PairCount = spec.PairCount;
    }

    /// <summary>Batch, head, three sequence axes and the pair axis.</summary>
    public const int IndexAxes = 6;

    /// <summary>Pairs rotated.</summary>
    public long PairCount { get; }

    /// <summary>Bytes per activation element: 4 (F32) or 2 (F16, BF16).</summary>
    public int ElementSize { get; }

    /// <summary>Stride from a pair's first activation element to its second.</summary>
    public long XSecond { get; }

    /// <summary>Stride from the first element's cosine to the second's (0 shares one entry).</summary>
    public long CosSecond { get; }

    /// <summary>Stride from the first element's sine to the second's.</summary>
    public long SinSecond { get; }

    public long XOffset { get; }

    public long CosOffset { get; }

    public long SinOffset { get; }

    /// <summary>Device position buffer, or 0.</summary>
    public ulong DevicePosition { get; }

    public int PositionSlot { get; }

    /// <summary>Table elements per position row when <see cref="DevicePosition"/> is set.</summary>
    public long PositionRowStride { get; }

    /// <summary>Extent of one index axis.</summary>
    public long Dim(int axis) => _dims[axis];

    public long XStride(int axis) => _x[axis];

    public long CosStride(int axis) => _cos[axis];

    public long SinStride(int axis) => _sin[axis];
}

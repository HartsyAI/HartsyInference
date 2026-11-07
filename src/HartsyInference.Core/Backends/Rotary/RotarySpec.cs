using HartsyInference.Core.Backends.Layout;

namespace HartsyInference.Core.Backends.Rotary;

/// <summary>The geometry of one rotation (<see cref="IBackend.Rotary"/>): three views over the canonical axes
/// <c>[B, H, S1, S2, S3, P, 2]</c> (batch, head, up to three sequence axes, pair, element of the pair). The activation
/// view says which two elements each pair holds; the table views say which cos and sin entries each element reads
/// (a broadcast stride reuses one entry for both). Build it with <see cref="RotaryViews"/>. Contract:
/// <c>docs/BACKEND_OPS.md</c>.</summary>
/// <param name="X">The rotated elements: <c>x[b,h,s,p,0]</c> and <c>x[b,h,s,p,1]</c> are one pair.</param>
/// <param name="Cos">The cosine each element reads; same extents as <paramref name="X"/>.</param>
/// <param name="Sin">The sine each element reads; same extents as <paramref name="X"/>.</param>
/// <param name="DevicePosition">A device buffer of <c>int</c> positions, or 0. When set, the table offsets grow by
/// <c>position[PositionSlot] · PositionRowStride</c> at run time, so a captured graph reads the step's row.</param>
/// <param name="PositionSlot">Index of the position within the device buffer (1 is the decode step's query offset).</param>
/// <param name="PositionRowStride">Table elements between consecutive positions' rows.</param>
public readonly record struct RotarySpec(StridedView X, StridedView Cos, StridedView Sin, ulong DevicePosition = 0, int PositionSlot = 0, long PositionRowStride = 0)
{
    /// <summary>Axes of every view: batch, head, three sequence axes, pair, pair element.</summary>
    public const int Rank = 7;

    /// <summary>The same geometry read at the row a device position buffer names.</summary>
    public RotarySpec AtDevicePosition(ulong handle, int slot, long rowStride) => this with { DevicePosition = handle, PositionSlot = slot, PositionRowStride = rowStride };

    /// <summary>Pairs rotated: the product of the batch, head, sequence and pair extents.</summary>
    public long PairCount => X.ElementCount / 2;
}

using System.Runtime.CompilerServices;

namespace HartsyInference.Core.Backends.Layout;

/// <summary>Six inline <see cref="long"/> slots: the extents or strides of a <see cref="StridedView"/>, one per axis up to <see cref="StridedView.MaxRank"/>.</summary>
[InlineArray(StridedView.MaxRank)]
public struct StridedExtents
{
    private long _element0;
}

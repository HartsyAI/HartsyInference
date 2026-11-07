using HartsyInference.Core.Backends.Norm;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The flattened geometry of a validated <see cref="NormModulateCall"/>: <see cref="Rows"/> rows of <see cref="Width"/>
/// elements, each row <c>r</c> starting at <c>(r / Spatial) · Width · Spatial + r % Spatial</c> with its elements
/// <see cref="Spatial"/> apart (1 along the last axis), and the modulation addressing of <see cref="ModulationSpec"/>.</summary>
public readonly record struct NormModulateGeometry(long Rows, int Width, long Spatial, long RowsPerMod, long RowStride, long ScaleOffset, long ShiftOffset)
{
    /// <summary>The first element of row <paramref name="row"/>.</summary>
    public long RowBase(long row) => row / Spatial * Width * Spatial + row % Spatial;

    /// <summary>The modulation row that row <paramref name="row"/> reads without an index: <c>row / RowsPerMod</c>, or 0 when every row shares one.</summary>
    public long ModulationRow(long row) => RowsPerMod == 0 ? 0 : row / RowsPerMod;
}

using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.GridScatter"/> and <see cref="IBackend.GridGather"/>: F32
/// <c>[voxels, channels]</c> features, an F32 grid of <c>channels · Depth · Height · Width</c> elements, an I32 coordinate table of
/// <c>voxels · CoordStride</c> entries. Coordinates are range-checked (and, for the scatter, checked for repeated cells) when their
/// host bytes are current; a device-resident table is the caller's to keep in range and, for the scatter, free of repeats.</summary>
public static class GridContract
{
    public static GridGeometry Validate(Tensor grid, Tensor features, Tensor coords, in GridSpec spec, bool scatter)
    {
        ArgumentNullException.ThrowIfNull(grid);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(coords);
        string op = scatter ? "GridScatter" : "GridGather";
        if (grid.DType != DType.F32 || features.DType != DType.F32)
            throw new NotSupportedException($"{op} takes an F32 grid and features; got {grid.DType} and {features.DType}.");
        if (coords.DType != DType.I32)
            throw new NotSupportedException($"{op} takes I32 coordinates; got {coords.DType}.");
        if (spec.Depth <= 0 || spec.Height <= 0 || spec.Width <= 0)
            throw new ArgumentOutOfRangeException(nameof(spec), spec, $"{op} grid extents must be positive.");
        if (spec.CoordStride < 3 || spec.FirstCoordColumn < 0 || spec.FirstCoordColumn + 3 > spec.CoordStride)
            throw new ArgumentOutOfRangeException(nameof(spec), spec, $"{op} coordinate rows must hold three columns from FirstCoordColumn within CoordStride.");
        if (features.Shape.Rank < 1)
            throw new ArgumentException($"{op} features need at least one dimension.", nameof(features));
        long channels = features.Shape[features.Shape.Rank - 1];
        if (channels <= 0 || channels > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(features), channels, $"{op} channel count must be in [1, {int.MaxValue}].");
        long voxels = features.ElementCount / channels;
        if (voxels > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(features), voxels, $"{op} addresses at most {int.MaxValue} voxels.");
        if (coords.ElementCount != voxels * spec.CoordStride)
            throw new ArgumentException($"{op} coordinates {coords.Shape} must hold {voxels} rows of {spec.CoordStride}; they hold {coords.ElementCount} entries.", nameof(coords));
        long cells = spec.Cells;
        if (grid.ElementCount != channels * cells)
            throw new ArgumentException($"{op} grid {grid.Shape} must hold {channels} planes of {cells} cells; it has {grid.ElementCount} elements.", nameof(grid));
        if (grid.HasOverlappingHostStorageWithoutSync(features))
            throw new ArgumentException($"{op} grid storage must not overlap the features.", nameof(grid));
        CheckCoords(op, coords, spec, (int)voxels, scatter);
        return new GridGeometry((int)voxels, (int)channels, cells);
    }

    /// <summary>The cell of coordinate row <paramref name="voxel"/>.</summary>
    public static unsafe long Cell(int* coords, int voxel, in GridSpec spec)
    {
        int* row = coords + (long)voxel * spec.CoordStride + spec.FirstCoordColumn;
        return ((long)row[0] * spec.Height + row[1]) * spec.Width + row[2];
    }

    private static unsafe void CheckCoords(string op, Tensor coords, in GridSpec spec, int voxels, bool unique)
    {
        if (!coords.HostBytesCurrent) return;
        int* p = (int*)coords.DataPointer;
        HashSet<long>? seen = unique && voxels > 1 ? new HashSet<long>(voxels) : null;
        for (int v = 0; v < voxels; v++)
        {
            int* row = p + (long)v * spec.CoordStride + spec.FirstCoordColumn;
            if (row[0] < 0 || row[0] >= spec.Depth || row[1] < 0 || row[1] >= spec.Height || row[2] < 0 || row[2] >= spec.Width)
                throw new ArgumentOutOfRangeException(nameof(coords), $"({row[0]}, {row[1]}, {row[2]})", $"{op} voxel {v} lies outside the {spec.Depth}×{spec.Height}×{spec.Width} grid.");
            if (seen is not null && !seen.Add(Cell(p, v, spec)))
                throw new ArgumentException($"{op} voxel {v} repeats an earlier voxel's cell; each cell is written at most once.", nameof(coords));
        }
    }
}

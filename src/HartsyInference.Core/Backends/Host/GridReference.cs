using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable forms of the grid contract (<c>docs/BACKEND_OPS.md</c>): the scatter writes each voxel's channels into
/// its cell of every channel plane, leaving the other cells as they were; the gather reads them back. The <see cref="IBackend"/>
/// defaults and the oracles every kernel is checked against.</summary>
public static unsafe class GridReference
{
    /// <summary><c>grid[c, cell(v)] = features[v, c]</c>.</summary>
    public static void Scatter(Tensor grid, Tensor features, Tensor coords, in GridSpec spec)
    {
        GridGeometry g = GridContract.Validate(grid, features, coords, spec, scatter: true);
        float* pGrid = (float*)grid.DataPointer;
        float* pFeat = (float*)features.DataPointer;
        int* pCoords = (int*)coords.DataPointer;
        for (int v = 0; v < g.Voxels; v++)
        {
            long cell = GridContract.Cell(pCoords, v, spec);
            for (int c = 0; c < g.Channels; c++) pGrid[c * g.Cells + cell] = pFeat[(long)v * g.Channels + c];
        }
    }

    /// <summary><c>features[v, c] = grid[c, cell(v)]</c>.</summary>
    public static void Gather(Tensor features, Tensor grid, Tensor coords, in GridSpec spec)
    {
        GridGeometry g = GridContract.Validate(grid, features, coords, spec, scatter: false);
        float* pGrid = (float*)grid.DataPointer;
        float* pFeat = (float*)features.DataPointer;
        int* pCoords = (int*)coords.DataPointer;
        for (int v = 0; v < g.Voxels; v++)
        {
            long cell = GridContract.Cell(pCoords, v, spec);
            for (int c = 0; c < g.Channels; c++) pFeat[(long)v * g.Channels + c] = pGrid[c * g.Cells + cell];
        }
    }
}

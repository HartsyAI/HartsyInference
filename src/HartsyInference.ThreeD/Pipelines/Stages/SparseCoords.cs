namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Coordinate layout helpers for single-sample sparse stages.</summary>
internal static class SparseCoords
{
    /// <summary>Expands rows of <c>(x, y, z)</c> to <c>(0, x, y, z)</c> rows for a single-sample sparse tensor.</summary>
    internal static int[] WithBatch(ReadOnlySpan<int> xyz)
    {
        int n = xyz.Length / 3;
        int[] coords = new int[n * 4];
        for (int i = 0; i < n; i++)
        {
            coords[i * 4 + 1] = xyz[i * 3];
            coords[i * 4 + 2] = xyz[i * 3 + 1];
            coords[i * 4 + 3] = xyz[i * 3 + 2];
        }

        return coords;
    }
}

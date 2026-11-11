namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// Voxel-set rules between the occupancy decode and the SLAT stage (<c>prune_sparse_structure</c> and
/// <c>downsample_sparse_structure</c> of <c>inference_utils.py</c>). Coordinates are interleaved <c>x, y, z</c> integers of one object.
/// </summary>
public static class SamSparseStructure
{
    /// <summary>The voxel count above which the mesh decoder would overflow a 32-bit index (<c>max(int32) / (64 · 768)</c>, rounded down).</summary>
    public const int MaxCoords = 42000;

    /// <summary>
    /// Keeps the voxels with at least one empty cell in their <c>(2d + 1)³</c> neighbourhood, so interior voxels are dropped and a
    /// surface shell remains. Cells outside the bounding box count as empty. The input order is kept; duplicates are kept.
    /// </summary>
    public static int[] Prune(ReadOnlySpan<int> coords, int maxNeighborAxesDist = 1)
    {
        RequireTriples(coords);
        if (maxNeighborAxesDist < 0)
            throw new ArgumentOutOfRangeException(nameof(maxNeighborAxesDist), maxNeighborAxesDist, "The distance must not be negative.");
        if (coords.Length == 0)
            return [];
        int[] min = [int.MaxValue, int.MaxValue, int.MaxValue];
        int[] max = [int.MinValue, int.MinValue, int.MinValue];
        for (int i = 0; i < coords.Length; i += 3)
        {
            for (int a = 0; a < 3; a++)
            {
                min[a] = Math.Min(min[a], coords[i + a]);
                max[a] = Math.Max(max[a], coords[i + a]);
            }
        }

        int dx = max[0] - min[0] + 1, dy = max[1] - min[1] + 1, dz = max[2] - min[2] + 1;
        bool[] occupied = new bool[checked(dx * dy * dz)];
        for (int i = 0; i < coords.Length; i += 3)
            occupied[Index(coords[i] - min[0], coords[i + 1] - min[1], coords[i + 2] - min[2], dy, dz)] = true;

        List<int> kept = new(coords.Length);
        for (int i = 0; i < coords.Length; i += 3)
        {
            int x = coords[i] - min[0], y = coords[i + 1] - min[1], z = coords[i + 2] - min[2];
            if (!IsInterior(occupied, x, y, z, dx, dy, dz, maxNeighborAxesDist))
            {
                kept.Add(coords[i]);
                kept.Add(coords[i + 1]);
                kept.Add(coords[i + 2]);
            }
        }

        return kept.ToArray();
    }

    /// <summary>
    /// Returns the voxels unchanged with factor 1 when there are at most <paramref name="maxCoords"/>; otherwise shrinks the grid about
    /// its centre by <paramref name="factor"/>, rounds half to even, clamps, removes duplicates (sorted) and, if still above the cap, keeps
    /// a random subset chosen by <paramref name="seed"/> (upstream's permutation is not reproducible).
    /// </summary>
    public static (int[] Coords, int Factor) Downsample(ReadOnlySpan<int> coords, int maxCoords = MaxCoords, int factor = 2, int seed = 0)
    {
        RequireTriples(coords);
        if (maxCoords <= 0 || factor <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxCoords), "The cap and the factor must be positive.");
        int count = coords.Length / 3;
        if (count <= maxCoords)
            return (coords.ToArray(), 1);

        float[] min = new float[3], max = new float[3], targetMin = new float[3], target = new float[3];
        int[] clampLow = new int[3], clampHigh = new int[3];
        for (int a = 0; a < 3; a++)
        {
            min[a] = float.MaxValue;
            max[a] = float.MinValue;
        }

        for (int i = 0; i < coords.Length; i += 3)
        {
            for (int a = 0; a < 3; a++)
            {
                min[a] = Math.Min(min[a], coords[i + a]);
                max[a] = Math.Max(max[a], coords[i + a]);
            }
        }

        for (int a = 0; a < 3; a++)
        {
            float original = max[a] - min[a] + 1f;
            target[a] = original / factor;
            float offset = (original - target[a]) / 2f;
            targetMin[a] = min[a] + offset;
            clampLow[a] = (int)targetMin[a];
            clampHigh[a] = (int)(min[a] + offset + target[a] - 1f);
        }

        SortedSet<(int, int, int)> unique = [];
        Span<int> mapped = stackalloc int[3];
        for (int i = 0; i < coords.Length; i += 3)
        {
            for (int a = 0; a < 3; a++)
            {
                float span = max[a] - min[a];
                float normalized = span > 0f ? (coords[i + a] - min[a]) / span : 0f;
                float scaled = MathF.Round(normalized * (target[a] - 1f) + targetMin[a], MidpointRounding.ToEven);
                // torch.clamp order: raise to the lower bound first, then cap at the upper one, so an inverted pair resolves to the upper.
                mapped[a] = Math.Min(Math.Max((int)scaled, clampLow[a]), clampHigh[a]);
            }

            unique.Add((mapped[0], mapped[1], mapped[2]));
        }

        List<(int, int, int)> voxels = [.. unique];
        if (voxels.Count > maxCoords)
        {
            Random random = new(seed);
            for (int i = 0; i < maxCoords; i++)
            {
                int j = random.Next(i, voxels.Count);
                (voxels[i], voxels[j]) = (voxels[j], voxels[i]);
            }

            voxels.RemoveRange(maxCoords, voxels.Count - maxCoords);
        }

        int[] result = new int[voxels.Count * 3];
        for (int i = 0; i < voxels.Count; i++)
        {
            result[i * 3] = voxels[i].Item1;
            result[i * 3 + 1] = voxels[i].Item2;
            result[i * 3 + 2] = voxels[i].Item3;
        }

        return (result, factor);
    }

    private static bool IsInterior(bool[] occupied, int x, int y, int z, int dx, int dy, int dz, int d)
    {
        for (int ox = -d; ox <= d; ox++)
        {
            for (int oy = -d; oy <= d; oy++)
            {
                for (int oz = -d; oz <= d; oz++)
                {
                    int nx = x + ox, ny = y + oy, nz = z + oz;
                    bool inside = nx >= 0 && nx < dx && ny >= 0 && ny < dy && nz >= 0 && nz < dz;
                    if (!inside || !occupied[Index(nx, ny, nz, dy, dz)])
                        return false;
                }
            }
        }

        return true;
    }

    private static int Index(int x, int y, int z, int dy, int dz) => (x * dy + y) * dz + z;

    private static void RequireTriples(ReadOnlySpan<int> coords)
    {
        if (coords.Length % 3 != 0)
            throw new ArgumentException($"Coordinates are x, y, z triples; got {coords.Length} values.", nameof(coords));
    }
}

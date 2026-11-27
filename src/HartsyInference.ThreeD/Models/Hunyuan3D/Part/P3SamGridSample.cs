namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>How <see cref="P3SamGridSample"/> picks the point kept in a voxel that holds several.</summary>
public enum P3SamVoxelPick
{
    /// <summary>The first point of the voxel in hash-sorted order. Deterministic.</summary>
    First,

    /// <summary>A random point of the voxel, as Sonata's <c>GridSample</c> does in its <c>train</c> mode (which Hunyuan3D-Part's <c>sonata.transform.default()</c> uses).</summary>
    Random,
}

/// <summary>Sonata's input transform for P3-SAM (<c>sonata.transform.default()</c>): <c>CenterShift(apply_z)</c>, then <c>GridSample</c> at 0.005 with the FNV
/// hash. Points are quantised to voxels, one point per voxel is kept in hash-sorted order, and every original point remembers its voxel's row.</summary>
public static class P3SamGridSample
{
    /// <summary>Voxel size of <c>GridSample</c> in Hunyuan3D-Part's transform.</summary>
    public const double GridSize = 0.005;

    /// <summary>The selected points and their mapping.</summary>
    /// <param name="Selected">Original index of the point kept for each voxel row, in FNV-hash order.</param>
    /// <param name="Inverse">For each original point, the row of its voxel in <paramref name="Selected"/>.</param>
    /// <param name="GridCoord">Voxel coordinates <c>[rows, 3]</c>, shifted so the smallest is 0.</param>
    /// <param name="Shifted">The centred coordinates <c>[points, 3]</c> (<c>CenterShift</c>): x and y centred on the bounding box, z starting at 0.</param>
    public sealed record Result(int[] Selected, int[] Inverse, int[] GridCoord, double[] Shifted);

    /// <summary>Runs <c>CenterShift</c> and <c>GridSample</c> on <paramref name="coord"/> (<c>[count, 3]</c>). <paramref name="random"/> is used for
    /// <see cref="P3SamVoxelPick.Random"/>.</summary>
    public static Result Run(double[] coord, int count, P3SamVoxelPick pick, Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(coord);
        if (coord.Length != count * 3 || count == 0)
            throw new ArgumentException("Coordinates must be [count, 3] with at least one point.", nameof(coord));
        double[] min = [double.MaxValue, double.MaxValue, double.MaxValue], max = [double.MinValue, double.MinValue, double.MinValue];
        for (int i = 0; i < count; i++)
            for (int a = 0; a < 3; a++)
            {
                min[a] = Math.Min(min[a], coord[i * 3 + a]);
                max[a] = Math.Max(max[a], coord[i * 3 + a]);
            }

        double[] shifted = new double[count * 3];
        double[] shift = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, min[2]];
        for (int i = 0; i < count; i++)
            for (int a = 0; a < 3; a++)
                shifted[i * 3 + a] = coord[i * 3 + a] - shift[a];

        long[] grid = new long[count * 3];
        long[] gridMin = [long.MaxValue, long.MaxValue, long.MaxValue];
        for (int i = 0; i < count; i++)
            for (int a = 0; a < 3; a++)
            {
                grid[i * 3 + a] = (long)Math.Floor(shifted[i * 3 + a] / GridSize);
                gridMin[a] = Math.Min(gridMin[a], grid[i * 3 + a]);
            }

        ulong[] key = new ulong[count];
        for (int i = 0; i < count; i++)
        {
            for (int a = 0; a < 3; a++)
                grid[i * 3 + a] -= gridMin[a];
            key[i] = Fnv(grid[i * 3], grid[i * 3 + 1], grid[i * 3 + 2]);
        }

        int[] order = [.. Enumerable.Range(0, count)];
        Array.Sort(order, (a, b) => key[a] != key[b] ? key[a].CompareTo(key[b]) : a.CompareTo(b));

        // Unique keys in sorted order: group starts and counts.
        List<int> starts = [];
        List<int> counts = [];
        int[] inverse = new int[count];
        for (int s = 0; s < count; s++)
        {
            if (s == 0 || key[order[s]] != key[order[s - 1]])
            {
                starts.Add(s);
                counts.Add(0);
            }

            counts[^1]++;
            inverse[order[s]] = starts.Count - 1;
        }

        int rows = starts.Count, maxCount = counts.Max();
        int[] selected = new int[rows], gridCoord = new int[rows * 3];
        for (int r = 0; r < rows; r++)
        {
            int offset = pick == P3SamVoxelPick.Random ? (random ?? throw new ArgumentNullException(nameof(random))).Next(0, maxCount) % counts[r] : 0;
            selected[r] = order[starts[r] + offset];
            for (int a = 0; a < 3; a++)
                gridCoord[r * 3 + a] = (int)grid[selected[r] * 3 + a];
        }

        return new Result(selected, inverse, gridCoord, shifted);
    }

    // FNV-1a over the three voxel coordinates as unsigned 64-bit integers, in the order and arithmetic of Sonata's fnv_hash_vec (multiply, then xor).
    private static ulong Fnv(long x, long y, long z)
    {
        ulong hash = 14695981039346656037UL;
        hash = unchecked(hash * 1099511628211UL) ^ (ulong)x;
        hash = unchecked(hash * 1099511628211UL) ^ (ulong)y;
        hash = unchecked(hash * 1099511628211UL) ^ (ulong)z;
        return hash;
    }
}

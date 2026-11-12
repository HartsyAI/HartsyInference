namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The corner lattice of the mesh decoder's output (upstream's <c>construct_voxel_grid</c>, <c>cubes_to_verts</c> and <c>sparse_cube2verts</c>): every voxel is a cube whose eight corners are
/// lattice points; a point shared by several cubes holds the mean of what those cubes say about it. The points are listed in lexicographic order of (x, y, z), as <c>torch.unique</c> does.
/// </summary>
public sealed class SamMeshCornerGrid
{
    /// <summary>Corner offsets in the order of upstream's <c>cube_corners</c>: <c>(x, y, z)</c> with x the fastest-changing bit.</summary>
    public static readonly int[][] CornerOffsets =
    [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ];

    private SamMeshCornerGrid(int[] points, int[] cubeCorners, float[] attributes, int channels)
    {
        Points = points;
        CubeCorners = cubeCorners;
        Attributes = attributes;
        Channels = channels;
    }

    /// <summary>The distinct corner points, <c>[V, 3]</c> flat, sorted by (x, y, z).</summary>
    public int[] Points { get; }

    /// <summary>For every cube the index of each of its eight corners in <see cref="Points"/>, <c>[M, 8]</c> flat, in <see cref="CornerOffsets"/> order.</summary>
    public int[] CubeCorners { get; }

    /// <summary>Per point the mean of the values of the cubes that share it, <c>[V, Channels]</c> flat.</summary>
    public float[] Attributes { get; }

    /// <summary>Channels per point.</summary>
    public int Channels { get; }

    /// <summary>Number of corner points.</summary>
    public int PointCount => Points.Length / 3;

    /// <summary>
    /// Builds the lattice of <paramref name="cubes"/> (<c>[M, 3]</c> flat voxel coordinates, no batch column) and averages <paramref name="values"/> (<c>[M, 8, channels]</c> flat: the value each
    /// cube gives to each of its corners) onto the points.
    /// </summary>
    /// <exception cref="ArgumentException">The arrays do not describe the same number of cubes.</exception>
    public static SamMeshCornerGrid Build(int[] cubes, float[] values, int channels)
    {
        ArgumentNullException.ThrowIfNull(cubes);
        ArgumentNullException.ThrowIfNull(values);
        if (cubes.Length % 3 != 0 || channels <= 0)
            throw new ArgumentException("Cube coordinates are triples and the channel count is positive.");
        int m = cubes.Length / 3;
        if (values.Length != (long)m * 8 * channels)
            throw new ArgumentException($"{m} cubes of 8 corners with {channels} channels hold {(long)m * 8 * channels} values; got {values.Length}.", nameof(values));

        // Every corner of every cube as a packed key, then the sorted distinct keys: lexicographic in (x, y, z) because the key packs x highest.
        long[] keys = new long[(long)m * 8];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                int x = cubes[i * 3] + CornerOffsets[j][0], y = cubes[i * 3 + 1] + CornerOffsets[j][1], z = cubes[i * 3 + 2] + CornerOffsets[j][2];
                keys[i * 8 + j] = Pack(x, y, z);
            }
        }

        long[] distinct = [.. keys.Distinct().Order()];
        Dictionary<long, int> index = new(distinct.Length);
        for (int p = 0; p < distinct.Length; p++)
            index[distinct[p]] = p;
        int[] points = new int[distinct.Length * 3];
        for (int p = 0; p < distinct.Length; p++)
            Unpack(distinct[p], out points[p * 3], out points[p * 3 + 1], out points[p * 3 + 2]);

        int[] corners = new int[keys.Length];
        float[] sums = new float[(long)distinct.Length * channels];
        int[] counts = new int[distinct.Length];
        for (long c = 0; c < keys.Length; c++)
        {
            int p = index[keys[c]];
            corners[c] = p;
            counts[p]++;
            for (int k = 0; k < channels; k++)
                sums[(long)p * channels + k] += values[c * channels + k];
        }

        for (int p = 0; p < distinct.Length; p++)
        {
            float inverse = 1f / counts[p];
            for (int k = 0; k < channels; k++)
                sums[(long)p * channels + k] *= inverse;
        }

        return new SamMeshCornerGrid(points, corners, sums, channels);
    }

    // 21 bits per axis: a corner of a 256 grid is at most 256, with room to spare.
    private static long Pack(int x, int y, int z) => ((long)x << 42) | ((long)y << 21) | (uint)z;

    private static void Unpack(long key, out int x, out int y, out int z)
    {
        x = (int)(key >> 42);
        y = (int)((key >> 21) & 0x1FFFFF);
        z = (int)(key & 0x1FFFFF);
    }
}

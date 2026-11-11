using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>FlashVDM hierarchical volume decoding: a coarse dense grid, then each finer level evaluated only in a band around the surface.</summary>
/// <remarks>Follows <c>FlashVDMVolumeDecoding</c> (adaptive KV selection, <c>mean</c> mode) from Tencent-Hunyuan/Hunyuan3D-2.
/// The finest grid has <c>resolutions[0] * 2^(levels-1) + 1</c> points per axis, not <c>octree_resolution + 1</c>; points the hierarchy never
/// evaluated are NaN.</remarks>
internal static class Hunyuan3DFlashVdm
{
    private const float Unevaluated = -10000f;
    private const int CellsPerAxis = 6;

    /// <summary>Decodes the occupancy of <paramref name="count"/> query points in one spatial group.</summary>
    /// <remarks>Latents are selected from every <paramref name="stride"/>-th query of the group.</remarks>
    internal delegate void QueryGroup(ReadOnlySpan<float> coordinates, int count, int stride, Span<float> logits);

    /// <summary>The per-axis cell counts of each level, coarsest first, for <paramref name="octreeResolution"/>.</summary>
    internal static int[] Resolutions(int octreeResolution, int miniGridNum = 4, int minResolution = 63)
    {
        List<int> list = [];
        int r = octreeResolution;
        if (r < minResolution) list.Add(r);
        while (r >= minResolution)
        {
            list.Add(r);
            r /= 2;
        }

        list.Reverse();
        int[] levels = [.. list];
        levels[0] = (int)Math.Round(levels[0] / (double)miniGridNum, MidpointRounding.ToEven) * miniGridNum - 1;
        for (int i = 1; i < levels.Length; i++)
            levels[i] = levels[0] * (1 << i);
        return levels;
    }

    internal static ScalarField3D Decode(QueryGroup query, int octreeResolution, float bound, float mcLevel, int miniGridNum = 4)
    {
        ArgumentNullException.ThrowIfNull(query);
        int[] levels = Resolutions(octreeResolution, miniGridNum);
        double size = 2.0 * bound;
        int n = levels[0] + 1;
        float[] axis = Linspace(-bound, bound, n);
        float[] grid = new float[(long)n * n * n];   // x slowest: [(x * n + y) * n + z]

        // Coarsest level: dense, queried in miniGridNum^3 blocks so each block picks its own latents.
        int m = n / miniGridNum;
        float[] coords = new float[m * m * m * 3];
        float[] logits = new float[m * m * m];
        for (int a = 0; a < miniGridNum; a++)
            for (int b = 0; b < miniGridNum; b++)
                for (int c = 0; c < miniGridNum; c++)
                {
                    int w = 0;
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++)
                            for (int k = 0; k < m; k++)
                            {
                                coords[w++] = axis[a * m + i]; coords[w++] = axis[b * m + j]; coords[w++] = axis[c * m + k];
                            }

                    query(coords, m * m * m, 100, logits);
                    int r = 0;
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++)
                            for (int k = 0; k < m; k++)
                                grid[(((long)(a * m + i)) * n + (b * m + j)) * n + (c * m + k)] = logits[r++];
                }

        for (int level = 1; level < levels.Length; level++)
        {
            int nextRes = levels[level];
            int next = nextRes + 1;
            bool last = level == levels.Length - 1;
            bool[] near = NearSurface(grid, n, mcLevel);
            if (!last) near = Dilate(near, n);

            bool[] nextIndex = new bool[(long)next * next * next];
            for (int x = 0; x < n; x++)
                for (int y = 0; y < n; y++)
                    for (int z = 0; z < n; z++)
                        if (near[((long)x * n + y) * n + z])
                            nextIndex[((long)(2 * x) * next + 2 * y) * next + 2 * z] = true;
            for (int i = 0; i < (last ? 2 : 1); i++)
                nextIndex = Dilate(nextIndex, next);

            float step = (float)(size / nextRes), origin = -bound;
            List<int> flat = [];
            for (long p = 0; p < nextIndex.LongLength; p++)
                if (nextIndex[p]) flat.Add((int)p);
            int count = flat.Count;
            float[] px = new float[count * 3];
            for (int p = 0; p < count; p++)
            {
                int f = flat[p];
                px[p * 3] = (f / (next * next)) * step + origin;
                px[p * 3 + 1] = (f / next % next) * step + origin;
                px[p * 3 + 2] = (f % next) * step + origin;
            }

            float[] nextLogits = new float[nextIndex.LongLength];
            Array.Fill(nextLogits, Unevaluated);
            int[] cell = CellOf(px, count);
            int[][] groups = GroupByCell(cell, count);
            foreach (int[] members in groups)
            {
                if (members.Length == 0) continue;
                float[] gc = new float[members.Length * 3], gl = new float[members.Length];
                for (int t = 0; t < members.Length; t++)
                {
                    gc[t * 3] = px[members[t] * 3]; gc[t * 3 + 1] = px[members[t] * 3 + 1]; gc[t * 3 + 2] = px[members[t] * 3 + 2];
                }

                query(gc, members.Length, 50, gl);
                for (int t = 0; t < members.Length; t++)
                    nextLogits[flat[members[t]]] = gl[t];
            }

            grid = nextLogits;
            n = next;
        }

        float[] values = new float[grid.Length];
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++)
                for (int z = 0; z < n; z++)
                {
                    float v = grid[((long)x * n + y) * n + z];
                    values[x + n * (y + n * z)] = v == Unevaluated ? float.NaN : v;
                }

        return new ScalarField3D
        {
            Values = values, ResX = n, ResY = n, ResZ = n, Min = (-bound, -bound, -bound), Max = (bound, bound, bound),
        };
    }

    /// <summary>Voxels whose sign differs from a neighbour (unevaluated neighbours count as equal), plus voxels with |logit| below 0.95.</summary>
    private static bool[] NearSurface(float[] grid, int n, float alpha)
    {
        bool[] mask = new bool[grid.LongLength];
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++)
                for (int z = 0; z < n; z++)
                {
                    long i = ((long)x * n + y) * n + z;
                    float raw = grid[i], v = raw + alpha;
                    if (v <= -9000f) continue;
                    int s = Math.Sign(v);
                    bool differs = Differs(grid, n, x - 1, y, z, alpha, v, s) || Differs(grid, n, x + 1, y, z, alpha, v, s)
                        || Differs(grid, n, x, y - 1, z, alpha, v, s) || Differs(grid, n, x, y + 1, z, alpha, v, s)
                        || Differs(grid, n, x, y, z - 1, alpha, v, s) || Differs(grid, n, x, y, z + 1, alpha, v, s);
                    mask[i] = differs || MathF.Abs(raw) < 0.95f;
                }

        return mask;
    }

    private static bool Differs(float[] grid, int n, int x, int y, int z, float alpha, float own, int ownSign)
    {
        if (x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n) return false;
        float v = grid[((long)x * n + y) * n + z] + alpha;
        if (v <= -9000f) v = own;
        return Math.Sign(v) != ownSign;
    }

    /// <summary>3x3x3 box dilation with zero padding.</summary>
    private static bool[] Dilate(bool[] src, int n)
    {
        bool[] dst = new bool[src.LongLength];
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++)
                for (int z = 0; z < n; z++)
                {
                    if (!src[((long)x * n + y) * n + z]) continue;
                    for (int dx = Math.Max(0, x - 1); dx <= Math.Min(n - 1, x + 1); dx++)
                        for (int dy = Math.Max(0, y - 1); dy <= Math.Min(n - 1, y + 1); dy++)
                            for (int dz = Math.Max(0, z - 1); dz <= Math.Min(n - 1, z + 1); dz++)
                                dst[((long)dx * n + dy) * n + dz] = true;
                }

        return dst;
    }

    /// <summary>The 6x6x6 spatial cell of each point, from float32 arithmetic on the points' bounding box.</summary>
    private static int[] CellOf(float[] px, int count)
    {
        float[] lo = [float.MaxValue, float.MaxValue, float.MaxValue], hi = [float.MinValue, float.MinValue, float.MinValue];
        for (int p = 0; p < count; p++)
            for (int a = 0; a < 3; a++)
            {
                lo[a] = MathF.Min(lo[a], px[p * 3 + a]); hi[a] = MathF.Max(hi[a], px[p * 3 + a]);
            }

        int[] cell = new int[count];
        for (int p = 0; p < count; p++)
        {
            int id = 0;
            for (int a = 0; a < 3; a++)
            {
                float scaled = (px[p * 3 + a] - lo[a]) / (hi[a] - lo[a]) * (CellsPerAxis - 0.001f);
                id = id * CellsPerAxis + (int)MathF.Floor(scaled);
            }

            cell[p] = id;
        }

        return cell;
    }

    /// <summary>Point indices per cell id, each in ascending (original) order, cells in ascending id order.</summary>
    private static int[][] GroupByCell(int[] cell, int count)
    {
        int cells = CellsPerAxis * CellsPerAxis * CellsPerAxis;
        List<int>[] lists = new List<int>[cells];
        for (int c = 0; c < cells; c++) lists[c] = [];
        for (int p = 0; p < count; p++) lists[cell[p]].Add(p);
        return [.. lists.Select(l => l.ToArray())];
    }

    /// <summary>numpy <c>linspace(-bound, bound, n, dtype=float32)</c>: float64 arithmetic, then a float32 cast.</summary>
    private static float[] Linspace(float lo, float hi, int n)
    {
        double start = lo, end = hi;
        double step = (end - start) / (n - 1);
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = (float)(i * step + start);
        values[n - 1] = (float)end;
        return values;
    }
}

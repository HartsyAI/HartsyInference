using System.Numerics;

namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Rebuilds a triangle surface as a clean closed shell on a regular grid, the pure-C# counterpart of CuMesh's <c>remesh_narrow_band_dc</c> that upstream TRELLIS.2's
/// <c>to_glb(remesh=True)</c> runs. A sparse grid is refined from about 32 cells up to <see cref="NarrowBandRemeshOptions.Resolution"/>, keeping the voxels whose centre lies within
/// the band of the surface at distance <c>band</c> voxels from the input (the unsigned distance comes from a <see cref="TriangleBvh"/>). The field at a grid corner is that
/// distance minus the band, so its zero set is the input offset by the band on both sides, a closed shell however torn the input was. Dual contouring then puts one vertex in each
/// voxel (the mean of the linearly interpolated zero crossings of its edges, the centre when there are none) and one quad on every edge whose end values change sign, taken
/// through the four voxels around the edge (a quad with a missing voxel is dropped) and split along the diagonal that keeps the two triangles flatter. Optionally the vertices
/// are moved back towards the input and the inner sheet is dropped.
/// </summary>
public static class NarrowBandRemesher
{
    private const int BaseCells = 32;
    private const double BandCellRatio = 0.87;
    private static readonly int[][] QuadOffsets = [[0, 0], [1, 0], [1, 1], [0, 1]];

    /// <summary>Remeshes <paramref name="mesh"/>; the result has positions and indices only.</summary>
    /// <param name="bvh">The hierarchy of <paramref name="mesh"/>, or null to build it here.</param>
    /// <exception cref="ArgumentException">The mesh has no triangles, or the resolution is not usable.</exception>
    public static Mesh Remesh(Mesh mesh, NarrowBandRemeshOptions options, TriangleBvh? bvh = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(options);
        Validate(mesh, options);
        bvh ??= new TriangleBvh(mesh);

        int res = options.Resolution;
        (Vector3 min, Vector3 max) = Bounds(mesh.Vertices);
        double extent = Math.Max(max.X - min.X, Math.Max(max.Y - min.Y, max.Z - min.Z));
        Vector3 center = options.Center ?? (min + max) * 0.5f;
        double scale = options.Scale ?? (res + 3 * options.Band) / res * extent;
        double voxel = scale / res;
        double eps = options.Band * voxel;
        Vector3 origin = center - new Vector3((float)(scale * 0.5));

        (int[] vx, int[] vy, int[] vz) = ActiveVoxels(bvh, origin, scale, res, eps, cancellationToken);
        int voxels = vx.Length;
        if (voxels == 0)
            return new Mesh { Vertices = [], Indices = [] };

        long stride = res + 1;
        long CornerKey(int x, int y, int z) => (x * stride + y) * stride + z;
        VoxelKeyMap voxelMap = new(voxels);
        for (int i = 0; i < voxels; i++)
            voxelMap.GetOrAdd(CornerKey(vx[i], vy[i], vz[i]), out _);

        VoxelKeyMap corners = new(voxels * 2);
        List<int> cornerX = new(voxels * 2), cornerY = new(voxels * 2), cornerZ = new(voxels * 2);
        for (int i = 0; i < voxels; i++)
        {
            if ((i & 4095) == 0)
                cancellationToken.ThrowIfCancellationRequested();
            for (int c = 0; c < 8; c++)
            {
                int x = vx[i] + (c & 1), y = vy[i] + ((c >> 1) & 1), z = vz[i] + (c >> 2);
                corners.GetOrAdd(CornerKey(x, y, z), out bool added);
                if (added)
                {
                    cornerX.Add(x);
                    cornerY.Add(y);
                    cornerZ.Add(z);
                }
            }
        }

        int cornerCount = corners.Count;
        float[] value = new float[cornerCount];
        ParallelOptions parallel = new() { CancellationToken = cancellationToken };
        Parallel.For(0, cornerCount, parallel, c =>
            value[c] = (float)(bvh.Distance(origin.X + cornerX[c] * voxel, origin.Y + cornerY[c] * voxel, origin.Z + cornerZ[c] * voxel) - eps));

        bool[]? exterior = options.OuterShellOnly ? ExteriorCorners(corners, value, cornerX, cornerY, cornerZ, CornerKey) : null;

        float[] vertex = new float[voxels * 3];
        sbyte[] flag = new sbyte[voxels * 3];
        Parallel.For(0, voxels, parallel, i => Contour(i, vx, vy, vz, corners, value, CornerKey, vertex, flag));

        List<int> triangles = new(voxels * 6);
        bool[] used = new bool[voxels];
        for (int i = 0; i < voxels; i++)
        {
            if ((i & 4095) == 0)
                cancellationToken.ThrowIfCancellationRequested();
            for (int axis = 0; axis < 3; axis++)
            {
                int f = flag[i * 3 + axis];
                if (f == 0)
                    continue;
                if (exterior is not null && !PositiveEndIsExterior(i, axis, f, vx, vy, vz, corners, exterior, CornerKey))
                    continue;
                AddQuad(i, axis, f, vx, vy, vz, voxelMap, CornerKey, vertex, triangles, used);
            }
        }

        return Finish(vertex, triangles, used, origin, voxel, bvh, options.ProjectBack);
    }

    private static void Validate(Mesh mesh, NarrowBandRemeshOptions options)
    {
        if (mesh.TriangleCount == 0)
            throw new ArgumentException("The mesh has no triangles.", nameof(mesh));
        if (options.Resolution is < BaseCells or > 4096)
            throw new ArgumentException($"The resolution must be between {BaseCells} and 4096; got {options.Resolution}.", nameof(options));
        int cells = options.Resolution;
        while (cells > BaseCells && cells % 2 == 0)
            cells /= 2;
        if (cells > 64)
            throw new ArgumentException($"The resolution must be {BaseCells} times a power of two, or a multiple of one such as 1536; got {options.Resolution}.", nameof(options));
        if (!(options.Band > 0) || !double.IsFinite(options.Band))
            throw new ArgumentOutOfRangeException(nameof(options), "The band must be positive.");
        if (!(options.ProjectBack >= 0 && options.ProjectBack <= 1))
            throw new ArgumentOutOfRangeException(nameof(options), "ProjectBack must be between 0 and 1.");
        if (options.Scale is { } s && !(s > 0))
            throw new ArgumentOutOfRangeException(nameof(options), "The scale must be positive.");
    }

    private static (Vector3 Min, Vector3 Max) Bounds(float[] positions)
    {
        Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        for (int i = 0; i + 2 < positions.Length; i += 3)
        {
            Vector3 p = new(positions[i], positions[i + 1], positions[i + 2]);
            min = Vector3.Min(min, p);
            max = Vector3.Max(max, p);
        }

        return (min, max);
    }

    // Refines from the base grid: a voxel stays while its centre is within 0.87 of a cell of the shell, and its children are tested at the next level.
    private static (int[] X, int[] Y, int[] Z) ActiveVoxels(TriangleBvh bvh, Vector3 origin, double scale, int res, double eps, CancellationToken cancellationToken)
    {
        int cells = res, levels = 0;
        while (cells > BaseCells && cells % 2 == 0)
        {
            cells /= 2;
            levels++;
        }

        List<int> x = new(), y = new(), z = new();
        for (int i = 0; i < cells; i++)
            for (int j = 0; j < cells; j++)
                for (int k = 0; k < cells; k++)
                {
                    x.Add(i);
                    y.Add(j);
                    z.Add(k);
                }

        int[] cx = [.. x], cy = [.. y], cz = [.. z];
        ParallelOptions parallel = new() { CancellationToken = cancellationToken };
        for (int level = 0; level <= levels; level++)
        {
            int resolution = cells << level;
            double cell = scale / resolution;
            bool[] keep = new bool[cx.Length];
            int[] px = cx, py = cy, pz = cz;
            Parallel.For(0, px.Length, parallel, i =>
            {
                double d = bvh.Distance(origin.X + (px[i] + 0.5) * cell, origin.Y + (py[i] + 0.5) * cell, origin.Z + (pz[i] + 0.5) * cell);
                keep[i] = Math.Abs(d - eps) < BandCellRatio * cell;
            });
            List<int> kx = [], ky = [], kz = [];
            for (int i = 0; i < keep.Length; i++)
            {
                if (!keep[i])
                    continue;
                if (level == levels)
                {
                    kx.Add(cx[i]);
                    ky.Add(cy[i]);
                    kz.Add(cz[i]);
                    continue;
                }

                for (int c = 0; c < 8; c++)
                {
                    kx.Add(cx[i] * 2 + (c & 1));
                    ky.Add(cy[i] * 2 + ((c >> 1) & 1));
                    kz.Add(cz[i] * 2 + (c >> 2));
                }
            }

            cx = [.. kx];
            cy = [.. ky];
            cz = [.. kz];
            if (cx.Length == 0)
                break;
        }

        return (cx, cy, cz);
    }

    private static void Contour(
        int i, int[] vx, int[] vy, int[] vz, VoxelKeyMap corners, float[] value, Func<int, int, int, long> key, float[] vertex, sbyte[] flag)
    {
        Span<float> v = stackalloc float[8];
        for (int c = 0; c < 8; c++)
            v[c] = value[corners.Find(key(vx[i] + (c & 1), vy[i] + ((c >> 1) & 1), vz[i] + (c >> 2)))];

        double sx = 0, sy = 0, sz = 0;
        int crossings = 0;
        for (int axis = 0; axis < 3; axis++)
        {
            int p = (axis + 1) % 3, q = (axis + 2) % 3;
            for (int u = 0; u < 2; u++)
                for (int w = 0; w < 2; w++)
                {
                    int start = (u << p) | (w << q), end = start | (1 << axis);
                    float a = v[start], b = v[end];
                    bool negativeToPositive = a < 0 && b >= 0, positiveToNegative = a >= 0 && b < 0;
                    if (u == 1 && w == 1)
                        flag[i * 3 + axis] = (sbyte)(negativeToPositive ? 1 : positiveToNegative ? -1 : 0);
                    if (!negativeToPositive && !positiveToNegative)
                        continue;
                    double t = -a / (double)(b - a);
                    sx += (start & 1) + (axis == 0 ? t : 0);
                    sy += ((start >> 1) & 1) + (axis == 1 ? t : 0);
                    sz += ((start >> 2) & 1) + (axis == 2 ? t : 0);
                    crossings++;
                }
        }

        if (crossings == 0)
        {
            sx = sy = sz = 0.5;
            crossings = 1;
        }

        vertex[i * 3] = (float)(vx[i] + sx / crossings);
        vertex[i * 3 + 1] = (float)(vy[i] + sy / crossings);
        vertex[i * 3 + 2] = (float)(vz[i] + sz / crossings);
    }

    private static void AddQuad(
        int i, int axis, int direction, int[] vx, int[] vy, int[] vz, VoxelKeyMap voxelMap, Func<int, int, int, long> key, float[] vertex, List<int> triangles, bool[] used)
    {
        int p = (axis + 1) % 3, q = (axis + 2) % 3;
        Span<int> quad = stackalloc int[4];
        for (int n = 0; n < 4; n++)
        {
            int[] o = QuadOffsets[n];
            int[] c = [vx[i], vy[i], vz[i]];
            c[p] += o[0];
            c[q] += o[1];
            int found = voxelMap.Find(key(c[0], c[1], c[2]));
            if (found < 0)
                return;
            quad[n] = found;
        }

        if (direction < 0)
        {
            (quad[1], quad[3]) = (quad[3], quad[1]);
        }

        // Of the two diagonals, split along the one whose triangles face most alike.
        double aligned02 = Alignment(vertex, quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
        double aligned13 = Alignment(vertex, quad[1], quad[2], quad[3], quad[1], quad[3], quad[0]);
        if (aligned02 >= aligned13)
            triangles.AddRange([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]);
        else
            triangles.AddRange([quad[1], quad[2], quad[3], quad[1], quad[3], quad[0]]);
        foreach (int v in quad)
            used[v] = true;
    }

    private static double Alignment(float[] vertex, int a0, int b0, int c0, int a1, int b1, int c1)
    {
        Vector3 n0 = Normal(vertex, a0, b0, c0), n1 = Normal(vertex, a1, b1, c1);
        return Vector3.Dot(n0, n1);
    }

    private static Vector3 Normal(float[] vertex, int a, int b, int c)
    {
        Vector3 pa = At(vertex, a), pb = At(vertex, b), pc = At(vertex, c);
        Vector3 n = Vector3.Cross(pb - pa, pc - pa);
        float len = n.Length();
        return len > 1e-20f ? n / len : Vector3.Zero;
    }

    private static Vector3 At(float[] v, int i) => new(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);

    // The corners outside the shell: the positive corners reachable from the one just beyond the leftmost corner of each negative (inside the shell) component.
    private static bool[] ExteriorCorners(
        VoxelKeyMap corners, float[] value, List<int> cx, List<int> cy, List<int> cz, Func<int, int, int, long> key)
    {
        int n = value.Length;
        bool[] exterior = new bool[n];
        bool[] seen = new bool[n];
        Queue<int> queue = new();
        int[] dx = [-1, 1, 0, 0, 0, 0], dy = [0, 0, -1, 1, 0, 0], dz = [0, 0, 0, 0, -1, 1];
        for (int start = 0; start < n; start++)
        {
            if (value[start] >= 0 || seen[start])
                continue;
            int leftmost = start;
            seen[start] = true;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int c = queue.Dequeue();
                if (cx[c] < cx[leftmost] || (cx[c] == cx[leftmost] && (cy[c] < cy[leftmost] || (cy[c] == cy[leftmost] && cz[c] < cz[leftmost]))))
                    leftmost = c;
                for (int d = 0; d < 6; d++)
                {
                    int next = corners.Find(key(cx[c] + dx[d], cy[c] + dy[d], cz[c] + dz[d]));
                    if (next >= 0 && value[next] < 0 && !seen[next])
                    {
                        seen[next] = true;
                        queue.Enqueue(next);
                    }
                }
            }

            int seed = corners.Find(key(cx[leftmost] - 1, cy[leftmost], cz[leftmost]));
            if (seed >= 0 && value[seed] >= 0 && !exterior[seed])
                Flood(seed);
        }

        return exterior;

        void Flood(int seed)
        {
            exterior[seed] = true;
            queue.Enqueue(seed);
            while (queue.Count > 0)
            {
                int c = queue.Dequeue();
                for (int d = 0; d < 6; d++)
                {
                    int next = corners.Find(key(cx[c] + dx[d], cy[c] + dy[d], cz[c] + dz[d]));
                    if (next >= 0 && value[next] >= 0 && !exterior[next])
                    {
                        exterior[next] = true;
                        queue.Enqueue(next);
                    }
                }
            }
        }
    }

    // A component whose seed corner was not found keeps both sheets, so its quads pass.
    private static bool PositiveEndIsExterior(
        int i, int axis, int direction, int[] vx, int[] vy, int[] vz, VoxelKeyMap corners, bool[] exterior, Func<int, int, int, long> key)
    {
        int p = (axis + 1) % 3, q = (axis + 2) % 3;
        int[] c = [vx[i], vy[i], vz[i]];
        c[p] += 1;
        c[q] += 1;
        if (direction > 0)
            c[axis] += 1;
        int corner = corners.Find(key(c[0], c[1], c[2]));
        return corner >= 0 && exterior[corner];
    }

    private static Mesh Finish(float[] vertex, List<int> triangles, bool[] used, Vector3 origin, double voxel, TriangleBvh bvh, double projectBack)
    {
        int[] remap = new int[used.Length];
        int kept = 0;
        for (int i = 0; i < used.Length; i++)
            remap[i] = used[i] ? kept++ : -1;
        float[] positions = new float[kept * 3];
        Parallel.For(0, used.Length, i =>
        {
            if (!used[i])
                return;
            double x = origin.X + vertex[i * 3] * voxel, y = origin.Y + vertex[i * 3 + 1] * voxel, z = origin.Z + vertex[i * 3 + 2] * voxel;
            if (projectBack > 0)
            {
                BvhHit hit = bvh.ClosestPoint(x, y, z);
                x -= projectBack * (x - hit.X);
                y -= projectBack * (y - hit.Y);
                z -= projectBack * (z - hit.Z);
            }

            positions[remap[i] * 3] = (float)x;
            positions[remap[i] * 3 + 1] = (float)y;
            positions[remap[i] * 3 + 2] = (float)z;
        });
        int[] indices = new int[triangles.Count];
        for (int i = 0; i < indices.Length; i++)
            indices[i] = remap[triangles[i]];
        return new Mesh { Vertices = positions, Indices = indices };
    }
}

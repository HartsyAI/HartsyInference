using System.Numerics;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// CPU port of TRELLIS.2 o-voxel <c>mesh_to_flexible_dual_grid_cpu</c>: scan-line intersection of voxel edges with
/// faces, QEF accumulation over faces and open-mesh boundary edges, then a constrained dual-vertex solve per voxel.
/// The vertices must already be shifted so the grid starts at the origin (the Python wrapper's <c>aabb[0]</c> offset).
/// </summary>
public static class DualGridVoxelizer
{
    // NumTraits<float>::epsilon(). Eigen's rank threshold is diagonalSize times this value.
    private const float FloatEpsilon = 1.1920929E-07f;

    // Eigen skips a Householder reflection when the tail norm is at or below the smallest normal float.
    private const float HouseholderTailTolerance = 1.17549435E-38f;

    // Eigen recomputes a column norm directly once its downdated estimate falls below sqrt(epsilon).
    private const float NormDowndateThreshold = 3.4526698E-4f;

    private const double DegenerateSegmentLength = 1e-6;

    /// <summary>Voxelizes a triangle mesh over the grid <c>[gridMin, gridMax)</c> with voxel edge lengths <paramref name="voxelSize"/>.</summary>
    /// <param name="vertices">Mesh vertices <c>[V·3]</c>.</param>
    /// <param name="faces">Triangle vertex indices <c>[F·3]</c>.</param>
    /// <param name="gridMin">Inclusive voxel-index minimum <c>[3]</c>.</param>
    /// <param name="gridMax">Exclusive voxel-index maximum <c>[3]</c>.</param>
    /// <param name="faceWeight">Enables the face pass when positive. The reference adds the unscaled plane quadric.</param>
    /// <param name="boundaryWeight">Enables the boundary pass when positive and scales its quadrics.</param>
    /// <param name="regularizationWeight">Enables the mean-position regularizer when positive.</param>
    public static DualGridVoxels Voxelize(
        float[] vertices,
        int[] faces,
        Vector3 voxelSize,
        int[] gridMin,
        int[] gridMax,
        float faceWeight,
        float boundaryWeight,
        float regularizationWeight)
    {
        ValidateArguments(vertices, faces, voxelSize, gridMin, gridMax);

        GridSpec grid = new(voxelSize, gridMin, gridMax);
        Vector3[] triangles = BuildTriangles(vertices, faces);
        Accumulator acc = new();
        IntersectQef(acc, triangles, grid);

        if (faceWeight > 0f)
        {
            FaceQef(acc, triangles, grid);
        }

        if (boundaryWeight > 0f)
        {
            BoundaryQef(acc, vertices, faces, grid, boundaryWeight);
        }

        return SolveDualVertices(acc, voxelSize, regularizationWeight);
    }

    private static void ValidateArguments(
        float[] vertices, int[] faces, Vector3 voxelSize, int[] gridMin, int[] gridMax)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(faces);
        ArgumentNullException.ThrowIfNull(gridMin);
        ArgumentNullException.ThrowIfNull(gridMax);
        if (vertices.Length % 3 != 0)
        {
            throw new ArgumentException("vertices must be [V*3].", nameof(vertices));
        }

        if (faces.Length % 3 != 0)
        {
            throw new ArgumentException("faces must be [F*3].", nameof(faces));
        }

        if (gridMin.Length != 3)
        {
            throw new ArgumentException("gridMin must have 3 elements.", nameof(gridMin));
        }

        if (gridMax.Length != 3)
        {
            throw new ArgumentException("gridMax must have 3 elements.", nameof(gridMax));
        }

        for (int axis = 0; axis < 3; axis++)
        {
            if (gridMax[axis] <= gridMin[axis])
            {
                throw new ArgumentException("gridMax must exceed gridMin on every axis.", nameof(gridMax));
            }
        }

        if (!(voxelSize.X > 0f && voxelSize.Y > 0f && voxelSize.Z > 0f))
        {
            throw new ArgumentException("voxelSize must be positive on every axis.", nameof(voxelSize));
        }

        for (int i = 0; i < vertices.Length; i++)
        {
            if (!float.IsFinite(vertices[i]))
            {
                throw new ArgumentException("vertices must be finite.", nameof(vertices));
            }
        }

        int vertexCount = vertices.Length / 3;
        for (int i = 0; i < faces.Length; i++)
        {
            if (faces[i] < 0 || faces[i] >= vertexCount)
            {
                throw new ArgumentException($"Face index {faces[i]} is outside [0, {vertexCount}).", nameof(faces));
            }
        }
    }

    private static Vector3[] BuildTriangles(float[] vertices, int[] faces)
    {
        Vector3[] triangles = new Vector3[faces.Length];
        for (int i = 0; i < faces.Length; i++)
        {
            int v = faces[i] * 3;
            triangles[i] = new Vector3(vertices[v], vertices[v + 1], vertices[v + 2]);
        }

        return triangles;
    }

    // ── Intersect QEF: scan-line fill of voxels whose edges cross each triangle ────────────────────

    private static void IntersectQef(Accumulator acc, Vector3[] triangles, GridSpec grid)
    {
        for (int t = 0; t < triangles.Length; t += 3)
        {
            Vector3 v0 = triangles[t];
            Vector3 v1 = triangles[t + 1];
            Vector3 v2 = triangles[t + 2];
            Vector3 n = FaceNormal(v0, v1, v2);
            float[] q = PlaneQuadric(n, -Vector3.Dot(n, v0));
            for (int ax2 = 0; ax2 < 3; ax2++)
            {
                ScanLineFill(acc, grid, ax2, v0, v1, v2, q);
            }
        }
    }

    private static void ScanLineFill(
        Accumulator acc, GridSpec grid, int ax2, Vector3 v0, Vector3 v1, Vector3 v2, float[] q)
    {
        int ax0 = (ax2 + 1) % 3;
        int ax1 = (ax2 + 2) % 3;
        Vec3d t0 = Permute(v0, ax0, ax1, ax2);
        Vec3d t1 = Permute(v1, ax0, ax1, ax2);
        Vec3d t2 = Permute(v2, ax0, ax1, ax2);
        SortByScanCoordinate(ref t0, ref t1, ref t2);

        int start = Math.Clamp((int)(t0[1] / At(grid.VoxelSize, ax1)), grid.Min[ax1], grid.Max[ax1] - 1);
        int mid = Math.Clamp((int)(t1[1] / At(grid.VoxelSize, ax1)), grid.Min[ax1], grid.Max[ax1] - 1);
        int end = Math.Clamp((int)(t2[1] / At(grid.VoxelSize, ax1)), grid.Min[ax1], grid.Max[ax1] - 1);

        ScanHalf(acc, grid, ax2, start, mid, t0, t1, t2, q);
        ScanHalf(acc, grid, ax2, mid, end, t2, t1, t0, q);
    }

    // Each half sweeps rows of the triangle between two of its edges. Components are in permuted order:
    // [0] = scan-plane x (ax0), [1] = scan coordinate (ax1), [2] = ax2.
    private static void ScanHalf(
        Accumulator acc, GridSpec grid, int ax2, int rowStart, int rowEnd, Vec3d t0, Vec3d t1, Vec3d t2, float[] q)
    {
        int ax0 = (ax2 + 1) % 3;
        int ax1 = (ax2 + 2) % 3;
        Span<int> coord = stackalloc int[3];
        Span<float> pos = stackalloc float[3];
        for (int yIdx = rowStart; yIdx < rowEnd; yIdx++)
        {
            // Reference computes (y_idx + 1) * voxel_size in float, then widens.
            double y = (float)(yIdx + 1) * At(grid.VoxelSize, ax1);
            Vec2d t3 = LerpPoint(t0[1], t1[1], y, new Vec2d(t0[0], t0[2]), new Vec2d(t1[0], t1[2]));
            Vec2d t4 = LerpPoint(t0[1], t2[1], y, new Vec2d(t0[0], t0[2]), new Vec2d(t2[0], t2[2]));
            if (t3.X > t4.X)
            {
                (t3, t4) = (t4, t3);
            }

            int lineStart = Math.Clamp((int)(t3.X / At(grid.VoxelSize, ax0)), grid.Min[ax0], grid.Max[ax0] - 1);
            int lineEnd = Math.Clamp((int)(t4.X / At(grid.VoxelSize, ax0)), grid.Min[ax0], grid.Max[ax0] - 1);
            for (int xIdx = lineStart; xIdx < lineEnd; xIdx++)
            {
                double x = (float)(xIdx + 1) * At(grid.VoxelSize, ax0);
                double z = LerpScalar(t3.X, t4.X, x, t3.Z, t4.Z);
                int zIdx = (int)(z / At(grid.VoxelSize, ax2));
                if (zIdx < grid.Min[ax2] || zIdx >= grid.Max[ax2])
                {
                    continue;
                }

                pos[ax0] = (float)x;
                pos[ax1] = (float)y;
                pos[ax2] = (float)z;
                for (int dx = 0; dx < 2; dx++)
                {
                    for (int dy = 0; dy < 2; dy++)
                    {
                        coord[ax0] = xIdx + dx;
                        coord[ax1] = yIdx + dy;
                        coord[ax2] = zIdx;
                        AccumulateCrossing(acc, coord, pos, ax2, dx == 0 && dy == 0, q);
                    }
                }
            }
        }
    }

    private static void AccumulateCrossing(
        Accumulator acc, ReadOnlySpan<int> coord, ReadOnlySpan<float> pos, int ax2, bool onEdge, float[] q)
    {
        (int, int, int) key = (coord[0], coord[1], coord[2]);
        if (!acc.Index.TryGetValue(key, out int i))
        {
            i = acc.Counts.Count;
            acc.Index.Add(key, i);
            acc.Coords.Add(coord[0]);
            acc.Coords.Add(coord[1]);
            acc.Coords.Add(coord[2]);
            acc.MeanSums.Add(pos[0]);
            acc.MeanSums.Add(pos[1]);
            acc.MeanSums.Add(pos[2]);
            acc.Counts.Add(1f);
            acc.Intersected.Add(false);
            acc.Intersected.Add(false);
            acc.Intersected.Add(false);
            acc.Qefs.Add((float[])q.Clone());
        }
        else
        {
            acc.MeanSums[i * 3] += pos[0];
            acc.MeanSums[i * 3 + 1] += pos[1];
            acc.MeanSums[i * 3 + 2] += pos[2];
            acc.Counts[i] += 1f;
            AddInPlace(acc.Qefs[i], q, 1f);
        }

        if (onEdge)
        {
            acc.Intersected[i * 3 + ax2] = true;
        }
    }

    // ── Face QEF: accumulate plane quadrics into voxels a triangle passes through ─────────────────

    private static void FaceQef(Accumulator acc, Vector3[] triangles, GridSpec grid)
    {
        Vector3 vs = grid.VoxelSize;
        Span<EdgeTest> edgeTests = stackalloc EdgeTest[9];
        for (int t = 0; t < triangles.Length; t += 3)
        {
            Vector3 v0 = triangles[t];
            Vector3 v1 = triangles[t + 1];
            Vector3 v2 = triangles[t + 2];
            Vector3 n = FaceNormal(v0, v1, v2);
            float[] q = PlaneQuadric(n, -Vector3.Dot(n, v0));

            Vector3 bbMinF = Vector3.Divide(Vector3.Min(Vector3.Min(v0, v1), v2), vs);
            Vector3 bbMaxF = Vector3.Divide(Vector3.Max(Vector3.Max(v0, v1), v2), vs);
            int bbMinX = Math.Max((int)bbMinF.X, grid.Min[0]);
            int bbMinY = Math.Max((int)bbMinF.Y, grid.Min[1]);
            int bbMinZ = Math.Max((int)bbMinF.Z, grid.Min[2]);
            int bbMaxX = Math.Min((int)(bbMaxF.X + 1f), grid.Max[0]);
            int bbMaxY = Math.Min((int)(bbMaxF.Y + 1f), grid.Max[1]);
            int bbMaxZ = Math.Min((int)(bbMaxF.Z + 1f), grid.Max[2]);

            Vector3 c = new(n.X > 0f ? vs.X : 0f, n.Y > 0f ? vs.Y : 0f, n.Z > 0f ? vs.Z : 0f);
            float d1 = Vector3.Dot(n, c - v0);
            float d2 = Vector3.Dot(n, vs - c - v0);
            BuildEdgeTests(edgeTests, n, v0, v1, v2, vs);

            for (int z = bbMinZ; z < bbMaxZ; z++)
            {
                for (int y = bbMinY; y < bbMaxY; y++)
                {
                    for (int x = bbMinX; x < bbMaxX; x++)
                    {
                        Vector3 p = new(vs.X * x, vs.Y * y, vs.Z * z);
                        float nDotP = Vector3.Dot(n, p);
                        if ((nDotP + d1) * (nDotP + d2) > 0f)
                        {
                            continue;
                        }

                        if (!InsideEdgeTests(edgeTests, p))
                        {
                            continue;
                        }

                        if (acc.Index.TryGetValue((x, y, z), out int i))
                        {
                            AddInPlace(acc.Qefs[i], q, 1f);
                        }
                    }
                }
            }
        }
    }

    // Each triangle edge gives one 2D half-plane test per coordinate plane (XY, YZ, ZX), in reference order.
    private static void BuildEdgeTests(
        Span<EdgeTest> tests, Vector3 n, Vector3 v0, Vector3 v1, Vector3 v2, Vector3 vs)
    {
        Vector3 e0 = v1 - v0;
        Vector3 e1 = v2 - v1;
        Vector3 e2 = v0 - v2;
        for (int m = 0; m < 3; m++)
        {
            tests[m * 3] = MakeEdgeTest(n, m, e0, v0, vs);
            tests[m * 3 + 1] = MakeEdgeTest(n, m, e1, v1, vs);
            tests[m * 3 + 2] = MakeEdgeTest(n, m, e2, v2, vs);
        }
    }

    // Plane m projects onto (u, w) = (m+1, m+2) mod 3. m = 2 is XY, m = 0 is YZ, m = 1 is ZX, matching the reference.
    private static EdgeTest MakeEdgeTest(Vector3 n, int m, Vector3 edge, Vector3 vertex, Vector3 vs)
    {
        int u = (m + 1) % 3;
        int w = (m + 2) % 3;
        float mul = At(n, m) < 0f ? -1f : 1f;
        Vector2 normal = new(-mul * At(edge, w), mul * At(edge, u));
        Vector2 positive = Vector2.Max(normal, Vector2.Zero);
        float offset = -Vector2.Dot(normal, new Vector2(At(vertex, u), At(vertex, w)))
            + Vector2.Dot(positive, new Vector2(At(vs, u), At(vs, w)));
        return new EdgeTest(normal, offset);
    }

    private static bool InsideEdgeTests(ReadOnlySpan<EdgeTest> tests, Vector3 p)
    {
        for (int m = 0; m < 3; m++)
        {
            Vector2 projected = new(At(p, (m + 1) % 3), At(p, (m + 2) % 3));
            for (int e = 0; e < 3; e++)
            {
                EdgeTest test = tests[m * 3 + e];
                if (Vector2.Dot(test.Normal, projected) + test.Offset < 0f)
                {
                    return false;
                }
            }
        }

        return true;
    }

    // ── Boundary QEF: lines of open-mesh edges, walked voxel by voxel (DDA) ───────────────────────

    private static void BoundaryQef(
        Accumulator acc, float[] vertices, int[] faces, GridSpec grid, float boundaryWeight)
    {
        // The reference walks a std::map, so boundary segments accumulate in ascending (min, max) vertex order.
        SortedDictionary<(int, int), int> edgeCount = new();
        for (int f = 0; f < faces.Length; f += 3)
        {
            for (int k = 0; k < 3; k++)
            {
                int e0 = faces[f + k];
                int e1 = faces[f + (k + 1) % 3];
                if (e0 > e1)
                {
                    (e0, e1) = (e1, e0);
                }

                edgeCount.TryGetValue((e0, e1), out int count);
                edgeCount[(e0, e1)] = count + 1;
            }
        }

        foreach (KeyValuePair<(int, int), int> edge in edgeCount)
        {
            if (edge.Value != 1)
            {
                continue;
            }

            Vector3 p0 = Position(vertices, edge.Key.Item1);
            Vector3 p1 = Position(vertices, edge.Key.Item2);
            BoundarySegment(acc, p0, p1, grid, boundaryWeight);
        }
    }

    private static void BoundarySegment(Accumulator acc, Vector3 v0, Vector3 v1, GridSpec grid, float boundaryWeight)
    {
        // Subtract in float and widen afterwards, as the reference does when building Vector3d(v1 - v0).
        Vec3d delta = new(v1.X - v0.X, v1.Y - v0.Y, v1.Z - v0.Z);
        double length = Math.Sqrt(delta.X * delta.X + delta.Y * delta.Y + delta.Z * delta.Z);
        if (length < DegenerateSegmentLength)
        {
            return;
        }

        Vec3d dir = new(delta.X / length, delta.Y / length, delta.Z / length);

        // A = I - d dᵀ (rounded to float), b = -A v0, c = v0ᵀ A v0 (row-vector product first, as Eigen evaluates it).
        Span<float> a = stackalloc float[9];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                a[r * 3 + c] = (r == c ? 1f : 0f) - (float)(dir[r] * dir[c]);
            }
        }

        Span<float> b = stackalloc float[3];
        for (int r = 0; r < 3; r++)
        {
            float rowDot = 0f;
            for (int c = 0; c < 3; c++)
            {
                rowDot += a[r * 3 + c] * At(v0, c);
            }

            b[r] = -rowDot;
        }

        float cScalar = 0f;
        for (int c = 0; c < 3; c++)
        {
            float column = 0f;
            for (int r = 0; r < 3; r++)
            {
                column += At(v0, r) * a[r * 3 + c];
            }

            cScalar += column * At(v0, c);
        }

        Span<float> q = stackalloc float[16];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                q[r * 4 + c] = a[r * 3 + c];
            }

            q[r * 4 + 3] = b[r];
            q[12 + r] = b[r];
        }

        q[15] = cScalar;

        Span<int> current = stackalloc int[3];
        Span<int> step = stackalloc int[3];
        Span<double> tMax = stackalloc double[3];
        Span<double> tDelta = stackalloc double[3];
        for (int axis = 0; axis < 3; axis++)
        {
            current[axis] = (int)MathF.Floor(At(v0, axis) / At(grid.VoxelSize, axis));
            step[axis] = dir[axis] > 0.0 ? 1 : -1;
        }

        for (int axis = 0; axis < 3; axis++)
        {
            double d = dir[axis];
            if (d == 0.0)
            {
                tMax[axis] = double.PositiveInfinity;
                tDelta[axis] = double.PositiveInfinity;
                continue;
            }

            // Reference computes the voxel border in float before the double division.
            float voxelBorder = At(grid.VoxelSize, axis) * (current[axis] + (step[axis] > 0 ? 1 : 0));
            tMax[axis] = (voxelBorder - At(v0, axis)) / d;
            tDelta[axis] = At(grid.VoxelSize, axis) / Math.Abs(d);
        }

        AccumulateBoundaryVoxel(acc, grid, current, q, boundaryWeight);
        while (true)
        {
            int axis;
            if (tMax[0] < tMax[1])
            {
                axis = tMax[0] < tMax[2] ? 0 : 2;
            }
            else
            {
                axis = tMax[1] < tMax[2] ? 1 : 2;
            }

            if (tMax[axis] > length)
            {
                break;
            }

            current[axis] += step[axis];
            tMax[axis] += tDelta[axis];
            AccumulateBoundaryVoxel(acc, grid, current, q, boundaryWeight);
        }
    }

    // A DDA walk visits each voxel at most once, so accumulating during the walk matches the reference's list pass.
    private static void AccumulateBoundaryVoxel(
        Accumulator acc, GridSpec grid, ReadOnlySpan<int> coord, ReadOnlySpan<float> q, float boundaryWeight)
    {
        if (coord[0] < grid.Min[0] || coord[0] >= grid.Max[0]
            || coord[1] < grid.Min[1] || coord[1] >= grid.Max[1]
            || coord[2] < grid.Min[2] || coord[2] >= grid.Max[2])
        {
            return;
        }

        if (acc.Index.TryGetValue((coord[0], coord[1], coord[2]), out int i))
        {
            AddInPlace(acc.Qefs[i], q, boundaryWeight);
        }
    }

    // ── Dual vertices: unconstrained solve, then the best constrained candidate if it leaves the voxel ─

    private static DualGridVoxels SolveDualVertices(Accumulator acc, Vector3 voxelSize, float regularizationWeight)
    {
        int count = acc.Counts.Count;
        float[] dual = new float[count * 3];
        for (int i = 0; i < count; i++)
        {
            int cx = acc.Coords[i * 3];
            int cy = acc.Coords[i * 3 + 1];
            int cz = acc.Coords[i * 3 + 2];
            Vector3 minCorner = new(cx * voxelSize.X, cy * voxelSize.Y, cz * voxelSize.Z);
            Vector3 maxCorner = new((cx + 1) * voxelSize.X, (cy + 1) * voxelSize.Y, (cz + 1) * voxelSize.Z);

            float[] q = acc.Qefs[i];
            if (regularizationWeight > 0f)
            {
                AddRegularization(q, acc, i, regularizationWeight);
            }

            Vector3 vNew = SolveUnconstrained(q);
            if (!InBox(vNew, minCorner, maxCorner))
            {
                vNew = new ConstrainedSearch(q, minCorner, maxCorner, vNew).Run();
            }

            dual[i * 3] = vNew.X;
            dual[i * 3 + 1] = vNew.Y;
            dual[i * 3 + 2] = vNew.Z;
        }

        return new DualGridVoxels(acc.Coords.ToArray(), dual, acc.Intersected.ToArray());
    }

    private static void AddRegularization(float[] q, Accumulator acc, int i, float regularizationWeight)
    {
        float cnt = acc.Counts[i];
        Vector3 p = new(acc.MeanSums[i * 3] / cnt, acc.MeanSums[i * 3 + 1] / cnt, acc.MeanSums[i * 3 + 2] / cnt);
        float[] qReg = new float[16];
        for (int k = 0; k < 3; k++)
        {
            qReg[k * 4 + k] = 1f;
            qReg[k * 4 + 3] = -At(p, k);
            qReg[12 + k] = -At(p, k);
        }

        qReg[15] = Vector3.Dot(p, p);
        AddInPlace(q, qReg, regularizationWeight * cnt);
    }

    private static Vector3 SolveUnconstrained(float[] q)
    {
        Span<float> a = stackalloc float[9];
        Span<float> b = stackalloc float[3];
        Span<float> x = stackalloc float[3];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                a[r * 3 + c] = q[r * 4 + c];
            }

            b[r] = -q[r * 4 + 3];
        }

        SolveColPivQr(a, b, x);
        return new Vector3(x[0], x[1], x[2]);
    }

    // Column-pivoted Householder QR in single precision, as Eigen 3.4 colPivHouseholderQr: column norms are downdated
    // between steps, and pivots under Eigen's nonzeroPivots() threshold are dropped. Only the leading nonzero pivots
    // are solved, because Gaussian elimination would divide by zero on rank-deficient QEFs.
    // Sum order in the reflections and back-substitution can differ from Eigen by an ulp.
    internal static void SolveColPivQr(ReadOnlySpan<float> matrix, ReadOnlySpan<float> rhs, Span<float> solution)
    {
        int n = rhs.Length;
        Span<float> qr = stackalloc float[n * n];
        Span<float> c = stackalloc float[n];
        Span<float> tau = stackalloc float[n];
        Span<int> perm = stackalloc int[n];
        Span<float> normDirect = stackalloc float[n];
        Span<float> normUpdated = stackalloc float[n];
        matrix.CopyTo(qr);
        rhs.CopyTo(c);
        float maxNorm = 0f;
        for (int j = 0; j < n; j++)
        {
            perm[j] = j;
            float colSq = 0f;
            for (int r = 0; r < n; r++)
            {
                colSq += qr[r * n + j] * qr[r * n + j];
            }

            normDirect[j] = MathF.Sqrt(colSq);
            normUpdated[j] = normDirect[j];
            maxNorm = MathF.Max(maxNorm, normDirect[j]);
        }

        float scaledMax = maxNorm * FloatEpsilon;
        float thresholdHelper = scaledMax * scaledMax / n;
        int nonzero = n;

        for (int k = 0; k < n; k++)
        {
            int pivot = k;
            float bestNorm = -1f;
            for (int j = k; j < n; j++)
            {
                if (normUpdated[j] > bestNorm)
                {
                    bestNorm = normUpdated[j];
                    pivot = j;
                }
            }

            // `<=` rather than Eigen's `<` so an all-zero matrix (threshold 0, norm 0) counts as rank 0 instead of
            // dividing by zero.
            if (nonzero == n && bestNorm * bestNorm <= thresholdHelper * (n - k))
            {
                nonzero = k;
            }

            if (pivot != k)
            {
                for (int r = 0; r < n; r++)
                {
                    (qr[r * n + k], qr[r * n + pivot]) = (qr[r * n + pivot], qr[r * n + k]);
                }

                (perm[k], perm[pivot]) = (perm[pivot], perm[k]);
                (normUpdated[k], normUpdated[pivot]) = (normUpdated[pivot], normUpdated[k]);
                (normDirect[k], normDirect[pivot]) = (normDirect[pivot], normDirect[k]);
            }

            float head = qr[k * n + k];
            float tailSq = 0f;
            for (int r = k + 1; r < n; r++)
            {
                tailSq += qr[r * n + k] * qr[r * n + k];
            }

            tau[k] = 0f;
            if (tailSq > HouseholderTailTolerance)
            {
                float beta = MathF.Sqrt(head * head + tailSq);
                if (head >= 0f)
                {
                    beta = -beta;
                }

                float scale = head - beta;
                for (int r = k + 1; r < n; r++)
                {
                    qr[r * n + k] /= scale;
                }

                tau[k] = (beta - head) / beta;
                for (int j = k + 1; j < n; j++)
                {
                    float w = qr[k * n + j];
                    for (int r = k + 1; r < n; r++)
                    {
                        w += qr[r * n + k] * qr[r * n + j];
                    }

                    float tw = tau[k] * w;
                    qr[k * n + j] -= tw;
                    for (int r = k + 1; r < n; r++)
                    {
                        qr[r * n + j] -= w * (tau[k] * qr[r * n + k]);
                    }
                }

                qr[k * n + k] = beta;
            }

            // Runs after skipped reflections too, as in Eigen.
            for (int j = k + 1; j < n; j++)
            {
                if (normUpdated[j] == 0f)
                {
                    continue;
                }

                float t = MathF.Abs(qr[k * n + j]) / normUpdated[j];
                t = MathF.Max(0f, (1f + t) * (1f - t));
                float ratio = normUpdated[j] / normDirect[j];
                if (t * (ratio * ratio) <= NormDowndateThreshold)
                {
                    float tailNormSq = 0f;
                    for (int r = k + 1; r < n; r++)
                    {
                        tailNormSq += qr[r * n + j] * qr[r * n + j];
                    }

                    normDirect[j] = MathF.Sqrt(tailNormSq);
                    normUpdated[j] = normDirect[j];
                }
                else
                {
                    normUpdated[j] *= MathF.Sqrt(t);
                }
            }
        }

        int rank = nonzero;

        solution.Clear();
        if (rank == 0)
        {
            return;
        }

        for (int k = 0; k < rank; k++)
        {
            float w = c[k];
            for (int r = k + 1; r < n; r++)
            {
                w += qr[r * n + k] * c[r];
            }

            float tw = tau[k] * w;
            c[k] -= tw;
            for (int r = k + 1; r < n; r++)
            {
                c[r] -= w * (tau[k] * qr[r * n + k]);
            }
        }

        for (int i = rank - 1; i >= 0; i--)
        {
            float s = c[i];
            for (int j = i + 1; j < rank; j++)
            {
                s -= qr[i * n + j] * c[j];
            }

            c[i] = s / qr[i * n + i];
        }

        for (int i = 0; i < rank; i++)
        {
            solution[perm[i]] = c[i];
        }
    }

    // Constrained minimum of the QEF over the voxel box: the best candidate on each face, edge and corner that lies in
    // the box. The unconstrained point is the fallback, as in the reference, if no candidate qualifies.
    private sealed class ConstrainedSearch
    {
        private readonly float[] _q;
        private readonly Vector3 _minCorner;
        private readonly Vector3 _maxCorner;
        private float _bestError = float.PositiveInfinity;

        public ConstrainedSearch(float[] q, Vector3 minCorner, Vector3 maxCorner, Vector3 fallback)
        {
            _q = q;
            _minCorner = minCorner;
            _maxCorner = maxCorner;
            Point = fallback;
        }

        public Vector3 Point { get; private set; }

        public Vector3 Run()
        {
            for (int axis = 0; axis < 3; axis++)
            {
                SingleConstraint(axis, At(_minCorner, axis));
                SingleConstraint(axis, At(_maxCorner, axis));
            }

            for (int free = 0; free < 3; free++)
            {
                int ax1 = (free + 1) % 3;
                int ax2 = (free + 2) % 3;
                TwoConstraint(free, ax1, ax2, At(_minCorner, ax1), At(_minCorner, ax2));
                TwoConstraint(free, ax1, ax2, At(_minCorner, ax1), At(_maxCorner, ax2));
                TwoConstraint(free, ax1, ax2, At(_maxCorner, ax1), At(_minCorner, ax2));
                TwoConstraint(free, ax1, ax2, At(_maxCorner, ax1), At(_maxCorner, ax2));
            }

            Span<float> corner = stackalloc float[4];
            corner[3] = 1f;
            for (int xc = 0; xc < 2; xc++)
            {
                for (int yc = 0; yc < 2; yc++)
                {
                    for (int zc = 0; zc < 2; zc++)
                    {
                        corner[0] = xc == 1 ? _minCorner.X : _maxCorner.X;
                        corner[1] = yc == 1 ? _minCorner.Y : _maxCorner.Y;
                        corner[2] = zc == 1 ? _minCorner.Z : _maxCorner.Z;
                        Consider(corner);
                    }
                }
            }

            return Point;
        }

        // Fixes one axis at a bound and solves the other two with the 2×2 system from the reference.
        private void SingleConstraint(int fixedAxis, float bound)
        {
            int ax1 = (fixedAxis + 1) % 3;
            int ax2 = (fixedAxis + 2) % 3;
            Span<float> a = stackalloc float[4];
            a[0] = _q[ax1 * 4 + ax1];
            a[1] = _q[ax1 * 4 + ax2];
            a[2] = _q[ax2 * 4 + ax1];
            a[3] = _q[ax2 * 4 + ax2];

            float b00 = _q[ax1 * 4 + fixedAxis];
            float b01 = _q[ax1 * 4 + 3];
            float b10 = _q[ax2 * 4 + fixedAxis];
            float b11 = _q[ax2 * 4 + 3];
            Span<float> rhs = stackalloc float[2];
            rhs[0] = -(b00 * bound + b01 * 1f);
            rhs[1] = -(b10 * bound + b11 * 1f);

            Span<float> x = stackalloc float[2];
            SolveColPivQr(a, rhs, x);
            if (!(x[0] >= At(_minCorner, ax1) && x[0] <= At(_maxCorner, ax1)
                && x[1] >= At(_minCorner, ax2) && x[1] <= At(_maxCorner, ax2)))
            {
                return;
            }

            Span<float> p = stackalloc float[4];
            p[fixedAxis] = bound;
            p[ax1] = x[0];
            p[ax2] = x[1];
            p[3] = 1f;
            Consider(p);
        }

        // Fixes two axes at a corner pair and solves the free axis in closed form.
        private void TwoConstraint(int free, int ax1, int ax2, float c1, float c2)
        {
            float a = _q[free * 4 + free];
            float bx = _q[free * 4 + ax1];
            float by = _q[free * 4 + ax2];
            float bw = _q[free * 4 + 3];
            float x = -(bx * c1 + by * c2 + bw) / a;
            if (!(x >= At(_minCorner, free) && x <= At(_maxCorner, free)))
            {
                return;
            }

            Span<float> p = stackalloc float[4];
            p[free] = x;
            p[ax1] = c1;
            p[ax2] = c2;
            p[3] = 1f;
            Consider(p);
        }

        private void Consider(ReadOnlySpan<float> p)
        {
            float err = QuadraticForm(_q, p);
            if (err < _bestError)
            {
                _bestError = err;
                Point = new Vector3(p[0], p[1], p[2]);
            }
        }
    }

    // ── Small helpers ──────────────────────────────────────────────────────────────────────────────

    private static bool InBox(Vector3 p, Vector3 minCorner, Vector3 maxCorner) =>
        p.X >= minCorner.X && p.X <= maxCorner.X
        && p.Y >= minCorner.Y && p.Y <= maxCorner.Y
        && p.Z >= minCorner.Z && p.Z <= maxCorner.Z;

    // pᵀ Q p, with the row product evaluated first as in the reference.
    private static float QuadraticForm(float[] q, ReadOnlySpan<float> p)
    {
        float err = 0f;
        for (int c = 0; c < 4; c++)
        {
            float column = 0f;
            for (int r = 0; r < 4; r++)
            {
                column += p[r] * q[r * 4 + c];
            }

            err += column * p[c];
        }

        return err;
    }

    private static float[] PlaneQuadric(Vector3 n, float d)
    {
        float[] plane = { n.X, n.Y, n.Z, d };
        float[] q = new float[16];
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                q[r * 4 + c] = plane[r] * plane[c];
            }
        }

        return q;
    }

    // Normalized cross(v1 - v0, v2 - v1). Like Eigen's normalized(), a zero-length normal is returned unchanged.
    private static Vector3 FaceNormal(Vector3 v0, Vector3 v1, Vector3 v2)
    {
        Vector3 n = Vector3.Cross(v1 - v0, v2 - v1);
        float length = n.Length();
        return length > 0f ? n / length : Vector3.Zero;
    }

    private static void AddInPlace(float[] dst, ReadOnlySpan<float> src, float scale)
    {
        for (int k = 0; k < 16; k++)
        {
            dst[k] += scale * src[k];
        }
    }

    private static void SortByScanCoordinate(ref Vec3d a, ref Vec3d b, ref Vec3d c)
    {
        // Stable insertion sort on three elements, the same order std::sort produces for this input.
        if (b[1] < a[1])
        {
            (a, b) = (b, a);
        }

        if (c[1] < b[1])
        {
            (b, c) = (c, b);
            if (b[1] < a[1])
            {
                (a, b) = (b, a);
            }
        }
    }

    // Lerp over a Vector2 value, with the same branch and operation order as the reference template.
    private static Vec2d LerpPoint(double a, double b, double t, Vec2d valA, Vec2d valB)
    {
        if (a == b)
        {
            return valA;
        }

        double alpha = (t - a) / (b - a);
        return new Vec2d((1 - alpha) * valA.X + alpha * valB.X, (1 - alpha) * valA.Z + alpha * valB.Z);
    }

    private static double LerpScalar(double a, double b, double t, double valA, double valB)
    {
        if (a == b)
        {
            return valA;
        }

        double alpha = (t - a) / (b - a);
        return (1 - alpha) * valA + alpha * valB;
    }

    private static Vec3d Permute(Vector3 v, int ax0, int ax1, int ax2) =>
        new(At(v, ax0), At(v, ax1), At(v, ax2));

    private static Vector3 Position(float[] vertices, int vertex) =>
        new(vertices[vertex * 3], vertices[vertex * 3 + 1], vertices[vertex * 3 + 2]);

    private static float At(Vector3 v, int axis) => axis switch
    {
        0 => v.X,
        1 => v.Y,
        _ => v.Z,
    };

    private readonly record struct GridSpec(Vector3 VoxelSize, int[] Min, int[] Max);

    private readonly record struct Vec2d(double X, double Z);

    private readonly record struct Vec3d(double X, double Y, double Z)
    {
        public double this[int index] => index switch
        {
            0 => X,
            1 => Y,
            _ => Z,
        };
    }

    private readonly record struct EdgeTest(Vector2 Normal, float Offset);

    private sealed class Accumulator
    {
        public readonly Dictionary<(int, int, int), int> Index = new();
        public readonly List<int> Coords = new();
        public readonly List<float> MeanSums = new();
        public readonly List<float> Counts = new();
        public readonly List<bool> Intersected = new();
        public readonly List<float[]> Qefs = new();
    }
}

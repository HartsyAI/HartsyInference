namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Checks that a planar chart is a valid parametrization: no triangle is mirrored and no two triangles overlap in area.</summary>
internal static class UvChartValidator
{
    /// <summary>True when every triangle keeps the winding of its source face and no two triangles overlap.</summary>
    public static bool IsValid(UvChart chart, double[] faceArea3d)
    {
        double[] uv = chart.Uv;
        int[] tri = chart.LocalTriangles;
        int count = tri.Length / 3;
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        for (int i = 0; i < uv.Length; i += 2)
        {
            minX = Math.Min(minX, uv[i]); maxX = Math.Max(maxX, uv[i]);
            minY = Math.Min(minY, uv[i + 1]); maxY = Math.Max(maxY, uv[i + 1]);
        }
        double extent = Math.Max(maxX - minX, maxY - minY);
        if (!(extent > 0))
            return count <= 1;
        double eps = 1e-7 * extent;

        List<int> solid = new(count);
        for (int t = 0; t < count; t++)
        {
            double signed = SignedArea(uv, tri, t);
            double area3d = faceArea3d[chart.Faces[t]];
            if (area3d <= 0) continue;
            if (signed < -1e-6 * area3d)
                return false;
            if (signed > 1e-14 * extent * extent)
                solid.Add(t);
        }
        return !HasOverlap(uv, tri, solid, minX, minY, maxX, maxY, eps);
    }

    private static bool HasOverlap(double[] uv, int[] tri, List<int> solid, double minX, double minY, double maxX, double maxY, double eps)
    {
        int m = solid.Count;
        if (m < 2) return false;
        int grid = Math.Clamp((int)Math.Ceiling(Math.Sqrt(m)), 1, 1024);
        double cellW = Math.Max((maxX - minX) / grid, 1e-300), cellH = Math.Max((maxY - minY) / grid, 1e-300);
        double[] box = new double[m * 4];
        int[] cellStart = new int[grid * grid + 1];
        for (int k = 0; k < m; k++)
        {
            Bounds(uv, tri, solid[k], out double x0, out double y0, out double x1, out double y1);
            box[k * 4] = x0; box[k * 4 + 1] = y0; box[k * 4 + 2] = x1; box[k * 4 + 3] = y1;
            CellRange(x0, y0, x1, y1, minX, minY, cellW, cellH, grid, out int cx0, out int cy0, out int cx1, out int cy1);
            for (int cy = cy0; cy <= cy1; cy++)
                for (int cx = cx0; cx <= cx1; cx++)
                    cellStart[cy * grid + cx + 1]++;
        }
        for (int c = 0; c < grid * grid; c++)
            cellStart[c + 1] += cellStart[c];
        int[] cellItems = new int[cellStart[grid * grid]];
        int[] fill = new int[grid * grid];
        for (int k = 0; k < m; k++)
        {
            CellRange(box[k * 4], box[k * 4 + 1], box[k * 4 + 2], box[k * 4 + 3], minX, minY, cellW, cellH, grid,
                out int cx0, out int cy0, out int cx1, out int cy1);
            for (int cy = cy0; cy <= cy1; cy++)
                for (int cx = cx0; cx <= cx1; cx++)
                {
                    int cell = cy * grid + cx;
                    cellItems[cellStart[cell] + fill[cell]++] = k;
                }
        }

        for (int cy = 0; cy < grid; cy++)
        {
            for (int cx = 0; cx < grid; cx++)
            {
                int cell = cy * grid + cx;
                for (int p = cellStart[cell]; p < cellStart[cell + 1]; p++)
                {
                    int a = cellItems[p];
                    for (int q = p + 1; q < cellStart[cell + 1]; q++)
                    {
                        int b = cellItems[q];
                        double rx = Math.Max(box[a * 4], box[b * 4]), ry = Math.Max(box[a * 4 + 1], box[b * 4 + 1]);
                        if (Cell(rx, minX, cellW, grid) != cx || Cell(ry, minY, cellH, grid) != cy)
                            continue;
                        if (box[a * 4 + 2] <= box[b * 4] + eps || box[b * 4 + 2] <= box[a * 4] + eps
                            || box[a * 4 + 3] <= box[b * 4 + 1] + eps || box[b * 4 + 3] <= box[a * 4 + 1] + eps)
                            continue;
                        if (Overlaps(uv, tri, solid[a], solid[b], eps))
                            return true;
                    }
                }
            }
        }
        return false;
    }

    private static void Bounds(double[] uv, int[] tri, int t, out double x0, out double y0, out double x1, out double y1)
    {
        x0 = y0 = double.MaxValue;
        x1 = y1 = double.MinValue;
        for (int k = 0; k < 3; k++)
        {
            double x = uv[tri[t * 3 + k] * 2], y = uv[tri[t * 3 + k] * 2 + 1];
            x0 = Math.Min(x0, x); x1 = Math.Max(x1, x);
            y0 = Math.Min(y0, y); y1 = Math.Max(y1, y);
        }
    }

    private static int Cell(double v, double min, double size, int grid) => Math.Clamp((int)((v - min) / size), 0, grid - 1);

    private static void CellRange(double x0, double y0, double x1, double y1, double minX, double minY, double cellW, double cellH, int grid,
        out int cx0, out int cy0, out int cx1, out int cy1)
    {
        cx0 = Cell(x0, minX, cellW, grid); cx1 = Cell(x1, minX, cellW, grid);
        cy0 = Cell(y0, minY, cellH, grid); cy1 = Cell(y1, minY, cellH, grid);
    }

    private static double SignedArea(double[] uv, int[] tri, int t)
    {
        int a = tri[t * 3] * 2, b = tri[t * 3 + 1] * 2, c = tri[t * 3 + 2] * 2;
        return 0.5 * ((uv[b] - uv[a]) * (uv[c + 1] - uv[a + 1]) - (uv[c] - uv[a]) * (uv[b + 1] - uv[a + 1]));
    }

    // Separating-axis test over the six edge normals; intervals that touch within eps count as separated.
    private static bool Overlaps(double[] uv, int[] tri, int ta, int tb, double eps)
    {
        Span<double> pa = stackalloc double[6];
        Span<double> pb = stackalloc double[6];
        for (int k = 0; k < 3; k++)
        {
            pa[k * 2] = uv[tri[ta * 3 + k] * 2]; pa[k * 2 + 1] = uv[tri[ta * 3 + k] * 2 + 1];
            pb[k * 2] = uv[tri[tb * 3 + k] * 2]; pb[k * 2 + 1] = uv[tri[tb * 3 + k] * 2 + 1];
        }
        return !Separated(pa, pb, eps) && !Separated(pb, pa, eps);
    }

    private static bool Separated(ReadOnlySpan<double> a, ReadOnlySpan<double> b, double eps)
    {
        for (int e = 0; e < 3; e++)
        {
            int n = (e + 1) % 3;
            double ex = a[n * 2] - a[e * 2], ey = a[n * 2 + 1] - a[e * 2 + 1];
            double len = Math.Sqrt(ex * ex + ey * ey);
            if (len <= 0) continue;
            double ax = -ey / len, ay = ex / len;
            double minA = double.MaxValue, maxA = double.MinValue, minB = double.MaxValue, maxB = double.MinValue;
            for (int k = 0; k < 3; k++)
            {
                double pa = a[k * 2] * ax + a[k * 2 + 1] * ay, pb = b[k * 2] * ax + b[k * 2 + 1] * ay;
                minA = Math.Min(minA, pa); maxA = Math.Max(maxA, pa);
                minB = Math.Min(minB, pb); maxB = Math.Max(maxB, pb);
            }
            if (maxA <= minB + eps || maxB <= minA + eps)
                return true;
        }
        return false;
    }
}

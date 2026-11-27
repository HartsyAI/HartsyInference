namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// CPU rasterizer that draws a mesh into UV space. Texel <c>(x, y)</c> samples UV <c>((x + 0.5) / width, (y + 0.5) / height)</c>, so row
/// index grows with v. Winding is ignored; where triangles overlap the lowest triangle index wins, which keeps the result deterministic
/// regardless of thread scheduling.
/// </summary>
public static class Rasterizer
{
    private const int BandRows = 16;
    private const double Tolerance = 1e-9;

    /// <summary>Rasterizes <paramref name="indices"/> triangles whose vertices carry <paramref name="uvs"/> (u,v pairs).</summary>
    public static RasterizedUvMap Rasterize(ReadOnlySpan<float> uvs, ReadOnlySpan<int> indices, int width, int height,
        CancellationToken cancellationToken = default)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Raster dimensions must be positive.");
        if (indices.Length % 3 != 0)
            throw new ArgumentException("Indices must be whole triangles.", nameof(indices));
        int vertexCount = uvs.Length / 2;
        foreach (int i in indices)
            if ((uint)i >= (uint)vertexCount)
                throw new ArgumentException("A triangle index lies outside the UV array.", nameof(indices));

        float[] uvArray = uvs.ToArray();
        int[] idxArray = indices.ToArray();
        int triangles = idxArray.Length / 3;
        int bands = (height + BandRows - 1) / BandRows;

        int[] bandStart = new int[bands + 1];
        for (int t = 0; t < triangles; t++)
        {
            (int y0, int y1) = RowRange(uvArray, idxArray, t, height);
            for (int b = y0 / BandRows; y0 <= y1 && b <= y1 / BandRows; b++)
                bandStart[b + 1]++;
        }
        for (int b = 0; b < bands; b++)
            bandStart[b + 1] += bandStart[b];
        int[] bandTriangles = new int[bandStart[bands]];
        int[] cursor = new int[bands];
        for (int t = 0; t < triangles; t++)
        {
            (int y0, int y1) = RowRange(uvArray, idxArray, t, height);
            for (int b = y0 / BandRows; y0 <= y1 && b <= y1 / BandRows; b++)
                bandTriangles[bandStart[b] + cursor[b]++] = t;
        }

        RasterizedUvMap map = new(width, height);
        ParallelOptions options = new() { CancellationToken = cancellationToken };
        Parallel.For(0, bands, options, band =>
        {
            int rowLo = band * BandRows, rowHi = Math.Min(height, rowLo + BandRows) - 1;
            for (int s = bandStart[band]; s < bandStart[band + 1]; s++)
                DrawTriangle(uvArray, idxArray, bandTriangles[s], width, height, rowLo, rowHi, map);
        });
        return map;
    }

    private static (int Y0, int Y1) RowRange(float[] uvs, int[] idx, int t, int height)
    {
        float minV = float.MaxValue, maxV = float.MinValue;
        for (int k = 0; k < 3; k++)
        {
            float v = uvs[idx[t * 3 + k] * 2 + 1];
            minV = MathF.Min(minV, v);
            maxV = MathF.Max(maxV, v);
        }
        int y0 = Math.Max(0, (int)Math.Ceiling(minV * height - 0.5 - Tolerance));
        int y1 = Math.Min(height - 1, (int)Math.Floor(maxV * height - 0.5 + Tolerance));
        return (y0, y1);
    }

    private static void DrawTriangle(float[] uvs, int[] idx, int t, int width, int height, int rowLo, int rowHi, RasterizedUvMap map)
    {
        int i0 = idx[t * 3] * 2, i1 = idx[t * 3 + 1] * 2, i2 = idx[t * 3 + 2] * 2;
        double x0 = uvs[i0] * (double)width, y0 = uvs[i0 + 1] * (double)height;
        double x1 = uvs[i1] * (double)width, y1 = uvs[i1 + 1] * (double)height;
        double x2 = uvs[i2] * (double)width, y2 = uvs[i2 + 1] * (double)height;
        double area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (Math.Abs(area) < 1e-12)
            return;
        double invArea = 1.0 / area;

        double minX = Math.Min(x0, Math.Min(x1, x2)), maxX = Math.Max(x0, Math.Max(x1, x2));
        double minY = Math.Min(y0, Math.Min(y1, y2)), maxY = Math.Max(y0, Math.Max(y1, y2));
        int xa = Math.Max(0, (int)Math.Ceiling(minX - 0.5 - Tolerance)), xb = Math.Min(width - 1, (int)Math.Floor(maxX - 0.5 + Tolerance));
        int ya = Math.Max(rowLo, (int)Math.Ceiling(minY - 0.5 - Tolerance));
        int yb = Math.Min(rowHi, (int)Math.Floor(maxY - 0.5 + Tolerance));
        for (int y = ya; y <= yb; y++)
        {
            double py = y + 0.5;
            for (int x = xa; x <= xb; x++)
            {
                int texel = y * width + x;
                if (map.TriangleIds[texel] >= 0)
                    continue;
                double px = x + 0.5;
                double b1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) * invArea;
                double b2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) * invArea;
                double b0 = 1.0 - b1 - b2;
                if (b0 < -Tolerance || b1 < -Tolerance || b2 < -Tolerance)
                    continue;
                map.TriangleIds[texel] = t;
                map.Barycentric[texel * 2] = (float)b1;
                map.Barycentric[texel * 2 + 1] = (float)b2;
            }
        }
    }
}

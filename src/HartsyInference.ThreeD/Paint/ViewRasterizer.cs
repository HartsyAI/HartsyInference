namespace HartsyInference.ThreeD.Paint;

/// <summary>Depth-buffered CPU rasterizer matching Tencent's <c>custom_rasterizer</c>: vertices are mapped as
/// <c>x_pix = (ndc_x / 2 + 0.5) * (W - 1) + 0.5</c> (likewise y, no flip), depth as <c>ndc_z * 0.49999 + 0.5</c>, a pixel is covered when its
/// centre lies inside the triangle whatever the winding, the smallest quantised depth wins (<c>depth * 2^18</c>, truncated) and a tie goes
/// to the lowest triangle index. Banded over rows, so threading never changes the result.</summary>
public static class ViewRasterizer
{
    private const int BandRows = 16;
    private const double DepthScale = 2 << 17;

    /// <summary>Rasterizes triangles whose vertices are homogeneous clip positions (x, y, z, w interleaved).</summary>
    public static ViewRaster Rasterize(ReadOnlySpan<float> clip, ReadOnlySpan<int> indices, int width, int height,
        CancellationToken cancellationToken = default)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Raster dimensions must be positive.");
        if (indices.Length % 3 != 0)
            throw new ArgumentException("Indices must be whole triangles.", nameof(indices));
        int vertexCount = clip.Length / 4;
        foreach (int i in indices)
            if ((uint)i >= (uint)vertexCount)
                throw new ArgumentException("A triangle index lies outside the vertex array.", nameof(indices));

        int triangles = indices.Length / 3;
        float[] clipArray = clip.ToArray();
        int[] idx = indices.ToArray();
        double[] screen = new double[triangles * 9];
        for (int t = 0; t < triangles; t++)
            for (int k = 0; k < 3; k++)
            {
                int v = idx[t * 3 + k] * 4;
                double w = clipArray[v + 3];
                screen[t * 9 + k * 3] = (clipArray[v] / w * 0.5 + 0.5) * (width - 1) + 0.5;
                screen[t * 9 + k * 3 + 1] = (clipArray[v + 1] / w * 0.5 + 0.5) * (height - 1) + 0.5;
                screen[t * 9 + k * 3 + 2] = clipArray[v + 2] / w * 0.49999 + 0.5;
            }

        int bands = (height + BandRows - 1) / BandRows;
        int[] bandStart = new int[bands + 1];
        for (int t = 0; t < triangles; t++)
        {
            (int first, int last) = BandRange(screen, t, height);
            for (int band = first; band <= last; band++)
                bandStart[band + 1]++;
        }
        for (int band = 0; band < bands; band++)
            bandStart[band + 1] += bandStart[band];
        int[] bandTriangles = new int[bandStart[bands]];
        int[] cursor = new int[bands];
        for (int t = 0; t < triangles; t++)
        {
            (int first, int last) = BandRange(screen, t, height);
            for (int band = first; band <= last; band++)
                bandTriangles[bandStart[band] + cursor[band]++] = t;
        }

        ViewRaster raster = new(width, height);
        long[] zbuffer = new long[width * height];
        Array.Fill(zbuffer, long.MaxValue);
        ParallelOptions options = new() { CancellationToken = cancellationToken };
        Parallel.For(0, bands, options, band =>
        {
            int rowLo = band * BandRows, rowHi = Math.Min(height, rowLo + BandRows) - 1;
            for (int s = bandStart[band]; s < bandStart[band + 1]; s++)
                Draw(screen, bandTriangles[s], width, rowLo, rowHi, zbuffer);
            for (int y = rowLo; y <= rowHi; y++)
                for (int x = 0; x < width; x++)
                    Resolve(clipArray, idx, screen, zbuffer[y * width + x], y * width + x, raster);
        });
        return raster;
    }

    private static (int First, int Last) BandRange(double[] screen, int t, int height)
    {
        double lo = Math.Min(screen[t * 9 + 1], Math.Min(screen[t * 9 + 4], screen[t * 9 + 7]));
        double hi = Math.Max(screen[t * 9 + 1], Math.Max(screen[t * 9 + 4], screen[t * 9 + 7]));
        if (!(hi >= 0) || !(lo < height))
            return (1, 0);
        int y0 = Math.Max(0, (int)lo), y1 = Math.Min(height - 1, (int)hi);
        return y0 > y1 ? (1, 0) : (y0 / BandRows, y1 / BandRows);
    }

    private static void Draw(double[] s, int t, int width, int rowLo, int rowHi, long[] zbuffer)
    {
        int o = t * 9;
        double area = (s[o + 3] - s[o]) * (s[o + 7] - s[o + 1]) - (s[o + 6] - s[o]) * (s[o + 4] - s[o + 1]);
        if (Math.Abs(area) < 1e-12)
            return;
        double xMin = Math.Min(s[o], Math.Min(s[o + 3], s[o + 6])), xMax = Math.Max(s[o], Math.Max(s[o + 3], s[o + 6]));
        double yMin = Math.Min(s[o + 1], Math.Min(s[o + 4], s[o + 7])), yMax = Math.Max(s[o + 1], Math.Max(s[o + 4], s[o + 7]));
        for (int px = Math.Max(0, (int)xMin); px < xMax + 1 && px < width; px++)
        {
            for (int py = Math.Max(rowLo, (int)yMin); py < yMax + 1 && py <= rowHi; py++)
            {
                double qx = px + 0.5, qy = py + 0.5;
                double b1 = ((qx - s[o]) * (s[o + 7] - s[o + 1]) - (s[o + 6] - s[o]) * (qy - s[o + 1])) / area;
                double b2 = ((s[o + 3] - s[o]) * (qy - s[o + 1]) - (qx - s[o]) * (s[o + 4] - s[o + 1])) / area;
                double b0 = 1.0 - b1 - b2;
                if (b0 < 0 || b1 < 0 || b2 < 0)
                    continue;
                double depth = b0 * s[o + 2] + b1 * s[o + 5] + b2 * s[o + 8];
                long token = (long)(int)(depth * DepthScale) * 0x1_0000_0000L + (t + 1);
                int pixel = py * width + px;
                if (token < zbuffer[pixel])
                    zbuffer[pixel] = token;
            }
        }
    }

    private static void Resolve(float[] clip, int[] indices, double[] s, long token, int pixel, ViewRaster raster)
    {
        if (token == long.MaxValue)
            return;
        int face = (int)(token & 0xFFFF_FFFFL) - 1;
        raster.FaceIds[pixel] = face + 1;
        int o = face * 9;
        double qx = pixel % raster.Width + 0.5, qy = pixel / raster.Width + 0.5;
        double area = (s[o + 3] - s[o]) * (s[o + 7] - s[o + 1]) - (s[o + 6] - s[o]) * (s[o + 4] - s[o + 1]);
        double b1 = ((qx - s[o]) * (s[o + 7] - s[o + 1]) - (s[o + 6] - s[o]) * (qy - s[o + 1])) / area;
        double b2 = ((s[o + 3] - s[o]) * (qy - s[o + 1]) - (qx - s[o]) * (s[o + 4] - s[o + 1])) / area;
        double b0 = 1.0 - b1 - b2;
        double w0 = b0 / clip[indices[face * 3] * 4 + 3], w1 = b1 / clip[indices[face * 3 + 1] * 4 + 3], w2 = b2 / clip[indices[face * 3 + 2] * 4 + 3];
        double norm = 1.0 / (w0 + w1 + w2);
        raster.Barycentric[pixel * 3] = (float)(w0 * norm);
        raster.Barycentric[pixel * 3 + 1] = (float)(w1 * norm);
        raster.Barycentric[pixel * 3 + 2] = (float)(w2 * norm);
    }
}

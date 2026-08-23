namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Continuous sampling of 3D feature volumes and triplanes — the read primitive every implicit-field decoder needs.</summary>
public static class GridSampler
{
    /// <summary>Trilinearly samples a single-channel grid <c>[resZ, resY, resX]</c> (x fastest) at normalized coordinate <paramref name="u"/>,<paramref name="v"/>,<paramref name="w"/> in [0,1]; out-of-range coords clamp to the edge.</summary>
    public static float Trilinear(ReadOnlySpan<float> grid, int resX, int resY, int resZ, float u, float v, float w)
    {
        float fx = Math.Clamp(u, 0f, 1f) * (resX - 1);
        float fy = Math.Clamp(v, 0f, 1f) * (resY - 1);
        float fz = Math.Clamp(w, 0f, 1f) * (resZ - 1);
        int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy), z0 = (int)MathF.Floor(fz);
        int x1 = Math.Min(x0 + 1, resX - 1), y1 = Math.Min(y0 + 1, resY - 1), z1 = Math.Min(z0 + 1, resZ - 1);
        float tx = fx - x0, ty = fy - y0, tz = fz - z0;

        float c000 = grid[Lin(x0, y0, z0, resX, resY)], c100 = grid[Lin(x1, y0, z0, resX, resY)];
        float c010 = grid[Lin(x0, y1, z0, resX, resY)], c110 = grid[Lin(x1, y1, z0, resX, resY)];
        float c001 = grid[Lin(x0, y0, z1, resX, resY)], c101 = grid[Lin(x1, y0, z1, resX, resY)];
        float c011 = grid[Lin(x0, y1, z1, resX, resY)], c111 = grid[Lin(x1, y1, z1, resX, resY)];

        float c00 = c000 + (c100 - c000) * tx, c10 = c010 + (c110 - c010) * tx;
        float c01 = c001 + (c101 - c001) * tx, c11 = c011 + (c111 - c011) * tx;
        float c0 = c00 + (c10 - c00) * ty, c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    /// <summary>Bilinearly samples one feature plane <c>[channels, h, w]</c> (channel-major) at normalized (u,v) in [0,1]; call once per orthogonal plane and sum for a triplane decoder.</summary>
    public static void BilinearPlane(ReadOnlySpan<float> plane, int channels, int h, int w, float u, float v, Span<float> dst)
    {
        float fx = Math.Clamp(u, 0f, 1f) * (w - 1);
        float fy = Math.Clamp(v, 0f, 1f) * (h - 1);
        int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
        int x1 = Math.Min(x0 + 1, w - 1), y1 = Math.Min(y0 + 1, h - 1);
        float tx = fx - x0, ty = fy - y0;
        int plane2d = h * w;
        for (int c = 0; c < channels; c++)
        {
            int b = c * plane2d;
            float c00 = plane[b + y0 * w + x0], c10 = plane[b + y0 * w + x1];
            float c01 = plane[b + y1 * w + x0], c11 = plane[b + y1 * w + x1];
            float c0 = c00 + (c10 - c00) * tx, c1 = c01 + (c11 - c01) * tx;
            dst[c] = c0 + (c1 - c0) * ty;
        }
    }

    /// <summary>Samples one feature plane like PyTorch <c>F.grid_sample(align_corners=False, padding_mode="zeros", mode="bilinear")</c> — coords in [−1,1], out-of-plane samples contribute 0 (unlike the [0,1]+clamp <see cref="BilinearPlane"/>).</summary>
    public static void GridSamplePlane(ReadOnlySpan<float> plane, int channels, int h, int w, float gx, float gy, Span<float> dst)
    {
        // align_corners=False: pixel = ((coord+1)*size - 1) / 2
        float fx = ((gx + 1f) * w - 1f) * 0.5f;
        float fy = ((gy + 1f) * h - 1f) * 0.5f;
        int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
        int x1 = x0 + 1, y1 = y0 + 1;
        float tx = fx - x0, ty = fy - y0;
        float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
        bool x0ok = x0 >= 0 && x0 < w, x1ok = x1 >= 0 && x1 < w, y0ok = y0 >= 0 && y0 < h, y1ok = y1 >= 0 && y1 < h;
        int plane2d = h * w;
        for (int c = 0; c < channels; c++)
        {
            int b = c * plane2d;
            float acc = 0f;
            if (y0ok && x0ok) acc += w00 * plane[b + y0 * w + x0];
            if (y0ok && x1ok) acc += w10 * plane[b + y0 * w + x1];
            if (y1ok && x0ok) acc += w01 * plane[b + y1 * w + x0];
            if (y1ok && x1ok) acc += w11 * plane[b + y1 * w + x1];
            dst[c] = acc;
        }
    }

    private static int Lin(int x, int y, int z, int resX, int resY) => x + resX * (y + resY * z);
}

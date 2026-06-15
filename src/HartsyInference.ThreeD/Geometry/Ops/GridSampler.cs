namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Continuous sampling of 3D feature volumes and triplanes — the read primitive every
/// implicit-field decoder needs (occupancy/SDF grids today; triplane/NeRF models later). Pure managed C#.</summary>
public static class GridSampler
{
    /// <summary>Trilinearly samples a single-channel grid <c>[resZ, resY, resX]</c> (x fastest) at the
    /// normalized coordinate <paramref name="u"/>,<paramref name="v"/>,<paramref name="w"/> in [0,1].
    /// Out-of-range coords clamp to the edge.</summary>
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

    /// <summary>Bilinearly samples one feature plane <c>[channels, h, w]</c> (channel-major) at normalized
    /// (u,v) in [0,1], writing <paramref name="channels"/> values into <paramref name="dst"/>. The building
    /// block for triplane decoders (call three times, one per orthogonal plane, then sum).</summary>
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

    private static int Lin(int x, int y, int z, int resX, int resY) => x + resX * (y + resY * z);
}

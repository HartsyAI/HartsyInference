namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Trilinear sampling of a <see cref="SparseAttributeVolume"/>, equivalent to flex_gemm <c>grid_sample_3d(mode='trilinear')</c>:
/// voxel <c>i</c> is centered at <c>i + 0.5</c> in voxel units, a sample blends the eight voxel centers around it with tent weights
/// <c>1 - |center - p|</c> per axis, empty neighbors drop out, and the remaining weights are renormalized (no occupied neighbor yields zeros).
/// </summary>
public static class SparseAttributeSampler
{
    /// <summary>Samples at a position in voxel units (grid space, origin at the grid corner). Returns false, with zeros in
    /// <paramref name="destination"/>, when none of the eight neighbors is occupied.</summary>
    public static bool SampleGrid(SparseAttributeVolume volume, double gx, double gy, double gz, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(volume);
        int channels = volume.Channels;
        if (destination.Length < channels)
            throw new ArgumentException("Destination is shorter than the channel count.", nameof(destination));

        double fx = gx - 0.5, fy = gy - 0.5, fz = gz - 0.5;
        int x0 = (int)Math.Floor(fx), y0 = (int)Math.Floor(fy), z0 = (int)Math.Floor(fz);
        double tx = fx - x0, ty = fy - y0, tz = fz - z0;
        Span<double> acc = stackalloc double[Math.Min(channels, 64)];
        double[]? heap = channels > 64 ? new double[channels] : null;
        Span<double> sum = heap ?? acc;
        sum[..channels].Clear();

        double weightSum = 0;
        float[] attrs = volume.Attributes;
        for (int dz = 0; dz < 2; dz++)
        {
            double wz = dz == 0 ? 1 - tz : tz;
            for (int dy = 0; dy < 2; dy++)
            {
                double wy = dy == 0 ? 1 - ty : ty;
                for (int dx = 0; dx < 2; dx++)
                {
                    double w = (dx == 0 ? 1 - tx : tx) * wy * wz;
                    if (w <= 0 || !volume.TryFindIndex(x0 + dx, y0 + dy, z0 + dz, out int idx))
                        continue;
                    weightSum += w;
                    int b = idx * channels;
                    for (int c = 0; c < channels; c++)
                        sum[c] += w * attrs[b + c];
                }
            }
        }

        if (weightSum <= 1e-12)
        {
            destination[..channels].Clear();
            return false;
        }
        for (int c = 0; c < channels; c++)
            destination[c] = (float)(sum[c] / weightSum);
        return true;
    }

    /// <summary>Samples at a world-space position: <c>grid = (p - Origin) / VoxelSize</c>.</summary>
    public static bool SampleWorld(SparseAttributeVolume volume, double x, double y, double z, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(volume);
        double inv = 1.0 / volume.VoxelSize;
        return SampleGrid(volume, (x - volume.Origin.X) * inv, (y - volume.Origin.Y) * inv, (z - volume.Origin.Z) * inv, destination);
    }
}

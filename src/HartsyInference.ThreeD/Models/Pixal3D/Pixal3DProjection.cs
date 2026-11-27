using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Models.Pixal3D;

/// <summary>
/// Pixal3D's view-aligned feature lookup (upstream <c>ProjGrid</c>): each cell of an <c>R³</c> grid is placed at <c>linspace(-1, 1, R)</c>
/// per axis (corners included), turned to the cube's frame by <c>(x, y, z) → (x, -z, y)</c> and divided by <c>2 · meshScale</c>, projected
/// through the <see cref="Pixal3DView"/> pinhole into an image of <c>imageSize</c> pixels, and the feature map is sampled there with a
/// bilinear, border-clamped <c>grid_sample</c>. The feature map's own size does not enter: the coordinate is normalized by the image size.
/// </summary>
public static class Pixal3DProjection
{
    /// <summary>Normalized <c>grid_sample</c> coordinates <c>(u, v)</c> for every cell of a dense <paramref name="gridResolution"/>³ grid, in the
    /// flows' x-major token order.</summary>
    public static float[] DenseCoordinates(int gridResolution, Pixal3DView view, int imageSize)
    {
        Validate(gridResolution, view, imageSize);
        float[] uv = new float[gridResolution * gridResolution * gridResolution * 2];
        int o = 0;
        for (int x = 0; x < gridResolution; x++)
        {
            for (int y = 0; y < gridResolution; y++)
            {
                for (int z = 0; z < gridResolution; z++)
                {
                    (uv[o], uv[o + 1]) = Cell(x, y, z, gridResolution, view, imageSize);
                    o += 2;
                }
            }
        }

        return uv;
    }

    /// <summary>Normalized <c>grid_sample</c> coordinates for voxels given as <c>(batch, x, y, z)</c> rows on a <paramref name="gridResolution"/>³ grid.</summary>
    public static float[] CellCoordinates(ReadOnlySpan<int> coords, int gridResolution, Pixal3DView view, int imageSize)
    {
        Validate(gridResolution, view, imageSize);
        if (coords.Length % 4 != 0)
            throw new ArgumentException("Coordinates must be (batch, x, y, z) rows.", nameof(coords));
        int n = coords.Length / 4;
        float[] uv = new float[n * 2];
        for (int i = 0; i < n; i++)
        {
            for (int a = 1; a <= 3; a++)
            {
                if ((uint)coords[i * 4 + a] >= (uint)gridResolution)
                    throw new ArgumentOutOfRangeException(nameof(coords), $"Voxel {i} lies outside the {gridResolution} grid.");
            }

            (uv[i * 2], uv[i * 2 + 1]) = Cell(coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3], gridResolution, view, imageSize);
        }

        return uv;
    }

    /// <summary>Samples <paramref name="rows"/> (<c>[h · w, channels]</c>, row-major pixels) at <paramref name="uv"/>, giving <c>[1, cells, channels]</c>.</summary>
    public static Tensor Sample(ReadOnlySpan<float> rows, int channels, int h, int w, ReadOnlySpan<float> uv)
    {
        if (channels <= 0 || h <= 0 || w <= 0 || rows.Length != (long)h * w * channels)
            throw new ArgumentException($"{rows.Length} values do not form a {h}x{w} map of {channels} channels.", nameof(rows));
        if (uv.Length % 2 != 0)
            throw new ArgumentException("Sampling coordinates must be (u, v) pairs.", nameof(uv));
        int cells = uv.Length / 2;
        Tensor result = new(new TensorShape(1, cells, channels), DType.F32);
        Span<float> output = result.AsSpan<float>();
        for (int i = 0; i < cells; i++)
            GridSampler.GridSampleRowsBorder(rows, channels, h, w, uv[i * 2], uv[i * 2 + 1], output.Slice(i * channels, channels));
        return result;
    }

    // Upstream: gp = grid point rotated by (x, -z, y) over 2 · meshScale; x_pix = f · gp.x / depth + S/2; y_pix = -f · gp.z / depth + S/2 with
    // depth = gp.y + distance and f = S / (2 tan(fov / 2)); the sampler sees (pix + 0.5) / S · 2 - 1.
    private static (float U, float V) Cell(int x, int y, int z, int resolution, Pixal3DView view, int imageSize)
    {
        double step = 2.0 / (resolution - 1), scale = 2.0 * view.MeshScale;
        double gx = (-1.0 + x * step) / scale, gy = -(-1.0 + z * step) / scale, gz = (-1.0 + y * step) / scale;
        double focal = imageSize / (2.0 * Math.Tan(view.FovRadians / 2.0));
        double depth = gy + view.Distance;
        double px = focal * gx / depth + imageSize / 2.0, py = -focal * gz / depth + imageSize / 2.0;
        return ((float)((px + 0.5) / imageSize * 2.0 - 1.0), (float)((py + 0.5) / imageSize * 2.0 - 1.0));
    }

    private static void Validate(int gridResolution, Pixal3DView view, int imageSize)
    {
        if (gridResolution < 2)
            throw new ArgumentOutOfRangeException(nameof(gridResolution), gridResolution, "The projection grid needs at least 2 cells per axis.");
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, "The image size must be positive.");
        view.Validate();
    }
}

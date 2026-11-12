using System.Numerics;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Render;

/// <summary>Draws a point cloud (a splat file's centres, a scanned set) as round dots with a depth test, through the same camera as <see cref="MeshRenderer"/>. Dots are flat-coloured; a cloud without colours is drawn light grey.</summary>
public static class PointCloudRenderer
{
    /// <summary>Draws <paramref name="cloud"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range.</exception>
    /// <exception cref="ArgumentException">The cloud has no points.</exception>
    public static MeshRender Render(PointCloud cloud, MeshRenderOptions options)
    {
        ArgumentNullException.ThrowIfNull(cloud);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (cloud.Count == 0)
            throw new ArgumentException("The point cloud has no points.", nameof(cloud));

        Vector3[] points = new Vector3[cloud.Count];
        Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = OrbitCamera.ToYUp(new Vector3(cloud.Positions[i * 3], cloud.Positions[i * 3 + 1], cloud.Positions[i * 3 + 2]), options.Up);
            min = Vector3.Min(min, points[i]);
            max = Vector3.Max(max, points[i]);
        }
        SampleBuffer buffer = new(options);
        OrbitCamera camera = new(min, max, options, buffer.Width, buffer.Height);
        float radius = options.PointSize * options.Samples * 0.5f;
        int reach = (int)MathF.Ceiling(radius);
        for (int i = 0; i < points.Length; i++)
        {
            (float x, float y, float w) = camera.Project(points[i]);
            if (w <= 0)
                continue;
            byte r = 200, g = 200, b = 200;
            if (cloud.Colors is { } colours)
            {
                r = (byte)MathF.Round(Math.Clamp(colours[i * 3], 0f, 1f) * 255f);
                g = (byte)MathF.Round(Math.Clamp(colours[i * 3 + 1], 0f, 1f) * 255f);
                b = (byte)MathF.Round(Math.Clamp(colours[i * 3 + 2], 0f, 1f) * 255f);
            }
            int cx = (int)MathF.Floor(x), cy = (int)MathF.Floor(y);
            for (int dy = -reach; dy <= reach; dy++)
            {
                for (int dx = -reach; dx <= reach; dx++)
                {
                    int px = cx + dx, py = cy + dy;
                    if (px < 0 || py < 0 || px >= buffer.Width || py >= buffer.Height)
                        continue;
                    float ox = px + 0.5f - x, oy = py + 0.5f - y;
                    if (ox * ox + oy * oy <= radius * radius + 0.25f)
                        buffer.TryWrite(px, py, w, r, g, b);
                }
            }
        }
        return new MeshRender(buffer.Resolve(options), []);
    }
}

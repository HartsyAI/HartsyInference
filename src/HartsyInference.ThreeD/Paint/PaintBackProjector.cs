namespace HartsyInference.ThreeD.Paint;

/// <summary>Projects one generated view back onto the texels it sees (<c>MeshRender.back_project</c>, <c>back_sample</c> mode). A texel is
/// painted when its point is inside the view, matches the view's depth, faces the camera within the angle limit and lies away from the
/// silhouette and depth edges; the colour is a bilinear sample and the weight is the viewing-angle cosine.</summary>
public static class PaintBackProjector
{
    // A view whose visible depth spans less than this is flat; normalising it would only amplify rounding noise into edges.
    private const float FlatDepthRange = 1e-4f;

    /// <summary>Back-projects <paramref name="image"/> (<paramref name="channels"/> floats per pixel in [0, 1], square, row 0 first).</summary>
    public static PaintViewBake Project(PaintViewRenderer renderer, PaintTexelSurface surface, ReadOnlySpan<float> image, int resolution, int channels,
        double elevationDegrees, double azimuthDegrees, PaintBakeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(renderer);
        ArgumentNullException.ThrowIfNull(surface);
        if (image.Length != resolution * resolution * channels)
            throw new ArgumentException("The image size does not match resolution and channels.", nameof(image));
        options ??= new PaintBakeOptions();
        PaintMeshFrame frame = renderer.Frame;
        PaintViewProjection view = renderer.Project(elevationDegrees, azimuthDegrees);
        ViewRaster raster = ViewRasterizer.Rasterize(view.Clip, frame.Indices, resolution, resolution, cancellationToken);

        float[] depth = InterpolateDepth(raster, view.Camera, frame.Indices);
        float[] cosine = ViewCosines(raster, view.Camera, frame.Indices, options.AngleThresholdDegrees);
        bool[] trusted = TrustedMask(raster, depth, options, resolution);
        for (int p = 0; p < cosine.Length; p++)
            if (!trusted[p])
                cosine[p] = 0f;

        double scale = 1.0 / (PaintCamera.OrthoScale * 0.5);
        double[] w2c = view.WorldToCamera;
        PaintViewBake bake = new(surface.Count, channels);
        float[] imageArray = image.ToArray();
        ParallelOptions parallel = new() { CancellationToken = cancellationToken };
        Parallel.For(0, surface.Count, parallel, i =>
        {
            (double cx, double cy, double cz, _) = PaintCamera.Transform(w2c, surface.Positions[i * 3], surface.Positions[i * 3 + 1], surface.Positions[i * 3 + 2]);
            double vx = cx * scale, vy = cy * scale;
            if (vx > 1.0 || vx < -1.0 || vy > 1.0 || vy < -1.0)
                return;
            int px = Math.Clamp((int)((vx * 0.5 + 0.5) * resolution), 0, resolution - 1);
            int py = Math.Clamp((int)((vy * 0.5 + 0.5) * resolution), 0, resolution - 1);
            int pixel = py * resolution + px;
            if (!(Math.Abs(cz - depth[pixel]) < options.DepthThreshold) || !(cosine[pixel] > 0f))
                return;
            double wx = (vx * 0.5 + 0.5) * resolution - px, wy = (vy * 0.5 + 0.5) * resolution - py;
            int px1 = Math.Min(px + 1, resolution - 1), py1 = Math.Min(py + 1, resolution - 1);
            for (int c = 0; c < channels; c++)
            {
                double top = imageArray[(py * resolution + px) * channels + c] * (1 - wx) + imageArray[(py * resolution + px1) * channels + c] * wx;
                double bottom = imageArray[(py1 * resolution + px) * channels + c] * (1 - wx) + imageArray[(py1 * resolution + px1) * channels + c] * wx;
                bake.Colors[i * channels + c] = (float)(top * (1 - wy) + bottom * wy);
            }
            bake.Weights[i] = cosine[pixel];
        });
        return bake;
    }

    private static float[] InterpolateDepth(ViewRaster raster, float[] camera, int[] indices)
    {
        float[] z = new float[camera.Length / 3];
        for (int i = 0; i < z.Length; i++)
            z[i] = camera[i * 3 + 2];
        return raster.Interpolate(z, indices, 1);
    }

    private static float[] ViewCosines(ViewRaster raster, float[] camera, int[] indices, double angleThresholdDegrees)
    {
        float threshold = (float)Math.Cos(angleThresholdDegrees * Math.PI / 180.0);
        float[] faceCosine = new float[indices.Length / 3];
        for (int t = 0; t < faceCosine.Length; t++)
        {
            int a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
            float e1x = camera[b] - camera[a], e1y = camera[b + 1] - camera[a + 1], e1z = camera[b + 2] - camera[a + 2];
            float e2x = camera[c] - camera[a], e2y = camera[c + 1] - camera[a + 1], e2z = camera[c + 2] - camera[a + 2];
            float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
            float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
            float cos = len > 1e-12f ? -nz / len : 0f;
            faceCosine[t] = cos < threshold ? 0f : cos;
        }
        float[] cosine = new float[raster.FaceIds.Length];
        for (int p = 0; p < cosine.Length; p++)
            if (raster.FaceIds[p] > 0)
                cosine[p] = faceCosine[raster.FaceIds[p] - 1];
        return cosine;
    }

    private static bool[] TrustedMask(ViewRaster raster, float[] depth, PaintBakeOptions options, int resolution)
    {
        int count = raster.FaceIds.Length;
        bool[] visible = new bool[count];
        float min = float.MaxValue, max = float.MinValue;
        for (int p = 0; p < count; p++)
            if (raster.FaceIds[p] > 0)
            {
                visible[p] = true;
                min = MathF.Min(min, depth[p]);
                max = MathF.Max(max, depth[p]);
            }
        int radius = (int)(options.UnreliableBoundaryScale / 512.0 * resolution);
        if (radius <= 0)
            return visible;

        bool[] hole = new bool[count];
        byte[] depthImage = new byte[count];
        float range = max - min;
        for (int p = 0; p < count; p++)
        {
            hole[p] = !visible[p];
            if (visible[p])
                depthImage[p] = range > FlatDepthRange ? (byte)Math.Clamp((int)((depth[p] - min) / range * 255f), 0, 255) : (byte)0;
        }
        bool[] nearHole = PaintImageFilters.Dilate(hole, resolution, resolution, radius);
        bool[] nearEdge = PaintImageFilters.Dilate(
            PaintImageFilters.Edges(depthImage, resolution, resolution, options.DepthEdgeThresholds.Low, options.DepthEdgeThresholds.High),
            resolution, resolution, radius);
        for (int p = 0; p < count; p++)
            visible[p] &= !nearHole[p] && !nearEdge[p];
        return visible;
    }
}

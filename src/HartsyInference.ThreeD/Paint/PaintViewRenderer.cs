namespace HartsyInference.ThreeD.Paint;

/// <summary>Renders the geometry maps Paint conditions on from one orthographic view of a <see cref="PaintMeshFrame"/>: world-space face
/// normals, positions and the covering-face raster. Images are RGB bytes, row 0 first, white background, quantised by truncation as
/// the upstream PIL conversion does.</summary>
public sealed class PaintViewRenderer
{
    private readonly PaintMeshFrame _frame;
    private readonly double[] _projection = PaintCamera.Orthographic();

    /// <summary>Creates a renderer for <paramref name="frame"/>.</summary>
    public PaintViewRenderer(PaintMeshFrame frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        _frame = frame;
    }

    /// <summary>The mesh in render frame being drawn.</summary>
    public PaintMeshFrame Frame => _frame;

    /// <summary>Camera-space and clip-space vertices of a view.</summary>
    public PaintViewProjection Project(double elevationDegrees, double azimuthDegrees)
    {
        double[] w2c = PaintCamera.WorldToCamera(elevationDegrees, azimuthDegrees);
        int n = _frame.Vertices.Length / 3;
        float[] clip = new float[n * 4];
        float[] camera = new float[n * 3];
        for (int i = 0; i < n; i++)
        {
            (double cx, double cy, double cz, _) = PaintCamera.Transform(w2c, _frame.Vertices[i * 3], _frame.Vertices[i * 3 + 1], _frame.Vertices[i * 3 + 2]);
            (double x, double y, double z, double w) = PaintCamera.Transform(_projection, cx, cy, cz);
            clip[i * 4] = (float)x;
            clip[i * 4 + 1] = (float)y;
            clip[i * 4 + 2] = (float)z;
            clip[i * 4 + 3] = (float)w;
            camera[i * 3] = (float)cx;
            camera[i * 3 + 1] = (float)cy;
            camera[i * 3 + 2] = (float)cz;
        }
        return new PaintViewProjection(w2c, clip, camera);
    }

    /// <summary>Rasterizes the mesh from a view.</summary>
    public ViewRaster Rasterize(double elevationDegrees, double azimuthDegrees, int resolution, CancellationToken cancellationToken = default) =>
        ViewRasterizer.Rasterize(Project(elevationDegrees, azimuthDegrees).Clip, _frame.Indices, resolution, resolution, cancellationToken);

    /// <summary>Face normals in the render frame mapped to colours as <c>(n + 1) / 2</c>.</summary>
    public byte[] RenderNormalMap(double elevationDegrees, double azimuthDegrees, int resolution, CancellationToken cancellationToken = default)
    {
        ViewRaster raster = Rasterize(elevationDegrees, azimuthDegrees, resolution, cancellationToken);
        float[] normals = FaceNormals(_frame);
        float[] colors = new float[raster.FaceIds.Length * 3];
        for (int p = 0; p < raster.FaceIds.Length; p++)
        {
            int face = raster.FaceIds[p] - 1;
            for (int c = 0; c < 3; c++)
                colors[p * 3 + c] = face < 0 ? 1f : normals[face * 3 + c] * 0.5f + 0.5f;
        }
        return Quantize(colors);
    }

    /// <summary>Interpolated positions encoded as <c>0.5 - p / scaleFactor</c> per channel.</summary>
    public byte[] RenderPositionMap(double elevationDegrees, double azimuthDegrees, int resolution, CancellationToken cancellationToken = default)
    {
        ViewRaster raster = Rasterize(elevationDegrees, azimuthDegrees, resolution, cancellationToken);
        float[] encoded = new float[_frame.Vertices.Length];
        for (int i = 0; i < encoded.Length; i++)
            encoded[i] = (float)(0.5 - _frame.Vertices[i] / _frame.ScaleFactor);
        float[] interpolated = raster.Interpolate(encoded, _frame.Indices, 3);
        for (int p = 0; p < raster.FaceIds.Length; p++)
            if (raster.FaceIds[p] == 0)
                interpolated[p * 3] = interpolated[p * 3 + 1] = interpolated[p * 3 + 2] = 1f;
        return Quantize(interpolated);
    }

    /// <summary>Unit face normals (area-independent), three per triangle.</summary>
    public static float[] FaceNormals(PaintMeshFrame frame)
    {
        int triangles = frame.TriangleCount;
        float[] normals = new float[triangles * 3];
        for (int t = 0; t < triangles; t++)
        {
            int a = frame.Indices[t * 3] * 3, b = frame.Indices[t * 3 + 1] * 3, c = frame.Indices[t * 3 + 2] * 3;
            float e1x = frame.Vertices[b] - frame.Vertices[a], e1y = frame.Vertices[b + 1] - frame.Vertices[a + 1], e1z = frame.Vertices[b + 2] - frame.Vertices[a + 2];
            float e2x = frame.Vertices[c] - frame.Vertices[a], e2y = frame.Vertices[c + 1] - frame.Vertices[a + 1], e2z = frame.Vertices[c + 2] - frame.Vertices[a + 2];
            float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
            float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-12f)
            {
                normals[t * 3] = nx / len;
                normals[t * 3 + 1] = ny / len;
                normals[t * 3 + 2] = nz / len;
            }
        }
        return normals;
    }

    private static byte[] Quantize(float[] colors)
    {
        byte[] image = new byte[colors.Length];
        for (int i = 0; i < colors.Length; i++)
            image[i] = (byte)Math.Clamp((int)(colors[i] * 255f), 0, 255);
        return image;
    }
}

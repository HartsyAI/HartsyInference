using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Paint;

/// <summary>A mesh moved into the frame Paint renders in (<c>MeshRender.set_mesh</c>): <c>(x, y, z) -> (-x, z, -y)</c>, centred on the
/// bounding-box centre and scaled so the farthest vertex sits at <c>scaleFactor / 2</c>. Keeps what is needed to map back.</summary>
public sealed class PaintMeshFrame
{
    /// <summary>Normalisation factor upstream uses.</summary>
    public const double DefaultScaleFactor = 1.15;

    private PaintMeshFrame(float[] vertices, int[] indices, double scaleFactor, double scale, double[] center)
    {
        Vertices = vertices;
        Indices = indices;
        ScaleFactor = scaleFactor;
        Scale = scale;
        Center = center;
    }

    /// <summary>Vertices in the render frame, <c>3 * count</c>.</summary>
    public float[] Vertices { get; }

    /// <summary>Triangle indices, unchanged.</summary>
    public int[] Indices { get; }

    /// <summary>Scale factor the farthest vertex is normalised to (diameter).</summary>
    public double ScaleFactor { get; }

    /// <summary>Applied uniform scale, <c>ScaleFactor / (2 * max |v - c|)</c>.</summary>
    public double Scale { get; }

    /// <summary>Centre in the axis-swapped frame before scaling.</summary>
    public double[] Center { get; }

    /// <summary>Number of triangles.</summary>
    public int TriangleCount => Indices.Length / 3;

    /// <summary>Builds the render-frame copy of <paramref name="mesh"/>.</summary>
    public static PaintMeshFrame Create(Mesh mesh, double scaleFactor = DefaultScaleFactor)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (mesh.VertexCount == 0 || mesh.Indices.Length == 0)
            throw new ArgumentException("The mesh is empty.", nameof(mesh));
        int n = mesh.VertexCount;
        double[] swapped = new double[n * 3];
        double[] min = [double.MaxValue, double.MaxValue, double.MaxValue], max = [double.MinValue, double.MinValue, double.MinValue];
        for (int i = 0; i < n; i++)
        {
            double x = mesh.Vertices[i * 3], y = mesh.Vertices[i * 3 + 1], z = mesh.Vertices[i * 3 + 2];
            double[] p = [-x, z, -y];
            for (int c = 0; c < 3; c++)
            {
                swapped[i * 3 + c] = p[c];
                min[c] = Math.Min(min[c], p[c]);
                max[c] = Math.Max(max[c], p[c]);
            }
        }
        double[] center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
        double radius = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = swapped[i * 3] - center[0], dy = swapped[i * 3 + 1] - center[1], dz = swapped[i * 3 + 2] - center[2];
            radius = Math.Max(radius, Math.Sqrt(dx * dx + dy * dy + dz * dz));
        }
        if (radius <= 0)
            throw new ArgumentException("The mesh has no extent.", nameof(mesh));
        double scale = scaleFactor / (radius * 2.0);
        float[] vertices = new float[n * 3];
        for (int i = 0; i < n * 3; i++)
            vertices[i] = (float)((swapped[i] - center[i % 3]) * scale);
        return new PaintMeshFrame(vertices, mesh.Indices, scaleFactor, scale, center);
    }
}

using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

public sealed class Hunyuan3DSurfaceTests
{
    private const int Resolution = 33;
    private const float Radius = 0.6f;

    [Fact]
    public void Extract_WindsTrianglesOutwardForLogitsThatArePositiveInside()
    {
        Mesh mesh = Hunyuan3DSurface.Extract(SphereLogits(), 0f);

        double expected = 4.0 / 3.0 * Math.PI * Radius * Radius * Radius;
        double volume = SignedVolume(mesh);
        Assert.True(volume > 0, $"signed volume {volume} is negative: triangles face inward.");
        Assert.InRange(volume, expected * 0.9, expected * 1.1);
    }

    [Fact]
    public void Extract_ReturnsAClosedSurface()
    {
        Mesh mesh = Hunyuan3DSurface.Extract(SphereLogits(), 0f);

        Dictionary<(int, int), int> edgeUses = [];
        for (int i = 0; i < mesh.Indices.Length; i += 3)
        {
            for (int k = 0; k < 3; k++)
            {
                int a = mesh.Indices[i + k], b = mesh.Indices[i + (k + 1) % 3];
                (int, int) key = a < b ? (a, b) : (b, a);
                edgeUses[key] = edgeUses.GetValueOrDefault(key) + 1;
            }
        }

        Assert.NotEmpty(edgeUses);
        Assert.All(edgeUses.Values, uses => Assert.Equal(2, uses));
    }

    [Fact]
    public void Extract_FillsOutwardVertexNormals()
    {
        Mesh mesh = Hunyuan3DSurface.Extract(SphereLogits(), 0f);

        Assert.NotNull(mesh.Normals);
        float[] normals = mesh.Normals!;
        for (int v = 0; v < mesh.VertexCount; v++)
        {
            float dot = mesh.Vertices[v * 3] * normals[v * 3] + mesh.Vertices[v * 3 + 1] * normals[v * 3 + 1]
                + mesh.Vertices[v * 3 + 2] * normals[v * 3 + 2];
            Assert.True(dot > 0, $"vertex {v} normal points toward the centre.");
        }
    }

    private static ScalarField3D SphereLogits()
    {
        float[] values = new float[Resolution * Resolution * Resolution];
        ScalarField3D field = new()
        {
            Values = values, ResX = Resolution, ResY = Resolution, ResZ = Resolution,
            Min = (-1f, -1f, -1f), Max = (1f, 1f, 1f),
        };
        for (int z = 0; z < Resolution; z++)
        {
            for (int y = 0; y < Resolution; y++)
            {
                for (int x = 0; x < Resolution; x++)
                {
                    (float px, float py, float pz) = field.WorldOf(x, y, z);
                    values[field.Index(x, y, z)] = Radius - MathF.Sqrt(px * px + py * py + pz * pz);
                }
            }
        }

        return field;
    }

    private static double SignedVolume(Mesh mesh)
    {
        double volume = 0;
        for (int i = 0; i < mesh.Indices.Length; i += 3)
        {
            (double ax, double ay, double az) = Vertex(mesh, mesh.Indices[i]);
            (double bx, double by, double bz) = Vertex(mesh, mesh.Indices[i + 1]);
            (double cx, double cy, double cz) = Vertex(mesh, mesh.Indices[i + 2]);
            volume += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
        }

        return volume / 6.0;
    }

    private static (double, double, double) Vertex(Mesh mesh, int index) =>
        (mesh.Vertices[index * 3], mesh.Vertices[index * 3 + 1], mesh.Vertices[index * 3 + 2]);
}

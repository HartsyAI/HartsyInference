using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Tests;

public sealed class ChainTubeMesherTests
{
    private const double Radius = ChainTubeMesher.DefaultRadius;
    private const int TubeVertices = 18;
    private const int TubeFaces = 32;
    private const int SphereVertices = 42;
    private const int SphereFaces = 80;

    [Fact]
    public void Single_segment_chain_is_a_tube_with_two_joint_spheres()
    {
        Mesh? mesh = ChainTubeMesher.Build([[0f, 0f, 0f, 0f, 0f, 0.1f]]);

        Assert.NotNull(mesh);
        Assert.Equal(TubeVertices + 2 * SphereVertices, mesh.VertexCount);
        Assert.Equal(TubeFaces + 2 * SphereFaces, mesh.TriangleCount);
    }

    [Fact]
    public void Tube_vertices_lie_on_the_tube_radius_around_the_segment()
    {
        Mesh? mesh = ChainTubeMesher.Build([[0f, 0f, 0f, 0f, 0f, 0.1f]]);

        Assert.NotNull(mesh);
        double halfLength = 0.05;
        for (int i = 0; i < TubeVertices; i++)
        {
            double x = mesh.Vertices[3 * i];
            double y = mesh.Vertices[3 * i + 1];
            double z = mesh.Vertices[3 * i + 2];
            Assert.InRange(z, halfLength - 0.05 - 1e-6, halfLength + 0.05 + 1e-6);
            double distance = Math.Sqrt(x * x + y * y);
            Assert.True(Math.Abs(distance - Radius) < 1e-7 || distance < 1e-7, $"vertex {i} at {distance}");
        }
    }

    [Fact]
    public void Diagonal_segment_tube_keeps_the_radius_about_its_axis()
    {
        double[] a = [0.01, 0.02, 0.03];
        double[] b = [0.07, -0.01, 0.05];
        Mesh? mesh = ChainTubeMesher.Build([ToFloat(a, b)]);

        Assert.NotNull(mesh);
        double[] direction = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        double length = Math.Sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
        double[] unit = [direction[0] / length, direction[1] / length, direction[2] / length];
        for (int i = 0; i < TubeVertices; i++)
        {
            double[] p = [mesh.Vertices[3 * i] - a[0], mesh.Vertices[3 * i + 1] - a[1], mesh.Vertices[3 * i + 2] - a[2]];
            double along = p[0] * unit[0] + p[1] * unit[1] + p[2] * unit[2];
            double rx = p[0] - along * unit[0];
            double ry = p[1] - along * unit[1];
            double rz = p[2] - along * unit[2];
            double radial = Math.Sqrt(rx * rx + ry * ry + rz * rz);
            Assert.True(Math.Abs(radial - Radius) < 1e-6 || radial < 1e-6, $"vertex {i} radial {radial}");
        }
    }

    [Fact]
    public void Joint_spheres_lie_on_1_9_times_the_tube_radius()
    {
        Mesh? mesh = ChainTubeMesher.Build([[0f, 0f, 0f, 0f, 0f, 0.1f]]);

        Assert.NotNull(mesh);
        double sphereRadius = 1.9 * Radius;
        double[] centres = [0.0, 0.1];
        for (int s = 0; s < centres.Length; s++)
        {
            for (int i = TubeVertices + s * SphereVertices; i < TubeVertices + (s + 1) * SphereVertices; i++)
            {
                double x = mesh.Vertices[3 * i];
                double y = mesh.Vertices[3 * i + 1];
                double z = mesh.Vertices[3 * i + 2] - centres[s];
                Assert.Equal(sphereRadius, Math.Sqrt(x * x + y * y + z * z), 7);
            }
        }
    }

    [Fact]
    public void Zero_length_segments_add_only_joint_spheres()
    {
        Mesh? mesh = ChainTubeMesher.Build([[0.1f, 0f, 0f, 0.1f, 0f, 0f]]);

        Assert.NotNull(mesh);
        Assert.Equal(2 * SphereVertices, mesh.VertexCount);
        Assert.Equal(2 * SphereFaces, mesh.TriangleCount);
    }

    [Fact]
    public void Indices_reference_only_existing_vertices()
    {
        Mesh? mesh = ChainTubeMesher.Build([[0f, 0f, 0f, 0.05f, 0f, 0f, 0.05f, 0.05f, 0f], [0f, 0f, 0.1f, 0f, 0.1f, 0.1f]]);

        Assert.NotNull(mesh);
        Assert.All(mesh.Indices, index => Assert.InRange(index, 0, mesh.VertexCount - 1));
        Assert.Equal(0, mesh.Indices.Length % 3);
    }

    [Fact]
    public void Empty_chain_list_returns_null()
    {
        Assert.Null(ChainTubeMesher.Build([]));
    }

    [Fact]
    public void Joints_must_be_xyz_triples()
    {
        Assert.Throws<ArgumentException>(() => ChainTubeMesher.Build([[0f, 0f]]));
    }

    private static float[] ToFloat(double[] a, double[] b) =>
        [(float)a[0], (float)a[1], (float)a[2], (float)b[0], (float)b[1], (float)b[2]];
}

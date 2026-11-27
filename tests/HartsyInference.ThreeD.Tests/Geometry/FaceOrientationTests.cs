using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class FaceOrientationTests
{
    // Outward-facing counter-clockwise octahedron: +x, -x, +y, -y, +z, -z.
    private static Mesh Octahedron(float offsetX = 0f) => new()
    {
        Vertices = [1f + offsetX, 0f, 0f, -1f + offsetX, 0f, 0f, offsetX, 1f, 0f, offsetX, -1f, 0f, offsetX, 0f, 1f, offsetX, 0f, -1f],
        Indices = [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5],
    };

    [Fact]
    public void ConsistentOutwardMesh_IsLeftAlone()
    {
        Mesh mesh = Octahedron();
        int[] before = (int[])mesh.Indices.Clone();

        int flipped = FaceOrientation.Unify(mesh);

        Assert.Equal(0, flipped);
        Assert.Equal(before, mesh.Indices);
    }

    [Fact]
    public void RandomlyFlippedClosedMesh_BecomesConsistentAndOutward()
    {
        Mesh mesh = Octahedron();
        Random random = new(7);
        for (int f = 0; f < mesh.TriangleCount; f++)
        {
            if (random.Next(2) == 0)
                Flip(mesh, f);
        }

        FaceOrientation.Unify(mesh);

        Assert.True(EveryEdgeIsTraversedOncePerDirection(mesh));
        Assert.True(SignedVolume(mesh) > 0.0);
    }

    [Fact]
    public void InsideOutClosedMesh_IsTurnedOutward()
    {
        Mesh mesh = Octahedron();
        for (int f = 0; f < mesh.TriangleCount; f++)
            Flip(mesh, f);
        Assert.True(SignedVolume(mesh) < 0.0);

        int flipped = FaceOrientation.Unify(mesh);

        Assert.Equal(mesh.TriangleCount, flipped);
        Assert.True(SignedVolume(mesh) > 0.0);
    }

    [Fact]
    public void SeparatePatches_AreOrientedIndependently()
    {
        Mesh a = Octahedron();
        Mesh b = Octahedron(offsetX: 10f);
        int[] indices = [.. a.Indices, .. b.Indices.Select(i => i + 6)];
        Mesh mesh = new() { Vertices = [.. a.Vertices, .. b.Vertices], Indices = indices };
        for (int f = 8; f < 16; f++)
            Flip(mesh, f);

        FaceOrientation.Unify(mesh);

        Assert.True(EveryEdgeIsTraversedOncePerDirection(mesh));
        Assert.True(SignedVolume(mesh, 0, 8) > 0.0);
        Assert.True(SignedVolume(mesh, 8, 16) > 0.0);
    }

    [Fact]
    public void OpenPatch_IsMadeConsistentWithoutLosingTriangles()
    {
        // A 3x3 vertex grid of 8 triangles with alternating winding: no closed volume, only consistency can be restored.
        float[] vertices = new float[9 * 3];
        for (int i = 0; i < 9; i++)
        {
            vertices[i * 3] = i % 3;
            vertices[i * 3 + 1] = i / 3;
        }

        List<int> indices = [];
        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 2; x++)
            {
                int v = y * 3 + x;
                indices.AddRange([v, v + 1, v + 4, v, v + 4, v + 3]);
            }
        }

        Mesh mesh = new() { Vertices = vertices, Indices = [.. indices] };
        Flip(mesh, 1);
        Flip(mesh, 4);

        FaceOrientation.Unify(mesh);

        Assert.Equal(8, mesh.TriangleCount);
        Assert.True(EveryEdgeIsTraversedOncePerDirection(mesh));
    }

    [Fact]
    public void DualGridCube_IsInconsistentFromTheMesherAndConsistentOutwardAfterUnify()
    {
        // A dual-grid quad is wound the same way whichever side is inside, so the six faces of a closed block disagree.
        int[] coords = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1];
        DualGridMesh raw = DualGridMesher.Extract(
            coords, Enumerable.Repeat(0.5f, 24).ToArray(), Enumerable.Repeat(true, 24).ToArray(), null, Vector3.Zero, Vector3.One);
        Mesh mesh = new() { Vertices = raw.Vertices, Indices = raw.Triangles };
        Assert.False(EveryEdgeIsTraversedOncePerDirection(mesh));

        FaceOrientation.Unify(mesh);

        Assert.Equal(12, mesh.TriangleCount);
        Assert.True(EveryEdgeIsTraversedOncePerDirection(mesh));
        Assert.True(SignedVolume(mesh) > 0.0);
    }

    [Fact]
    public void EmptyMesh_FlipsNothing()
    {
        Mesh mesh = new() { Vertices = [], Indices = [] };

        Assert.Equal(0, FaceOrientation.Unify(mesh));
    }

    private static void Flip(Mesh mesh, int face) => (mesh.Indices[face * 3 + 1], mesh.Indices[face * 3 + 2]) = (mesh.Indices[face * 3 + 2], mesh.Indices[face * 3 + 1]);

    // In a consistently wound mesh no directed edge appears twice.
    private static bool EveryEdgeIsTraversedOncePerDirection(Mesh mesh)
    {
        HashSet<(int, int)> seen = [];
        for (int f = 0; f < mesh.TriangleCount; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                if (!seen.Add((mesh.Indices[f * 3 + k], mesh.Indices[f * 3 + (k + 1) % 3])))
                    return false;
            }
        }

        return true;
    }

    private static double SignedVolume(Mesh mesh) => SignedVolume(mesh, 0, mesh.TriangleCount);

    private static double SignedVolume(Mesh mesh, int firstFace, int endFace)
    {
        double volume = 0.0;
        for (int f = firstFace; f < endFace; f++)
        {
            Vector3 a = Vertex(mesh, mesh.Indices[f * 3]), b = Vertex(mesh, mesh.Indices[f * 3 + 1]), c = Vertex(mesh, mesh.Indices[f * 3 + 2]);
            volume += Vector3.Dot(a, Vector3.Cross(b, c)) / 6.0;
        }

        return volume;
    }

    private static Vector3 Vertex(Mesh mesh, int index) => new(mesh.Vertices[index * 3], mesh.Vertices[index * 3 + 1], mesh.Vertices[index * 3 + 2]);
}

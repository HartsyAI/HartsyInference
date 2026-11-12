using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: each cleaning step on a purpose-built defect.</summary>
public sealed class MeshCleanerTests
{
    private static Mesh Tetrahedron(float x = 0, float scale = 1f) => new()
    {
        Vertices = [x, 0, 0, x + scale, 0, 0, x, scale, 0, x, 0, scale],
        Indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
    };

    [Fact]
    public void RemoveDuplicateFaces_DropsRepeatsInEitherWinding()
    {
        Mesh tet = Tetrahedron();
        Mesh doubled = new() { Vertices = tet.Vertices, Indices = [.. tet.Indices, 0, 2, 1, 1, 2, 0] };

        Mesh cleaned = MeshCleaner.RemoveDuplicateFaces(doubled);

        Assert.Equal(4, cleaned.TriangleCount);
        Assert.True(SyntheticMeshes.IsClosedManifold(cleaned));
    }

    [Fact]
    public void RemoveDegenerateFaces_DropsRepeatedVerticesAndZeroArea()
    {
        Mesh tet = Tetrahedron();
        Mesh dirty = new()
        {
            Vertices = [.. tet.Vertices, 2, 2, 2, 3, 3, 3],
            Indices = [.. tet.Indices, 0, 0, 1, 0, 4, 5],
        };

        Assert.Equal(4, MeshCleaner.RemoveDegenerateFaces(dirty).TriangleCount);
    }

    // Three fins share the edge 0-1; the fin opposite in winding to the first should be kept with it.
    private static Mesh ThreeFins() => new()
    {
        Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1],
        Indices = [0, 1, 2, 1, 0, 3, 0, 1, 4],
    };

    [Fact]
    public void RemoveNonManifoldFaces_KeepsAtMostTwoFacesPerEdge()
    {
        Mesh fins = ThreeFins();
        Assert.Equal(1, SyntheticMeshes.Topology(fins).NonManifold);

        Mesh repaired = MeshCleaner.RemoveNonManifoldFaces(fins);

        Assert.Equal(2, repaired.TriangleCount);
        Assert.Equal(0, SyntheticMeshes.Topology(repaired).NonManifold);
        Assert.True(SyntheticMeshes.Topology(repaired).ConsistentWinding);
    }

    [Fact]
    public void RepairNonManifoldEdges_SplitsVerticesAndKeepsEveryFace()
    {
        Mesh fins = ThreeFins();

        Mesh repaired = MeshCleaner.RepairNonManifoldEdges(fins);

        // All three faces survive; the third fin is cut off along the edge with copies of its two vertices.
        Assert.Equal(3, repaired.TriangleCount);
        Assert.Equal(fins.VertexCount + 2, repaired.VertexCount);
        Assert.Equal(0, SyntheticMeshes.Topology(repaired).NonManifold);
        Assert.True(SyntheticMeshes.Topology(repaired).ConsistentWinding);
        for (int f = 0; f < 3; f++)
            for (int c = 0; c < 3; c++)
                for (int axis = 0; axis < 3; axis++)
                    Assert.Equal(fins.Vertices[fins.Indices[f * 3 + c] * 3 + axis], repaired.Vertices[repaired.Indices[f * 3 + c] * 3 + axis]);
    }

    // Three fans of triangles hang off the line v0-v1-v2-v3, so each of its three edges has one face per fan. Within a fan the triangles
    // are joined through manifold edges (at v1 and v2 by a connector triangle). Fan A is wound the other way, so the pair kept at an edge
    // is A with whichever of B and C has the larger triangle there. heights[fan][i] is the height of that fan's triangle on edge i.
    private static Mesh ThreeFanChain(float[][] heights)
    {
        List<float> vertices = [];
        for (int i = 0; i < 4; i++) vertices.AddRange([i, 0f, 0f]);
        List<int> indices = [];
        for (int fan = 0; fan < 3; fan++)
        {
            double angle = fan * 2 * Math.PI / 3;
            int[] apex = new int[3];
            for (int i = 0; i < 3; i++)
            {
                apex[i] = vertices.Count / 3;
                vertices.AddRange([i + 0.5f, heights[fan][i] * (float)Math.Cos(angle), heights[fan][i] * (float)Math.Sin(angle)]);
            }
            List<int[]> triangles = [[0, 1, apex[0]], [1, 2, apex[1]], [2, 3, apex[2]], [apex[0], 1, apex[1]], [apex[1], 2, apex[2]]];
            foreach (int[] t in triangles)
                indices.AddRange(fan == 0 ? [t[2], t[1], t[0]] : t);
        }
        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    [Fact]
    public void RepairNonManifoldEdges_CutsAnEdgeWhoseKeptPairIsJoinedBackByItsNeighbours()
    {
        // The kept pair is (A, B) on the first edge, (A, C) on the middle one and (A, B) on the last. Joins made for the first and last edge
        // link A, B and C at v1 and v2 alike, so the middle edge would keep all three faces unless it is cut.
        Mesh chain = ThreeFanChain([[1f, 1f, 1f], [2f, 1f, 2f], [1f, 2f, 1f]]);
        Assert.Equal(3, SyntheticMeshes.Topology(chain).NonManifold);

        Mesh repaired = MeshCleaner.RepairNonManifoldEdges(chain);

        Assert.Equal(chain.TriangleCount, repaired.TriangleCount);
        Assert.Equal(0, SyntheticMeshes.Topology(repaired).NonManifold);
    }

    [Fact]
    public void RepairNonManifoldEdges_LeavesNoEdgeWithMoreThanTwoFaces_ForEveryHeightPattern()
    {
        float[] values = [1f, 2f, 3f];
        for (int code = 0; code < 19683; code++)
        {
            float[][] heights = [new float[3], new float[3], new float[3]];
            int c = code;
            for (int fan = 0; fan < 3; fan++)
                for (int i = 0; i < 3; i++, c /= 3)
                    heights[fan][i] = values[c % 3];
            Mesh chain = ThreeFanChain(heights);

            Mesh repaired = MeshCleaner.RepairNonManifoldEdges(chain);

            Assert.True(SyntheticMeshes.Topology(repaired).NonManifold == 0, $"heights {string.Join("|", heights.Select(h => string.Join(",", h)))}");
            Assert.Equal(chain.TriangleCount, repaired.TriangleCount);
        }
    }

    [Fact]
    public void RepairNonManifoldEdges_ReturnsACompactedCopy()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(1);
        Mesh withStray = new() { Vertices = [.. sphere.Vertices, 9f, 9f, 9f], Indices = sphere.Indices };

        Mesh repaired = MeshCleaner.RepairNonManifoldEdges(withStray);

        Assert.NotSame(withStray, repaired);
        Assert.Equal(sphere.VertexCount, repaired.VertexCount);
    }

    [Fact]
    public void FillHoles_DoesNotCapATriangleWithADuplicateOfItself()
    {
        // A fin split off a surface is a lone triangle whose three edges are a boundary loop; capping it would add the same triangle again.
        Mesh fin = new() { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2] };

        Mesh filled = MeshCleaner.FillHoles(fin, 100);

        Assert.Equal(1, filled.TriangleCount);
    }

    [Fact]
    public void RepairNonManifoldEdges_ReturnsAManifoldMeshUnchanged()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);

        Mesh repaired = MeshCleaner.RepairNonManifoldEdges(sphere);

        Assert.Equal(sphere.Indices, repaired.Indices);
        Assert.Equal(sphere.Vertices, repaired.Vertices);
    }

    [Fact]
    public void Clean_KeepsTheSurfaceClosedWhereAFinMeetsIt()
    {
        // A fin glued to one edge of a closed sphere makes that edge non-manifold. Dropping a face (the old repair) opened a crack in the
        // sphere or lost the fin; splitting vertices leaves the sphere closed and the fin as its own three boundary edges.
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        int a = sphere.Indices[0], b = sphere.Indices[1];
        float[] vertices = [.. sphere.Vertices, 3f, 3f, 3f];
        int[] indices = [.. sphere.Indices, b, a, sphere.VertexCount];
        Mesh withFin = new() { Vertices = vertices, Indices = indices };
        Assert.Equal(1, SyntheticMeshes.Topology(withFin).NonManifold);

        Mesh cleaned = MeshCleaner.Clean(withFin, new MeshCleanOptions { SmallComponentAreaRatio = 0, HoleFillMaxPerimeter = 0, UnifyOrientation = false });

        (int _, int boundary, int nonManifold, bool _) = SyntheticMeshes.Topology(cleaned);
        Assert.Equal(sphere.TriangleCount + 1, cleaned.TriangleCount);
        Assert.Equal(0, nonManifold);
        Assert.Equal(3, boundary);
    }

    [Fact]
    public void RemoveSmallComponents_DropsOnlyFragmentsBelowTheAreaRatio()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        Mesh speck = Tetrahedron(10f, 0.01f);
        int offset = sphere.VertexCount;
        Mesh combined = new()
        {
            Vertices = [.. sphere.Vertices, .. speck.Vertices],
            Indices = [.. sphere.Indices, .. speck.Indices.Select(i => i + offset)],
        };

        Mesh cleaned = MeshCleaner.RemoveSmallComponents(combined, 1e-3);
        Mesh kept = MeshCleaner.RemoveSmallComponents(combined, 0);

        Assert.Equal(sphere.TriangleCount, cleaned.TriangleCount);
        Assert.Equal(sphere.TriangleCount + 4, kept.TriangleCount);
    }

    [Fact]
    public void RemoveSmallComponents_NeverRemovesTheLargestComponent()
    {
        Mesh tet = Tetrahedron();

        Assert.Equal(4, MeshCleaner.RemoveSmallComponents(tet, 5.0).TriangleCount);
    }

    [Fact]
    public void FillHoles_ClosesSmallLoopsAndHonoursThePerimeterLimit()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        Mesh holed = RemoveFacesAroundVertex(sphere, 0);
        Assert.True(SyntheticMeshes.Topology(holed).Boundary > 0);
        double perimeter = BoundaryPerimeter(holed);

        Mesh filled = MeshCleaner.FillHoles(holed, perimeter * 1.01);
        Mesh untouched = MeshCleaner.FillHoles(holed, perimeter * 0.5);

        Assert.True(SyntheticMeshes.IsClosedManifold(filled));
        Assert.Equal(2, SyntheticMeshes.EulerCharacteristic(filled));
        Assert.True(SyntheticMeshes.SignedVolume(filled) > 0);
        Assert.Equal(SyntheticMeshes.Topology(holed).Boundary, SyntheticMeshes.Topology(untouched).Boundary);
    }

    [Fact]
    public void FillHoles_ClosesASingleTriangleHole()
    {
        Mesh tet = Tetrahedron();
        Mesh open = new() { Vertices = tet.Vertices, Indices = tet.Indices[..9] };

        Mesh filled = MeshCleaner.FillHoles(open, 100);

        Assert.Equal(4, filled.TriangleCount);
        Assert.True(SyntheticMeshes.IsClosedManifold(filled));
    }

    [Fact]
    public void UnifyOrientation_RepairsRandomlyFlippedFacesAndPointsOutward()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        int[] indices = (int[])sphere.Indices.Clone();
        Random rng = new(7);
        for (int f = 0; f < indices.Length / 3; f++)
            if (rng.Next(3) == 0) (indices[f * 3 + 1], indices[f * 3 + 2]) = (indices[f * 3 + 2], indices[f * 3 + 1]);
        Mesh scrambled = new() { Vertices = sphere.Vertices, Indices = indices };
        Assert.False(SyntheticMeshes.IsClosedManifold(scrambled));

        Mesh fixedMesh = MeshCleaner.UnifyOrientation(scrambled);

        Assert.True(SyntheticMeshes.IsClosedManifold(fixedMesh));
        Assert.True(SyntheticMeshes.SignedVolume(fixedMesh) > 0);
    }

    [Fact]
    public void UnifyOrientation_FlipsAnInsideOutClosedMesh()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(1);
        int[] inverted = (int[])sphere.Indices.Clone();
        for (int f = 0; f < inverted.Length / 3; f++) (inverted[f * 3 + 1], inverted[f * 3 + 2]) = (inverted[f * 3 + 2], inverted[f * 3 + 1]);
        Mesh inside = new() { Vertices = sphere.Vertices, Indices = inverted };

        Assert.True(SyntheticMeshes.SignedVolume(MeshCleaner.UnifyOrientation(inside, orientOutward: true)) > 0);
        Assert.True(SyntheticMeshes.SignedVolume(MeshCleaner.UnifyOrientation(inside, orientOutward: false)) < 0);
    }

    [Fact]
    public void WeldVertices_MergesCoincidentPositions()
    {
        Mesh cube = SyntheticMeshes.Cube(2);
        // Unweld by giving every triangle corner its own vertex.
        float[] verts = new float[cube.Indices.Length * 3];
        int[] idx = new int[cube.Indices.Length];
        for (int i = 0; i < idx.Length; i++)
        {
            Array.Copy(cube.Vertices, cube.Indices[i] * 3, verts, i * 3, 3);
            idx[i] = i;
        }

        Mesh welded = MeshCleaner.WeldVertices(new Mesh { Vertices = verts, Indices = idx });

        Assert.Equal(cube.VertexCount, welded.VertexCount);
        Assert.True(SyntheticMeshes.IsClosedManifold(welded));
        Assert.Equal(cube.VertexCount, MeshCleaner.WeldVertices(new Mesh { Vertices = verts, Indices = idx }, 1e-4).VertexCount);
    }

    [Fact]
    public void Clean_RunsEveryStepAndLeavesAWatertightMesh()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        int[] indices = (int[])sphere.Indices.Clone();
        (indices[1], indices[2]) = (indices[2], indices[1]);
        Mesh dirty = new() { Vertices = sphere.Vertices, Indices = [.. indices, .. sphere.Indices[..3]] };

        Mesh cleaned = MeshCleaner.Clean(dirty);

        Assert.Equal(sphere.TriangleCount, cleaned.TriangleCount);
        Assert.True(SyntheticMeshes.IsClosedManifold(cleaned));
        Assert.True(SyntheticMeshes.SignedVolume(cleaned) > 0);
    }

    [Fact]
    public void RejectsIndicesOutsideTheVertexArray()
    {
        Mesh bad = new() { Vertices = new float[9], Indices = [0, 1, 3] };
        Assert.Throws<ArgumentException>(() => MeshCleaner.Clean(bad));
    }

    private static Mesh RemoveFacesAroundVertex(Mesh mesh, int vertex)
    {
        List<int> keep = [];
        for (int f = 0; f < mesh.TriangleCount; f++)
        {
            bool touches = mesh.Indices[f * 3] == vertex || mesh.Indices[f * 3 + 1] == vertex || mesh.Indices[f * 3 + 2] == vertex;
            if (!touches) keep.AddRange([mesh.Indices[f * 3], mesh.Indices[f * 3 + 1], mesh.Indices[f * 3 + 2]]);
        }
        return new Mesh { Vertices = mesh.Vertices, Indices = [.. keep] };
    }

    private static double BoundaryPerimeter(Mesh mesh)
    {
        Dictionary<(int, int), int> counts = [];
        for (int f = 0; f < mesh.TriangleCount; f++)
            for (int c = 0; c < 3; c++)
            {
                int a = mesh.Indices[f * 3 + c], b = mesh.Indices[f * 3 + (c + 1) % 3];
                (int, int) key = a < b ? (a, b) : (b, a);
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }
        double sum = 0;
        foreach (((int a, int b), int count) in counts)
        {
            if (count != 1) continue;
            double dx = mesh.Vertices[a * 3] - mesh.Vertices[b * 3], dy = mesh.Vertices[a * 3 + 1] - mesh.Vertices[b * 3 + 1];
            double dz = mesh.Vertices[a * 3 + 2] - mesh.Vertices[b * 3 + 2];
            sum += Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
        return sum;
    }
}

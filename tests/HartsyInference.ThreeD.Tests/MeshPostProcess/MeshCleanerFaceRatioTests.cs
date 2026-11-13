using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>The face-count floater rule of the Hunyuan3D shape models (pymeshlab <c>nbfaceratio</c> 0.005): edge-connected components with fewer faces than the ratio times the largest component's face count go.</summary>
public sealed class MeshCleanerFaceRatioTests
{
    /// <summary>A strip of <paramref name="faces"/> triangles (edge-connected, open) starting at <paramref name="x"/>; the pieces are placed far apart so no vertex is shared.</summary>
    private static void AddStrip(List<float> vertices, List<int> indices, int faces, float x)
    {
        int first = vertices.Count / 3;
        for (int i = 0; i < faces + 2; i++)
        {
            vertices.Add(x + i * 0.1f);
            vertices.Add(i % 2 * 0.1f);
            vertices.Add(0);
        }

        for (int f = 0; f < faces; f++)
        {
            // Alternate the winding so the strip stays consistently oriented.
            if (f % 2 == 0)
                indices.AddRange([first + f, first + f + 1, first + f + 2]);
            else
                indices.AddRange([first + f + 1, first + f, first + f + 2]);
        }
    }

    private static Mesh Pieces(params int[] faceCounts)
    {
        List<float> vertices = [];
        List<int> indices = [];
        for (int i = 0; i < faceCounts.Length; i++)
            AddStrip(vertices, indices, faceCounts[i], i * 1000f);
        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    [Fact]
    public void ALargestOf1000FacesKeepsTheFiveAndSixFacePiecesAndDropsTheFourFacePiece()
    {
        Mesh mesh = Pieces(1000, 4, 5, 6);

        Mesh result = MeshCleaner.RemoveSmallComponentsByFaceCount(mesh, 0.005);

        Assert.Equal(1000 + 5 + 6, result.TriangleCount); // 0.005 * 1000 = 5: a piece stays when it has at least 5 faces
    }

    [Fact]
    public void TheThresholdIsTruncatedLikeMeshLabs_FiveFacesStayUnderALargestOf1100()
    {
        // MeshLab: uint th = largest * 0.005f; a component stays at size >= th. 1,100 * 0.005 = 5.5 truncates to 5.
        Mesh result = MeshCleaner.RemoveSmallComponentsByFaceCount(Pieces(1100, 4, 5), 0.005);

        Assert.Equal(1100 + 5, result.TriangleCount);
        // 2,000 * 0.005 = 10: nine faces go, ten stay.
        Assert.Equal(2000 + 10, MeshCleaner.RemoveSmallComponentsByFaceCount(Pieces(2000, 9, 10), 0.005).TriangleCount);
    }

    [Fact]
    public void ThreeFacesOnOneEdgeAreOnePieceSoTheRuleMustRunBeforeTheNonManifoldSplit()
    {
        // Three triangles sharing the edge (0, 1), a non-manifold edge: MeshLab's face-face ring joins them into one component of 3 faces.
        Mesh fin = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1],
            Indices = [0, 1, 2, 0, 1, 3, 0, 1, 4],
        };
        List<float> vertices = [.. fin.Vertices];
        List<int> indices = [.. fin.Indices];
        AddStrip(vertices, indices, 4, 50f);
        Mesh mesh = new() { Vertices = [.. vertices], Indices = [.. indices] };

        // largest 4 faces, ratio 0.75: threshold 3, so the 3-face fin stays only if its three faces count as one component (as separate faces they would be 1 each and go).
        Assert.Equal(7, MeshCleaner.RemoveSmallComponentsByFaceCount(mesh, 0.75).TriangleCount);
    }

    [Fact]
    public void TheLargestAlwaysStaysAndARatioOfZeroKeepsEverything()
    {
        Mesh mesh = Pieces(10, 3);

        Assert.Equal(13, MeshCleaner.RemoveSmallComponentsByFaceCount(mesh, 0).TriangleCount);
        Assert.Equal(10, MeshCleaner.RemoveSmallComponentsByFaceCount(mesh, 1).TriangleCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshCleaner.RemoveSmallComponentsByFaceCount(mesh, -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshCleaner.RemoveSmallComponentsByFaceCount(mesh, 2)); // would remove the largest component too
    }

    [Fact]
    public void FacesThatOnlyTouchAtAVertexAreSeparateComponents()
    {
        // Two triangles sharing only vertex 0: edge-connected components are one face each, so the ratio compares 1 face against 1 face and both stay;
        // a 3-face fan on the other side makes the lone triangles the small ones.
        Mesh bowTie = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0],
            Indices = [0, 1, 2, 0, 3, 4],
        };
        Assert.Equal(2, MeshCleaner.RemoveSmallComponentsByFaceCount(bowTie, 0.5).TriangleCount);

        List<float> vertices = [.. bowTie.Vertices];
        List<int> indices = [.. bowTie.Indices];
        AddStrip(vertices, indices, 3, 50f);
        Mesh withStrip = new() { Vertices = [.. vertices], Indices = [.. indices] };
        // largest 3 faces, ratio 0.75: the threshold is (uint)2.25 = 2, so both single triangles go (they would stay if they counted as one 2-face component).
        Assert.Equal(3, MeshCleaner.RemoveSmallComponentsByFaceCount(withStrip, 0.75).TriangleCount);
    }

    [Fact]
    public void TheShapeFinisherAppliesTheRuleBeforeItsOwnRepair()
    {
        Mesh mesh = Pieces(1000, 4, 5, 6);

        Mesh finished = ShapeMeshFinisher.Finish(mesh, clean: true, smallComponentAreaRatio: 0, smallComponentFaceRatio: 0.005);

        // The three kept strips are open, so hole filling adds faces; the dropped 4-face piece is gone whatever it adds: compare the number of connected pieces left.
        Assert.Equal(3, Components(finished));
        Assert.Equal(4, Components(ShapeMeshFinisher.Finish(mesh, clean: true, smallComponentAreaRatio: 0, smallComponentFaceRatio: 0)));
    }

    private static int Components(Mesh mesh)
    {
        int[] parent = Enumerable.Range(0, mesh.VertexCount).ToArray();
        int Find(int x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; }
        for (int f = 0; f < mesh.TriangleCount; f++)
        {
            parent[Find(mesh.Indices[f * 3 + 1])] = Find(mesh.Indices[f * 3]);
            parent[Find(mesh.Indices[f * 3 + 2])] = Find(mesh.Indices[f * 3]);
        }

        return Enumerable.Range(0, mesh.TriangleCount).Select(f => Find(mesh.Indices[f * 3])).Distinct().Count();
    }
}

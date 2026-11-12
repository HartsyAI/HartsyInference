using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamFlexiCubesDataTests
{
    private static readonly int[] CubeEdges = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4];

    private static int[] Group(SamFlexiCubesTables t, int caseId, int group) =>
        [.. t.Dmc.Slice((caseId * t.MaxVertices + group) * SamFlexiCubesTables.EdgesPerVertex, SamFlexiCubesTables.EdgesPerVertex).ToArray().Where(e => e >= 0)];

    [Fact]
    public void Tables_HaveTheReleasedShapes()
    {
        SamFlexiCubesTables t = SamFlexiCubesData.Tables;

        Assert.Equal(256 * 4 * 7, t.Dmc.Length);
        Assert.Equal(256, t.NumVd.Length);
        Assert.Equal(256 * 5, t.Check.Length);
        Assert.Equal(0, t.NumVd[0]);
        Assert.Equal(0, t.NumVd[255]);
        Assert.Equal(4, t.NumVd.ToArray().Max());
    }

    [Fact]
    public void EveryDualVertexAveragesCrossingEdgesAndTheGroupsPartitionAllOfThem()
    {
        SamFlexiCubesTables t = SamFlexiCubesData.Tables;
        for (int c = 0; c < 256; c++)
        {
            List<int> crossing = [];
            for (int e = 0; e < 12; e++)
            {
                if (((c >> CubeEdges[e * 2]) & 1) != ((c >> CubeEdges[e * 2 + 1]) & 1))
                    crossing.Add(e);
            }

            List<int> listed = [];
            for (int g = 0; g < t.NumVd[c]; g++)
                listed.AddRange(Group(t, c, g));
            // The edges of all the case's dual vertices are exactly its crossing edges, each once.
            Assert.Equal(crossing.Order(), listed.Order());
        }
    }

    [Fact]
    public void ACaseWithOneInsideCornerHasOneDualVertexOnTheThreeEdgesOfThatCorner()
    {
        SamFlexiCubesTables t = SamFlexiCubesData.Tables;
        for (int corner = 0; corner < 8; corner++)
        {
            int[] incident = [.. Enumerable.Range(0, 12).Where(e => CubeEdges[e * 2] == corner || CubeEdges[e * 2 + 1] == corner)];
            foreach (int c in new[] { 1 << corner, 255 - (1 << corner) })
            {
                Assert.Equal(1, t.NumVd[c]);
                Assert.Equal(incident.Order(), Group(t, c, 0).Order());
            }
        }
    }

    [Fact]
    public void TheCheckTableFlagsExactly36AmbiguousCasesEachPointingOneStepAlongOneAxisAtTheOppositeCase()
    {
        SamFlexiCubesTables t = SamFlexiCubesData.Tables;
        List<int> flagged = [.. Enumerable.Range(0, 256).Where(c => t.Check[c * 5] == 1)];

        Assert.Equal(36, flagged.Count);
        foreach (int c in flagged)
        {
            int[] offset = [t.Check[c * 5 + 1], t.Check[c * 5 + 2], t.Check[c * 5 + 3]];
            Assert.Equal(1, offset.Count(o => o != 0));
            Assert.All(offset, o => Assert.InRange(o, -1, 1));
            int replacement = t.Check[c * 5 + 4];
            Assert.Equal(255 - c, replacement);
            Assert.Equal(1, t.NumVd[c]);
            Assert.Equal(2, t.NumVd[replacement]);
            Assert.NotEqual(1, t.Check[replacement * 5]);
        }

        // Every unflagged row is zeros.
        Assert.All(Enumerable.Range(0, 256).Where(c => t.Check[c * 5] != 1), c => Assert.True(t.Check.Slice(c * 5, 5).ToArray().All(v => v == 0)));
    }

    [Fact]
    public void AFlaggedCubeWhoseNeighbourIsFlaggedTakesTheOppositeCaseAndTheOthersKeepTheirs()
    {
        SamFlexiCubesTables t = SamFlexiCubesData.Tables;
        int c = Enumerable.Range(0, 256).First(x => t.Check[x * 5] == 1);
        (int dx, int dy, int dz) = (t.Check[c * 5 + 1], t.Check[c * 5 + 2], t.Check[c * 5 + 3]);
        // A and B = A + offset, both case c; C far away, also case c. The look-up is one-directional: A looks at B, B looks at B + offset (no cube there), C at nothing.
        long[] keys = [SamMeshCornerGrid.Pack(8, 8, 8), SamMeshCornerGrid.Pack(8 + dx, 8 + dy, 8 + dz), SamMeshCornerGrid.Pack(20, 20, 20)];
        int[] cases = [c, c, c];

        SamSparseFeatures2Mesh.ResolveAmbiguousCases(keys, [0, 1, 2], cases, t, 32);

        Assert.Equal([255 - c, c, c], cases);
    }
}

using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamFlexiCubesDataTests
{
    private static readonly int[] CubeEdges = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4];

    private static int[] Group(SamFlexiCubesTables t, int caseId, int group) =>
        [.. t.Dmc.Skip((caseId * t.MaxVertices + group) * SamFlexiCubesTables.EdgesPerVertex).Take(SamFlexiCubesTables.EdgesPerVertex).Where(e => e >= 0)];

    [Fact]
    public void Tables_HaveTheReleasedShapes()
    {
        SamFlexiCubesTables t = SamFlexiCubesData.Tables;

        Assert.Equal(256 * 4 * 7, t.Dmc.Length);
        Assert.Equal(256, t.NumVd.Length);
        Assert.Equal(256 * 5, t.Check.Length);
        Assert.Equal(0, t.NumVd[0]);
        Assert.Equal(0, t.NumVd[255]);
        Assert.Equal(4, t.NumVd.Max());
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
    public void TheCheckTableFlagsTheAmbiguousCasesAndNamesARealReplacement()
    {
        SamFlexiCubesTables t = SamFlexiCubesData.Tables;
        int flagged = 0;
        for (int c = 0; c < 256; c++)
        {
            if (t.Check[c * 5] != 1)
                continue;
            flagged++;
            Assert.InRange(t.Check[c * 5 + 4], 0, 255);
            Assert.All(new[] { t.Check[c * 5 + 1], t.Check[c * 5 + 2], t.Check[c * 5 + 3] }, offset => Assert.InRange(offset, -1, 1));
        }

        Assert.True(flagged > 0);
    }
}

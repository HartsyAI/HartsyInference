using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.Idol;
using HartsyInference.ThreeD.Models.Idol.Budget;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol.Budget;

public sealed class BudgetGeometryTests
{
    [Fact]
    public void TheEmbeddedPositionsAreAnExactSixteenCubedByTwoOctree()
    {
        (int[] toOctree, int[] fromOctree) = BudgetOctree.Permutation(IdolWeights.EmbeddedPositions());

        Assert.Equal(Enumerable.Range(0, 8192), toOctree.OrderBy(x => x));
        for (int k = 0; k < 8192; k++)
            Assert.Equal(k, fromOctree[toOctree[k]]);
    }

    [Fact]
    public void ADuplicatedSlotIsRefused()
    {
        float[] positions = IdolWeights.EmbeddedPositions();
        Array.Copy(positions, 0, positions, 3, 3);   // point 1 lands on point 0's slot

        Assert.Throws<HartsyInferenceException>(() => BudgetOctree.Permutation(positions));
    }

    [Fact]
    public void TheCubeTableCoversEveryCellOnceAndAgreesWithTheCellToCubeMap()
    {
        int[] cells = BudgetOctree.CubeToCells();

        Assert.Equal(Enumerable.Range(0, 4096), cells.OrderBy(x => x));
        for (int j = 0; j < 512; j++)
            for (int o = 0; o < 8; o++)
                Assert.Equal(j, BudgetOctree.CubeOfCell(cells[j * 8 + o]));
        // Child order is (a, b, c) over the x, y and z halves.
        Assert.Equal([0, 1, 16, 17, 256, 257, 272, 273], cells[..8]);
    }

    [Theory]
    [InlineData(512, 512)]
    [InlineData(513, 512)]
    [InlineData(515, 512)]
    [InlineData(516, 519)]
    [InlineData(519, 519)]
    [InlineData(100, 512)]
    [InlineData(9000, 8192)]
    public void TokenCountsSnapToARealisableOne(int requested, int expected) => Assert.Equal(expected, BudgetLayout.SnapTokens(requested));

    [Fact]
    public void ExactBudgetsKeepTheOctreeCovered()
    {
        Random rng = new(3);
        float[] cube = Enumerable.Range(0, 512).Select(_ => (float)rng.NextDouble() * 4).ToArray(), cell = Enumerable.Range(0, 4096).Select(_ => (float)rng.NextDouble() * 4).ToArray();
        foreach (int tokens in new[] { 512, 519, 700, 1024, 2048, 4096, 6000, 8192 })
        {
            BudgetLayout layout = BudgetLayout.Build(cube, cell, tokens);

            Assert.Equal(tokens, layout.Tokens);
            Assert.Equal(tokens, 512 + 7 * layout.K4 + layout.K2);
            Assert.InRange(layout.K2, 0, 8 * layout.K4);
            // Every cell is covered once: by an opened pair, a whole cell, or one of the eight of a whole cube.
            HashSet<int> covered = [.. layout.Pairs, .. layout.Cells, .. layout.Cubes.SelectMany(j => BudgetOctree.CubeToCells().Skip(j * 8).Take(8))];
            Assert.Equal(4096, covered.Count);
            Assert.Equal(4096, layout.K2 + layout.CellTokens + 8 * layout.CubeTokens);
            Assert.Equal(layout.Cubes.OrderBy(x => x), layout.Cubes);
        }
    }

    [Fact]
    public void TheDenseLayoutIsIdolsAndABudgetOfTheFullSetBuildsIt()
    {
        BudgetLayout dense = BudgetLayout.Dense();
        BudgetLayout built = BudgetLayout.Build(new float[512], new float[4096], 8192);

        Assert.True(dense.IsDense);
        Assert.True(built.IsDense);
        Assert.Equal(8192, dense.Tokens);
        Assert.Equal(dense.Pairs, built.Pairs);
    }

    [Fact]
    public void TheHighestScoreOpensAndATieGoesToTheLowerIndex()
    {
        // 519 tokens is one open cube and no open cell, whatever the scores say.
        float[] cube = new float[512], cell = new float[4096];
        cube[10] = 5f;
        cube[3] = 4f;
        BudgetLayout one = BudgetLayout.Build(cube, cell, 519);
        Assert.Equal((1, 0), (one.K4, one.K2));
        Assert.DoesNotContain(10, one.Cubes);
        Assert.Contains(3, one.Cubes);

        cube[10] = 0f;
        cube[3] = 5f;
        cube[7] = 5f;   // ties with cube 3: the lower index opens
        BudgetLayout tie = BudgetLayout.Build(cube, cell, 519);
        Assert.DoesNotContain(3, tie.Cubes);
        Assert.Contains(7, tie.Cubes);
    }

    [Fact]
    public void ABudgetOutsideTheRealisableRangeIsRefused() => Assert.Throws<ArgumentOutOfRangeException>(() => BudgetLayout.K4Bounds(300));

    [Fact]
    public void DetailScoresAreZeroForAConstantLatentAndGrowWithTheSpread()
    {
        float[] flat = new float[8192 * 16];
        Array.Fill(flat, 0.7f);
        (float[] cube0, float[] cell0) = BudgetOctree.DetailScores(flat, 16);
        Assert.All(cube0, s => Assert.True(s < 5e-3f));
        Assert.All(cell0, s => Assert.True(s < 5e-3f));

        float[] spread = (float[])flat.Clone();
        spread[0] = 3f;   // one point of cell 0 (and cube 0) differs
        (float[] cube1, float[] cell1) = BudgetOctree.DetailScores(spread, 16);
        Assert.True(cell1[0] > cell0[0] + 0.1f);
        Assert.True(cube1[0] > cube0[0] + 0.01f);
        Assert.True(cell1[0] > cube1[0]);   // two points vary more per member than sixteen
        Assert.Equal(cell0[5], cell1[5]);
    }
}

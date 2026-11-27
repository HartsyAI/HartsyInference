using HartsyInference.ThreeD.Geometry;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class DualGridVoxelizerSolverTests
{
    [Fact]
    public void SolveColPivQr_FullRank_RecoversKnownSolution()
    {
        float[] matrix = { 2f, 1f, 0f, 1f, 3f, 1f, 0f, 1f, 4f };
        float[] rhs = { 4f, 10f, 14f };
        Span<float> solution = stackalloc float[3];

        DualGridVoxelizer.SolveColPivQr(matrix, rhs, solution);

        Assert.InRange(solution[0], 0.9999f, 1.0001f);
        Assert.InRange(solution[1], 1.9999f, 2.0001f);
        Assert.InRange(solution[2], 2.9999f, 3.0001f);
    }

    [Fact]
    public void SolveColPivQr_TiltedRankOne_ReturnsBasicSolutionThatSatisfiesSystem()
    {
        // A = v vᵀ with v = (1, 2, 2): rank 1, so only the leading pivot counts and the basic solution must
        // still satisfy the system where b lies in the range of A.
        float[] matrix = { 1f, 2f, 2f, 2f, 4f, 4f, 2f, 4f, 4f };
        float[] rhs = { 3f, 6f, 6f };
        Span<float> solution = stackalloc float[3];

        DualGridVoxelizer.SolveColPivQr(matrix, rhs, solution);

        for (int r = 0; r < 3; r++)
        {
            float row = 0f;
            for (int c = 0; c < 3; c++)
            {
                row += matrix[r * 3 + c] * solution[c];
            }

            Assert.InRange(row, rhs[r] - 1e-3f, rhs[r] + 1e-3f);
        }
    }

    [Fact]
    public void SolveColPivQr_ZeroMatrix_ReturnsZeros()
    {
        float[] matrix = new float[9];
        float[] rhs = { 1f, 2f, 3f };
        Span<float> solution = stackalloc float[3] { 9f, 9f, 9f };

        DualGridVoxelizer.SolveColPivQr(matrix, rhs, solution);

        Assert.Equal(0f, solution[0]);
        Assert.Equal(0f, solution[1]);
        Assert.Equal(0f, solution[2]);
    }

    [Fact]
    public void SolveColPivQr_PivotInRankVsNonzeroPivotsBand_KeepsSecondPivot()
    {
        // diag(1, 2e-7, 0): Eigen keeps two nonzero pivots, while a 3·eps cutoff on |R_11| would drop the second.
        float[] matrix = { 1f, 0f, 0f, 0f, 2e-7f, 0f, 0f, 0f, 0f };
        float[] rhs = { 1f, 2e-7f, 0f };
        Span<float> solution = stackalloc float[3];

        DualGridVoxelizer.SolveColPivQr(matrix, rhs, solution);

        Assert.InRange(solution[0], 0.999f, 1.001f);
        Assert.InRange(solution[1], 0.999f, 1.001f);
        Assert.Equal(0f, solution[2]);
    }

    [Fact]
    public void SolveColPivQr_NearlyParallelColumns_RecomputesDowndatedNormAndSolvesFullRankSystem()
    {
        // Column 0 is almost parallel to the pivot column 1, so its downdated norm collapses and is recomputed.
        // The solve must still be exact for a full-rank matrix; the test does not check which pivot was chosen.
        float[] matrix = { 2f, 2f, 0f, 0f, 1e-3f, 0f, 0f, 0f, 1f };
        float[] expected = { 1f, -2f, 3f };
        float[] rhs = { -2f, -2e-3f, 3f };
        Span<float> solution = stackalloc float[3];

        DualGridVoxelizer.SolveColPivQr(matrix, rhs, solution);

        for (int c = 0; c < 3; c++)
        {
            Assert.InRange(solution[c], expected[c] - 1e-3f, expected[c] + 1e-3f);
        }
    }
}

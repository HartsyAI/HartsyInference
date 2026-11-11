using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2RopeTableTests
{
    [Fact]
    public void Table_MatchesTheClosedFormPhaseLayout()
    {
        // headDim 12: f = 2 pairs per axis, so 6 pairs ordered x0 x1 y0 y1 z0 z1 with frequencies 1 and 10000^(-1/2).
        float[] position = [3f, 5f, 7f];
        using Trellis2RopeTable table = Trellis2RopeTable.Build(12, position, 1);

        ReadOnlySpan<float> cos = table.Cos.AsReadOnlySpan<float>(), sin = table.Sin.AsReadOnlySpan<float>();
        Assert.Equal(6, cos.Length);
        for (int axis = 0; axis < 3; axis++)
        {
            for (int k = 0; k < 2; k++)
            {
                double angle = position[axis] * Math.Pow(10000.0, -k / 2.0);
                Assert.Equal(Math.Cos(angle), cos[axis * 2 + k], 5);
                Assert.Equal(Math.Sin(angle), sin[axis * 2 + k], 5);
            }
        }
    }

    [Fact]
    public void ShippedHeadWidth_RotatesSixtyThreePairsAndLeavesTwoChannels()
    {
        // 128 // 2 // 3 = 21 pairs per axis: 126 rotated channels, 2 passed through.
        Assert.Equal(21, Trellis2RopeTable.PairsPerAxis(128));
        using Trellis2RopeTable table = Trellis2RopeTable.Build(128, [1f, 2f, 3f], 1);

        Assert.Equal(63, table.Cos.Shape[2]);
    }

    [Fact]
    public void Apply_RotatesAdjacentChannelPairsAndLeavesTheTailUntouched()
    {
        // headDim 8: one pair per axis (6 channels), the last 2 pass through. Two heads, two tokens.
        float[] positions = [0f, 0f, 0f, 2f, 3f, 4f];
        using Trellis2RopeTable table = Trellis2RopeTable.Build(8, positions, 2);
        using Tensor x = Trellis2TestData.Random(new Random(7), 1, 2, 2, 8);
        float[] before = Trellis2TestData.ToArray(x);
        using CpuBackend backend = new();

        table.Apply(backend, x);

        float[] after = Trellis2TestData.ToArray(x);
        for (int head = 0; head < 2; head++)
        {
            for (int token = 0; token < 2; token++)
            {
                int row = (head * 2 + token) * 8;
                for (int axis = 0; axis < 3; axis++)
                {
                    double angle = positions[token * 3 + axis];
                    float a = before[row + 2 * axis], b = before[row + 2 * axis + 1];
                    Assert.Equal(a * Math.Cos(angle) - b * Math.Sin(angle), after[row + 2 * axis], 4);
                    Assert.Equal(a * Math.Sin(angle) + b * Math.Cos(angle), after[row + 2 * axis + 1], 4);
                }

                Assert.Equal(before[row + 6], after[row + 6]);
                Assert.Equal(before[row + 7], after[row + 7]);
            }
        }
    }

    [Fact]
    public void DenseGridPositions_AreXMajor()
    {
        float[] positions = Trellis2RopeTable.DenseGridPositions(3);

        Assert.Equal(27 * 3, positions.Length);
        Assert.Equal(new float[] { 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0 }, positions[..12]);
        Assert.Equal(new float[] { 2, 2, 2 }, positions[^3..]);
    }

    [Fact]
    public void VoxelPositions_DropTheBatchColumn()
    {
        Assert.Equal(new float[] { 4, 5, 6, 7, 8, 9 }, Trellis2RopeTable.VoxelPositions([0, 4, 5, 6, 0, 7, 8, 9]));
    }

    [Fact]
    public void Build_RejectsAHeadTooNarrowForThreeAxes()
    {
        Assert.Throws<ArgumentException>(() => Trellis2RopeTable.Build(4, [0f, 0f, 0f], 1));
    }
}

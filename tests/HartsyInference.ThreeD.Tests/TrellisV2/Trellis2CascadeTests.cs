using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2CascadeTests
{
    [Fact]
    public void OccupiedVoxels_ListsPositiveLogitsInXMajorOrder()
    {
        float[] logits = new float[4 * 4 * 4];
        logits[(0 * 4 + 1) * 4 + 2] = 1f;   // (0, 1, 2)
        logits[(3 * 4 + 0) * 4 + 3] = 0.1f; // (3, 0, 3)
        logits[(1 * 4 + 1) * 4 + 1] = 0f;   // not above zero
        logits[(2 * 4 + 2) * 4 + 2] = -3f;

        int[] coords = Trellis2Cascade.OccupiedVoxels(logits, 4, 4);

        Assert.Equal([0, 0, 1, 2, 0, 3, 0, 3], coords);
    }

    [Fact]
    public void OccupiedVoxels_PoolsABlockToOneVoxelWhenAnyCellIsOccupied()
    {
        // 4 -> 2: cells (0,0,0) and (1,1,1) fall in the same 2x2x2 block; (3,3,3) is the far block.
        float[] logits = new float[64];
        logits[0] = 1f;
        logits[(1 * 4 + 1) * 4 + 1] = 2f;
        logits[(3 * 4 + 3) * 4 + 3] = 1f;

        int[] coords = Trellis2Cascade.OccupiedVoxels(logits, 4, 2);

        Assert.Equal([0, 0, 0, 0, 0, 1, 1, 1], coords);
    }

    [Fact]
    public void OccupiedVoxels_RejectsAnIncompatibleTarget()
    {
        Assert.Throws<ArgumentException>(() => Trellis2Cascade.OccupiedVoxels(new float[64], 4, 3));
        Assert.Throws<ArgumentException>(() => Trellis2Cascade.OccupiedVoxels(new float[63], 4, 2));
    }

    [Fact]
    public void QuantizeUnique_MapsEachVoxelToTheFloorOfItsScaledCentreAndDeduplicates()
    {
        // 512 -> 1024: grid 64, so (c + 0.5) / 512 · 64 = c/8 + 1/16: voxels 0..7 map to 0, 8..15 to 1.
        int[] upsampled = [0, 0, 0, 0, 0, 7, 7, 7, 0, 8, 0, 0, 0, 15, 15, 15, 0, 8, 7, 7, 0, 0, 0, 0];

        int[] coords = Trellis2Cascade.QuantizeUnique(upsampled, 512, 1024);

        // (0,0,0) twice and (7,7,7) map to cell (0,0,0); (8,0,0) and (8,7,7) to (1,0,0); (15,15,15) to (1,1,1). Output is sorted.
        Assert.Equal([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], coords);
    }

    [Fact]
    public void QuantizeUnique_MatchesUpstreamsFloatFormulaForEveryCoordinate()
    {
        // The reference computes ((c + 0.5) / 512 · grid).int() in float32; the integer form must agree for both grids.
        foreach (int hr in new[] { 1024, 1152, 1280, 1408, 1536 })
        {
            int grid = hr / 16;
            for (int c = 0; c < 512; c++)
            {
                int expected = (int)((c + 0.5f) / 512f * grid);
                int[] coords = Trellis2Cascade.QuantizeUnique([0, c, 0, 0], 512, hr);
                Assert.Equal(expected, coords[1]);
            }
        }
    }

    [Fact]
    public void FitTokenCap_KeepsTheTargetWhenTheCountIsUnderTheCap()
    {
        int[] upsampled = [0, 0, 0, 0, 0, 100, 100, 100, 0, 200, 200, 200];

        (int[] coords, int resolution) = Trellis2Cascade.FitTokenCap(upsampled, 512, 1536, maxTokens: 4);

        Assert.Equal(1536, resolution);
        Assert.Equal(3, coords.Length / 4);
    }

    [Fact]
    public void FitTokenCap_RetreatsInSteps128UntilTheCountFitsOrTheFloorIsReached()
    {
        // 600 voxels along x at stride 5 on the 512 grid. At grid g = res/16 a voxel maps to floor((c + 0.5) · g / 512),
        // so the unique count shrinks as the resolution falls.
        List<int> upsampled = [];
        for (int i = 0; i < 100; i++)
            upsampled.AddRange([0, i * 5, 0, 0]);
        int CountAt(int res) => Trellis2Cascade.QuantizeUnique(upsampled.ToArray(), 512, res).Length / 4;
        Assert.True(CountAt(1536) > CountAt(1024));

        // A cap just above the 1280 count and below the 1408 count must settle at 1280.
        int cap = CountAt(1280) + 1;
        Assert.True(CountAt(1408) >= cap);
        (int[] coords, int resolution) = Trellis2Cascade.FitTokenCap(upsampled.ToArray(), 512, 1536, cap);
        Assert.Equal(1280, resolution);
        Assert.Equal(CountAt(1280), coords.Length / 4);

        // A cap nothing satisfies stops at the floor, as upstream's literal 1024 does.
        (_, int floor) = Trellis2Cascade.FitTokenCap(upsampled.ToArray(), 512, 1536, 1);
        Assert.Equal(1024, floor);
    }

    [Fact]
    public void FitTokenCap_TheDefaultTargetNeverRetreats()
    {
        int[] upsampled = [0, 0, 0, 0, 0, 100, 100, 100, 0, 200, 200, 200];

        (_, int resolution) = Trellis2Cascade.FitTokenCap(upsampled, 512, 1024, maxTokens: 1);

        Assert.Equal(1024, resolution);
    }

    [Fact]
    public void FitTokenCap_RejectsTargetsOffTheStepGridAndEmptyInput()
    {
        Assert.Throws<ArgumentException>(() => Trellis2Cascade.FitTokenCap([0, 1, 1, 1], 512, 1500, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => Trellis2Cascade.FitTokenCap([0, 1, 1, 1], 512, 512, 10));
        Assert.Throws<HartsyInferenceException>(() => Trellis2Cascade.FitTokenCap([], 512, 1024, 10));
    }

    [Fact]
    public void DefaultCapMatchesUpstream()
    {
        Assert.Equal(49152, Trellis2Cascade.DefaultMaxTokens);
        Assert.Equal(1024, Trellis2Cascade.FloorResolution);
        Assert.Equal(128, Trellis2Cascade.ResolutionStep);
    }
}

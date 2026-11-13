using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

namespace HartsyInference.ThreeD.Tests.Omni;

public sealed class OmniControlInputsTests
{
    // A box [0, 2] x [0, 1] x [0, 1] as a triangle mesh.
    private static readonly float[] Box =
        [0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1, 2, 0, 1, 2, 1, 1, 0, 1, 1];

    private static readonly int[] BoxFaces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5];

    [Fact]
    public void Point_CentresThePointsAndScalesTheLongestSideToTwiceTheScale_OneRowPerPoint()
    {
        float[] rows = OmniControlInputs.Point(Box);

        Assert.Equal(8 * 6, rows.Length);
        // x spans 2 (the longest side): -0.98 .. 0.98; y and z span 1: -0.49 .. 0.49; each point repeats to 6 values.
        for (int p = 0; p < 8; p++)
        {
            Assert.Equal(rows[p * 6], rows[p * 6 + 3]);
            Assert.Equal((Box[p * 3] - 1f) * 0.98f, rows[p * 6], 5);
            Assert.Equal((Box[p * 3 + 1] - 0.5f) * 0.98f, rows[p * 6 + 1], 5);
            Assert.Equal((Box[p * 3 + 2] - 0.5f) * 0.98f, rows[p * 6 + 2], 5);
        }
    }

    [Fact]
    public void Voxel_TurnsTheMeshUprightThenQuantisesTheSurfaceSamplesToSixteenCubed()
    {
        float[] rows = OmniControlInputs.Voxel(Box, BoxFaces, seed: 1);

        // After (x, y, z) -> (x, z, -y) the box is 2 long in x and 1 in y and z; normalised to 1.9998 on x it covers all 16 voxel columns in x and 8 in y and z.
        Assert.True(rows.Length % 6 == 0);
        int voxels = rows.Length / 6;
        Assert.InRange(voxels, 16 * 8 * 2, 16 * 8 * 8);
        float lowY = float.MaxValue, highY = float.MinValue, highX = float.MinValue;
        for (int r = 0; r < voxels; r++)
        {
            lowY = Math.Min(lowY, rows[r * 6 + 1]);
            highY = Math.Max(highY, rows[r * 6 + 1]);
            highX = Math.Max(highX, rows[r * 6]);
            Assert.Equal(rows[r * 6], rows[r * 6 + 3]);
        }

        Assert.True(highX > 0.9f);
        Assert.InRange(highY - lowY, 0.7f, 1.1f);
        Assert.Equal(rows, OmniControlInputs.Voxel(Box, BoxFaces, seed: 1));
    }

    [Fact]
    public void BoundingBox_AndPose_PassTheirValuesThroughAndRejectWhatIsOutOfRange()
    {
        Assert.Equal([0.5f, 0.25f, 1f, 0.5f, 0.25f, 1f], OmniControlInputs.BoundingBox(0.5f, 0.25f, 1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => OmniControlInputs.BoundingBox(0f, 0.5f, 0.5f));
        Assert.Throws<ArgumentOutOfRangeException>(() => OmniControlInputs.BoundingBox(0.5f, 1.5f, 0.5f));
        Assert.Throws<ArgumentOutOfRangeException>(() => OmniControlInputs.BoundingBox(0.5f, 0.5f, float.NaN));
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], OmniControlInputs.Pose([1, 2, 3, 4, 5, 6]));
        Assert.Throws<ArgumentException>(() => OmniControlInputs.Pose([1, 2, 3, 4, 5]));
    }

    [Fact]
    public void Inputs_RejectEmptyNonFiniteAndFlatGeometry()
    {
        Assert.Throws<ArgumentException>(() => OmniControlInputs.Point([]));
        Assert.Throws<ArgumentException>(() => OmniControlInputs.Point([0, 0, float.NaN]));
        Assert.Throws<ArgumentException>(() => OmniControlInputs.Point([1, 1, 1, 1, 1, 1]));
        Assert.Throws<ArgumentException>(() => OmniControlInputs.Voxel(Box, [], 1));
        Assert.Throws<ArgumentException>(() => OmniControlInputs.Voxel(Box, [0, 1, 99], 1));
    }
}

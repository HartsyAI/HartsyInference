using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed unsafe class SparseSpatialChannelTests
{
    [Fact]
    public void SpatialToChannel_MergesChildrenIntoOrderedChannelGroups()
    {
        // Cell (0,0,0) holds children (0,0,0) and (1,1,1); cell (1,0,0) holds (2,0,0).
        int[] coords = { 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0 };
        float[] values = { 1f, 2f, 5f };
        using Tensor feats = Feats(values, 3, 1);

        SparseSpatialChannel.Plan plan = SparseSpatialChannel.PlanSpatialToChannel(coords, 4, 2);
        using Tensor merged = SparseSpatialChannel.SpatialToChannelFeats(new CpuBackend(), feats, plan, 1);

        Assert.Equal(2, plan.Coords.Length / 4);
        Assert.Equal(2, plan.Resolution);
        Assert.Equal(new[] { 0, 0, 0, 0, 0, 1, 0, 0 }, plan.Coords);
        // Row 0 is cell (0,0,0): child 0 holds 1, child 7 holds 2. Row 1 is cell (1,0,0): child 0 holds 5.
        float[] expected = new float[2 * 8];
        expected[0] = 1f; expected[7] = 2f; expected[8] = 5f;
        Assert.Equal(expected, ShapeVaeTestData.Read(merged));
    }

    [Fact]
    public void SpatialToChannel_OutputIsSortedByBatchThenCoords()
    {
        // Cells: (1,1,1,1) from batch 1, (0,0,0,0) holding two inputs, (0,1,1,1).
        int[] coords = { 1, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1 };

        SparseSpatialChannel.Plan plan = SparseSpatialChannel.PlanSpatialToChannel(coords, 4, 2);

        Assert.Equal(new[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 }, plan.Coords);
        // Input 0 is row 2, child 7; input 1 is row 0, child 0; input 2 is row 1, child 0; input 3 is row 0, child 4.
        Assert.Equal(new[] { 2 * 8 + 7, 0, 8, 4 }, plan.Index);
    }

    [Fact]
    public void SpatialToChannel_CountsUniqueCoarseCells_AndChannelToSpatialRestoresThem()
    {
        Random rng = new(31);
        int res = 8, n = 40;
        int[] coords = RandomCoords(rng, n, res);
        using Tensor feats = RandomFeats(rng, n, 3);
        CpuBackend backend = new();

        SparseSpatialChannel.Plan down = SparseSpatialChannel.PlanSpatialToChannel(coords, res, 2);
        int coarse = down.Coords.Length / 4;
        Assert.InRange(coarse, 1, n);
        using Tensor merged = SparseSpatialChannel.SpatialToChannelFeats(backend, feats, down, 3);

        // Occupancy mask: every child that exists in the input is kept, so the expansion mirrors the input exactly.
        Dictionary<(int, int, int, int), int> present = new();
        for (int i = 0; i < n; i++) present[(coords[i * 4], coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3])] = i;
        using Tensor logits = Logits(down.Coords, coarse, present);
        SparseSpatialChannel.Plan up = SparseSpatialChannel.PlanChannelToSpatial(down.Coords, down.Resolution, 2, logits);
        using Tensor restored = SparseSpatialChannel.ChannelToSpatialFeats(backend, merged, up, 3);

        Assert.Equal(n, up.Coords.Length / 4);
        Assert.Equal(res, up.Resolution);
        Assert.Equal(SortedKeys(coords), SortedKeys(up.Coords));
        AssertFeatsMatch(coords, feats, up.Coords, restored);
    }

    [Fact]
    public void ChannelToSpatial_ExpandsOnlyPositiveChildrenInAscendingOrder()
    {
        int[] coords = { 0, 1, 1, 1 };
        using Tensor logits = Feats(new float[] { -1f, 1f, 0f, 2f, -3f, 0.5f, 0f, 1f }, 1, 8);

        SparseSpatialChannel.Plan plan = SparseSpatialChannel.PlanChannelToSpatial(coords, 4, 2, logits);

        // Parent (1,1,1) doubles to (2,2,2); children 1, 3, 5, 7 offset it by (x%2, y%2, z%2) in ascending order.
        Assert.Equal(new[] { 0, 3, 2, 2, 0, 3, 3, 2, 0, 3, 2, 3, 0, 3, 3, 3 }, plan.Coords);
        Assert.Equal(new[] { 1, 3, 5, 7 }, plan.Index.Select(i => i % 8).ToArray());
        Assert.Equal(8, plan.Resolution);
    }

    [Fact]
    public void MeanChannelGroups_AveragesContiguousGroups()
    {
        using Tensor feats = Feats(new float[] { 1f, 3f, 5f, 7f, 2f, 2f, 4f, 8f }, 2, 4);

        using Tensor mean = SparseSpatialChannel.MeanChannelGroups(feats, 2);

        Assert.Equal(new float[] { 2f, 6f, 2f, 6f }, ShapeVaeTestData.Read(mean));
    }

    [Fact]
    public void RepeatChannels_CopiesEachChannelInPlace()
    {
        using Tensor feats = Feats(new float[] { 1f, 2f }, 1, 2);

        using Tensor repeated = SparseSpatialChannel.RepeatChannels(feats, 3);

        Assert.Equal(new float[] { 1f, 1f, 1f, 2f, 2f, 2f }, ShapeVaeTestData.Read(repeated));
    }

    [Fact]
    public void ValidateCoords_RejectsOutOfRangeVoxels()
    {
        Assert.Throws<ArgumentException>(() => SparseSpatialChannel.ValidateCoords(new[] { 0, 4, 0, 0 }, 4));
        Assert.Throws<ArgumentException>(() => SparseSpatialChannel.ValidateCoords(new[] { 0, -1, 0, 0 }, 4));
        Assert.Throws<ArgumentException>(() => SparseSpatialChannel.ValidateCoords(new[] { 0, 0, 0 }, 4));
    }

    private static Tensor Feats(float[] values, int rows, int channels)
    {
        Tensor t = new(new TensorShape(rows, channels), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < values.Length; i++) p[i] = values[i];
        return t;
    }

    private static Tensor RandomFeats(Random rng, int rows, int channels)
    {
        Tensor t = new(new TensorShape(rows, channels), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static int[] RandomCoords(Random rng, int n, int res)
    {
        HashSet<(int, int, int)> seen = new();
        List<int> coords = new();
        while (seen.Count < n)
        {
            int x = rng.Next(res), y = rng.Next(res), z = rng.Next(res);
            if (seen.Add((x, y, z))) coords.AddRange(new[] { 0, x, y, z });
        }
        return coords.ToArray();
    }

    /// <summary>Logits for each parent child slot: positive where the child exists in <paramref name="present"/>.</summary>
    private static Tensor Logits(int[] parentCoords, int parents, Dictionary<(int, int, int, int), int> present)
    {
        Tensor t = new(new TensorShape(parents, 8), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < parents; i++)
            for (int s = 0; s < 8; s++)
            {
                int cx = parentCoords[i * 4 + 1] * 2 + s % 2;
                int cy = parentCoords[i * 4 + 2] * 2 + s / 2 % 2;
                int cz = parentCoords[i * 4 + 3] * 2 + s / 4 % 2;
                p[i * 8 + s] = present.ContainsKey((parentCoords[i * 4], cx, cy, cz)) ? 1f : -1f;
            }
        return t;
    }

    private static List<(int, int, int, int)> SortedKeys(int[] coords)
    {
        List<(int, int, int, int)> keys = new();
        for (int i = 0; i < coords.Length; i += 4) keys.Add((coords[i], coords[i + 1], coords[i + 2], coords[i + 3]));
        keys.Sort();
        return keys;
    }

    private static void AssertFeatsMatch(int[] inputCoords, Tensor inputFeats, int[] outputCoords, Tensor outputFeats)
    {
        Dictionary<(int, int, int, int), float> byCoord = new();
        float* f = (float*)inputFeats.DataPointer;
        for (int i = 0; i < inputCoords.Length / 4; i++)
            byCoord[(inputCoords[i * 4], inputCoords[i * 4 + 1], inputCoords[i * 4 + 2], inputCoords[i * 4 + 3])] = f[i * 3];

        float[] restored = ShapeVaeTestData.Read(outputFeats);
        for (int i = 0; i < outputCoords.Length / 4; i++)
        {
            (int, int, int, int) key = (outputCoords[i * 4], outputCoords[i * 4 + 1], outputCoords[i * 4 + 2], outputCoords[i * 4 + 3]);
            Assert.Equal(byCoord[key], restored[i * 3], 5);
        }
    }
}

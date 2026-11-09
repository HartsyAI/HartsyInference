using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed class UtoniaGridPoolingTests
{
    [Fact]
    public void Forward_ProducesOneRowPerCoarseVoxelPerSampleInGridOrder()
    {
        CpuBackend backend = new();
        UtoniaGridPooling pool = new(backend, Identity(backend, 3), UnitNorm(backend, 3), stride: 2);
        // Batch 0: (0,0,0) and (1,1,1) share a coarse cell; (2,0,0) and (3,3,3) do not. Batch 1: (1,1,1) and (5,0,0).
        int[] grid = { 0, 0, 0, 1, 1, 1, 2, 0, 0, 3, 3, 3, 1, 1, 1, 5, 0, 0 };
        int[] batch = { 0, 0, 0, 0, 1, 1 };
        UtoniaPoint input = Input(backend, grid, batch, 3);

        UtoniaPoint output = pool.Forward(input, new[] { UtoniaSerialOrder.ZOrder });

        Assert.Equal(5, output.Count);
        Assert.Equal(new[] { 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 0 }, output.GridCoord);
        Assert.Equal(new[] { 0, 0, 0, 1, 1 }, output.Batch);
        Assert.Equal(new[] { 0, 0, 1, 2, 3, 4 }, output.PoolingInverse);
        Assert.Same(input, output.PoolingParent);
        Assert.NotNull(output.Serialized);
        Assert.NotNull(output.SparseCoords);
    }

    [Fact]
    public void Forward_AveragesCoordinatesWithinEachCluster()
    {
        CpuBackend backend = new();
        UtoniaGridPooling pool = new(backend, Identity(backend, 3), UnitNorm(backend, 3), stride: 2);
        UtoniaPoint input = Input(backend, new[] { 0, 0, 0, 1, 1, 1 }, new[] { 0, 0 }, 3);

        UtoniaPoint output = pool.Forward(input, new[] { UtoniaSerialOrder.ZOrder });

        Assert.Equal(1, output.Count);
        foreach (float v in output.Coord)
        {
            Assert.Equal(0.005f, v, 6);
        }
    }

    private static UtoniaPoint Input(CpuBackend backend, int[] grid, int[] batch, int channels)
    {
        int n = batch.Length;
        float[] feat = new float[n * channels];
        for (int i = 0; i < feat.Length; i++)
        {
            feat[i] = (i % 3) + 1;
        }

        float[] coord = grid.Select(v => v * 0.01f).ToArray();
        return new UtoniaPoint(UtoniaTestData.Make(new long[] { n, channels }, feat), coord, grid, batch);
    }

    private static UtoniaLinear Identity(CpuBackend backend, int channels)
    {
        float[] w = new float[channels * channels];
        for (int i = 0; i < channels; i++)
        {
            w[i * channels + i] = 1f;
        }

        return new UtoniaLinear(backend, UtoniaTestData.Make(new long[] { channels, channels }, w),
            UtoniaTestData.Make(new long[] { channels }, new float[channels]));
    }

    private static UtoniaLayerNorm UnitNorm(CpuBackend backend, int channels)
    {
        float[] ones = Enumerable.Repeat(1f, channels).ToArray();
        return new UtoniaLayerNorm(backend, UtoniaTestData.Make(new long[] { channels }, ones),
            UtoniaTestData.Make(new long[] { channels }, new float[channels]));
    }
}

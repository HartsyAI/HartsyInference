using HartsyInference.LLM.Transformer.Dsa;
using Xunit;

namespace HartsyInference.LLM.Tests.Dsa;

/// <summary>The helpers the compressed-pool indexers share: the gated softmax pool, the descending top-k with ties to the lower index, and the row packing.</summary>
public sealed class DsaCompressedPoolTests
{
    [Fact]
    public void GatedSoftmaxPool_WeightsEachSlotBySoftmaxOfGateAndApe()
    {
        // One group of two slots, one channel, stride 1: gates 0 and ln 3 give weights 1/4 and 3/4 of the values.
        float[] values = [10f, 20f];
        float[] gates = [0f, MathF.Log(3f)];
        float[] ape = [0f, 0f];
        float[] pooled = new float[1];

        DsaCompressedPool.GatedSoftmaxPool(values, gates, 1, ape, groups: 1, slots: 2, channels: 1, pooled);

        Assert.Equal(0.25f * 10f + 0.75f * 20f, pooled[0], 5);
    }

    [Fact]
    public void GatedSoftmaxPool_ReadsValuesAndGatesAtTheirStride()
    {
        // Interleaved rows [value, gate] of width 2, two slots, one group: the gate view starts one column in.
        float[] store = [1f, 0f, 3f, 0f];
        float[] ape = [0f, 0f];
        float[] pooled = new float[1];

        DsaCompressedPool.GatedSoftmaxPool(store, store.AsSpan(1), 2, ape, groups: 1, slots: 2, channels: 1, pooled);

        Assert.Equal(2f, pooled[0], 5);
    }

    [Fact]
    public void TopK_IsDescendingWithTiesToTheLowerIndex()
    {
        float[] scores = [1f, 3f, 3f, 2f];

        Assert.Equal([1, 2, 3], DsaCompressedPool.TopK(scores, 3));
        Assert.Equal([1, 2, 3, 0], DsaCompressedPool.TopK(scores, 10));
    }

    [Fact]
    public void Pack_PadsEachRowWithMinusOneToTheWidestRow()
    {
        List<int>[] rows = [[0, 1, 2], [5]];

        DsaKeySelection selection = DsaCompressedPool.Pack(rows);

        Assert.Equal(3, selection.Width);
        Assert.Equal([0, 1, 2, 5, -1, -1], selection.Rows);
        Assert.Equal([3, 1], selection.Counts);
    }
}

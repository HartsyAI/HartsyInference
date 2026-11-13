using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Transformer.AttnRes;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The attention-residual source rule of the Kimi decoder: an attention mix reads the running partial sum only while a block is
/// open, and the MLP and output mixes always read it. The decoder's own mix must equal the shared mix over exactly those sources.</summary>
public sealed class KimiK31AttnResSourceTests
{
    private const float Tol = 1e-5f;

    [Fact]
    public void AttentionMix_ReadsThePartialOnlyInsideABlock()
    {
        // Block size 2: layers 0 and 2 open a block with no partial yet; layers 1 and 3 continue one.
        Assert.False(KimiK31Model.AttentionMixIncludesPartial(0, 2));
        Assert.True(KimiK31Model.AttentionMixIncludesPartial(1, 2));
        Assert.False(KimiK31Model.AttentionMixIncludesPartial(2, 2));
        Assert.True(KimiK31Model.AttentionMixIncludesPartial(3, 2));
    }

    [Fact]
    public void Mix_WithoutPartial_IsTheMixOfTheCompletedBlocksOnly()
    {
        KimiK31Config c = KimiK31ModelTests.Config;
        KimiK31Weights weights = KimiK31ModelTests.RandomWeights(new Random(9));
        using KimiK31Model model = new(new CpuBackend(), c, weights, maxPositions: 16);
        using KimiK31State state = new(c, capacity: 16);
        float[] embedding = Vector(c.Hidden, 1), block = Vector(c.Hidden, 2), partial = Vector(c.Hidden, 3);
        state.Blocks.Clear();
        state.Blocks.Add(embedding);
        state.Blocks.Add(block);
        // The state a block's first layer sees: the running sum was just closed into Blocks, so nothing is running.
        Array.Clear(state.Partial);

        KimiK31LayerWeights lw = weights.Layers[2];
        float[] actual = model.Mix(state, lw.AttnResAttnQuery, lw.AttnResAttnNorm, includePartial: false);
        AssertClose(Reference(c, [embedding, block], lw.AttnResAttnQuery, lw.AttnResAttnNorm), actual);
    }

    [Fact]
    public void Mix_WithPartial_AddsTheRunningSumAsTheLastSource()
    {
        KimiK31Config c = KimiK31ModelTests.Config;
        KimiK31Weights weights = KimiK31ModelTests.RandomWeights(new Random(9));
        using KimiK31Model model = new(new CpuBackend(), c, weights, maxPositions: 16);
        using KimiK31State state = new(c, capacity: 16);
        float[] embedding = Vector(c.Hidden, 1), block = Vector(c.Hidden, 2), partial = Vector(c.Hidden, 3);
        state.Blocks.Clear();
        state.Blocks.Add(embedding);
        state.Blocks.Add(block);
        partial.CopyTo(state.Partial, 0);

        KimiK31LayerWeights lw = weights.Layers[3];
        float[] actual = model.Mix(state, lw.AttnResMlpQuery, lw.AttnResMlpNorm, includePartial: true);
        AssertClose(Reference(c, [embedding, block, partial], lw.AttnResMlpQuery, lw.AttnResMlpNorm), actual);
    }

    // The shared mix over the given sources, in order: the reference the decoder's source assembly must match.
    private static float[] Reference(KimiK31Config c, IReadOnlyList<float[]> sources, Tensor query, Tensor norm)
    {
        using Tensor stacked = new(new TensorShape(sources.Count, c.Hidden), DType.F32);
        for (int i = 0; i < sources.Count; i++) sources[i].CopyTo(stacked.AsSpan<float>().Slice(i * c.Hidden, c.Hidden));
        using Tensor output = new(new TensorShape(1, c.Hidden), DType.F32);
        AttentionResidual.Mix(new CpuBackend(), stacked, norm, query, output);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private static float[] Vector(int n, float seed)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = MathF.Sin(seed * (i + 1)) + 0.5f * seed;
        return v;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], Tol);
    }
}

using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.DeepSeekV41.Engram;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>A draft tap on a block that has Engram is the hc-mean of the stream after the Engram step, as upstream's <c>h.mean(dim=2)</c> after <c>layer.engram</c>. The
/// expected value applies the same Engram module to the embedding stream, with the hash ids the host computes, and the tap must match it and must differ from the
/// mean before the step, so the test cannot pass with the placement wrong.</summary>
public sealed class DeepSeekV41EngramTapTests
{
    [Fact]
    public void A_Tap_On_A_Block_With_Engram_Is_Taken_After_The_Engram_Step()
    {
        using CpuBackend cpu = new();
        const int dim = 16, hc = 2, headDim = 4;
        int columns = EngramConstants.ColumnsPerLayer;
        Random random = new(11);
        float[] Values(int count, float scale) => Enumerable.Range(0, count).Select(_ => (float)((random.NextDouble() * 2 - 1) * scale)).ToArray();

        // non-zero weights of useful size, so the step moves the stream visibly; the table rows are fp16 4.0 everywhere
        void Gather(ReadOnlySpan<long> rows, Span<ushort> dest) => dest.Fill((ushort)0x4400);
        const int slot = 1;
        DeepSeekV41EngramModule engram = new(dim, hc, columns, headDim, 1e-6f, Values(dim * (hc + 1) * columns * headDim, 1f),
            Values(hc * dim, 2f), Values(hc * dim, 2f), Gather);
        DeepSeekV41Block[] blocks = [DeepSeekV41HostModelTests.BuildBlock(cpu, 0, engram, slot), DeepSeekV41HostModelTests.BuildBlock(cpu, 1)];

        float[] embed = DeepSeekV41DSparkFixture.TargetEmbedAndHead().Embed;
        int vocab = embed.Length / dim;
        DeepSeekV41HostModel model = new(dim, hc, vocab, 1e-6f, embed, blocks, Enumerable.Repeat(1f, dim).ToArray(),
            DeepSeekV41DSparkFixture.TargetEmbedAndHead().Head, mainHiddenLayers: new[] { 0 });
        int[] ids = DeepSeekV41DSparkFixture.Ids[..6];
        float[] tap = new float[ids.Length * model.MainHiddenWidth];
        model.Forward(ids, model.CreateState(DeepSeekV41DSparkFixture.MaxTokens), new float[ids.Length * dim], tap);

        // the embedding copied into every hc stream, as the host builds it
        float[] stream = new float[ids.Length * hc * dim];
        for (int t = 0; t < ids.Length; t++)
            for (int c = 0; c < hc; c++) embed.AsSpan(ids[t] * dim, dim).CopyTo(stream.AsSpan((t * hc + c) * dim, dim));
        float[] before = HcMean(stream, ids.Length, hc, dim);

        // the same hash ids the host gives this block's slot
        EngramHasher hasher = new();
        long[] all = new long[ids.Length * hasher.ValuesPerPosition];
        hasher.Hash(ids, Enumerable.Repeat(true, ids.Length).ToArray(), 0, all);
        int hashLayers = hasher.ValuesPerPosition / columns;
        long[] mine = new long[ids.Length * columns];
        for (int t = 0; t < ids.Length; t++) all.AsSpan((t * hashLayers + slot) * columns, columns).CopyTo(mine.AsSpan(t * columns, columns));
        engram.Apply(stream, ids.Length, mine, ReadOnlySpan<bool>.Empty);
        float[] expected = HcMean(stream, ids.Length, hc, dim);

        float moved = 0;
        for (int i = 0; i < expected.Length; i++) moved = Math.Max(moved, Math.Abs(expected[i] - before[i]));
        Assert.True(moved > 1e-3f, $"the Engram step moved the mean by at most {moved}, so this test would not tell the placements apart");
        for (int i = 0; i < expected.Length; i++)
        {
            int t = i / dim, d = i % dim;
            Assert.True(Math.Abs(expected[i] - tap[t * model.MainHiddenWidth + d]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"tap[{t}][{d}]: {tap[t * model.MainHiddenWidth + d]} vs expected {expected[i]}");
        }
    }

    /// <summary>The mean over the hc copies of each position, <c>[tokens, dim]</c>.</summary>
    private static float[] HcMean(float[] stream, int tokens, int hc, int dim)
    {
        float[] mean = new float[tokens * dim];
        for (int t = 0; t < tokens; t++)
            for (int d = 0; d < dim; d++)
            {
                float sum = 0f;
                for (int c = 0; c < hc; c++) sum += stream[(t * hc + c) * dim + d];
                mean[t * dim + d] = sum / hc;
            }
        return mean;
    }
}

using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer.Compression;
using Xunit;

namespace HartsyInference.LLM.Tests.Transformer;

/// <summary>The batched HCA attention (one <c>SparseLatentAttention</c> call per chunk, the GPU path) against the per-query path on the CPU, over
/// a prefill, a multi-token chunk past position 0 and decode steps: the window ring wraps, and compressor groups complete inside the chunk and
/// during decode.</summary>
public sealed class HcaBatchedAttentionTests : IDisposable
{
    private const int Dim = 32, Window = 6, Ratio = 8;
    private readonly List<Tensor> _owned = [];
    private readonly Random _rng = new(17);

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
    }

    [Theory]
    [InlineData(new[] { 21, 1, 1, 1 })]          // prefill, then decode across the group ending at position 23
    [InlineData(new[] { 13, 10, 1, 1, 1, 1 })]   // a 10-token chunk at position 13 completes the group at 15 and wraps the ring, then decode
    [InlineData(new[] { 3, 1, 9, 2 })]           // a short prefill, a decode, then chunks that cross groups 7 and 15
    public void Batched_attention_matches_the_per_query_path(int[] chunks)
    {
        HcaAttentionSettings settings = new()
        {
            Dim = Dim, Heads = 4, HeadDim = 16, RopeDim = 8, QLoraRank = 12, OGroups = 4, OLoraRank = 8, Window = Window,
            CompressRatio = Ratio, NormEps = 1e-6f, QueryHeadNorm = true,
        };
        int total = chunks.Sum(), h = settings.Heads, hd = settings.HeadDim, grouped = settings.OGroups * settings.OLoraRank;
        HcaAttentionWeights weights = new()
        {
            WqA = Own(settings.QLoraRank, Dim), QNorm = Gains(settings.QLoraRank), WqB = Own(h * hd, settings.QLoraRank),
            Wkv = Own(hd, Dim), KvNorm = Gains(hd), CompressorWkv = Own(hd, Dim), CompressorWgate = Own(hd, Dim),
            CompressorApe = Values(Ratio * hd, 0.1f), CompressorNorm = [.. Values(hd, 0.2f).Select(v => 1f + v)],
            Sink = Own(h), WoA = Own(grouped, settings.GroupDim), WoB = Own(Dim, grouped), OGate = Own(grouped, settings.QLoraRank),
            HeadMix = Own(h, h),
        };
        DeepSeekV41RopeTable rope = DeepSeekV41RopeTable.Build(settings.RopeDim, total, 10000, null);
        float[] x = Values(total * Dim, 1f);

        float[] perQuery = Run(settings, weights, rope, x, chunks, batched: false);
        float[] batched = Run(settings, weights, rope, x, chunks, batched: true);
        for (int i = 0; i < perQuery.Length; i++)
            Assert.True(MathF.Abs(perQuery[i] - batched[i]) <= 1e-4f * MathF.Max(1f, MathF.Abs(perQuery[i])),
                $"output {i} (position {i / Dim}): per query {perQuery[i]}, batched {batched[i]}");
    }

    private static float[] Run(HcaAttentionSettings settings, HcaAttentionWeights weights, DeepSeekV41RopeTable rope, float[] x, int[] chunks,
        bool batched)
    {
        using CpuBackend cpu = new();
        using HcaAttention layer = new(cpu, settings, weights, rope) { BatchedOnCpu = batched };
        HcaAttentionState state = new(x.Length / Dim, settings);
        float[] y = new float[x.Length];
        int pos = 0;
        foreach (int n in chunks)
        {
            layer.Forward(x.AsSpan(pos * Dim, n * Dim), n, pos, state, y.AsSpan(pos * Dim, n * Dim));
            pos += n;
        }
        return y;
    }

    private Tensor Own(params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        Values((int)t.ElementCount, 0.2f).CopyTo(t.AsSpan<float>());
        _owned.Add(t);
        return t;
    }

    private Tensor Gains(int count)
    {
        Tensor t = new(new TensorShape(count), DType.F32);
        Values(count, 0.2f).Select(v => 1f + v).ToArray().CopyTo(t.AsSpan<float>());
        _owned.Add(t);
        return t;
    }

    private float[] Values(int count, float amplitude)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)((_rng.NextDouble() * 2 - 1) * amplitude);
        return v;
    }
}

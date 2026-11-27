using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.LLM.Tests;

public sealed class MossVlVisionTowerTests
{
    private const int Hidden = 8, Heads = 2, Inter = 16, Out = 6, Merge = 2, Depth = 3, Side = 4;
    private const int Gx = 4, Gy = 4;

    private static Qwen3VlVisionConfig TinyConfig() => new()
    {
        Depth = Depth, HiddenSize = Hidden, NumHeads = Heads, IntermediateSize = Inter, InChannels = 3,
        PatchSize = 2, TemporalPatchSize = 1, SpatialMergeSize = Merge, OutHiddenSize = Out,
        NumPositionEmbeddings = Side * Side, DeepstackVisualIndexes = [0, 1],
        UnifiedDeepstackMerger = true, MergerExactGelu = true,
    };

    /// <summary>Synthetic F32 weights for the tiny tower, keyed as the checkpoint names them after the prefix strip.</summary>
    private static Dictionary<string, Tensor> TinyWeights(int seed, Func<string, bool>? skip = null)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        void Add(string key, params long[] dims)
        {
            if (skip?.Invoke(key) == true) return;
            Tensor t = new(new TensorShape(dims), DType.F32);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            w[key] = t;
        }
        void Norm(string key, int n)
        {
            Add(key + ".weight", n);
            Add(key + ".bias", n);
        }
        Add("patch_embed.proj.weight", Hidden, 3, 1, 2, 2);
        Add("patch_embed.proj.bias", Hidden);
        Add("pos_embed.weight", Side * Side, Hidden);
        for (int b = 0; b < Depth; b++)
        {
            string p = $"blocks.{b}";
            Norm($"{p}.norm1", Hidden);
            Add($"{p}.attn.qkv.weight", 3 * Hidden, Hidden);
            Add($"{p}.attn.qkv.bias", 3 * Hidden);
            Add($"{p}.attn.proj.weight", Hidden, Hidden);
            Add($"{p}.attn.proj.bias", Hidden);
            Norm($"{p}.norm2", Hidden);
            Add($"{p}.mlp.linear_fc1.weight", Inter, Hidden);
            Add($"{p}.mlp.linear_fc1.bias", Inter);
            Add($"{p}.mlp.linear_fc2.weight", Hidden, Inter);
            Add($"{p}.mlp.linear_fc2.bias", Hidden);
        }
        int streams = 1 + 2;
        int mergedIn = Hidden * streams * Merge * Merge;
        for (int s = 0; s < streams; s++) Norm($"merger.norms.{s}", Hidden);
        Add("merger.linear_fc1.weight", mergedIn, mergedIn);
        Add("merger.linear_fc1.bias", mergedIn);
        Add("merger.linear_fc2.weight", Out, mergedIn);
        Add("merger.linear_fc2.bias", Out);
        return w;
    }

    private static Tensor Patches(int seq, int flatDim, int seed)
    {
        Random rng = new(seed);
        Tensor t = new(new TensorShape(seq, flatDim), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Fact]
    public void MergesEveryStreamIntoOneRowPerMergeBlock()
    {
        using Qwen3VlVisionEncoder encoder = new(TinyConfig());
        encoder.LoadWeights(TinyWeights(7));
        using Tensor patches = Patches(Gx * Gy, 3 * 1 * 2 * 2, seed: 11);
        using IBackend backend = new CpuBackend();

        Qwen3VlVisionEncoder.VisionOutput output = encoder.Forward(backend, patches, 1, Gy, Gx);
        try
        {
            Assert.Equal((long)(Gx * Gy / (Merge * Merge)), output.MergedTokens.Shape[0]);
            Assert.Equal((long)Out, output.MergedTokens.Shape[1]);
            Assert.Empty(output.DeepstackFeatures);
            Span<float> values = output.MergedTokens.AsSpan<float>();
            foreach (float v in values) Assert.True(float.IsFinite(v));
        }
        finally
        {
            output.MergedTokens.Dispose();
        }
    }

    [Fact]
    public void DeepstackTapWeightsReachTheMergedOutput()
    {
        using Tensor patches = Patches(Gx * Gy, 3 * 1 * 2 * 2, seed: 11);
        float[] baseline = Merged(TinyWeights(7), patches);
        float[] tapNormChanged = Merged(TinyWeights(7), patches, bumpTapNorm: true);

        Assert.NotEqual(baseline, tapNormChanged);
    }

    [Fact]
    public void NamesAMissingMergerWeight()
    {
        using Qwen3VlVisionEncoder encoder = new(TinyConfig());
        IReadOnlyDictionary<string, Tensor> w = TinyWeights(7, key => key == "merger.norms.2.weight");
        Assert.Throws<KeyNotFoundException>(() => encoder.LoadWeights(w));
    }

    private static float[] Merged(Dictionary<string, Tensor> weights, Tensor patches, bool bumpTapNorm = false)
    {
        if (bumpTapNorm)
        {
            Span<float> g = weights["merger.norms.2.weight"].AsSpan<float>();
            for (int i = 0; i < g.Length; i++) g[i] *= 2f;
        }
        using Qwen3VlVisionEncoder encoder = new(TinyConfig());
        encoder.LoadWeights(weights);
        using IBackend backend = new CpuBackend();
        Qwen3VlVisionEncoder.VisionOutput output = encoder.Forward(backend, patches, 1, Gy, Gx);
        try
        {
            return output.MergedTokens.AsSpan<float>().ToArray();
        }
        finally
        {
            output.MergedTokens.Dispose();
        }
    }
}

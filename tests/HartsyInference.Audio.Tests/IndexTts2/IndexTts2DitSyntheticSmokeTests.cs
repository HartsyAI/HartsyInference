using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Tiny random weights through <see cref="IndexTts2Dit"/> (the S2Mel flow-matching estimator): shapes,
/// finiteness, and U-ViT skip-bookkeeping correctness (depth=5 so both an emit half, a dead-center layer, and
/// a receive half are exercised) only. Says nothing about parity with the real <c>s2mel.pth</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IndexTts2DitSyntheticSmokeTests : IDisposable
{
    private const int Hidden = 8;
    private const int Heads = 2;
    private const int Depth = 5;          // half=2: emit {0,1}, dead-center {2}, receive {3,4}.
    // IndexTts2DitConfig.FfnDim is a derived property (find_multiple(2*4*hidden/3, 256), matching the real
    // checkpoint's own formula) — not independently settable, so the test's synthetic weights must be built
    // to whatever it actually computes for Hidden, not an arbitrary small constant.
    private static readonly int FfnDim = Cfg().FfnDim;
    private const int InChannels = 4;
    private const int ContentDim = 6;
    private const int StyleDim = 3;
    private const int WavenetHidden = 8;
    private const int WavenetKernel = 3;
    private const int WavenetLayers = 2;
    private const int FreqEmbedSize = 8;

    private readonly List<Tensor> _owned = [];
    private readonly CpuBackend _backend = new();

    private Tensor Rand(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * 0.1);
        _owned.Add(t);
        return t;
    }

    private Tensor Ones(params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = 1f;
        _owned.Add(t);
        return t;
    }

    private void AddTimestepEmbedder(Dictionary<string, Tensor> w, Random rng, string prefix, int hidden)
    {
        w[$"{prefix}.freqs"] = Rand(rng, FreqEmbedSize / 2);
        w[$"{prefix}.mlp.0.weight"] = Rand(rng, hidden, FreqEmbedSize);
        w[$"{prefix}.mlp.0.bias"] = Rand(rng, hidden);
        w[$"{prefix}.mlp.2.weight"] = Rand(rng, hidden, hidden);
        w[$"{prefix}.mlp.2.bias"] = Rand(rng, hidden);
    }

    private void AddAdaptiveNorm(Dictionary<string, Tensor> w, Random rng, string prefix, int hidden)
    {
        w[$"{prefix}.norm.weight"] = Ones(hidden);
        w[$"{prefix}.project_layer.weight"] = Rand(rng, 2 * hidden, hidden);
        w[$"{prefix}.project_layer.bias"] = Rand(rng, 2 * hidden);
    }

    private Dictionary<string, Tensor> BuildWeights(Random rng, string prefix)
    {
        Dictionary<string, Tensor> w = [];
        for (int i = 0; i < Depth; i++)
        {
            string bp = $"{prefix}.transformer.layers.{i}";
            AddAdaptiveNorm(w, rng, $"{bp}.attention_norm", Hidden);
            AddAdaptiveNorm(w, rng, $"{bp}.ffn_norm", Hidden);
            w[$"{bp}.attention.wqkv.weight"] = Rand(rng, 3 * Hidden, Hidden);
            w[$"{bp}.attention.wo.weight"] = Rand(rng, Hidden, Hidden);
            w[$"{bp}.feed_forward.w1.weight"] = Rand(rng, FfnDim, Hidden);
            w[$"{bp}.feed_forward.w2.weight"] = Rand(rng, Hidden, FfnDim);
            w[$"{bp}.feed_forward.w3.weight"] = Rand(rng, FfnDim, Hidden);
            w[$"{bp}.skip_in_linear.weight"] = Rand(rng, Hidden, 2 * Hidden);
            w[$"{bp}.skip_in_linear.bias"] = Rand(rng, Hidden);
        }
        AddAdaptiveNorm(w, rng, $"{prefix}.transformer.norm", Hidden);
        AddTimestepEmbedder(w, rng, $"{prefix}.t_embedder", Hidden);
        AddTimestepEmbedder(w, rng, $"{prefix}.t_embedder2", WavenetHidden);

        w[$"{prefix}.cond_projection.weight"] = Rand(rng, Hidden, ContentDim);
        w[$"{prefix}.cond_projection.bias"] = Rand(rng, Hidden);
        w[$"{prefix}.cond_x_merge_linear.weight"] = Rand(rng, Hidden, InChannels + InChannels + Hidden + StyleDim);
        w[$"{prefix}.cond_x_merge_linear.bias"] = Rand(rng, Hidden);
        w[$"{prefix}.skip_linear.weight"] = Rand(rng, Hidden, Hidden + InChannels);
        w[$"{prefix}.skip_linear.bias"] = Rand(rng, Hidden);

        w[$"{prefix}.conv1.weight"] = Rand(rng, WavenetHidden, Hidden);
        w[$"{prefix}.conv1.bias"] = Rand(rng, WavenetHidden);
        w[$"{prefix}.conv2.weight"] = Rand(rng, InChannels, WavenetHidden, 1);
        w[$"{prefix}.conv2.bias"] = Rand(rng, InChannels);
        w[$"{prefix}.res_projection.weight"] = Rand(rng, WavenetHidden, Hidden);
        w[$"{prefix}.res_projection.bias"] = Rand(rng, WavenetHidden);

        w[$"{prefix}.final_layer.adaLN_modulation.1.weight"] = Rand(rng, 2 * WavenetHidden, Hidden);
        w[$"{prefix}.final_layer.adaLN_modulation.1.bias"] = Rand(rng, 2 * WavenetHidden);
        w[$"{prefix}.final_layer.linear.weight_g"] = Ones(WavenetHidden, 1, 1);
        w[$"{prefix}.final_layer.linear.weight_v"] = Rand(rng, WavenetHidden, WavenetHidden);
        w[$"{prefix}.final_layer.linear.bias"] = Rand(rng, WavenetHidden);

        for (int i = 0; i < WavenetLayers; i++)
        {
            int resSkipCh = i < WavenetLayers - 1 ? 2 * WavenetHidden : WavenetHidden;
            w[$"{prefix}.wavenet.in_layers.{i}.conv.conv.weight"] = Rand(rng, 2 * WavenetHidden, WavenetHidden, WavenetKernel);
            w[$"{prefix}.wavenet.in_layers.{i}.conv.conv.bias"] = Rand(rng, 2 * WavenetHidden);
            w[$"{prefix}.wavenet.res_skip_layers.{i}.conv.conv.weight"] = Rand(rng, resSkipCh, WavenetHidden, 1);
            w[$"{prefix}.wavenet.res_skip_layers.{i}.conv.conv.bias"] = Rand(rng, resSkipCh);
        }
        w[$"{prefix}.wavenet.cond_layer.conv.conv.weight"] = Rand(rng, 2 * WavenetHidden * WavenetLayers, WavenetHidden, 1);
        w[$"{prefix}.wavenet.cond_layer.conv.conv.bias"] = Rand(rng, 2 * WavenetHidden * WavenetLayers);

        return w;
    }

    private static IndexTts2DitConfig Cfg() => new()
    {
        HiddenDim = Hidden,
        NumHeads = Heads,
        Depth = Depth,
        InChannels = InChannels,
        ContentDim = ContentDim,
        StyleDim = StyleDim,
        WavenetHiddenDim = WavenetHidden,
        WavenetKernelSize = WavenetKernel,
        WavenetDilationRate = 1,
        WavenetNumLayers = WavenetLayers,
        FreqEmbedSize = FreqEmbedSize,
    };

    private IndexTts2Dit BuildEstimator(Random rng)
    {
        Dictionary<string, Tensor> w = BuildWeights(rng, "est");
        IndexTts2Dit dit = new(Cfg());
        dit.LoadWeights(w, "est");
        return dit;
    }

    [Fact]
    public void Estimate_ProducesFiniteOutput_OfInputShape()
    {
        Random rng = new(1);
        using IndexTts2Dit dit = BuildEstimator(rng);
        const int t = 9;
        Tensor x = Rand(rng, 1, InChannels, t);
        Tensor mu = Rand(rng, 1, t, ContentDim);
        Tensor spk = Rand(rng, 1, StyleDim);
        Tensor cond = Rand(rng, 1, InChannels, t);

        using Tensor velocity = dit.Estimate(_backend, x, mu, 0.5f, spk, cond);

        Assert.Equal(new TensorShape(1, InChannels, t), velocity.Shape);
        foreach (float v in velocity.AsSpan<float>()) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Estimate_IsDeterministic_ForTheSameInputs()
    {
        Random rng = new(2);
        using IndexTts2Dit dit = BuildEstimator(rng);
        const int t = 7;
        Tensor x = Rand(rng, 1, InChannels, t);
        Tensor mu = Rand(rng, 1, t, ContentDim);
        Tensor spk = Rand(rng, 1, StyleDim);
        Tensor cond = Rand(rng, 1, InChannels, t);

        using Tensor a = dit.Estimate(_backend, x, mu, 0.3f, spk, cond);
        using Tensor b = dit.Estimate(_backend, x, mu, 0.3f, spk, cond);

        float[] av = a.AsSpan<float>().ToArray(), bv = b.AsSpan<float>().ToArray();
        for (int i = 0; i < av.Length; i++) Assert.Equal(av[i], bv[i], 5);
    }

    [Fact]
    public void Solve_ViaConditionalCfm_WithPromptLen_ProducesFiniteZeroPrefixedMel()
    {
        Random rng = new(3);
        using IndexTts2Dit dit = BuildEstimator(rng);
        ConditionalCfm cfm = new(dit, InChannels);

        const int t = 10, promptLen = 3;
        Tensor mu = Rand(rng, 1, t, ContentDim);
        Tensor spk = Rand(rng, 1, StyleDim);
        Tensor cond = Rand(rng, 1, InChannels, t);

        using Tensor result = cfm.Solve(_backend, mu, spk, cond, numSteps: 3, cfgRate: 0.5f, seed: 7, promptLen: promptLen);

        Assert.Equal(new TensorShape(1, InChannels, t), result.Shape);
        Span<float> data = result.AsSpan<float>();
        for (int c = 0; c < InChannels; c++)
            for (int j = 0; j < promptLen; j++)
                Assert.Equal(0f, data[c * t + j]);
        foreach (float v in data) Assert.True(float.IsFinite(v));
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}

/// <summary>Loads the real IndexTTS-2.5 <c>s2mel.pth</c> (a training checkpoint — <c>net.cfm.estimator.*</c>
/// weights plus an <c>optimizer.*</c> section this class ignores; too large to bundle — point
/// <c>INDEXTTS2_S2MEL_PTH_PATH</c> at a local copy). A clean <c>LoadWeights</c> confirms every key name this
/// class expects matches the real checkpoint; one <c>Estimate</c> call on random input confirms the full
/// forward graph (13-layer transformer, U-ViT skips, long skip, WaveNet final stage) runs end to end and
/// produces finite output.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2DitRealWeightTests
{
    [Fact]
    public void Estimate_SucceedsAgainstRealS2MelPth()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS2_S2MEL_PTH_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        try
        {
            using IndexTts2Dit dit = new(IndexTts2DitConfig.IndexTts2);
            dit.LoadWeights(weights, "net.cfm.estimator");

            const int t = 20;
            IndexTts2DitConfig cfg = IndexTts2DitConfig.IndexTts2;
            Random rng = new(5);
            Tensor x = new(new TensorShape(1, cfg.InChannels, t), DType.F32);
            Tensor mu = new(new TensorShape(1, t, cfg.ContentDim), DType.F32);
            Tensor spk = new(new TensorShape(1, cfg.StyleDim), DType.F32);
            Tensor cond = new(new TensorShape(1, cfg.InChannels, t), DType.F32);
            foreach (Tensor tn in new[] { x, mu, spk, cond })
                foreach (ref float v in tn.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

            using CpuBackend backend = new();
            using Tensor velocity = dit.Estimate(backend, x, mu, 0.5f, spk, cond);
            x.Dispose(); mu.Dispose(); spk.Dispose(); cond.Dispose();

            Assert.Equal(new TensorShape(1, cfg.InChannels, t), velocity.Shape);
            foreach (float v in velocity.AsSpan<float>()) Assert.True(float.IsFinite(v));
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Stable Audio Open 1.0's sampling path, from stable-audio-tools' <c>generate_diffusion_cond</c> and k-diffusion's
/// <c>VDenoiser</c> / <c>get_sigmas_polyexponential</c>:
/// <list type="bullet">
///   <item>the noise schedule, time map and input scale;</item>
///   <item>picking the variant from checkpoint shapes;</item>
///   <item>a DiT that reads unprojected cross-attention tokens without QK-norm;</item>
///   <item>the pipeline's v-prediction run with CFG on a tiny model and a stand-in VAE.</item>
/// </list></summary>
public sealed class StableAudioOpen1Tests
{
    [Fact]
    public void Sigmas_AreExponentialFromMaxToMinThenZero()
    {
        float[] s = StableAudioEdm.Sigmas(3);

        Assert.Equal(4, s.Length);
        Assert.Equal(500f, s[0], 3);
        // rho 1: the middle level is the geometric mean of the ends.
        Assert.Equal(MathF.Sqrt(500f * 0.3f), s[1], 3);
        Assert.Equal(0.3f, s[2], 5);
        Assert.Equal(0f, s[3]);
        Assert.Equal(new[] { 500f, 0f }, StableAudioEdm.Sigmas(1));
    }

    [Fact]
    public void TimeAndInputScale_FollowVDenoiser()
    {
        Assert.Equal(0f, StableAudioEdm.TimestepOf(0f));
        Assert.Equal(0.5f, StableAudioEdm.TimestepOf(1f), 6);
        Assert.InRange(StableAudioEdm.TimestepOf(500f), 0.998f, 1f);
        Assert.Equal(1f, StableAudioEdm.InputScale(0f));
        Assert.Equal(1f / MathF.Sqrt(2f), StableAudioEdm.InputScale(1f), 6);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void FromShapes_PicksTheReleasedVariant(bool open1)
    {
        StableAudioDitConfig expected = open1 ? StableAudioDitConfig.Open1_0 : StableAudioDitConfig.OpenSmall;

        StableAudioDitConfig picked = StableAudioDitConfig.FromShapes(Shapes(expected, condInput: open1 ? 768 : expected.HiddenSize));

        Assert.Same(open1 ? StableAudioDitConfig.Open1_0.GetType() : StableAudioDitConfig.OpenSmall.GetType(), picked.GetType());
        Assert.Equal((expected.HiddenSize, expected.NumLayers, expected.HasSecondsStart), (picked.HiddenSize, picked.NumLayers, picked.HasSecondsStart));
    }

    [Fact]
    public void FromShapes_RefusesAShapeNeitherVariantHas()
    {
        Dictionary<string, TensorShape> shapes = Shapes(StableAudioDitConfig.Open1_0, condInput: 768);
        shapes["transformer.layers.0.ff.ff.0.proj.weight"] = new TensorShape(2 * 4096, 1536);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => StableAudioDitConfig.FromShapes(shapes));
        Assert.Contains("FFN 4096", ex.Message);
    }

    [Fact]
    public void Dit_ReadsUnprojectedTokensWithoutQkNorm()
    {
        StableAudioDitConfig cfg = Tiny;
        Dictionary<string, Tensor> w = DitWeights(cfg);
        try
        {
            using CpuBackend backend = new CpuBackend();
            using StableAudioDit dit = new StableAudioDit(cfg);
            dit.LoadWeights(w);
            using Tensor latent = Rand(new TensorShape(1, cfg.IoChannels, cfg.MaxLatentTokens), 11, 1f);
            using Tensor cond = Rand(new TensorShape(1, 5, cfg.CondTokenDim), 12, 1f);
            using Tensor global = Rand(new TensorShape(1, 1, cfg.GlobalCondDim), 13, 1f);

            using Tensor v = dit.Forward(backend, latent, cond, global, 0.5f);

            Assert.Equal(new TensorShape(1, cfg.IoChannels, cfg.MaxLatentTokens), v.Shape);
            Assert.All(v.AsSpan<float>().ToArray(), x => Assert.True(float.IsFinite(x)));
        }
        finally
        {
            DisposeAll(w);
        }
    }

    [Fact]
    public void Open1T5Weights_UsesComfyTextEncoderNamespace()
    {
        using Tensor shared = Fill(4, 0.5f);
        Dictionary<string, Tensor> all = new(StringComparer.Ordinal)
        {
            ["text_encoders.t5base.transformer.shared.weight"] = shared,
        };

        Dictionary<string, Tensor> selected = StableAudioMusicModel.SelectOpen1T5Weights(all);

        Assert.Same(shared, selected["shared.weight"]);
    }

    [Fact]
    public void Pipeline_RunsVPredictionWithGuidanceAndTrimsToTheWindow()
    {
        StableAudioDitConfig cfg = Tiny;
        Dictionary<string, Tensor> w = DitWeights(cfg);
        Dictionary<string, Tensor> timing = TimingWeights(cfg);
        try
        {
            using CpuBackend backend = new CpuBackend();
            using StableAudioDit dit = new StableAudioDit(cfg);
            dit.LoadWeights(w);
            StableAudioNumberEmbedder total = new(0f, (float)cfg.TimingMaxSeconds);
            total.LoadWeights(timing, "total");
            StableAudioNumberEmbedder start = new(0f, (float)cfg.TimingMaxSeconds);
            start.LoadWeights(timing, "start");
            using StableAudioPipeline pipeline = new(backend, dit, new StandInVae(cfg.VaeDownsample), total, cfg, start);
            using Tensor text = Rand(new TensorShape(1, 4, cfg.CondTokenDim), 21, 1f);

            (float[] l1, float[] r1, int rate, int seed) = pipeline.GenerateVPrediction(text, 0.0, 1.0, steps: 4, cfgScale: 7f, seed: 5);
            (float[] l2, _, _, _) = pipeline.GenerateVPrediction(text, 0.0, 1.0, steps: 4, cfgScale: 7f, seed: 5);
            (float[] noCfg, _, _, _) = pipeline.GenerateVPrediction(text, 0.0, 1.0, steps: 4, cfgScale: 1f, seed: 5);

            Assert.Equal((cfg.SampleRate, 5), (rate, seed));
            Assert.Equal(cfg.SampleRate, l1.Length); // 1 s at the stand-in rate.
            Assert.Equal(l1.Length, r1.Length);
            Assert.Equal(l1, l2);
            Assert.All(l1, x => Assert.True(float.IsFinite(x)));
            Assert.NotEqual(l1, noCfg);
        }
        finally
        {
            DisposeAll(w);
            DisposeAll(timing);
        }
    }

    [Fact]
    public void Pipeline_VPredictionNeedsTheStartConditioner()
    {
        StableAudioDitConfig cfg = Tiny;
        Dictionary<string, Tensor> w = DitWeights(cfg);
        Dictionary<string, Tensor> timing = TimingWeights(cfg);
        try
        {
            using CpuBackend backend = new CpuBackend();
            using StableAudioDit dit = new StableAudioDit(cfg);
            dit.LoadWeights(w);
            StableAudioNumberEmbedder total = new(0f, (float)cfg.TimingMaxSeconds);
            total.LoadWeights(timing, "total");
            using StableAudioPipeline pipeline = new(backend, dit, new StandInVae(cfg.VaeDownsample), total, cfg);
            using Tensor text = Rand(new TensorShape(1, 2, cfg.CondTokenDim), 31, 1f);

            Assert.Throws<InvalidOperationException>(() => pipeline.GenerateVPrediction(text, 0, 1, 2, 7f, 1));
        }
        finally
        {
            DisposeAll(w);
            DisposeAll(timing);
        }
    }

    /// <summary>Open 1.0's structure at a tiny width: unprojected 8-wide condition tokens read as one KV head of 8 by two
    /// query heads, no QK-norm, two timing tokens.</summary>
    private static StableAudioDitConfig Tiny => new()
    {
        HiddenSize = 16, NumLayers = 1, NumHeads = 2, CrossKvHeads = 1, HeadDim = 8, FfInner = 24, RopeDim = 4, QkNorm = false,
        IoChannels = 4, CondTokenDim = 8, GlobalCondDim = 16, TimestepFeaturesDim = 8, HasSecondsStart = true,
        MaxLatentTokens = 6, TimingMaxSeconds = 10, SampleRate = 16, VaeDownsample = 4,
    };

    private static Dictionary<string, TensorShape> Shapes(StableAudioDitConfig c, int condInput)
    {
        Dictionary<string, TensorShape> s = new(StringComparer.Ordinal)
        {
            ["transformer.project_in.weight"] = new TensorShape(c.HiddenSize, c.IoChannels),
            ["transformer.rotary_pos_emb.inv_freq"] = new TensorShape(c.RopeDim / 2),
            ["to_global_embed.0.weight"] = new TensorShape(c.HiddenSize, c.GlobalCondDim),
            ["transformer.layers.0.ff.ff.0.proj.weight"] = new TensorShape(2 * c.FfInner, c.HiddenSize),
            ["transformer.layers.0.cross_attn.to_kv.weight"] = new TensorShape(2 * c.CrossKvHeads * c.HeadDim, condInput),
        };
        for (int i = 0; i < c.NumLayers; i++)
            s[$"transformer.layers.{i}.self_attn.to_qkv.weight"] = new TensorShape(3 * c.HiddenSize, c.HiddenSize);
        if (c.QkNorm)
            s["transformer.layers.0.self_attn.q_norm.weight"] = new TensorShape(c.HeadDim);
        return s;
    }

    private static Dictionary<string, Tensor> DitWeights(StableAudioDitConfig c)
    {
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        uint seed = 100;
        void Mat(string k, params long[] dims) => w[k] = Rand(new TensorShape(dims), seed++, 0.2f);
        void Norm(string k, int n)
        {
            w[k + ".gamma"] = Fill(n, 1f);
            w[k + ".beta"] = Fill(n, 0f);
        }
        int dim = c.HiddenSize, io = c.IoChannels, kv = c.CrossKvHeads * c.HeadDim;
        Mat("preprocess_conv.weight", io, io, 1);
        Mat("postprocess_conv.weight", io, io, 1);
        Mat("timestep_features.weight", c.TimestepFeaturesDim / 2);
        Mat("to_timestep_embed.0.weight", dim, c.TimestepFeaturesDim);
        Mat("to_timestep_embed.2.weight", dim, dim);
        Mat("to_global_embed.0.weight", 12, c.GlobalCondDim);
        Mat("to_global_embed.2.weight", dim, 12);
        Mat("to_cond_embed.0.weight", c.CondTokenDim, c.CondTokenDim);
        Mat("to_cond_embed.2.weight", c.CondTokenDim, c.CondTokenDim);
        Mat("transformer.project_in.weight", dim, io);
        Mat("transformer.project_out.weight", io, dim);
        w["transformer.rotary_pos_emb.inv_freq"] = Fill(c.RopeDim / 2, 0.5f);
        string p = "transformer.layers.0";
        Norm($"{p}.pre_norm", dim);
        Norm($"{p}.cross_attend_norm", dim);
        Norm($"{p}.ff_norm", dim);
        Mat($"{p}.self_attn.to_qkv.weight", 3 * dim, dim);
        Mat($"{p}.self_attn.to_out.weight", dim, dim);
        Mat($"{p}.cross_attn.to_q.weight", dim, dim);
        Mat($"{p}.cross_attn.to_kv.weight", 2 * kv, c.CondTokenDim);
        Mat($"{p}.cross_attn.to_out.weight", dim, dim);
        Mat($"{p}.ff.ff.0.proj.weight", 2 * c.FfInner, dim);
        Mat($"{p}.ff.ff.2.weight", dim, c.FfInner);
        return w;
    }

    private static Dictionary<string, Tensor> TimingWeights(StableAudioDitConfig c)
    {
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        uint seed = 500;
        foreach (string name in new[] { "total", "start" })
        {
            w[$"{name}.embedder.embedding.0.weights"] = Rand(new TensorShape(4), seed++, 1f);
            w[$"{name}.embedder.embedding.1.weight"] = Rand(new TensorShape(c.CondTokenDim, 9), seed++, 0.3f);
        }
        return w;
    }

    /// <summary>Repeats each latent frame <c>hop</c> times per channel: enough to check lengths, trimming and determinism.</summary>
    private sealed class StandInVae(int hop) : IAudioLatentDecoder
    {
        public Tensor Decode(IBackend backend, Tensor latent)
        {
            int t = (int)latent.Shape[2];
            Tensor pcm = new Tensor(new TensorShape(1, 2, t * hop), DType.F32);
            ReadOnlySpan<float> z = latent.AsSpan<float>();
            Span<float> o = pcm.AsSpan<float>();
            for (int ch = 0; ch < 2; ch++)
                for (int i = 0; i < t * hop; i++)
                    o[ch * t * hop + i] = z[ch * t + i / hop];
            return pcm;
        }

        public IEnumerable<Tensor> EnumerateWeights() => [];
    }

    private static Tensor Rand(TensorShape shape, uint seed, float scale)
    {
        Tensor t = new Tensor(shape, DType.F32);
        Random r = new Random((int)seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++)
            s[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static Tensor Fill(int n, float value)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values)
            t.Dispose();
    }
}

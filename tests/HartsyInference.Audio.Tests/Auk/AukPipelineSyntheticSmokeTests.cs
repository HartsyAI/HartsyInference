using System.Text;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.QwenOmni;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>End-to-end AuK pipeline on tiny random weights (every stage, both models, with and without a reference): shapes, finiteness, seeding and cancellation. Says nothing about parity with the real checkpoints.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class AukPipelineSyntheticSmokeTests : IDisposable
{
    private const int Hidden = 16;
    private const int Layers = 2;
    private const int Hop = 6;
    private const int Rate = 600;

    private static readonly AukConfig DitCfg = AukDitReference.Tiny with { Hop = Hop, SampleRate = Rate, FusionLayers = Layers };
    private static readonly AukVaeConfig VaeCfg = AukVaeTestData.Tiny() with { LatentDim = DitCfg.LatentDim, SampleRate = Rate };
    private static readonly QwenOmniConfig OmniCfg = new()
    {
        NumMelBins = 4, DModel = 8, Layers = 2, Heads = 2, FfnDim = 16, NWindow = 4, OutputDim = Hidden, MaxSamples = 48_000,
    };

    private static readonly Qwen2Config LmCfg = new()
    {
        HiddenSize = Hidden, NumHiddenLayers = Layers, NumAttentionHeads = 2, NumKeyValueHeads = 1, IntermediateSize = 16,
        VocabSize = 151_936, MaxPositionEmbeddings = 512, TieWordEmbeddings = false,
    };

    private readonly List<Tensor> _owned = [];

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private Tensor Rand(Random rng, double amp, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * amp);
        return Own(t);
    }

    private Tensor Const(float value, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        t.AsSpan<float>().Fill(value);
        return Own(t);
    }

    private Dictionary<string, Tensor> AukWeights(bool flash)
    {
        Dictionary<string, Tensor> w = AukDitReference.ToTensors(AukDitReference.RandomWeights(DitCfg, flash ? 11 : 12));
        foreach (Tensor t in w.Values) Own(t);
        w[AukLayerFusion.LayerWeightsKey] = Rand(new Random(5), 1.0, Layers);
        w[AukLayerFusion.LayerScaleKey] = Const(1.5f, 1);
        return w;
    }

    private Dictionary<string, Tensor> VaeWeights()
    {
        Dictionary<string, Tensor> w = AukVaeTestData.BuildDecoder(VaeCfg, 21);
        foreach ((string k, Tensor t) in AukVaeTestData.BuildEncoder(VaeCfg, 22)) w[k] = t;
        foreach ((string k, Tensor t) in AukVaeTestData.BuildStats(VaeCfg, 23)) w[k] = t;
        foreach (Tensor t in w.Values) Own(t);
        return w;
    }

    private Dictionary<string, Tensor> OmniWeights()
    {
        Random rng = new(31);
        Dictionary<string, Tensor> w = [];
        foreach ((string key, long[] shape) in new QwenOmniAudioEncoder(OmniCfg).ExpectedWeights())
        {
            bool norm = key.EndsWith("norm.weight", StringComparison.Ordinal) || key == "ln_post.weight";
            Tensor t = Rand(rng, norm ? 0.1 : 0.4, shape);
            if (norm) foreach (ref float v in t.AsSpan<float>()) v += 1f;
            w[$"thinker.audio_tower.{key}"] = t;
        }
        w["thinker.model.embed_tokens.weight"] = Rand(rng, 0.5, LmCfg.VocabSize, Hidden);
        w["thinker.model.norm.weight"] = Const(1f, Hidden);
        long kv = LmCfg.KvHiddenSize;
        for (int l = 0; l < Layers; l++)
        {
            string p = $"thinker.model.layers.{l}";
            w[$"{p}.input_layernorm.weight"] = Const(1f, Hidden);
            w[$"{p}.post_attention_layernorm.weight"] = Const(1f, Hidden);
            w[$"{p}.self_attn.q_proj.weight"] = Rand(rng, 0.3, Hidden, Hidden);
            w[$"{p}.self_attn.q_proj.bias"] = Rand(rng, 0.1, Hidden);
            w[$"{p}.self_attn.k_proj.weight"] = Rand(rng, 0.3, kv, Hidden);
            w[$"{p}.self_attn.k_proj.bias"] = Rand(rng, 0.1, kv);
            w[$"{p}.self_attn.v_proj.weight"] = Rand(rng, 0.3, kv, Hidden);
            w[$"{p}.self_attn.v_proj.bias"] = Rand(rng, 0.1, kv);
            w[$"{p}.self_attn.o_proj.weight"] = Rand(rng, 0.3, Hidden, Hidden);
            w[$"{p}.mlp.gate_proj.weight"] = Rand(rng, 0.3, LmCfg.IntermediateSize, Hidden);
            w[$"{p}.mlp.up_proj.weight"] = Rand(rng, 0.3, LmCfg.IntermediateSize, Hidden);
            w[$"{p}.mlp.down_proj.weight"] = Rand(rng, 0.3, Hidden, LmCfg.IntermediateSize);
        }
        return w;
    }

    private AukPipeline Build(bool flash)
    {
        AukPipeline pipeline = new(flash ? "flash" : "base", flash, s => [.. Encoding.UTF8.GetBytes(s).Select(b => (int)b)],
            DitCfg, VaeCfg, LmCfg, OmniCfg);
        pipeline.LoadWeights(AukWeights(flash), VaeWeights(), OmniWeights());
        return pipeline;
    }

    private static float[] Tone(int samples, int rate)
    {
        float[] x = new float[samples];
        for (int i = 0; i < samples; i++) x[i] = 0.4f * MathF.Sin(2f * MathF.PI * 220f * i / rate);
        return x;
    }

    private static void AssertFinite(float[] pcm)
    {
        foreach (float v in pcm) Assert.True(float.IsFinite(v) && Math.Abs(v) <= 1f, "sample out of range");
    }

    [Fact]
    public void Flash_InstructTts_NoReference_ProducesFiniteAudioOfTheRequestedLength()
    {
        using AukPipeline pipeline = Build(flash: true);
        CpuBackend backend = new();
        float[] pcm = pipeline.Generate(backend, "Say hello", null, 0, 0.5, new AukOptions { Seed = 3 });
        Assert.Equal(50 * Hop, pcm.Length);
        AssertFinite(pcm);
        float[] again = pipeline.Generate(backend, "Say hello", null, 0, 0.5, new AukOptions { Seed = 3 });
        Assert.Equal(pcm, again);
        float[] other = pipeline.Generate(backend, "Say hello", null, 0, 0.5, new AukOptions { Seed = 4 });
        Assert.NotEqual(pcm, other);
    }

    [Fact]
    public void Base_WithReference_RunsGuidedSamplingAndMatchesTheClipLength()
    {
        using AukPipeline pipeline = Build(flash: false);
        CpuBackend backend = new();
        float[] source = Tone(8_000, 16_000);
        float[] pcm = pipeline.Generate(backend, "Remove the noise", source, 16_000, null,
            new AukOptions { Seed = 1, Steps = 3, CfgScale = 2f });
        Assert.Equal(50 * Hop, pcm.Length);
        AssertFinite(pcm);

        float[] unguided = pipeline.Generate(backend, "Remove the noise", source, 16_000, null,
            new AukOptions { Seed = 1, Steps = 3, CfgScale = 0f });
        Assert.NotEqual(pcm, unguided);
    }

    [Fact]
    public void Base_ZeroShot_ScalesDurationByTextLength_AndNonSequentialResidencyMatches()
    {
        using AukPipeline pipeline = Build(flash: false);
        CpuBackend backend = new();
        float[] source = Tone(Rate, Rate);
        AukOptions opts = new() { Seed = 2, Steps = 2, GenText = "twelve chars", RefText = "six ch" };
        float[] pcm = pipeline.Generate(backend, "Say the following with the same voice", source, Rate, null, opts);
        Assert.Equal(source.Length / Hop * Hop * 2, pcm.Length);
        float[] resident = pipeline.Generate(backend, "Say the following with the same voice", source, Rate, null,
            opts with { SequentialResidency = false });
        Assert.Equal(pcm, resident);
    }

    [Fact]
    public void Generate_RejectsMissingDurationOverBudgetAndCancellation()
    {
        using AukPipeline pipeline = Build(flash: true);
        CpuBackend backend = new();
        Assert.Throws<ArgumentException>(() => pipeline.Generate(backend, "x", null, 0, null));
        Assert.Throws<ArgumentException>(() => pipeline.Generate(backend, "x", null, 0, 31.0));
        Assert.Throws<ArgumentException>(() => pipeline.Generate(backend, "x", new float[16_000 * 31], 16_000, null));
        using CancellationTokenSource cts = new();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => pipeline.Generate(backend, "x", null, 0, 0.5, null, cts.Token));
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
    }
}

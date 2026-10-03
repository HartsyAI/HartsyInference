using System.Text;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.QwenOmni;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Exceptions;
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

    [Theory]
    [InlineData(0, 0, 100, false)] // a backend that can't report VRAM never claims it fits
    [InlineData(1_000, 0, 100, false)]
    [InlineData(99, 1_000_000, 100, false)] // free is below required + a third margin
    [InlineData(132, 1_000_000, 100, false)] // just under the margin (100 + 100/3 = 133, integer division)
    [InlineData(133, 1_000_000, 100, true)] // exactly at it
    [InlineData(1_000_000, 1_000_000, 100, true)]
    public void ResidentWithinBudget_ComparesFreeAgainstRequiredPlusAThird(long free, long total, long required, bool expected)
        => Assert.Equal(expected, AukPipeline.ResidentWithinBudget(free, total, required));

    /// <summary>Simulates AukPipeline.FitsResident's effectiveFree = freeBytes + residentBytes pattern across a
    /// call sequence, standing in for the live device (no backend needed since the formula itself is pure). Two
    /// properties a naive "re-read GetVramInfo every call" check gets wrong: going resident must not immediately
    /// look like it no longer fits just because the preload it caused shows up as less free device memory (that
    /// would evict every other call, exactly what going resident is meant to avoid), and a real drop in free VRAM
    /// from something else entirely must still be able to flip it back.</summary>
    [Fact]
    public void FitsResidentFormula_StaysStableOnOwnFootprint_ButReactsToExternalPressure()
    {
        const long total = 10_000;
        const long required = 900; // margin threshold: 900 + 900/3 = 1200
        long deviceFree = 2_000;
        long residentBytes = 0;

        bool fits1 = AukPipeline.ResidentWithinBudget(deviceFree + residentBytes, total, required);
        Assert.True(fits1);
        residentBytes = required; // the pipeline preloads and keeps `required` bytes resident
        deviceFree -= required; // which the device now reports as used

        // Call 2: a plain re-read of deviceFree (1100) would be below the 1200 margin and flip to sequential.
        // effectiveFree adds the pipeline's own footprint back, so it reads as unchanged from call 1.
        bool fits2 = AukPipeline.ResidentWithinBudget(deviceFree + residentBytes, total, required);
        Assert.True(fits2);
        residentBytes = required;

        // Something unrelated now consumes real device VRAM (another engine's pipeline, not this one).
        deviceFree -= 1_000;
        bool fits3 = AukPipeline.ResidentWithinBudget(deviceFree + residentBytes, total, required);
        Assert.False(fits3); // must still catch genuine external pressure, not stay stuck on the old "true"
    }

    [Theory]
    [InlineData(500, 900, true)] // device shows less in use than we think we alone hold: at least ours was freed
    [InlineData(900, 900, false)] // exactly what we think we hold -- nothing missing
    [InlineData(1_500, 900, false)] // more in use than just ours -- plausible without anything of ours freed
    public void WasSweptExternally_DetectsDeviceUsageBelowOwnFootprint(long usedBytes, long residentBytes, bool expected)
        => Assert.Equal(expected, AukPipeline.WasSweptExternally(usedBytes, residentBytes));

    [Fact]
    public void FitsResidentFormula_RecoversAfterAnExternalSweepFreesOwnFootprint()
    {
        const long total = 10_000;
        const long required = 900;
        long deviceFree = 2_000;
        long residentBytes = 0;

        bool fits1 = AukPipeline.ResidentWithinBudget(deviceFree + residentBytes, total, required);
        Assert.True(fits1);
        residentBytes = required;
        deviceFree -= required; // deviceFree=1100, usedBytes=8900

        // AudioRuntime.UnloadOthers evicts a sibling model and sweeps the WHOLE device clean -- including this
        // pipeline's own resident weights -- without ever throwing through this pipeline's own call.
        deviceFree = total; // the device now reports fully free; usedBytes drops to 0
        Assert.True(AukPipeline.WasSweptExternally(total - deviceFree, residentBytes)); // 0 < 900: detected
        residentBytes = 0; // what FitsResident does on detecting it

        bool fits2 = AukPipeline.ResidentWithinBudget(deviceFree + residentBytes, total, required);
        Assert.True(fits2); // re-measured from a clean baseline, not inflated by the no-longer-real old footprint
    }

    [Fact]
    public void WeightBytes_SumsElementCountTimesDtypeSize()
    {
        Tensor a = Own(new Tensor(new TensorShape(2, 3), DType.F32)); // 6 * 4 bytes
        Tensor b = Own(new Tensor(new TensorShape(4), DType.F32)); // 4 * 4 bytes
        Assert.Equal(40, AukPipeline.WeightBytes([a, b]));
    }

    [Fact]
    public void IsOutOfVram_TrueForTheExceptionItselfOrAnywhereInItsInnerChain()
    {
        Assert.True(AukPipeline.IsOutOfVram(new OutOfVramException(100, 10)));
        Assert.True(AukPipeline.IsOutOfVram(new InvalidOperationException("wrapped", new OutOfVramException(100, 10))));
        Assert.True(AukPipeline.IsOutOfVram(new InvalidOperationException("double-wrapped",
            new InvalidOperationException("inner", new OutOfVramException(100, 10)))));
    }

    [Fact]
    public void IsOutOfVram_FalseWhenNoOutOfVramExceptionAppearsInTheChain()
    {
        Assert.False(AukPipeline.IsOutOfVram(new InvalidOperationException("unrelated")));
        Assert.False(AukPipeline.IsOutOfVram(new InvalidOperationException("wrapped", new ArgumentException("inner"))));
        Assert.False(AukPipeline.IsOutOfVram(new OperationCanceledException()));
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

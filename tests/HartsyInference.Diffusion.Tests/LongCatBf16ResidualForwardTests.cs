using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Shared;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The residual rounding on a forward: a block on BF16 weights, or on INT8 weights widened to F32, rounds its residual sums to the BF16 grid
/// (every output value is a BF16 value) when the knob is on; an F32 block, or the knob off, leaves them off the grid. The cached build path rounds too.</summary>
[Collection(GlobalKnobStore.Name)]
public sealed class LongCatBf16ResidualForwardTests
{
    private const int Frames = 5, GridH = 2, GridW = 3, TextRows = 4;

    [Fact]
    public void AnF32BlockLeavesItsResidualsOffTheBf16Grid()
    {
        float[] values = ForwardBlock(bf16Weights: false, int8Release: false, knob: true, cached: false);
        Assert.Contains(values, v => !OnBf16Grid(v));
    }

    [Fact]
    public void ABf16WeightBlockRoundsItsResidualsOnTheBf16Grid()
    {
        float[] values = ForwardBlock(bf16Weights: true, int8Release: false, knob: true, cached: false);
        Assert.All(values, v => Assert.True(OnBf16Grid(v), $"{v:R} is not a BF16 value"));
    }

    [Fact]
    public void AnInt8ReleaseRoundsItsResidualsEvenWhenItsWeightsWidenToF32()
    {
        // The widened weights are F32 here, so only the release flag (not the weight dtype) can turn the rounding on.
        float[] values = ForwardBlock(bf16Weights: false, int8Release: true, knob: true, cached: false);
        Assert.All(values, v => Assert.True(OnBf16Grid(v), $"{v:R} is not a BF16 value"));
    }

    [Fact]
    public void TheKnobOffKeepsABf16WeightBlockOffTheGrid()
    {
        float[] values = ForwardBlock(bf16Weights: true, int8Release: false, knob: false, cached: false);
        Assert.Contains(values, v => !OnBf16Grid(v));
    }

    [Fact]
    public void TheCachedBuildPathRoundsItsResidualsToo()
    {
        float[] on = ForwardBlock(bf16Weights: true, int8Release: false, knob: true, cached: true);
        Assert.All(on, v => Assert.True(OnBf16Grid(v), $"{v:R} is not a BF16 value"));
        float[] off = ForwardBlock(bf16Weights: false, int8Release: false, knob: true, cached: true);
        Assert.Contains(off, v => !OnBf16Grid(v));
    }

    /// <summary>Every probe stage of a BF16 avatar block with audio is on the BF16 grid: the sub-layer outputs and the residual sums.</summary>
    [Fact]
    public void TheAudioBlockProbeStagesAreOnTheBf16GridWithTheKnobOn()
    {
        Dictionary<string, float[]> stages = AudioBlockStages(knob: true);
        foreach (string name in AudioStageNames)
        {
            Assert.True(stages.ContainsKey(name), $"no probe stage {name}");
            Assert.All(stages[name], v => Assert.True(OnBf16Grid(v), $"{name}: {v:R} is not a BF16 value"));
        }
    }

    /// <summary>With the knob off, the same audio block keeps its sub-layer outputs and residuals off the BF16 grid.</summary>
    [Fact]
    public void TheAudioBlockProbeStagesLeaveTheGridWithTheKnobOff()
    {
        Dictionary<string, float[]> stages = AudioBlockStages(knob: false);
        foreach (string name in AudioStageNames)
        {
            Assert.True(stages.ContainsKey(name), $"no probe stage {name}");
            Assert.Contains(stages[name], v => !OnBf16Grid(v));
        }
    }

    private static readonly string[] AudioStageNames =
    [
        "block0.attn_in", "block0.self_attn", "block0.after_self", "block0.cross_norm", "block0.cross_attn", "block0.after_cross",
        "block0.audio_cross_attn_norm", "block0.audio_cross_attn_modulated", "block0.audio_cross_attn", "block0.after_audio",
        "block0.ffn_in", "block0.ffn", "block0.out",
    ];

    /// <summary>Runs the avatar block (BF16 qkv, the audio step) with a probe, the knob on or off, and returns each stage.</summary>
    private static Dictionary<string, float[]> AudioBlockStages(bool knob)
    {
        LongCatConfig config = LongCatTestData.TinyAvatar;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, 8);
        Tensor qkv = weights["blocks.0.attn.qkv.weight"];
        weights["blocks.0.attn.qkv.weight"] = qkv.CastTo(DType.BF16);
        qkv.Dispose();
        List<Tensor> owned = [];
        LongCatBlock block = new(config);
        using (KnobProfile.Create("bf16-audio-probe").With(EngineKnobs.LongcatBf16Residual, knob).Push())
        {
            block.LoadWeights(weights, "blocks.0", owned);
        }
        Assert.Equal(knob, block.Bf16Residual);

        const int frames = 5, gridH = 2, gridW = 3, textRows = 4;
        int hw = gridH * gridW, tokens = frames * hw;
        using Tensor hidden = LongCatTestData.Random(new TensorShape(tokens, config.HiddenSize), 31);
        using Tensor text = LongCatTestData.Random(new TensorShape(textRows, config.HiddenSize), 32);
        using Tensor timeEmbed = LongCatTestData.Random(new TensorShape(frames, config.AdalnTembedDim), 33);
        using Tensor audio = LongCatTestData.Random(new TensorShape(frames * config.AudioContextTokens, config.AudioOutputDim), 34);
        WanRope rope = new(config.HeadDim, LongCatConfig.RopeTheta, 0);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(frames, gridH, gridW);
        using Tensor index = new(new TensorShape(tokens), DType.I32);
        Span<int> rowFrame = index.AsSpan<int>();
        for (int i = 0; i < tokens; i++)
            rowFrame[i] = i / hw;
        LongCatBlockInputs inputs = new(frames, hw, 0, cos, sin, [AttentionSegment.Dense(tokens)], index);

        Dictionary<string, float[]> stages = new(StringComparer.Ordinal);
        using Tensor output = block.Forward(backend, hidden, text, timeEmbed, in inputs,
            (name, t) => stages[name] = t.AsReadOnlySpan<float>().ToArray(), "block0", audio);

        cos.Dispose();
        sin.Dispose();
        foreach (Tensor t in owned) t.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
        return stages;
    }

    /// <summary>The transformer's patch embedding and caption projection are on the BF16 grid on the BF16 route. The final linear and the velocity stay
    /// F32: autocast('cuda', float32) runs the final linear in F32 (checked on CUDA), so they are off the grid even with the knob on.</summary>
    [Fact]
    public void TheTransformerEmbeddingAndCaptionAreOnTheBf16GridAndTheFinalLayerIsF32()
    {
        Dictionary<string, float[]> stages = TransformerStages(knob: true);
        foreach (string name in new[] { "x_embedder", "y_embedder" })
        {
            Assert.True(stages.ContainsKey(name), $"no probe {name}");
            Assert.All(stages[name], v => Assert.True(OnBf16Grid(v), $"{name}: {v:R} is not a BF16 value"));
        }
        Assert.Contains(stages["final_layer"], v => !OnBf16Grid(v));
        Assert.Contains(stages["velocity"], v => !OnBf16Grid(v));
    }

    [Fact]
    public void TheTransformerPatchEmbeddingLeavesTheGridWithTheKnobOff()
    {
        Dictionary<string, float[]> stages = TransformerStages(knob: false);
        Assert.Contains(stages["x_embedder"], v => !OnBf16Grid(v));
        Assert.Contains(stages["velocity"], v => !OnBf16Grid(v));
    }

    /// <summary>The audio projection's output and inputs are on the BF16 grid on the BF16 route; with the route off they are not.</summary>
    [Fact]
    public void TheAudioProjectionIsOnTheBf16GridOnTheBf16Route()
    {
        Assert.All(AudioProjectionOutput(route: true), v => Assert.True(OnBf16Grid(v), $"audio projection: {v:R} is not a BF16 value"));
        Assert.Contains(AudioProjectionOutput(route: false), v => !OnBf16Grid(v));
    }

    private static float[] AudioProjectionOutput(bool route)
    {
        LongCatConfig config = LongCatTestData.TinyAvatar;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, 12);
        using LongCatAudioProjection projection = new(config.AudioWindow, config.AudioWindow + config.AudioVaeScale - 1, config.AudioBlock,
            config.AudioChannel, config.AudioIntermediateDim, config.AudioOutputDim, config.AudioContextTokens);
        projection.LoadWeights(weights, "audio_proj.");
        projection.Bf16Route = route;
        int inFirst = config.AudioWindow * config.AudioBlock * config.AudioChannel;
        int inLatter = (config.AudioWindow + config.AudioVaeScale - 1) * config.AudioBlock * config.AudioChannel;
        using Tensor first = LongCatTestData.Random(new TensorShape(1, inFirst), 13);
        using Tensor latter = LongCatTestData.Random(new TensorShape(2, inLatter), 14);
        using Tensor output = projection.Forward(backend, first, latter);
        float[] values = output.AsReadOnlySpan<float>().ToArray();
        foreach (Tensor t in weights.Values) t.Dispose();
        return values;
    }

    /// <summary>The probes of one tiny-transformer forward on the BF16 route (knob on) or the F32 one, by stage name.</summary>
    private static Dictionary<string, float[]> TransformerStages(bool knob)
    {
        LongCatConfig config = LongCatTestData.Tiny;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 9);
        Tensor proj = weights["x_embedder.proj.weight"];
        weights["x_embedder.proj.weight"] = proj.CastTo(DType.BF16);
        proj.Dispose();
        Dictionary<string, float[]> stages = new(StringComparer.Ordinal);
        using (KnobProfile.Create("bf16-transformer-probe").With(EngineKnobs.LongcatBf16Residual, knob).Push())
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            transformer.Probe = (name, t) => stages[name] = t.AsReadOnlySpan<float>().ToArray();
            using Tensor latent = LongCatTestData.Random(new TensorShape(new long[] { 1, config.InChannels, 2, 4, 6 }), 10);
            using Tensor encoder = LongCatTestData.Encoder(config, 4, 11);
            using Tensor velocity = transformer.Forward(backend, latent, encoder, [600f], LongCatConditioning.None);
        }
        foreach (Tensor t in weights.Values) t.Dispose();
        return stages;
    }

    /// <summary>The cached denoise pass runs two steps with the knob on and off. The caption projection stays owned by its cache, so the second step
    /// reads it again (it is not disposed by the first), and the pass finishes with finite velocities of the noisy latent's shape.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheCachedPassRunsTwoStepsWithTheCaptionProjectionOwnedByItsCache(bool knob)
    {
        LongCatConfig config = LongCatTestData.Tiny;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 15);
        Tensor proj = weights["x_embedder.proj.weight"];
        weights["x_embedder.proj.weight"] = proj.CastTo(DType.BF16);
        proj.Dispose();
        using LongCatTransformer transformer = new(config);
        using (KnobProfile.Create("bf16-cached-two-steps").With(EngineKnobs.LongcatBf16Residual, knob).Push())
        {
            transformer.LoadWeights(weights);
        }

        LongCatConditioning conditioning = new(NumCondLatents: 1);
        using Tensor condLatent = LongCatTestData.Random(new TensorShape(new long[] { 1, config.InChannels, 1, 4, 6 }), 16);
        using Tensor noisy = LongCatTestData.Random(new TensorShape(new long[] { 1, config.InChannels, 2, 4, 6 }), 18);
        using Tensor encoder = LongCatTestData.Encoder(config, 4, 17);
        using LongCatKvCache cache = transformer.BuildKvCache(backend, condLatent, conditioning);

        float[] first, second;
        using (Tensor v1 = transformer.ForwardCached(backend, noisy, encoder, [600f], in conditioning, cache))
            first = v1.AsReadOnlySpan<float>().ToArray();
        using (Tensor v2 = transformer.ForwardCached(backend, noisy, encoder, [400f], in conditioning, cache))
            second = v2.AsReadOnlySpan<float>().ToArray();

        Assert.All(first, v => Assert.True(float.IsFinite(v)));
        Assert.All(second, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(first.Length, second.Length);
        Assert.Equal(1, transformer.TextProjectionBuilds);
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    /// <summary>The uncached pass reuses the caption projection across calls with the same encoder: the key is the caller's encoder, so the second call
    /// hits. With the knob on, the cached projection is the rounded one.</summary>
    [Fact]
    public void TheUncachedPassBuildsTheCaptionProjectionOnce()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 19);
        Tensor proj = weights["x_embedder.proj.weight"];
        weights["x_embedder.proj.weight"] = proj.CastTo(DType.BF16);
        proj.Dispose();
        using LongCatTransformer transformer = new(config);
        using (KnobProfile.Create("bf16-uncached-hit").With(EngineKnobs.LongcatBf16Residual, true).Push())
        {
            transformer.LoadWeights(weights);
        }
        using Tensor latent = LongCatTestData.Random(new TensorShape(new long[] { 1, config.InChannels, 2, 4, 6 }), 20);
        using Tensor encoder = LongCatTestData.Encoder(config, 4, 21);
        using (Tensor v1 = transformer.Forward(backend, latent, encoder, [600f], LongCatConditioning.None)) { }
        using (Tensor v2 = transformer.Forward(backend, latent, encoder, [400f], LongCatConditioning.None)) { }
        Assert.Equal(1, transformer.TextProjectionBuilds);
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    private static bool OnBf16Grid(float v) => BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(v)) == v;

    /// <summary>One block forward on the tiny config, with the given weight dtype, release flag and knob. The cached path runs the build pass into a
    /// fresh store.</summary>
    private static float[] ForwardBlock(bool bf16Weights, bool int8Release, bool knob, bool cached)
    {
        LongCatConfig config = LongCatTestData.Tiny;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 6);
        if (bf16Weights)
        {
            Tensor qkv = weights["blocks.0.attn.qkv.weight"];
            weights["blocks.0.attn.qkv.weight"] = qkv.CastTo(DType.BF16);
            qkv.Dispose();
        }
        List<Tensor> owned = [];
        LongCatBlock block = new(config);
        using (KnobProfile.Create("bf16-forward").With(EngineKnobs.LongcatBf16Residual, knob).Push())
        {
            block.LoadWeights(weights, "blocks.0", owned, int8Release);
        }

        int hw = GridH * GridW, tokens = Frames * hw;
        using Tensor hidden = LongCatTestData.Random(new TensorShape(tokens, config.HiddenSize), 21);
        using Tensor text = LongCatTestData.Random(new TensorShape(TextRows, config.HiddenSize), 22);
        using Tensor timeEmbed = LongCatTestData.Random(new TensorShape(Frames, config.AdalnTembedDim), 23);
        WanRope rope = new(config.HeadDim, LongCatConfig.RopeTheta, 0);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(Frames, GridH, GridW);
        using Tensor index = new(new TensorShape(tokens), DType.I32);
        Span<int> rowFrame = index.AsSpan<int>();
        for (int i = 0; i < tokens; i++)
            rowFrame[i] = i / hw;
        AttentionSegment[] dense = [AttentionSegment.Dense(tokens)];
        LongCatBlockInputs inputs = new(Frames, hw, 0, cos, sin, dense, index);

        float[] values;
        if (cached)
        {
            using KvSegmentStore store = new(backend, layers: 1, tokenAxis: 0);
            using Tensor output = block.ForwardCacheBuild(backend, hidden, timeEmbed, in inputs, store, 0);
            values = output.AsReadOnlySpan<float>().ToArray();
        }
        else
        {
            using Tensor output = block.Forward(backend, hidden, text, timeEmbed, in inputs);
            values = output.AsReadOnlySpan<float>().ToArray();
        }

        cos.Dispose();
        sin.Dispose();
        foreach (Tensor t in owned) t.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
        return values;
    }
}

/// <summary>The Vulkan F32 widening of a BF16 tensor: the bits of each value widen by its top 16, which the F16 path used to misread.</summary>
public sealed class VulkanBf16WidenTests
{
    [VulkanBf16Fact]
    public void VulkanWidensBf16ByItsBits()
    {
        float[] values = [1.0f, 1.00390625f, -2.71828f, 3.14159f, 1e-3f, 65504f];
        using Tensor bf16 = new(new TensorShape(values.Length), DType.BF16);
        ushort[] bits = values.Select(BFloat16Bits.FromFloat).ToArray();
        bits.AsSpan().CopyTo(bf16.AsSpan<ushort>());
        using Tensor f32 = new(new TensorShape(values.Length), DType.F32);
        using VulkanBackend vulkan = GpuDevices.OpenVulkan();

        vulkan.CastToF32(f32, bf16);

        ReadOnlySpan<float> got = f32.AsReadOnlySpan<float>();
        for (int i = 0; i < values.Length; i++)
            Assert.Equal(BFloat16Bits.ToFloat(bits[i]), got[i]);
    }
}

/// <summary>A fact reported as skipped, never passed, when no Vulkan device opens.</summary>
public sealed class VulkanBf16FactAttribute : FactAttribute
{
    public VulkanBf16FactAttribute()
    {
        if (!VulkanContext.IsAvailable()) Skip = "SKIPPED: no Vulkan device.";
    }
}

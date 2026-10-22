using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

/// <summary>The op sequence and attention call of Wave E sites on the recording stub (no compute): the Flash-entry sites the shared layer cannot
/// reach today, pinned so a migration shows any change in order, positions, window or scale.</summary>
public sealed class AttentionWaveETraceTests
{
    private static string[] Names(AttentionTraceBackend b) => [.. b.Ops.Select(o => o.Split(' ')[0])];

    [Fact]
    public void DiaFlashPath_ProjectsPermutesAppendsFlashesAndProjects()
    {
        Weights w = new(1);
        w.Add("a.q_proj.weight", 32, 16); w.Add("a.k_proj.weight", 16, 16); w.Add("a.v_proj.weight", 16, 16); w.Add("a.o_proj.weight", 16, 32);
        DiaAttention attn = new(4, 2, 8, 16, 16, 16, useRope: false);
        attn.LoadWeights(w, "a");
        using FixedKvCache cache = new(1, 1, 2, 8, 16);
        AttentionTraceBackend trace = new();
        using Tensor x = new(new TensorShape(1, 1, 16), DType.F32);
        using Tensor y = attn.SelfForwardFlash(trace, x, 1, 0, cache, 0, x, x);
        Assert.Equal(["Linear", "Linear", "Linear", "Permute0213", "Permute0213", "Permute0213", "KvCacheAppend", "KvCacheAppend",
            "FlashAttention", "Permute0213", "Linear"], Names(trace));
        FlashCall call = Assert.Single(trace.FlashCalls);
        Assert.Equal(new FlashCall(KvLen: 1, KvGroup: 2, Causal: true, QOffset: 0, Scale: 1f, Softcap: 0f, HasSinks: false, SlidingWindow: 0, DevicePos: 0), call);
    }

    [Fact]
    public void MusicGenStep_AppendsOnDeviceThenFlashesSelfWithPositionsAndCrossNonCausal()
    {
        const int h = 16;
        MusicGenConfig cfg = new() { Hidden = h, NumLayers = 1, NumHeads = 2 };
        Weights w = new(2);
        foreach (string n in new[] { "self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm" }) { w.AddNorm($"m.{n}.weight", h); w.AddBias($"m.{n}.bias", h); }
        foreach (string a in new[] { "self_attn", "encoder_attn" })
            foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Add($"m.{a}.{n}.weight", h, h);
        w.Add("m.fc1.weight", 4 * h, h); w.Add("m.fc2.weight", h, 4 * h);
        using MusicGenBlock block = new(cfg);
        block.LoadWeights(w, "m");
        using MusicGenKvCache cache = new(1, 1, 2, 8, 8);
        AttentionTraceBackend trace = new();
        block.PrimeCross(trace, new Tensor(new TensorShape(1, 3, h), DType.F32), cache.Layers[0]);
        using Tensor x = new(new TensorShape(1, 1, h), DType.F32);
        try { block.ForwardStep(trace, x, cache.Layers[0], 2, 3, devicePos: 7); } catch (NotSupportedException) { /* the closing residual/MLP ops outside the stub */ }
        Assert.Collection(trace.FlashCalls,
            self => Assert.Equal(new FlashCall(KvLen: 3, KvGroup: 1, Causal: true, QOffset: 2, Scale: 1f / MathF.Sqrt(8), Softcap: 0f, HasSinks: false, SlidingWindow: 0, DevicePos: 7), self),
            cross => Assert.Equal(new FlashCall(KvLen: 3, KvGroup: 1, Causal: false, QOffset: 0, Scale: 1f / MathF.Sqrt(8), Softcap: 0f, HasSinks: false, SlidingWindow: 0, DevicePos: 0), cross));
        Assert.Equal(2, trace.Ops.Count(o => o.StartsWith("KvCacheAppendDev", StringComparison.Ordinal)));
    }
}

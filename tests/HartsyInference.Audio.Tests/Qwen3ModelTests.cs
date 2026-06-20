using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Streaming;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the reusable Qwen3 backbone (Qwen3-TTS talker / code-predictor; future Qwen3 models):
/// config (decoupled head_dim) + a synthetic-weights forward exercising per-head q/k-norm + GQA + RoPE + SwiGLU.</summary>
public sealed unsafe class Qwen3ModelTests
{
    private static uint _rng = 0x39E3u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void Talker1_7B_DecouplesHeadDim()
    {
        Qwen3Config c = Qwen3Config.Talker1_7B;
        Assert.Equal(2_048, c.HiddenSize);
        Assert.Equal(28, c.NumHiddenLayers);
        Assert.Equal(16, c.NumAttentionHeads);
        Assert.Equal(8, c.NumKeyValueHeads);
        Assert.Equal(128, c.HeadDim);
        Assert.Equal(2_048, c.QDim);          // 16×128 (== hidden here, but decoupled in general)
        Assert.Equal(1_024, c.KvDim);         // 8×128
        Assert.Equal(2, c.KvGroup);
    }

    [Fact]
    public void SyntheticForward_QkNorm_GQA_IsFinite()
    {
        // Decoupled head_dim: hidden 16 but q_proj outputs 4×8=32.
        Qwen3Config c = new()
        {
            HiddenSize = 16, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 2,
            HeadDim = 8, IntermediateSize = 32, MaxPositionEmbeddings = 64,
        };
        using CpuBackend backend = new();
        using Qwen3Model m = new(c);
        m.LoadWeightsHeadless(Weights(c), "model");
        int t = 4;
        using StreamingKvCache cache = new(c.NumHiddenLayers, 1, c.NumKeyValueHeads, 16, c.HeadDim);
        using Tensor embeds = F3(1, t, c.HiddenSize);
        using Tensor o = m.ForwardEmbeds(backend, embeds, t, 0, cache);
        Assert.Equal(new TensorShape(1, t, c.HiddenSize), o.Shape);
        float* p = (float*)o.DataPointer;
        for (long i = 0; i < o.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> Weights(Qwen3Config c)
    {
        int h = c.HiddenSize, hd = c.HeadDim;
        Dictionary<string, Tensor> w = new() { ["model.norm.weight"] = Ones(h) };
        for (int i = 0; i < c.NumHiddenLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(c.QDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(c.KvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(c.KvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, c.QDim);
            w[$"{p}.self_attn.q_norm.weight"] = Ones(hd);
            w[$"{p}.self_attn.k_norm.weight"] = Ones(hd);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }
}

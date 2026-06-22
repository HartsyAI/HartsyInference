using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Streaming;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Parity: the config-driven <see cref="GenericTransformer"/> must reproduce the per-family reference
/// decoders on the CPU backend. Qwen2 exercises QKV bias + coupled head_dim; Qwen3 exercises per-head Q/K norm
/// + decoupled head_dim + bias-free attention. Comparison is on the <c>ForwardEmbeds</c> hidden state across
/// prefill + two incremental decode steps with shared synthetic weights and inputs.</summary>
public sealed unsafe class GenericTransformerParityTests
{
    private static uint _rng = 0x12345u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Embeds(int t, int h) => Fill(new Tensor(new TensorShape(1, t, h), DType.F32));

    [Fact]
    public void Qwen2_MatchesReferenceDecoder()
    {
        Qwen2Config refCfg = new()
        {
            HiddenSize = 16, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 2,
            IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, TieWordEmbeddings = true, AttentionBias = true,
        };
        TransformerConfig genCfg = new()
        {
            HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
            IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = true, QkNorm = false,
        };
        Dictionary<string, Tensor> w = Qwen2Weights(genCfg);

        using CpuBackend backend = new();
        using Qwen2Model reference = new(refCfg);
        using GenericTransformer generic = new(genCfg);
        reference.LoadWeights(w, "model");
        generic.LoadWeights(w, "model");

        using StreamingKvCache refCache = new(refCfg.NumHiddenLayers, 1, refCfg.NumKeyValueHeads, 32, refCfg.HeadDim);
        using KvCache genCache = new(genCfg.NumLayers, 1, genCfg.NumKvHeads, genCfg.HeadDim);

        RunParity(genCfg.HiddenSize,
            (embeds, t, pos) => reference.ForwardEmbeds(backend, embeds, 1, t, pos, refCache),
            (embeds, t, pos) => generic.ForwardEmbeds(backend, embeds, t, pos, genCache),
            () => refCache.CurrentLength, () => genCache.CurrentLength);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void Qwen3_StructuralForward_IsFinite()
    {
        // Exercises the three Qwen3 axes: per-head q/k RMSNorm, decoupled head_dim (4*8=32 != hidden 16),
        // and bias-free attention. The repo's Qwen3Model is the *TTS* backbone (interleaved RoPE), so it is
        // NOT a valid parity oracle for HF Qwen3 *text* (split-half RoPE) — text correctness is proved by the
        // end-to-end Qwen3-0.6B coherence run. Here we assert the config-driven path is structurally sound.
        TransformerConfig cfg = new()
        {
            HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 8,
            IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = false, QkNorm = true,
        };
        Dictionary<string, Tensor> w = Qwen3Weights(cfg);

        using CpuBackend backend = new();
        using GenericTransformer generic = new(cfg);
        generic.LoadWeights(w, "model");
        using KvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);

        // Prefill 4 + two decode steps; every output must be finite and shaped correctly.
        using (Tensor e = Embeds(4, cfg.HiddenSize))
        using (Tensor o = generic.ForwardEmbeds(backend, e, 4, 0, cache))
        {
            Assert.Equal(new TensorShape(1, 4, cfg.HiddenSize), o.Shape);
            AssertFinite(o, "prefill");
        }
        for (int step = 0; step < 2; step++)
        {
            using Tensor e = Embeds(1, cfg.HiddenSize);
            using Tensor o = generic.ForwardEmbeds(backend, e, 1, cache.CurrentLength, cache);
            AssertFinite(o, $"decode-{step}");
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static void RunParity(int hidden,
        Func<Tensor, int, int, Tensor> reference, Func<Tensor, int, int, Tensor> generic,
        Func<int> refLen, Func<int> genLen)
    {
        // Prefill (t = 4) then two single-token decode steps, shared inputs.
        using (Tensor e = Embeds(4, hidden))
        using (Tensor r = reference(e, 4, 0))
        using (Tensor g = generic(e, 4, 0))
            AssertClose(r, g, "prefill", 2e-3f);
        Assert.Equal(refLen(), genLen());

        for (int step = 0; step < 2; step++)
        {
            using Tensor e = Embeds(1, hidden);
            using Tensor r = reference(e, 1, refLen());
            using Tensor g = generic(e, 1, genLen());
            AssertClose(r, g, $"decode-{step}", 2e-3f);
            Assert.Equal(refLen(), genLen());
        }
    }

    private static Dictionary<string, Tensor> Qwen2Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(c.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
            w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
            w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    private static Dictionary<string, Tensor> Qwen3Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim, hd = c.HeadDim;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(c.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.self_attn.q_norm.weight"] = Ones(hd);
            w[$"{p}.self_attn.k_norm.weight"] = Ones(hd);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    private static void AssertFinite(Tensor t, string label)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]), $"{label}: non-finite at {i}");
    }

    private static void AssertClose(Tensor a, Tensor b, string label, float tol)
    {
        Assert.Equal(a.Shape, b.Shape);
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        float maxDiff = 0f;
        for (long i = 0; i < a.ElementCount; i++)
        {
            Assert.True(float.IsFinite(pb[i]), $"{label}: non-finite at {i}");
            maxDiff = MathF.Max(maxDiff, MathF.Abs(pa[i] - pb[i]));
        }
        Assert.True(maxDiff <= tol, $"{label}: max abs diff {maxDiff} exceeds {tol}");
    }
}

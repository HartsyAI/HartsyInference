using HartsyInference.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Structural validation of the Multi-head Latent Attention path (DeepSeek-V2/V3, Kimi-K2) through
/// <see cref="GenericTransformer"/>. The real models exceed local VRAM, so this exercises the MLA forward on a
/// tiny synthetic 1-layer model: query projection, KV down→RMSNorm→up, the no-position/RoPE split, the shared
/// RoPE-key broadcast, value zero-padding, and the latent-KV cache through both prefill and decode. It asserts
/// the path runs and produces finite, correctly-shaped output (the crash/NaN/shape failure modes that matter for
/// a build-defer feature), and that decode is consistent with appending to the prefill.</summary>
public sealed unsafe class MlaTests
{
    private static uint _rng = 0x515Au;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    private static TransformerConfig Config()
    {
        MlaConfig mla = new() { KvLoraRank = 8, QLoraRank = 0, QkNopeHeadDim = 4, QkRopeHeadDim = 2, VHeadDim = 4 };
        return new TransformerConfig
        {
            HiddenSize = 16, NumLayers = 1, NumHeads = 4, NumKvHeads = 4, HeadDim = mla.QkHeadDim /* 6 */,
            IntermediateSize = 20, VocabSize = 32, MaxPositionEmbeddings = 64,
            AttentionBias = false, QkNorm = false, TieWordEmbeddings = true, Mla = mla,
        };
    }

    private static Dictionary<string, Tensor> Weights(TransformerConfig cfg)
    {
        MlaConfig m = cfg.Mla!;
        int h = cfg.HiddenSize, hq = cfg.NumHeads;
        const string p = "model.layers.0";
        return new Dictionary<string, Tensor>
        {
            ["model.embed_tokens.weight"] = F2(cfg.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
            [$"{p}.input_layernorm.weight"] = Ones(h),
            [$"{p}.post_attention_layernorm.weight"] = Ones(h),
            [$"{p}.self_attn.q_proj.weight"] = F2(hq * m.QkHeadDim, h),
            [$"{p}.self_attn.kv_a_proj.weight"] = F2(m.KvLoraRank + m.QkRopeHeadDim, h),
            [$"{p}.self_attn.kv_a_norm.weight"] = Ones(m.KvLoraRank),
            [$"{p}.self_attn.kv_b_proj.weight"] = F2(hq * (m.QkNopeHeadDim + m.VHeadDim), m.KvLoraRank),
            [$"{p}.self_attn.o_proj.weight"] = F2(h, hq * m.VHeadDim),
            [$"{p}.mlp.gate_proj.weight"] = F2(cfg.IntermediateSize, h),
            [$"{p}.mlp.up_proj.weight"] = F2(cfg.IntermediateSize, h),
            [$"{p}.mlp.down_proj.weight"] = F2(h, cfg.IntermediateSize),
        };
    }

    /// <summary>DeepSeek-V3 / Kimi-K2 q-LoRA config: the query is compressed (q_a_proj → q_a_norm → q_b_proj)
    /// instead of a direct q_proj.</summary>
    private static (TransformerConfig, Dictionary<string, Tensor>) QLoraModel(int qLora)
    {
        MlaConfig mla = new() { KvLoraRank = 8, QLoraRank = qLora, QkNopeHeadDim = 4, QkRopeHeadDim = 2, VHeadDim = 4 };
        TransformerConfig cfg = new()
        {
            HiddenSize = 16, NumLayers = 1, NumHeads = 4, NumKvHeads = 4, HeadDim = mla.QkHeadDim,
            IntermediateSize = 20, VocabSize = 32, MaxPositionEmbeddings = 64,
            AttentionBias = false, QkNorm = false, TieWordEmbeddings = true, Mla = mla,
        };
        int h = cfg.HiddenSize, hq = cfg.NumHeads;
        const string p = "model.layers.0";
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(cfg.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
            [$"{p}.input_layernorm.weight"] = Ones(h),
            [$"{p}.post_attention_layernorm.weight"] = Ones(h),
            [$"{p}.self_attn.q_a_proj.weight"] = F2(qLora, h),
            [$"{p}.self_attn.q_a_norm.weight"] = Ones(qLora),
            [$"{p}.self_attn.q_b_proj.weight"] = F2(hq * mla.QkHeadDim, qLora),
            [$"{p}.self_attn.kv_a_proj.weight"] = F2(mla.KvLoraRank + mla.QkRopeHeadDim, h),
            [$"{p}.self_attn.kv_a_norm.weight"] = Ones(mla.KvLoraRank),
            [$"{p}.self_attn.kv_b_proj.weight"] = F2(hq * (mla.QkNopeHeadDim + mla.VHeadDim), mla.KvLoraRank),
            [$"{p}.self_attn.o_proj.weight"] = F2(h, hq * mla.VHeadDim),
            [$"{p}.mlp.gate_proj.weight"] = F2(cfg.IntermediateSize, h),
            [$"{p}.mlp.up_proj.weight"] = F2(cfg.IntermediateSize, h),
            [$"{p}.mlp.down_proj.weight"] = F2(h, cfg.IntermediateSize),
        };
        return (cfg, w);
    }

    /// <summary>The q-LoRA query block <c>q = q_b_proj(q_a_norm(q_a_proj(x)))</c> matches an independent host
    /// reference (matmul → RMSNorm → matmul). This is the only new arithmetic the DeepSeek-V3 / Kimi-K2 MLA path
    /// adds over the validated V2-Lite direct-q path.</summary>
    [Fact]
    public void Mla_QLora_QueryBlock_MatchesReference()
    {
        const int h = 16, qLora = 12, qOut = 24, t = 3; const float eps = 1e-6f;
        using CpuBackend backend = new();
        IBackend b = backend;
        using Tensor preW = F2(qLora, h);     // q_a_proj [qLora, h]
        using Tensor norm = Fill(new Tensor(new TensorShape(qLora), DType.F32));   // q_a_norm [qLora] (non-trivial weights)
        using Tensor upW = F2(qOut, qLora);   // q_b_proj [qOut, qLora]
        using Tensor x = Fill(new Tensor(new TensorShape(1, t, h), DType.F32));

        // Engine path: Linear → RmsNorm → Linear (the F32 ops GenericTransformer.MlaForward's q-LoRA block runs).
        using Tensor qA = new(new TensorShape(1, t, qLora), DType.F32);
        b.Linear(qA, x, preW, null);
        using Tensor qAN = new(new TensorShape(1, t, qLora), DType.F32);
        b.RmsNorm(qAN, qA, norm, eps);
        using Tensor q = new(new TensorShape(1, t, qOut), DType.F32);
        b.Linear(q, qAN, upW, null);

        // Reference: x·q_a_projᵀ → rmsnorm·weight → ·q_b_projᵀ.
        float[] xh = HostArr(x), pw = HostArr(preW), nm = HostArr(norm), uw = HostArr(upW);
        float* qp = (float*)q.DataPointer;
        float maxDiff = 0f;
        for (int r = 0; r < t; r++)
        {
            float[] a = new float[qLora];
            for (int i = 0; i < qLora; i++) { float s = 0f; for (int j = 0; j < h; j++) s += xh[r * h + j] * pw[i * h + j]; a[i] = s; }
            float ms = 0f; for (int i = 0; i < qLora; i++) ms += a[i] * a[i]; ms /= qLora;
            float inv = 1f / MathF.Sqrt(ms + eps);
            for (int i = 0; i < qLora; i++) a[i] = a[i] * inv * nm[i];
            for (int o = 0; o < qOut; o++) { float s = 0f; for (int i = 0; i < qLora; i++) s += a[i] * uw[o * qLora + i]; maxDiff = MathF.Max(maxDiff, MathF.Abs(s - qp[r * qOut + o])); }
        }
        Assert.True(maxDiff <= 1e-4f, $"q-LoRA query block diverges from reference by {maxDiff:E3}");
    }

    /// <summary>The full q-LoRA MLA layer runs through prefill + decode, stays finite, and advances the cache —
    /// the build-defer structural gate for DeepSeek-V3 / Kimi-K2 (which exceed local VRAM for a real e2e run).</summary>
    [Fact]
    public void Mla_QLora_PrefillAndDecode_StayFinite()
    {
        (TransformerConfig cfg, Dictionary<string, Tensor> w) = QLoraModel(qLora: 12);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 32);
        using (Tensor prompt = Fill(new Tensor(new TensorShape(1, 4, cfg.HiddenSize), DType.F32)))
        using (Tensor _ = model.ForwardEmbeds(backend, prompt, 4, 0, cache)) { }
        using Tensor step = Fill(new Tensor(new TensorShape(1, 1, cfg.HiddenSize), DType.F32));
        using Tensor outp = model.ForwardEmbeds(backend, step, 1, cache.CurrentLength, cache);
        float* po = (float*)outp.DataPointer;
        for (int i = 0; i < cfg.HiddenSize; i++) Assert.True(float.IsFinite(po[i]), "q-LoRA MLA decode non-finite");
        Assert.Equal(5, cache.CurrentLength);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static float[] HostArr(Tensor t) { float[] r = new float[t.ElementCount]; float* p = (float*)t.DataPointer; for (long i = 0; i < r.Length; i++) r[i] = p[i]; return r; }

    [Fact]
    public void Mla_Prefill_ProducesFiniteOutput()
    {
        TransformerConfig cfg = Config();
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(Weights(cfg), "model");

        const int t = 5;
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 32);
        using Tensor embeds = Fill(new Tensor(new TensorShape(1, t, cfg.HiddenSize), DType.F32));
        using Tensor outp = model.ForwardEmbeds(backend, embeds, t, 0, cache);

        Assert.Equal(cfg.HiddenSize, (int)outp.Shape[2]);
        float* po = (float*)outp.DataPointer;
        for (long i = 0; i < outp.ElementCount; i++)
            Assert.True(float.IsFinite(po[i]), $"MLA prefill produced non-finite output at {i}");
        Assert.Equal(t, cache.CurrentLength);   // the latent-KV cache advanced once per token
    }

    [Fact]
    public void Mla_Decode_AppendsAndStaysFinite()
    {
        TransformerConfig cfg = Config();
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(Weights(cfg), "model");

        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 32);
        using (Tensor prompt = Fill(new Tensor(new TensorShape(1, 3, cfg.HiddenSize), DType.F32)))
        using (Tensor _ = model.ForwardEmbeds(backend, prompt, 3, 0, cache)) { }

        // Decode one token: attends the cached prefix; must run and stay finite.
        using Tensor step = Fill(new Tensor(new TensorShape(1, 1, cfg.HiddenSize), DType.F32));
        using Tensor outp = model.ForwardEmbeds(backend, step, 1, cache.CurrentLength, cache);
        float* po = (float*)outp.DataPointer;
        for (int i = 0; i < cfg.HiddenSize; i++)
            Assert.True(float.IsFinite(po[i]), "MLA decode produced non-finite output");
        Assert.Equal(4, cache.CurrentLength);
    }
}

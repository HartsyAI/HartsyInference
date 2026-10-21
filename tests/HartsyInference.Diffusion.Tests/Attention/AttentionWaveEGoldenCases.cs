using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Tests.Common.Attention;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>Forward passes of the Wave E Diffusion-package leftovers (T5 block, CLIP text and vision layers, ACE-Step 1.5 attention) on tiny
/// pseudo-random weights and inputs, each reduced to SHA-256 digests of its output bits. Everything runs on the CPU backend, so the same cases
/// produce the golden fixture from the pre-migration source and must reproduce it after the attention refactor.</summary>
internal static unsafe class AttentionWaveEGoldenCases
{
    private const float NegInf = -1e9f;

    private static readonly Dictionary<string, Func<CpuBackend, string>> Cases = Build();

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } = [.. Cases.Keys];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name)
    {
        using CpuBackend cpu = new();
        return Cases.TryGetValue(name, out Func<CpuBackend, string>? run)
            ? run(cpu)
            : throw new ArgumentException($"Unknown attention golden case '{name}'.", nameof(name));
    }

    private static Dictionary<string, Func<CpuBackend, string>> Build() => new(StringComparer.Ordinal)
    {
        ["t5block_b1"] = cpu => T5(cpu, batch: 1, padded: false, qTau: 1f, gated: true, scale: null),
        ["t5block_b2"] = cpu => T5(cpu, batch: 2, padded: false, qTau: 1f, gated: true, scale: null),
        ["t5block_padded"] = cpu => T5(cpu, batch: 2, padded: true, qTau: 1f, gated: true, scale: null),
        ["t5block_qtau"] = cpu => T5(cpu, batch: 1, padded: false, qTau: 0.01f, gated: true, scale: null),
        ["t5block_v10_relu_scale1"] = cpu => T5(cpu, batch: 1, padded: true, qTau: 1f, gated: false, scale: 1f),
        ["clip_text_layer"] = ClipTextLayer,
        ["clip_vision_layer"] = ClipVisionLayer,
        ["acestep15_attention_self_sliding"] = cpu => AceStep(cpu, mode: "sliding"),
        ["acestep15_attention_self_full"] = cpu => AceStep(cpu, mode: "full"),
        ["acestep15_attention_cross"] = cpu => AceStep(cpu, mode: "cross"),
        ["diamond_attn"] = DiamondAttn,
        ["qwenimage_vae_attn_b1"] = cpu => QwenImageAttn(cpu, 1),
        ["qwenimage_vae_attn_b2"] = cpu => QwenImageAttn(cpu, 2),
        ["h3_video_vae_vit_block"] = H3VitBlock,
        ["qwen25vl_vision_block_zero_mask"] = cpu => Qwen25Block(cpu, windowed: false),
        ["qwen25vl_vision_block_window_mask"] = cpu => Qwen25Block(cpu, windowed: true),
        ["qwen3vl_vision_block"] = Qwen3Block,
        ["stableaudio_block"] = StableAudioBlock,
    };

    private static Tensor Seq(Owner o, uint seed, float scale, params long[] dims) => o.Own(Seeded(new TensorShape(dims), seed, scale));

    private static Tensor AdditiveMask(int rows, int cols, Func<int, int, bool> allowed)
    {
        Tensor mask = new(new TensorShape(1, 1, rows, cols), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                p[i * cols + j] = allowed(i, j) ? 0f : NegInf;
        return mask;
    }

    private static void LayerNormW(Weights w, string name, int n) { w.AddNorm($"{name}.weight", n); w.AddBias($"{name}.bias", n); }

    // ── T5 block (feeds Flux, SD3, Wan, AuraFlow, MusicGen, ACE-Step, Cosmos) ──────────────────────────────────────

    private static string T5(CpuBackend cpu, int batch, bool padded, float qTau, bool gated, float? scale)
    {
        const int d = 16, heads = 2, hd = 8, ff = 24, seq = 7;
        Owner o = new();
        Weights w = new(400u + (uint)batch + (padded ? 10u : 0u) + (gated ? 0u : 20u) + (qTau == 1f ? 0u : 30u));
        w.AddNorm("t.layer.0.layer_norm.weight", d); w.AddNorm("t.layer.1.layer_norm.weight", d);
        foreach (string n in new[] { "q", "k", "v" }) w.Add($"t.layer.0.SelfAttention.{n}.weight", heads * hd, d);
        w.Add("t.layer.0.SelfAttention.o.weight", d, heads * hd);
        if (gated) { w.Add("t.layer.1.DenseReluDense.wi_0.weight", ff, d); w.Add("t.layer.1.DenseReluDense.wi_1.weight", ff, d); }
        else w.Add("t.layer.1.DenseReluDense.wi.weight", ff, d);
        w.Add("t.layer.1.DenseReluDense.wo.weight", d, ff);
        T5Block block = new(d, heads, hd, 1e-6f, gatedFeedForward: gated, useReluActivation: !gated, attentionScale: scale ?? 1f / MathF.Sqrt(hd));
        block.LoadWeights(w, "t");
        Tensor x = Seq(o, 4001, 1f, batch, seq, d), bias = Seq(o, 4002, 0.5f, 1, heads, seq, seq);
        Tensor? mask = null;
        if (padded)
        {
            mask = o.Own(new Tensor(new TensorShape(batch, seq), DType.F32));
            float* p = (float*)mask.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int j = 0; j < seq; j++) p[b * seq + j] = j < seq - 2 * (b + 1) + 1 ? 1f : 0f;
        }
        Tensor y = o.Own(block.Forward(cpu, x, bias, mask, qTau));
        return Finish(o, ("out", y));
    }

    // ── CLIP text (causal) and vision (bidirectional) transformer layers ────────────────────────────────────────────

    private static void ClipWeights(Weights w, int h, int inter)
    {
        LayerNormW(w, "c.layer_norm1", h); LayerNormW(w, "c.layer_norm2", h);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"c.self_attn.{n}", h, h, true);
        w.Linear("c.mlp.fc1", inter, h, true); w.Linear("c.mlp.fc2", h, inter, true);
    }

    private static string ClipTextLayer(CpuBackend cpu)
    {
        const int h = 16, inter = 24, seq = 7;
        Owner o = new();
        Weights w = new(410);
        ClipWeights(w, h, inter);
        ClipTransformerLayer layer = new(h, 2, inter, 1e-5f, useQuickGelu: true);
        layer.LoadWeights(w, "c");
        Tensor x = Seq(o, 4101, 1f, 2, seq, h);
        Tensor causal = o.Own(AdditiveMask(seq, seq, (i, j) => j <= i));
        Tensor y = o.Own(layer.Forward(cpu, x, causal));
        return Finish(o, ("out", y));
    }

    private static string ClipVisionLayer(CpuBackend cpu)
    {
        const int h = 16, inter = 24, seq = 7;
        Owner o = new();
        Weights w = new(411);
        ClipWeights(w, h, inter);
        ClipVisionTransformerLayer layer = new(h, 2, inter, 1e-5f, useQuickGelu: true);
        layer.LoadWeights(w, "c");
        Tensor y = o.Own(layer.Forward(cpu, Seq(o, 4111, 1f, 2, seq, h)));
        return Finish(o, ("out", y));
    }

    // ── ACE-Step 1.5 attention (GQA, per-head QK norm, host RoPE tables expanded per call, sliding or cross) ──────

    private static string AceStep(CpuBackend cpu, string mode)
    {
        const int hidden = 32, heads = 4, kvHeads = 2, hd = 8, s = 9, l = 6;
        Owner o = new();
        AceStep15Config cfg = new() { HiddenSize = hidden, NumHeads = heads, NumKvHeads = kvHeads, HeadDim = hd, RmsNormEps = 1e-6f };
        Weights w = new(420);
        w.Add("a.q_proj.weight", heads * hd, hidden); w.Add("a.k_proj.weight", kvHeads * hd, hidden);
        w.Add("a.v_proj.weight", kvHeads * hd, hidden); w.Add("a.o_proj.weight", hidden, heads * hd);
        w.AddNorm("a.q_norm.weight", hd); w.AddNorm("a.k_norm.weight", hd);
        AceStep15Attention attn = new(cfg);
        attn.LoadWeights(w, "a");
        Tensor x = Seq(o, 4201, 1f, 1, s, hidden);
        Tensor y;
        if (mode == "cross")
        {
            Tensor kv = Seq(o, 4202, 1f, 1, l, hidden);
            y = o.Own(attn.Forward(cpu, x, kv, null, null, null));
        }
        else
        {
            (float[] cos, float[] sin) = AceStep15Attention.BuildRopeTables(s, hd, 10_000.0);
            Tensor? mask = mode == "sliding" ? o.Own(AceStep15Attention.BuildSlidingMask(s, 2)) : null;
            y = o.Own(attn.Forward(cpu, x, null, cos, sin, mask));
        }
        return Finish(o, ("out", y));
    }

    // ── Leftover sites inside larger models ─────────────────────────────────────────────────────────────────────────

    private static object NewPrivate(Type owner, string nested, params object[] ctorArgs) =>
        Activator.CreateInstance(owner.GetNestedType(nested, BindingFlags.NonPublic | BindingFlags.Public)!,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, ctorArgs, null)!;

    private static object Invoke(object target, string method, params object?[] args) =>
        target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(target, args)!;

    private static string DiamondAttn(CpuBackend cpu)
    {
        const int c = 16, h = 4, w = 4;
        Owner o = new();
        DiamondConfig cfg = new() { AttnHeadDim = 8, GnGroupSize = 8 };
        Tensor x = Seq(o, 4301, 1f, 1, c, h, w);
        Tensor normW = o.Own(Seeded(new TensorShape(c), 4302, 0.25f)), normB = Seq(o, 4303, 0.1f, c);
        Tensor qkvW = Seq(o, 4304, 0.3f, 3 * c, c, 1, 1), qkvB = Seq(o, 4305, 0.1f, 3 * c);
        Tensor outW = Seq(o, 4306, 0.3f, c, c, 1, 1), outB = Seq(o, 4307, 0.1f, c);
        float* nw = (float*)normW.DataPointer;
        for (int i = 0; i < c; i++) nw[i] += 1f;
        Tensor y = o.Own(DiamondOps.SelfAttention(cpu, x, normW, normB, qkvW, qkvB, outW, outB, cfg));
        return Finish(o, ("out", y));
    }

    private static string QwenImageAttn(CpuBackend cpu, int batch)
    {
        const int dim = 16, h = 4, w = 3;
        Owner o = new();
        Weights wt = new(430u + (uint)batch);
        wt.AddTensor("a.norm.gamma", 0.5f, dim, 1, 1);
        wt.AddTensor("a.to_qkv.weight", 0.3f, 3 * dim, dim, 1, 1); wt.AddBias("a.to_qkv.bias", 3 * dim);
        wt.AddTensor("a.proj.weight", 0.3f, dim, dim, 1, 1); wt.AddBias("a.proj.bias", dim);
        QwenImageAttentionBlock block = new(dim);
        block.LoadWeights(wt, "a");
        Tensor y = o.Own(block.Forward(cpu, Seq(o, 4311, 1f, batch, dim, h, w)));
        return Finish(o, ("out", y));
    }

    private static string H3VitBlock(CpuBackend cpu)
    {
        const int seq = 9;
        Owner o = new();
        MiniMaxH3VideoVaeConfig cfg = new() { Heads = 2, DimHead = 8 };
        int dim = cfg.Dim, hd = cfg.DimHead, inner = cfg.FfnInner;
        Weights w = new(440);
        w.AddNorm("v.norm1.weight", dim); w.AddNorm("v.norm2.weight", dim);
        w.AddTensor("v.scale1", 0.5f, dim); w.AddTensor("v.scale2", 0.5f, dim);
        w.Linear("v.attn.to_qkv", 3 * dim, dim, true); w.Linear("v.attn.to_out", dim, dim, true);
        w.Linear("v.ff.w1", 2 * inner, dim, true); w.Linear("v.ff.w2", dim, inner, true);
        List<Tensor> owned = [];
        MiniMaxH3VideoVaeVitBlock block = new(cfg);
        block.LoadWeights(w, "v", owned);
        foreach (Tensor t in owned) o.Own(t);
        Tensor h = Seq(o, 4401, 1f, seq, dim), cos = Seq(o, 4402, 1f, seq, hd), sin = Seq(o, 4403, 1f, seq, hd);
        Tensor ones = o.Own(new Tensor(new TensorShape(hd), DType.F32));
        new Span<float>((float*)ones.DataPointer, hd).Fill(1f);
        block.Forward(cpu, h, cos, sin, ones);   // updates h in place
        return Finish(o, ("out", h));
    }

    private static string Qwen25Block(CpuBackend cpu, bool windowed)
    {
        const int hidden = 16, heads = 2, hd = 8, inter = 24, np = 12;
        Owner o = new();
        Qwen25VlVisionConfig cfg = new() { HiddenSize = hidden, NumHeads = heads, IntermediateSize = inter };
        Weights w = new(windowed ? 451u : 450u);
        w.AddNorm("b.norm1.weight", hidden); w.AddNorm("b.norm2.weight", hidden);
        w.Linear("b.attn.qkv", 3 * hidden, hidden, true); w.Linear("b.attn.proj", hidden, hidden, true);
        w.Linear("b.mlp.gate_proj", inter, hidden, true); w.Linear("b.mlp.up_proj", inter, hidden, true); w.Linear("b.mlp.down_proj", hidden, inter, true);
        object block = NewPrivate(typeof(Qwen25VlVisionEncoder), "Block", cfg);
        Invoke(block, "LoadWeights", w, "b");
        float[] cos = new float[np * hd], sin = new float[np * hd];
        for (int p = 0; p < np; p++)
            for (int e = 0; e < hd; e++)
            {
                double ang = p * Math.Pow(10_000.0, -2.0 * (e % (hd / 2)) / hd);
                cos[p * hd + e] = (float)Math.Cos(ang); sin[p * hd + e] = (float)Math.Sin(ang);
            }
        // Full layers carry an all-zero mask (not null); windowed layers a block-diagonal one.
        Tensor mask = o.Own(AdditiveMask(np, np, (i, j) => !windowed || i / 4 == j / 4));
        Tensor x = Seq(o, 4501, 1f, 1, np, hidden);
        Tensor y = o.Own((Tensor)Invoke(block, "Forward", cpu, x, cos, sin, np, mask));
        return Finish(o, ("out", y));
    }

    private static string Qwen3Block(CpuBackend cpu)
    {
        const int hidden = 16, heads = 2, hd = 8, half = hd / 2, inter = 24, seq = 12;
        Owner o = new();
        Qwen3VlVisionConfig cfg = new() { HiddenSize = hidden, NumHeads = heads, IntermediateSize = inter };
        Weights w = new(452);
        LayerNormW(w, "b.norm1", hidden); LayerNormW(w, "b.norm2", hidden);
        w.Linear("b.attn.qkv", 3 * hidden, hidden, true); w.Linear("b.attn.proj", hidden, hidden, true);
        w.Linear("b.mlp.linear_fc1", inter, hidden, true); w.Linear("b.mlp.linear_fc2", hidden, inter, true);
        object block = NewPrivate(typeof(Qwen3VlVisionEncoder), "VisionBlock", cfg);
        Invoke(block, "LoadWeights", w, "b");
        float[] cos = new float[seq * half], sin = new float[seq * half];
        for (int p = 0; p < seq; p++)
            for (int i = 0; i < half; i++)
            {
                double ang = p * Math.Pow(10_000.0, -2.0 * i / hd);
                cos[p * half + i] = (float)Math.Cos(ang); sin[p * half + i] = (float)Math.Sin(ang);
            }
        Tensor x = Seq(o, 4521, 1f, 1, seq, hidden);
        Tensor y = o.Own((Tensor)Invoke(block, "Forward", cpu, x, cos, sin, seq));
        return Finish(o, ("out", y));
    }

    private static string StableAudioBlock(CpuBackend cpu)
    {
        const int dim = 32, heads = 4, hd = 8, kvHeads = 2, inner = 48, rows = 9, condLen = 5;
        Owner o = new();
        StableAudioDitConfig cfg = new() { HiddenSize = dim, NumHeads = heads, HeadDim = hd, CrossKvHeads = kvHeads, FfInner = inner, RopeDim = 4 };
        Weights w = new(460);
        LayerNormW(w, "b.pre_norm", dim); LayerNormW(w, "b.cross_attend_norm", dim); LayerNormW(w, "b.ff_norm", dim);
        w.Add("b.self_attn.to_qkv.weight", 3 * dim, dim); w.Add("b.self_attn.to_out.weight", dim, dim);
        LayerNormW(w, "b.self_attn.q_norm", hd); LayerNormW(w, "b.self_attn.k_norm", hd);
        w.Add("b.cross_attn.to_q.weight", dim, dim); w.Add("b.cross_attn.to_kv.weight", 2 * kvHeads * hd, dim); w.Add("b.cross_attn.to_out.weight", dim, dim);
        LayerNormW(w, "b.cross_attn.q_norm", hd); LayerNormW(w, "b.cross_attn.k_norm", hd);
        w.Linear("b.ff.ff.0.proj", 2 * inner, dim, true); w.Linear("b.ff.ff.2", dim, inner, true);
        object block = NewPrivate(typeof(StableAudioDit), "Block", cfg);
        Invoke(block, "LoadWeights", w, "b");
        Tensor x = Seq(o, 4601, 1f, 1, rows, dim), cond = Seq(o, 4602, 1f, 1, condLen, dim);
        Tensor cos = Seq(o, 4603, 1f, rows, hd), sin = Seq(o, 4604, 1f, rows, hd);
        Tensor y = o.Own((Tensor)Invoke(block, "Forward", cpu, x, cond, rows, condLen, cos, sin));
        return Finish(o, ("out", y));
    }
}

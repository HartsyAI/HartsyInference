using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.TextEncoders.Bert;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Detection.Blocks;
using HartsyInference.Vision.Detection.GroundingDino;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.DinoVit;
using HartsyInference.Vision.Segmentation.Sam2;
using HartsyInference.Vision.Siglip;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Vision.Tests.Attention;

/// <summary>Forward passes of the Wave E vision attention sites (DINOv2, DINO ViT, SigLIP, SAM2 decoder and Hiera, Swin window attention,
/// RT-DETR / GroundingDINO multi-head attention, BERT, PSA) on tiny pseudo-random weights and inputs, each reduced to SHA-256 digests of
/// its output bits. Everything runs on the CPU backend, so the same cases produce the golden fixture from the pre-migration source and
/// must reproduce it after the attention refactor. Weights come from the per-case-salted <c>Weights</c> harness.</summary>
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
        ["dinov2_b1"] = cpu => Dinov2(cpu, 1),
        ["dinov2_b2"] = cpu => Dinov2(cpu, 2),
        ["dinovit_layer"] = DinoVit,
        ["siglip_layer"] = SiglipLayer,
        ["siglip_attention_pool_b1"] = cpu => SiglipPool(cpu, 1),
        ["siglip_attention_pool_b2"] = cpu => SiglipPool(cpu, 2),
        ["sam2_decoder_self"] = cpu => Sam2(cpu, internalDim: 16, cross: false),
        ["sam2_decoder_cross"] = cpu => Sam2(cpu, internalDim: 8, cross: true),
        ["hiera_windowed"] = cpu => Hiera(cpu, dimIn: 8, dimOut: 8, ws: 2, qPool: false, h: 5, w: 5),
        ["hiera_pooled"] = cpu => Hiera(cpu, dimIn: 8, dimOut: 16, ws: 4, qPool: true, h: 6, w: 6),
        ["hiera_global"] = cpu => Hiera(cpu, dimIn: 8, dimOut: 8, ws: 0, qPool: false, h: 4, w: 4),
        ["swin_window_unshifted"] = cpu => SwinWindow(cpu, shift: 0),
        ["swin_window_shifted_padded"] = cpu => SwinWindow(cpu, shift: 2),
        ["rtdetr_decoder_self"] = cpu => RtDetrMultiHead(cpu, mask: false),
        ["rtdetr_aifi_masked"] = cpu => RtDetrMultiHead(cpu, mask: true),
        ["groundingdino_mha_masked"] = GroundingDinoMha,
        ["bert_masked"] = cpu => Bert(cpu, masked: true),
        ["bert_null"] = cpu => Bert(cpu, masked: false),
        ["psa"] = Psa,
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

    // ── ViT family (DINOv2 with LayerScale, DINO ViT, SigLIP) ───────────────────────────────────────────────────────

    private static string Dinov2(CpuBackend cpu, int batch)
    {
        const int h = 16, inter = 24, seq = 7;
        Owner o = new();
        Weights w = new(200u + (uint)batch);
        LayerNormW(w, "l.norm1", h); LayerNormW(w, "l.norm2", h);
        foreach (string n in new[] { "query", "key", "value" }) w.Linear($"l.attention.attention.{n}", h, h, true);
        w.Linear("l.attention.output.dense", h, h, true);
        w.AddTensor("l.layer_scale1.lambda1", 0.5f, h); w.AddTensor("l.layer_scale2.lambda1", 0.5f, h);
        w.Linear("l.mlp.fc1", inter, h, true); w.Linear("l.mlp.fc2", h, inter, true);
        Dinov2Layer layer = new(h, 2, inter, 1e-6f);
        layer.LoadWeights(w, "l");
        Tensor y = o.Own(layer.Forward(cpu, Seq(o, 2001, 1f, batch, seq, h)));
        return Finish(o, ("out", y));
    }

    private static string DinoVit(CpuBackend cpu)
    {
        const int h = 16, inter = 24, seq = 7;
        Owner o = new();
        Weights w = new(203);
        LayerNormW(w, "l.layernorm_before", h); LayerNormW(w, "l.layernorm_after", h);
        foreach (string n in new[] { "query", "key", "value" }) w.Linear($"l.attention.attention.{n}", h, h, true);
        w.Linear("l.attention.output.dense", h, h, true);
        w.Linear("l.intermediate.dense", inter, h, true); w.Linear("l.output.dense", h, inter, true);
        DinoViTLayer layer = new(h, 2, inter, 1e-6f);
        layer.LoadWeights(w, "l");
        Tensor y = o.Own(layer.Forward(cpu, Seq(o, 2031, 1f, 2, seq, h)));
        return Finish(o, ("out", y));
    }

    private static string SiglipLayer(CpuBackend cpu)
    {
        const int h = 16, inter = 24, seq = 7;
        Owner o = new();
        Weights w = new(204);
        LayerNormW(w, "l.layer_norm1", h); LayerNormW(w, "l.layer_norm2", h);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"l.self_attn.{n}", h, h, true);
        w.Linear("l.mlp.fc1", inter, h, true); w.Linear("l.mlp.fc2", h, inter, true);
        SiglipTransformerLayer layer = new(h, 2, inter, 1e-6f);
        layer.LoadWeights(w, "l");
        Tensor y = o.Own(layer.Forward(cpu, Seq(o, 2041, 1f, 2, seq, h)));
        return Finish(o, ("out", y));
    }

    /// <summary>The multi-head attention-pooling head (a private method of the encoder, driven through its private fields): a learned probe attends over the patch tokens.</summary>
    private static string SiglipPool(CpuBackend cpu, int batch)
    {
        const int h = 16, patches = 4;
        Owner o = new();
        SiglipPreset preset = new()
        {
            Name = "tiny", HiddenSize = h, NumLayers = 1, NumHeads = 2, IntermediateSize = 24, ImageSize = 8, PatchSize = 4, EmbeddingDim = h,
        };
        SiglipVisionEncoder enc = new(preset);
        void Set(string field, Tensor value) =>
            typeof(SiglipVisionEncoder).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(enc, o.Own(value));
        Set("_headProbe", Seeded(new TensorShape(1, 1, h), 2201, 0.5f));
        Set("_headAttnInProjWeight", Seeded(new TensorShape(3 * h, h), 2202, 0.3f)); Set("_headAttnInProjBias", Seeded(new TensorShape(3 * h), 2203, 0.1f));
        Set("_headAttnOutProjWeight", Seeded(new TensorShape(h, h), 2204, 0.3f)); Set("_headAttnOutProjBias", Seeded(new TensorShape(h), 2205, 0.1f));
        Tensor tokens = Seq(o, 2206, 1f, batch, patches, h);
        Tensor y = o.Own((Tensor)typeof(SiglipVisionEncoder).GetMethod("AttentionPool", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(enc, [cpu, tokens, batch, h])!);
        return Finish(o, ("out", y));
    }

    // ── SAM2: mask-decoder two-way attention (internal dim below the embed dim for cross) and Hiera blocks ──────────

    private static string Sam2(CpuBackend cpu, int internalDim, bool cross)
    {
        const int embed = 16, nq = 5, nk = 9;
        Owner o = new();
        Weights w = new(cross ? 205u : 206u);
        w.Linear("a.q_proj", internalDim, embed, true); w.Linear("a.k_proj", internalDim, embed, true);
        w.Linear("a.v_proj", internalDim, embed, true); w.Linear("a.out_proj", embed, internalDim, true);
        DecoderAttention attn = new(2);
        attn.LoadWeights(w, "a.");
        Tensor q = Seq(o, 2051, 1f, 2, nq, embed);
        Tensor k = cross ? Seq(o, 2052, 1f, 2, nk, embed) : Seq(o, 2051, 1f, 2, nq, embed);
        Tensor v = cross ? Seq(o, 2053, 1f, 2, nk, embed) : Seq(o, 2051, 1f, 2, nq, embed);
        Tensor y = o.Own(attn.Forward(cpu, q, k, v));
        return Finish(o, ("out", y));
    }

    private static string Hiera(CpuBackend cpu, int dimIn, int dimOut, int ws, bool qPool, int h, int w)
    {
        Owner o = new();
        Weights wt = new(207u + (uint)dimOut + (uint)ws);
        LayerNormW(wt, "h.norm1", dimIn); LayerNormW(wt, "h.norm2", dimOut);
        wt.Linear("h.attn.qkv", 3 * dimOut, dimIn, true); wt.Linear("h.attn.proj", dimOut, dimOut, true);
        wt.Linear("h.mlp.layers.0", 4 * dimOut, dimOut, true); wt.Linear("h.mlp.layers.1", dimOut, 4 * dimOut, true);
        if (dimIn != dimOut) wt.Linear("h.proj", dimOut, dimIn, true);
        HieraBlock block = new(dimIn, dimOut, 2, ws, qPool, 4, 1e-6f);
        block.LoadWeights(wt, "h.");
        Tensor x = Seq(o, 2071, 1f, 1, h, w, dimIn);
        (Tensor y, int oh, int ow) = block.Forward(cpu, x, h, w);
        o.Own(y);
        Tensor dims = o.Own(new Tensor(new TensorShape(2), DType.F32));
        ((float*)dims.DataPointer)[0] = oh; ((float*)dims.DataPointer)[1] = ow;
        return Finish(o, ("out", y), ("dims", dims));
    }

    // ── Swin window attention (private instance method of SwinBackbone; relative-position bias plus the -100 shift mask) ──

    private static string SwinWindow(CpuBackend cpu, int shift)
    {
        const int dim = 8, heads = 2, ws = 4, hp = 8, wp = 8;
        Owner o = new();
        using SwinBackbone backbone = new(new GroundingDinoConfig());
        Tensor q = Seq(o, 2081, 1f, 1, hp * wp, dim), k = Seq(o, 2082, 1f, 1, hp * wp, dim), v = Seq(o, 2083, 1f, 1, hp * wp, dim);
        Tensor rel = Seq(o, 2084, 0.5f, (2 * ws - 1) * (2 * ws - 1), heads);
        MethodInfo method = typeof(SwinBackbone).GetMethod("WindowAttention", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Tensor y = o.Own((Tensor)method.Invoke(backbone, [cpu, q, k, v, hp, wp, dim, heads, ws, shift, rel])!);
        return Finish(o, ("out", y));
    }

    // ── RT-DETR / GroundingDINO ─────────────────────────────────────────────────────────────────────────────────────

    private static string RtDetrMultiHead(CpuBackend cpu, bool mask)
    {
        const int heads = 2, hd = 8, sq = 6, skv = 6;
        Owner o = new();
        Tensor q = Seq(o, 2091, 1f, 1, sq, heads * hd), k = Seq(o, 2092, 1f, 1, skv, heads * hd), v = Seq(o, 2093, 1f, 1, skv, heads * hd);
        Tensor? m = mask ? o.Own(AdditiveMask(sq, skv, (i, j) => j <= i + 1)) : null;
        Tensor y = o.Own(RtDetrAttention.MultiHead(cpu, q, k, v, heads, hd, m));
        return Finish(o, ("out", y));
    }

    private static string GroundingDinoMha(CpuBackend cpu)
    {
        // Image queries over text keys with a key-padding mask, then the swapped roles (text queries over image keys).
        const int hidden = 16, heads = 2, nImg = 7, nTxt = 5;
        Owner o = new();
        Weights w = new(210);
        foreach (string n in new[] { "query", "key", "value", "out_proj" }) w.Linear($"m.{n}", hidden, hidden, true);
        using GroundingDinoMultiheadAttention mha = new(hidden, heads);
        mha.LoadWeights(w, "m");
        Tensor img = Seq(o, 2101, 1f, 1, nImg, hidden), txt = Seq(o, 2102, 1f, 1, nTxt, hidden);
        Tensor pad = o.Own(AdditiveMask(nImg, nTxt, (i, j) => j < nTxt - 1));
        Tensor y = o.Own(mha.Forward(cpu, img, txt, txt, pad));
        Tensor swapped = o.Own(mha.Forward(cpu, txt, img, img, null));
        return Finish(o, ("masked", y), ("swapped", swapped));
    }

    // ── BERT (GroundingDINO text tower mask, MeloTTS null mask) ─────────────────────────────────────────────────────

    private static string Bert(CpuBackend cpu, bool masked)
    {
        const int h = 16, inter = 24, heads = 2, vocab = 20, t = 7;
        Owner o = new();
        BertConfig cfg = new() { Hidden = h, NumLayers = 2, NumHeads = heads, Intermediate = inter, VocabSize = vocab, MaxPositions = 16 };
        Weights w = new(masked ? 211u : 212u);
        w.Add("bert.embeddings.word_embeddings.weight", vocab, h); w.Add("bert.embeddings.position_embeddings.weight", 16, h);
        w.Add("bert.embeddings.token_type_embeddings.weight", 2, h);
        LayerNormW(w, "bert.embeddings.LayerNorm", h);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"bert.encoder.layer.{i}";
            foreach (string n in new[] { "query", "key", "value" }) w.Linear($"{p}.attention.self.{n}", h, h, true);
            w.Linear($"{p}.attention.output.dense", h, h, true); LayerNormW(w, $"{p}.attention.output.LayerNorm", h);
            w.Linear($"{p}.intermediate.dense", inter, h, true); w.Linear($"{p}.output.dense", h, inter, true);
            LayerNormW(w, $"{p}.output.LayerNorm", h);
        }
        using BertModel bert = new(cfg);
        bert.LoadWeights(w);
        int[] ids = [3, 7, 1, 12, 5, 9, 2];
        // Head-invariant block mask [1, heads, t, t]: the last two tokens are masked out as keys.
        Tensor? mask = null;
        if (masked)
        {
            mask = o.Own(new Tensor(new TensorShape(1, heads, t, t), DType.F32));
            float* p = (float*)mask.DataPointer;
            for (int hh = 0; hh < heads; hh++)
                for (int i = 0; i < t; i++)
                    for (int j = 0; j < t; j++) p[((hh * t) + i) * t + j] = j < t - 2 ? 0f : NegInf;
        }
        Tensor y = o.Own(bert.Forward(cpu, ids, default, default, mask));
        return Finish(o, ("out", y));
    }

    // ── YOLO11 PSA (channel-first Q/K of width kd, V of head width, plus a depthwise positional conv) ───────────────

    private static string Psa(CpuBackend cpu)
    {
        const int dim = 16, heads = 2, hd = dim / heads, kd = hd / 2, side = 4;
        Owner o = new();
        Weights w = new(213);
        w.AddTensor("p.qkv.conv.weight", 0.3f, dim + 2 * heads * kd, dim, 1, 1); w.AddBias("p.qkv.conv.bias", dim + 2 * heads * kd);
        w.AddTensor("p.proj.conv.weight", 0.3f, dim, dim, 1, 1); w.AddBias("p.proj.conv.bias", dim);
        w.AddTensor("p.pe.conv.weight", 0.3f, dim, 1, 3, 3); w.AddBias("p.pe.conv.bias", dim);
        PsaAttention psa = new(dim, heads);
        psa.LoadWeights(w, "p");
        Tensor y = o.Own(psa.Forward(cpu, Seq(o, 2131, 1f, 1, dim, side, side)));
        return Finish(o, ("out", y));
    }
}

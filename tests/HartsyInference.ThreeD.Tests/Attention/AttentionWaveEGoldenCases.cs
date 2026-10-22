using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.TripoSr;
using HartsyInference.ThreeD.Models.Trellis;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.ThreeD.Tests.Attention;

/// <summary>Forward passes of the Wave E 3D attention sites (TripoSR block, Hunyuan3D shape-VAE blocks and DiT blocks, Trellis sparse-structure
/// flow and Gaussian decoder blocks) on tiny pseudo-random weights and inputs where the width is configurable, or on the fixed real width where it
/// is a constant, each reduced to SHA-256 digests of its output bits. Everything runs on the CPU backend, so the same cases produce the golden
/// fixture from the pre-migration source and must reproduce it after the attention refactor.</summary>
internal static unsafe class AttentionWaveEGoldenCases
{
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
        ["triposr_self_cross_block"] = TripoSr,
        ["hunyuan3d_vae_resblock"] = VaeResBlock,
        ["hunyuan3d_geo_query_chunked"] = GeoQuery,
        ["hunyuan3d_dit_double"] = DitDouble,
        ["hunyuan3d_dit_single"] = DitSingle,
        ["trellis_ssflow"] = SsFlow,
        ["trellis_gs"] = Gs,
    };

    private static Tensor Seq(Owner o, uint seed, float scale, params long[] dims) => o.Own(Seeded(new TensorShape(dims), seed, scale));

    private static void LayerNormW(Weights w, string name, int n) { w.AddNorm($"{name}.weight", n); w.AddBias($"{name}.bias", n); }

    // ── TripoSR transformer block ───────────────────────────────────────────────────────────────────────────────────

    private static string TripoSr(CpuBackend cpu)
    {
        const int width = 16, heads = 2, hd = 8, cross = 12, n = 9, s = 5;
        Owner o = new();
        TripoSrConfig cfg = new()
        {
            TriplaneChannels = 4, PlaneSize = 2, NumChannels = width, Width = width, Depth = 1, NumHeads = heads, HeadDim = hd,
            CrossAttentionDim = cross, NerfHidden = 8, NerfMidLayers = 1,
        };
        Weights w = new(300);
        LayerNormW(w, "b.norm1", width); LayerNormW(w, "b.norm2", width); LayerNormW(w, "b.norm3", width);
        foreach (string n1 in new[] { "to_q", "to_k", "to_v" }) w.Add($"b.attn1.{n1}.weight", width, width);
        w.Linear("b.attn1.to_out.0", width, width, true);
        w.Add("b.attn2.to_q.weight", width, width); w.Add("b.attn2.to_k.weight", width, cross); w.Add("b.attn2.to_v.weight", width, cross);
        w.Linear("b.attn2.to_out.0", width, width, true);
        w.Linear("b.ff.net.0.proj", 2 * 24, width, true); w.Linear("b.ff.net.2", width, 24, true);
        TripoSrBlock block = new(cfg);
        block.LoadWeights(w, "b");
        Tensor h = Seq(o, 3001, 1f, 1, n, width), img = Seq(o, 3002, 1f, 1, s, cross);
        Tensor self = o.Own((Tensor)typeof(TripoSrBlock).GetMethod("SelfAttn", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(block, [cpu, h, n])!);
        Tensor crossOut = o.Own((Tensor)typeof(TripoSrBlock).GetMethod("CrossAttn", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(block, [cpu, h, img, n, s])!);
        Tensor full = o.Own(block.Forward(cpu, h, img));
        return Finish(o, ("self", self), ("cross", crossOut), ("block", full));
    }

    // ── Hunyuan3D shape VAE ─────────────────────────────────────────────────────────────────────────────────────────

    private static string VaeResBlock(CpuBackend cpu)
    {
        const int width = 16, heads = 2, hd = 8, n = 9;
        Owner o = new();
        Weights w = new(301);
        LayerNormW(w, "r.ln_1", width); LayerNormW(w, "r.ln_2", width);
        w.Add("r.attn.c_qkv.weight", 3 * width, width); w.Linear("r.attn.c_proj", width, width, true);
        LayerNormW(w, "r.attn.attention.q_norm", hd); LayerNormW(w, "r.attn.attention.k_norm", hd);
        w.Linear("r.mlp.c_fc", 4 * width, width, true); w.Linear("r.mlp.c_proj", width, 4 * width, true);
        Hunyuan3DVaeResBlock block = new(width, heads);
        block.LoadWeights(w, "r");
        Tensor y = o.Own(block.Forward(cpu, Seq(o, 3011, 1f, 1, n, width)));
        return Finish(o, ("out", y));
    }

    private static string GeoQuery(CpuBackend cpu)
    {
        const int width = 16, heads = 2, hd = 8, bands = 2, fourier = 3 + 3 * 2 * bands, nl = 10;
        Owner o = new();
        Weights w = new(302);
        w.Linear("g.query_proj", width, fourier, true);
        const string c = "g.cross_attn_decoder";
        foreach (string n in new[] { "ln_1", "ln_2", "ln_3" }) LayerNormW(w, $"{c}.{n}", width);
        w.Add($"{c}.attn.c_q.weight", width, width); w.Add($"{c}.attn.c_kv.weight", 2 * width, width);
        w.Linear($"{c}.attn.c_proj", width, width, true);
        LayerNormW(w, $"{c}.attn.attention.q_norm", hd); LayerNormW(w, $"{c}.attn.attention.k_norm", hd);
        w.Linear($"{c}.mlp.c_fc", 4 * width, width, true); w.Linear($"{c}.mlp.c_proj", width, 4 * width, true);
        LayerNormW(w, "g.ln_post", width); w.Linear("g.output_proj", 1, width, true);
        Hunyuan3DGeoDecoder dec = new(width, heads, bands, fourier);
        dec.LoadWeights(w, "g");
        Tensor latents = Seq(o, 3021, 1f, 1, nl, width);
        (Tensor kP, Tensor vP) = dec.PrepareKv(cpu, latents);
        o.Own(kP); o.Own(vP);
        // Two chunks of query points against the same K/V.
        Tensor c0 = Seq(o, 3022, 1f, 6, 3), c1 = Seq(o, 3023, 1f, 4, 3);
        Tensor y0 = o.Own(dec.Query(cpu, (kP, vP), c0, 6));
        Tensor y1 = o.Own(dec.Query(cpu, (kP, vP), c1, 4));
        return Finish(o, ("chunk0", y0), ("chunk1", y1), ("k", kP), ("v", vP));
    }

    // ── Hunyuan3D DiT (CPU F32 only: the real path is native F16 and graph-captured) ──────────────────────────────

    private static string DitDouble(CpuBackend cpu)
    {
        const int width = 16, heads = 2, hd = 8, mlp = 32, nImg = 8, nTxt = 5;
        Owner o = new();
        Weights w = new(303);
        foreach (string s in new[] { "img", "txt" })
        {
            w.Linear($"d.{s}_mod.lin", 6 * width, width, true);
            w.Linear($"d.{s}_attn.qkv", 3 * width, width, true); w.Linear($"d.{s}_attn.proj", width, width, true);
            w.AddNorm($"d.{s}_attn.norm.query_norm.scale", hd); w.AddNorm($"d.{s}_attn.norm.key_norm.scale", hd);
            w.Linear($"d.{s}_mlp.0", mlp, width, true); w.Linear($"d.{s}_mlp.2", width, mlp, true);
        }
        Hunyuan3DDoubleBlock block = new(width, heads, mlp);
        block.LoadWeights(w, "d");
        Tensor img = Seq(o, 3031, 1f, 1, nImg, width), txt = Seq(o, 3032, 1f, 1, nTxt, width), vec = Seq(o, 3033, 1f, 1, width);
        (Tensor i, Tensor t) = block.Forward(cpu, img, txt, vec);
        o.Own(i); o.Own(t);
        return Finish(o, ("img", i), ("txt", t));
    }

    private static string DitSingle(CpuBackend cpu)
    {
        const int width = 16, heads = 2, hd = 8, mlp = 32, n = 11;
        Owner o = new();
        Weights w = new(304);
        w.Linear("s.modulation.lin", 3 * width, width, true);
        w.Linear("s.linear1", 3 * width + mlp, width, true); w.Linear("s.linear2", width, width + mlp, true);
        w.AddNorm("s.norm.query_norm.scale", hd); w.AddNorm("s.norm.key_norm.scale", hd);
        Hunyuan3DSingleBlock block = new(width, heads, mlp);
        block.LoadWeights(w, "s");
        Tensor x = Seq(o, 3041, 1f, 1, n, width), vec = Seq(o, 3042, 1f, 1, width);
        Tensor y = o.Own(block.Forward(cpu, x, vec));
        return Finish(o, ("out", y));
    }

    // ── Trellis (fixed 1024-wide sparse-structure flow block and 768-wide Gaussian-decoder block) ─────────────────

    private static string SsFlow(CpuBackend cpu)
    {
        const int w = 1024, hd = 64, mlp = 4096, n = 9, lc = 6;
        const float s = 0.02f;
        Owner o = new();
        Weights wt = new(305);
        wt.AddTensor("b.adaLN_modulation.1.weight", s, 6 * w, w); wt.AddBias("b.adaLN_modulation.1.bias", 6 * w);
        LayerNormW(wt, "b.norm2", w);
        wt.AddTensor("b.self_attn.to_qkv.weight", s, 3 * w, w); wt.AddBias("b.self_attn.to_qkv.bias", 3 * w);
        wt.AddTensor("b.self_attn.to_out.weight", s, w, w); wt.AddBias("b.self_attn.to_out.bias", w);
        wt.AddTensor("b.self_attn.q_rms_norm.gamma", 0.25f, 16, hd); wt.AddTensor("b.self_attn.k_rms_norm.gamma", 0.25f, 16, hd);
        wt.AddTensor("b.cross_attn.to_q.weight", s, w, w); wt.AddBias("b.cross_attn.to_q.bias", w);
        wt.AddTensor("b.cross_attn.to_kv.weight", s, 2 * w, w); wt.AddBias("b.cross_attn.to_kv.bias", 2 * w);
        wt.AddTensor("b.cross_attn.to_out.weight", s, w, w); wt.AddBias("b.cross_attn.to_out.bias", w);
        wt.AddTensor("b.mlp.mlp.0.weight", s, mlp, w); wt.AddBias("b.mlp.mlp.0.bias", mlp);
        wt.AddTensor("b.mlp.mlp.2.weight", s, w, mlp); wt.AddBias("b.mlp.mlp.2.bias", w);
        SsFlowBlock block = SsFlowBlock.Load(wt, "b");
        Tensor ones = o.Own(new Tensor(new TensorShape(hd), DType.F32));
        new Span<float>((float*)ones.DataPointer, hd).Fill(1f);
        Tensor x = Seeded(new TensorShape(1, n, w), 3051), vec = Seq(o, 3052, 1f, 1, w), cond = Seq(o, 3053, 1f, 1, lc, w);
        Tensor y = o.Own(block.Forward(cpu, x, vec, cond, ones));   // consumes x
        return Finish(o, ("out", y));
    }

    private static string Gs(CpuBackend cpu)
    {
        const int w = 768, mlp = 3072, n = 12;
        const float s = 0.02f;
        Owner o = new();
        Weights wt = new(306);
        wt.AddTensor("b.attn.to_qkv.weight", s, 3 * w, w); wt.AddBias("b.attn.to_qkv.bias", 3 * w);
        wt.AddTensor("b.attn.to_out.weight", s, w, w); wt.AddBias("b.attn.to_out.bias", w);
        wt.AddTensor("b.mlp.mlp.0.weight", s, mlp, w); wt.AddBias("b.mlp.mlp.0.bias", mlp);
        wt.AddTensor("b.mlp.mlp.2.weight", s, w, mlp); wt.AddBias("b.mlp.mlp.2.bias", w);
        GsBlock block = GsBlock.Load(wt, "b");
        Tensor x = Seq(o, 3061, 1f, 1, n, w);
        // Block-diagonal window mask: rows 0-5 and 6-11 are separate windows.
        Tensor mask = o.Own(new Tensor(new TensorShape(1, 1, n, n), DType.F32));
        float* m = (float*)mask.DataPointer;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) m[i * n + j] = i / 6 == j / 6 ? 0f : -1e30f;
        Tensor y = o.Own(block.Forward(cpu, x, mask, n));
        return Finish(o, ("out", y));
    }
}

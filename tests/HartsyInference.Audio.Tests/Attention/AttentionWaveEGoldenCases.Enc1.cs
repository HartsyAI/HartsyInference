using System.Reflection;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterEnc1(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["hubert_layer"] = HubertLayer;
        d["neucodec_txblock"] = NeuCodecTxBlock;
        d["chatterbox_perceiver"] = ChatterboxPerceiverCase;
        d["auk_attend_op"] = AukAttend;
        d["auk_single"] = AukSingle;
        d["auk_double"] = AukDouble;
        d["controlfoley_vit"] = ControlFoleyVit;
        d["controlfoley_clip_text"] = cpu => ControlFoleyClip(cpu, causal: true);
        d["controlfoley_clip_vision"] = cpu => ControlFoleyClip(cpu, causal: false);
        d["controlfoley_block_attend"] = ControlFoleyBlockAttend;
        d["controlfoley_vae_attn"] = ControlFoleyVaeAttn;
    }

    /// <summary>A private nested class's instance, its <c>LoadWeights</c>/<c>Load</c> call and its <c>Forward</c>, all by reflection (test-only, no src seam).</summary>
    private static object NewPrivate(Type owner, string nested, params object[] ctorArgs) =>
        Activator.CreateInstance(owner.GetNestedType(nested, BindingFlags.NonPublic | BindingFlags.Public)!,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, ctorArgs, null)!;

    private static object Invoke(object target, string method, params object?[] args) =>
        target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(target, args)!;

    // ── HuBERT encoder layer (private nested Layer) ─────────────────────────────────────────────────────────────

    private static string HubertLayer(CpuBackend cpu)
    {
        const int h = 24, heads = 2, ffn = 32, t = 9;
        Owner o = new();
        HubertConfig cfg = new() { Hidden = h, NumHeads = heads, FfnDim = ffn };
        Weights w = new(130);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"l.attention.{n}", h, h, true);
        w.AddNorm("l.layer_norm.weight", h); w.AddBias("l.layer_norm.bias", h);
        w.Linear("l.feed_forward.intermediate_dense", ffn, h, true); w.Linear("l.feed_forward.output_dense", h, ffn, true);
        w.AddNorm("l.final_layer_norm.weight", h); w.AddBias("l.final_layer_norm.bias", h);
        object layer = NewPrivate(typeof(Hubert), "Layer", cfg);
        Invoke(layer, "LoadWeights", w, "l");
        Tensor x = Seq(o, 1301, 1f, 1, t, h);
        Tensor y = o.Own((Tensor)Invoke(layer, "Forward", cpu, x, t));
        return Finish(o, ("out", y));
    }

    // ── NeuCodec decoder transformer block (private TxBlock over a private TxWeights) ───────────────────────────

    private static string NeuCodecTxBlock(CpuBackend cpu)
    {
        const int dim = 16, mlp = 24, t = 9;
        Owner o = new();
        NeuCodecConfig cfg = new() { BackboneDim = dim, NumHeads = 2, MlpDim = mlp };
        Weights w = new(131);
        w.AddNorm("b.input_layernorm.weight", dim); w.AddNorm("b.post_attention_layernorm.weight", dim);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "o_proj" }) w.Add($"b.self_attn.{n}.weight", dim, dim);
        w.Add("b.mlp.fc1.weight", mlp, dim); w.Add("b.mlp.fc2.weight", dim, mlp);
        NeuCodecDecoder dec = new(cfg);
        object txWeights = typeof(NeuCodecDecoder).GetMethod("LoadTx", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, [w, "b"])!;
        Tensor x = Seq(o, 1311, 1f, 1, t, dim);
        Tensor y = o.Own((Tensor)typeof(NeuCodecDecoder).GetMethod("TxBlock", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(dec, [cpu, x, txWeights, t])!);
        return Finish(o, ("out", y));
    }

    // ── Chatterbox perceiver (fixed 1024 wide; learned-query cross-attention then self-attention) ───────────────

    private static string ChatterboxPerceiverCase(CpuBackend cpu)
    {
        const int dim = 1024;
        const float s = 0.03f;
        Owner o = new();
        Weights w = new(132);
        w.AddTensor("p.pre_attention_query", 0.5f, 1, 32, dim);
        w.AddNorm("p.attn.norm.weight", dim); w.AddBias("p.attn.norm.bias", dim);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "proj_out" })
        {
            w.AddTensor($"p.attn.{n}.weight", s, dim, dim); w.AddBias($"p.attn.{n}.bias", dim);
        }
        using ChatterboxPerceiver perceiver = new();
        perceiver.LoadWeights(w, "p");
        Tensor prompt = Seq(o, 1321, 1f, 1, 11, dim);
        Tensor y = o.Own(perceiver.Forward(cpu, prompt));
        return Finish(o, ("out", y));
    }

    // ── AuK: the shared operator, the single block and the double block ─────────────────────────────────────────

    private const int AkDim = 16, AkHeads = 2, AkHd = AkDim / AkHeads;

    private static string AukAttend(CpuBackend cpu)
    {
        Owner o = new();
        Tensor q = Seq(o, 1401, 1f, 1, 7, AkHeads, AkHd), k = Seq(o, 1402, 1f, 1, 7, AkHeads, AkHd), v = Seq(o, 1403, 1f, 1, 7, AkHeads, AkHd);
        Tensor y = o.Own(AukOps.Attend(cpu, q, k, v, 7, AkHeads, AkHd));
        return Finish(o, ("out", y));
    }

    private static void AukAttnWeights(Weights w, string p, string qkv, string qn, string kn, string outName)
    {
        w.Linear($"{p}.attn.{qkv}", 3 * AkDim, AkDim, true);
        w.AddNorm($"{p}.attn.{qn}.weight", AkHd); w.AddNorm($"{p}.attn.{kn}.weight", AkHd);
        w.Linear($"{p}.attn.{outName}", AkDim, AkDim, true);
    }

    private static void AukFf(Weights w, string p)
    {
        w.Add($"{p}.linear_in.weight", 4 * AkDim, AkDim); w.Add($"{p}.linear_out.weight", AkDim, 2 * AkDim);
    }

    private static string AukSingle(CpuBackend cpu)
    {
        const int n = 8;
        Owner o = new();
        AukConfig cfg = new() { Dim = AkDim, Heads = AkHeads };
        Weights w = new(141);
        w.Linear("s.attn_norm.linear", 6 * AkDim, AkDim, true);
        AukAttnWeights(w, "s", "to_qkv", "q_norm", "k_norm", "to_out.0"); AukFf(w, "s.ff");
        AukSingleBlock block = new(cfg);
        block.LoadWeights(w, "s");
        (Tensor cos, Tensor sin) = AukRope.BuildTables(AukRope.TheoryInvFreq(AkHd), 32);
        o.Own(cos); o.Own(sin);
        Tensor x = Seq(o, 1411, 1f, 1, n, AkDim), time = Seq(o, 1412, 1f, 1, 1, AkDim);
        Tensor y = o.Own(block.Forward(cpu, x, time, n, cos, sin));
        return Finish(o, ("out", y));
    }

    private static string AukDouble(CpuBackend cpu)
    {
        const int n = 8, nt = 5;
        Owner o = new();
        AukConfig cfg = new() { Dim = AkDim, Heads = AkHeads };
        Weights w = new(142);
        w.Linear("d.attn_norm_x.linear", 6 * AkDim, AkDim, true); w.Linear("d.attn_norm_c.linear", 6 * AkDim, AkDim, true);
        AukAttnWeights(w, "d", "to_qkv", "q_norm", "k_norm", "to_out.0");
        w.Linear("d.attn.to_qkv_c", 3 * AkDim, AkDim, true); w.AddNorm("d.attn.c_q_norm.weight", AkHd); w.AddNorm("d.attn.c_k_norm.weight", AkHd);
        w.Linear("d.attn.to_out_c", AkDim, AkDim, true);
        AukFf(w, "d.ff_x"); AukFf(w, "d.ff_c");
        AukDoubleBlock block = new(cfg);
        block.LoadWeights(w, "d");
        (Tensor cos, Tensor sin) = AukRope.BuildTables(AukRope.TheoryInvFreq(AkHd), 32);
        o.Own(cos); o.Own(sin);
        Tensor x = Seq(o, 1421, 1f, 1, n, AkDim), c = Seq(o, 1422, 1f, 1, nt, AkDim), time = Seq(o, 1423, 1f, 1, 1, AkDim);
        (Tensor cOut, Tensor xOut) = block.Forward(cpu, x, c, time, n, nt, cos, sin);
        o.Own(cOut); o.Own(xOut);
        return Finish(o, ("audio", xOut), ("text", cOut));
    }

    // ── ControlFoley ────────────────────────────────────────────────────────────────────────────────────────────

    private static string ControlFoleyVit(CpuBackend cpu)
    {
        const int dim = 16, heads = 2, n = 7;
        Owner o = new();
        Tensor qkvW = Seq(o, 1501, 0.3f, 3 * dim, dim), qkvB = Seq(o, 1502, 0.1f, 3 * dim), projW = Seq(o, 1503, 0.3f, dim, dim), projB = Seq(o, 1504, 0.1f, dim);
        Tensor x1 = Seq(o, 1505, 1f, 1, n, dim), x2 = Seq(o, 1506, 1f, 2, n, dim);
        Tensor y1 = o.Own(ControlFoleyVitOps.SelfAttention(cpu, x1, qkvW, qkvB, projW, projB, heads));
        Tensor y2 = o.Own(ControlFoleyVitOps.SelfAttention(cpu, x2, qkvW, qkvB, projW, projB, heads));
        return Finish(o, ("b1", y1), ("b2", y2));
    }

    private static string ControlFoleyClip(CpuBackend cpu, bool causal)
    {
        const int width = 16, heads = 2, ratio = 2, n = 6;
        Owner o = new();
        Weights w = new(causal ? 151u : 152u);
        for (int i = 0; i < 2; i++)
        {
            string p = $"c.resblocks.{i}";
            w.AddNorm($"{p}.ln_1.weight", width); w.AddBias($"{p}.ln_1.bias", width);
            w.Add($"{p}.attn.in_proj_weight", 3 * width, width); w.AddBias($"{p}.attn.in_proj_bias", 3 * width);
            w.Linear($"{p}.attn.out_proj", width, width, true);
            w.AddNorm($"{p}.ln_2.weight", width); w.AddBias($"{p}.ln_2.bias", width);
            w.Linear($"{p}.mlp.c_fc", width * ratio, width, true); w.Linear($"{p}.mlp.c_proj", width, width * ratio, true);
        }
        ControlFoleyClipTransformer tr = new(w, "c", 2, width, heads, ratio, 1e-5f);
        Tensor x = Seq(o, 1510, 1f, n, width);
        float[] input = new ReadOnlySpan<float>(x.DataPointer, n * width).ToArray();
        float[] y = tr.Forward(cpu, input, n, causal);
        Finish(o);
        return Digest("out", y);
    }

    private static string ControlFoleyBlockAttend(CpuBackend cpu)
    {
        const int heads = 2, hd = 8, n = 7;
        Owner o = new();
        Tensor q = Seq(o, 1521, 1f, 2, heads, n, hd), k = Seq(o, 1522, 1f, 2, heads, n, hd), v = Seq(o, 1523, 1f, 2, heads, n, hd);
        Tensor y = o.Own(ControlFoleyBlock.Attend(cpu, q, k, v));
        return Finish(o, ("out", y));
    }

    private static string ControlFoleyVaeAttn(CpuBackend cpu)
    {
        // The single-head pixel-normed attention (D = channels) of the VAE decoder, through its private AttnBlock loader.
        const int c = 32, t = 9;
        Owner o = new();
        Weights w = new(153);
        w.AddTensor("a.qkv.weight", 0.3f, 3 * c, c, 1); w.AddTensor("a.proj_out.weight", 0.3f, c, c, 1);
        ControlFoleyVaeDecoder dec = new(new ControlFoleyVaeConfig { DataDim = 8, EmbedDim = 4, HiddenDim = 8 });
        object block = Invoke(dec, "Attn", w, "a", c);
        Tensor x = Seq(o, 1531, 1f, 1, c, t);
        Tensor y = o.Own((Tensor)Invoke(dec, "AttnForward", cpu, x, block));
        return Finish(o, ("out", y));
    }
}

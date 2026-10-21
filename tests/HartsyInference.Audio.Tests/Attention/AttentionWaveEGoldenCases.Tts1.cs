using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterTts1(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["dia_enc"] = cpu => DiaSelf(cpu, interleaved: false);
        d["dia_self_interleaved"] = cpu => DiaSelf(cpu, interleaved: true);
        d["dia_dec_host_steps"] = DiaDecoderHost;
        d["dia_cross"] = DiaCross;
        d["dia_flash_cpu"] = cpu => DiaFlash(cpu, interleaved: false);
        d["dia_flash_cpu_interleaved"] = cpu => DiaFlash(cpu, interleaved: true);
        d["mimi_forward_b1"] = cpu => MimiForward(cpu, 1, dsm: false);
        d["mimi_forward_b2"] = cpu => MimiForward(cpu, 2, dsm: false);
        d["mimi_forward_dsm_b1"] = cpu => MimiForward(cpu, 1, dsm: true);
        d["mimi_stream"] = cpu => MimiStream(cpu, dsm: false);
        d["mimi_stream_dsm"] = cpu => MimiStream(cpu, dsm: true);
        d["moshi_forward"] = cpu => MoshiForward(cpu, tq: 6, s: 5, salt: 61);
        d["moshi_cross_t1"] = cpu => MoshiForward(cpu, tq: 1, s: 9, salt: 62);
        d["moshi_step"] = MoshiStep;
        d["moshi_depformer"] = MoshiDepformerFrame;
    }

    // ── Dia (also Zonos' attention class) ──────────────────────────────────────────────────────────────────────

    private static void DiaWeightsFor(Weights w, string p, int qHeads, int kvHeads, int hd, int qIn, int kvIn, int outDim)
    {
        w.Add($"{p}.q_proj.weight", qHeads * hd, qIn);
        w.Add($"{p}.k_proj.weight", kvHeads * hd, kvIn);
        w.Add($"{p}.v_proj.weight", kvHeads * hd, kvIn);
        w.Add($"{p}.o_proj.weight", outDim, qHeads * hd);
    }

    private static (float[] Cos, float[] Sin) DiaTables(int hd) =>
        HartsyInference.Audio.Models.Moonshine.RotaryEmbedding.GetTables(hd, 10_000f, 32);

    private static string DiaSelf(CpuBackend cpu, bool interleaved)
    {
        Owner o = new();
        Weights w = new(interleaved ? 32u : 31u);
        DiaWeightsFor(w, "a", 2, 2, 8, 16, 16, 16);
        DiaAttention attn = new(2, 2, 8, 16, 16, 16, useRope: true, interleavedRope: interleaved, attnScale: interleaved ? 1f / MathF.Sqrt(8) : 1f);
        attn.LoadWeights(w, "a");
        (float[] cos, float[] sin) = DiaTables(8);
        Tensor x = Seq(o, 311, 1f, 1, 6, 16);
        Tensor y = o.Own(attn.SelfForward(cpu, x, 6, 0, null, 0, null, cos, sin));
        return Finish(o, ("out", y));
    }

    private static string DiaDecoderHost(CpuBackend cpu)
    {
        // GQA 4/2, scale 1.0, a causal prefill then single-token steps at posStart > 0 through the host cache.
        Owner o = new();
        Weights w = new(33);
        DiaWeightsFor(w, "a", 4, 2, 8, 16, 16, 16);
        DiaAttention attn = new(4, 2, 8, 16, 16, 16, useRope: true);
        attn.LoadWeights(w, "a");
        (float[] cos, float[] sin) = DiaTables(8);
        using StreamingKvCache cache = new(1, 1, 2, 16, 8);
        Tensor causal = o.Own(AdditiveMask(3, 3, (i, j) => j <= i));
        Tensor x0 = Seq(o, 331, 1f, 1, 3, 16), x1 = Seq(o, 332, 1f, 1, 1, 16), x2 = Seq(o, 333, 1f, 1, 1, 16);
        Tensor y0 = o.Own(attn.SelfForward(cpu, x0, 3, 0, cache, 0, causal, cos, sin));
        cache.AdvanceLength(3);
        Tensor y1 = o.Own(attn.SelfForward(cpu, x1, 1, 3, cache, 0, null, cos, sin));
        cache.AdvanceLength(1);
        Tensor y2 = o.Own(attn.SelfForward(cpu, x2, 1, 4, cache, 0, null, cos, sin));
        return Finish(o, ("prefill", y0), ("step3", y1), ("step4", y2));
    }

    private static string DiaCross(CpuBackend cpu)
    {
        Owner o = new();
        Weights w = new(34);
        DiaWeightsFor(w, "c", 2, 2, 8, 16, 12, 16);
        DiaAttention attn = new(2, 2, 8, 16, 12, 16, useRope: false);
        attn.LoadWeights(w, "c");
        Tensor enc = Seq(o, 341, 1f, 1, 7, 12);
        attn.PrecomputeCrossKv(cpu, enc, 7);
        Tensor x = Seq(o, 342, 1f, 1, 3, 16);
        Tensor y = o.Own(attn.CrossForward(cpu, x, 3));
        Tensor x1 = Seq(o, 343, 1f, 1, 1, 16);
        Tensor y1 = o.Own(attn.CrossForward(cpu, x1, 1));
        return Finish(o, ("t3", y), ("t1", y1));
    }

    /// <summary>Device-layout cos/sin <c>[1, t, headDim]</c>; split-half duplicates each table half, interleaved repeats each entry in a pair.</summary>
    private static (Tensor Cos, Tensor Sin) DiaDeviceRope(Owner o, int posStart, int t, int hd, bool interleaved)
    {
        (float[] cosT, float[] sinT) = DiaTables(hd);
        int half = hd / 2;
        Tensor cos = o.Own(new Tensor(new TensorShape(1, t, hd), DType.F32)), sin = o.Own(new Tensor(new TensorShape(1, t, hd), DType.F32));
        float* pc = (float*)cos.DataPointer, ps = (float*)sin.DataPointer;
        for (int p = 0; p < t; p++)
            for (int i = 0; i < half; i++)
            {
                float c = cosT[(posStart + p) * half + i], s = sinT[(posStart + p) * half + i];
                if (interleaved) { pc[p * hd + 2 * i] = pc[p * hd + 2 * i + 1] = c; ps[p * hd + 2 * i] = ps[p * hd + 2 * i + 1] = s; }
                else { pc[p * hd + i] = pc[p * hd + i + half] = c; ps[p * hd + i] = ps[p * hd + i + half] = s; }
            }
        return (cos, sin);
    }

    private static string DiaFlash(CpuBackend cpu, bool interleaved)
    {
        // The CPU Flash reference through SelfForwardFlash: GQA 4/2, scale 1.0, prefill then a step at posStart 3.
        Owner o = new();
        Weights w = new(interleaved ? 36u : 35u);
        DiaWeightsFor(w, "a", 4, 2, 8, 16, 16, 16);
        DiaAttention attn = new(4, 2, 8, 16, 16, 16, useRope: true, interleavedRope: interleaved);
        attn.LoadWeights(w, "a");
        using FixedKvCache cache = new(1, 1, 2, 8, 16);
        Tensor x0 = Seq(o, 351, 1f, 1, 3, 16), x1 = Seq(o, 352, 1f, 1, 1, 16);
        (Tensor c0, Tensor s0) = DiaDeviceRope(o, 0, 3, 8, interleaved);
        Tensor y0 = o.Own(attn.SelfForwardFlash(cpu, x0, 3, 0, cache, 0, c0, s0));
        cache.AdvanceLength(3);
        (Tensor c1, Tensor s1) = DiaDeviceRope(o, 3, 1, 8, interleaved);
        Tensor y1 = o.Own(attn.SelfForwardFlash(cpu, x1, 1, 3, cache, 0, c1, s1));
        return Finish(o, ("prefill", y0), ("step3", y1));
    }

    // ── Mimi transformer (HF split-half layout and the DSM/moshi interleaved layout) ───────────────────────────────

    private const int MmD = 16, MmFfn = 24, MmLayers = 2;

    private static MimiConfig MimiTiny() => new()
    {
        TransformerDim = MmD, TransformerLayers = MmLayers, TransformerHeads = 2, TransformerFfnDim = MmFfn, TransformerContext = 6,
    };

    private static Weights MimiWeights(uint salt, bool dsm)
    {
        Weights w = new(salt);
        for (int i = 0; i < MmLayers; i++)
        {
            if (dsm)
            {
                string p = $"m.transformer.layers.{i}";
                w.AddNorm($"{p}.norm1.weight", MmD); w.AddBias($"{p}.norm1.bias", MmD);
                w.AddNorm($"{p}.norm2.weight", MmD); w.AddBias($"{p}.norm2.bias", MmD);
                w.Add($"{p}.self_attn.in_proj_weight", 3 * MmD, MmD); w.Add($"{p}.self_attn.out_proj.weight", MmD, MmD);
                w.Add($"{p}.linear1.weight", MmFfn, MmD); w.Add($"{p}.linear2.weight", MmD, MmFfn);
                w.AddTensor($"{p}.layer_scale_1.scale", 0.5f, MmD); w.AddTensor($"{p}.layer_scale_2.scale", 0.5f, MmD);
            }
            else
            {
                string p = $"m.layers.{i}";
                w.AddNorm($"{p}.input_layernorm.weight", MmD); w.AddBias($"{p}.input_layernorm.bias", MmD);
                w.AddNorm($"{p}.post_attention_layernorm.weight", MmD); w.AddBias($"{p}.post_attention_layernorm.bias", MmD);
                foreach (string n in new[] { "q", "k", "v", "o" }) w.Add($"{p}.self_attn.{n}_proj.weight", MmD, MmD);
                w.Add($"{p}.mlp.fc1.weight", MmFfn, MmD); w.Add($"{p}.mlp.fc2.weight", MmD, MmFfn);
                w.AddTensor($"{p}.self_attn_layer_scale.scale", 0.5f, MmD); w.AddTensor($"{p}.mlp_layer_scale.scale", 0.5f, MmD);
            }
        }
        return w;
    }

    private static string MimiForward(CpuBackend cpu, int batch, bool dsm)
    {
        // t = 10 against a 6-key context, so the sliding window masks real keys.
        Owner o = new();
        MimiTransformer tr = new(MimiTiny(), "m");
        tr.LoadWeights(MimiWeights((uint)batch * 10u + (dsm ? 41u : 40u), dsm));
        Tensor x = Seq(o, 410 + (uint)batch, 1f, batch, 10, MmD);
        Tensor y = o.Own(tr.Forward(cpu, x, batch, 10));
        return Finish(o, ("out", y));
    }

    private static string MimiStream(CpuBackend cpu, bool dsm)
    {
        Owner o = new();
        MimiTransformer tr = new(MimiTiny(), "m");
        tr.LoadWeights(MimiWeights(dsm ? 43u : 42u, dsm));
        using MimiTransformerCache cache = new();
        List<(string, Tensor)> outs = [];
        int[] chunks = [4, 3, 5];
        for (int i = 0; i < chunks.Length; i++)
        {
            Tensor x = Seq(o, 420 + (uint)i, 1f, 1, chunks[i], MmD);
            outs.Add(($"chunk{i}", o.Own(tr.ForwardStreaming(cpu, x, chunks[i], cache))));
        }
        return Finish(o, [.. outs]);
    }

    // ── Moshi (fixed 2048-wide backbone, one layer; the depformer through its layer-count seam) ─────────────────

    private const int MoD = MoshiTransformer.Dim, MoInner = MoshiTransformer.GateInner;

    private static Weights MoshiWeights(uint salt)
    {
        const float s = 0.04f;
        Weights w = new(salt);
        w.AddTensor("t.layers.0.self_attn.in_proj_weight", s, 3 * MoD, MoD);
        w.AddTensor("t.layers.0.self_attn.out_proj.weight", s, MoD, MoD);
        w.AddTensor("t.layers.0.cross_attention.in_proj_weight", s, 3 * MoD, MoD);
        w.AddTensor("t.layers.0.cross_attention.out_proj.weight", s, MoD, MoD);
        w.AddTensor("t.layers.0.norm1.alpha", 0.25f, MoD);
        w.AddTensor("t.layers.0.norm2.alpha", 0.25f, MoD);
        w.AddNorm("t.layers.0.norm_cross.weight", MoD); w.AddBias("t.layers.0.norm_cross.bias", MoD);
        w.AddTensor("t.layers.0.gating.linear_in.weight", s, 2 * MoInner, MoD);
        w.AddTensor("t.layers.0.gating.linear_out.weight", s, MoD, MoInner);
        w.AddTensor("out_norm.alpha", 0.25f, MoD);
        return w;
    }

    private static string MoshiForward(CpuBackend cpu, int tq, int s, uint salt)
    {
        Owner o = new();
        using MoshiTransformer tr = new(layers: 1);
        tr.LoadWeights(MoshiWeights(salt), "t");
        Tensor input = Seq(o, 610 + salt, 1f, 1, tq, MoD), cross = Seq(o, 620 + salt, 1f, 1, s, MoD);
        Tensor y = o.Own(tr.Forward(cpu, input, cross));
        return Finish(o, ("out", y));
    }

    private static string MoshiStep(CpuBackend cpu)
    {
        // Three single-token steps through the FixedKvCache + FlashAttention path, with precomputed cross K/V.
        Owner o = new();
        using MoshiTransformer tr = new(layers: 1);
        tr.LoadWeights(MoshiWeights(63), "t");
        Tensor cross = Seq(o, 630, 1f, 1, 5, MoD);
        using MoshiTransformer.CrossKvCache crossKv = tr.PrecomputeCrossKv(cpu, cross);
        using FixedKvCache cache = new(1, 1, MoshiTransformer.Heads, MoshiTransformer.HeadDim, 8);
        List<(string, Tensor)> outs = [];
        for (int pos = 0; pos < 3; pos++)
        {
            Tensor frame = Seq(o, 640 + (uint)pos, 1f, 1, 1, MoD);
            outs.Add(($"step{pos}", o.Own(tr.StepForward(cpu, frame, crossKv, cache, pos))));
        }
        return Finish(o, [.. outs]);
    }

    private static string MoshiDepformerFrame(CpuBackend cpu)
    {
        // One depformer layer through the 32-codebook depth decode: causal FlashAttention over a growing depth cache.
        const int dim = MoshiDepformer.Dim, set = MoshiDepformer.Sets, depQ = MoshiDepformer.DepQ, inner = MoshiDepformer.GateInner;
        const float s = 0.04f;
        Owner o = new();
        Weights w = new(64);
        for (int i = 0; i < set; i++) w.AddTensor($"depformer_in.{i}.weight", s, dim, MoshiDepformer.MainDim);
        for (int k = 0; k < depQ - 1; k++)
        {
            w.AddTensor($"depformer_emb.{k}.weight", 0.5f, MoshiDepformer.Card + 1, MoshiDepformer.LowRank);
            w.AddTensor($"depformer_emb.{k}.low_rank.weight", s, dim, MoshiDepformer.LowRank);
        }
        w.AddTensor("depformer_text_emb.weight", 0.5f, MoshiDepformer.TextCard + 1, MoshiDepformer.LowRank);
        w.AddTensor("depformer_text_emb.out1.weight", s, dim, MoshiDepformer.LowRank);
        w.AddTensor("depformer_text_emb.out2.weight", s, dim, MoshiDepformer.LowRank);
        w.AddTensor("depformer.layers.0.self_attn.in_proj_weight", s, set * 3 * dim, dim);
        w.AddTensor("depformer.layers.0.self_attn.out_proj.weight", s, set * dim, dim);
        w.AddTensor("depformer.layers.0.norm1.alpha", 0.25f, dim);
        w.AddTensor("depformer.layers.0.norm2.alpha", 0.25f, dim);
        for (int i = 0; i < set; i++)
        {
            w.AddTensor($"depformer.layers.0.gating.{i}.linear_in.weight", s, 2 * inner, dim);
            w.AddTensor($"depformer.layers.0.gating.{i}.linear_out.weight", s, dim, inner);
        }
        for (int cb = 0; cb < depQ; cb++) w.AddTensor($"linears.{cb}.weight", s, MoshiDepformer.Card, dim);
        MoshiDepformer dep = new(layers: 1);
        dep.LoadWeights(w);
        Tensor ctx = Seq(o, 650, 1f, 1, 1, MoshiDepformer.MainDim);
        Tensor logits = o.Own(dep.DecodeFrameGreedy(cpu, ctx, 5, out int[] tokens));
        Tensor tok = o.Own(new Tensor(new TensorShape(depQ), DType.F32));
        for (int i = 0; i < depQ; i++) ((float*)tok.DataPointer)[i] = tokens[i];
        return Finish(o, ("logits", logits), ("tokens", tok));
    }
}

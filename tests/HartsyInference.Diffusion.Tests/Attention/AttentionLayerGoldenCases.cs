using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>Forward passes of the attention blocks (LTX-2, Krea2, Z-Image, Lumina2, OmniGen2, Boogu, Lance) on tiny
/// pseudo-random weights and inputs, each reduced to SHA-256 digests of its output bits. Everything runs on the CPU backend
/// through the blocks' public constructor / LoadWeights / Forward, so the same cases produce the golden fixture from the
/// pre-migration source and must reproduce it after the attention refactor.</summary>
internal static unsafe class AttentionLayerGoldenCases
{
    private const float NegInf = -1e9f;

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "ltx2_self_gated_split", "ltx2_self_ungated_split", "ltx2_self_gated_interleaved", "ltx2_self_ungated_interleaved",
        "ltx2_self_gated_rope_null", "ltx2_cross_gated_text", "ltx2_cross_ungated_text_masked", "ltx2_cross_modal_split_two_ropes",
        "ltx2_cross_modal_interleaved_two_ropes", "ltx2_self_split_masked",
        "krea2_gqa_4_2_rope", "krea2_gqa_4_1_rope", "krea2_mha_rope", "krea2_gqa_rope_vrowscale", "krea2_gqa_rope_attnbias",
        "krea2_gqa_rope_batch2", "krea2_text_fusion_attn_no_rope", "krea2_text_fusion_block",
        "zimage_turbo_rope", "zimage_base_rope", "zimage_turbo_rope_batch2", "zimage_base_bias_no_rope",
        "zimage_ctx_refiner_turbo", "zimage_ctx_refiner_base", "zimage_ctx_refiner_batch2", "zimage_ctx_refiner_no_rope",
        "lumina2_gqa_rope", "lumina2_mha_rope", "lumina2_gqa_no_rope",
        "omnigen2_modulated_joint", "omnigen2_modulated_image", "omnigen2_modulated_text", "omnigen2_plain_text",
        "omnigen2_modulated_joint_mha",
        "boogu_single_modulated_joint", "boogu_single_plain_text",
        "lance_mot_mask_qknorm", "lance_mot_nomask_qknorm", "lance_mot_mask_noqknorm", "lance_mot_mask_no_und_prefix",
    ];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name)
    {
        using CpuBackend cpu = new();
        return name switch
        {
            _ when name.StartsWith("ltx2_", StringComparison.Ordinal) => Ltx2(cpu, name),
            _ when name.StartsWith("krea2_", StringComparison.Ordinal) => Krea2(cpu, name),
            _ when name.StartsWith("zimage_ctx_", StringComparison.Ordinal) => ZImageContext(cpu, name),
            _ when name.StartsWith("zimage_", StringComparison.Ordinal) => ZImage(cpu, name),
            _ when name.StartsWith("lumina2_", StringComparison.Ordinal) => Lumina2(cpu, name),
            _ when name.StartsWith("omnigen2_", StringComparison.Ordinal) => OmniGen2(cpu, name),
            _ when name.StartsWith("boogu_", StringComparison.Ordinal) => Boogu(cpu, name),
            _ when name.StartsWith("lance_", StringComparison.Ordinal) => Lance(cpu, name),
            _ => throw new ArgumentException($"Unknown attention golden case '{name}'.", nameof(name)),
        };
    }

    // ── LTX-2 ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private const int LtxHeads = 4, LtxHeadDim = 8, LtxInner = LtxHeads * LtxHeadDim;
    private const int LtxFrames = 2, LtxGridH = 2, LtxGridW = 3, LtxVideoSeq = LtxFrames * LtxGridH * LtxGridW;
    private const int LtxTextSeq = 7, LtxTextDim = 24, LtxAudioSeq = 5;

    private static string Ltx2(CpuBackend cpu, string name)
    {
        bool gated = !name.Contains("ungated", StringComparison.Ordinal);
        bool interleaved = name.Contains("interleaved", StringComparison.Ordinal);
        LtxVideo2Rope.RopeType flavor = interleaved ? LtxVideo2Rope.RopeType.Interleaved : LtxVideo2Rope.RopeType.Split;
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }

        LtxVideo2Rope VideoSelf() => LtxVideo2Rope.ForVideoSelf(LtxInner, 10000.0, 20, 64, 64, [8, 32, 32], 1, flavor, LtxHeads, LtxHeadDim);
        // Cross-modal ropes: video side temporal-only, audio side its own 1-axis table, same inner width.
        LtxVideo2Rope VideoCross() => LtxVideo2Rope.ForVideoCross(LtxInner, 10000.0, 20, 64, 64, [8, 32, 32], 1, flavor, LtxHeads, LtxHeadDim);
        LtxVideo2Rope Audio() => LtxVideo2Rope.ForAudio(LtxInner, 10000.0, 20, 4, 0, 16000, 160, flavor, LtxHeads, LtxHeadDim);

        switch (name)
        {
            case "ltx2_self_gated_split":
            case "ltx2_self_ungated_split":
            case "ltx2_self_gated_interleaved":
            case "ltx2_self_ungated_interleaved":
            case "ltx2_self_split_masked":
            {
                LtxVideo2Attention attn = Ltx2Block(LtxInner, LtxInner, gated, 11);
                LtxVideo2Rope rope = VideoSelf();
                (Tensor cos, Tensor sin) = rope.BuildVideo(LtxFrames, LtxGridH, LtxGridW, 24.0);
                Own(cos); Own(sin);
                Tensor x = Own(Seeded(new TensorShape(LtxVideoSeq, LtxInner), 101));
                Tensor? mask = name.EndsWith("masked", StringComparison.Ordinal)
                    ? Own(AdditiveMask(LtxVideoSeq, LtxVideoSeq, (i, j) => j <= i + 2)) : null;
                Tensor o = Own(attn.Forward(cpu, x, x, rope, cos, sin, rope, cos, sin, mask));
                return Finish(owned, ("out", o));
            }
            case "ltx2_self_gated_rope_null":
            {
                LtxVideo2Attention attn = Ltx2Block(LtxInner, LtxInner, gated: true, 12);
                Tensor x = Own(Seeded(new TensorShape(LtxVideoSeq, LtxInner), 102));
                Tensor o = Own(attn.Forward(cpu, x, x, null, null, null, null, null, null, null));
                return Finish(owned, ("out", o));
            }
            case "ltx2_cross_gated_text":
            case "ltx2_cross_ungated_text_masked":
            {
                // Different q/kv widths (text cross-attention, no RoPE); the masked case blanks the last two text tokens.
                LtxVideo2Attention attn = Ltx2Block(LtxInner, LtxTextDim, gated, 13, outDim: 40);
                Tensor x = Own(Seeded(new TensorShape(LtxVideoSeq, LtxInner), 103));
                Tensor text = Own(Seeded(new TensorShape(LtxTextSeq, LtxTextDim), 104));
                Tensor? mask = name.Contains("masked", StringComparison.Ordinal)
                    ? Own(AdditiveMask(LtxVideoSeq, LtxTextSeq, (_, j) => j < LtxTextSeq - 2)) : null;
                Tensor o = Own(attn.Forward(cpu, x, text, null, null, null, null, null, null, mask));
                return Finish(owned, ("out", o));
            }
            case "ltx2_cross_modal_split_two_ropes":
            case "ltx2_cross_modal_interleaved_two_ropes":
            {
                // a2v: video queries (video-cross rope) attend to audio keys (audio rope).
                LtxVideo2Attention attn = Ltx2Block(LtxInner, LtxInner, gated: true, 14);
                LtxVideo2Rope qRope = VideoCross(), kRope = Audio();
                (Tensor qCos, Tensor qSin) = qRope.BuildVideo(LtxFrames, LtxGridH, LtxGridW, 24.0);
                (Tensor kCos, Tensor kSin) = kRope.BuildAudio(LtxAudioSeq);
                Own(qCos); Own(qSin); Own(kCos); Own(kSin);
                Tensor x = Own(Seeded(new TensorShape(LtxVideoSeq, LtxInner), 105));
                Tensor audio = Own(Seeded(new TensorShape(LtxAudioSeq, LtxInner), 106));
                Tensor o = Own(attn.Forward(cpu, x, audio, qRope, qCos, qSin, kRope, kCos, kSin, null));
                return Finish(owned, ("out", o));
            }
            // No F16-query / F32-kv case: the CPU backend is F32-only (LinearTransB rejects F16 activations).
            default:
                throw new ArgumentException($"Unknown LTX-2 case '{name}'.", nameof(name));
        }
    }

    private static LtxVideo2Attention Ltx2Block(int qIn, int kvIn, bool gated, uint salt, int? outDim = null)
    {
        int od = outDim ?? qIn;
        LtxVideo2Attention attn = new(qIn, kvIn, LtxHeads, LtxHeadDim, od, 1e-6f);
        Weights w = new(salt);
        w.Add("a.to_q.weight", LtxInner, qIn); w.AddBias("a.to_q.bias", LtxInner);
        w.Add("a.to_k.weight", LtxInner, kvIn); w.AddBias("a.to_k.bias", LtxInner);
        w.Add("a.to_v.weight", LtxInner, kvIn); w.AddBias("a.to_v.bias", LtxInner);
        w.Add("a.to_out.0.weight", od, LtxInner); w.AddBias("a.to_out.0.bias", od);
        w.AddNorm("a.q_norm.weight", LtxInner);
        w.AddNorm("a.k_norm.weight", LtxInner);
        if (gated) { w.Add("a.to_gate_logits.weight", LtxHeads, qIn); w.AddBias("a.to_gate_logits.bias", LtxHeads); }
        attn.LoadWeights(w, "a");
        return attn;
    }

    // ── Krea2 ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private const int KHidden = 32, KHeads = 4, KHeadDim = 8, KText = 3, KGridH = 2, KGridW = 3, KSeq = KText + KGridH * KGridW;

    private static string Krea2(CpuBackend cpu, string name)
    {
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }
        switch (name)
        {
            case "krea2_gqa_4_2_rope":
            case "krea2_gqa_4_1_rope":
            case "krea2_mha_rope":
            case "krea2_gqa_rope_vrowscale":
            case "krea2_gqa_rope_attnbias":
            case "krea2_gqa_rope_batch2":
            {
                int kv = name switch { "krea2_gqa_4_1_rope" => 1, "krea2_mha_rope" => 4, _ => 2 };
                int batch = name.EndsWith("batch2", StringComparison.Ordinal) ? 2 : 1;
                Krea2Attention attn = Krea2Block(KHidden, KHeads, kv, 21);
                FluxRope rope = new([2, 2, 4], 1000);
                Tensor ids = FluxRope.BuildPositionIds(KText, KGridH, KGridW);
                rope.Precompute(ids);
                ids.Dispose();
                Tensor x = Own(Seeded(new TensorShape(batch, KSeq, KHidden), 201));
                Tensor? bias = name.EndsWith("attnbias", StringComparison.Ordinal)
                    ? Own(AdditiveMask(KSeq, KSeq, (i, j) => j <= i + 3)) : null;
                Tensor? vScale = null;
                if (name.EndsWith("vrowscale", StringComparison.Ordinal))
                {
                    // Per-token weights (SwarmUI token weighting) expanded across the kv heads.
                    vScale = Own(new Tensor(new TensorShape(batch, KSeq, kv, KHeadDim), DType.F32));
                    float* p = (float*)vScale.DataPointer;
                    for (int s = 0; s < KSeq; s++)
                        for (int d = 0; d < kv * KHeadDim; d++) p[s * kv * KHeadDim + d] = 0.5f + 0.125f * (s % 5);
                }
                Tensor o = Own(attn.Forward(cpu, x, rope, batch, KSeq, bias, vScale));
                string digest = Finish(owned, ("out", o));
                (rope as IDisposable)?.Dispose();
                return digest;
            }
            case "krea2_text_fusion_attn_no_rope":
            {
                // Text fusion calls the attention with rope null, full MHA, over [B·S, layers, dim] stacks.
                Krea2Attention attn = Krea2Block(KHidden, KHeads, KHeads, 22);
                Tensor x = Own(Seeded(new TensorShape(6, 5, KHidden), 202));
                Tensor o = Own(attn.Forward(cpu, x, null, 6, 5));
                return Finish(owned, ("out", o));
            }
            case "krea2_text_fusion_block":
            {
                Krea2TextFusionBlock block = new(KHidden, KHeads, KHeads, 48, 1e-5f);
                Weights w = new(23);
                w.AddNorm("b.norm1.weight", KHidden, zeroCentered: true);
                w.AddNorm("b.norm2.weight", KHidden, zeroCentered: true);
                AddKrea2AttnWeights(w, "b.attn", KHidden, KHeads, KHeads);
                w.Add("b.ff.gate.weight", 48, KHidden);
                w.Add("b.ff.up.weight", 48, KHidden);
                w.Add("b.ff.down.weight", KHidden, 48);
                block.LoadWeights(w, "b");
                Tensor x = Own(Seeded(new TensorShape(3, 6, KHidden), 203));
                Tensor o = Own(block.Forward(cpu, x, 3, 6));
                return Finish(owned, ("out", o));
            }
            default:
                throw new ArgumentException($"Unknown Krea2 case '{name}'.", nameof(name));
        }
    }

    private static Krea2Attention Krea2Block(int hidden, int heads, int kvHeads, uint salt)
    {
        Krea2Attention attn = new(hidden, heads, kvHeads);
        Weights w = new(salt);
        AddKrea2AttnWeights(w, "a", hidden, heads, kvHeads);
        attn.LoadWeights(w, "a");
        return attn;
    }

    private static void AddKrea2AttnWeights(Weights w, string p, int hidden, int heads, int kvHeads)
    {
        int headDim = hidden / heads;
        w.Add($"{p}.to_q.weight", hidden, hidden);
        w.Add($"{p}.to_k.weight", kvHeads * headDim, hidden);
        w.Add($"{p}.to_v.weight", kvHeads * headDim, hidden);
        w.Add($"{p}.to_gate.weight", hidden, hidden);
        w.Add($"{p}.to_out.0.weight", hidden, hidden);
        w.AddNorm($"{p}.norm_q.weight", headDim, zeroCentered: true);
        w.AddNorm($"{p}.norm_k.weight", headDim, zeroCentered: true);
    }

    // ── Z-Image ────────────────────────────────────────────────────────────────────────────────────────────────────

    private const int ZHidden = 32, ZHeads = 4, ZHeadDim = 8, ZFfn = 48, ZEmbed = 16, ZCap = 3, ZGridH = 2, ZGridW = 3;

    private static ZImageRope ZRope(Tensor ids)
    {
        ZImageRope rope = new([2, 2, 4], 256f);
        rope.Precompute(ids);
        ids.Dispose();
        return rope;
    }

    private static string ZImage(CpuBackend cpu, string name)
    {
        bool allowF16 = name.StartsWith("zimage_turbo", StringComparison.Ordinal);
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal);
        bool noRope = name.EndsWith("no_rope", StringComparison.Ordinal);
        int seq = ZCap + ZGridH * ZGridW;
        int batch = batch2 ? 2 : 1;
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }

        ZImageBlock block = new(ZHidden, ZHeads, ZFfn, 1e-5f, useF16SandwichDamp: false, allowF16Attention: allowF16);
        Weights w = new(31);
        w.Add("b.adaLN_modulation.0.weight", 4 * ZHidden, ZEmbed); w.AddBias("b.adaLN_modulation.0.bias", 4 * ZHidden);
        AddZImageCommon(w, "b");
        block.LoadWeights(w, "b");

        ZImageRope? rope = noRope ? null : ZRope(ZImageRope.BuildPositionIds(ZCap, ZGridH, ZGridW, ZGridH * ZGridW));
        Tensor x = Own(Seeded(new TensorShape(batch, seq, ZHidden), 301));
        Tensor tEmb = Own(Seeded(new TensorShape(batch, ZEmbed), 302));
        Tensor? bias = name.Contains("bias", StringComparison.Ordinal) ? Own(AdditiveMask(seq, seq, (i, j) => j <= i + 2)) : null;
        Tensor o = Own(block.Forward(cpu, x, tEmb, rope, bias));
        string digest = Finish(owned, ("out", o));
        rope?.ReleaseGpuTables(cpu);
        return digest;
    }

    private static string ZImageContext(CpuBackend cpu, string name)
    {
        bool allowF16 = name.EndsWith("turbo", StringComparison.Ordinal);
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal);
        bool noRope = name.EndsWith("no_rope", StringComparison.Ordinal);
        const int cap = 5;
        int batch = batch2 ? 2 : 1;
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }

        ZImageContextRefinerBlock block = new(ZHidden, ZHeads, ZFfn, 1e-5f, allowF16);
        Weights w = new(32);
        AddZImageCommon(w, "b");
        block.LoadWeights(w, "b");

        ZImageRope? rope = noRope ? null : ZRope(ZImageRope.BuildCaptionPositionIds(cap));
        Tensor x = Own(Seeded(new TensorShape(batch, cap, ZHidden), 311));
        Tensor o = Own(block.Forward(cpu, x, rope));
        string digest = Finish(owned, ("out", o));
        rope?.ReleaseGpuTables(cpu);
        return digest;
    }

    private static void AddZImageCommon(Weights w, string p)
    {
        w.Add($"{p}.attention.qkv.weight", 3 * ZHidden, ZHidden);
        w.Add($"{p}.attention.out.weight", ZHidden, ZHidden);
        w.AddNorm($"{p}.attention.q_norm.weight", ZHeadDim);
        w.AddNorm($"{p}.attention.k_norm.weight", ZHeadDim);
        foreach (string n in new[] { "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2" })
            w.AddNorm($"{p}.{n}.weight", ZHidden);
        w.Add($"{p}.feed_forward.w1.weight", ZFfn, ZHidden);
        w.Add($"{p}.feed_forward.w2.weight", ZHidden, ZFfn);
        w.Add($"{p}.feed_forward.w3.weight", ZFfn, ZHidden);
    }

    // ── Lumina2 ────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string Lumina2(CpuBackend cpu, string name)
    {
        int kv = name.Contains("mha", StringComparison.Ordinal) ? ZHeads : 2;
        int seq = ZCap + ZGridH * ZGridW;
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }

        Lumina2Block block = new(ZHidden, ZHeads, kv, ZHeadDim, ZFfn, ZEmbed);
        Weights w = new(41);
        w.AddNorm("b.norm1.norm.weight", ZHidden);
        w.Add("b.norm1.linear.weight", 4 * ZHidden, ZEmbed); w.AddBias("b.norm1.linear.bias", 4 * ZHidden);
        w.Add("b.attn.to_q.weight", ZHeads * ZHeadDim, ZHidden);
        w.Add("b.attn.to_k.weight", kv * ZHeadDim, ZHidden);
        w.Add("b.attn.to_v.weight", kv * ZHeadDim, ZHidden);
        w.Add("b.attn.to_out.0.weight", ZHidden, ZHeads * ZHeadDim);
        w.AddNorm("b.attn.norm_q.weight", ZHeadDim);
        w.AddNorm("b.attn.norm_k.weight", ZHeadDim);
        w.AddNorm("b.norm2.weight", ZHidden);
        w.AddNorm("b.ffn_norm1.weight", ZHidden);
        w.AddNorm("b.ffn_norm2.weight", ZHidden);
        w.Add("b.feed_forward.linear_1.weight", ZFfn, ZHidden);
        w.Add("b.feed_forward.linear_2.weight", ZHidden, ZFfn);
        w.Add("b.feed_forward.linear_3.weight", ZFfn, ZHidden);
        block.LoadWeights(w, "b");

        ZImageRope? rope = name.EndsWith("no_rope", StringComparison.Ordinal)
            ? null : ZRope(ZImageRope.BuildPositionIds(ZCap, ZGridH, ZGridW, ZGridH * ZGridW));
        Tensor x = Own(Seeded(new TensorShape(1, seq, ZHidden), 401));
        Tensor tEmb = Own(Seeded(new TensorShape(1, ZEmbed), 402));
        Tensor o = Own(block.Forward(cpu, x, tEmb, rope));
        string digest = Finish(owned, ("out", o));
        rope?.ReleaseGpuTables(cpu);
        return digest;
    }

    // ── OmniGen2 / Boogu (shared Lumina-style weights) ─────────────────────────────────────────────────────────────

    private const int OHidden = 48, OHeads = 6, OKvHeads = 2, OHeadDim = 8, OFfn = 64, OCond = 16, OText = 3, OGridH = 2, OGridW = 3;

    private static Weights OmniWeights(bool modulation, uint salt)
    {
        Weights w = new(salt);
        w.Add("b.attn.to_q.weight", OHidden, OHidden);
        w.Add("b.attn.to_k.weight", OKvHeads * OHeadDim, OHidden);
        w.Add("b.attn.to_v.weight", OKvHeads * OHeadDim, OHidden);
        w.Add("b.attn.to_out.0.weight", OHidden, OHidden);
        w.AddNorm("b.attn.norm_q.weight", OHeadDim);
        w.AddNorm("b.attn.norm_k.weight", OHeadDim);
        w.Add("b.feed_forward.linear_1.weight", OFfn, OHidden);
        w.Add("b.feed_forward.linear_2.weight", OHidden, OFfn);
        w.Add("b.feed_forward.linear_3.weight", OFfn, OHidden);
        if (modulation)
        {
            w.Add("b.norm1.linear.weight", 4 * OHidden, OCond); w.AddBias("b.norm1.linear.bias", 4 * OHidden);
            w.AddNorm("b.norm1.norm.weight", OHidden);
        }
        else
        {
            w.AddNorm("b.norm1.weight", OHidden);
        }
        w.AddNorm("b.norm2.weight", OHidden);
        w.AddNorm("b.ffn_norm1.weight", OHidden);
        w.AddNorm("b.ffn_norm2.weight", OHidden);
        return w;
    }

    private static string OmniGen2(CpuBackend cpu, string name)
    {
        bool modulation = !name.Contains("plain", StringComparison.Ordinal);
        int kv = name.EndsWith("mha", StringComparison.Ordinal) ? OHeads : OKvHeads;
        int imgTokens = OGridH * OGridW;
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }

        OmniGen2Block block = new(OHidden, OHeads, kv, OHeadDim, OFfn, OCond, modulation);
        Weights w = OmniWeights(modulation, 51);
        if (kv != OKvHeads)
        {
            w.Replace("b.attn.to_k.weight", kv * OHeadDim, OHidden);
            w.Replace("b.attn.to_v.weight", kv * OHeadDim, OHidden);
        }
        block.LoadWeights(w, "b");
        using OmniGen2Rope rope = new([2, 2, 4]);
        Tensor? temb = modulation ? Own(Seeded(new TensorShape(1, OCond), 502)) : null;

        (RopeApplyMode mode, int seq) = name switch
        {
            _ when name.Contains("joint", StringComparison.Ordinal) => (RopeApplyMode.Joint, OText + imgTokens),
            _ when name.Contains("image", StringComparison.Ordinal) => (RopeApplyMode.Image, imgTokens),
            _ => (RopeApplyMode.Text, 5),
        };
        Tensor x = Own(Seeded(new TensorShape(1, seq, OHidden), 501));
        Tensor o = Own(block.Forward(cpu, x, rope, mode, OGridH, OGridW, mode == RopeApplyMode.Image ? OText : 0, temb));
        return Finish(owned, ("out", o));
    }

    private static string Boogu(CpuBackend cpu, string name)
    {
        bool modulation = !name.Contains("plain", StringComparison.Ordinal);
        int imgTokens = OGridH * OGridW;
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }

        BooguImageSingleBlock block = new(OHidden, OHeads, OKvHeads, OHeadDim, OFfn, OCond, modulation);
        block.LoadWeights(OmniWeights(modulation, 61), "b");
        using OmniGen2Rope rope = new([2, 2, 4]);
        (float[] cosT, float[] sinT) = modulation ? rope.BuildJointTable(OText, OGridH, OGridW) : rope.BuildTextTable(5);
        int seq = modulation ? OText + imgTokens : 5;
        (Tensor cos, Tensor sin) = rope.ExpandToDeviceTables(cosT, sinT);
        Own(cos); Own(sin);
        Tensor? temb = modulation ? Own(Seeded(new TensorShape(1, OCond), 602)) : null;
        Tensor x = Own(Seeded(new TensorShape(1, seq, OHidden), 601));
        Tensor o = Own(block.Forward(cpu, x, cos, sin, temb));
        return Finish(owned, ("out", o));
    }

    // ── Lance ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private const int LHidden = 32, LHeads = 4, LKvHeads = 2, LHeadDim = 8, LFfn = 48, LSeq = 9;

    private static string Lance(CpuBackend cpu, string name)
    {
        bool qkNorm = !name.EndsWith("noqknorm", StringComparison.Ordinal);
        bool masked = !name.Contains("nomask", StringComparison.Ordinal);
        int nPre = name.EndsWith("no_und_prefix", StringComparison.Ordinal) ? 0 : 3;
        int nGen = nPre == 0 ? 5 : 4;
        List<Tensor> owned = [];
        Tensor Own(Tensor t) { owned.Add(t); return t; }

        LanceMoTBlock block = new(LHidden, LHeads, LKvHeads, LFfn, 1e-6f, qkNorm);
        Weights w = new(71);
        foreach (string suffix in new[] { "", "_moe_gen" })
        {
            string at = $"b.self_attn.";
            w.Add($"{at}q_proj{suffix}.weight", LHidden, LHidden); w.AddBias($"{at}q_proj{suffix}.bias", LHidden);
            w.Add($"{at}k_proj{suffix}.weight", LKvHeads * LHeadDim, LHidden); w.AddBias($"{at}k_proj{suffix}.bias", LKvHeads * LHeadDim);
            w.Add($"{at}v_proj{suffix}.weight", LKvHeads * LHeadDim, LHidden); w.AddBias($"{at}v_proj{suffix}.bias", LKvHeads * LHeadDim);
            w.Add($"{at}o_proj{suffix}.weight", LHidden, LHidden);
            string mlp = suffix.Length == 0 ? "b.mlp" : "b.mlp_moe_gen";
            w.Add($"{mlp}.gate_proj.weight", LFfn, LHidden);
            w.Add($"{mlp}.up_proj.weight", LFfn, LHidden);
            w.Add($"{mlp}.down_proj.weight", LHidden, LFfn);
            w.AddNorm($"b.input_layernorm{suffix}.weight", LHidden);
            w.AddNorm($"b.post_attention_layernorm{suffix}.weight", LHidden);
            if (qkNorm)
            {
                w.AddNorm($"{at}q_norm{suffix}.weight", LHeadDim);
                w.AddNorm($"{at}k_norm{suffix}.weight", LHeadDim);
            }
        }
        block.LoadWeights(w, "b");

        // M-RoPE cos/sin [seq, headDim] from mixed text/latent positions.
        Tensor ids = new(new TensorShape(LSeq, 3), DType.F32);
        float* pid = (float*)ids.DataPointer;
        for (int i = 0; i < LSeq; i++)
        {
            bool latent = i >= nPre && i < nPre + nGen;
            pid[3 * i] = latent ? nPre + (i - nPre) / 2 : i;
            pid[3 * i + 1] = latent ? nPre + (i - nPre) % 2 : i;
            pid[3 * i + 2] = latent ? nPre + (i - nPre) % 3 : i;
        }
        (Tensor cos, Tensor sin) = new Multimodal3DRope(LHeadDim, 1_000_000.0, (2, 1, 1)).BuildCosSin(ids);
        ids.Dispose();
        Own(cos); Own(sin);

        // Und rows are causal; gen rows see everything (the sparse additive mask of the gen pipelines).
        Tensor? mask = masked
            ? Own(AdditiveMask(LSeq, LSeq, (i, j) => (i >= nPre && i < nPre + nGen) || j <= i)) : null;
        Tensor x = Own(Seeded(new TensorShape(LSeq, LHidden), 701));
        Tensor o = Own(block.Forward(cpu, x, cos, sin, nPre, nGen, mask));
        return Finish(owned, ("out", o));
    }

    // ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Digests the outputs, then releases every tensor the case allocated.</summary>
    private static string Finish(List<Tensor> owned, params (string Label, Tensor Tensor)[] outputs)
    {
        foreach ((string label, Tensor t) in outputs)
            for (long i = 0; i < t.ElementCount; i++)
                if (!float.IsFinite(((float*)t.DataPointer)[i]))
                    throw new InvalidOperationException($"Non-finite value in '{label}' at {i}.");
        string digests = string.Join(' ', outputs.Select(entry => Hex(entry.Label, Bytes(entry.Tensor))));
        foreach (Tensor t in owned) t.Dispose();
        return digests;
    }

    private static ReadOnlySpan<byte> Bytes(Tensor t)
    {
        int size = t.DType == DType.F16 ? sizeof(ushort) : sizeof(float);
        return new ReadOnlySpan<byte>(t.DataPointer, checked((int)(t.ElementCount * size)));
    }

    private static string Hex(string label, ReadOnlySpan<byte> bytes) => $"{label}={Convert.ToHexStringLower(SHA256.HashData(bytes))}";

    /// <summary>Additive F32 mask <c>[1, 1, rows, cols]</c>: 0 where <paramref name="allowed"/>, a large negative elsewhere.</summary>
    private static Tensor AdditiveMask(int rows, int cols, Func<int, int, bool> allowed)
    {
        Tensor mask = new(new TensorShape(1, 1, rows, cols), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                p[i * cols + j] = allowed(i, j) ? 0f : NegInf;
        return mask;
    }

    private static Tensor Seeded(TensorShape shape, uint seed, float scale = 1f)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = seed * 2654435761u + 0x9E3779B9u;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * scale;
        }
        return tensor;
    }

    /// <summary>A checkpoint stand-in: each tensor is seeded from its own name (FNV-1a) plus a per-case salt.</summary>
    private sealed class Weights(uint salt) : Dictionary<string, Tensor>
    {
        public void Add(string name, long rows, long cols) => base.Add(name, Seeded(new TensorShape(rows, cols), Seed(name), 0.3f));

        public void Replace(string name, long rows, long cols) => this[name] = Seeded(new TensorShape(rows, cols), Seed(name), 0.3f);

        public void AddBias(string name, long n) => base.Add(name, Seeded(new TensorShape(n), Seed(name), 0.1f));

        /// <summary>Norm scale near 1 (near 0 when stored zero-centered, as Krea2's are).</summary>
        public void AddNorm(string name, long n, bool zeroCentered = false)
        {
            Tensor t = Seeded(new TensorShape(n), Seed(name), 0.25f);
            if (!zeroCentered)
            {
                float* p = (float*)t.DataPointer;
                for (long i = 0; i < n; i++) p[i] += 1f;
            }
            base.Add(name, t);
        }

        private uint Seed(string name)
        {
            uint h = 2166136261u ^ salt;
            foreach (char c in name) h = (h ^ c) * 16777619u;
            return h;
        }
    }
}

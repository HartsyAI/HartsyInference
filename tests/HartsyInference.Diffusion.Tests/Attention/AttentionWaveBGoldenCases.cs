using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>Forward passes of the Wave B attention blocks (Flux, Flux2, Flux3, Chroma, Qwen-Image, Qwen-Image 2.1, HunyuanImage,
/// HunyuanVideo, AuraFlow, SD3, HiDream, Ideogram4, Anima, Kandinsky5, Ernie, MOSS-Image, F-Lite) on tiny pseudo-random weights and
/// inputs, each reduced to SHA-256 digests of its output bits. Everything runs on the CPU backend through the blocks' public
/// constructor / LoadWeights / Forward, so the same cases produce the golden fixture from the pre-migration source and must
/// reproduce it after the attention refactor.</summary>
internal static unsafe class AttentionWaveBGoldenCases
{
    private const float NegInf = -1e9f;
    private const int Hidden = 32, Heads = 4, HeadDim = 8, Mlp = 64, Txt = 3, GridH = 2, GridW = 3, Img = GridH * GridW;
    private const int RefH = 2, RefW = 2, Refs = RefH * RefW;

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "flux_double_base", "flux_double_attnbias", "flux_double_nobias", "flux_double_kontext", "flux_double_batch2",
        "flux_single_base", "flux_single_attnbias", "flux_single_nobias", "flux_single_kontext", "flux_single_batch2",
        "flux2_double_base", "flux2_double_attnbias", "flux2_double_qkvbias", "flux2_double_batch2",
        "flux2_single_base", "flux2_single_attnbias", "flux2_single_qkvbias", "flux2_single_batch2",
        "flux3_block_uniform", "flux3_block_spans", "flux3_block_refs",
        "chroma_double_split", "chroma_double_fused", "chroma_double_mask", "chroma_double_batch2",
        "chroma_single_split", "chroma_single_fused", "chroma_single_mask", "chroma_single_batch2",
        "qwenimage_block_base", "qwenimage_block_refs", "qwenimage_block_tembzero", "qwenimage_block_batch2",
        "qwenimage21_block_prefix", "qwenimage21_block_target_with_prefix", "qwenimage21_block_target_no_prefix",
        "hunyuanimage_double_image", "hunyuanimage_double_video", "hunyuanimage_double_nobias",
        "hunyuanimage_double_image_batch2", "hunyuanimage_double_video_batch2",
        "hunyuanimage_single_image", "hunyuanimage_single_video", "hunyuanimage_single_nobias",
        "hunyuanimage_single_image_batch2", "hunyuanimage_single_video_batch2",
        "hunyuanvideo_refiner_batch1", "hunyuanvideo_refiner_batch2",
        "auraflow_joint_base", "auraflow_joint_qknorm", "auraflow_joint_batch2",
        "auraflow_single_base", "auraflow_single_qknorm", "auraflow_single_batch2",
        "sd3_joint_qknorm", "sd3_joint_noqknorm", "sd3_joint_preonly", "sd3_joint_dual", "sd3_joint_norm2_affine", "sd3_joint_batch2",
        "hidream_double_base", "hidream_double_qkvbias", "hidream_double_batch2",
        "hidream_single_base", "hidream_single_qkvbias", "hidream_single_batch2",
        "ideogram4_block_split_ffn", "ideogram4_block_fused_ffn", "ideogram4_block_masked", "ideogram4_block_batch2",
        "anima_block_base", "anima_block_cross_mask", "anima_block_batch2",
        "kandinsky5_encoder_base", "kandinsky5_encoder_batch2",
        "kandinsky5_decoder_image", "kandinsky5_decoder_video", "kandinsky5_decoder_batch2",
        "ernie_block_base", "ernie_block_masked", "ernie_block_batch2",
        "moss_block_plain_qkv", "moss_block_fused_qkv", "moss_block_prefix", "moss_block_cached_refs", "moss_block_segmented_refs",
        "moss_text_base",
        "flite_block_default", "flite_block_vresidual", "flite_block_biasrms", "flite_block_no_rope", "flite_block_batch2",
    ];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name)
    {
        using CpuBackend cpu = new();
        return name switch
        {
            _ when name.StartsWith("flux_double_", StringComparison.Ordinal) => FluxDouble(cpu, name),
            _ when name.StartsWith("flux_single_", StringComparison.Ordinal) => FluxSingle(cpu, name),
            _ when name.StartsWith("flux2_double_", StringComparison.Ordinal) => Flux2Double(cpu, name),
            _ when name.StartsWith("flux2_single_", StringComparison.Ordinal) => Flux2Single(cpu, name),
            _ when name.StartsWith("flux3_", StringComparison.Ordinal) => Flux3(cpu, name),
            _ when name.StartsWith("chroma_double_", StringComparison.Ordinal) => ChromaDouble(cpu, name),
            _ when name.StartsWith("chroma_single_", StringComparison.Ordinal) => ChromaSingle(cpu, name),
            _ when name.StartsWith("qwenimage21_", StringComparison.Ordinal) => QwenImage21(cpu, name),
            _ when name.StartsWith("qwenimage_", StringComparison.Ordinal) => QwenImage(cpu, name),
            _ when name.StartsWith("hunyuanimage_double_", StringComparison.Ordinal) => HunyuanDouble(cpu, name),
            _ when name.StartsWith("hunyuanimage_single_", StringComparison.Ordinal) => HunyuanSingle(cpu, name),
            _ when name.StartsWith("hunyuanvideo_refiner_", StringComparison.Ordinal) => HunyuanRefiner(cpu, name),
            _ when name.StartsWith("auraflow_joint_", StringComparison.Ordinal) => AuraJoint(cpu, name),
            _ when name.StartsWith("auraflow_single_", StringComparison.Ordinal) => AuraSingle(cpu, name),
            _ when name.StartsWith("sd3_", StringComparison.Ordinal) => Sd3(cpu, name),
            _ when name.StartsWith("hidream_double_", StringComparison.Ordinal) => HiDream(cpu, name, single: false),
            _ when name.StartsWith("hidream_single_", StringComparison.Ordinal) => HiDream(cpu, name, single: true),
            _ when name.StartsWith("ideogram4_", StringComparison.Ordinal) => Ideogram4(cpu, name),
            _ when name.StartsWith("anima_", StringComparison.Ordinal) => Anima(cpu, name),
            _ when name.StartsWith("kandinsky5_encoder_", StringComparison.Ordinal) => Kandinsky5Encoder(cpu, name),
            _ when name.StartsWith("kandinsky5_decoder_", StringComparison.Ordinal) => Kandinsky5Decoder(cpu, name),
            _ when name.StartsWith("ernie_", StringComparison.Ordinal) => Ernie(cpu, name),
            _ when name.StartsWith("moss_text_", StringComparison.Ordinal) => MossText(cpu),
            _ when name.StartsWith("moss_", StringComparison.Ordinal) => Moss(cpu, name),
            _ when name.StartsWith("flite_", StringComparison.Ordinal) => FLite(cpu, name),
            _ => throw new ArgumentException($"Unknown attention golden case '{name}'.", nameof(name)),
        };
    }

    // ── Flux ───────────────────────────────────────────────────────────────────────────────────────────────────────

    private static FluxRope FluxRopeFor(Tensor ids, int[] axes, int theta)
    {
        FluxRope rope = new(axes, theta);
        rope.Precompute(ids);
        ids.Dispose();
        return rope;
    }

    /// <summary>Flux.1 ids: text rows, then the image grid, then (kontext) a reference grid.</summary>
    private static FluxRope Flux1Rope(bool kontext) => FluxRopeFor(
        kontext ? FluxRope.BuildPositionIdsKontext(Txt, GridH, GridW, RefH, RefW) : FluxRope.BuildPositionIds(Txt, GridH, GridW),
        [2, 2, 4], 10000);

    private static string FluxDouble(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), kontext = name.EndsWith("kontext", StringComparison.Ordinal);
        bool qkvBias = !name.EndsWith("nobias", StringComparison.Ordinal), masked = name.EndsWith("attnbias", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, img = kontext ? Img + Refs : Img, seq = Txt + img;
        Owner o = new();
        FluxDoubleStreamBlock block = new(Hidden, Heads, Mlp, qkvBias);
        Weights w = new(101);
        w.Linear("b.norm1.linear", 6 * Hidden, Hidden, true);
        w.Linear("b.norm1_context.linear", 6 * Hidden, Hidden, true);
        DualAttnWeights(w, "b", qkvBias);
        GeluFfnWeights(w, "b.ff");
        GeluFfnWeights(w, "b.ff_context");
        block.LoadWeights(w, "b");
        using FluxRope rope = Flux1Rope(kontext);
        Tensor image = o.Own(Seeded(new TensorShape(batch, img, Hidden), 1001));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1002));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1003, 0.5f));
        Tensor? bias = masked ? o.Own(AdditiveMask(1, seq, seq, (i, j) => j <= i + 3)) : null;
        (Tensor outImg, Tensor outTxt) = block.Forward(cpu, image, text, temb, rope, bias);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    private static string FluxSingle(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), kontext = name.EndsWith("kontext", StringComparison.Ordinal);
        bool biases = !name.EndsWith("nobias", StringComparison.Ordinal), masked = name.EndsWith("attnbias", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, seq = Txt + Img + (kontext ? Refs : 0);
        Owner o = new();
        FluxSingleStreamBlock block = new(Hidden, Heads, Mlp);
        Weights w = new(102);
        w.Linear("b.norm.linear", 3 * Hidden, Hidden, true);
        w.AddNorm("b.attn.norm_q.weight", HeadDim);
        w.AddNorm("b.attn.norm_k.weight", HeadDim);
        foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", Hidden, Hidden, biases);
        w.Linear("b.proj_mlp", Mlp, Hidden, biases);
        w.Linear("b.proj_out", Hidden, Hidden + Mlp, biases);
        block.LoadWeights(w, "b");
        using FluxRope rope = Flux1Rope(kontext);
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 1011));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1012, 0.5f));
        Tensor? bias = masked ? o.Own(AdditiveMask(1, seq, seq, (i, j) => j <= i + 3)) : null;
        Tensor result = o.Own(block.Forward(cpu, x, temb, rope, bias));
        return Finish(o, ("out", result));
    }

    // ── Flux2 ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private static FluxRope Flux2RopeFor() => FluxRopeFor(Flux2PosEmbed.BuildPositionIds(Txt, GridH, GridW), [2, 2, 2, 2], 2000);

    private static Tensor[] Mods(Owner o, int count, int batch, uint seed)
    {
        Tensor[] mods = new Tensor[count];
        for (int i = 0; i < count; i++) mods[i] = o.Own(Seeded(new TensorShape(batch, Hidden), seed + (uint)i, 0.5f));
        return mods;
    }

    private static string Flux2Double(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), qkvBias = name.EndsWith("qkvbias", StringComparison.Ordinal);
        bool masked = name.EndsWith("attnbias", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, seq = Txt + Img;
        Owner o = new();
        Flux2DoubleBlock block = new(Hidden, Heads, Mlp, qkvBias);
        Weights w = new(111);
        DualAttnWeights(w, "b", qkvBias, outBias: qkvBias);
        foreach (string ff in new[] { "b.ff", "b.ff_context" })
        {
            w.Linear($"{ff}.linear_in_gate", Mlp, Hidden);
            w.Linear($"{ff}.linear_in_up", Mlp, Hidden);
            w.Linear($"{ff}.linear_out", Hidden, Mlp);
        }
        block.LoadWeights(w, "b");
        using FluxRope rope = Flux2RopeFor();
        Tensor image = o.Own(Seeded(new TensorShape(batch, Img, Hidden), 1101));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1102));
        Tensor[] imgMod = Mods(o, 6, batch, 1110), txtMod = Mods(o, 6, batch, 1120);
        Tensor? bias = masked ? o.Own(AdditiveMask(1, seq, seq, (i, j) => j <= i + 3)) : null;
        (Tensor outImg, Tensor outTxt) = block.Forward(cpu, image, text, imgMod, txtMod, rope, bias);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    private static string Flux2Single(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), qkvBias = name.EndsWith("qkvbias", StringComparison.Ordinal);
        bool masked = name.EndsWith("attnbias", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, seq = Txt + Img;
        Owner o = new();
        Flux2SingleBlock block = new(Hidden, Heads, Mlp, qkvBias);
        Weights w = new(112);
        foreach (string n in new[] { "to_q", "to_k" }) w.Linear($"b.attn.{n}", Hidden, Hidden, qkvBias);
        w.Linear("b.attn.to_v", Hidden, Hidden, qkvBias);
        w.Linear("b.attn.linear_in_gate", Mlp, Hidden, qkvBias);
        w.Linear("b.attn.linear_in_up", Mlp, Hidden, qkvBias);
        w.Linear("b.attn.to_out", Hidden, Hidden + Mlp, qkvBias);
        w.AddNorm("b.attn.norm_q.weight", HeadDim);
        w.AddNorm("b.attn.norm_k.weight", HeadDim);
        block.LoadWeights(w, "b");
        using FluxRope rope = Flux2RopeFor();
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 1111));
        Tensor[] mod = Mods(o, 3, batch, 1130);
        Tensor? bias = masked ? o.Own(AdditiveMask(1, seq, seq, (i, j) => j <= i + 3)) : null;
        Tensor result = o.Own(block.Forward(cpu, x, mod, rope, bias));
        return Finish(o, ("out", result));
    }

    // ── Flux3 ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string Flux3(CpuBackend cpu, string name)
    {
        bool spans = name.EndsWith("spans", StringComparison.Ordinal), refs = name.EndsWith("refs", StringComparison.Ordinal);
        int refTokens = refs ? Refs : 0, seq = Txt + Img + refTokens;
        Owner o = new();
        Flux3Block block = new(Hidden, Heads, Mlp, 1e-6f, 1e-6f);
        Weights w = new(121);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj" }) w.Linear($"b.{n}", Hidden, Hidden);
        w.Linear("b.mlp_in", 2 * Mlp, Hidden);
        w.Linear("b.attn_out", Hidden, Hidden);
        w.Linear("b.mlp_out", Hidden, Mlp);
        w.AddNorm("b.norm.query_norm.weight", HeadDim);
        w.AddNorm("b.norm.key_norm.weight", HeadDim);
        block.LoadWeights(w, "b");

        // Ids as the transformer lays them out: text, image grid, then reference grids.
        List<Tensor> parts = [Flux3PosEmbed.TextIds(Txt), Flux3PosEmbed.ImageIds(GridH, GridW, 0f, 1f)];
        if (refs) parts.Add(Flux3PosEmbed.ReferenceIds([(RefH, RefW)], 1f));
        Tensor ids = Flux3PosEmbed.Concat(parts.ToArray());
        foreach (Tensor part in parts) part.Dispose();
        using FluxRope rope = FluxRopeFor(ids, [2, 2, 2, 2], 2000);

        Tensor x = o.Own(Seeded(new TensorShape(1, seq, Hidden), 1201));
        (Tensor, Tensor, Tensor) Row(uint seed) => (o.Own(Seeded(new TensorShape(1, Hidden), seed, 0.5f)),
            o.Own(Seeded(new TensorShape(1, Hidden), seed + 1, 0.5f)), o.Own(Seeded(new TensorShape(1, Hidden), seed + 2, 0.5f)));
        Flux3Modulation modulation;
        if (spans || refs)
        {
            // One modulation row per span: text, image tokens, then (refs) the reference tokens.
            List<(Tensor, Tensor, Tensor)> rows = [Row(1210), Row(1220)];
            List<int> lengths = [Txt, Img];
            if (refs)
            {
                rows.Add(Row(1230));
                lengths.Add(refTokens);
            }
            modulation = Flux3Modulation.Spans(cpu, rows, lengths);
        }
        else
        {
            (Tensor shift, Tensor scale, Tensor gate) = Row(1210);
            // Uniform takes ownership of its rows, so hand it private copies.
            modulation = Flux3Modulation.Uniform(Copy(shift), Copy(scale), Copy(gate));
        }
        using (modulation)
        {
            Tensor result = o.Own(block.Forward(cpu, x, modulation, rope));
            return Finish(o, ("out", result));
        }
    }

    // ── Chroma ─────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string ChromaDouble(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), fused = name.EndsWith("fused", StringComparison.Ordinal);
        bool masked = name.EndsWith("mask", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, seq = Txt + Img, mlp = 4 * Hidden;
        Owner o = new();
        ChromaDoubleStreamBlock block = new(Hidden, Heads, HeadDim);
        Weights w = new(131);
        if (fused)
        {
            w.Linear("b.attn.qkv", 3 * Hidden, Hidden, true);
            w.Linear("b.attn.add_qkv", 3 * Hidden, Hidden, true);
            w.Linear("b.attn.to_out.0", Hidden, Hidden, true);
            w.Linear("b.attn.to_add_out", Hidden, Hidden, true);
            foreach (string n in new[] { "norm_q", "norm_k", "norm_added_q", "norm_added_k" }) w.AddNorm($"b.attn.{n}.weight", HeadDim);
        }
        else
        {
            DualAttnWeights(w, "b", true);
        }
        GeluFfnWeights(w, "b.ff", mlp);
        GeluFfnWeights(w, "b.ff_context", mlp);
        block.LoadWeights(w, "b");
        using FluxRope rope = FluxRopeFor(FluxRope.BuildPositionIds(Txt, GridH, GridW), [2, 2, 4], 10000);
        Tensor image = o.Own(Seeded(new TensorShape(batch, Img, Hidden), 1301));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1302));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, 12, Hidden), 1303, 0.5f));
        Tensor? mask = masked ? o.Own(AdditiveMask(batch, seq, seq, (i, j) => j <= i + 3)) : null;
        (Tensor outImg, Tensor outTxt) = block.Forward(cpu, image, text, temb, rope, mask);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    private static string ChromaSingle(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), fused = name.EndsWith("fused", StringComparison.Ordinal);
        bool masked = name.EndsWith("mask", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, seq = Txt + Img, mlp = 4 * Hidden;
        Owner o = new();
        ChromaSingleStreamBlock block = new(Hidden, Heads, HeadDim);
        Weights w = new(132);
        w.AddNorm("b.attn.norm_q.weight", HeadDim);
        w.AddNorm("b.attn.norm_k.weight", HeadDim);
        if (fused)
        {
            w.Linear("b.linear1", 3 * Hidden + mlp, Hidden, true);
        }
        else
        {
            foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", Hidden, Hidden, true);
            w.Linear("b.proj_mlp", mlp, Hidden, true);
        }
        w.Linear("b.proj_out", Hidden, Hidden + mlp, true);
        block.LoadWeights(w, "b");
        using FluxRope rope = FluxRopeFor(FluxRope.BuildPositionIds(Txt, GridH, GridW), [2, 2, 4], 10000);
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 1311));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, 3, Hidden), 1312, 0.5f));
        Tensor? mask = masked ? o.Own(AdditiveMask(batch, seq, seq, (i, j) => j <= i + 3)) : null;
        Tensor result = o.Own(block.Forward(cpu, x, temb, rope, mask));
        return Finish(o, ("out", result));
    }

    // ── Qwen-Image ─────────────────────────────────────────────────────────────────────────────────────────────────

    private static string QwenImage(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), refs = name.EndsWith("refs", StringComparison.Ordinal);
        bool tembZero = name.EndsWith("tembzero", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, img = Img + (refs || tembZero ? Refs : 0);
        Owner o = new();
        QwenImageBlock block = new(Hidden, Heads, HeadDim, Mlp);
        Weights w = new(141);
        w.Linear("b.img_mod.1", 6 * Hidden, Hidden, true);
        w.Linear("b.txt_mod.1", 6 * Hidden, Hidden, true);
        DualAttnWeights(w, "b", true);
        GeluFfnWeights(w, "b.img_mlp");
        GeluFfnWeights(w, "b.txt_mlp");
        block.LoadWeights(w, "b");
        using QwenImageRope rope = new([2, 2, 4], 10000);
        int start = QwenImageRope.ComputeTextPositionStart(GridH, GridW);
        (int H, int W)[]? grids = refs || tembZero ? [(RefH, RefW)] : null;
        Tensor image = o.Own(Seeded(new TensorShape(batch, img, Hidden), 1401));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1402));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1403, 0.5f));
        Tensor? zero = tembZero ? o.Own(Seeded(new TensorShape(batch, Hidden), 1404, 0.5f)) : null;
        (Tensor outImg, Tensor outTxt) = block.Forward(cpu, image, text, temb, rope, GridH, GridW, start, grids, zero, tembZero ? Img : 0);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    // ── Qwen-Image 2.1 ─────────────────────────────────────────────────────────────────────────────────────────────

    private static string QwenImage21(CpuBackend cpu, string name)
    {
        Owner o = new();
        QwenImage21Block block = new(Hidden, Heads, HeadDim, Mlp);
        Weights w = new(151);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w.Linear($"b.attn.{n}", Hidden, Hidden);
        w.AddNorm("b.attn.norm_q.weight", HeadDim);
        w.AddNorm("b.attn.norm_k.weight", HeadDim);
        w.Linear("b.img_mlp.gate_up", 2 * Mlp, Hidden);
        w.Linear("b.img_mlp.out", Hidden, Mlp);
        block.LoadWeights(w, "b");
        using QwenImage21Rope rope = new([2, 2, 4], 10000);
        Tensor Row(uint seed, bool plusOne) { Tensor t = Seeded(new TensorShape(1, Hidden), seed, 0.5f); if (plusOne) AddInPlace(t, 1f); return o.Own(t); }
        QwenImage21Modulation mod = new(Row(1501, true), Row(1502, false), Row(1503, true), Row(1504, false));
        const int textLen = 5;
        (Tensor textCos, Tensor textSin) = rope.GetOrBuildTextTables(cpu, textLen);
        Tensor text = o.Own(Seeded(new TensorShape(1, textLen, Hidden), 1511));
        // The prefix pass keeps its post-RoPE K and V for the target pass to attend over.
        Tensor causal = o.Own(AdditiveMask(1, textLen, textLen, (i, j) => j <= i));
        Tensor prefixOut = o.Own(block.ForwardPrefix(cpu, text, mod, textCos, textSin, causal, out Tensor keys, out Tensor values));
        o.Own(keys); o.Own(values);
        if (name == "qwenimage21_block_prefix")
            return Finish(o, ("out", prefixOut), ("k", keys), ("v", values));

        (Tensor imgCos, Tensor imgSin) = rope.GetOrBuildImageTables(cpu, textLen, GridH, GridW);
        Tensor image = o.Own(Seeded(new TensorShape(1, Img, Hidden), 1512));
        bool withPrefix = name.EndsWith("with_prefix", StringComparison.Ordinal);
        Tensor result = o.Own(block.ForwardTarget(cpu, image, mod, imgCos, imgSin, withPrefix ? keys : null, withPrefix ? values : null));
        return Finish(o, ("out", result));
    }

    // ── HunyuanImage / HunyuanVideo ────────────────────────────────────────────────────────────────────────────────

    private static HunyuanImageRope HunyuanRopeFor(bool video) => video ? new HunyuanImageRope([2, 2, 4], 256f) : new HunyuanImageRope([4, 4], 256f);

    private static string HunyuanDouble(CpuBackend cpu, string name)
    {
        bool video = name.Contains("_video", StringComparison.Ordinal), batch2 = name.EndsWith("batch2", StringComparison.Ordinal);
        bool qkvBias = !name.EndsWith("nobias", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, frames = video ? 2 : 1, img = frames * Img;
        Owner o = new();
        HunyuanImageBlock block = new(Hidden, Heads, HeadDim, Mlp);
        Weights w = new(161);
        w.Linear("b.norm1.linear", 6 * Hidden, Hidden, true);
        w.Linear("b.norm1_context.linear", 6 * Hidden, Hidden, true);
        DualAttnWeights(w, "b", qkvBias);
        GeluFfnWeights(w, "b.ff");
        GeluFfnWeights(w, "b.ff_context");
        block.LoadWeights(w, "b");
        using HunyuanImageRope rope = HunyuanRopeFor(video);
        Tensor image = o.Own(Seeded(new TensorShape(batch, img, Hidden), 1601));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1602));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1603, 0.5f));
        (Tensor outImg, Tensor outTxt) = block.Forward(cpu, image, text, temb, rope, GridH, GridW, frames);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    private static string HunyuanSingle(CpuBackend cpu, string name)
    {
        bool video = name.Contains("_video", StringComparison.Ordinal), batch2 = name.EndsWith("batch2", StringComparison.Ordinal);
        bool biases = !name.EndsWith("nobias", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, frames = video ? 2 : 1, img = frames * Img;
        Owner o = new();
        HunyuanImageSingleBlock block = new(Hidden, Heads, HeadDim, Mlp);
        Weights w = new(162);
        w.Linear("b.norm.linear", 3 * Hidden, Hidden, true);
        foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", Hidden, Hidden, biases);
        w.AddNorm("b.attn.norm_q.weight", HeadDim);
        w.AddNorm("b.attn.norm_k.weight", HeadDim);
        w.Linear("b.proj_mlp", Mlp, Hidden, biases);
        w.Linear("b.proj_out", Hidden, Hidden + Mlp, biases);
        block.LoadWeights(w, "b");
        using HunyuanImageRope rope = HunyuanRopeFor(video);
        Tensor image = o.Own(Seeded(new TensorShape(batch, img, Hidden), 1611));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1612));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1613, 0.5f));
        (Tensor outImg, Tensor outTxt) = block.Forward(cpu, image, text, temb, rope, GridH, GridW, frames);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    private static string HunyuanRefiner(CpuBackend cpu, string name)
    {
        const int textDim = 24, textLen = 5;
        int batch = name.EndsWith("batch2", StringComparison.Ordinal) ? 2 : 1;
        Owner o = new();
        HunyuanVideoTokenRefiner refiner = new(textDim, Hidden, Heads, numBlocks: 2);
        Weights w = new(171);
        w.Linear("txt_in.input_embedder", Hidden, textDim, true);
        w.Linear("txt_in.t_embedder.in_layer", Hidden, 256, true);
        w.Linear("txt_in.t_embedder.out_layer", Hidden, Hidden, true);
        w.Linear("txt_in.c_embedder.in_layer", Hidden, textDim, true);
        w.Linear("txt_in.c_embedder.out_layer", Hidden, Hidden, true);
        for (int i = 0; i < 2; i++)
        {
            string p = $"txt_in.individual_token_refiner.blocks.{i}";
            w.AddNorm($"{p}.norm1.weight", Hidden); w.AddBias($"{p}.norm1.bias", Hidden);
            w.AddNorm($"{p}.norm2.weight", Hidden); w.AddBias($"{p}.norm2.bias", Hidden);
            w.Linear($"{p}.self_attn.qkv", 3 * Hidden, Hidden, true);
            w.Linear($"{p}.self_attn.proj", Hidden, Hidden, true);
            w.Linear($"{p}.mlp.0", 4 * Hidden, Hidden, true);
            w.Linear($"{p}.mlp.2", Hidden, 4 * Hidden, true);
            w.Linear($"{p}.adaLN_modulation.1", 2 * Hidden, Hidden, true);
        }
        refiner.LoadWeights(w);
        Tensor text = o.Own(Seeded(new TensorShape(batch, textLen, textDim), 1701));
        Tensor result = o.Own(refiner.Forward(cpu, text, 0.37f));
        return Finish(o, ("out", result));
    }

    // ── AuraFlow ───────────────────────────────────────────────────────────────────────────────────────────────────

    private static void AuraNorms(Weights w, string p, bool added)
    {
        w.AddNorm($"{p}.attn.norm_q.weight", HeadDim);
        w.AddNorm($"{p}.attn.norm_k.weight", HeadDim);
        if (!added) return;
        w.AddNorm($"{p}.attn.norm_added_q.weight", HeadDim);
        w.AddNorm($"{p}.attn.norm_added_k.weight", HeadDim);
    }

    private static string AuraJoint(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), qkNorm = name.EndsWith("qknorm", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1;
        Owner o = new();
        AuraFlowJointBlock block = new(Hidden, Heads, HeadDim, Mlp);
        Weights w = new(181);
        w.Linear("b.norm1.linear", 6 * Hidden, Hidden);
        w.Linear("b.norm1_context.linear", 6 * Hidden, Hidden);
        DualAttnWeights(w, "b", false, outBias: false, norms: false);
        // The checkpoint ships no QK-norm scales (they default to ones); "qknorm" cases load explicit ones.
        if (qkNorm) AuraNorms(w, "b", added: true);
        foreach (string ff in new[] { "b.ff", "b.ff_context" })
        {
            w.Linear($"{ff}.linear_1", Mlp, Hidden);
            w.Linear($"{ff}.linear_2", Mlp, Hidden);
            w.Linear($"{ff}.out_projection", Hidden, Mlp);
        }
        block.LoadWeights(w, "b");
        Tensor image = o.Own(Seeded(new TensorShape(batch, Img, Hidden), 1801));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1802));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1803, 0.5f));
        (Tensor outImg, Tensor outTxt) = block.Forward(cpu, image, text, temb);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    private static string AuraSingle(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), qkNorm = name.EndsWith("qknorm", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1, seq = Txt + Img;
        Owner o = new();
        AuraFlowSingleBlock block = new(Hidden, Heads, HeadDim, Mlp);
        Weights w = new(182);
        w.Linear("b.norm1.linear", 6 * Hidden, Hidden);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w.Linear($"b.attn.{n}", Hidden, Hidden);
        if (qkNorm) AuraNorms(w, "b", added: false);
        w.Linear("b.ff.linear_1", Mlp, Hidden);
        w.Linear("b.ff.linear_2", Mlp, Hidden);
        w.Linear("b.ff.out_projection", Hidden, Mlp);
        block.LoadWeights(w, "b");
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 1811));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1812, 0.5f));
        Tensor result = o.Own(block.Forward(cpu, x, temb));
        return Finish(o, ("out", result));
    }

    // ── SD3 ────────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string Sd3(CpuBackend cpu, string name)
    {
        bool preOnly = name.EndsWith("preonly", StringComparison.Ordinal), dual = name.EndsWith("dual", StringComparison.Ordinal);
        bool qkNorm = !name.EndsWith("noqknorm", StringComparison.Ordinal), batch2 = name.EndsWith("batch2", StringComparison.Ordinal);
        bool norm2Affine = name.EndsWith("norm2_affine", StringComparison.Ordinal);
        int batch = batch2 ? 2 : 1;
        Owner o = new();
        JointBlock block = new(Hidden, Heads, Mlp, preOnly, qkNorm, dual);
        Weights w = new(191u + (preOnly ? 1u : dual ? 2u : 0u));
        w.Linear("b.norm1.linear", (dual ? 9 : 6) * Hidden, Hidden, true);
        w.Linear("b.norm1_context.linear", (preOnly ? 2 : 6) * Hidden, Hidden, true);
        DualAttnWeights(w, "b", true, toAddOut: !preOnly, norms: qkNorm);
        if (dual)
        {
            foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w.Linear($"b.attn2.{n}", Hidden, Hidden, true);
            if (qkNorm)
            {
                w.AddNorm("b.attn2.norm_q.weight", HeadDim);
                w.AddNorm("b.attn2.norm_k.weight", HeadDim);
            }
        }
        GeluFfnWeights(w, "b.ff");
        if (!preOnly) GeluFfnWeights(w, "b.ff_context");
        if (norm2Affine)
        {
            w.AddNorm("b.norm2.weight", Hidden); w.AddBias("b.norm2.bias", Hidden);
            w.AddNorm("b.norm2_context.weight", Hidden); w.AddBias("b.norm2_context.bias", Hidden);
        }
        block.LoadWeights(w, "b");
        Tensor image = o.Own(Seeded(new TensorShape(batch, Img, Hidden), 1901));
        Tensor context = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 1902));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 1903, 0.5f));
        (Tensor outImg, Tensor outCtx) = block.Forward(cpu, image, context, temb);
        o.Own(outImg);
        // The pre-only block hands the context back untouched, so only the image stream is digested.
        if (preOnly) return Finish(o, ("img", outImg));
        o.Own(outCtx);
        return Finish(o, ("img", outImg), ("ctx", outCtx));
    }

    // ── HiDream ────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string HiDream(CpuBackend cpu, string name, bool single)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), biases = name.EndsWith("qkvbias", StringComparison.Ordinal);
        const int experts = 4, sharedFf = 48, routedFf = 64;
        int batch = batch2 ? 2 : 1, seq = Img + Txt;
        Owner o = new();
        HiDreamBlock block = new(Hidden, Heads, HeadDim, 4 * Hidden, single, experts, 2);
        Weights w = new(201);
        w.Linear("b.adaLN_modulation.1", (single ? 6 : 12) * Hidden, Hidden, true);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out" }) w.Linear($"b.attn1.{n}", Hidden, Hidden, biases);
        // The Q/K RMSNorm runs over the whole hidden width, not per head.
        w.AddNorm("b.attn1.q_rms_norm.weight", Hidden);
        w.AddNorm("b.attn1.k_rms_norm.weight", Hidden);
        if (!single)
        {
            foreach (string n in new[] { "to_q_t", "to_k_t", "to_v_t", "to_out_t" }) w.Linear($"b.attn1.{n}", Hidden, Hidden, biases);
            w.AddNorm("b.attn1.q_rms_norm_t.weight", Hidden);
            w.AddNorm("b.attn1.k_rms_norm_t.weight", Hidden);
            AddSwiGlu(w, "b.ff_t", routedFf);
        }
        AddSwiGlu(w, "b.ff_i.shared_experts", sharedFf);
        for (int e = 0; e < experts; e++) AddSwiGlu(w, $"b.ff_i.experts.{e}", routedFf);
        w.Linear("b.ff_i.gate", experts, Hidden);
        block.LoadWeights(w, "b");

        HiDreamRope rope = new([2, 2, 4], 10000);
        Tensor ids = HiDreamRope.BuildPositionIds(Img, GridH, GridW, Txt);
        rope.Precompute(ids);
        ids.Dispose();
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 2001, 0.5f));
        if (single)
        {
            Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 2002));
            Tensor result = o.Own(block.ForwardSingle(cpu, x, temb, rope, Img, seq));
            return Finish(o, ("out", result));
        }
        Tensor image = o.Own(Seeded(new TensorShape(batch, Img, Hidden), 2003));
        Tensor text = o.Own(Seeded(new TensorShape(batch, Txt, Hidden), 2004));
        (Tensor outImg, Tensor outTxt) = block.ForwardDouble(cpu, image, text, temb, rope, seq);
        o.Own(outImg); o.Own(outTxt);
        return Finish(o, ("img", outImg), ("txt", outTxt));
    }

    private static void AddSwiGlu(Weights w, string p, int inner)
    {
        w.Linear($"{p}.w1", inner, Hidden);
        w.Linear($"{p}.w3", inner, Hidden);
        w.Linear($"{p}.w2", Hidden, inner);
    }

    // ── Ideogram4 ──────────────────────────────────────────────────────────────────────────────────────────────────

    private static string Ideogram4(CpuBackend cpu, string name)
    {
        bool fused = name.EndsWith("fused_ffn", StringComparison.Ordinal), masked = name.EndsWith("masked", StringComparison.Ordinal);
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal);
        const int adaln = 16, ffn = 48;
        int batch = batch2 ? 2 : 1, seq = Txt + Img;
        Owner o = new();
        Ideogram4Block block = new(Hidden, Heads, ffn, 1e-6f);
        Weights w = new(211);
        foreach (string n in new[] { "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2" }) w.AddNorm($"b.{n}.weight", Hidden);
        w.Linear("b.attention.qkv", 3 * Hidden, Hidden);
        w.Linear("b.attention.o", Hidden, Hidden);
        w.AddNorm("b.attention.norm_q.weight", HeadDim);
        w.AddNorm("b.attention.norm_k.weight", HeadDim);
        if (fused)
        {
            w.Linear("b.feed_forward.w13", 2 * ffn, Hidden);
        }
        else
        {
            w.Linear("b.feed_forward.w1", ffn, Hidden);
            w.Linear("b.feed_forward.w3", ffn, Hidden);
        }
        w.Linear("b.feed_forward.w2", Hidden, ffn);
        w.Linear("b.adaln_modulation", 4 * Hidden, adaln, true);
        block.LoadWeights(w, "b");

        // Text rows at (i, i, i), then the image grid at the 65536 offset, as the pipeline lays them out.
        Tensor ids = o.Own(new Tensor(new TensorShape(batch, seq, 3), DType.F32));
        float* p = (float*)ids.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seq; t++)
            {
                long at = 3L * (b * seq + t);
                if (t < Txt) { p[at] = p[at + 1] = p[at + 2] = t + b; }
                else { int i = t - Txt; p[at] = 65536; p[at + 1] = 65536 + i / GridW; p[at + 2] = 65536 + i % GridW; }
            }
        (Tensor cos, Tensor sin) = new Ideogram4Mrope(HeadDim, 10000, (2, 1, 1)).BuildCosSin(ids);
        o.Own(cos); o.Own(sin);
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 2101));
        Tensor c = o.Own(Seeded(new TensorShape(batch, adaln), 2102, 0.5f));
        Tensor? mask = masked ? o.Own(AdditiveMask(1, seq, seq, (i, j) => j <= i + 2)) : null;
        Tensor result = o.Own(block.Forward(cpu, x, c, cos, sin, mask));
        return Finish(o, ("out", result));
    }

    // ── Anima ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string Anima(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), crossMask = name.EndsWith("cross_mask", StringComparison.Ordinal);
        const int hidden = 24, heads = 2, headDim = 12, ffn = 48, rank = 8, kvDim = 16, textLen = 5;
        int batch = batch2 ? 2 : 1;
        Owner o = new();
        AnimaBlock block = new(hidden, heads, headDim, ffn, rank);
        Weights w = new(221);
        foreach (string a in new[] { "self_attn", "cross_attn" }) w.AddNorm($"b.{a}.q_norm.weight", headDim);
        foreach (string a in new[] { "self_attn", "cross_attn" }) w.AddNorm($"b.{a}.k_norm.weight", headDim);
        foreach (string m in new[] { "adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp" })
        {
            w.Linear($"b.{m}.1", rank, hidden);
            w.Linear($"b.{m}.2", 3 * hidden, rank);
        }
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "output_proj" }) w.Linear($"b.self_attn.{n}", hidden, hidden);
        w.Linear("b.cross_attn.q_proj", hidden, hidden);
        w.Linear("b.cross_attn.k_proj", hidden, kvDim);
        w.Linear("b.cross_attn.v_proj", hidden, kvDim);
        w.Linear("b.cross_attn.output_proj", hidden, hidden);
        w.Linear("b.mlp.layer1", ffn, hidden);
        w.Linear("b.mlp.layer2", hidden, ffn);
        block.LoadWeights(w, "b");
        AnimaRope rope = new(headDim, 10000f, (1f, 2f, 2f));
        (Tensor cos, Tensor sin) = rope.BuildFreqs(1, GridH, GridW);
        o.Own(cos); o.Own(sin);
        Tensor x = o.Own(Seeded(new TensorShape(batch, Img, hidden), 2201));
        Tensor text = o.Own(Seeded(new TensorShape(batch, textLen, kvDim), 2202));
        Tensor embedded = o.Own(Seeded(new TensorShape(batch, hidden), 2203, 0.5f));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, 3 * hidden), 2204, 0.5f));
        Tensor? mask = crossMask ? o.Own(AdditiveMask(1, Img, textLen, (_, j) => j < textLen - 2)) : null;
        Tensor result = o.Own(block.Forward(cpu, x, text, embedded, temb, cos, sin, rope, mask));
        return Finish(o, ("out", result));
    }

    // ── Kandinsky5 ─────────────────────────────────────────────────────────────────────────────────────────────────

    private const int KTime = 16, KFf = 48;

    private static void Kandinsky5AttnWeights(Weights w, string p)
    {
        foreach (string n in new[] { "to_query", "to_key", "to_value", "out_layer" }) w.Linear($"{p}.{n}", Hidden, Hidden, true);
        w.AddNorm($"{p}.query_norm.weight", HeadDim);
        w.AddNorm($"{p}.key_norm.weight", HeadDim);
    }

    private static string Kandinsky5Encoder(CpuBackend cpu, string name)
    {
        int batch = name.EndsWith("batch2", StringComparison.Ordinal) ? 2 : 1;
        const int seq = 7;
        Owner o = new();
        Kandinsky5EncoderBlock block = new(Hidden, KTime, KFf, HeadDim);
        Weights w = new(231);
        w.Linear("b.text_modulation.out_layer", 6 * Hidden, KTime, true);
        Kandinsky5AttnWeights(w, "b.self_attention");
        w.Linear("b.feed_forward.in_layer", KFf, Hidden);
        w.Linear("b.feed_forward.out_layer", Hidden, KFf);
        block.LoadWeights(w, "b");
        Kandinsky5Rope rope = new(HeadDim);
        rope.Precompute1D([0, 1, 2, 3, 4, 5, 6]);
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 2301));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, KTime), 2302, 0.5f));
        Tensor result = o.Own(block.Forward(cpu, x, temb, rope));
        return Finish(o, ("out", result));
    }

    private static string Kandinsky5Decoder(CpuBackend cpu, string name)
    {
        bool video = name.EndsWith("video", StringComparison.Ordinal);
        int batch = name.EndsWith("batch2", StringComparison.Ordinal) ? 2 : 1, frames = video ? 2 : 1, vis = frames * Img;
        Owner o = new();
        Kandinsky5DecoderBlock block = new(Hidden, KTime, KFf, HeadDim);
        Weights w = new(232);
        w.Linear("b.visual_modulation.out_layer", 9 * Hidden, KTime, true);
        Kandinsky5AttnWeights(w, "b.self_attention");
        Kandinsky5AttnWeights(w, "b.cross_attention");
        w.Linear("b.feed_forward.in_layer", KFf, Hidden);
        w.Linear("b.feed_forward.out_layer", Hidden, KFf);
        block.LoadWeights(w, "b");
        Kandinsky5Rope rope = new(HeadDim);
        if (video) rope.Precompute3D([2, 2, 4], frames, GridH, GridW, 3, 1.5f, 2f, 2f);
        else rope.Precompute3D([2, 2, 4], frames, GridH, GridW);
        Tensor visual = o.Own(Seeded(new TensorShape(batch, vis, Hidden), 2311));
        Tensor text = o.Own(Seeded(new TensorShape(batch, 5, Hidden), 2312));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, KTime), 2313, 0.5f));
        Tensor result = o.Own(block.Forward(cpu, visual, text, temb, rope));
        return Finish(o, ("out", result));
    }

    // ── Ernie ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string Ernie(CpuBackend cpu, string name)
    {
        bool batch2 = name.EndsWith("batch2", StringComparison.Ordinal), masked = !name.EndsWith("base", StringComparison.Ordinal);
        const int ffn = 48, textMax = 4;
        int batch = batch2 ? 2 : 1, seq = textMax + Img;
        int[] textLens = batch2 ? [4, 2] : [3];
        Owner o = new();
        ErnieImageBlock block = new(Hidden, Heads, ffn);
        Weights w = new(241);
        w.AddNorm("b.adaLN_sa_ln.weight", Hidden);
        w.AddNorm("b.adaLN_mlp_ln.weight", Hidden);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w.Linear($"b.self_attention.{n}", Hidden, Hidden);
        w.AddNorm("b.self_attention.norm_q.weight", HeadDim);
        w.AddNorm("b.self_attention.norm_k.weight", HeadDim);
        w.Linear("b.mlp.gate_proj", ffn, Hidden);
        w.Linear("b.mlp.up_proj", ffn, Hidden);
        w.Linear("b.mlp.linear_fc2", Hidden, ffn);
        block.LoadWeights(w, "b");
        Tensor ids = ErnieImageRope.BuildPositionIds(batch, textLens, GridH, GridW, textMax);
        Tensor freqs = new ErnieImageRope([2, 2, 4], 256f).BuildFreqs(ids);
        ids.Dispose();
        Tensor cos = o.Own(new Tensor(new TensorShape(batch, seq, HeadDim), DType.F32));
        Tensor sin = o.Own(new Tensor(new TensorShape(batch, seq, HeadDim), DType.F32));
        float* f = (float*)freqs.DataPointer;
        for (int t = 0; t < batch * seq; t++)
            for (int d = 0; d < HeadDim; d++)
            {
                ((float*)cos.DataPointer)[t * HeadDim + d] = f[t * 2 * HeadDim + d];
                ((float*)sin.DataPointer)[t * HeadDim + d] = f[t * 2 * HeadDim + HeadDim + d];
            }
        freqs.Dispose();
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 2401));
        Tensor[] mod = Mods(o, 6, batch, 2410);
        // Padding mask: text keys past each item's text length are blanked.
        Tensor? mask = null;
        if (masked)
        {
            mask = o.Own(new Tensor(new TensorShape(batch, 1, seq, seq), DType.F32));
            float* m = (float*)mask.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < seq; i++)
                    for (int j = 0; j < seq; j++)
                        m[((long)b * seq + i) * seq + j] = j < textMax && j >= textLens[b] ? NegInf : 0f;
        }
        Tensor result = o.Own(block.Forward(cpu, x, mod[0], mod[1], mod[2], mod[3], mod[4], mod[5], cos, sin, mask));
        return Finish(o, ("out", result));
    }

    // ── MOSS-Image ─────────────────────────────────────────────────────────────────────────────────────────────────

    private static MossImageBlock MossBlock(bool fusedQkv)
    {
        const int ffn = 48;
        MossImageBlock block = new(Hidden, Heads, ffn, 1e-6f);
        Weights w = new(251);
        w.AddNorm("b.norm1.weight", Hidden);
        w.AddNorm("b.norm2.weight", Hidden);
        w.AddNorm("b.q_norm.weight", HeadDim);
        w.AddNorm("b.k_norm.weight", HeadDim);
        if (fusedQkv)
        {
            w.Linear("b.proj.qkv", 3 * Hidden, Hidden);
        }
        else
        {
            foreach (string n in new[] { "q", "k", "v" }) w.Linear($"b.proj.{n}", Hidden, Hidden);
        }
        w.Linear("b.proj.o", Hidden, Hidden);
        w.Linear("b.proj.ffn_up_gate", 2 * ffn, Hidden);
        w.Linear("b.proj.ffn_down", Hidden, ffn);
        block.LoadWeights(w, "b");
        return block;
    }

    private static string Moss(CpuBackend cpu, string name)
    {
        Owner o = new();
        MossImageBlock block = MossBlock(fusedQkv: name.Contains("fused_qkv", StringComparison.Ordinal));
        MossImageModulation Modulation()
        {
            Tensor rows = o.Own(Seeded(new TensorShape(6, Hidden), 2501, 0.5f));
            Tensor head = o.Own(Seeded(new TensorShape(2, Hidden), 2502, 0.5f));
            return MossImageModulation.Split(cpu, rows, head, Hidden);
        }
        using MossImageModulation mod = Modulation();
        const int textLen = 3, refTokens = 4;
        int seq = Img + textLen;
        (Tensor cos, Tensor sin) = MossImageRope.Joint(GridH, GridW, textLen, HeadDim, 10000f);
        o.Own(cos); o.Own(sin);
        Tensor x = o.Own(Seeded(new TensorShape(1, seq, Hidden), 2503));

        if (name.EndsWith("plain_qkv", StringComparison.Ordinal) || name.EndsWith("fused_qkv", StringComparison.Ordinal))
        {
            Tensor result = o.Own(block.Forward(cpu, x, mod, cos, sin, DType.F32));
            return Finish(o, ("out", result));
        }
        if (name.EndsWith("segmented_refs", StringComparison.Ordinal))
        {
            // Joint references: [image, text] on the live modulation, the reference tokens on a second (timestep-zero) group.
            (Tensor cosJ, Tensor sinJ) = MossImageRope.Joint(GridH, GridW, textLen, [(RefH, RefW)], HeadDim, 10000f);
            o.Own(cosJ); o.Own(sinJ);
            Tensor joint = o.Own(Seeded(new TensorShape(1, seq + refTokens, Hidden), 2504));
            Tensor rowsZero = o.Own(Seeded(new TensorShape(6, Hidden), 2505, 0.5f));
            Tensor headZero = o.Own(Seeded(new TensorShape(2, Hidden), 2506, 0.5f));
            using MossImageModulation zero = MossImageModulation.Split(cpu, rowsZero, headZero, Hidden);
            using TokenSegmentLayout layout = new([seq, refTokens], [0, 1]);
            using SegmentModulation attention = SegmentModulation.Build(cpu,
                [(mod.ShiftAttn, mod.ScaleAttn, mod.GateAttn), (zero.ShiftAttn, zero.ScaleAttn, zero.GateAttn)], layout);
            using SegmentModulation ffn = SegmentModulation.Build(cpu,
                [(mod.ShiftFfn, mod.ScaleFfn, mod.GateFfn), (zero.ShiftFfn, zero.ScaleFfn, zero.GateFfn)], layout);
            Tensor result = o.Own(block.ForwardSegmented(cpu, joint, attention, ffn, cosJ, sinJ, DType.F32));
            return Finish(o, ("out", result));
        }

        // Reference tokens through the prefix pass: their post-RoPE K and V are what the live pass can attend over.
        (Tensor refCos, Tensor refSin) = MossImageRope.References([(RefH, RefW)], textLen, HeadDim, 10000f);
        o.Own(refCos); o.Own(refSin);
        Tensor refX = o.Own(Seeded(new TensorShape(1, refTokens, Hidden), 2507));
        Tensor prefix = o.Own(block.ForwardPrefix(cpu, refX, mod, refCos, refSin, DType.F32, out Tensor keys, out Tensor values));
        o.Own(keys); o.Own(values);
        if (name.EndsWith("prefix", StringComparison.Ordinal))
            return Finish(o, ("out", prefix), ("k", keys), ("v", values));
        Tensor cached = o.Own(block.Forward(cpu, x, mod, cos, sin, DType.F32, keys, values));
        return Finish(o, ("out", cached));
    }

    private static string MossText(CpuBackend cpu)
    {
        const int ffn = 48, len = 5;
        Owner o = new();
        MossImageTextBlock block = new(Hidden, Heads, ffn, 1e-6f);
        Weights w = new(252);
        w.AddNorm("b.norm1.weight", Hidden);
        w.AddNorm("b.norm2.weight", Hidden);
        w.AddNorm("b.q_norm.weight", HeadDim);
        w.AddNorm("b.k_norm.weight", HeadDim);
        w.Linear("b.qkv", 3 * Hidden, Hidden, true);
        w.Linear("b.proj", Hidden, Hidden, true);
        w.Linear("b.ffn_gate", ffn, Hidden, true);
        w.Linear("b.ffn_up", ffn, Hidden, true);
        w.Linear("b.ffn_down", Hidden, ffn, true);
        block.LoadWeights(w, "b");
        (Tensor cos, Tensor sin) = MossImageRope.Text(len, HeadDim, 10000f);
        o.Own(cos); o.Own(sin);
        Tensor x = o.Own(Seeded(new TensorShape(1, len, Hidden), 2511));
        Tensor result = o.Own(block.Forward(cpu, x, cos, sin));
        return Finish(o, ("out", result));
    }

    // ── F-Lite ─────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string FLite(CpuBackend cpu, string name)
    {
        bool biasRms = name.EndsWith("biasrms", StringComparison.Ordinal), noRope = name.EndsWith("no_rope", StringComparison.Ordinal);
        bool vResidual = name.EndsWith("vresidual", StringComparison.Ordinal), batch2 = name.EndsWith("batch2", StringComparison.Ordinal);
        const int cross = 24, ctxLen = 5, registers = 2;
        int batch = batch2 ? 2 : 1, seq = registers + Img;
        FLiteConfig config = new()
        {
            HiddenSize = Hidden, NumHeads = Heads, Depth = 1, CrossAttnInputSize = cross, MlpRatio = 2f,
            TrainBiasAndRms = biasRms, ResidualV = true, NumRegisterTokens = registers, RopeMaxGrid = 16,
        };
        Owner o = new();
        FLiteBlock block = new(config);
        Weights w = new(261);
        w.Linear("b.self_attn.qkv", 3 * Hidden, Hidden, biasRms);
        w.Linear("b.self_attn.proj", Hidden, Hidden, biasRms);
        w.Linear("b.cross_attn.q", Hidden, Hidden, biasRms);
        w.Linear("b.cross_attn.context_kv", 2 * Hidden, cross, biasRms);
        w.Linear("b.cross_attn.proj", Hidden, Hidden, biasRms);
        w.AddConst("b.self_attn.lambda_param", 0.6f);
        if (biasRms)
        {
            foreach (string a in new[] { "self_attn", "cross_attn" })
            {
                w.AddNorm($"b.{a}.qk_norm.query_norm.weight", HeadDim);
                w.AddNorm($"b.{a}.qk_norm.key_norm.weight", HeadDim);
            }
            foreach (string n in new[] { "norm1", "norm2", "norm3" }) w.AddNorm($"b.{n}.weight", Hidden);
        }
        w.Linear("b.mlp.0", config.MlpDim, Hidden, biasRms);
        w.Linear("b.mlp.2", Hidden, config.MlpDim, biasRms);
        w.Linear("b.adaLN_modulation.1", 9 * Hidden, Hidden, biasRms);
        block.LoadWeights(w, "b");

        Tensor? cos = null, sin = null;
        if (!noRope)
        {
            (Tensor c, Tensor s) = new FLiteRope(HeadDim, 10000, 16).BuildDeviceTables(GridH, GridW, registers);
            o.Own(c); o.Own(s);
            // The device rotary reads one table row per (batch, token), so batch 2 needs the table repeated per item.
            cos = batch2 ? o.Own(Repeat(c, batch)) : c;
            sin = batch2 ? o.Own(Repeat(s, batch)) : s;
        }
        Tensor x = o.Own(Seeded(new TensorShape(batch, seq, Hidden), 2601));
        Tensor context = o.Own(Seeded(new TensorShape(batch, ctxLen, cross), 2602));
        Tensor temb = o.Own(Seeded(new TensorShape(batch, Hidden), 2603, 0.5f));
        (Tensor outX, Tensor v) = block.Forward(cpu, x, context, temb, null, cos, sin);
        o.Own(outX); o.Own(v);
        if (!vResidual) return Finish(o, ("out", outX), ("v", v));

        // Second block step: the first step's mixed V feeds the lambda residual.
        (Tensor outX2, Tensor v2) = block.Forward(cpu, outX, context, temb, v, cos, sin);
        o.Own(outX2); o.Own(v2);
        return Finish(o, ("out", outX), ("v", v), ("out2", outX2), ("v2", v2));
    }

    // ── shared weight builders ─────────────────────────────────────────────────────────────────────────────────────

    /// <summary>diffusers dual-stream attention keys (to_q/k/v, add_q/k/v_proj, both output projections, four QK norms).</summary>
    private static void DualAttnWeights(Weights w, string p, bool qkvBias, bool outBias = true, bool norms = true, bool toAddOut = true)
    {
        foreach (string n in new[] { "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj" })
            w.Linear($"{p}.attn.{n}", Hidden, Hidden, qkvBias);
        w.Linear($"{p}.attn.to_out.0", Hidden, Hidden, outBias);
        if (toAddOut) w.Linear($"{p}.attn.to_add_out", Hidden, Hidden, outBias);
        if (!norms) return;
        foreach (string n in new[] { "norm_q", "norm_k", "norm_added_q", "norm_added_k" }) w.AddNorm($"{p}.attn.{n}.weight", HeadDim);
    }

    private static void GeluFfnWeights(Weights w, string p, int mlp = Mlp)
    {
        w.Linear($"{p}.net.0.proj", mlp, Hidden, true);
        w.Linear($"{p}.net.2", Hidden, mlp, true);
    }

    // ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────

    private sealed class Owner : List<Tensor>
    {
        public Tensor Own(Tensor t) { Add(t); return t; }
    }

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

    private static ReadOnlySpan<byte> Bytes(Tensor t) => new(t.DataPointer, checked((int)(t.ElementCount * sizeof(float))));

    private static string Hex(string label, ReadOnlySpan<byte> bytes) => $"{label}={Convert.ToHexStringLower(SHA256.HashData(bytes))}";

    /// <summary>Additive F32 mask <c>[batch, 1, rows, cols]</c>: 0 where <paramref name="allowed"/>, a large negative elsewhere.</summary>
    private static Tensor AdditiveMask(int batch, int rows, int cols, Func<int, int, bool> allowed)
    {
        Tensor mask = new(new TensorShape(batch, 1, rows, cols), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    p[((long)b * rows + i) * cols + j] = allowed(i, j) ? 0f : NegInf;
        return mask;
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor copy = new(source.Shape, DType.F32);
        new ReadOnlySpan<float>(source.DataPointer, checked((int)source.ElementCount)).CopyTo(new Span<float>(copy.DataPointer, checked((int)source.ElementCount)));
        return copy;
    }

    private static void AddInPlace(Tensor t, float value)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] += value;
    }

    /// <summary>Stacks <paramref name="table"/> <c>[1, S, D]</c> <paramref name="batch"/> times into <c>[batch, S, D]</c>.</summary>
    private static Tensor Repeat(Tensor table, int batch)
    {
        Tensor result = new(new TensorShape(batch, table.Shape[1], table.Shape[2]), DType.F32);
        long count = table.ElementCount;
        for (int b = 0; b < batch; b++)
            new ReadOnlySpan<float>(table.DataPointer, checked((int)count)).CopyTo(new Span<float>((float*)result.DataPointer + b * count, checked((int)count)));
        return result;
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

        public void AddBias(string name, long n) => base.Add(name, Seeded(new TensorShape(n), Seed(name), 0.1f));

        /// <summary>A weight <c>[out, in]</c> under <c>prefix.weight</c>, plus <c>prefix.bias</c> when asked.</summary>
        public void Linear(string prefix, long outFeatures, long inFeatures, bool bias = false)
        {
            Add($"{prefix}.weight", outFeatures, inFeatures);
            if (bias) AddBias($"{prefix}.bias", outFeatures);
        }

        public void AddConst(string name, float value)
        {
            Tensor t = new(new TensorShape(1), DType.F32);
            ((float*)t.DataPointer)[0] = value;
            base.Add(name, t);
        }

        /// <summary>Norm scale near 1.</summary>
        public void AddNorm(string name, long n)
        {
            Tensor t = Seeded(new TensorShape(n), Seed(name), 0.25f);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++) p[i] += 1f;
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

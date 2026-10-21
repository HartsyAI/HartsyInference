using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Models.Wav2Vec2Bert;
using HartsyInference.Audio.Models.ZipVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterHost2(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["sovits_refenc_fused_qkv"] = cpu => SoVitsRefEncCase(cpu, fused: true);
        d["sovits_refenc_split_qkv"] = cpu => SoVitsRefEncCase(cpu, fused: false);
        d["zipformer_attention_weights"] = ZipformerWeights;
        d["muq_embedder"] = MuqCase;
        d["w2vbert_conformer"] = W2vBertConformer;
        d["indextts_conformer"] = IndexTtsConformer;
        d["demucs_self"] = cpu => DemucsLayer(cpu, cross: false);
        d["demucs_cross"] = cpu => DemucsLayer(cpu, cross: true);
    }

    // ── GPT-SoVITS reference encoder (scale 1/sqrt(hidden), not head dim; fused or split projections) ───────────────

    private static string SoVitsRefEncCase(CpuBackend cpu, bool fused)
    {
        const int mels = 24, hidden = 16, style = 12, heads = 2, kernel = 5, t = 9;
        Owner o = new();
        Weights w = new(fused ? 610u : 611u);
        w.Linear("r.spectral.0.fc", hidden, mels, true); w.Linear("r.spectral.3.fc", hidden, hidden, true);
        w.AddTensor("r.temporal.0.conv1.conv.weight", 0.3f, 2 * hidden, hidden, kernel); w.AddBias("r.temporal.0.conv1.conv.bias", 2 * hidden);
        w.AddTensor("r.temporal.1.conv1.conv.weight", 0.3f, 2 * hidden, hidden, kernel); w.AddBias("r.temporal.1.conv1.conv.bias", 2 * hidden);
        if (fused) w.Linear("r.slf_attn.qkv", 3 * hidden, hidden, true);
        else
        {
            w.Linear("r.slf_attn.w_qs", hidden, hidden, true); w.Linear("r.slf_attn.w_ks", hidden, hidden, true); w.Linear("r.slf_attn.w_vs", hidden, hidden, true);
        }
        w.Linear("r.slf_attn.fc", hidden, hidden, true); w.Linear("r.fc.fc", style, hidden, true);
        SoVitsRefEnc enc = new(mels, hidden, style, heads, kernel);
        enc.LoadWeights(w, "r");
        Tensor y = o.Own(enc.Forward(cpu, Seq(o, 6101, 1f, 1, mels, t), t));
        return Finish(o, ("out", y));
    }

    // ── Zipformer attention weights (softmax probabilities shared by two value paths; cannot be expressed on the shared layer) ──

    private static string ZipformerWeights(CpuBackend cpu)
    {
        const int dim = 16, heads = 2, qhd = 4, phd = 2, posDim = 8, t = 7;
        Owner o = new();
        Weights w = new(612);
        w.Linear("z.in_proj", 2 * heads * qhd + heads * phd, dim, true);
        w.Add("z.linear_pos.weight", heads * phd, posDim);
        ZipformerAttentionWeights attn = new(dim, heads, qhd, phd, posDim);
        attn.LoadWeights(w, "z");
        Tensor pos = o.Own(ZipformerRelPosEncoding.Build(t, posDim));
        Tensor y = o.Own(attn.Forward(cpu, Seq(o, 6121, 1f, 1, t, dim), pos, t));
        return Finish(o, ("probs", y));
    }

    // ── HT-Demucs transformer layers (private nested Layer; host Q/K/V matmul, group-norm over all tokens) ─────────

    private static string DemucsLayer(CpuBackend cpu, bool cross)
    {
        const int dim = 16, ffn = 24, nq = 7, nkv = 5;
        Owner o = new();
        HtDemucsConfig cfg = new() { BottomChannels = dim, THeads = 2, HiddenScale = 1.5f };
        Weights w = new(cross ? 620u : 621u);
        string attn = cross ? "cross_attn" : "self_attn";
        w.Add($"l.{attn}.in_proj_weight", 3 * dim, dim); w.AddBias($"l.{attn}.in_proj_bias", 3 * dim);
        w.Linear($"l.{attn}.out_proj", dim, dim, true);
        foreach (string n in cross ? new[] { "norm1", "norm2", "norm3", "norm_out" } : new[] { "norm1", "norm2", "norm_out" })
        {
            w.AddNorm($"l.{n}.weight", dim); w.AddBias($"l.{n}.bias", dim);
        }
        w.Linear("l.linear1", ffn, dim, true); w.Linear("l.linear2", dim, ffn, true);
        w.AddTensor("l.gamma_1.scale", 0.5f, dim); w.AddTensor("l.gamma_2.scale", 0.5f, dim);
        using DemucsCastOwner casts = new();
        object layer = NewPrivate(typeof(DemucsCrossTransformer), "Layer", cfg, cross);
        Invoke(layer, "LoadWeights", w, "l", casts);
        Tensor q = Seq(o, 6201, 1f, 1, nq, dim), kv = cross ? Seq(o, 6202, 1f, 1, nkv, dim) : q;
        Tensor y = o.Own((Tensor)Invoke(layer, "Forward", cpu, q, kv, nq, cross ? nkv : nq));
        return Finish(o, ("out", y));
    }

    // ── MuQ embedder (HeartMula): conv stem into a pre-norm transformer ─────────────────────────────────────────────

    private static string MuqCase(CpuBackend cpu)
    {
        const int mels = 12, muqDim = 10, lmHidden = 14, hidden = 16, layers = 2, heads = 2, ffn = 24, t = 20;
        Owner o = new();
        Weights w = new(630);
        w.AddTensor("m.conv_stem.0.weight", 0.3f, hidden, mels, 3); w.AddBias("m.conv_stem.0.bias", hidden);
        w.AddTensor("m.conv_stem.1.weight", 0.3f, hidden, hidden, 3); w.AddBias("m.conv_stem.1.bias", hidden);
        for (int i = 0; i < layers; i++)
        {
            string p = $"m.layers.{i}";
            foreach (string n in new[] { "ln1", "ln2" }) { w.AddNorm($"{p}.{n}.weight", hidden); w.AddBias($"{p}.{n}.bias", hidden); }
            foreach (string n in new[] { "q", "k", "v", "o" }) w.Linear($"{p}.{n}", hidden, hidden, true);
            w.Linear($"{p}.fc1", ffn, hidden, true); w.Linear($"{p}.fc2", hidden, ffn, true);
        }
        w.AddNorm("m.norm.weight", hidden); w.AddBias("m.norm.bias", hidden);
        w.Linear("m.proj", muqDim, hidden, true); w.Linear("m.muq_linear", lmHidden, muqDim, true);
        using MuqEmbedder muq = new(mels, muqDim, lmHidden, hidden, layers, heads, ffn);
        muq.LoadWeights(w, "m");
        Tensor y = o.Own(muq.Embed(cpu, Seq(o, 6301, 1f, 1, mels, t), t));
        return Finish(o, ("out", y));
    }

    // ── Wav2Vec2-BERT conformer layer (host relative-key attention with a clamped distance embedding: not a plain SDPA) ──

    private static string W2vBertConformer(CpuBackend cpu)
    {
        const int h = 16, inter = 24, heads = 2, kernel = 5, left = 4, right = 2, t = 9;
        Owner o = new();
        Wav2Vec2BertConfig cfg = new()
        {
            HiddenSize = h, IntermediateSize = inter, NumAttentionHeads = heads, ConvDepthwiseKernelSize = kernel,
            LeftMaxPositionEmbeddings = left, RightMaxPositionEmbeddings = right, FeatureProjectionInputDim = 8, NumLayersToRun = 1,
        };
        Weights w = new(631);
        foreach (string f in new[] { "ffn1", "ffn2" })
        {
            w.AddNorm($"c.{f}_layer_norm.weight", h); w.AddBias($"c.{f}_layer_norm.bias", h);
            w.Linear($"c.{f}.intermediate_dense", inter, h, true); w.Linear($"c.{f}.output_dense", h, inter, true);
        }
        w.AddNorm("c.self_attn_layer_norm.weight", h); w.AddBias("c.self_attn_layer_norm.bias", h);
        foreach (string n in new[] { "linear_q", "linear_k", "linear_v", "linear_out" }) w.Linear($"c.self_attn.{n}", h, h, true);
        w.AddTensor("c.self_attn.distance_embedding.weight", 0.3f, left + right + 1, h / heads);
        w.AddNorm("c.conv_module.layer_norm.weight", h); w.AddBias("c.conv_module.layer_norm.bias", h);
        w.AddTensor("c.conv_module.pointwise_conv1.weight", 0.3f, 2 * h, h, 1); w.AddTensor("c.conv_module.depthwise_conv.weight", 0.3f, h, 1, kernel);
        w.AddNorm("c.conv_module.depthwise_layer_norm.weight", h); w.AddBias("c.conv_module.depthwise_layer_norm.bias", h);
        w.AddTensor("c.conv_module.pointwise_conv2.weight", 0.3f, h, h, 1);
        w.AddNorm("c.final_layer_norm.weight", h); w.AddBias("c.final_layer_norm.bias", h);
        Wav2Vec2BertConformerLayer layer = new(cfg);
        layer.LoadWeights(w, "c");
        Tensor seq = Seq(o, 6311, 1f, 1, t, h);
        Tensor full = o.Own(layer.Forward(cpu, seq, t, t));
        Tensor padded = o.Own(layer.Forward(cpu, seq, t, t - 2));
        return Finish(o, ("all_frames", full), ("padded", padded));
    }

    // ── IndexTTS conformer layer (host Transformer-XL attention with pos_bias_u/v and no rel_shift) ─────────────────

    private static string IndexTtsConformer(CpuBackend cpu)
    {
        const int c = 16, heads = 2, ffn = 24, kernel = 5, t = 9;
        Owner o = new();
        IndexTtsConformerConfig cfg = new() { InputSize = 8, OutputSize = c, AttentionHeads = heads, LinearUnits = ffn, NumBlocks = 1, ConvKernel = kernel };
        Weights w = new(632);
        foreach (string n in new[] { "norm_mha", "norm_conv", "norm_ff", "norm_final" }) { w.AddNorm($"c.{n}.weight", c); w.AddBias($"c.{n}.bias", c); }
        foreach (string n in new[] { "linear_q", "linear_k", "linear_v", "linear_out" }) w.Linear($"c.self_attn.{n}", c, c, true);
        w.Add("c.self_attn.linear_pos.weight", c, c);
        w.AddTensor("c.self_attn.pos_bias_u", 0.3f, heads, c / heads); w.AddTensor("c.self_attn.pos_bias_v", 0.3f, heads, c / heads);
        w.AddTensor("c.conv_module.pointwise_conv1.weight", 0.3f, 2 * c, c, 1); w.AddBias("c.conv_module.pointwise_conv1.bias", 2 * c);
        w.AddTensor("c.conv_module.depthwise_conv.weight", 0.3f, c, 1, kernel); w.AddBias("c.conv_module.depthwise_conv.bias", c);
        w.AddNorm("c.conv_module.norm.weight", c); w.AddBias("c.conv_module.norm.bias", c);
        w.AddTensor("c.conv_module.pointwise_conv2.weight", 0.3f, c, c, 1); w.AddBias("c.conv_module.pointwise_conv2.bias", c);
        w.Linear("c.feed_forward.w_1", ffn, c, true); w.Linear("c.feed_forward.w_2", c, ffn, true);
        IndexTtsConformerLayer layer = new(cfg);
        layer.LoadWeights(w, "c");
        Tensor y = o.Own(layer.Forward(cpu, Seq(o, 6321, 1f, 1, t, c), Seq(o, 6322, 1f, 1, t, c), t));
        return Finish(o, ("out", y));
    }
}

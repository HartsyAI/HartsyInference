using HartsyInference.Audio.Models.BreezeTts;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterHost1(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["t5gemma2_tiny"] = T5Gemma2Tiny;
        d["vits_fft"] = VitsFft;
        d["vits_textenc"] = VitsTextEnc;
    }

    // ── T5Gemma2 text encoder (host float[] encoder: GQA 2/1, alternating sliding window and full layers) ───────────

    private static string T5Gemma2Tiny(CpuBackend cpu)
    {
        const int hidden = 16, inter = 24, heads = 2, kvHeads = 1, hd = 8, vocab = 20;
        T5Gemma2TextEncoderConfig cfg = new()
        {
            VocabSize = vocab, HiddenSize = hidden, IntermediateSize = inter, NumLayers = 3, NumHeads = heads, NumKvHeads = kvHeads,
            HeadDim = hd, QueryPreAttnScalar = hd, SlidingWindow = 3, SlidingPatternPeriod = 2, EoiTokenIndex = 19,
        };
        Weights w = new(600);
        w.Add("t.embed_tokens.weight", vocab, hidden);
        w.AddBias("t.norm.weight", hidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"t.layers.{i}";
            foreach (string n in new[] { "pre_self_attn_layernorm", "post_self_attn_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm" })
                w.AddBias($"{p}.{n}.weight", hidden);
            w.AddBias($"{p}.self_attn.q_norm.weight", hd); w.AddBias($"{p}.self_attn.k_norm.weight", hd);
            w.Add($"{p}.self_attn.q_proj.weight", heads * hd, hidden); w.Add($"{p}.self_attn.k_proj.weight", kvHeads * hd, hidden);
            w.Add($"{p}.self_attn.v_proj.weight", kvHeads * hd, hidden); w.Add($"{p}.self_attn.o_proj.weight", hidden, heads * hd);
            w.Add($"{p}.mlp.gate_proj.weight", inter, hidden); w.Add($"{p}.mlp.up_proj.weight", inter, hidden); w.Add($"{p}.mlp.down_proj.weight", hidden, inter);
        }
        using T5Gemma2TextEncoder enc = new(cfg);
        enc.LoadWeights(w, "t");
        float[] y = enc.Encode(cpu, new[] { 3, 7, 1, 12, 5, 9, 2, 14, 6 });
        return Digest("out", y);
    }

    private static void VitsEncoderWeights(Weights w, string prefix, int layers, int hidden, int heads, int filter, int kernel, int window)
    {
        int kCh = hidden / heads;
        for (int i = 0; i < layers; i++)
        {
            string p = $"{prefix}.attn_layers.{i}";
            foreach (string n in new[] { "conv_q", "conv_k", "conv_v", "conv_o" })
            {
                w.AddTensor($"{p}.{n}.weight", 0.3f, hidden, hidden, 1); w.AddBias($"{p}.{n}.bias", hidden);
            }
            w.AddTensor($"{p}.emb_rel_k", 0.3f, 1, 2 * window + 1, kCh); w.AddTensor($"{p}.emb_rel_v", 0.3f, 1, 2 * window + 1, kCh);
            foreach (string n in new[] { "norm_layers_1", "norm_layers_2" })
            {
                w.AddNorm($"{prefix}.{n}.{i}.gamma", hidden); w.AddBias($"{prefix}.{n}.{i}.beta", hidden);
            }
            w.AddTensor($"{prefix}.ffn_layers.{i}.conv_1.weight", 0.3f, filter, hidden, kernel); w.AddBias($"{prefix}.ffn_layers.{i}.conv_1.bias", filter);
            w.AddTensor($"{prefix}.ffn_layers.{i}.conv_2.weight", 0.3f, hidden, filter, kernel); w.AddBias($"{prefix}.ffn_layers.{i}.conv_2.bias", hidden);
        }
    }

    // ── VITS FFT block (relative-position attention with a q-dependent K term and a V term: cannot be expressed on the shared layer) ──

    private static string VitsFft(CpuBackend cpu)
    {
        const int hidden = 16, heads = 2, filter = 24, kernel = 3, window = 4, layers = 2, t = 10;
        Owner o = new();
        Weights w = new(601);
        VitsEncoderWeights(w, "f", layers, hidden, heads, filter, kernel, window);
        VitsFftBlock block = new(layers, hidden, heads, filter, kernel, window);
        block.LoadWeights(w, "f");
        Tensor x = Seeded(new TensorShape(1, hidden, t), 6011);   // Run consumes its input
        Tensor y = o.Own(block.Run(cpu, x, t));
        return Finish(o, ("out", y));
    }

    private static string VitsTextEnc(CpuBackend cpu)
    {
        const int hidden = 16, inter = 12, heads = 2, filter = 24, kernel = 3, window = 4, layers = 2, vocab = 20;
        Owner o = new();
        VitsConfig cfg = new()
        {
            NumVocab = vocab, InterChannels = inter, HiddenChannels = hidden, FilterChannels = filter, NumHeads = heads, NumEncoderLayers = layers,
            EncoderKernelSize = kernel, WindowSize = window,
        };
        Weights w = new(602);
        w.Add("e.emb.weight", vocab, hidden);
        VitsEncoderWeights(w, "e.encoder", layers, hidden, heads, filter, kernel, window);
        w.AddTensor("e.proj.weight", 0.3f, 2 * inter, hidden, 1); w.AddBias("e.proj.bias", 2 * inter);
        VitsTextEncoder enc = new(cfg);
        enc.LoadWeights(w, "e");
        (Tensor hiddenOut, Tensor mp, Tensor logs) = enc.Forward(cpu, new[] { 3, 7, 1, 12, 5, 9, 2, 14 });
        o.Own(hiddenOut); o.Own(mp); o.Own(logs);
        return Finish(o, ("hidden", hiddenOut), ("m_p", mp), ("logs_p", logs));
    }
}

using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Models.Mert2;
using HartsyInference.Audio.Models.SheetSage2;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterDevice2(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["heartcodec_b1"] = cpu => HeartCodec(cpu, 1);
        d["heartcodec_b2"] = cpu => HeartCodec(cpu, 2);
        d["mert2_layer"] = Mert2Layer;
        d["gpt_uncached_bias_causal"] = cpu => GptUncached(cpu, bias: true, causal: true);
        d["gpt_uncached_nobias_noncausal"] = cpu => GptUncached(cpu, bias: false, causal: false);
        d["gpt_cached"] = GptCached;
        d["sheetsage2_prefill_decode_cross"] = SheetSage2;
    }

    private static string Chain(params string[] digests) => string.Join(' ', digests);

    // ── HeartCodec estimator (stage 1: heads x headDim, stage 2: heads x 2*headDim) ─────────────────────────────

    private static string HeartCodec(CpuBackend cpu, int b)
    {
        const int inCh = 16, outCh = 8, heads = 2, hd = 8, inner = heads * hd, inner2 = 2 * inner, t = 6;
        Owner o = new();
        Weights w = new(90u + (uint)b);
        void Project(string p, int inDim, int filter)
        {
            w.AddTensor($"{p}.ffn_1.weight", 0.3f, filter, inDim, 3); w.AddBias($"{p}.ffn_1.bias", filter);
            w.Linear($"{p}.ffn_2", filter, filter, true);
        }
        void AdaLn(string p, int dim)
        {
            w.Linear($"{p}.emb.timestep_embedder.linear_1", dim, 512, true); w.Linear($"{p}.emb.timestep_embedder.linear_2", dim, dim, true);
            w.Linear($"{p}.linear", 6 * dim, dim, true);
        }
        void Blocks(string p, int dim)
        {
            w.AddNorm($"{p}.attn_norm.weight", dim); w.AddNorm($"{p}.mlp_norm.weight", dim);
            w.AddTensor($"{p}.scale_shift_table", 0.3f, 6, dim);
            foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "o_proj" }) w.Add($"{p}.attn.{n}.weight", dim, dim);
            w.Add($"{p}.mlp.gate.weight", 2 * dim, dim); w.Add($"{p}.mlp.up.weight", 2 * dim, dim); w.Add($"{p}.mlp.down.weight", dim, 2 * dim);
        }
        Project("e.proj_in", inCh, inner); Project("e.connection_proj", inCh + inner, inner2); Project("e.proj_out", inner2, outCh);
        AdaLn("e.adaln_single", inner); AdaLn("e.adaln_single_2", inner2);
        w.AddTensor("e.scale_shift_table", 0.3f, 2, inner); w.AddTensor("e.scale_shift_table_2", 0.3f, 2, inner2);
        Blocks("e.transformer_blocks.0", inner); Blocks("e.transformer_blocks_2.0", inner2);
        HeartCodecEstimator est = new(inCh, outCh, heads, hd, numLayers: 1, numLayers2: 1);
        est.LoadWeights(w, "e");
        Tensor input = Seq(o, 901, 1f, b, t, inCh);
        Tensor y = o.Own(est.Forward(cpu, input, b == 1 ? [0.4f] : [0.4f, 0.8f]));
        return Finish(o, ("out", y));
    }

    // ── Mert2 conformer layer (writes into preallocated scratch) ────────────────────────────────────────────────

    private static string Mert2Layer(CpuBackend cpu)
    {
        const int dim = 16, inter = 24, tokens = 9, kernel = 5;
        Owner o = new();
        Mert2Config cfg = new() { Dim = dim, Intermediate = inter, Heads = 2, ConformerConvKernel = kernel };
        Weights w = new(91);
        foreach (string f in new[] { "ffn1", "ffn2" })
        {
            w.AddNorm($"l.{f}_layer_norm.weight", dim); w.AddBias($"l.{f}_layer_norm.bias", dim);
            w.Linear($"l.{f}.w_1", inter, dim, true); w.Linear($"l.{f}.w_2", dim, inter, true);
        }
        w.AddNorm("l.attn_layer_norm.weight", dim); w.AddBias("l.attn_layer_norm.bias", dim);
        foreach (string n in new[] { "query_proj", "key_proj", "value_proj", "out_proj" }) w.Linear($"l.attn.{n}", dim, dim, true);
        w.AddNorm("l.conv_module.layer_norm.weight", dim); w.AddBias("l.conv_module.layer_norm.bias", dim);
        w.AddTensor("l.conv_module.conv_block.1.weight", 0.3f, 2 * dim, dim, 1);
        w.AddTensor("l.conv_module.conv_block.3.weight", 0.3f, dim, 1, kernel);
        w.AddNorm("l.conv_module.conv_block.4.1.weight", dim); w.AddBias("l.conv_module.conv_block.4.1.bias", dim);
        w.AddTensor("l.conv_module.conv_block.6.weight", 0.3f, dim, dim, 1);
        w.AddNorm("l.final_layer_norm.weight", dim); w.AddBias("l.final_layer_norm.bias", dim);
        List<Tensor> owned = [];
        Mert2ConformerLayer layer = new(cfg);
        layer.LoadWeights(w, "l", owned);
        foreach (Tensor t in owned) o.Own(t);
        using Mert2ConformerLayer.Scratch scratch = new(cfg, tokens);
        Tensor input = Seq(o, 911, 1f, tokens, dim), output = o.Own(new Tensor(new TensorShape(tokens, dim), DType.F32));
        layer.Forward(cpu, input, output, scratch);
        return Finish(o, ("out", output));
    }

    // ── GPT block (Bark/IndexTTS) ───────────────────────────────────────────────────────────────────────────────

    private static Weights GptWeights(uint salt, GptConfig cfg)
    {
        int h = cfg.Hidden;
        Weights w = new(salt);
        w.AddNorm("g.layernorm_1.weight", h); w.AddNorm("g.layernorm_2.weight", h);
        w.Add("g.attn.att_proj.weight", 3 * h, h); w.Add("g.attn.out_proj.weight", h, h);
        w.Add("g.mlp.in_proj.weight", cfg.MlpDim, h); w.Add("g.mlp.out_proj.weight", h, cfg.MlpDim);
        if (cfg.Bias)
        {
            w.AddBias("g.layernorm_1.bias", h); w.AddBias("g.layernorm_2.bias", h);
            w.AddBias("g.attn.att_proj.bias", 3 * h); w.AddBias("g.attn.out_proj.bias", h);
            w.AddBias("g.mlp.in_proj.bias", cfg.MlpDim); w.AddBias("g.mlp.out_proj.bias", h);
        }
        return w;
    }

    private static string GptUncached(CpuBackend cpu, bool bias, bool causal)
    {
        const int t = 7;
        Owner o = new();
        GptConfig cfg = new() { Hidden = 32, NumLayers = 1, NumHeads = 2, Bias = bias };
        using GptBlock block = new(cfg);
        block.LoadWeights(GptWeights(bias ? 921u : 922u, cfg), "g");
        Tensor x = Seq(o, 920, 1f, 1, t, 32);
        Tensor? mask = causal ? o.Own(AdditiveMask(t, t, (i, j) => j <= i)) : null;
        Tensor y = o.Own(block.Forward(cpu, x, mask));
        return Finish(o, ("out", y));
    }

    private static string GptCached(CpuBackend cpu)
    {
        Owner o = new();
        GptConfig cfg = new() { Hidden = 32, NumLayers = 1, NumHeads = 2, Bias = true };
        using GptBlock block = new(cfg);
        block.LoadWeights(GptWeights(923, cfg), "g");
        using FixedKvCache cache = new(1, 1, 2, 16, 16);
        Tensor x0 = Seq(o, 924, 1f, 1, 3, 32), x1 = Seq(o, 925, 1f, 1, 1, 32), x2 = Seq(o, 926, 1f, 1, 1, 32);
        Tensor y0 = o.Own(block.ForwardCached(cpu, x0, cache, 0, 0));
        cache.AdvanceLength(3);
        Tensor y1 = o.Own(block.ForwardCached(cpu, x1, cache, 0, 3));
        cache.AdvanceLength(1);
        Tensor y2 = o.Own(block.ForwardCached(cpu, x2, cache, 0, 4));
        return Finish(o, ("prefill", y0), ("step3", y1), ("step4", y2));
    }

    // ── SheetSage2 decoder: causal prefill, one-token decode steps and the cross-attention against the memory ────

    private static string SheetSage2(CpuBackend cpu)
    {
        const int dim = 16, inner = 24, memTokens = 6;
        Owner o = new();
        SheetSage2Config cfg = new() { Dim = dim, IntermediateSize = inner, NumHeads = 2, Layers = 2, MaxTokens = 16 };
        ScoreTokenizer tokenizer = new();
        int vocab = tokenizer.TokenCount;
        Weights w = new(92);
        w.Add("token_embedding.weight", vocab, dim);
        w.AddTensor("decoder.embed_positions.weight", 0.3f, cfg.PositionRows, dim);
        w.AddNorm("decoder.layernorm_embedding.weight", dim); w.AddBias("decoder.layernorm_embedding.bias", dim);
        w.Add("output_projection.weight", vocab, dim);
        for (int i = 0; i < cfg.Layers; i++)
        {
            string p = $"decoder.layers.{i}";
            foreach (string a in new[] { "self_attn", "encoder_attn" })
                foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"{p}.{a}.{n}", dim, dim, true);
            foreach (string n in new[] { "self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm" })
            {
                w.AddNorm($"{p}.{n}.weight", dim); w.AddBias($"{p}.{n}.bias", dim);
            }
            w.Linear($"{p}.fc1", inner, dim, true); w.Linear($"{p}.fc2", dim, inner, true);
        }
        using SheetSage2Decoder dec = new(cfg, tokenizer);
        dec.LoadWeights(w);
        Tensor memory = Seq(o, 931, 1f, 1, memTokens, dim);
        using SheetSage2DecodeState state = dec.StartDecode(cpu, memory, 12);
        List<string> digests = [];
        int[][] feeds = [[1, 5, 9, 3], [7], [11], [4]];
        for (int i = 0; i < feeds.Length; i++)
        {
            dec.Decode(cpu, feeds[i], state);
            digests.Add(Digest($"step{i}", state.Logits));
        }
        Finish(o);
        return Chain([.. digests]);
    }
}

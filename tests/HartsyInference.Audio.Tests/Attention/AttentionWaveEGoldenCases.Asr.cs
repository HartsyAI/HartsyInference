using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterAsr(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["whisper_enc_b1"] = cpu => WhisperEncoder(cpu, 1, 11);
        d["whisper_enc_b2"] = cpu => WhisperEncoder(cpu, 2, 12);
        d["whisper_dec_prefill4_steps3"] = WhisperDecoderSteps;
        d["moonshine_enc"] = MoonshineEncoderLayer;
        d["moonshine_dec_prefill_steps"] = MoonshineDecoderSteps;
        d["moonshine_stream_enc"] = MoonshineStreamEncoder;
        d["moonshine_stream_dec"] = MoonshineStreamDecoder;
    }

    // ── Whisper ─────────────────────────────────────────────────────────────────────────────────────────────────

    private const int WhD = 16, WhHeads = 2, WhInter = 32, WhVocab = 24, WhMels = 8, WhMaxText = 16, WhMaxAudio = 16;

    private static WhisperConfig WhisperTiny() => new()
    {
        VocabSize = WhVocab, NumMelBins = WhMels, MaxAudioPositions = WhMaxAudio, MaxTextPositions = WhMaxText,
        EncoderLayers = 2, DecoderLayers = 2, HiddenSize = WhD, NumHeads = WhHeads, IntermediateSize = WhInter,
    };

    private static void WhisperLayerWeights(Weights w, string p, bool decoder)
    {
        void Block(string ln, string attn)
        {
            w.AddNorm($"{p}.{ln}.weight", WhD); w.AddBias($"{p}.{ln}.bias", WhD);
            w.Linear($"{p}.{attn}.q_proj", WhD, WhD, true);
            w.Add($"{p}.{attn}.k_proj.weight", WhD, WhD);
            w.Linear($"{p}.{attn}.v_proj", WhD, WhD, true);
            w.Linear($"{p}.{attn}.out_proj", WhD, WhD, true);
        }
        Block("self_attn_layer_norm", "self_attn");
        if (decoder) Block("encoder_attn_layer_norm", "encoder_attn");
        w.Linear($"{p}.fc1", WhInter, WhD, true);
        w.Linear($"{p}.fc2", WhD, WhInter, true);
        w.AddNorm($"{p}.final_layer_norm.weight", WhD); w.AddBias($"{p}.final_layer_norm.bias", WhD);
    }

    private static string WhisperEncoder(CpuBackend cpu, int batch, uint salt)
    {
        Owner o = new();
        WhisperConfig cfg = WhisperTiny();
        Weights w = new(salt);
        w.AddTensor("e.conv1.weight", 0.3f, WhD, WhMels, 3); w.AddBias("e.conv1.bias", WhD);
        w.AddTensor("e.conv2.weight", 0.3f, WhD, WhD, 3); w.AddBias("e.conv2.bias", WhD);
        w.AddTensor("e.embed_positions.weight", 0.3f, WhMaxAudio, WhD);
        for (int i = 0; i < cfg.EncoderLayers; i++) WhisperLayerWeights(w, $"e.layers.{i}", false);
        w.AddNorm("e.layer_norm.weight", WhD); w.AddBias("e.layer_norm.bias", WhD);
        using WhisperEncoder enc = new(cfg);
        enc.LoadWeights(w, "e");
        Tensor mel = Seq(o, 100 + salt, 1f, batch, WhMels, 20);
        Tensor y = o.Own(enc.Forward(cpu, mel));
        return Finish(o, ("out", y));
    }

    private static string WhisperDecoderSteps(CpuBackend cpu)
    {
        Owner o = new();
        WhisperConfig cfg = WhisperTiny();
        Weights w = new(13);
        w.Add("d.embed_tokens.weight", WhVocab, WhD);
        w.AddTensor("d.embed_positions.weight", 0.3f, WhMaxText, WhD);
        for (int i = 0; i < cfg.DecoderLayers; i++) WhisperLayerWeights(w, $"d.layers.{i}", true);
        w.AddNorm("d.layer_norm.weight", WhD); w.AddBias("d.layer_norm.bias", WhD);
        using WhisperDecoder dec = new(cfg);
        dec.LoadWeights(w, "d");
        Tensor enc = Seq(o, 131, 1f, 1, 9, WhD);
        using WhisperDecoder.DecodeState state = dec.StartDecode(cpu, enc);
        // Prefill four tokens, then three one-token steps (n==1 takes the all-zero mask path).
        int[][] feeds = [[3, 5, 7, 2], [9], [4], [11]];
        List<(string, Tensor)> outs = [];
        for (int i = 0; i < feeds.Length; i++)
            outs.Add(($"step{i}", o.Own(dec.DecodeStep(cpu, feeds[i], state))));
        outs.Add(("pos", o.Own(PosTensor(state.CurrentPos))));
        return Finish(o, [.. outs]);
    }

    private static Tensor PosTensor(int n)
    {
        Tensor t = new(new TensorShape(1), DType.F32);
        ((float*)t.DataPointer)[0] = n;
        return t;
    }

    // ── Moonshine ───────────────────────────────────────────────────────────────────────────────────────────────

    private static MoonshineConfig MoonTiny() => new()
    {
        HiddenSize = 16, NumHeads = 2, IntermediateSize = 24, EncoderLayers = 1, DecoderLayers = 2, VocabSize = 20,
        MaxTextPositions = 16,
    };

    private static void MoonAttn(Weights w, string p, string name, int d, int qkv)
    {
        w.Add($"{p}.{name}.q_proj.weight", qkv, d);
        w.Add($"{p}.{name}.k_proj.weight", qkv, d);
        w.Add($"{p}.{name}.v_proj.weight", qkv, d);
        w.Add($"{p}.{name}.o_proj.weight", d, qkv);
    }

    private static string MoonshineEncoderLayer(CpuBackend cpu)
    {
        Owner o = new();
        MoonshineConfig cfg = MoonTiny();
        Weights w = new(21);
        w.AddNorm("l.input_layernorm.weight", 16);
        MoonAttn(w, "l", "self_attn", 16, 16);
        w.AddNorm("l.post_attention_layernorm.weight", 16);
        w.Linear("l.mlp.fc1", 24, 16, true); w.Linear("l.mlp.fc2", 16, 24, true);
        MoonshineEncoderLayer layer = new(cfg);
        layer.LoadWeights(w, "l");
        (float[] cos, float[] sin) = HartsyInference.Audio.Models.Moonshine.RotaryEmbedding.GetTables(cfg.RotaryDim, cfg.RopeTheta, 64);
        Tensor x = Seq(o, 211, 1f, 1, 10, 16);
        Tensor y = o.Own(layer.Forward(cpu, x, 10, cos, sin));
        return Finish(o, ("out", y));
    }

    private static void MoonDecoderWeights(Weights w, string p, MoonshineConfig cfg, int encDim)
    {
        int d = cfg.HiddenSize, inter = cfg.IntermediateSize;
        w.Add($"{p}.embed_tokens.weight", cfg.VocabSize, d);
        for (int i = 0; i < cfg.DecoderLayers; i++)
        {
            string l = $"{p}.layers.{i}";
            w.AddNorm($"{l}.input_layernorm.weight", d);
            MoonAttn(w, l, "self_attn", d, d);
            w.AddNorm($"{l}.post_attention_layernorm.weight", d);
            MoonAttn(w, l, "encoder_attn", d, d);
            w.AddNorm($"{l}.final_layernorm.weight", d);
            w.Linear($"{l}.mlp.fc1", 2 * inter, d, true); w.Linear($"{l}.mlp.fc2", d, inter, true);
        }
        w.AddNorm($"{p}.norm.weight", d);
    }

    private static string RunDecoderFeeds(CpuBackend cpu, Owner o, Func<ReadOnlySpan<int>, Tensor> step)
    {
        int[][] feeds = [[1, 4, 6], [8], [3], [5]];
        List<(string, Tensor)> outs = [];
        for (int i = 0; i < feeds.Length; i++) outs.Add(($"step{i}", o.Own(step(feeds[i]))));
        return Finish(o, [.. outs]);
    }

    private static string MoonshineDecoderSteps(CpuBackend cpu)
    {
        Owner o = new();
        MoonshineConfig cfg = MoonTiny();
        Weights w = new(22);
        MoonDecoderWeights(w, "d", cfg, 16);
        using MoonshineDecoder dec = new(cfg);
        dec.LoadWeights(w, "d");
        Tensor enc = Seq(o, 221, 1f, 1, 7, 16);
        using MoonshineDecoder.DecodeState state = dec.StartDecode(cpu, enc);
        return RunDecoderFeeds(cpu, o, ids => dec.DecodeStep(cpu, ids, state));
    }

    private static MoonshineConfig MoonStreamTiny() => new()
    {
        HiddenSize = 16, NumHeads = 2, IntermediateSize = 24, DecoderLayers = 2, VocabSize = 20, MaxTextPositions = 16,
        TieWordEmbeddings = false, EncoderHiddenSize = 12, EncoderNumHeads = 2, EncoderHeadDim = 6, EncoderIntermediateSize = 20,
    };

    private static string MoonshineStreamEncoder(CpuBackend cpu)
    {
        // seq 40 > the 16-key left window, so both the (16, 4) and (16, 0) windows mask real keys.
        Owner o = new();
        MoonshineConfig cfg = MoonStreamTiny();
        const int d = 12, qkv = 12, inter = 20, seq = 40;
        Weights w = new(23);
        foreach (string p in new[] { "a", "b" })
        {
            w.AddTensor($"{p}.input_layernorm.gamma", 0.2f, d);
            MoonAttn(w, p, "self_attn", d, qkv);
            w.AddTensor($"{p}.post_attention_layernorm.gamma", 0.2f, d);
            w.Linear($"{p}.mlp.fc1", inter, d, true); w.Linear($"{p}.mlp.fc2", d, inter, true);
        }
        MoonshineStreamingEncoderLayer a = new(cfg, (16, 4)), b = new(cfg, (16, 0));
        a.LoadWeights(w, "a"); b.LoadWeights(w, "b");
        Tensor x = Seq(o, 231, 1f, 1, seq, d);
        Tensor ya = o.Own(a.Forward(cpu, x, seq));
        Tensor yb = o.Own(b.Forward(cpu, x, seq));
        return Finish(o, ("win16_4", ya), ("win16_0", yb));
    }

    private static string MoonshineStreamDecoder(CpuBackend cpu)
    {
        Owner o = new();
        MoonshineConfig cfg = MoonStreamTiny();
        Weights w = new(24);
        MoonDecoderWeights(w, "d", cfg, 12);
        w.Add("proj_out.weight", cfg.VocabSize, cfg.HiddenSize);
        w.AddTensor("d.pos_emb.weight", 0.3f, 16, 12);
        w.Add("d.proj.weight", cfg.HiddenSize, 12);
        using MoonshineStreamingDecoder dec = new(cfg);
        dec.LoadWeights(w, "d");
        Tensor enc = Seq(o, 241, 1f, 1, 7, 12);
        using MoonshineDecoder.DecodeState state = dec.StartDecode(cpu, enc);
        return RunDecoderFeeds(cpu, o, ids => dec.DecodeStep(cpu, ids, state));
    }
}

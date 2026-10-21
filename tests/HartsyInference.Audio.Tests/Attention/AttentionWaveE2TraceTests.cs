using System.Reflection;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Audio.Models.StyleTts2;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

/// <summary>The op sequence of each host-glue and encoder attention site on the recording stub (no compute): every backend op in order, the
/// attention call spelled out (route, scale, F16 assertion, operand and mask shapes) and which operands the site had freed by the end. The pins
/// were recorded from the code before the sites moved onto the shared layer's operator stage and must not change. Host reshapes, rotations and
/// masks are not backend ops, so only their results show, through the operand shapes.</summary>
public sealed class AttentionWaveE2TraceTests
{
    private static Tensor T(params long[] dims) => new(new TensorShape(dims), DType.F32);

    private static void Pin(string site, string[] actual)
    {
        if (!AttentionTraceOps.TryRecord(site, actual)) Assert.Equal(AttentionWaveE2TracePins.All[site], actual);
    }

    private static object NewPrivate(Type owner, string nested, params object[] ctorArgs) =>
        Activator.CreateInstance(owner.GetNestedType(nested, BindingFlags.NonPublic | BindingFlags.Public)!,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, ctorArgs, null)!;

    private static object? Invoke(object target, string method, params object?[] args) =>
        target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(target, args);

    // ── Reshape family: host reshapes stay, the attention call moved ────────────────────────────────────────────

    [Fact]
    public void HubertLayer_ReshapesOnTheHostThenAttendsDenseUnmasked()
    {
        const int h = 24, ffn = 32, t = 9;
        HubertConfig cfg = new() { Hidden = h, NumHeads = 2, FfnDim = ffn };
        Weights w = new(130);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"l.attention.{n}", h, h, true);
        w.AddNorm("l.layer_norm.weight", h); w.AddBias("l.layer_norm.bias", h);
        w.Linear("l.feed_forward.intermediate_dense", ffn, h, true); w.Linear("l.feed_forward.output_dense", h, ffn, true);
        w.AddNorm("l.final_layer_norm.weight", h); w.AddBias("l.final_layer_norm.bias", h);
        object layer = NewPrivate(typeof(Hubert), "Layer", cfg);
        Invoke(layer, "LoadWeights", w, "l");
        Pin("hubert", AttentionTraceOps.Record(b => Invoke(layer, "Forward", b, T(1, t, h), t)));
    }

    [Fact]
    public void NeuCodecTxBlock_AttendsDenseUnmasked()
    {
        const int dim = 16, mlp = 24, t = 9;
        NeuCodecConfig cfg = new() { BackboneDim = dim, NumHeads = 2, MlpDim = mlp };
        Weights w = new(131);
        w.AddNorm("b.input_layernorm.weight", dim); w.AddNorm("b.post_attention_layernorm.weight", dim);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "o_proj" }) w.Add($"b.self_attn.{n}.weight", dim, dim);
        w.Add("b.mlp.fc1.weight", mlp, dim); w.Add("b.mlp.fc2.weight", dim, mlp);
        NeuCodecDecoder dec = new(cfg);
        object tx = typeof(NeuCodecDecoder).GetMethod("LoadTx", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, [w, "b"])!;
        MethodInfo block = typeof(NeuCodecDecoder).GetMethod("TxBlock", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Pin("neucodec_txblock", AttentionTraceOps.Record(b => block.Invoke(dec, [b, T(1, t, dim), tx, t])));
    }

    [Fact]
    public void MuqEmbedder_AttendsDenseUnmaskedAfterTheConvStem()
    {
        const int mels = 12, muqDim = 10, lmHidden = 14, hidden = 16, layers = 2, ffn = 24, t = 20;
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
        using MuqEmbedder muq = new(mels, muqDim, lmHidden, hidden, layers, 2, ffn);
        muq.LoadWeights(w, "m");
        Pin("muq", AttentionTraceOps.Record(b => muq.Embed(b, T(1, mels, t), t), serveConvs: true));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SoVitsRefEnc_AttendsWithTheModelWidthScale(bool fused)
    {
        const int mels = 24, hidden = 16, style = 12, kernel = 5, t = 9;
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
        SoVitsRefEnc enc = new(mels, hidden, style, 2, kernel);
        enc.LoadWeights(w, "r");
        Pin(fused ? "sovits_refenc_fused" : "sovits_refenc_split", AttentionTraceOps.Record(b => enc.Forward(b, T(1, mels, t), t), serveConvs: true));
    }

    [Fact]
    public void ChatterboxPerceiver_AttendsCrossThenSelf()
    {
        const int dim = 1024;
        Weights w = new(132);
        w.AddTensor("p.pre_attention_query", 0.5f, 1, 32, dim);
        w.AddNorm("p.attn.norm.weight", dim); w.AddBias("p.attn.norm.bias", dim);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "proj_out" }) { w.AddTensor($"p.attn.{n}.weight", 0.03f, dim, dim); w.AddBias($"p.attn.{n}.bias", dim); }
        using ChatterboxPerceiver perceiver = new();
        perceiver.LoadWeights(w, "p");
        MethodInfo block = typeof(ChatterboxPerceiver).GetMethod("AttentionBlock", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Pin("chatterbox_perceiver", AttentionTraceOps.Record(b =>
        {
            Tensor queries = T(1, 32, dim), prompt = T(1, 11, dim);
            block.Invoke(perceiver, [b, queries, prompt]);
            block.Invoke(perceiver, [b, queries, queries]);
        }));
    }

    [Fact]
    public void S3Block_AttendsDenseUnmaskedAroundTheFsmnBranch()
    {
        const int n = 1280, ff = 2560, t = 9;
        Weights w = new(160);
        w.AddNorm("b.attn_ln.weight", n); w.AddBias("b.attn_ln.bias", n);
        foreach (string x in new[] { "query", "value", "out" }) { w.AddTensor($"b.attn.{x}.weight", 0.02f, n, n); w.AddBias($"b.attn.{x}.bias", n); }
        w.AddTensor("b.attn.key.weight", 0.02f, n, n);
        w.AddTensor("b.attn.fsmn_block.weight", 0.3f, n, 1, 31);
        w.AddNorm("b.mlp_ln.weight", n); w.AddBias("b.mlp_ln.bias", n);
        w.AddTensor("b.mlp.0.weight", 0.02f, ff, n); w.AddBias("b.mlp.0.bias", ff);
        w.AddTensor("b.mlp.2.weight", 0.02f, n, ff); w.AddBias("b.mlp.2.bias", n);
        S3Block block = new();
        block.LoadWeights(w, "b");
        float[] cos = new float[t * 32], sin = new float[t * 32];
        Pin("s3_block", AttentionTraceOps.Record(b => block.Forward(b, T(1, t, n), t, cos, sin), serveConvs: true));
    }

    [Fact]
    public void Text2SemanticBlock_AttendsWithThePrefixMaskAsBias()
    {
        const int h = 32, ffn = 48, t = 9;
        Text2SemanticConfig cfg = new() { Hidden = h, NumHeads = 2, FfnDim = ffn };
        Weights w = new(161);
        w.Linear("b.self_attn.out_proj", h, h, true);
        w.Add("b.self_attn.in_proj_weight", 3 * h, h); w.AddBias("b.self_attn.in_proj_bias", 3 * h);
        w.Linear("b.linear1", ffn, h, true); w.Linear("b.linear2", h, ffn, true);
        foreach (string n in new[] { "norm1", "norm2" }) { w.AddNorm($"b.{n}.weight", h); w.AddBias($"b.{n}.bias", h); }
        object block = NewPrivate(typeof(Text2Semantic), "Block", cfg);
        Invoke(block, "LoadWeights", w, "b");
        using Tensor mask = T(t, t);
        Pin("text2semantic", AttentionTraceOps.Record(b => Invoke(block, "Forward", b, T(1, t, h), t, mask)));
    }

    // ── Operator stage only ─────────────────────────────────────────────────────────────────────────────────────

    private static MoonshineConfig MoonTiny() => new()
    {
        HiddenSize = 16, NumHeads = 2, IntermediateSize = 24, EncoderLayers = 1, DecoderLayers = 2, VocabSize = 20, MaxTextPositions = 16,
    };

    private static void MoonAttn(Weights w, string p, string name, int d, int qkv)
    {
        w.Add($"{p}.{name}.q_proj.weight", qkv, d); w.Add($"{p}.{name}.k_proj.weight", qkv, d);
        w.Add($"{p}.{name}.v_proj.weight", qkv, d); w.Add($"{p}.{name}.o_proj.weight", d, qkv);
    }

    [Fact]
    public void MoonshineEncoderLayer_AttendsDenseUnmasked()
    {
        MoonshineConfig cfg = MoonTiny();
        Weights w = new(21);
        w.AddNorm("l.input_layernorm.weight", 16);
        MoonAttn(w, "l", "self_attn", 16, 16);
        w.AddNorm("l.post_attention_layernorm.weight", 16);
        w.Linear("l.mlp.fc1", 24, 16, true); w.Linear("l.mlp.fc2", 16, 24, true);
        MoonshineEncoderLayer layer = new(cfg);
        layer.LoadWeights(w, "l");
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(cfg.RotaryDim, cfg.RopeTheta, 64);
        Pin("moonshine_enc", AttentionTraceOps.Record(b => layer.Forward(b, T(1, 10, 16), 10, cos, sin)));
    }

    [Fact]
    public void MoonshineDecoderLayer_AttendsSelfWithTheCausalMaskAndCrossOverBorrowedKv()
    {
        MoonshineConfig cfg = MoonTiny();
        const int d = 16, enc = 7;
        Weights w = new(22);
        w.AddNorm("l.input_layernorm.weight", d);
        MoonAttn(w, "l", "self_attn", d, d);
        w.AddNorm("l.post_attention_layernorm.weight", d);
        MoonAttn(w, "l", "encoder_attn", d, d);
        w.AddNorm("l.final_layernorm.weight", d);
        w.Linear("l.mlp.fc1", 2 * cfg.IntermediateSize, d, true); w.Linear("l.mlp.fc2", d, cfg.IntermediateSize, true);
        MoonshineDecoderLayer layer = new(cfg);
        layer.LoadWeights(w, "l");
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(cfg.RotaryDim, cfg.RopeTheta, 64);
        using Tensor selfK = T(1, 2, 16, 8), selfV = T(1, 2, 16, 8), crossK = T(1, 2, enc, 8), crossV = T(1, 2, enc, 8);
        Pin("moonshine_dec", AttentionTraceOps.Record(b =>
        {
            layer.PrecomputeCrossKv(b, T(1, enc, d), crossK, crossV, enc);
            layer.Forward(b, T(1, 3, d), selfK, selfV, crossK, crossV, 0, 3, cos, sin).Dispose();
            layer.Forward(b, T(1, 1, d), selfK, selfV, crossK, crossV, 3, 1, cos, sin).Dispose();
        }));
    }

    [Fact]
    public void MoonshineStreamingEncoderLayer_AttendsWithItsWindowMaskAndFreesIt()
    {
        MoonshineConfig cfg = new()
        {
            HiddenSize = 16, NumHeads = 2, IntermediateSize = 24, DecoderLayers = 2, VocabSize = 20, MaxTextPositions = 16,
            TieWordEmbeddings = false, EncoderHiddenSize = 12, EncoderNumHeads = 2, EncoderHeadDim = 6, EncoderIntermediateSize = 20,
        };
        Weights w = new(23);
        foreach (string p in new[] { "a", "b" })
        {
            w.AddTensor($"{p}.input_layernorm.gamma", 0.2f, 12);
            MoonAttn(w, p, "self_attn", 12, 12);
            w.AddTensor($"{p}.post_attention_layernorm.gamma", 0.2f, 12);
            w.Linear($"{p}.mlp.fc1", 20, 12, true); w.Linear($"{p}.mlp.fc2", 12, 20, true);
        }
        MoonshineStreamingEncoderLayer a = new(cfg, (16, 4)), b2 = new(cfg, (16, 0));
        a.LoadWeights(w, "a"); b2.LoadWeights(w, "b");
        Pin("moonshine_stream_enc", AttentionTraceOps.Record(b =>
        {
            a.Forward(b, T(1, 40, 12), 40).Dispose();
            b2.Forward(b, T(1, 40, 12), 40).Dispose();
        }));
    }

    private const int MmD = 16, MmLayers = 2, MmFfn = 24;

    private static MimiTransformer MimiTiny()
    {
        MimiConfig cfg = new() { TransformerDim = MmD, TransformerLayers = MmLayers, TransformerHeads = 2, TransformerFfnDim = MmFfn, TransformerContext = 6 };
        Weights w = new(40);
        for (int i = 0; i < MmLayers; i++)
        {
            string p = $"m.layers.{i}";
            w.AddNorm($"{p}.input_layernorm.weight", MmD); w.AddBias($"{p}.input_layernorm.bias", MmD);
            w.AddNorm($"{p}.post_attention_layernorm.weight", MmD); w.AddBias($"{p}.post_attention_layernorm.bias", MmD);
            foreach (string n in new[] { "q", "k", "v", "o" }) w.Add($"{p}.self_attn.{n}_proj.weight", MmD, MmD);
            w.Add($"{p}.mlp.fc1.weight", MmFfn, MmD); w.Add($"{p}.mlp.fc2.weight", MmD, MmFfn);
            w.AddTensor($"{p}.self_attn_layer_scale.scale", 0.5f, MmD); w.AddTensor($"{p}.mlp_layer_scale.scale", 0.5f, MmD);
        }
        MimiTransformer tr = new(cfg, "m");
        tr.LoadWeights(w);
        return tr;
    }

    [Fact]
    public void MimiForward_AttendsABatchOfTwoThroughTheWindowMask()
    {
        MimiTransformer tr = MimiTiny();
        Pin("mimi_forward", AttentionTraceOps.Record(b => tr.Forward(b, T(2, 10, MmD), 2, 10)));
    }

    [Fact]
    public void MimiStreaming_AttendsOverTheCacheOwnedKvAndNeverFreesIt()
    {
        MimiTransformer tr = MimiTiny();
        using MimiTransformerCache cache = new();
        Pin("mimi_stream", AttentionTraceOps.Record(b =>
        {
            tr.ForwardStreaming(b, T(1, 4, MmD), 4, cache).Dispose();
            tr.ForwardStreaming(b, T(1, 3, MmD), 3, cache).Dispose();
        }));
    }

    private const int PkD = 32, PkHeads = 4, PkFfn = 48, PkLayers = 2;

    private static Weights PocketWeights(uint salt, bool layerScale)
    {
        Weights w = new(salt);
        for (int i = 0; i < PkLayers; i++)
        {
            string p = $"p.layers.{i}";
            w.AddNorm($"{p}.norm1.weight", PkD); w.AddBias($"{p}.norm1.bias", PkD);
            w.AddNorm($"{p}.norm2.weight", PkD); w.AddBias($"{p}.norm2.bias", PkD);
            w.Add($"{p}.self_attn.in_proj.weight", 3 * PkD, PkD); w.Add($"{p}.self_attn.out_proj.weight", PkD, PkD);
            w.Add($"{p}.linear1.weight", PkFfn, PkD); w.Add($"{p}.linear2.weight", PkD, PkFfn);
            if (layerScale) { w.AddTensor($"{p}.layer_scale_1.scale", 0.5f, PkD); w.AddTensor($"{p}.layer_scale_2.scale", 0.5f, PkD); }
        }
        return w;
    }

    [Fact]
    public void PocketTtsBackbone_AttendsABatchOfTwoThroughTheWindowMask()
    {
        PocketTtsStreamingTransformer tr = new("p", PkD, PkHeads, PkLayers, PkFfn, context: 6, layerScale: true);
        tr.LoadWeights(PocketWeights(72, layerScale: true));
        Pin("pockettts_backbone", AttentionTraceOps.Record(b => tr.Forward(b, T(2, 10, PkD), 2, 10)));
    }

    [Fact]
    public void PocketTtsPrimed_AttendsOverThePrefixConcatenatedKv()
    {
        const int prefixLen = 4, t = 3, hd = PkD / PkHeads;
        PocketTtsStreamingTransformer open = new("p", PkD, PkHeads, PkLayers, PkFfn, context: null, layerScale: false);
        PocketTtsStreamingTransformer windowed = new("p", PkD, PkHeads, PkLayers, PkFfn, context: 5, layerScale: false);
        open.LoadWeights(PocketWeights(73, layerScale: false));
        windowed.LoadWeights(PocketWeights(73, layerScale: false));
        Tensor[] pk = [T(1, PkHeads, prefixLen, hd), T(1, PkHeads, prefixLen, hd)], pv = [T(1, PkHeads, prefixLen, hd), T(1, PkHeads, prefixLen, hd)];
        Pin("pockettts_primed", AttentionTraceOps.Record(b =>
        {
            open.ForwardPrimed(b, T(1, t, PkD), t, pk, pv, prefixLen).Dispose();
            windowed.ForwardPrimed(b, T(1, t, PkD), t, pk, pv, prefixLen).Dispose();
        }));
    }

    [Fact]
    public void PocketTtsMimiDecoder_AttendsThroughTheWindowMask()
    {
        const float s = 0.05f;
        Weights w = new(74);
        w.AddTensor("mimi.quantizer.output_proj.weight", s, 512, 32, 1);
        w.AddTensor("mimi.upsample.convtr.convtr.weight", s, 512, 1, 32);
        for (int i = 0; i < 2; i++)
        {
            string p = $"mimi.decoder_transformer.transformer.layers.{i}";
            w.AddNorm($"{p}.norm1.weight", 512); w.AddBias($"{p}.norm1.bias", 512);
            w.AddNorm($"{p}.norm2.weight", 512); w.AddBias($"{p}.norm2.bias", 512);
            w.AddTensor($"{p}.self_attn.in_proj.weight", s, 1536, 512); w.AddTensor($"{p}.self_attn.out_proj.weight", s, 512, 512);
            w.AddTensor($"{p}.linear1.weight", s, 2048, 512); w.AddTensor($"{p}.linear2.weight", s, 512, 2048);
            w.AddTensor($"{p}.layer_scale_1.scale", 0.5f, 512); w.AddTensor($"{p}.layer_scale_2.scale", 0.5f, 512);
        }
        string d = "mimi.decoder.model";
        w.AddTensor($"{d}.0.conv.weight", s, 512, 512, 7); w.AddBias($"{d}.0.conv.bias", 512);
        int[] upIdx = [2, 5, 8], rbIdx = [3, 6, 9], ratios = [6, 5, 4];
        for (int i = 0; i < 3; i++)
        {
            int inCh = (8 >> i) * 64, outCh = inCh / 2;
            w.AddTensor($"{d}.{upIdx[i]}.convtr.weight", s, inCh, outCh, 2 * ratios[i]); w.AddBias($"{d}.{upIdx[i]}.convtr.bias", outCh);
            w.AddTensor($"{d}.{rbIdx[i]}.block.1.conv.weight", s, outCh / 2, outCh, 3); w.AddBias($"{d}.{rbIdx[i]}.block.1.conv.bias", outCh / 2);
            w.AddTensor($"{d}.{rbIdx[i]}.block.3.conv.weight", s, outCh, outCh / 2, 1); w.AddBias($"{d}.{rbIdx[i]}.block.3.conv.bias", outCh);
        }
        w.AddTensor($"{d}.11.conv.weight", s, 1, 64, 3); w.AddBias($"{d}.11.conv.bias", 1);
        PocketTtsMimiDecoder dec = new();
        dec.LoadWeights(w);
        MethodInfo run = typeof(PocketTtsMimiDecoder).GetMethod("RunTransformer", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Pin("pockettts_mimi", AttentionTraceOps.Record(b => run.Invoke(dec, [b, T(1, 256, 512), 1, 256])));
    }

    [Fact]
    public void Qwen3TtsTransformerLayer_AttendsWithTheWindowMaskThenWithoutOneAtT1()
    {
        const int dim = 16, heads = 2, hd = 8, ffn = 24;
        Qwen3TtsVocoderConfig cfg = new() { TransformerDim = dim, TransformerHeads = heads, TransformerHeadDim = hd, TransformerFfnDim = ffn, SlidingWindow = 4 };
        Weights w = new(75);
        w.AddNorm("l.input_layernorm.weight", dim); w.AddNorm("l.post_attention_layernorm.weight", dim);
        w.Linear("l.self_attn.q_proj", heads * hd, dim, true); w.Linear("l.self_attn.k_proj", heads * hd, dim, true);
        w.Linear("l.self_attn.v_proj", heads * hd, dim, true); w.Linear("l.self_attn.o_proj", dim, heads * hd, true);
        w.Add("l.mlp.gate_proj.weight", ffn, dim); w.Add("l.mlp.up_proj.weight", ffn, dim); w.Add("l.mlp.down_proj.weight", dim, ffn);
        w.AddTensor("l.self_attn_layer_scale.scale", 0.5f, dim); w.AddTensor("l.mlp_layer_scale.scale", 0.5f, dim);
        Type layerType = typeof(Qwen3TtsVocoder).GetNestedType("TransformerLayer", BindingFlags.NonPublic)!;
        object layer = Activator.CreateInstance(layerType, cfg)!;
        layerType.GetMethod("LoadWeights")!.Invoke(layer, [w, "l"]);
        MethodInfo forward = layerType.GetMethod("Forward")!;
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(hd, cfg.TransformerRopeTheta, 64);
        using Tensor mask = Qwen3TtsVocoder.BuildSlidingCausalMask(12, cfg.SlidingWindow);
        Pin("qwen3tts_layer", AttentionTraceOps.Record(b =>
        {
            forward.Invoke(layer, [b, T(1, 12, dim), 12, mask, cos, sin]);
            forward.Invoke(layer, [b, T(1, 1, dim), 1, null, cos, sin]);
        }));
    }

    [Fact]
    public void StyleTransformer1d_AttendsSelfOverOneTokenAndCrossOverTheContext()
    {
        const int c = 16, heads = 2, hf = 8, inner = heads * hf, mult = 2, ctxLen = 5;
        Weights w = new(76);
        Tensor Wt(string name, long rows, long cols) { w.Add(name, rows, cols); return w[name]; }
        Tensor Bs(string name, long n) { w.AddBias(name, n); return w[name]; }
        Tensor Nm(string name) { w.AddNorm(name, c); return w[name]; }
        StyleTransformerLayer layer = new()
        {
            AdaLnW = Wt("ada", 2 * c, c), AdaLnB = Bs("ada.b", 2 * c),
            SelfNormW = Nm("n1"), SelfNormB = Bs("n1.b", c),
            SelfQW = Wt("q", inner, c), SelfQB = Bs("q.b", inner), SelfKW = Wt("k", inner, c), SelfKB = Bs("k.b", inner),
            SelfVW = Wt("v", inner, c), SelfVB = Bs("v.b", inner), SelfOW = Wt("o", c, inner), SelfOB = Bs("o.b", c),
            CrossNormW = Nm("n2"), CrossNormB = Bs("n2.b", c),
            CrossQW = Wt("cq", inner, c), CrossQB = Bs("cq.b", inner), CrossKW = Wt("ck", inner, c), CrossKB = Bs("ck.b", inner),
            CrossVW = Wt("cv", inner, c), CrossVB = Bs("cv.b", inner), CrossOW = Wt("co", c, inner), CrossOB = Bs("co.b", c),
            FfNormW = Nm("n3"), FfNormB = Bs("n3.b", c),
            FfW1 = Wt("f1", c * mult, c), FfB1 = Bs("f1.b", c * mult), FfW2 = Wt("f2", c, c * mult), FfB2 = Bs("f2.b", c),
        };
        StyleTransformer1d net = new(c, heads, hf, c, mult, [layer], null, null, null, null, null, null, null, null);
        Pin("style_transformer1d", AttentionTraceOps.Record(b =>
        {
            Invoke(net, "SelfAttention", b, layer, T(1, 1, c));
            Invoke(net, "CrossAttention", b, layer, T(1, 1, c), T(1, ctxLen, c));
        }));
    }

    [Fact]
    public void ControlFoleyVitSelfAttention_AttendsABatchOfTwoAndKeepsItsOperandsUntilTheEnd()
    {
        const int dim = 16, heads = 2, n = 7;
        Tensor qkvW = T(3 * dim, dim), qkvB = T(3 * dim), projW = T(dim, dim), projB = T(dim);
        Pin("controlfoley_vit", AttentionTraceOps.Record(b => ControlFoleyVitOps.SelfAttention(b, T(2, n, dim), qkvW, qkvB, projW, projB, heads)));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ControlFoleyClipTransformer_AttendsWithTheCausalMaskOrWithout(bool causal)
    {
        const int width = 16, heads = 2, ratio = 2, n = 6;
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
        Pin(causal ? "controlfoley_clip_text" : "controlfoley_clip_vision", AttentionTraceOps.Record(b => tr.Forward(b, new float[n * width], n, causal)));
    }

    [Fact]
    public void ControlFoleyBlockAttend_AttendsDenseUnmasked()
    {
        const int heads = 2, hd = 8, n = 7;
        Pin("controlfoley_block_attend", AttentionTraceOps.Record(b => ControlFoleyBlock.Attend(b, T(2, heads, n, hd), T(2, heads, n, hd), T(2, heads, n, hd))));
    }

    [Fact]
    public void ControlFoleyVaeAttention_AttendsASingleHeadOfTheFullChannelWidth()
    {
        const int c = 32, t = 9;
        Weights w = new(153);
        w.AddTensor("a.qkv.weight", 0.3f, 3 * c, c, 1); w.AddTensor("a.proj_out.weight", 0.3f, c, c, 1);
        ControlFoleyVaeDecoder dec = new(new ControlFoleyVaeConfig { DataDim = 8, EmbedDim = 4, HiddenDim = 8 });
        object block = Invoke(dec, "Attn", w, "a", c)!;
        Pin("controlfoley_vae_attn", AttentionTraceOps.Record(b => Invoke(dec, "AttnForward", b, T(1, c, t), block), serveConvs: true));
    }

    [Fact]
    public void ControlFoleySynchformer_AttendsTheClsRowThenTheGroupsThenTheAggregate()
    {
        ControlFoleySynchformerConfig cfg = new() { EmbedDim = 16, Depth = 1, Heads = 2, PatchSize = 16, InputSize = 32 };
        ControlFoleySynchformer sync = new(cfg);
        const int d = 16;
        Type attnType = typeof(ControlFoleySynchformer).GetNestedType("Attn", BindingFlags.NonPublic)!;
        object attn = Activator.CreateInstance(attnType, T(3 * d, d), T(3 * d), T(d, d), T(d))!;
        int total = 1 + cfg.TemporalTokens * cfg.SpatialTokens;
        MethodInfo divided = typeof(ControlFoleySynchformer).GetMethod("DividedAttention", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Pin("controlfoley_synchformer_divided", AttentionTraceOps.Record(b =>
        {
            ((Tensor)divided.Invoke(sync, [b, T(1, total, d), attn, true])!).Dispose();
            ((Tensor)divided.Invoke(sync, [b, T(1, total, d), attn, false])!).Dispose();
        }));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DemucsTransformerLayer_AttendsSelfOrCrossDenseUnmasked(bool cross)
    {
        const int dim = 16, ffn = 24, nq = 7, nkv = 5;
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
        Pin(cross ? "demucs_cross" : "demucs_self", AttentionTraceOps.Record(b =>
        {
            Tensor q = T(1, nq, dim);
            Invoke(layer, "Forward", b, q, cross ? T(1, nkv, dim) : q, nq, cross ? nkv : nq);
        }));
    }

    [Fact]
    public void ControlFoleySynchformerAggregate_AttendsTheClsQueryOverEveryKey()
    {
        ControlFoleySynchformerConfig cfg = new() { EmbedDim = 16, Depth = 1, Heads = 2, PatchSize = 16, InputSize = 32 };
        ControlFoleySynchformer sync = new(cfg);
        const int d = 16;
        Type aggType = typeof(ControlFoleySynchformer).GetNestedType("Agg", BindingFlags.NonPublic)!;
        object agg = Activator.CreateInstance(aggType, T(1, 1, d), T(3 * d, d), T(3 * d), T(d, d), T(d), T(d), T(d), T(d), T(d), T(d, d), T(d), T(d, d), T(d))!;
        BindingFlags inst = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(ControlFoleySynchformer).GetField("_agg", inst)!.SetValue(sync, agg);
        typeof(ControlFoleySynchformer).GetField("_normW", inst)!.SetValue(sync, T(d));
        typeof(ControlFoleySynchformer).GetField("_normB", inst)!.SetValue(sync, T(d));
        MethodInfo aggregate = typeof(ControlFoleySynchformer).GetMethod("Aggregate", inst)!;
        Pin("controlfoley_synchformer_aggregate", AttentionTraceOps.Record(b => aggregate.Invoke(sync, [b, T(1, 1 + cfg.TemporalTokens * cfg.SpatialTokens, d)])));
    }

    [Fact]
    public void Wav2Vec2EncoderAttention_AttendsDenseUnmasked()
    {
        const int h = 16, t = 9;
        Wav2Vec2EncoderConfig cfg = new()
        {
            ConvDims = [8], ConvStrides = [5], ConvKernels = [10], ConvBias = false, GroupNormFirstConvOnly = true,
            Hidden = h, NumLayers = 1, NumHeads = 2, Intermediate = 32, PosConvKernel = 8, PosConvGroups = 2,
        };
        Wav2Vec2Encoder enc = new(cfg);
        Type layerType = typeof(Wav2Vec2Encoder).GetNestedType("Layer", BindingFlags.NonPublic)!;
        object layer = Activator.CreateInstance(layerType)!;
        foreach (string n in new[] { "QW", "KW", "VW", "OW" }) layerType.GetField(n)!.SetValue(layer, T(h, h));
        foreach (string n in new[] { "QB", "KB", "VB", "OB" }) layerType.GetField(n)!.SetValue(layer, T(h));
        MethodInfo attention = typeof(Wav2Vec2Encoder).GetMethod("Attention", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Pin("wav2vec2_attention", AttentionTraceOps.Record(b => attention.Invoke(enc, [b, T(t, h), layer, t])));
    }
}

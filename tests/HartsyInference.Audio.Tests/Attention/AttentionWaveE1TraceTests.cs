using System.Reflection;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.F5Tts;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Models.Mert2;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Models.SheetSage2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

/// <summary>The op sequence of each device-resident audio attention site on the recording stub (no compute): every backend op in order, with the
/// attention call pinned to its route, scale, F16 assertion and mask. The sequences were recorded from the code before the sites moved onto
/// the shared layer and must not change. Ops the stub leaves to the interface's host defaults (norms, rope, residuals) are not recorded.</summary>
public sealed class AttentionWaveE1TraceTests
{
    /// <summary>The ops with each attention call spelled out: the legacy SDPA record and the layer's <c>Attention</c> record read the same.</summary>
    private static string[] Ops(AttentionTraceBackend b)
    {
        Queue<AttentionCall> dense = new(b.Calls.Where(c => c.Route is not (AttentionRoute.Flash or AttentionRoute.FlashDevice)));
        int sdpa = 0;
        List<string> ops = new();
        foreach (string op in b.Ops)
        {
            if (op.StartsWith("SDPA ", StringComparison.Ordinal))
            {
                SdpaCall c = b.SdpaCalls[sdpa++];
                ops.Add($"Attn Dense scale={c.Scale:0.######} f16={c.AllowF16} mask={c.Masked}");
            }
            else if (op.StartsWith("Attention ", StringComparison.Ordinal))
            {
                AttentionCall c = dense.Dequeue();
                ops.Add($"Attn {c.Route} scale={c.Scale:0.######} f16={c.Numerics.LegacyAssertedBounded} mask={c.Mask.Bias is not null}");
            }
            else
            {
                ops.Add(op);
            }
        }
        return [.. ops];
    }

    private static string[] Run(Action<AttentionTraceBackend> site)
    {
        AttentionTraceBackend trace = new();
        try { site(trace); }
        catch (NotSupportedException) { /* ops past the attention that the stub does not serve */ }
        catch (TargetInvocationException e) when (e.InnerException is NotSupportedException) { /* the same, through reflection */ }
        return Ops(trace);
    }

    private static Tensor T(params long[] dims) => new(new TensorShape(dims), DType.F32);

    private static void Pin(string site, string[] actual) => Assert.Equal(Pins[site], actual);

    [Fact]
    public void F5Block_ProjectsRopesPermutesAttendsAndProjectsOn_Dense()
    {
        Weights w = new(81);
        w.Linear("b.attn_norm.linear", 6 * 32, 32, true);
        foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", 32, 32, true);
        w.Linear("b.attn.to_out.0", 32, 32, true);
        w.Linear("b.ff.ff.0.0", 64, 32, true); w.Linear("b.ff.ff.2", 32, 64, true);
        F5DitBlock block = new(new F5TtsConfig { Dim = 32, Heads = 2, FfMult = 2 });
        block.LoadWeights(w, "b");
        Pin("f5", Run(b => block.Forward(b, T(1, 9, 32), T(1, 1, 32), 9, T(32, 16), T(32, 16))));
    }

    [Fact]
    public void IndexTts2Block_AttendsThroughTheFusedWeightViews()
    {
        const int hidden = 32;
        Weights w = new(82);
        foreach (string n in new[] { "attention_norm", "ffn_norm" })
        {
            w.AddNorm($"b.{n}.norm.weight", hidden); w.Linear($"b.{n}.project_layer", 2 * hidden, hidden, true);
        }
        w.Add("b.attention.wqkv.weight", 3 * hidden, hidden); w.Add("b.attention.wo.weight", hidden, hidden);
        w.Add("b.feed_forward.w1.weight", 48, hidden); w.Add("b.feed_forward.w3.weight", 48, hidden); w.Add("b.feed_forward.w2.weight", hidden, 48);
        IndexTts2DitBlock block = new(hidden, 2, 48);
        block.LoadWeights(w, "b");
        Pin("indextts2", Run(b => block.Forward(b, T(1, 9, hidden), T(1, hidden), 9, T(32, 16), T(32, 16), null)));
    }

    [Fact]
    public void AlbertLayer_ProjectsPermutesAndAttendsOnTheSharedWeights()
    {
        const int hidden = 24;
        KokoroConfig.PlBertConfig cfg = new() { HiddenSize = hidden, NumAttentionHeads = 2, IntermediateSize = 32, MaxPositionEmbeddings = 16, NumHiddenLayers = 2, EmbeddingSize = 8 };
        Weights w = new(83);
        const string p = "l";
        foreach (string n in new[] { "query", "key", "value", "dense" }) w.Linear($"{p}.attention.{n}", hidden, hidden, true);
        w.AddNorm($"{p}.attention.LayerNorm.weight", hidden); w.AddBias($"{p}.attention.LayerNorm.bias", hidden);
        w.Linear($"{p}.ffn", 32, hidden, true); w.Linear($"{p}.ffn_output", hidden, 32, true);
        w.AddNorm($"{p}.full_layer_layer_norm.weight", hidden); w.AddBias($"{p}.full_layer_layer_norm.bias", hidden);
        AlbertLayer layer = new(cfg);
        layer.LoadWeights(w, p);
        Pin("albert", Run(b => layer.Forward(b, T(1, 7, hidden), 7)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MatchaBlock_AttendsOverAnInnerWidthWiderThanTheModelWidth(bool masked)
    {
        const int ch = 16, heads = 4, hd = 8, t = 10;
        Weights w = new(84);
        w.AddNorm("m.norm1.weight", ch); w.AddBias("m.norm1.bias", ch);
        w.Add("m.attn1.to_q.weight", heads * hd, ch); w.Add("m.attn1.to_k.weight", heads * hd, ch); w.Add("m.attn1.to_v.weight", heads * hd, ch);
        w.Linear("m.attn1.to_out.0", ch, heads * hd, true);
        w.AddNorm("m.norm3.weight", ch); w.AddBias("m.norm3.bias", ch);
        w.Linear("m.ff.net.0.proj", 40, ch, true); w.Linear("m.ff.net.2", ch, 40, true);
        MatchaTransformerBlock block = new(ch, heads, hd);
        block.LoadWeights(w, "m");
        Pin(masked ? "matcha_masked" : "matcha", Run(b => block.Forward(b, T(1, t, ch), t, masked ? T(t, t) : null)));
    }

    [Fact]
    public void WhisperEncoderLayer_AttendsABatchOfTwo()
    {
        WhisperConfig cfg = Whisper();
        Weights w = new(11);
        WhisperLayer(w, "l", decoder: false);
        WhisperEncoderLayer layer = new(cfg);
        layer.LoadWeights(w, "l");
        Pin("whisper_enc", Run(b => layer.Forward(b, T(2, 6, 16))));
    }

    [Theory]
    [InlineData(0, 4)]
    [InlineData(4, 1)]
    public void WhisperDecoderLayer_AttendsSelfWithTheMaskTensorThenCrossWithoutOne(int posStart, int newCount)
    {
        WhisperConfig cfg = Whisper();
        Weights w = new(13);
        WhisperLayer(w, "l", decoder: true);
        WhisperDecoderLayer layer = new(cfg);
        layer.LoadWeights(w, "l");
        Tensor mask = WhisperDecoderLayer.BuildIncrementalCausalMask(posStart, newCount, posStart + newCount);
        string[] ops = Run(b => layer.Forward(b, T(1, newCount, 16), T(1, 2, 16, 8), T(1, 2, 16, 8), T(1, 2, 9, 8), T(1, 2, 9, 8), posStart, newCount, mask));
        Pin($"whisper_dec_{posStart}_{newCount}", ops);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void GptBlock_Uncached_AttendsWithTheCausalMaskOrNone(bool bias, bool causal)
    {
        GptConfig cfg = new() { Hidden = 32, NumLayers = 1, NumHeads = 2, Bias = bias };
        using GptBlock block = new(cfg);
        block.LoadWeights(GptWeights(cfg), "g");
        Pin($"gpt_uncached_{bias}", Run(b => block.Forward(b, T(1, 7, 32), causal ? T(1, 1, 7, 7) : null)));
    }

    [Fact]
    public void GptBlock_Cached_AppendsThenFlashesCausalOverTheValidPrefix()
    {
        GptConfig cfg = new() { Hidden = 32, NumLayers = 1, NumHeads = 2, Bias = true };
        using GptBlock block = new(cfg);
        block.LoadWeights(GptWeights(cfg), "g");
        using FixedKvCache cache = new(1, 1, 2, 16, 16);
        List<string> ops = new();
        ops.AddRange(Run(b => block.ForwardCached(b, T(1, 3, 32), cache, 0, 0)));
        cache.AdvanceLength(3);
        ops.AddRange(Run(b => block.ForwardCached(b, T(1, 1, 32), cache, 0, 3)));
        Pin("gpt_cached", [.. ops]);
    }

    [Fact]
    public void MusicGenStep_WithoutDevicePositions_KeepsTheDevEntryAtAZeroHandle()
    {
        const int h = 16;
        MusicGenConfig cfg = new() { Hidden = h, NumLayers = 1, NumHeads = 2 };
        Weights w = new(2);
        foreach (string n in new[] { "self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm" }) { w.AddNorm($"m.{n}.weight", h); w.AddBias($"m.{n}.bias", h); }
        foreach (string a in new[] { "self_attn", "encoder_attn" })
            foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Add($"m.{a}.{n}.weight", h, h);
        w.Add("m.fc1.weight", 4 * h, h); w.Add("m.fc2.weight", h, 4 * h);
        using MusicGenBlock block = new(cfg);
        block.LoadWeights(w, "m");
        using MusicGenKvCache cache = new(1, 1, 2, 8, 8);
        AttentionTraceBackend prime = new();
        block.PrimeCross(prime, T(1, 3, h), cache.Layers[0]);
        Pin("musicgen_host", Run(b => block.ForwardStep(b, T(1, 1, h), cache.Layers[0], 2, 3)));
        Pin("musicgen_dev", Run(b => block.ForwardStep(b, T(1, 1, h), cache.Layers[0], 2, 3, devicePos: 7)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DiaFlashPath_RotatesQThenKPerTensorBeforeThePermutes(bool interleaved)
    {
        Weights w = new(1);
        w.Add("a.q_proj.weight", 32, 16); w.Add("a.k_proj.weight", 16, 16); w.Add("a.v_proj.weight", 16, 16); w.Add("a.o_proj.weight", 16, 32);
        DiaAttention attn = new(4, 2, 8, 16, 16, 16, useRope: true, interleavedRope: interleaved);
        attn.LoadWeights(w, "a");
        using FixedKvCache cache = new(1, 1, 2, 8, 16);
        Pin($"dia_flash_{interleaved}", Run(b => attn.SelfForwardFlash(b, T(1, 2, 16), 2, 0, cache, 0, T(1, 2, 8), T(1, 2, 8))));
    }

    [Fact]
    public void AukOps_Attend_PermutesAttendsAndMergesThroughTheDenseRoute()
    {
        Pin("auk_attend", Run(b => AukOps.Attend(b, T(1, 7, 2, 8), T(1, 7, 2, 8), T(1, 7, 2, 8), 7, 2, 8)));
    }

    [Fact]
    public void Mert2Layer_AttendsIntoItsScratchWithTheF16Assertion()
    {
        const int dim = 16, tokens = 9;
        Mert2Config cfg = new() { Dim = dim, Intermediate = 24, Heads = 2, ConformerConvKernel = 5 };
        Weights w = new(91);
        foreach (string f in new[] { "ffn1", "ffn2" })
        {
            w.AddNorm($"l.{f}_layer_norm.weight", dim); w.AddBias($"l.{f}_layer_norm.bias", dim);
            w.Linear($"l.{f}.w_1", 24, dim, true); w.Linear($"l.{f}.w_2", dim, 24, true);
        }
        w.AddNorm("l.attn_layer_norm.weight", dim); w.AddBias("l.attn_layer_norm.bias", dim);
        foreach (string n in new[] { "query_proj", "key_proj", "value_proj", "out_proj" }) w.Linear($"l.attn.{n}", dim, dim, true);
        w.AddNorm("l.conv_module.layer_norm.weight", dim); w.AddBias("l.conv_module.layer_norm.bias", dim);
        w.AddTensor("l.conv_module.conv_block.1.weight", 0.3f, 2 * dim, dim, 1);
        w.AddTensor("l.conv_module.conv_block.3.weight", 0.3f, dim, 1, 5);
        w.AddNorm("l.conv_module.conv_block.4.1.weight", dim); w.AddBias("l.conv_module.conv_block.4.1.bias", dim);
        w.AddTensor("l.conv_module.conv_block.6.weight", 0.3f, dim, dim, 1);
        w.AddNorm("l.final_layer_norm.weight", dim); w.AddBias("l.final_layer_norm.bias", dim);
        Mert2ConformerLayer layer = new(cfg);
        layer.LoadWeights(w, "l", []);
        using Mert2ConformerLayer.Scratch scratch = new(cfg, tokens);
        Pin("mert2", Run(b => layer.Forward(b, T(tokens, dim), T(tokens, dim), scratch)));
    }


    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void HeartCodecBlock_AttendsWithRopeTablesOverBatchTimesTokens(int b)
    {
        const int dim = 16, heads = 2, hd = 8, t = 6;
        Weights w = new(90);
        w.AddNorm("p.attn_norm.weight", dim); w.AddNorm("p.mlp_norm.weight", dim);
        w.AddTensor("p.scale_shift_table", 0.3f, 6, dim);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "o_proj" }) w.Add($"p.attn.{n}.weight", dim, dim);
        w.Add("p.mlp.gate.weight", 2 * dim, dim); w.Add("p.mlp.up.weight", 2 * dim, dim); w.Add("p.mlp.down.weight", dim, 2 * dim);
        Type type = typeof(HeartCodecEstimator).GetNestedType("Block", BindingFlags.NonPublic)!;
        object block = Activator.CreateInstance(type, dim, heads, hd)!;
        type.GetMethod("Load")!.Invoke(block, [w, "p"]);
        MethodInfo forward = type.GetMethod("Forward")!;
        Pin($"heartcodec_{b}", Run(backend => forward.Invoke(block, [backend, T(b, t, dim), T(b, 6 * dim), b, t, T(b * t, hd), T(b * t, hd)])));
    }

    [Theory]
    [InlineData(1, new[] { 3, 1 })]
    [InlineData(2, new[] { 2 })]
    public void MiniMaxDepthLayer_AttendsOverTheWholeCacheBufferWithItsMask(int batch, int[] steps)
    {
        const int h = 4096, heads = 16, hd = h / heads;
        Weights w = new(110);
        w.AddNorm("l.input_layernorm.weight", h); w.AddNorm("l.post_attention_layernorm.weight", h);
        // Placeholder projections: the stub reads only a weight's first dim, which names the op.
        int row = 0;
        foreach (string n in new[] { "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "gate_proj", "up_proj", "down_proj" })
            w.Add($"l.{n}.weight", 100 + row++, 1);
        using MiniMaxMusic3DepthDecoder owner = new();
        Type type = typeof(MiniMaxMusic3DepthDecoder).GetNestedType("Layer", BindingFlags.NonPublic)!;
        object layer = Activator.CreateInstance(type, nonPublic: true)!;
        type.GetMethod("Load")!.Invoke(layer, [owner, w, "l"]);
        MethodInfo forward = type.GetMethod("Forward")!;
        using MiniMaxMusic3DepthCache cache = new(1, batch * heads, hd, 8);
        List<string> ops = new();
        foreach (int n in steps)
        {
            ops.AddRange(Run(backend => forward.Invoke(layer, [backend, T(batch, n, h), batch, n, cache, 0])));
            cache.Advance(n);
        }
        Pin($"minimax_depth_{batch}", [.. ops]);
    }

    [Fact]
    public void SheetSage2Decoder_AppendsAndFlashesOnDecodeAndAttendsDensePrefillAndCross()
    {
        const int dim = 16, inner = 24, memTokens = 6;
        SheetSage2Config cfg = new() { Dim = dim, IntermediateSize = inner, NumHeads = 2, Layers = 1, MaxTokens = 16 };
        ScoreTokenizer tokenizer = new();
        Weights w = new(92);
        w.Add("token_embedding.weight", tokenizer.TokenCount, dim);
        w.AddTensor("decoder.embed_positions.weight", 0.3f, cfg.PositionRows, dim);
        w.AddNorm("decoder.layernorm_embedding.weight", dim); w.AddBias("decoder.layernorm_embedding.bias", dim);
        w.Add("output_projection.weight", tokenizer.TokenCount, dim);
        const string p = "decoder.layers.0";
        foreach (string a in new[] { "self_attn", "encoder_attn" })
            foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"{p}.{a}.{n}", dim, dim, true);
        foreach (string n in new[] { "self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm" }) { w.AddNorm($"{p}.{n}.weight", dim); w.AddBias($"{p}.{n}.bias", dim); }
        w.Linear($"{p}.fc1", inner, dim, true); w.Linear($"{p}.fc2", dim, inner, true);
        using SheetSage2Decoder dec = new(cfg, tokenizer);
        dec.LoadWeights(w);
        AttentionTraceBackend trace = new();
        using SheetSage2DecodeState state = dec.StartDecode(trace, T(1, memTokens, dim), 12);
        foreach (int[] feed in (int[][])[[1, 5, 9, 3], [7]])
            try { dec.Decode(trace, feed, state); } catch (NotSupportedException) { /* ops past the attention that the stub does not serve */ }
        Pin("sheetsage2", Ops(trace));
    }

    [Fact]
    public void Yue2Velocity_AttendsOverThePrefixBuffersAsMhaWithTheF16Assertion()
    {
        const int hidden = 32, layers = 2, heads = 4, kvHeads = 2, hd = 8, inter = 48, latent = 8, tsWidth = 16, frames = 4, arLength = 5;
        Qwen3Config nar = new()
        {
            HiddenSize = hidden, NumHiddenLayers = layers, NumAttentionHeads = heads, NumKeyValueHeads = kvHeads, HeadDim = hd,
            IntermediateSize = inter, MaxPositionEmbeddings = 64, RopeTheta = 10_000f,
        };
        Weights w = new(120);
        w.Linear("vae2llm", hidden, latent, true); w.Linear("llm2vae", latent, hidden, true);
        w.Linear("time_embedder.mlp.0", hidden, tsWidth, true); w.Linear("time_embedder.mlp.2", hidden, hidden, true);
        w.AddTensor("latent_pos_embed.pe", 0.3f, 32, hidden);
        w.AddNorm("model.norm.weight", hidden);
        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            w.Add($"{p}.self_attn.q_proj.weight", heads * hd, hidden); w.Add($"{p}.self_attn.k_proj.weight", kvHeads * hd, hidden);
            w.Add($"{p}.self_attn.v_proj.weight", kvHeads * hd, hidden); w.Add($"{p}.self_attn.o_proj.weight", hidden, heads * hd);
            w.AddNorm($"{p}.self_attn.q_norm.weight", hd); w.AddNorm($"{p}.self_attn.k_norm.weight", hd);
            w.Add($"{p}.mlp.gate_proj.weight", inter, hidden); w.Add($"{p}.mlp.up_proj.weight", inter, hidden); w.Add($"{p}.mlp.down_proj.weight", hidden, inter);
            w.AddNorm($"{p}.input_layernorm.weight", hidden); w.AddNorm($"{p}.post_attention_layernorm.weight", hidden);
        }
        using Yue2AcousticTransformer tr = new(new Yue2Config { Ar = nar, Nar = nar, LatentDim = latent, TimestepEmbedWidth = tsWidth });
        tr.LoadWeights(w);
        (Tensor Key, Tensor Value)[] prefix = [(T(1, kvHeads, arLength, hd), T(1, kvHeads, arLength, hd)), (T(1, kvHeads, arLength, hd), T(1, kvHeads, arLength, hd))];
        Pin("yue2", Run(b => tr.Velocity(b, new float[frames * latent], 0.3f, prefix, arLength, new float[frames * latent])));
    }

    /// <summary>One backbone layer at the fixed 2048 width. Only the cross projection is read on the host at load, so every other weight is a
    /// placeholder whose first dim names its op in the trace.</summary>
    private static MoshiTransformer Moshi()
    {
        const int dim = MoshiTransformer.Dim;
        Dictionary<string, Tensor> w = new()
        {
            ["t.layers.0.self_attn.in_proj_weight"] = T(301, 1), ["t.layers.0.self_attn.out_proj.weight"] = T(302, 1),
            ["t.layers.0.cross_attention.in_proj_weight"] = T(3 * dim, dim), ["t.layers.0.cross_attention.out_proj.weight"] = T(303, 1),
            ["t.layers.0.norm1.alpha"] = T(dim), ["t.layers.0.norm2.alpha"] = T(dim),
            ["t.layers.0.norm_cross.weight"] = T(dim), ["t.layers.0.norm_cross.bias"] = T(dim),
            ["t.layers.0.gating.linear_in.weight"] = T(304, 1), ["t.layers.0.gating.linear_out.weight"] = T(305, 1),
            ["out_norm.alpha"] = T(dim),
        };
        MoshiTransformer tr = new(layers: 1);
        tr.LoadWeights(w, "t");
        return tr;
    }

    [Fact]
    public void MoshiBackbone_AttendsDenseWithTheCausalMaskThenCrossWithoutOne()
    {
        using MoshiTransformer tr = Moshi();
        Pin("moshi_forward", Run(b => tr.Forward(b, T(1, 3, MoshiTransformer.Dim), T(1, 4, MoshiTransformer.Dim))));
    }

    [Fact]
    public void MoshiBackbone_StepAppendsThenFlashesNonCausalOverThePrefixAndAttendsCrossDense()
    {
        using MoshiTransformer tr = Moshi();
        AttentionTraceBackend prime = new();
        using MoshiTransformer.CrossKvCache crossKv = tr.PrecomputeCrossKv(prime, T(1, 4, MoshiTransformer.Dim));
        using FixedKvCache cache = new(1, 1, MoshiTransformer.Heads, MoshiTransformer.HeadDim, 8);
        Pin("moshi_step", Run(b => tr.StepForward(b, T(1, 1, MoshiTransformer.Dim), crossKv, cache, 2)));
    }

    private static WhisperConfig Whisper() => new()
    {
        VocabSize = 24, NumMelBins = 8, MaxAudioPositions = 16, MaxTextPositions = 16, EncoderLayers = 2, DecoderLayers = 2,
        HiddenSize = 16, NumHeads = 2, IntermediateSize = 32,
    };

    private static void WhisperLayer(Weights w, string p, bool decoder)
    {
        void Block(string ln, string attn)
        {
            w.AddNorm($"{p}.{ln}.weight", 16); w.AddBias($"{p}.{ln}.bias", 16);
            w.Linear($"{p}.{attn}.q_proj", 16, 16, true);
            w.Add($"{p}.{attn}.k_proj.weight", 16, 16);
            w.Linear($"{p}.{attn}.v_proj", 16, 16, true);
            w.Linear($"{p}.{attn}.out_proj", 16, 16, true);
        }
        Block("self_attn_layer_norm", "self_attn");
        if (decoder) Block("encoder_attn_layer_norm", "encoder_attn");
        w.Linear($"{p}.fc1", 32, 16, true);
        w.Linear($"{p}.fc2", 16, 32, true);
        w.AddNorm($"{p}.final_layer_norm.weight", 16); w.AddBias($"{p}.final_layer_norm.bias", 16);
    }

    private static Weights GptWeights(GptConfig cfg)
    {
        int h = cfg.Hidden;
        Weights w = new(923);
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

    private static readonly Dictionary<string, string[]> Pins = new()
    {
        ["albert"] =
        [
            "Linear w24", "Linear w24", "Linear w24", "Permute0213 7,2,12", "Permute0213 7,2,12", "Permute0213 7,2,12",
            "Attn Dense scale=0.288675 f16=False mask=False", "Permute0213 2,7,12", "Linear w24", "Add", "LayerNorm w24", "Linear w32", "Gelu",
            "Linear w24", "Add", "LayerNorm w24",
        ],
        ["auk_attend"] =
        [
            "Permute0213 7,2,8", "Permute0213 7,2,8", "Permute0213 7,2,8", "Attn Dense scale=0.353553 f16=False mask=False", "Permute0213 2,7,8",
        ],
        ["dia_flash_False"] =
        [
            "Linear w32", "Linear w16", "Linear w16", "ApplyRopeSingle", "ApplyRopeSingle", "Permute0213 2,4,8", "Permute0213 2,2,8",
            "Permute0213 2,2,8", "KvCacheAppend @0 [1, 2, 2, 8]", "KvCacheAppend @0 [1, 2, 2, 8]",
            "FlashAttention kv=2 group=2 causal=True qOffset=0 window=0 sinks=False", "Permute0213 4,2,8", "Linear w16",
        ],
        ["dia_flash_True"] =
        [
            "Linear w32", "Linear w16", "Linear w16", "ApplyRopeInterleaved", "ApplyRopeInterleaved", "Permute0213 2,4,8", "Permute0213 2,2,8",
            "Permute0213 2,2,8", "KvCacheAppend @0 [1, 2, 2, 8]", "KvCacheAppend @0 [1, 2, 2, 8]",
            "FlashAttention kv=2 group=2 causal=True qOffset=0 window=0 sinks=False", "Permute0213 4,2,8", "Linear w16",
        ],
        ["f5"] =
        [
            "Linear w192", "SliceLastDim @0", "SliceLastDim @32", "SliceLastDim @64", "SliceLastDim @96", "SliceLastDim @128", "SliceLastDim @160",
            "Linear w32", "Linear w32", "Linear w32", "WanRopeInterleaved heads=2 d=16 seq=9", "WanRopeInterleaved heads=2 d=16 seq=9",
            "Permute0213 9,2,16", "Permute0213 9,2,16", "Permute0213 9,2,16", "Attn Dense scale=0.25 f16=False mask=False", "Permute0213 2,9,16",
            "Linear w32", "Linear w64", "Gelu", "Linear w32",
        ],
        ["gpt_cached"] =
        [
            "LayerNorm w32", "Linear w96", "SliceLastDim @0", "SliceLastDim @32", "SliceLastDim @64", "Permute0213 3,2,16", "Permute0213 3,2,16",
            "Permute0213 3,2,16", "KvCacheAppend @0 [1, 2, 3, 16]", "KvCacheAppend @0 [1, 2, 3, 16]",
            "FlashAttention kv=3 group=1 causal=True qOffset=0 window=0 sinks=False", "Permute0213 2,3,16", "Linear w32", "Add", "LayerNorm w32",
            "Linear w128", "Gelu", "Linear w32", "Add", "LayerNorm w32", "Linear w96", "SliceLastDim @0", "SliceLastDim @32", "SliceLastDim @64",
            "Permute0213 1,2,16", "Permute0213 1,2,16", "Permute0213 1,2,16", "KvCacheAppend @3 [1, 2, 1, 16]", "KvCacheAppend @3 [1, 2, 1, 16]",
            "FlashAttention kv=4 group=1 causal=True qOffset=3 window=0 sinks=False", "Permute0213 2,1,16", "Linear w32", "Add", "LayerNorm w32",
            "Linear w128", "Gelu", "Linear w32", "Add",
        ],
        ["gpt_uncached_False"] =
        [
            "LayerNorm w32", "Linear w96", "SliceLastDim @0", "SliceLastDim @32", "SliceLastDim @64", "Permute0213 7,2,16", "Permute0213 7,2,16",
            "Permute0213 7,2,16", "Attn Dense scale=0.25 f16=False mask=False", "Permute0213 2,7,16", "Linear w32", "Add", "LayerNorm w32",
            "Linear w128", "Gelu", "Linear w32", "Add",
        ],
        ["gpt_uncached_True"] =
        [
            "LayerNorm w32", "Linear w96", "SliceLastDim @0", "SliceLastDim @32", "SliceLastDim @64", "Permute0213 7,2,16", "Permute0213 7,2,16",
            "Permute0213 7,2,16", "Attn Dense scale=0.25 f16=False mask=True", "Permute0213 2,7,16", "Linear w32", "Add", "LayerNorm w32",
            "Linear w128", "Gelu", "Linear w32", "Add",
        ],
        ["heartcodec_1"] =
        [
            "Add", "SliceLastDim @0", "SliceLastDim @16", "SliceLastDim @32", "SliceLastDim @48", "SliceLastDim @64", "SliceLastDim @80",
            "RmsNorm w16", "Linear w16", "Linear w16", "Linear w16", "WanRopeInterleaved heads=2 d=8 seq=6", "WanRopeInterleaved heads=2 d=8 seq=6",
            "Permute0213 6,2,8", "Permute0213 6,2,8", "Permute0213 6,2,8", "Attn Dense scale=0.353553 f16=True mask=False", "Permute0213 2,6,8",
            "Linear w16", "RmsNorm w16", "Linear w32", "Silu", "Linear w32", "Mul", "Linear w16",
        ],
        ["heartcodec_2"] =
        [
            "Add", "SliceLastDim @0", "SliceLastDim @16", "SliceLastDim @32", "SliceLastDim @48", "SliceLastDim @64", "SliceLastDim @80",
            "RmsNorm w16", "Linear w16", "Linear w16", "Linear w16", "WanRopeInterleaved heads=2 d=8 seq=12",
            "WanRopeInterleaved heads=2 d=8 seq=12", "Permute0213 6,2,8", "Permute0213 6,2,8", "Permute0213 6,2,8",
            "Attn Dense scale=0.353553 f16=True mask=False", "Permute0213 2,6,8", "Linear w16", "RmsNorm w16", "Linear w32", "Silu", "Linear w32",
            "Mul", "Linear w16",
        ],
        ["indextts2"] =
        [
            "RmsNorm w32", "Linear w32", "Linear w32", "Linear w32", "Linear w32", "Linear w32", "WanRopeInterleaved heads=2 d=16 seq=9",
            "WanRopeInterleaved heads=2 d=16 seq=9", "Permute0213 9,2,16", "Permute0213 9,2,16", "Permute0213 9,2,16",
            "Attn Dense scale=0.25 f16=False mask=False", "Permute0213 2,9,16", "Linear w32", "Add", "RmsNorm w32", "Linear w32", "Linear w32",
            "Linear w48", "Silu", "Linear w48", "Mul", "Linear w32", "Add",
        ],
        ["matcha"] =
        [
            "LayerNorm w16", "Linear w32", "Linear w32", "Linear w32", "Permute0213 10,4,8", "Permute0213 10,4,8", "Permute0213 10,4,8",
            "Attn Dense scale=0.353553 f16=False mask=False", "Permute0213 4,10,8", "Linear w16", "Add", "LayerNorm w16", "Linear w40", "Linear w16",
            "Add",
        ],
        ["matcha_masked"] =
        [
            "LayerNorm w16", "Linear w32", "Linear w32", "Linear w32", "Permute0213 10,4,8", "Permute0213 10,4,8", "Permute0213 10,4,8",
            "Attn Dense scale=0.353553 f16=False mask=True", "Permute0213 4,10,8", "Linear w16", "Add", "LayerNorm w16", "Linear w40", "Linear w16",
            "Add",
        ],
        ["mert2"] =
        [
            "LayerNorm w16", "Linear w24", "Linear w16", "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16", "ApplyRope", "Permute0213 9,2,8",
            "Permute0213 9,2,8", "Permute0213 9,2,8", "Attn Dense scale=0.353553 f16=True mask=False", "Permute0213 2,9,8", "Linear w16", "Add",
            "LayerNorm w16", "Transpose2D 9,16",
        ],
        ["minimax_depth_1"] =
        [
            "RmsNorm w4096", "Linear w100", "Linear w101", "Linear w102", "Permute0213 3,16,256", "Permute0213 3,16,256", "Permute0213 3,16,256",
            "KvCacheAppend @0 [1, 16, 3, 256]", "KvCacheAppend @0 [1, 16, 3, 256]", "Attn Dense scale=0.0625 f16=False mask=True",
            "Permute0213 16,3,256", "Linear w103", "Add", "RmsNorm w4096", "Linear w104", "Linear w105", "Silu", "Mul", "Linear w106", "Add",
            "RmsNorm w4096", "Linear w100", "Linear w101", "Linear w102", "KvCacheAppend @3 [1, 16, 1, 256]", "KvCacheAppend @3 [1, 16, 1, 256]",
            "Attn Dense scale=0.0625 f16=False mask=True", "Linear w103", "Add", "RmsNorm w4096", "Linear w104", "Linear w105", "Silu", "Mul",
            "Linear w106", "Add",
        ],
        ["minimax_depth_2"] =
        [
            "RmsNorm w4096", "Linear w100", "Linear w101", "Linear w102", "Permute0213 2,16,256", "Permute0213 2,16,256", "Permute0213 2,16,256",
            "KvCacheAppend @0 [1, 32, 2, 256]", "KvCacheAppend @0 [1, 32, 2, 256]", "Attn Dense scale=0.0625 f16=False mask=True",
            "Permute0213 16,2,256", "Linear w103", "Add", "RmsNorm w4096", "Linear w104", "Linear w105", "Silu", "Mul", "Linear w106", "Add",
        ],
        ["moshi_forward"] =
        [
            "Linear w4096", "Permute0213 4,16,128", "Permute0213 4,16,128", "RmsNorm w2048", "Linear w301", "ApplyRopeInterleaved",
            "ApplyRopeInterleaved", "Permute0213 3,16,128", "Permute0213 3,16,128", "Permute0213 3,16,128",
            "Attn Dense scale=0.088388 f16=False mask=True", "Linear w302", "Add", "LayerNorm w2048", "Linear w2048", "Permute0213 3,16,128",
            "Attn Dense scale=0.088388 f16=False mask=False", "Linear w303", "Add", "RmsNorm w2048", "Linear w304", "Linear w305", "Add",
            "RmsNorm w2048",
        ],
        ["moshi_step"] =
        [
            "RmsNorm w2048", "Linear w301", "ApplyRopeInterleaved", "ApplyRopeInterleaved", "Permute0213 1,16,128", "Permute0213 1,16,128",
            "Permute0213 1,16,128", "KvCacheAppend @0 [1, 16, 1, 128]", "KvCacheAppend @0 [1, 16, 1, 128]",
            "FlashAttention kv=3 group=1 causal=False qOffset=0 window=0 sinks=False", "Linear w302", "Add", "LayerNorm w2048", "Linear w2048",
            "Permute0213 1,16,128", "Attn Dense scale=0.088388 f16=False mask=False", "Linear w303", "Add", "RmsNorm w2048", "Linear w304",
            "Linear w305", "Add", "RmsNorm w2048",
        ],
        ["musicgen_dev"] =
        [
            "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16", "Permute0213 1,2,8", "Permute0213 1,2,8", "Permute0213 1,2,8",
            "KvCacheAppendDev @2 dev=True", "KvCacheAppendDev @2 dev=True", "FlashAttentionDev kv=3 group=1 causal=True qOffset=2 window=0 dev=True",
            "Permute0213 2,1,8", "Linear w16", "Add", "LayerNorm w16", "Linear w16", "Permute0213 1,2,8",
            "FlashAttention kv=3 group=1 causal=False qOffset=0 window=0 sinks=False", "Permute0213 2,1,8", "Linear w16", "Add", "LayerNorm w16",
            "Linear w64", "Gelu", "Linear w16", "Add",
        ],
        ["musicgen_host"] =
        [
            "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16", "Permute0213 1,2,8", "Permute0213 1,2,8", "Permute0213 1,2,8",
            "KvCacheAppendDev @2 dev=False", "KvCacheAppendDev @2 dev=False",
            "FlashAttentionDev kv=3 group=1 causal=True qOffset=2 window=0 dev=False", "Permute0213 2,1,8", "Linear w16", "Add", "LayerNorm w16",
            "Linear w16", "Permute0213 1,2,8", "FlashAttention kv=3 group=1 causal=False qOffset=0 window=0 sinks=False", "Permute0213 2,1,8",
            "Linear w16", "Add", "LayerNorm w16", "Linear w64", "Gelu", "Linear w16", "Add",
        ],
        ["sheetsage2"] =
        [
            "Linear w16", "Linear w16", "Permute0213 6,2,8", "Permute0213 6,2,8", "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16",
            "Permute0213 4,2,8", "Permute0213 4,2,8", "Permute0213 4,2,8", "KvCacheAppend @0 [1, 2, 4, 8]", "KvCacheAppend @0 [1, 2, 4, 8]",
            "Attn Dense scale=0.353553 f16=False mask=True", "Permute0213 2,4,8", "Linear w16", "Add", "LayerNorm w16", "Linear w16",
            "Permute0213 4,2,8", "Attn Dense scale=0.353553 f16=True mask=False", "Permute0213 2,4,8", "Linear w16", "Add", "LayerNorm w16",
            "Linear w24", "Linear w16", "Add", "LayerNorm w16", "Linear w31678", "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16",
            "Permute0213 1,2,8", "Permute0213 1,2,8", "Permute0213 1,2,8", "KvCacheAppend @4 [1, 2, 1, 8]", "KvCacheAppend @4 [1, 2, 1, 8]",
            "FlashAttention kv=5 group=1 causal=True qOffset=4 window=0 sinks=False", "Permute0213 2,1,8", "Linear w16", "Add", "LayerNorm w16",
            "Linear w16", "Permute0213 1,2,8", "Attn Dense scale=0.353553 f16=True mask=False", "Permute0213 2,1,8", "Linear w16", "Add",
            "LayerNorm w16", "Linear w24", "Linear w16", "Add", "LayerNorm w16", "Linear w31678",
        ],
        ["whisper_dec_0_4"] =
        [
            "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16", "Permute0213 4,2,8", "Permute0213 4,2,8", "Permute0213 4,2,8",
            "KvCacheAppend @0 [1, 2, 4, 8]", "KvCacheAppend @0 [1, 2, 4, 8]", "SliceTimeRange 0+4", "SliceTimeRange 0+4",
            "Attn Dense scale=0.353553 f16=False mask=True", "Permute0213 2,4,8", "Linear w16", "Add", "LayerNorm w16", "Linear w16",
            "Permute0213 4,2,8", "Attn Dense scale=0.353553 f16=False mask=False", "Permute0213 2,4,8", "Linear w16", "Add", "LayerNorm w16",
            "Linear w32", "Linear w16", "Add",
        ],
        ["whisper_dec_4_1"] =
        [
            "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16", "Permute0213 1,2,8", "Permute0213 1,2,8", "Permute0213 1,2,8",
            "KvCacheAppend @4 [1, 2, 1, 8]", "KvCacheAppend @4 [1, 2, 1, 8]", "SliceTimeRange 0+5", "SliceTimeRange 0+5",
            "Attn Dense scale=0.353553 f16=False mask=True", "Permute0213 2,1,8", "Linear w16", "Add", "LayerNorm w16", "Linear w16",
            "Permute0213 1,2,8", "Attn Dense scale=0.353553 f16=False mask=False", "Permute0213 2,1,8", "Linear w16", "Add", "LayerNorm w16",
            "Linear w32", "Linear w16", "Add",
        ],
        ["whisper_enc"] =
        [
            "LayerNorm w16", "Linear w16", "Linear w16", "Linear w16", "Permute0213 6,2,8", "Permute0213 6,2,8", "Permute0213 6,2,8",
            "Attn Dense scale=0.353553 f16=False mask=False", "Permute0213 2,6,8", "Linear w16", "Add", "LayerNorm w16", "Linear w32", "Linear w16",
            "Add",
        ],
        ["yue2"] =
        [
            "RepeatKvHeads 2x2", "RepeatKvHeads 2x2", "KvCacheAppend @0 [1, 4, 5, 8]", "KvCacheAppend @0 [1, 4, 5, 8]", "RepeatKvHeads 2x2",
            "RepeatKvHeads 2x2", "KvCacheAppend @0 [1, 4, 5, 8]", "KvCacheAppend @0 [1, 4, 5, 8]", "Linear w32", "Linear w32", "Silu", "Linear w32",
            "GatherRows 6", "Add", "RmsNorm w32", "Linear w32", "Linear w16", "Linear w16", "RmsNorm w8", "RmsNorm w8", "ApplyRopeSingle",
            "ApplyRopeSingle", "Permute0213 6,4,8", "Permute0213 6,2,8", "Permute0213 6,2,8", "RepeatKvHeads 2x2", "RepeatKvHeads 2x2",
            "KvCacheAppend @5 [1, 4, 6, 8]", "KvCacheAppend @5 [1, 4, 6, 8]", "Attn Dense scale=0.353553 f16=True mask=False", "Permute0213 4,6,8",
            "Linear w32", "Add", "RmsNorm w32", "Linear w48", "Linear w48", "Silu", "Mul", "Linear w32", "Add", "RmsNorm w32", "Linear w32",
            "Linear w16", "Linear w16", "RmsNorm w8", "RmsNorm w8", "ApplyRopeSingle", "ApplyRopeSingle", "Permute0213 6,4,8", "Permute0213 6,2,8",
            "Permute0213 6,2,8", "RepeatKvHeads 2x2", "RepeatKvHeads 2x2", "KvCacheAppend @5 [1, 4, 6, 8]", "KvCacheAppend @5 [1, 4, 6, 8]",
            "Attn Dense scale=0.353553 f16=True mask=False", "Permute0213 4,6,8", "Linear w32", "Add", "RmsNorm w32", "Linear w48", "Linear w48",
            "Silu", "Mul", "Linear w32", "Add", "RmsNorm w32", "Linear w8",
        ],
    };
}

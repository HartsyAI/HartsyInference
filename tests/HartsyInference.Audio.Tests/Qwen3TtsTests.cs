using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Audio.Streaming;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weight forwards for Qwen3-TTS on TINY configs: the talker emits a codebook-0 token, the
/// MTP predicts the acoustic codebooks, the custom vocoder turns a codebook grid into finite 24 kHz audio, the
/// 3D mRoPE rotates finitely, and the ECAPA encoder produces a finite embedding. Mirrors the random-weight CPU
/// pattern in <see cref="Qwen3ModelTests"/>.</summary>
public sealed unsafe class Qwen3TtsTests
{
    private static uint _rng = 0x1234u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Zeros(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 0f; return t; }
    private static void AssertFinite(float* p, long n) { for (long i = 0; i < n; i++) Assert.True(float.IsFinite(p[i])); }

    private static Qwen3Config TinyTalker => new()
    {
        HiddenSize = 16, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 2,
        HeadDim = 8, IntermediateSize = 32, MaxPositionEmbeddings = 64,
    };

    private static Qwen3Config TinyMtp => new()
    {
        HiddenSize = 12, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 2,
        HeadDim = 6, IntermediateSize = 24, MaxPositionEmbeddings = 64,
    };

    private static Qwen3TtsConfig TinyCfg => new()
    {
        Talker = TinyTalker,
        CodePredictor = TinyMtp,
        Codec = MimiConfig.Mimi24kHz with { AcousticCodebooks = 15 },
        Vocoder = TinyVocoder,
        TextVocabSize = 32,
        CodecVocabSize = 40,
        CodecHeadOut = 40,
        NumCodeGroups = 16,
        MropeSection = [2, 1, 1],
        MtpCodebooks = 15,
        MtpVocabSize = 20,
        CodecRealVocab = 24,
        CodecEos = 30,
        CodecBos = 29,
        CodecPad = 28,
        CodecNoThink = 27, CodecThinkBos = 26, CodecThinkEos = 25,
        CustomVoiceSpeakerIds = [24, 23, 22],
        MinNewTokens = 1,
        MaxNewTokens = 6,
    };

    private static Qwen3TtsVocoderConfig TinyVocoder => new()
    {
        SemanticCodebookSize = 32, SemanticCodebookDim = 8,
        AcousticCodebookSize = 20, AcousticCodebookDim = 10, AcousticCodebooks = 15,
        LatentDim = 8,
        PreConvOut = 12, PreConvKernel = 3,
        TransformerDim = 8, TransformerLayers = 2, TransformerHeads = 2, TransformerHeadDim = 4,
        SlidingWindow = 4, TransformerFfnDim = 16,
        ConvNeXtUpsampleRates = [2, 2], ConvNeXtDim = 12,
        DecoderInKernel = 3, DecoderInChannels = 16, UpsampleRates = [2, 2],
        ResidualDilations = [1, 3], ResidualKernel = 3, OutKernel = 3,
        SampleRate = 24_000,
    };

    [Fact]
    public void Mrope_RotatesFinitely()
    {
        Qwen3Mrope m = new(8, 1e6f, [2, 1, 1]);
        (float[] cos, float[] sin) = m.ForPositions(4, 0);
        Tensor heads = Fill(new Tensor(new TensorShape(1, 2, 4, 8), DType.F32));
        m.ApplyInPlace(heads, 2, 4, cos, sin);
        AssertFinite((float*)heads.DataPointer, heads.ElementCount);
        heads.Dispose();
    }

    [Fact]
    public void Talker_EmitsFiniteCodebook0()
    {
        Qwen3TtsConfig cfg = TinyCfg;
        using CpuBackend backend = new();
        using Qwen3TtsTalker talker = new(cfg);
        talker.LoadWeights(TalkerWeights(cfg));
        using StreamingKvCache cache = new(cfg.Talker.NumHiddenLayers, 1, cfg.Talker.NumKeyValueHeads, 16, cfg.Talker.HeadDim);

        using Tensor e = talker.EmbedStep(backend, 3, cfg.CodecBos);
        AssertFinite((float*)e.DataPointer, e.ElementCount);
        using Tensor hidden = talker.Forward(backend, e, 1, 0, cache);
        Assert.Equal(new TensorShape(1, 1, cfg.Talker.HiddenSize), hidden.Shape);
        uint rng = 0xABCDu;
        int tok = talker.SampleCodebook0(backend, hidden, ref rng, ReadOnlySpan<int>.Empty);
        Assert.InRange(tok, 0, cfg.CodecHeadOut - 1);
    }

    [Fact]
    public void Mtp_PredictsFiniteCodebooks()
    {
        Qwen3TtsConfig cfg = TinyCfg;
        using CpuBackend backend = new();
        using Qwen3MtpCodePredictor mtp = new(cfg);
        mtp.LoadWeights(MtpWeights(cfg));
        using Tensor talkerHidden = F3(1, 1, cfg.Talker.HiddenSize);
        uint rng = 0x55u;
        int[] codes = mtp.PredictFrame(backend, talkerHidden, 5, 0.9f, 50, 1.0f, ref rng);
        Assert.Equal(cfg.MtpCodebooks, codes.Length);
        foreach (int c in codes) Assert.InRange(c, 0, cfg.MtpVocabSize - 1);
    }

    [Fact]
    public void Vocoder_DecodesFiniteAudio()
    {
        Qwen3TtsVocoderConfig vc = TinyVocoder;
        using CpuBackend backend = new();
        using Qwen3TtsVocoder voc = new(vc);
        voc.LoadWeights(VocoderWeights(vc));
        int frames = 3;
        int[,] grid = new int[16, frames];
        for (int s = 0; s < frames; s++)
        {
            grid[0, s] = (s * 7) % vc.SemanticCodebookSize;
            for (int k = 1; k < 16; k++) grid[k, s] = (s + k) % vc.AcousticCodebookSize;
        }
        float[] pcm = voc.Decode(backend, grid);
        Assert.Equal(frames * vc.TotalUpsample, pcm.Length);
        foreach (float v in pcm) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Vocoder_TotalUpsampleIs1920OnDefaultConfig()
    {
        Assert.Equal(1920, Qwen3TtsVocoderConfig.Default.TotalUpsample);
    }

    [Fact]
    public void Ecapa_ProducesFiniteEmbedding()
    {
        EcapaConfig cfg = new()
        {
            InputChannels = 8, StemChannels = 8,
            Channels = [8], KernelSizes = [3], Dilations = [2],
            Res2Scale = 2, SeBottleneck = 4, EmbeddingDim = 12, ConditioningDim = 16,
        };
        using CpuBackend backend = new();
        using EcapaSpeakerEncoder enc = new(cfg);
        enc.LoadWeights(EcapaWeights(cfg));
        using Tensor mel = F3(1, cfg.InputChannels, 10);
        using Tensor cond = enc.Encode(backend, mel);
        Assert.Equal(new TensorShape(1, cfg.ConditioningDim), cond.Shape);
        AssertFinite((float*)cond.DataPointer, cond.ElementCount);
    }

    // ── synthetic weight builders ───────────────────────────────────────

    private static void AddBackbone(Dictionary<string, Tensor> w, Qwen3Config c, string prefix)
    {
        int h = c.HiddenSize, hd = c.HeadDim;
        w[$"{prefix}.norm.weight"] = Ones(h);
        for (int i = 0; i < c.NumHiddenLayers; i++)
        {
            string p = $"{prefix}.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(c.QDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(c.KvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(c.KvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, c.QDim);
            w[$"{p}.self_attn.q_norm.weight"] = Ones(hd);
            w[$"{p}.self_attn.k_norm.weight"] = Ones(hd);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
    }

    private static Dictionary<string, Tensor> TalkerWeights(Qwen3TtsConfig cfg)
    {
        int h = cfg.Talker.HiddenSize;
        Dictionary<string, Tensor> w = new()
        {
            ["talker.text_embedding.weight"] = F2(cfg.TextVocabSize, h),
            ["talker.codec_embedding.weight"] = F2(cfg.CodecVocabSize, h),
            ["talker.text_projection.0.weight"] = F2(h, h),
            ["talker.text_projection.2.weight"] = F2(h, h),
            ["talker.codec_head.weight"] = F2(cfg.CodecHeadOut, h),
        };
        AddBackbone(w, cfg.Talker, "talker.model");
        return w;
    }

    private static Dictionary<string, Tensor> MtpWeights(Qwen3TtsConfig cfg)
    {
        int mh = cfg.CodePredictor.HiddenSize;
        Dictionary<string, Tensor> w = new()
        {
            ["code_predictor.small_to_mtp_projection.weight"] = F2(mh, cfg.Talker.HiddenSize),
        };
        for (int k = 0; k < cfg.MtpCodebooks; k++)
        {
            w[$"code_predictor.codec_embedding.{k}.weight"] = F2(cfg.MtpVocabSize, mh);
            w[$"code_predictor.lm_head.{k}.weight"] = F2(cfg.MtpVocabSize, mh);
        }
        AddBackbone(w, cfg.CodePredictor, "code_predictor.model");
        return w;
    }

    private static Dictionary<string, Tensor> VocoderWeights(Qwen3TtsVocoderConfig vc)
    {
        Dictionary<string, Tensor> w = new()
        {
            ["vocoder.quantizer.semantic.codebook"] = F2(vc.SemanticCodebookSize, vc.SemanticCodebookDim),
            ["vocoder.quantizer.semantic.in_proj.weight"] = F2(vc.LatentDim, vc.SemanticCodebookDim),
        };
        for (int k = 0; k < vc.AcousticCodebooks; k++)
        {
            w[$"vocoder.quantizer.acoustic.{k}.codebook"] = F2(vc.AcousticCodebookSize, vc.AcousticCodebookDim);
            w[$"vocoder.quantizer.acoustic.{k}.in_proj.weight"] = F2(vc.LatentDim, vc.AcousticCodebookDim);
        }
        // Transformer.
        int dim = vc.TransformerDim, hd = vc.TransformerHeadDim, heads = vc.TransformerHeads;
        for (int i = 0; i < vc.TransformerLayers; i++)
        {
            string p = $"vocoder.transformer.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(dim);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(dim);
            w[$"{p}.self_attn.q_proj.weight"] = F2(heads * hd, dim);
            w[$"{p}.self_attn.k_proj.weight"] = F2(heads * hd, dim);
            w[$"{p}.self_attn.v_proj.weight"] = F2(heads * hd, dim);
            w[$"{p}.self_attn.o_proj.weight"] = F2(dim, heads * hd);
            w[$"{p}.mlp.gate_proj.weight"] = F2(vc.TransformerFfnDim, dim);
            w[$"{p}.mlp.up_proj.weight"] = F2(vc.TransformerFfnDim, dim);
            w[$"{p}.mlp.down_proj.weight"] = F2(dim, vc.TransformerFfnDim);
            w[$"{p}.layer_scale_1.scale"] = Ones(dim);
            w[$"{p}.layer_scale_2.scale"] = Ones(dim);
        }
        // pre_conv: transformerDim → convnextDim.
        w["vocoder.pre_conv.weight"] = F3(vc.ConvNeXtDim, vc.TransformerDim, vc.PreConvKernel);
        // upsample stages.
        for (int i = 0; i < vc.ConvNeXtUpsampleRates.Count; i++)
        {
            string p = $"vocoder.upsample.{i}";
            int rate = vc.ConvNeXtUpsampleRates[i];
            w[$"{p}.conv.weight"] = F3(vc.ConvNeXtDim, vc.ConvNeXtDim, rate * 2);   // [Cin, Cout, K]
            AddConvNeXt(w, $"{p}.convnext", vc.ConvNeXtDim);
        }
        // decoder in_conv convnextDim→decoderInChannels.
        w["vocoder.decoder.in_conv.weight"] = F3(vc.DecoderInChannels, vc.ConvNeXtDim, vc.DecoderInKernel);
        int ch = vc.DecoderInChannels;
        for (int i = 0; i < vc.UpsampleRates.Count; i++)
        {
            int outCh = ch / 2;
            int rate = vc.UpsampleRates[i];
            string p = $"vocoder.decoder.blocks.{i}";
            w[$"{p}.snake.alpha"] = F3(1, ch, 1);
            w[$"{p}.up_conv.weight"] = F3(ch, outCh, rate * 2);
            for (int r = 0; r < vc.ResidualDilations.Count; r++)
            {
                string rp = $"{p}.residuals.{r}";
                w[$"{rp}.snake1.alpha"] = F3(1, outCh, 1);
                w[$"{rp}.conv1.weight"] = F3(outCh, outCh, vc.ResidualKernel);
                w[$"{rp}.snake2.alpha"] = F3(1, outCh, 1);
                w[$"{rp}.conv2.weight"] = F3(outCh, outCh, 1);
            }
            ch = outCh;
        }
        w["vocoder.decoder.final_snake.alpha"] = F3(1, ch, 1);
        w["vocoder.decoder.out_conv.weight"] = F3(1, ch, vc.OutKernel);
        return w;
    }

    private static void AddConvNeXt(Dictionary<string, Tensor> w, string p, int c)
    {
        int hidden = c * 2;
        w[$"{p}.dwconv.weight"] = F3(c, 1, 7);          // depthwise [C,1,K]
        w[$"{p}.norm.weight"] = Ones(c);
        w[$"{p}.norm.bias"] = Zeros(c);
        w[$"{p}.pwconv1.weight"] = F2(hidden, c);
        w[$"{p}.pwconv2.weight"] = F2(c, hidden);
        w[$"{p}.gamma"] = Ones(c);
    }

    private static Dictionary<string, Tensor> EcapaWeights(EcapaConfig cfg)
    {
        Dictionary<string, Tensor> w = new();
        AddTdnn(w, "speaker_encoder.tdnn", cfg.InputChannels, cfg.StemChannels, 5);
        int inCh = cfg.StemChannels;
        int concat = 0;
        for (int i = 0; i < cfg.Channels.Count; i++)
        {
            string p = $"speaker_encoder.blocks.{i}";
            int outCh = cfg.Channels[i];
            w[$"{p}.conv1.weight"] = F3(outCh, inCh, 1); AddBn(w, $"{p}.bn1", outCh);
            w[$"{p}.res2.weight"] = F3(outCh, outCh / cfg.Res2Scale, cfg.KernelSizes[i]); AddBn(w, $"{p}.bn2", outCh);
            w[$"{p}.conv3.weight"] = F3(outCh, outCh, 1); AddBn(w, $"{p}.bn3", outCh);
            w[$"{p}.se.fc1.weight"] = F2(cfg.SeBottleneck, outCh);
            w[$"{p}.se.fc2.weight"] = F2(outCh, cfg.SeBottleneck);
            inCh = outCh;
            concat += outCh;
        }
        int aggCh = concat;
        w["speaker_encoder.mfa.weight"] = F3(aggCh, concat, 1);
        int attDim = 8;
        w["speaker_encoder.asp.tdnn.weight"] = F3(attDim, 3 * aggCh, 1);
        w["speaker_encoder.asp.conv.weight"] = F3(aggCh, attDim, 1);
        w["speaker_encoder.fc.weight"] = F2(cfg.EmbeddingDim, 2 * aggCh);
        AddBnFlat(w, "speaker_encoder.bn", cfg.EmbeddingDim);
        w["speaker_encoder.proj.weight"] = F2(cfg.ConditioningDim, cfg.EmbeddingDim);
        return w;
    }

    private static void AddTdnn(Dictionary<string, Tensor> w, string p, int inCh, int outCh, int k)
    {
        w[$"{p}.conv.weight"] = F3(outCh, inCh, k);
        AddBn(w, $"{p}.bn", outCh);
    }

    private static void AddBn(Dictionary<string, Tensor> w, string p, int c)
    {
        w[$"{p}.weight"] = Ones(c);
        w[$"{p}.bias"] = Zeros(c);
        w[$"{p}.running_mean"] = Zeros(c);
        w[$"{p}.running_var"] = Ones(c);
    }

    private static void AddBnFlat(Dictionary<string, Tensor> w, string p, int c) => AddBn(w, p, c);
}

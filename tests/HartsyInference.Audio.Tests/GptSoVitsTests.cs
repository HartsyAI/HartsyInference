using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for GPT-SoVITS stage 1 (T2S AR GPT): config, and a synthetic-weights generation producing
/// valid in-range semantic tokens (exercises the post-LN transformer + sinusoidal positions + text/audio
/// span masking + sampling).</summary>
public sealed unsafe class GptSoVitsTests
{
    private static uint _rng = 0x50A1u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void V2_Config()
    {
        Text2SemanticConfig c = Text2SemanticConfig.V2;
        Assert.Equal(512, c.Hidden);
        Assert.Equal(24, c.NumLayers);
        Assert.Equal(16, c.NumHeads);
        Assert.Equal(32, c.HeadDim);
        Assert.Equal(1_025, c.SemanticVocab);
        Assert.Equal(1_024, c.EosToken);
        Assert.Equal(15, c.TopK);
    }

    [Fact]
    public void T2S_SyntheticForward_GeneratesValidSemanticTokens()
    {
        Text2SemanticConfig c = new()
        {
            Hidden = 16, NumLayers = 2, NumHeads = 2, FfnDim = 32,
            PhonemeVocab = 10, SemanticVocab = 8, EosToken = 7, BertDim = 6,
        };
        using CpuBackend backend = new();
        using Text2Semantic t2s = new(c);
        t2s.LoadWeights(Weights(c));

        int[] phon = [2, 5, 1], prefix = [3, 1];
        using Tensor bert = F3(1, c.BertDim, phon.Length);
        List<int> tokens = t2s.Generate(backend, phon, bert, prefix, maxNew: 5, seed: 9);
        Assert.True(tokens.Count <= 5);
        foreach (int tok in tokens) { Assert.InRange(tok, 0, c.SemanticVocab - 1); Assert.NotEqual(c.EosToken, tok); }
    }

    [Fact]
    public void SoVits_SyntheticForward_SemanticTokensToFiniteAudio()
    {
        int gin = 4, ssl = 6;
        VitsConfig core = new()
        {
            InterChannels = 8, HiddenChannels = 8, GinChannels = gin, FlowLayers = 2, FlowFlows = 2, FlowKernelSize = 3,
            ResBlock = "2", ResBlockKernelSizes = [3], ResBlockDilations = [[1, 2]],
            UpsampleRates = [4, 4], UpsampleInitialChannel = 16, UpsampleKernelSizes = [8, 8], SampleRate = 32_000,
        };
        using CpuBackend backend = new();
        using SoVitsSynthesizer sv = new(core, sslDim: ssl);
        sv.LoadWeights(SoVitsWeights(core, gin, ssl, codebookSize: 5));

        int[] semantic = [2, 0, 4];
        using Tensor ge = F3(1, gin, 1);
        float[] audio = sv.Forward(backend, semantic, ge, seed: 4);
        Assert.True(audio.Length > 0);
        foreach (float s in audio) Assert.True(float.IsFinite(s));
    }

    private static Dictionary<string, Tensor> SoVitsWeights(VitsConfig c, int gin, int ssl, int codebookSize)
    {
        int h = c.HiddenChannels, inter = c.InterChannels, half = inter / 2;
        Dictionary<string, Tensor> w = new()
        {
            ["quantizer.vq.layers.0._codebook.embed"] = F2(codebookSize, ssl),
            ["enc_p.ssl_proj.weight"] = F3(inter, ssl, 1), ["enc_p.ssl_proj.bias"] = F1(inter),
            ["enc_p.encoder_ssl_proj.weight"] = F3(inter, inter, 1), ["enc_p.encoder_ssl_proj.bias"] = F1(inter),
            ["enc_p.proj.weight"] = F3(2 * inter, inter, 1), ["enc_p.proj.bias"] = F1(2 * inter),
            ["dec.conv_pre.weight"] = F3(c.UpsampleInitialChannel, inter, 7), ["dec.conv_pre.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.cond.weight"] = F3(c.UpsampleInitialChannel, gin, 1), ["dec.cond.bias"] = F1(c.UpsampleInitialChannel),
            ["dec.conv_post.weight"] = F3(1, c.UpsampleInitialChannel >> c.UpsampleRates.Count, 7), ["dec.conv_post.bias"] = F1(1),
        };
        for (int i = 0; i < c.FlowFlows; i++)
        {
            string f = $"flow.flows.{2 * i}";
            w[$"{f}.pre.weight"] = F3(h, half, 1); w[$"{f}.pre.bias"] = F1(h);
            w[$"{f}.post.weight"] = F3(half, h, 1); w[$"{f}.post.bias"] = F1(half);
            w[$"{f}.enc.cond_layer.weight"] = F3(2 * h * c.FlowLayers, gin, 1); w[$"{f}.enc.cond_layer.bias"] = F1(2 * h * c.FlowLayers);
            for (int j = 0; j < c.FlowLayers; j++)
            {
                w[$"{f}.enc.in_layers.{j}.weight"] = F3(2 * h, h, c.FlowKernelSize); w[$"{f}.enc.in_layers.{j}.bias"] = F1(2 * h);
                int rs = j < c.FlowLayers - 1 ? 2 * h : h;
                w[$"{f}.enc.res_skip_layers.{j}.weight"] = F3(rs, h, 1); w[$"{f}.enc.res_skip_layers.{j}.bias"] = F1(rs);
            }
        }
        for (int i = 0; i < c.UpsampleRates.Count; i++)
        {
            int inCh = c.UpsampleInitialChannel >> i, outCh = inCh >> 1;
            w[$"dec.ups.{i}.weight"] = F3(inCh, outCh, c.UpsampleKernelSizes[i]); w[$"dec.ups.{i}.bias"] = F1(outCh);
            for (int dn = 0; dn < c.ResBlockDilations[0].Count; dn++)
            { w[$"dec.resblocks.{i}.convs1.{dn}.weight"] = F3(outCh, outCh, 3); w[$"dec.resblocks.{i}.convs1.{dn}.bias"] = F1(outCh); }
        }
        return w;
    }

    private static Dictionary<string, Tensor> Weights(Text2SemanticConfig c)
    {
        int h = c.Hidden;
        Dictionary<string, Tensor> w = new()
        {
            ["model.ar_text_embedding.word_embeddings.weight"] = F2(c.PhonemeVocab, h),
            ["model.ar_audio_embedding.word_embeddings.weight"] = F2(c.SemanticVocab, h),
            ["model.bert_proj.weight"] = F2(h, c.BertDim), ["model.bert_proj.bias"] = F1(h),
            ["model.ar_predict_layer.weight"] = F2(c.SemanticVocab, h),
            ["model.ar_text_position.alpha"] = F1(1), ["model.ar_audio_position.alpha"] = F1(1),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.h.layers.{i}";
            w[$"{p}.self_attn.in_proj_weight"] = F2(3 * h, h); w[$"{p}.self_attn.in_proj_bias"] = F1(3 * h);
            w[$"{p}.self_attn.out_proj.weight"] = F2(h, h); w[$"{p}.self_attn.out_proj.bias"] = F1(h);
            w[$"{p}.linear1.weight"] = F2(c.FfnDim, h); w[$"{p}.linear1.bias"] = F1(c.FfnDim);
            w[$"{p}.linear2.weight"] = F2(h, c.FfnDim); w[$"{p}.linear2.bias"] = F1(h);
            w[$"{p}.norm1.weight"] = F1(h); w[$"{p}.norm1.bias"] = F1(h);
            w[$"{p}.norm2.weight"] = F1(h); w[$"{p}.norm2.bias"] = F1(h);
        }
        return w;
    }
}

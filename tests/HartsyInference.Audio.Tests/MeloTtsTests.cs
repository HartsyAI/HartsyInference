using HartsyInference.Audio.Models.MeloTts;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for MeloTTS: the config, and a synthetic-weights forward through the extended text encoder
/// (phoneme + tone + language + BERT summed embedding → reused VITS encoder layers) asserting finite prior.</summary>
public sealed unsafe class MeloTtsTests
{
    private static uint _rng = 0xBADF00Du;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void EnglishV3_ConfigReusesVitsCore()
    {
        MeloTtsConfig c = MeloTtsConfig.EnglishV3;
        Assert.Equal(192, c.Core.InterChannels);
        Assert.Equal(256, c.Core.GinChannels);        // multispeaker
        Assert.Equal(1_024, c.BertDim);
        Assert.Equal(768, c.JaBertDim);
        Assert.Equal(0.2f, c.SdpRatio);
    }

    [Fact]
    public void TextEncoder_SyntheticForward_IsFinite()
    {
        MeloTtsConfig c = new()
        {
            Core = new VitsConfig
            {
                NumVocab = 16, InterChannels = 8, HiddenChannels = 8, FilterChannels = 16, NumHeads = 2,
                NumEncoderLayers = 2, WindowSize = 4,
            },
            NumTones = 4, NumLanguages = 3, BertDim = 6, JaBertDim = 5,
        };
        using CpuBackend backend = new();
        MeloTtsTextEncoder enc = new(c);
        enc.LoadWeights(Weights(c));

        int t = 3;
        int[] phon = [2, 5, 1], tones = [1, 0, 3], langs = [0, 2, 1];
        using Tensor bert = F3(1, c.BertDim, t);
        using Tensor jaBert = F3(1, c.JaBertDim, t);
        (Tensor hidden, Tensor mP, Tensor logsP) = enc.Forward(backend, phon, tones, langs, bert, jaBert);
        Assert.Equal(new TensorShape(1, c.Core.InterChannels, t), mP.Shape);
        AssertFinite(mP); AssertFinite(logsP); AssertFinite(hidden);
        hidden.Dispose(); mP.Dispose(); logsP.Dispose();
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> Weights(MeloTtsConfig c)
    {
        int h = c.Core.HiddenChannels, inter = c.Core.InterChannels, kc = h / c.Core.NumHeads;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_p.emb.weight"] = F2(c.Core.NumVocab, h),
            ["enc_p.tone_emb.weight"] = F2(c.NumTones, h),
            ["enc_p.language_emb.weight"] = F2(c.NumLanguages, h),
            ["enc_p.bert_proj.weight"] = F3(h, c.BertDim, 1), ["enc_p.bert_proj.bias"] = F1(h),
            ["enc_p.ja_bert_proj.weight"] = F3(h, c.JaBertDim, 1), ["enc_p.ja_bert_proj.bias"] = F1(h),
            ["enc_p.proj.weight"] = F3(2 * inter, h, 1), ["enc_p.proj.bias"] = F1(2 * inter),
        };
        for (int i = 0; i < c.Core.NumEncoderLayers; i++)
        {
            string e = "enc_p.encoder";
            foreach (string p in new[] { "conv_q", "conv_k", "conv_v", "conv_o" })
            { w[$"{e}.attn_layers.{i}.{p}.weight"] = F3(h, h, 1); w[$"{e}.attn_layers.{i}.{p}.bias"] = F1(h); }
            w[$"{e}.attn_layers.{i}.emb_rel_k"] = F3(1, 2 * c.Core.WindowSize + 1, kc);
            w[$"{e}.attn_layers.{i}.emb_rel_v"] = F3(1, 2 * c.Core.WindowSize + 1, kc);
            w[$"{e}.norm_layers_1.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_1.{i}.beta"] = F1(h);
            w[$"{e}.norm_layers_2.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_2.{i}.beta"] = F1(h);
            w[$"{e}.ffn_layers.{i}.conv_1.weight"] = F3(c.Core.FilterChannels, h, 3); w[$"{e}.ffn_layers.{i}.conv_1.bias"] = F1(c.Core.FilterChannels);
            w[$"{e}.ffn_layers.{i}.conv_2.weight"] = F3(h, c.Core.FilterChannels, 3); w[$"{e}.ffn_layers.{i}.conv_2.bias"] = F1(h);
        }
        return w;
    }
}

using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the real GPT-SoVITS SoVITS prior encoder (<see cref="SoVitsEncP"/> with MRTE
/// cross-attention) and the reference encoder (<see cref="SoVitsRefEnc"/> MelStyleEncoder). Synthetic-weights
/// forwards on the CPU backend with tiny dims: assert finite, correctly-shaped outputs.</summary>
public sealed unsafe class SoVitsEncPTests
{
    private static uint _rng = 0x1357u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void RefEnc_SyntheticForward_ProducesFiniteGe()
    {
        int nMels = 12, hidden = 8, styleDim = 10, nHead = 2, kernel = 3, t = 7;
        using CpuBackend backend = new();
        SoVitsRefEnc refEnc = new(nMels, hidden, styleDim, nHead, kernel);
        refEnc.LoadWeights(RefEncWeights(nMels, hidden, styleDim, kernel));

        using Tensor spec = F3(1, nMels, t);
        using Tensor ge = refEnc.Forward(backend, spec, t);
        Assert.Equal(1, (int)ge.Shape[0]);
        Assert.Equal(styleDim, (int)ge.Shape[1]);
        Assert.Equal(1, (int)ge.Shape[2]);
        float* p = (float*)ge.DataPointer;
        for (int i = 0; i < styleDim; i++) Assert.True(float.IsFinite(p[i]));
    }

    [Fact]
    public void EncP_SyntheticForward_ProducesFinitePrior()
    {
        // ge is added directly in the MRTE channel space, so gin_channels == mrte hidden_size.
        int gin = 8, ssl = 6, mrteHidden = 8, mrteHeads = 2;
        VitsConfig core = new()
        {
            InterChannels = 4, HiddenChannels = 4, FilterChannels = 8, NumHeads = 2,
            NumEncoderLayers = 2, WindowSize = 2, GinChannels = gin,
        };
        using CpuBackend backend = new();
        SoVitsEncP encP = new(core, sslDim: ssl, sslLayers: 2, textLayers: 2, enc2Layers: 2,
            mrteHidden: mrteHidden, mrteHeads: mrteHeads);
        encP.LoadWeights(EncPWeights(core, ssl, gin, mrteHidden, codebook: 5));

        int t = 6;
        int[] textTokens = [1, 3, 0, 4, 2, 1];
        using Tensor latent = F3(1, ssl, t);
        using Tensor ge = F3(1, gin, 1);
        (Tensor mP, Tensor logsP) = encP.Forward(backend, latent, t, textTokens, ge);
        using (mP)
        using (logsP)
        {
            Assert.Equal(core.InterChannels, (int)mP.Shape[1]);
            Assert.Equal(t, (int)mP.Shape[2]);
            float* mp = (float*)mP.DataPointer;
            float* lp = (float*)logsP.DataPointer;
            for (long i = 0; i < mP.ElementCount; i++) { Assert.True(float.IsFinite(mp[i])); Assert.True(float.IsFinite(lp[i])); }
        }
    }

    private static Dictionary<string, Tensor> RefEncWeights(int nMels, int hidden, int styleDim, int kernel)
    {
        return new Dictionary<string, Tensor>
        {
            ["ref_enc.spectral.0.fc.weight"] = F2(hidden, nMels), ["ref_enc.spectral.0.fc.bias"] = F1(hidden),
            ["ref_enc.spectral.3.fc.weight"] = F2(hidden, hidden), ["ref_enc.spectral.3.fc.bias"] = F1(hidden),
            ["ref_enc.temporal.0.conv1.conv.weight"] = F3(2 * hidden, hidden, kernel),
            ["ref_enc.temporal.0.conv1.conv.bias"] = F1(2 * hidden),
            ["ref_enc.temporal.1.conv1.conv.weight"] = F3(2 * hidden, hidden, kernel),
            ["ref_enc.temporal.1.conv1.conv.bias"] = F1(2 * hidden),
            ["ref_enc.slf_attn.w_qs.weight"] = F2(hidden, hidden), ["ref_enc.slf_attn.w_qs.bias"] = F1(hidden),
            ["ref_enc.slf_attn.w_ks.weight"] = F2(hidden, hidden), ["ref_enc.slf_attn.w_ks.bias"] = F1(hidden),
            ["ref_enc.slf_attn.w_vs.weight"] = F2(hidden, hidden), ["ref_enc.slf_attn.w_vs.bias"] = F1(hidden),
            ["ref_enc.slf_attn.fc.weight"] = F2(hidden, hidden), ["ref_enc.slf_attn.fc.bias"] = F1(hidden),
            ["ref_enc.fc.fc.weight"] = F2(styleDim, hidden), ["ref_enc.fc.fc.bias"] = F1(styleDim),
        };
    }

    private static Dictionary<string, Tensor> EncPWeights(VitsConfig c, int ssl, int gin, int mrteHidden, int codebook)
    {
        int h = c.HiddenChannels, inter = c.InterChannels, f = c.FilterChannels, kc = h / c.NumHeads, win = c.WindowSize;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_p.ssl_proj.weight"] = F3(h, ssl, 1), ["enc_p.ssl_proj.bias"] = F1(h),
            ["enc_p.text_embedding.weight"] = F2(codebook, h),
            ["enc_p.proj.weight"] = F3(2 * inter, h, 1), ["enc_p.proj.bias"] = F1(2 * inter),
            // c_pre/text_pre lift h→mrte_hidden; cross-attention runs in mrte_hidden; c_post maps back to h.
            ["enc_p.mrte.c_pre.weight"] = F3(mrteHidden, h, 1), ["enc_p.mrte.c_pre.bias"] = F1(mrteHidden),
            ["enc_p.mrte.c_post.weight"] = F3(h, mrteHidden, 1), ["enc_p.mrte.c_post.bias"] = F1(h),
            ["enc_p.mrte.cross_attention.conv_q.weight"] = F3(mrteHidden, mrteHidden, 1), ["enc_p.mrte.cross_attention.conv_q.bias"] = F1(mrteHidden),
            ["enc_p.mrte.cross_attention.conv_k.weight"] = F3(mrteHidden, mrteHidden, 1), ["enc_p.mrte.cross_attention.conv_k.bias"] = F1(mrteHidden),
            ["enc_p.mrte.cross_attention.conv_v.weight"] = F3(mrteHidden, mrteHidden, 1), ["enc_p.mrte.cross_attention.conv_v.bias"] = F1(mrteHidden),
            ["enc_p.mrte.cross_attention.conv_o.weight"] = F3(mrteHidden, mrteHidden, 1), ["enc_p.mrte.cross_attention.conv_o.bias"] = F1(mrteHidden),
            ["enc_p.mrte.text_pre.weight"] = F3(mrteHidden, h, 1), ["enc_p.mrte.text_pre.bias"] = F1(mrteHidden),
        };
        AddEncoder(w, "enc_p.encoder_ssl", 2, h, f, kc, win);
        AddEncoder(w, "enc_p.encoder_text", 2, h, f, kc, win);
        AddEncoder(w, "enc_p.encoder2", 2, h, f, kc, win);
        return w;
    }

    private static void AddEncoder(Dictionary<string, Tensor> w, string p, int layers, int h, int f, int kc, int win)
    {
        for (int i = 0; i < layers; i++)
        {
            w[$"{p}.attn_layers.{i}.conv_q.weight"] = F3(h, h, 1); w[$"{p}.attn_layers.{i}.conv_q.bias"] = F1(h);
            w[$"{p}.attn_layers.{i}.conv_k.weight"] = F3(h, h, 1); w[$"{p}.attn_layers.{i}.conv_k.bias"] = F1(h);
            w[$"{p}.attn_layers.{i}.conv_v.weight"] = F3(h, h, 1); w[$"{p}.attn_layers.{i}.conv_v.bias"] = F1(h);
            w[$"{p}.attn_layers.{i}.conv_o.weight"] = F3(h, h, 1); w[$"{p}.attn_layers.{i}.conv_o.bias"] = F1(h);
            w[$"{p}.attn_layers.{i}.emb_rel_k"] = F3(1, 2 * win + 1, kc);
            w[$"{p}.attn_layers.{i}.emb_rel_v"] = F3(1, 2 * win + 1, kc);
            w[$"{p}.norm_layers_1.{i}.gamma"] = F1(h); w[$"{p}.norm_layers_1.{i}.beta"] = F1(h);
            w[$"{p}.norm_layers_2.{i}.gamma"] = F1(h); w[$"{p}.norm_layers_2.{i}.beta"] = F1(h);
            w[$"{p}.ffn_layers.{i}.conv_1.weight"] = F3(f, h, 3); w[$"{p}.ffn_layers.{i}.conv_1.bias"] = F1(f);
            w[$"{p}.ffn_layers.{i}.conv_2.weight"] = F3(h, f, 3); w[$"{p}.ffn_layers.{i}.conv_2.bias"] = F1(h);
        }
    }
}

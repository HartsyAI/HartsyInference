using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for resemble-enhance's LCFM enhancer: config, and a synthetic-weights forward of the WN CFM
/// velocity net + IRMAE (encoder loaded, decoder exercised) using the real checkpoint key layout
/// (<c>lcfm.cfm.net.layers.i.{gconv,dconv,lconv,out}</c>, Sequential-indexed <c>lcfm.ae.*</c> with legacy
/// <c>weight_g</c>/<c>weight_v</c> pairs) — the structure verified against upstream <c>wn.py</c>/<c>irmae.py</c>
/// and the enhancer_stage2 DeepSpeed checkpoint.</summary>
public sealed unsafe class ResembleEnhanceTests
{
    private static uint _rng = 0xEA0Cu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor Ones3(int a) { Tensor t = new(new TensorShape(a, 1, 1), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < a; i++) p[i] = 1f; return t; }
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void Default_Config()
    {
        ResembleEnhanceConfig c = ResembleEnhanceConfig.Default;
        Assert.Equal(44_100, c.SampleRate);
        Assert.Equal(128, c.NMels);
        Assert.Equal(64, c.LatentDim);
        Assert.Equal(30, c.WnLayers);
        Assert.Equal(512, c.WnHidden);
        Assert.Equal("midpoint", c.Solver);
        // enhancer_stage2/hparams.yaml pins lcfm_z_scale to 6 (the upstream code default of 5 is stale).
        Assert.Equal(6f, c.LatentScale);
    }

    [Fact]
    public void Lcfm_SyntheticForward_CondMelToDecodedFeatures()
    {
        ResembleEnhanceConfig c = new()
        {
            NMels = 8, LatentDim = 4, WnLayers = 2, WnHidden = 8, WnKernel = 3, WnDilationCycle = 2, TimeEmbDim = 8,
            AeHidden = 16, AeResBlocks = 2, AeDilations = [1, 2], IrmaeGroupNorm = 4, NumIrms = 2,
            VocoderExtraDim = 4, Nfe = 4, Solver = "euler",
        };
        using CpuBackend backend = new();
        using ResembleEnhancePipeline pipe = new(c);
        pipe.LoadWeights(Weights(c));

        int t = 5;
        using Tensor condMel = F3(1, c.NMels, t);
        using Tensor decoded = pipe.EnhanceMel(backend, condMel, seed: 2);
        Assert.Equal(new TensorShape(1, c.NMels + c.VocoderExtraDim, t), decoded.Shape);
        float* p = (float*)decoded.DataPointer;
        for (long i = 0; i < decoded.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> Weights(ResembleEnhanceConfig c)
    {
        int h = c.WnHidden, lat = c.LatentDim, mels = c.NMels, ae = c.AeHidden;
        int aeOut = mels + c.VocoderExtraDim;
        Dictionary<string, Tensor> w = new()
        {
            // WN velocity net: start/end 1×1 convs + per-layer gconv/dconv/lconv/out.
            ["lcfm.cfm.net.start.weight"] = F3(h, lat, 1), ["lcfm.cfm.net.start.bias"] = F1(h),
            ["lcfm.cfm.net.end.weight"] = F3(lat, h, 1), ["lcfm.cfm.net.end.bias"] = F1(lat),
            // IRMAE encoder/decoder input convs (Sequential index 0).
            ["lcfm.ae.encoder.0.weight"] = F3(ae, mels, 3), ["lcfm.ae.encoder.0.bias"] = F1(ae),
            ["lcfm.ae.decoder.0.weight"] = F3(ae, lat, 3), ["lcfm.ae.decoder.0.bias"] = F1(ae),
        };
        for (int i = 0; i < c.WnLayers; i++)
        {
            string p = $"lcfm.cfm.net.layers.{i}";
            w[$"{p}.gconv.weight"] = F3(h, c.TimeEmbDim, 1); w[$"{p}.gconv.bias"] = F1(h);
            w[$"{p}.dconv.weight"] = F3(2 * h, h, c.WnKernel); w[$"{p}.dconv.bias"] = F1(2 * h);
            w[$"{p}.lconv.weight"] = F3(2 * h, mels, 1); w[$"{p}.lconv.bias"] = F1(2 * h);
            w[$"{p}.out.weight"] = F3(2 * h, h, 1); w[$"{p}.out.bias"] = F1(2 * h);
        }
        for (int b = 0; b < c.AeResBlocks; b++)
        {
            AddResBlock(w, $"lcfm.ae.encoder.{b + 1}", ae, c.AeDilations.Count);
            AddResBlock(w, $"lcfm.ae.decoder.{b + 1}", ae, c.AeDilations.Count);
        }
        // Bias-free IRM 1×1 chain after the encoder ResBlocks, then the decoder output conv to mels+extra.
        for (int i = 0; i < c.NumIrms; i++)
        {
            w[$"lcfm.ae.encoder.{c.AeResBlocks + 1 + i}.weight"] = i == 0 ? F3(lat, ae, 1) : F3(lat, lat, 1);
        }
        w[$"lcfm.ae.decoder.{c.AeResBlocks + 1}.weight"] = F3(aeOut, ae, 1);
        w[$"lcfm.ae.decoder.{c.AeResBlocks + 1}.bias"] = F1(aeOut);
        return w;
    }

    /// <summary>One upstream irmae.ResBlock: per stage i, Sequential index 3i is a GroupNorm and 3i+2 a
    /// weight-normed dilated conv.</summary>
    private static void AddResBlock(Dictionary<string, Tensor> w, string p, int ch, int stages)
    {
        for (int i = 0; i < stages; i++)
        {
            w[$"{p}.{3 * i}.weight"] = Ones(ch); w[$"{p}.{3 * i}.bias"] = F1(ch);
            w[$"{p}.{3 * i + 2}.weight_g"] = Ones3(ch); w[$"{p}.{3 * i + 2}.weight_v"] = F3(ch, ch, 3);
            w[$"{p}.{3 * i + 2}.bias"] = F1(ch);
        }
    }
}

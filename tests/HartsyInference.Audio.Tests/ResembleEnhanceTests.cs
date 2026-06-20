using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for resemble-enhance's LCFM enhancer: config, and a synthetic-weights forward of the WN CFM
/// velocity net + IRMAE decoder driven by the reused OT-CFM solver (conditioning mel → enhanced mel, finite).</summary>
public sealed unsafe class ResembleEnhanceTests
{
    private static uint _rng = 0xEA0Cu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
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
    }

    [Fact]
    public void Lcfm_SyntheticForward_CondMelToEnhancedMel()
    {
        ResembleEnhanceConfig c = new()
        {
            NMels = 8, LatentDim = 4, WnLayers = 2, WnHidden = 8, WnKernel = 3, WnDilationCycle = 2, TimeEmbDim = 8,
            AeHidden = 16, AeResBlocks = 2, AeDilations = [1, 2], Nfe = 4, Solver = "euler",
        };
        using CpuBackend backend = new();
        using ResembleEnhancePipeline pipe = new(c);
        pipe.LoadWeights(Weights(c));

        int t = 5;
        using Tensor condMel = F3(1, c.NMels, t);
        using Tensor mel = pipe.EnhanceMel(backend, condMel, t, seed: 2);
        Assert.Equal(new TensorShape(1, c.NMels, t), mel.Shape);
        float* p = (float*)mel.DataPointer;
        for (long i = 0; i < mel.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> Weights(ResembleEnhanceConfig c)
    {
        int h = c.WnHidden, lat = c.LatentDim, mels = c.NMels, ae = c.AeHidden;
        Dictionary<string, Tensor> w = new()
        {
            // WN estimator.
            ["lcfm.cfm.net.start.weight"] = F3(h, lat, 1), ["lcfm.cfm.net.start.bias"] = F1(h),
            ["lcfm.cfm.net.global_cond.weight"] = F2(h, c.TimeEmbDim), ["lcfm.cfm.net.global_cond.bias"] = F1(h),
            ["lcfm.cfm.net.local_cond.weight"] = F3(2 * h, mels, 1), ["lcfm.cfm.net.local_cond.bias"] = F1(2 * h),
            ["lcfm.cfm.net.out.weight"] = F3(lat, h, 1), ["lcfm.cfm.net.out.bias"] = F1(lat),
            // IRMAE decoder.
            ["lcfm.ae.decoder.in_conv.weight"] = F3(ae, lat, 3), ["lcfm.ae.decoder.in_conv.bias"] = F1(ae),
            ["lcfm.ae.decoder.out_conv.weight"] = F3(mels, ae, 1), ["lcfm.ae.decoder.out_conv.bias"] = F1(mels),
            ["lcfm.ae.decoder.head.0.weight"] = F3(ae, mels, 3), ["lcfm.ae.decoder.head.0.bias"] = F1(ae),
            ["lcfm.ae.decoder.head.2.weight"] = F3(mels, ae, 1), ["lcfm.ae.decoder.head.2.bias"] = F1(mels),
        };
        for (int i = 0; i < c.WnLayers; i++)
        {
            w[$"lcfm.cfm.net.layers.{i}.dilated.weight"] = F3(2 * h, h, c.WnKernel); w[$"lcfm.cfm.net.layers.{i}.dilated.bias"] = F1(2 * h);
            w[$"lcfm.cfm.net.layers.{i}.res_skip.weight"] = F3(2 * h, h, 1); w[$"lcfm.cfm.net.layers.{i}.res_skip.bias"] = F1(2 * h);
        }
        for (int i = 0; i < c.AeResBlocks; i++)
        {
            string p = $"lcfm.ae.decoder.res.{i}";
            w[$"{p}.norm1.weight"] = Ones(ae); w[$"{p}.norm1.bias"] = F1(ae);
            w[$"{p}.conv1.weight"] = F3(ae, ae, 3); w[$"{p}.conv1.bias"] = F1(ae);
            w[$"{p}.norm2.weight"] = Ones(ae); w[$"{p}.norm2.bias"] = F1(ae);
            w[$"{p}.conv2.weight"] = F3(ae, ae, 3); w[$"{p}.conv2.bias"] = F1(ae);
        }
        return w;
    }
}

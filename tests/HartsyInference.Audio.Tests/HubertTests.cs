using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the HuBERT content encoder (shared by GPT-SoVITS + RVC): config, and a synthetic-weights
/// forward through the conv feature extractor + positional conv + post-LN transformer producing finite
/// content features.</summary>
public sealed unsafe class HubertTests
{
    private static uint _rng = 0xB0BBu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void ChineseHubertBase_Config()
    {
        HubertConfig c = HubertConfig.ChineseHubertBase;
        Assert.Equal(768, c.Hidden);
        Assert.Equal(12, c.NumLayers);
        Assert.Equal(64, c.HeadDim);
        Assert.Equal(320, c.Downsample);          // 50 Hz at 16 kHz
        Assert.Equal(16_000, c.SampleRate);
    }

    [Fact]
    public void SyntheticForward_ProducesFiniteContentFeatures()
    {
        HubertConfig c = new()
        {
            ConvDim = 16, Hidden = 24, NumLayers = 2, NumHeads = 2, FfnDim = 32,
            PosConvKernel = 8, PosConvGroups = 2,
        };
        using CpuBackend backend = new();
        using Hubert hub = new(c);
        hub.LoadWeights(Weights(c));

        int tPcm = 1400;                          // → ~4 frames after 320× downsample
        using Tensor pcm = F3(1, 1, tPcm);
        using Tensor feats = hub.Forward(backend, pcm, tPcm);
        Assert.Equal(c.Hidden, (int)feats.Shape[1]);
        Assert.True((int)feats.Shape[2] > 0);
        float* p = (float*)feats.DataPointer;
        for (long i = 0; i < feats.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> Weights(HubertConfig c)
    {
        int cd = c.ConvDim, h = c.Hidden;
        Dictionary<string, Tensor> w = new()
        {
            ["feature_extractor.conv_layers.0.layer_norm.weight"] = F1(cd),
            ["feature_extractor.conv_layers.0.layer_norm.bias"] = F1(cd),
            ["feature_projection.layer_norm.weight"] = F1(cd), ["feature_projection.layer_norm.bias"] = F1(cd),
            ["feature_projection.projection.weight"] = F2(h, cd), ["feature_projection.projection.bias"] = F1(h),
            ["encoder.pos_conv_embed.conv.weight"] = F3(h, h / c.PosConvGroups, c.PosConvKernel),
            ["encoder.pos_conv_embed.conv.bias"] = F1(h),
            ["encoder.layer_norm.weight"] = F1(h), ["encoder.layer_norm.bias"] = F1(h),
        };
        for (int i = 0; i < c.ConvKernels.Count; i++)
            w[$"feature_extractor.conv_layers.{i}.conv.weight"] = F3(cd, i == 0 ? 1 : cd, c.ConvKernels[i]);
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"encoder.layers.{i}";
            foreach (string a in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
            { w[$"{p}.attention.{a}.weight"] = F2(h, h); w[$"{p}.attention.{a}.bias"] = F1(h); }
            w[$"{p}.layer_norm.weight"] = F1(h); w[$"{p}.layer_norm.bias"] = F1(h);
            w[$"{p}.feed_forward.intermediate_dense.weight"] = F2(c.FfnDim, h); w[$"{p}.feed_forward.intermediate_dense.bias"] = F1(c.FfnDim);
            w[$"{p}.feed_forward.output_dense.weight"] = F2(h, c.FfnDim); w[$"{p}.feed_forward.output_dense.bias"] = F1(h);
            w[$"{p}.final_layer_norm.weight"] = F1(h); w[$"{p}.final_layer_norm.bias"] = F1(h);
        }
        return w;
    }
}

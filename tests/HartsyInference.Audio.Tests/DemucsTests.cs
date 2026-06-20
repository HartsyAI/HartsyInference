using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for HTDemucs's novel components: the cross-domain transformer (self/cross alternating layers
/// over two token streams) and the HEnc/HDec GLU conv block — each synthetic-forward verified to finite output.</summary>
public sealed unsafe class DemucsTests
{
    private static uint _rng = 0xDE3Cu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor F4(int a, int b, int c, int d) => Fill(new Tensor(new TensorShape(a, b, c, d), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static void AssertFinite(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i])); }

    [Fact]
    public void Config_FourStereoStems()
    {
        HtDemucsConfig c = HtDemucsConfig.Htdemucs;
        Assert.Equal(4, c.NumSources);
        Assert.Equal(new[] { "drums", "bass", "other", "vocals" }, c.Sources.ToArray());
        Assert.Equal(4, c.SpecInChannels);       // 2 (re/im) × 2 stereo
        Assert.Equal(512, c.BottomChannels);
        Assert.Equal(2048, c.TransformerFfn);
    }

    [Fact]
    public void CrossTransformer_SyntheticForward_MixesStreamsFinite()
    {
        HtDemucsConfig c = new() { BottomChannels = 8, THeads = 2, TLayers = 3, HiddenScale = 2.0f };
        using CpuBackend backend = new();
        DemucsCrossTransformer xf = new(c);
        xf.LoadWeights(XfWeights(c));
        int nSpec = 6, nTime = 4;
        using Tensor spec = F3(1, c.BottomChannels, nSpec);
        using Tensor time = F3(1, c.BottomChannels, nTime);
        (Tensor s, Tensor t) = xf.Forward(backend, spec, nSpec, time, nTime);
        Assert.Equal(new TensorShape(1, c.BottomChannels, nSpec), s.Shape);
        Assert.Equal(new TensorShape(1, c.BottomChannels, nTime), t.Shape);
        AssertFinite(s); AssertFinite(t);
        s.Dispose(); t.Dispose();
    }

    [Fact]
    public void ConvBlock_EncodeDecode_Finite()
    {
        using CpuBackend backend = new();
        // 2D encoder: freq /4.
        DemucsConvBlock enc2 = new(4, 8, kernel: 8, stride: 4, is2d: true, decoder: false);
        enc2.LoadWeights(new Dictionary<string, Tensor> { ["e.conv.weight"] = F4(8, 4, 8, 1), ["e.conv.bias"] = F1(8), ["e.rewrite.weight"] = F4(16, 8, 1, 1), ["e.rewrite.bias"] = F1(16) }, "e");
        using Tensor x2 = F4(1, 4, 16, 5);
        using Tensor o2 = enc2.EncodeForward(backend, x2, 16, 5);
        Assert.Equal(8, (int)o2.Shape[1]); AssertFinite(o2);

        // 1D encoder: time /4.
        DemucsConvBlock enc1 = new(4, 8, 8, 4, is2d: false, decoder: false);
        enc1.LoadWeights(new Dictionary<string, Tensor> { ["e.conv.weight"] = F3(8, 4, 8), ["e.conv.bias"] = F1(8), ["e.rewrite.weight"] = F3(16, 8, 1), ["e.rewrite.bias"] = F1(16) }, "e");
        using Tensor x1 = F3(1, 4, 20);
        using Tensor o1 = enc1.EncodeForward(backend, x1, 1, 20);
        Assert.Equal(8, (int)o1.Shape[1]); AssertFinite(o1);

        // 1D decoder: time ×4.
        DemucsConvBlock dec1 = new(8, 8, 8, 4, is2d: false, decoder: true);
        dec1.LoadWeights(new Dictionary<string, Tensor> { ["d.conv.weight"] = F3(8, 8, 8), ["d.conv.bias"] = F1(8), ["d.rewrite.weight"] = F3(16, 8, 1), ["d.rewrite.bias"] = F1(16) }, "d");
        using Tensor d1 = dec1.DecodeForward(backend, o1, 1, (int)o1.Shape[2]);
        Assert.Equal(8, (int)d1.Shape[1]); AssertFinite(d1);
    }

    private static Dictionary<string, Tensor> XfWeights(HtDemucsConfig c)
    {
        int d = c.BottomChannels, ffn = c.TransformerFfn;
        Dictionary<string, Tensor> w = new()
        {
            ["crosstransformer.norm_in.weight"] = Ones(d), ["crosstransformer.norm_in.bias"] = F1(d),
            ["crosstransformer.norm_in_t.weight"] = Ones(d), ["crosstransformer.norm_in_t.bias"] = F1(d),
            ["crosstransformer.weight_pos_embed"] = F1(1),
        };
        for (int i = 0; i < c.TLayers; i++)
        {
            bool cross = i % 2 == 1;
            foreach (string stream in new[] { "layers", "layers_t" })
            {
                string p = $"crosstransformer.{stream}.{i}";
                string a = cross ? "cross_attn" : "self_attn";
                foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
                { w[$"{p}.{a}.{proj}.weight"] = F2(d, d); w[$"{p}.{a}.{proj}.bias"] = F1(d); }
                w[$"{p}.norm1.weight"] = Ones(d); w[$"{p}.norm1.bias"] = F1(d);
                w[$"{p}.norm2.weight"] = Ones(d); w[$"{p}.norm2.bias"] = F1(d);
                w[$"{p}.norm3.weight"] = Ones(d); w[$"{p}.norm3.bias"] = F1(d);
                w[$"{p}.linear1.weight"] = F2(ffn, d); w[$"{p}.linear1.bias"] = F1(ffn);
                w[$"{p}.linear2.weight"] = F2(d, ffn); w[$"{p}.linear2.bias"] = F1(d);
                w[$"{p}.gamma_1.scale"] = F1(d); w[$"{p}.gamma_2.scale"] = F1(d);
            }
        }
        return w;
    }
}

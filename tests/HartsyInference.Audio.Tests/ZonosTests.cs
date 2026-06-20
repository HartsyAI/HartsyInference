using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Streaming;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Zonos: the backbone preset, the k+1 delay pattern, the Fourier
/// conditioner math, the multi-codebook embed/head, and a synthetic-weights backbone forward (LayerNorm
/// blocks reusing the Dia attention/MLP) asserting finite output.</summary>
public sealed unsafe class ZonosTests
{
    private static uint _rng = 0x2468aceu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Zeros(int n) => new(new TensorShape(n), DType.F32);

    [Fact]
    public void V0_1Transformer_BackbonePreset()
    {
        ZonosConfig c = ZonosConfig.V0_1Transformer;
        Assert.Equal(2_048, c.Hidden);
        Assert.Equal(26, c.NumLayers);
        Assert.Equal(16, c.NumHeads);
        Assert.Equal(4, c.NumKvHeads);          // GQA 4:1
        Assert.Equal(128, c.HeadDim);
        Assert.Equal(8_192, c.FfnIntermediate);
        Assert.Equal(9, c.Channels);
        Assert.Equal(1_026, c.InputVocab);      // 1024 + EOS + masked
        Assert.Equal(1_025, c.OutputVocab);
        Assert.Equal(1_024, c.EosToken);
        Assert.Equal(1_025, c.MaskedToken);
        Assert.Equal(2.0f, c.CfgScale);
    }

    [Fact]
    public void Delays_AreKPlusOne()
    {
        int[] d = ZonosConfig.V0_1Transformer.BuildDelays();
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, d);
    }

    [Fact]
    public void Fourier_ProducesCosSinHalves()
    {
        // weight all-zero → f=0 → cos=1, sin=0 for every pair, regardless of input.
        ZonosFourierConditioner fc = new(inputDim: 1, outputDim: 8, min: 0, max: 40);
        Dictionary<string, Tensor> w = new() { ["c.weight"] = Zeros(4 * 1), ["c.uncond_vector"] = F1(8) };
        fc.LoadWeights(w, "c");
        float[] outv = fc.Forward([15f]);
        Assert.Equal(8, outv.Length);
        for (int i = 0; i < 4; i++) { Assert.Equal(1f, outv[i], 5); Assert.Equal(0f, outv[4 + i], 5); }
    }

    [Fact]
    public void Codebooks_EmbedSumsNine_HeadsStackNine()
    {
        ZonosConfig c = ZonosConfig.V0_1Transformer with { Hidden = 16 };
        ZonosCodebooks cb = new(c);
        Dictionary<string, Tensor> w = new();
        for (int k = 0; k < 9; k++) { w[$"embeddings.{k}.weight"] = F(c.InputVocab, 16); w[$"heads.{k}.weight"] = F(c.OutputVocab, 16); }
        cb.LoadWeights(w);
        using CpuBackend backend = new();
        Span<int> codes = stackalloc int[9];
        for (int k = 0; k < 9; k++) codes[k] = k + 1;
        using Tensor emb = cb.EmbedFrame(codes);
        Assert.Equal(new TensorShape(1, 1, 16), emb.Shape);
        float[][] logits = cb.Heads(backend, emb);
        Assert.Equal(9, logits.Length);
        Assert.Equal(c.OutputVocab, logits[0].Length);
    }

    [Fact]
    public void Backbone_SyntheticForward_IsFinite()
    {
        ZonosConfig c = ZonosConfig.V0_1Transformer with
        {
            Hidden = 32, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 8, FfnIntermediate = 64, MaxPositions = 64,
        };
        using CpuBackend backend = new();
        using ZonosBackbone bb = new(c);
        bb.LoadWeights(BackboneWeights(c));
        int t = 5;
        using StreamingKvCache cache = new(c.NumLayers, 1, c.NumKvHeads, 16, c.HeadDim);
        using Tensor embeds = F(t, c.Hidden).Reshape(new TensorShape(1, t, c.Hidden));
        using Tensor outT = bb.Forward(backend, embeds, t, 0, cache);
        Assert.Equal(new TensorShape(1, t, c.Hidden), outT.Shape);
        float* p = (float*)outT.DataPointer;
        for (long i = 0; i < outT.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> BackboneWeights(ZonosConfig c)
    {
        int h = c.Hidden, qkv = (c.NumHeads + 2 * c.NumKvHeads) * c.HeadDim, q = c.NumHeads * c.HeadDim;
        Dictionary<string, Tensor> w = new()
        {
            ["backbone.norm_f.weight"] = Ones(h),
            ["backbone.norm_f.bias"] = Zeros(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"backbone.layers.{i}";
            w[$"{p}.norm.weight"] = Ones(h);
            w[$"{p}.norm.bias"] = Zeros(h);
            w[$"{p}.norm2.weight"] = Ones(h);
            w[$"{p}.norm2.bias"] = Zeros(h);
            w[$"{p}.mixer.in_proj.weight"] = F(qkv, h);
            w[$"{p}.mixer.out_proj.weight"] = F(h, q);
            w[$"{p}.mlp.fc1.weight"] = F(2 * c.FfnIntermediate, h);
            w[$"{p}.mlp.fc2.weight"] = F(h, c.FfnIntermediate);
        }
        return w;
    }
}

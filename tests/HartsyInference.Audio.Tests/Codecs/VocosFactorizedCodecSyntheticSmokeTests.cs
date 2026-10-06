using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Codecs;

/// <summary>Tiny random weights through <see cref="VocosFactorizedCodec"/> (both the no-resample "RepCodec"
/// shape and the 2x-resample "EnhancedCodec" shape): shapes, finiteness, code range and determinism only. Says
/// nothing about parity with the real <c>amphion/MaskGCT</c> or IndexTTS-2.5 <c>codec.pth</c> checkpoints.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class VocosFactorizedCodecSyntheticSmokeTests : IDisposable
{
    private const int Hidden = 16;
    private const int CodebookSize = 32;
    private const int CodebookDim = 4;
    private const int VocosDim = 8;
    private const int VocosIntermediate = 16;
    private const int VocosLayers = 2;

    private readonly List<Tensor> _owned = [];
    private readonly CpuBackend _backend = new();

    private Tensor Rand(Random rng, double amp, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * amp);
        _owned.Add(t);
        return t;
    }

    private Tensor NormWeight(Random rng, long dim)
    {
        Tensor t = Rand(rng, 0.1, dim);
        foreach (ref float v in t.AsSpan<float>()) v += 1f;
        return t;
    }

    private void AddVocosBackboneWeights(Dictionary<string, Tensor> w, Random rng, string prefix)
    {
        w[$"{prefix}.embed.weight"] = Rand(rng, 0.1, VocosDim, Hidden, 7);
        w[$"{prefix}.embed.bias"] = Rand(rng, 0.02, VocosDim);
        w[$"{prefix}.norm.weight"] = NormWeight(rng, VocosDim);
        w[$"{prefix}.norm.bias"] = Rand(rng, 0.02, VocosDim);
        for (int i = 0; i < VocosLayers; i++)
        {
            string p = $"{prefix}.convnext.{i}";
            w[$"{p}.dwconv.weight"] = Rand(rng, 0.1, VocosDim, 1, 7);
            w[$"{p}.dwconv.bias"] = Rand(rng, 0.02, VocosDim);
            w[$"{p}.norm.weight"] = NormWeight(rng, VocosDim);
            w[$"{p}.norm.bias"] = Rand(rng, 0.02, VocosDim);
            w[$"{p}.pwconv1.weight"] = Rand(rng, 0.1, VocosIntermediate, VocosDim);
            w[$"{p}.pwconv1.bias"] = Rand(rng, 0.02, VocosIntermediate);
            w[$"{p}.pwconv2.weight"] = Rand(rng, 0.1, VocosDim, VocosIntermediate);
            w[$"{p}.pwconv2.bias"] = Rand(rng, 0.02, VocosDim);
            w[$"{p}.gamma"] = Rand(rng, 0.1, VocosDim);
        }
        w[$"{prefix}.final_layer_norm.weight"] = NormWeight(rng, VocosDim);
        w[$"{prefix}.final_layer_norm.bias"] = Rand(rng, 0.02, VocosDim);
    }

    private Dictionary<string, Tensor> BuildWeights(Random rng, int downsampleScale, bool withDecoder)
    {
        Dictionary<string, Tensor> w = [];
        AddVocosBackboneWeights(w, rng, "encoder.0");
        w["encoder.1.weight"] = Rand(rng, 0.1, Hidden, VocosDim);
        w["encoder.1.bias"] = Rand(rng, 0.02, Hidden);

        w["quantizer.quantizers.0.in_project.weight_g"] = Rand(rng, 0.5, CodebookDim, 1, 1);
        w["quantizer.quantizers.0.in_project.weight_v"] = Rand(rng, 0.1, CodebookDim, Hidden, 1);
        w["quantizer.quantizers.0.in_project.bias"] = Rand(rng, 0.02, CodebookDim);
        w["quantizer.quantizers.0.out_project.weight_g"] = Rand(rng, 0.5, Hidden, 1, 1);
        w["quantizer.quantizers.0.out_project.weight_v"] = Rand(rng, 0.1, Hidden, CodebookDim, 1);
        w["quantizer.quantizers.0.out_project.bias"] = Rand(rng, 0.02, Hidden);
        w["quantizer.quantizers.0.codebook.weight"] = Rand(rng, 1.0, CodebookSize, CodebookDim);

        if (downsampleScale > 1)
        {
            w["down.weight"] = Rand(rng, 0.1, Hidden, Hidden, 3);
            w["down.bias"] = Rand(rng, 0.02, Hidden);
        }

        if (withDecoder)
        {
            AddVocosBackboneWeights(w, rng, "decoder.0");
            w["decoder.1.weight"] = Rand(rng, 0.1, Hidden, VocosDim);
            w["decoder.1.bias"] = Rand(rng, 0.02, Hidden);
            if (downsampleScale > 1)
            {
                w["up.weight"] = Rand(rng, 0.1, Hidden, Hidden, 3);
                w["up.bias"] = Rand(rng, 0.02, Hidden);
            }
        }
        return w;
    }

    private static VocosFactorizedCodecConfig Cfg(int downsampleScale) => new()
    {
        CodebookSize = CodebookSize,
        HiddenSize = Hidden,
        CodebookDim = CodebookDim,
        VocosDim = VocosDim,
        VocosIntermediateDim = VocosIntermediate,
        VocosNumLayers = VocosLayers,
        DownsampleScale = downsampleScale,
    };

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Quantize_ProducesValidCodesAndFiniteContinuous(int downsampleScale)
    {
        Random rng = new(1);
        Dictionary<string, Tensor> weights = BuildWeights(rng, downsampleScale, withDecoder: false);
        using VocosFactorizedCodec codec = new(Cfg(downsampleScale));
        codec.LoadWeights(weights, loadDecoder: false);

        const int t = 12;
        Tensor input = Rand(rng, 1.0, 1, t, Hidden);
        (int[] codes, Tensor continuous) = codec.Quantize(_backend, input, t);
        try
        {
            int expectedT = downsampleScale > 1 ? (t + 1) / 2 : t;
            Assert.Equal(expectedT, codes.Length);
            foreach (int c in codes) Assert.InRange(c, 0, CodebookSize - 1);

            Assert.Equal(expectedT, (int)continuous.Shape[1]);
            Assert.Equal(Hidden, (int)continuous.Shape[2]);
            foreach (float v in continuous.AsSpan<float>()) Assert.True(float.IsFinite(v));
        }
        finally { continuous.Dispose(); }
    }

    [Fact]
    public void Decode_WithoutLoadedDecoder_Throws()
    {
        Random rng = new(2);
        Dictionary<string, Tensor> weights = BuildWeights(rng, downsampleScale: 1, withDecoder: false);
        using VocosFactorizedCodec codec = new(Cfg(1));
        codec.LoadWeights(weights, loadDecoder: false);

        Assert.Throws<InvalidOperationException>(() => codec.Decode(_backend, [0, 1, 2]));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Decode_ProducesFiniteOutput_OfExpectedLength(int downsampleScale)
    {
        Random rng = new(3);
        Dictionary<string, Tensor> weights = BuildWeights(rng, downsampleScale, withDecoder: true);
        using VocosFactorizedCodec codec = new(Cfg(downsampleScale));
        codec.LoadWeights(weights, loadDecoder: true);

        int[] codes = [1, 2, 3, 4, 5];
        using Tensor output = codec.Decode(_backend, codes);

        int expectedT = downsampleScale > 1 ? codes.Length * 2 : codes.Length;
        Assert.Equal(expectedT, (int)output.Shape[1]);
        Assert.Equal(Hidden, (int)output.Shape[2]);
        foreach (float v in output.AsSpan<float>()) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void VqToEmbedding_MatchesQuantizesContinuousOutput_WithoutTheDecoderLoaded()
    {
        // vq2emb is exactly the out_project(codebook[codes]) half of Quantize — so for the codes Quantize itself
        // returned, the two must agree, and it must work with loadDecoder: false (IndexTTS-2.0's configuration).
        Random rng = new(5);
        Dictionary<string, Tensor> weights = BuildWeights(rng, downsampleScale: 1, withDecoder: false);
        using VocosFactorizedCodec codec = new(Cfg(1));
        codec.LoadWeights(weights, loadDecoder: false);

        const int t = 9;
        Tensor input = Rand(rng, 1.0, 1, t, Hidden);
        (int[] codes, Tensor continuous) = codec.Quantize(_backend, input, t);
        using Tensor embedding = codec.VqToEmbedding(_backend, codes);
        try
        {
            Assert.Equal(new TensorShape(1, t, Hidden), embedding.Shape);
            float[] a = embedding.AsSpan<float>().ToArray(), b = continuous.AsSpan<float>().ToArray();
            Assert.Equal(b.Length, a.Length);
            for (int i = 0; i < a.Length; i++) Assert.Equal(b[i], a[i], 5);
        }
        finally { continuous.Dispose(); }
    }

    [Fact]
    public void Quantize_IsDeterministic_AcrossRepeatedCalls()
    {
        Random rng = new(4);
        Dictionary<string, Tensor> weights = BuildWeights(rng, downsampleScale: 2, withDecoder: false);
        using VocosFactorizedCodec codec = new(Cfg(2));
        codec.LoadWeights(weights, loadDecoder: false);

        const int t = 10;
        Tensor input = Rand(rng, 1.0, 1, t, Hidden);
        (int[] codesA, Tensor contA) = codec.Quantize(_backend, input, t);
        (int[] codesB, Tensor contB) = codec.Quantize(_backend, input, t);
        try
        {
            Assert.Equal(codesA, codesB);
            float[] a = contA.AsSpan<float>().ToArray(), b = contB.AsSpan<float>().ToArray();
            for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 6);
        }
        finally { contA.Dispose(); contB.Dispose(); }
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}

/// <summary>Loads the real <c>amphion/MaskGCT</c> <c>semantic_codec/model.safetensors</c> (IndexTTS-2.0's
/// dependency; too large to bundle — point <c>MASKGCT_SEMANTIC_CODEC_SAFETENSORS_PATH</c> at a local copy). A
/// clean <c>LoadWeights</c> confirms every key name matches the real checkpoint; shape/finite are checked past
/// that. <c>loadDecoder: false</c> matches the real <c>infer_v2.py</c>, which never calls <c>.decode()</c> on
/// this codec.</summary>
[Trait("Category", "Integration")]
public sealed class VocosFactorizedCodecRealWeightTests
{
    [Fact]
    public void Quantize_SucceedsAgainstRealMaskGctCheckpoint()
    {
        string? path = Environment.GetEnvironmentVariable("MASKGCT_SEMANTIC_CODEC_SAFETENSORS_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        try
        {
            using VocosFactorizedCodec codec = new(VocosFactorizedCodecConfig.IndexTts2V0);
            codec.LoadWeights(weights, loadDecoder: false);

            const int t = 20;
            Random rng = new(5);
            Tensor input = new(new TensorShape(1, t, VocosFactorizedCodecConfig.IndexTts2V0.HiddenSize), DType.F32);
            foreach (ref float v in input.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

            using CpuBackend backend = new();
            (int[] codes, Tensor continuous) = codec.Quantize(backend, input, t);
            try
            {
                Assert.Equal(t, codes.Length);
                foreach (int c in codes) Assert.InRange(c, 0, VocosFactorizedCodecConfig.IndexTts2V0.CodebookSize - 1);
                foreach (float v in continuous.AsSpan<float>()) Assert.True(float.IsFinite(v));
            }
            finally { input.Dispose(); continuous.Dispose(); }
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}

/// <summary>Loads the real IndexTTS-2.5 <c>codec.pth</c> (a training checkpoint — <c>model.*</c> weights plus an
/// <c>optimizer.state.*</c> section this class ignores; too large to bundle — point
/// <c>INDEXTTS2_CODEC_PTH_PATH</c> at a local copy). Unlike IndexTTS-2.0's dependency, this checkpoint's decoder
/// IS live (the real <c>infer_v2_5.py</c> calls <c>.decode()</c> on AR-generated codes), so this test exercises
/// both halves plus the 2x down/up resample.</summary>
[Trait("Category", "Integration")]
public sealed class VocosFactorizedCodecIndexTts25RealWeightTests
{
    [Fact]
    public void Quantize_AndDecode_SucceedAgainstRealCodecPth()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS2_CODEC_PTH_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        try
        {
            using VocosFactorizedCodec codec = new(VocosFactorizedCodecConfig.IndexTts2V5);
            codec.LoadWeights(weights, prefix: "model", loadDecoder: true);

            const int t = 20;
            Random rng = new(6);
            Tensor input = new(new TensorShape(1, t, VocosFactorizedCodecConfig.IndexTts2V5.HiddenSize), DType.F32);
            foreach (ref float v in input.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

            using CpuBackend backend = new();
            (int[] codes, Tensor continuous) = codec.Quantize(backend, input, t);
            try
            {
                Assert.Equal((t + 1) / 2, codes.Length);
                foreach (int c in codes) Assert.InRange(c, 0, VocosFactorizedCodecConfig.IndexTts2V5.CodebookSize - 1);
                foreach (float v in continuous.AsSpan<float>()) Assert.True(float.IsFinite(v));
            }
            finally { input.Dispose(); continuous.Dispose(); }

            using Tensor decoded = codec.Decode(backend, codes);
            Assert.Equal(codes.Length * 2, (int)decoded.Shape[1]);
            foreach (float v in decoded.AsSpan<float>()) Assert.True(float.IsFinite(v));
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}

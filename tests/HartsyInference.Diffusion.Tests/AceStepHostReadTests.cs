using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>ACE-Step v1 reads some checkpoint tensors on the host through <c>float*</c> (norm biases, the DCAE's depthwise
/// and grouped convs). The Comfy all-in-one checkpoint ships them as bf16, which GPU backends keep, so each model must give
/// the same output from bf16 copies of those tensors as from the same values in F32.</summary>
public sealed class AceStepHostReadTests
{
    [Fact]
    public void DcaeDecode_Bf16HostReadTensors_MatchTheSameValuesInF32()
    {
        Dictionary<string, Tensor> weights = AceStepSyntheticWeights.BuildDcaeDecoder();
        // The tiny build has no multiscale branch or block norm biases; the real decoder has both.
        const string Attention = "decoder.up_blocks.3.0";
        weights[$"{Attention}.attn.to_qkv_multiscale.0.proj_in.weight"] = Values([96, 1, 5, 5], 1);
        weights[$"{Attention}.attn.to_qkv_multiscale.0.proj_in.bias"] = Values([96], 2);
        weights[$"{Attention}.attn.to_qkv_multiscale.0.proj_out.weight"] = Values([96, 32, 1, 1], 3);
        weights[$"{Attention}.attn.to_qkv_multiscale.0.proj_out.bias"] = Values([96], 4);
        weights[$"{Attention}.attn.to_out.weight"] = Values([32, 64], 5);
        weights[$"{Attention}.conv_out.norm.bias"] = Values([32], 6);
        int[] dims = [4, 8, 16];
        for (int i = 0; i < dims.Length; i++)
        {
            weights[$"decoder.up_blocks.{i}.1.norm.bias"] = Values([dims[i]], 7 + i);
        }
        (Dictionary<string, Tensor> half, Dictionary<string, Tensor> full) = SplitHostRead(weights,
            key => key.EndsWith("norm.bias", StringComparison.Ordinal) || key.EndsWith("norm_out.bias", StringComparison.Ordinal)
                || key.Contains(".conv_depth.", StringComparison.Ordinal) || key.Contains(".to_qkv_multiscale.", StringComparison.Ordinal));
        MusicDcaeDecoder reference = NewDcae();
        reference.LoadWeights(full);
        MusicDcaeDecoder dcae = NewDcae();
        dcae.LoadWeights(half);
        using CpuBackend backend = new CpuBackend();
        using Tensor latent = Values([1, 8, 4, 6], 20);

        using Tensor expected = reference.Decode(backend, latent);
        using Tensor actual = dcae.Decode(backend, latent);

        Assert.Equal(expected.AsSpan<float>().ToArray(), actual.AsSpan<float>().ToArray());
    }

    [Fact]
    public void VocoderDecode_Bf16NormBiases_MatchTheSameValuesInF32()
    {
        Dictionary<string, Tensor> weights = AceStepSyntheticWeights.BuildVocoder(melBins: 32);
        (Dictionary<string, Tensor> half, Dictionary<string, Tensor> full) = SplitHostRead(weights,
            key => key is "backbone.channel_layers.0.1.bias" or "backbone.norm.bias"
                || (key.StartsWith("backbone.channel_layers.", StringComparison.Ordinal) && key.EndsWith(".0.bias", StringComparison.Ordinal)
                    && !key.StartsWith("backbone.channel_layers.0.", StringComparison.Ordinal))
                || (key.StartsWith("backbone.stages.", StringComparison.Ordinal) && key.EndsWith(".norm.bias", StringComparison.Ordinal)));
        AdaMosHiFiGanV1 reference = NewVocoder();
        reference.LoadWeights(full);
        AdaMosHiFiGanV1 vocoder = NewVocoder();
        vocoder.LoadWeights(half);
        using CpuBackend backend = new CpuBackend();
        using Tensor mel = Values([1, 32, 10], 21);

        using Tensor expected = reference.Decode(backend, mel);
        using Tensor actual = vocoder.Decode(backend, mel);

        Assert.Equal(expected.AsSpan<float>().ToArray(), actual.AsSpan<float>().ToArray());
    }

    [Fact]
    public void LyricEncoder_Bf16NormBiases_MatchTheSameValuesInF32()
    {
        AceStepConfig config = AceStepSyntheticWeights.TinyConfig;
        (Dictionary<string, Tensor> half, Dictionary<string, Tensor> full) = SplitHostRead(AceStepSyntheticWeights.BuildDit(config),
            key => key.StartsWith("lyric_encoder.", StringComparison.Ordinal) && key.EndsWith(".bias", StringComparison.Ordinal)
                && (key.Contains("norm", StringComparison.Ordinal) || key.Contains("embed.out.1", StringComparison.Ordinal)));
        using AceStepDit reference = new AceStepDit(config);
        reference.LoadWeights(full);
        using AceStepDit dit = new AceStepDit(config);
        dit.LoadWeights(half);
        using CpuBackend backend = new CpuBackend();
        using Tensor text = new Tensor(new TensorShape(2, config.TextDim), DType.F32);

        using Tensor expected = reference.BuildContext(backend, text, [1, 2, 3], null);
        using Tensor actual = dit.BuildContext(backend, text, [1, 2, 3], null);

        Assert.Equal(expected.AsSpan<float>().ToArray(), actual.AsSpan<float>().ToArray());
    }

    private static MusicDcaeDecoder NewDcae() =>
        new MusicDcaeDecoder(latentChannels: 8, outChannels: 2, blockOutChannels: [4, 8, 16, 32], layersPerBlock: [1, 1, 1, 1]);

    private static AdaMosHiFiGanV1 NewVocoder() =>
        new AdaMosHiFiGanV1(depths: [1, 1, 1, 1], dims: [4, 6, 8, 8], upsampleRates: [2, 2], upsampleKernels: [4, 4],
            resblockKernels: [3]);

    /// <summary>The selected tensors as bf16 in one dictionary and as the same rounded values in F32 in the other; every
    /// other tensor is shared.</summary>
    private static (Dictionary<string, Tensor> Half, Dictionary<string, Tensor> Full) SplitHostRead(
        Dictionary<string, Tensor> weights, Func<string, bool> hostRead)
    {
        Dictionary<string, Tensor> half = new Dictionary<string, Tensor>(weights);
        Dictionary<string, Tensor> full = new Dictionary<string, Tensor>(weights);
        int split = 0;
        foreach ((string key, Tensor value) in weights)
        {
            if (!hostRead(key))
            {
                continue;
            }
            Tensor bf16 = value.CastTo(DType.BF16);
            half[key] = bf16;
            full[key] = bf16.CastTo(DType.F32);
            split++;
        }
        Assert.True(split > 0, "no host-read tensor selected");
        return (half, full);
    }

    private static Tensor Values(int[] dims, int seed)
    {
        Tensor t = new Tensor(new TensorShape(Array.ConvertAll(dims, d => (long)d)), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = 0.05f * MathF.Sin(seed * 12.9898f + i * 0.731f);
        }
        return t;
    }
}

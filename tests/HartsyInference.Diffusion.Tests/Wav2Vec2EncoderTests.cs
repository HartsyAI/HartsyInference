using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config smoke test for the Wav2Vec2 audio feature extractor (S2V's Phase S0 front-end): a strided
/// Conv1d feature encoder → projection → grouped positional conv → transformer, harvesting all hidden states
/// <c>[T, numStates, hidden]</c>. Numerics validation-pending (structural).</summary>
public unsafe class Wav2Vec2EncoderTests
{
    [Fact]
    public void EncodeAllLayers_ProducesStackedHiddenStates()
    {
        CpuBackend backend = new();
        Wav2Vec2EncoderConfig cfg = new()
        {
            ConvDims = [8, 8, 8], ConvStrides = [5, 2, 2], ConvKernels = [10, 3, 3], ConvBias = false,
            GroupNormFirstConvOnly = true, Hidden = 16, NumLayers = 2, NumHeads = 2, Intermediate = 32,
            PosConvKernel = 8, PosConvGroups = 2,
        };
        Wav2Vec2Encoder enc = new(cfg);
        enc.LoadWeights(BuildWeights(cfg));

        float[] waveform = new float[400];
        Random rng = new(31);
        for (int i = 0; i < waveform.Length; i++) waveform[i] = (float)(rng.NextDouble() * 2 - 1);

        Tensor states = enc.EncodeAllLayers(backend, waveform);

        Assert.Equal(3, states.Shape.Rank);
        Assert.Equal(cfg.NumHiddenStates, (int)states.Shape[1]);   // 2 layers + 1 = 3
        Assert.Equal(cfg.Hidden, (int)states.Shape[2]);
        Assert.True((int)states.Shape[0] > 0);
        float* p = (float*)states.DataPointer;
        for (long i = 0; i < states.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static Dictionary<string, Tensor> BuildWeights(Wav2Vec2EncoderConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int prevC = 1;
        for (int i = 0; i < c.ConvDims.Length; i++)
        {
            w[$"feature_extractor.conv_layers.{i}.conv.weight"] = R([c.ConvDims[i], prevC, c.ConvKernels[i]]);
            prevC = c.ConvDims[i];
        }
        w["feature_extractor.conv_layers.0.layer_norm.weight"] = R([c.ConvDims[0]]);
        w["feature_extractor.conv_layers.0.layer_norm.bias"] = R([c.ConvDims[0]]);
        w["feature_projection.layer_norm.weight"] = R([c.ConvDims[^1]]); w["feature_projection.layer_norm.bias"] = R([c.ConvDims[^1]]);
        w["feature_projection.projection.weight"] = R([c.Hidden, c.ConvDims[^1]]); w["feature_projection.projection.bias"] = R([c.Hidden]);
        w["encoder.pos_conv_embed.conv.weight"] = R([c.Hidden, c.Hidden / c.PosConvGroups, c.PosConvKernel]);
        w["encoder.pos_conv_embed.conv.bias"] = R([c.Hidden]);
        w["encoder.layer_norm.weight"] = R([c.Hidden]); w["encoder.layer_norm.bias"] = R([c.Hidden]);
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"encoder.layers.{i}";
            foreach (string a in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
            { w[$"{p}.attention.{a}.weight"] = R([c.Hidden, c.Hidden]); w[$"{p}.attention.{a}.bias"] = R([c.Hidden]); }
            w[$"{p}.layer_norm.weight"] = R([c.Hidden]); w[$"{p}.layer_norm.bias"] = R([c.Hidden]);
            w[$"{p}.feed_forward.intermediate_dense.weight"] = R([c.Intermediate, c.Hidden]); w[$"{p}.feed_forward.intermediate_dense.bias"] = R([c.Intermediate]);
            w[$"{p}.feed_forward.output_dense.weight"] = R([c.Hidden, c.Intermediate]); w[$"{p}.feed_forward.output_dense.bias"] = R([c.Hidden]);
            w[$"{p}.final_layer_norm.weight"] = R([c.Hidden]); w[$"{p}.final_layer_norm.bias"] = R([c.Hidden]);
        }
        return w;
    }

    private static int _seed = 1000;
    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}

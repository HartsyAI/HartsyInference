using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural test for <see cref="LtxVideo2VaeDecoder"/>'s data-driven loading on the LTX-2.3 <b>22B</b> VAE
/// topology, which differs from the 19B: 4 up-stages (not 3), a 2-resnet mid block, per-stage resnet counts
/// [2,4,6,4], irregular channels, and MIXED upsampler strides (spatio-temporal / temporal / spatial). The decoder
/// must infer all of this from the weight shapes alone. Uses a channel-scaled-down analog that preserves the real
/// stride-product / upscale ratios (8,8,2,4 and 2,1,2,2). Guards the real-checkpoint
/// <c>KeyNotFoundException: decoder.mid_block.resnets.2</c> reported on the 22B load. Numerics validation-pending.</summary>
public unsafe class LtxVideo2VaeDecoderTests
{
    private static int _seed = 31;

    [Fact]
    public void Decode_22bTopology_InfersStructureAndProducesFiniteRgb()
    {
        CpuBackend backend = new();
        LtxVideo2VaeDecoder decoder = new(latentChannels: 128, patchSize: 2, isCausal: false);
        decoder.LoadWeights(Build22bAnalog());

        // 4 up-stages; temporal upscale on stages 0,1 (spatio-temporal) and 2 (temporal) → 3 temporal doublings.
        int tLat = 2;
        Assert.Equal(((tLat * 2 - 1) * 2 - 1) * 2 - 1, decoder.OutputFrames(tLat));   // 2→3→5→9

        Tensor latent = Rand([1, 128, tLat, 2, 2]);
        Tensor rgb = decoder.Decode(backend, latent);

        Assert.Equal(3, (int)rgb.Shape[1]);
        Assert.Equal(decoder.OutputFrames(tLat), (int)rgb.Shape[2]);
        float* p = (float*)rgb.DataPointer;
        for (long i = 0; i < rgb.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    /// <summary>22B decoder weights (diffusers grouping, channels scaled ÷16 from the real model). Per stage the
    /// upsampler conv shape encodes (strideProduct, upscale): stage0/1 = (8,2)/(8,1) spatio-temporal, stage2 = (2,2)
    /// temporal, stage3 = (4,2) spatial — exactly the ratios the loader inverts.</summary>
    private static Dictionary<string, Tensor> Build22bAnalog()
    {
        Dictionary<string, Tensor> w = new();
        int mid = 64;
        AddConv(w, "decoder.conv_in", mid, 128);
        for (int j = 0; j < 2; j++) AddResnet(w, $"decoder.mid_block.resnets.{j}", mid);

        // (convOut, convIn, outNom, nRes) per up-stage. convIn = incoming channels; convOut = outNom·strideProd.
        (int convOut, int convIn, int outNom, int nRes)[] stages =
        [
            (256, 64, 32, 2),   // stage0: strideProd 8 (2,2,2), upscale 2
            (256, 32, 32, 4),   // stage1: strideProd 8 (2,2,2), upscale 1
            (32, 32, 16, 6),    // stage2: strideProd 2 (2,1,1), upscale 2
            (32, 16, 8, 4),     // stage3: strideProd 4 (1,2,2), upscale 2
        ];
        for (int i = 0; i < stages.Length; i++)
        {
            (int convOut, int convIn, int outNom, int nRes) = stages[i];
            AddConv(w, $"decoder.up_blocks.{i}.upsamplers.0.conv", convOut, convIn);
            for (int j = 0; j < nRes; j++) AddResnet(w, $"decoder.up_blocks.{i}.resnets.{j}", outNom);
        }
        AddConv(w, "decoder.conv_out", 3 * 2 * 2, 8);   // outChannels·patch² = 12, in = last stage outNom (8)
        return w;
    }

    private static void AddResnet(Dictionary<string, Tensor> w, string p, int ch)
    {
        AddConv(w, $"{p}.conv1", ch, ch);
        AddConv(w, $"{p}.conv2", ch, ch);
    }

    private static void AddConv(Dictionary<string, Tensor> w, string p, int outC, int inC)
    {
        w[$"{p}.conv.weight"] = Rand([outC, inC, 3, 3, 3]);
        w[$"{p}.conv.bias"] = Rand([outC]);
    }

    private static Tensor Rand(long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}

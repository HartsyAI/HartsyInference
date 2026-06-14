using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Video;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end structural test for the LTX-Video T2V pipeline: tiny-config <see cref="LtxVideoTransformer"/> + base <see cref="LtxVideoVaeDecoder"/> driven through <see cref="LtxVideoPipeline"/> on CPU. Proves the flow-match denoise + pack/unpack + VAE decode + frame output compose and yield the right frame count/shape. Numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class LtxVideoPipelineTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideoPipelineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GenerateFromEmbeddings_TinyConfig_ProducesExpandedFrames()
    {
        CpuBackend backend = new();
        LtxVideoConfig cfg = new()
        {
            InChannels = 8, OutChannels = 8, NumHeads = 2, HeadDim = 4,
            CrossAttentionDim = 8, NumLayers = 1, CaptionChannels = 16,
            VaeSpatialCompression = 16, VaeTemporalCompression = 8,
            NumInferenceSteps = 2, GuidanceScale = 3,
        };
        LtxVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(LtxSyntheticWeights.BuildTransformer(cfg));

        int[] blockOut = [8, 16, 16, 16];
        bool[] scaling = [true, true, true, false];
        int[] layers = [1, 1, 1, 1, 1];
        LtxVideoVaeDecoder vae = new(8, 3, blockOut, scaling, layers, patchSize: 2, isCausal: false);
        vae.LoadWeights(LtxSyntheticWeights.BuildVaeDecoder(8, 3, blockOut, scaling, layers, 2));

        LtxVideoPipeline pipeline = new(backend, transformer, vae, cfg);

        Tensor promptEmbeds = RandRows(3, cfg.CaptionChannels, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.CaptionChannels, seed: 2);
        int numFrames = 9;   // (9-1)/8 + 1 = 2 latent frames
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 3, Seed = 42 };

        (byte[][] frames, int w, int h, int seed) = pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames, frameRate: 25,
            p => _output.WriteLine($"step {p.Step}/{p.TotalSteps}"));

        Assert.Equal(numFrames, frames.Length);   // (T_lat-1)*8+1 = 9
        Assert.Equal(32, w);
        Assert.Equal(32, h);
        foreach (byte[] f in frames) Assert.Equal(32 * 32 * 3, f.Length);
        _ = seed;
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}

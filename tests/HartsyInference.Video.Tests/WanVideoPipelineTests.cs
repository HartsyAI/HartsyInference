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

/// <summary>End-to-end structural test for the Wan2.2 TI2V-5B T2V pipeline: tiny-config <see cref="WanVideoTransformer"/> + the (reused) <see cref="Wan22VaeDecoder"/> driven through <see cref="WanVideoPipeline"/> on CPU. The DiT denoises in latent space; the VAE is the same Wan2.2 decoder built for Lance. Numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class WanVideoPipelineTests
{
    private readonly ITestOutputHelper _output;
    public WanVideoPipelineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GenerateFromEmbeddings_TinyConfig_ProducesExpandedFrames()
    {
        CpuBackend backend = new();
        // Latent channels stay 48 (the Wan2.2 VAE denorm is fixed to z=48); the DiT is tiny otherwise.
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5,
        };
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));

        // Reuse the Wan2.2 VAE decoder (z=48), same builder as Lance.
        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));

        WanVideoPipeline pipeline = new(backend, transformer, vae, cfg);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        int numFrames = 5;   // (5-1)/4 + 1 = 2 latent frames
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, int seed) = pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames,
            p => _output.WriteLine($"step {p.Step}/{p.TotalSteps}"));

        Assert.Equal(numFrames, frames.Length);   // (T_lat-1)*4+1 = 5
        Assert.Equal(32, w);
        Assert.Equal(32, h);
        foreach (byte[] f in frames) Assert.Equal(32 * 32 * 3, f.Length);
        _ = seed;
    }

    [Fact]
    public void GenerateFromEmbeddings_I2V_FirstFrameLatent_ProducesFrames()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5,
        };
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));

        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));

        WanVideoPipeline pipeline = new(backend, transformer, vae, cfg);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        Tensor condition = Rand5d(1, 48, 1, 2, 2, seed: 9);   // normalized first-frame latent for 32x32
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames: 5,
            onProgress: null, firstFrameLatent: condition);

        Assert.Equal(5, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);
        foreach (byte[] f in frames) Assert.Equal(32 * 32 * 3, f.Length);

        // Wrong spatial shape must fail fast.
        Tensor bad = Rand5d(1, 48, 1, 4, 4, seed: 10);
        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames: 5, onProgress: null, firstFrameLatent: bad));
    }

    [Fact]
    public void GenerateFromEmbeddings_I2V_FromRgbBytes_EncodesAndProducesFrames()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 1, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5,
        };
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));

        int[] dimMult = [1, 2, 4, 4];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [false, true, true]);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, [false, true, true]));
        Wan22VaeEncoder encoder = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalDownsample: [true, true, false]);
        encoder.LoadWeights(LanceSyntheticWeights.BuildVaeEncoder(8, 48, dimMult, 2, [true, true, false]));

        WanVideoPipeline pipeline = new(backend, transformer, vae, cfg, encoder);

        byte[] firstFrame = new byte[32 * 32 * 3];
        new Random(5).NextBytes(firstFrame);
        Tensor condition = pipeline.EncodeFirstFrame(firstFrame, width: 32, height: 32);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames: 5,
            onProgress: null, firstFrameLatent: condition);
        condition.Dispose();

        Assert.Equal(5, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);

        // Without an encoder, the RGB entry fails fast.
        WanVideoPipeline noEncoder = new(backend, transformer, vae, cfg);
        Assert.Throws<InvalidOperationException>(() => noEncoder.EncodeFirstFrame(firstFrame, 32, 32));
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Rand5d(int b, int c, int t, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}

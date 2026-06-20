using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end structural test for the Wan2.2-S2V single-chunk pipeline: tiny-config
/// <see cref="WanS2VTransformer"/> + <see cref="WanS2VAudioEncoder"/> + reused Wan VAE decode driven through
/// <see cref="WanS2VPipeline"/> on CPU (pre-computed audio features → frames). Reconstructed; validation-pending.</summary>
public unsafe class WanS2VPipelineTests
{
    [Fact]
    public void GenerateFromAudioFeatures_TinyConfig_ProducesFrames()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48, VaeLatentChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 4, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5,
        };
        int[] inject = [0, 2];
        const int numLayers = 3, audioDim = 10, tokens = 3;

        WanS2VTransformer transformer = new(cfg, inject);
        transformer.LoadWeights(WanSyntheticWeights.BuildS2VTransformer(cfg, inject));
        WanS2VAudioEncoder audioEnc = new(numLayers, audioDim, cfg.InnerDim, tokensPerFrame: tokens);
        audioEnc.LoadWeights(WanSyntheticWeights.BuildAudioEncoder("audio_encoder", numLayers, audioDim, cfg.InnerDim, tokens), "audio_encoder");

        int[] dimMult = [1, 2, 4, 4];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [false, true, true]);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, [false, true, true]));

        WanS2VPipeline pipeline = new(backend, transformer, audioEnc, vae, cfg);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        int numFrames = 5;                       // → tLat = 2 latent frames
        Tensor audio = Rand3d(2, numLayers, audioDim, seed: 9);   // [gt=2, numLayers, audioDim]
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateFromAudioFeatures(promptEmbeds, negEmbeds, audio, req, numFrames);
        Assert.Equal(numFrames, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);
    }

    [Fact]
    public void GenerateChunked_ReferenceAndMotion_ProducesFrames()
    {
        CpuBackend backend = new();
        // Reference-conditioned S2V: InChannels = 2·z (noisy + reference broadcast).
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 96, OutChannels = 48, VaeLatentChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 4, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5,
        };
        int[] inject = [0, 2];
        const int numLayers = 3, audioDim = 10, tokens = 3;

        WanS2VTransformer transformer = new(cfg, inject);
        transformer.LoadWeights(WanSyntheticWeights.BuildS2VTransformer(cfg, inject));
        WanS2VAudioEncoder audioEnc = new(numLayers, audioDim, cfg.InnerDim, tokensPerFrame: tokens);
        audioEnc.LoadWeights(WanSyntheticWeights.BuildAudioEncoder("audio_encoder", numLayers, audioDim, cfg.InnerDim, tokens), "audio_encoder");

        int[] dimMult = [1, 2, 4, 4];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [false, true, true]);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, [false, true, true]));
        Wan22VaeEncoder enc = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalDownsample: [true, true, false]);
        enc.LoadWeights(LanceSyntheticWeights.BuildVaeEncoder(8, 48, dimMult, 2, [true, true, false]));

        WanS2VPipeline pipeline = new(backend, transformer, audioEnc, vae, cfg, enc);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        Tensor audio = Rand3d(3, numLayers, audioDim, seed: 9);   // 3 latent frames total → 2 clips (gt 2, motion 1)
        byte[] reference = new byte[32 * 32 * 3];
        new Random(5).NextBytes(reference);
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateChunked(
            promptEmbeds, negEmbeds, audio, reference, req, framesPerClip: 5, motionFrames: 1);
        Assert.True(frames.Length > 5, $"multi-chunk should extend past one clip: {frames.Length}");
        Assert.Equal(32, w);
        Assert.Equal(32, h);
        foreach (byte[] f in frames) Assert.Equal(32 * 32 * 3, f.Length);
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Rand3d(int a, int b, int c, int seed)
    {
        Tensor x = new Tensor(new TensorShape(a, b, c), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}

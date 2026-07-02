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

/// <summary>End-to-end structural tests for the Wan2.2-S2V single-clip pipeline (ComfyUI-reference flow): tiny-config
/// <see cref="WanS2VTransformer"/> + <see cref="WanS2VAudioEncoder"/> + Wan VAE driven through
/// <see cref="WanS2VPipeline"/> on CPU — per-video-frame audio features (zeroed for the negative CFG pass) and an
/// optional reference image passed as appended tokens.</summary>
public unsafe class WanS2VPipelineTests
{
    [Fact]
    public void GenerateFromAudioFeatures_TinyConfig_ProducesFrames()
    {
        (WanS2VPipeline pipeline, WanVideoConfig cfg) = BuildPipeline(withEncoder: false);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        int numFrames = 5;                                                       // → tLat = 2, tVideo = 8
        Tensor audio = Rand3d(8, cfg.AudioLayers, cfg.AudioDim, seed: 9);        // [4·tLat, layers, audioDim]
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateFromAudioFeatures(promptEmbeds, negEmbeds, audio, req, numFrames);
        Assert.Equal(numFrames, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);
    }

    [Fact]
    public void GenerateFromAudioFeatures_WithReferenceImage_ProducesFrames()
    {
        (WanS2VPipeline pipeline, WanVideoConfig cfg) = BuildPipeline(withEncoder: true);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        int numFrames = 5;
        // Longer than 4·tLat on purpose — the pipeline slices to the clip length like the reference node.
        Tensor audio = Rand3d(10, cfg.AudioLayers, cfg.AudioDim, seed: 9);
        byte[] reference = new byte[32 * 32 * 3];
        new Random(5).NextBytes(reference);
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateFromAudioFeatures(
            promptEmbeds, negEmbeds, audio, req, numFrames, reference);
        Assert.Equal(numFrames, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);
        foreach (byte[] f in frames) Assert.Equal(32 * 32 * 3, f.Length);
    }

    [Fact]
    public void ResampleAudioFeatures_MapsFiftyHzToVideoFrames()
    {
        // 50 features at 50 Hz = 1 s of audio → 30 frames at 30 fps; 16 fps sampling reaches index round(i·1.875).
        Tensor allLayers = Rand3d(50, 2, 4, seed: 3);
        Tensor resampled = WanS2VPipeline.ResampleAudioFeatures(allLayers, 20);
        Assert.Equal(new long[] { 20, 2, 4 }, new[] { resampled.Shape[0], resampled.Shape[1], resampled.Shape[2] });
        float* p = (float*)resampled.DataPointer;
        for (long i = 0; i < resampled.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
        // Video frames past the audio (i=16 → bucket index 30 ≥ 30 frames) are zero-padded.
        bool tailZero = true;
        for (long i = 16 * 2 * 4; i < resampled.Shape.ElementCount; i++) tailZero &= p[i] == 0f;
        Assert.True(tailZero, "frames past the audio must be zero");
        // Frame 0 samples 30 fps frame 0 = 50 Hz frame 0 exactly (align_corners).
        float* src = (float*)allLayers.DataPointer;
        for (int j = 0; j < 8; j++) Assert.Equal(src[j], p[j], 5);
    }

    private static (WanS2VPipeline Pipeline, WanVideoConfig Config) BuildPipeline(bool withEncoder)
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48, VaeLatentChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 4, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5,
            AudioInjectLayers = [0, 2], AudioDim = 10, AudioTokens = 3, AudioLayers = 3,
        };

        WanS2VTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildS2VTransformer(cfg));
        WanS2VAudioEncoder audioEnc = new(cfg.AudioLayers, cfg.AudioDim, cfg.InnerDim, numTokens: cfg.AudioTokens);
        audioEnc.LoadWeights(WanSyntheticWeights.BuildAudioEncoder(cfg.AudioLayers, cfg.AudioDim, cfg.InnerDim, cfg.AudioTokens));

        int[] dimMult = [1, 2, 4, 4];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [false, true, true]);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, [false, true, true]));

        Wan22VaeEncoder? enc = null;
        if (withEncoder)
        {
            enc = new Wan22VaeEncoder(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalDownsample: [true, true, false]);
            enc.LoadWeights(LanceSyntheticWeights.BuildVaeEncoder(8, 48, dimMult, 2, [true, true, false]));
        }

        return (new WanS2VPipeline(backend, transformer, audioEnc, vae, cfg, enc), cfg);
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

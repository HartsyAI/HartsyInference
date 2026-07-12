using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end structural test for the Wan2.2-Animate pipeline: tiny-config
/// <see cref="WanAnimateTransformer"/> (reference-frame concat conditioning + pose latent add + motion/face pathway
/// with the black-face negative branch) + reused Wan VAE encode/decode driven through
/// <see cref="WanAnimatePipeline"/> on CPU. Numerics validation-pending.</summary>
public unsafe class WanAnimatePipelineTests
{
    [Fact]
    public void GenerateAnimation_TinyConfig_ProducesFrames()
    {
        CpuBackend backend = new();
        // InChannels = 2·z + 4 (noise + [mask, cond-latent] concat), matching the real Animate 36 = 2·16 + 4 layout.
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 100, OutChannels = 48, VaeLatentChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 5, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5, IsAnimate = true,
        };
        const int poseCh = 48, motionSize = 16, styleDim = 8, motionDim = 4, fcLayers = 2;
        const int faceHidden = 16, faceHeads = 2;

        WanAnimateTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildAnimateTransformer(cfg, poseCh, motionSize, styleDim,
            motionDim, fcLayers, faceHidden, faceHeads));

        int[] dimMult = [1, 2, 4, 4];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [false, true, true]);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, [false, true, true]));
        Wan22VaeEncoder encoder = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalDownsample: [true, true, false]);
        encoder.LoadWeights(LanceSyntheticWeights.BuildVaeEncoder(8, 48, dimMult, 2, [true, true, false]));

        WanAnimatePipeline pipeline = new(backend, transformer, vae, encoder, cfg);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        Tensor referenceRgb = Rand5d(1, 3, 1, 32, 32, seed: 8);                 // ref frame → 1 trim latent frame
        Tensor poseClip = Rand5d(1, 3, 5, 32, 32, seed: 9);                     // 5 pose frames → 2 latent frames
        Tensor faceClip = Rand5d(1, 3, 4, motionSize, motionSize, seed: 10);    // 4 face frames → motion T'=1 (+zero)
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _, WanAnimateConditioning cond) = pipeline.GenerateAnimation(
            promptEmbeds, negEmbeds, referenceRgb, poseClip, faceClip, req);
        cond.Dispose();
        Assert.Equal(5, frames.Length);   // ref latent frame trimmed before decode → the 2 generated latent frames
        Assert.Equal(32, w);
        Assert.Equal(32, h);
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

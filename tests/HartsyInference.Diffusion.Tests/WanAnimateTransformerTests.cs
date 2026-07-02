using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config end-to-end smoke tests for the Wan2.2-Animate DiT (<see cref="WanAnimateTransformer"/>):
/// the pose patch-embed addition (frames 1..), the motion-encoder → face-encoder → face-adapter pathway (fused every
/// 5th block), the i2v CLIP image context, and the ref_conv token-prepend/trim, producing a finite, correctly-shaped
/// velocity. Numerics validation-pending vs the real checkpoint.</summary>
public unsafe class WanAnimateTransformerTests
{
    private const int PoseCh = 8, MotionSize = 16, StyleDim = 8, MotionDim = 4, FcLayers = 2;
    private const int FaceHidden = 16, FaceHeads = 2;

    private static WanVideoConfig TinyConfig(int imageDim = 0) => new()
    {
        NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8,
        TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 5, PatchSize = (1, 2, 2),
        ImageDim = imageDim, AddedKvProjDim = imageDim, PosEmbedSeqLen = imageDim > 0 ? 3 : 0,
        IsAnimate = true,
    };

    [Fact]
    public void Forward_PoseFaceAndClipConditioning_ProducesLatentShape()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = TinyConfig(imageDim: 16);
        WanAnimateTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildAnimateTransformer(cfg, PoseCh, MotionSize, StyleDim,
            MotionDim, FcLayers, FaceHidden, FaceHeads));

        Tensor latent = Rand5d(1, cfg.InChannels, 2, 4, 4, seed: 61);          // gt=2, S=8 tokens
        Tensor pose = Rand5d(1, PoseCh, 1, 4, 4, seed: 62);                    // 1 pose frame → added to latent frame 1
        Tensor face = Rand5d(1, 3, 4, MotionSize, MotionSize, seed: 63);       // 4 face frames → 1 + zero-prepend = gt
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 64);
        Tensor clip = RandRows(3, cfg.ImageDim, seed: 65);

        Tensor outVel = transformer.Forward(backend, latent, pose, face, encoder, timestep: 0.5f, clipImageEmbeds: clip);

        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(2, (int)outVel.Shape[2]);
        Assert.Equal(4, (int)outVel.Shape[3]);
        AssertFinite(outVel);
    }

    [Fact]
    public void Forward_RefConvTokenPrepend_TrimsToLatentShape()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = TinyConfig();
        WanAnimateTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildAnimateTransformer(cfg, PoseCh, MotionSize, StyleDim,
            MotionDim, FcLayers, FaceHidden, FaceHeads, refConvChannels: cfg.InChannels));
        Assert.True(transformer.HasRefConv);

        Tensor latent = Rand5d(1, cfg.InChannels, 2, 4, 4, seed: 71);
        Tensor pose = Rand5d(1, PoseCh, 1, 4, 4, seed: 72);
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 73);
        Tensor reference = Rand4d(1, cfg.InChannels, 4, 4, seed: 74);

        Tensor outVel = transformer.Forward(backend, latent, pose, facePixels: null, encoder, timestep: 0.5f,
            referenceLatent: reference);

        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(2, (int)outVel.Shape[2]);   // ref tokens trimmed after the head
        AssertFinite(outVel);
    }

    [Fact]
    public void Forward_RefConvPlusFacePathway_Throws()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = TinyConfig();
        WanAnimateTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildAnimateTransformer(cfg, PoseCh, MotionSize, StyleDim,
            MotionDim, FcLayers, FaceHidden, FaceHeads, refConvChannels: cfg.InChannels));

        Tensor latent = Rand5d(1, cfg.InChannels, 2, 4, 4, seed: 81);
        Tensor face = Rand5d(1, 3, 4, MotionSize, MotionSize, seed: 82);
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 83);
        Tensor reference = Rand4d(1, cfg.InChannels, 4, 4, seed: 84);

        Assert.Throws<ArgumentException>(() => transformer.Forward(backend, latent, pose: null, face, encoder,
            timestep: 0.5f, referenceLatent: reference));
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Rand4d(int b, int c, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape(b, c, h, w), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
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

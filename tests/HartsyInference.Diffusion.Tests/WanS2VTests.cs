using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config structural tests for the Wan2.2-S2V engine pieces, ported from ComfyUI's reference: the
/// <see cref="WanS2VAudioEncoder"/> (weighted-mean layer combine + causal conv stack, 4× temporal downsample →
/// per-latent-frame local+global tokens) and the <see cref="WanS2VTransformer"/> (audio injector, trainable cond
/// mask, reference-latent token append with per-frame timesteps).</summary>
public unsafe class WanS2VTests
{
    [Fact]
    public void AudioEncoder_DownsamplesToLatentFrames()
    {
        CpuBackend backend = new();
        const int numLayers = 3, audioDim = 10, dim = 24, tokens = 3, videoFrames = 8;   // → 2 latent frames
        WanS2VAudioEncoder enc = new(numLayers, audioDim, dim, numTokens: tokens);
        enc.LoadWeights(WanSyntheticWeights.BuildAudioEncoder(numLayers, audioDim, dim, tokens));

        Tensor audio = Rand3d(videoFrames, numLayers, audioDim, seed: 71);
        (Tensor global, Tensor local) = enc.Forward(backend, audio);

        Assert.Equal(new[] { 2L, tokens + 1, dim }, new[] { local.Shape[0], local.Shape[1], local.Shape[2] });
        Assert.Equal(new[] { 2L, 1, dim }, new[] { global.Shape[0], global.Shape[1], global.Shape[2] });
        float* lp = (float*)local.DataPointer;
        for (long i = 0; i < local.Shape.ElementCount; i++) Assert.True(float.IsFinite(lp[i]), $"non-finite local at {i}");
        float* gp = (float*)global.DataPointer;
        for (long i = 0; i < global.Shape.ElementCount; i++) Assert.True(float.IsFinite(gp[i]), $"non-finite global at {i}");

        // The last local token per frame is the learnable padding token — identical across frames.
        for (int d = 0; d < dim; d++)
            Assert.Equal(lp[((0L * (tokens + 1)) + tokens) * dim + d], lp[((1L * (tokens + 1)) + tokens) * dim + d]);
    }

    [Fact]
    public void Transformer_AudioAndReferenceConditioningInfluenceOutput()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8, VaeLatentChannels = 8,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 4, PatchSize = (1, 2, 2),
            AudioInjectLayers = [0, 2], AudioDim = 10, AudioTokens = 3, AudioLayers = 3,
        };
        WanS2VTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildS2VTransformer(cfg));

        Tensor latent = Rand5d(1, cfg.InChannels, 2, 4, 4, seed: 81);   // gt=2, 4 tokens/frame, S=8
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 82);
        Tensor audioLocal = Rand3d(2, cfg.AudioTokens + 1, cfg.InnerDim, seed: 83);
        Tensor audioGlobal = Rand3d(2, 1, cfg.InnerDim, seed: 84);

        Tensor outVel = transformer.Forward(backend, latent, encoder, timestep: 500f, audioLocal, audioGlobal);
        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(2, (int)outVel.Shape[2]);
        float* p = (float*)outVel.DataPointer;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");

        // Different audio tokens must change the output (the injector is wired in).
        Tensor audioLocal2 = Rand3d(2, cfg.AudioTokens + 1, cfg.InnerDim, seed: 93);
        Tensor audioGlobal2 = Rand3d(2, 1, cfg.InnerDim, seed: 94);
        Tensor outAudio2 = transformer.Forward(backend, latent, encoder, timestep: 500f, audioLocal2, audioGlobal2);
        Assert.True(MaxDiff(outVel, outAudio2) > 1e-6f, "audio conditioning must influence the output");

        // A reference latent (appended tokens, timestep 0, far RoPE) must also change the output.
        Tensor refLatent = Rand5d(1, cfg.InChannels, 1, 4, 4, seed: 95);
        Tensor outRef = transformer.Forward(backend, latent, encoder, timestep: 500f, audioLocal, audioGlobal, refLatent);
        Assert.Equal(2, (int)outRef.Shape[2]);   // reference tokens are dropped from the prediction
        Assert.True(MaxDiff(outVel, outRef) > 1e-6f, "reference conditioning must influence the output");
    }

    private static float MaxDiff(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer;
        float maxDiff = 0;
        for (long i = 0; i < a.Shape.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(ap[i] - bp[i]));
        return maxDiff;
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

    private static Tensor Rand5d(int b, int c, int t, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}

using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config smoke tests for the Wan2.2-S2V engine pieces: the <see cref="WanS2VAudioEncoder"/> (stacked
/// audio layers → per-frame tokens) and the <see cref="WanS2VTransformer"/> (base Wan + audio-injector cross-attn).
/// S2V has no diffusers reference — these verify the reconstructed wiring runs + that audio influences the output.</summary>
public unsafe class WanS2VTests
{
    [Fact]
    public void AudioEncoder_ProducesPerFrameTokens()
    {
        CpuBackend backend = new();
        const int numLayers = 3, audioDim = 10, dim = 24, tokens = 3, frames = 2;
        WanS2VAudioEncoder enc = new(numLayers, audioDim, dim, tokensPerFrame: tokens);
        enc.LoadWeights(WanSyntheticWeights.BuildAudioEncoder("audio_encoder", numLayers, audioDim, dim, tokens), "audio_encoder");

        Tensor audio = Rand3d(frames, numLayers, audioDim, seed: 71);
        Tensor outT = enc.Forward(backend, audio);

        Assert.Equal(3, outT.Shape.Rank);
        Assert.Equal(frames, (int)outT.Shape[0]);
        Assert.Equal(tokens, (int)outT.Shape[1]);
        Assert.Equal(dim, (int)outT.Shape[2]);
        float* p = (float*)outT.DataPointer;
        for (long i = 0; i < outT.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void Transformer_TinyConfig_AudioConditioningInfluencesOutput()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 4, PatchSize = (1, 2, 2),
        };
        int[] inject = [0, 2];
        WanS2VTransformer transformer = new(cfg, inject);
        transformer.LoadWeights(WanSyntheticWeights.BuildS2VTransformer(cfg, inject));

        Tensor latent = Rand5d(1, cfg.InChannels, 2, 4, 4, seed: 81);   // S=8, gt=2
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 82);
        Tensor audioTokens = Rand3d(2, 3, cfg.InnerDim, seed: 83);      // [gt=2, tokens=3, dim], gt | S

        Tensor outVel = transformer.Forward(backend, latent, audioTokens, encoder, timestep: 0.5f);
        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(2, (int)outVel.Shape[2]);
        float* p = (float*)outVel.DataPointer;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");

        // Zero audio must change the output (audio injector is wired in).
        Tensor zeroAudio = new Tensor(audioTokens.Shape, DType.F32);
        new Span<float>((float*)zeroAudio.DataPointer, (int)zeroAudio.Shape.ElementCount).Clear();
        Tensor outZero = transformer.Forward(backend, latent, zeroAudio, encoder, timestep: 0.5f);
        float* z = (float*)outZero.DataPointer;
        float maxDiff = 0;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(p[i] - z[i]));
        Assert.True(maxDiff > 1e-6f, $"audio conditioning must influence the output: maxDiff={maxDiff}");
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

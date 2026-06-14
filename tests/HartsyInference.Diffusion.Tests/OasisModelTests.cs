using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config CPU tests for the Oasis-500m model classes: the ViT-VAE round trip and the spatio-temporal
/// DiT — including the temporal-causality gate (future frames must not influence past predictions) and action
/// sensitivity. Numerics vs the reference checkpoint are validation-pending.</summary>
public unsafe class OasisModelTests
{
    private static OasisDitConfig TinyDit => new()
    {
        HiddenSize = 32, NumHeads = 2, Depth = 2, InChannels = 16, PatchSize = 2,
        FreqDim = 16, MaxFrames = 8, SpatialRopeDimPerAxis = 8, SpatialRopeMaxFreq = 256,
    };

    [Fact]
    public void VitVae_EncodeDecode_RoundTripsShapes()
    {
        CpuBackend backend = new();
        OasisVitVae vae = new(latentDim: 16, patchSize: 4, dim: 32, heads: 2, encDepth: 2, decDepth: 2);
        vae.LoadWeights(OasisSyntheticWeights.BuildVitVae(16, 4, 32, 2, 2));

        Tensor rgb = Rand4d(1, 3, 16, 16, seed: 3);
        Tensor latent = vae.Encode(backend, rgb);
        Assert.Equal(16, (int)latent.Shape[1]);
        Assert.Equal(4, (int)latent.Shape[2]);
        Assert.Equal(4, (int)latent.Shape[3]);

        Tensor decoded = vae.Decode(backend, latent);
        Assert.Equal(3, (int)decoded.Shape[1]);
        Assert.Equal(16, (int)decoded.Shape[2]);
        Assert.Equal(16, (int)decoded.Shape[3]);
        float* p = (float*)decoded.DataPointer;
        for (long i = 0; i < decoded.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    [Fact]
    public void Dit_Forward_ProducesVAndRespectsTemporalCausality()
    {
        CpuBackend backend = new();
        OasisDitConfig cfg = TinyDit;
        using OasisDit dit = new(cfg);
        dit.LoadWeights(OasisSyntheticWeights.BuildDit(cfg));

        Tensor latents = Rand4d(3, 16, 4, 4, seed: 7);
        Tensor actions = RandRows(3, 25, seed: 8);
        int[] noiseIdx = [14, 14, 500];

        Tensor v1 = dit.Forward(backend, latents, noiseIdx, actions);
        Assert.Equal(3, (int)v1.Shape[0]);
        Assert.Equal(16, (int)v1.Shape[1]);
        Assert.Equal(4, (int)v1.Shape[2]);
        Assert.Equal(4, (int)v1.Shape[3]);

        // Causality: perturb ONLY the last frame; frames 0 and 1 must be bit-identical.
        Tensor perturbed = Clone(latents);
        float* pp = (float*)perturbed.DataPointer;
        long perFrame = 16L * 4 * 4;
        for (long i = 2 * perFrame; i < 3 * perFrame; i++) pp[i] += 1.0f;
        Tensor v2 = dit.Forward(backend, perturbed, noiseIdx, actions);

        float* a = (float*)v1.DataPointer;
        float* b = (float*)v2.DataPointer;
        for (long i = 0; i < 2 * perFrame; i++)
            Assert.True(a[i] == b[i], $"future frame leaked into past prediction at {i}");
        float lastDiff = 0;
        for (long i = 2 * perFrame; i < 3 * perFrame; i++) lastDiff = MathF.Max(lastDiff, MathF.Abs(a[i] - b[i]));
        Assert.True(lastDiff > 1e-6f, "perturbing the target frame must change its own prediction");

        // Action sensitivity: a different action stream changes the output.
        Tensor actions2 = RandRows(3, 25, seed: 99);
        Tensor v3 = dit.Forward(backend, latents, noiseIdx, actions2);
        float* c = (float*)v3.DataPointer;
        float anyDiff = 0;
        for (long i = 0; i < v1.Shape.ElementCount; i++) anyDiff = MathF.Max(anyDiff, MathF.Abs(a[i] - c[i]));
        Assert.True(anyDiff > 1e-6f, "actions must condition the prediction");
    }

    private static Tensor Rand4d(int a, int b, int c, int d, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)a, b, c, d]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Clone(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, DType.F32);
        long bytes = x.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }
}

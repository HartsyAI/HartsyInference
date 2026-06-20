using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config end-to-end smoke test for the Wan VACE DiT (<see cref="WanVaceTransformer"/>): the control
/// branch runs and its per-layer hints are added into the main stream, producing a correctly-shaped, finite velocity.
/// Numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class WanVaceTransformerTests
{
    [Fact]
    public void Forward_TinyConfig_ControlBranchProducesLatentShape()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8, VaeLatentChannels = 8,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 4, PatchSize = (1, 2, 2),
            VaceLayers = [0, 2], VaceInChannels = 12,
        };
        WanVaceTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildVaceTransformer(cfg));

        Tensor latent = Rand5d(1, cfg.InChannels, 2, 4, 4, seed: 51);          // main
        Tensor control = Rand5d(1, cfg.VaceInChannels, 2, 4, 4, seed: 52);     // VACE control context (same grid)
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 53);

        Tensor outVel = transformer.Forward(backend, latent, control, encoder, timestep: 0.5f, controlScales: [1f, 0.5f]);

        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(2, (int)outVel.Shape[2]);
        Assert.Equal(4, (int)outVel.Shape[3]);
        float* p = (float*)outVel.DataPointer;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");

        // A non-zero control hint must change the output vs. zero control.
        Tensor zeroControl = new Tensor(control.Shape, DType.F32);
        new Span<float>((float*)zeroControl.DataPointer, (int)zeroControl.Shape.ElementCount).Clear();
        Tensor outZero = transformer.Forward(backend, latent, zeroControl, encoder, timestep: 0.5f, controlScales: [1f, 0.5f]);
        float* z = (float*)outZero.DataPointer;
        float maxDiff = 0;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(p[i] - z[i]));
        Assert.True(maxDiff > 1e-5f, $"control must influence the output: maxDiff={maxDiff}");
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

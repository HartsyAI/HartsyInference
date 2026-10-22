using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Wan context-parallel split on two CPU backends: the transformer hands every block a
/// <see cref="CpExchangeRewrite"/> over the real <see cref="CpKvExchange"/>, and the concatenated rank outputs must match the
/// single-backend forward (the GPU twin in the Video suite returns early without CUDA).</summary>
public sealed unsafe class WanContextParallelCpuTests
{
    private static WanVideoConfig TinyConfig => new()
    {
        NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
        TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
        VaeSpatialCompression = 16, VaeTemporalCompression = 4,
        NumInferenceSteps = 3, GuidanceScale = 5, FlowShift = 5,
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheSplitForwardMatchesTheSingleBackendForward(bool perFrameTimesteps)
    {
        WanVideoConfig cfg = TinyConfig;
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));
        using CpuBackend backend1 = new(), backend2 = new();
        const int tLat = 3, hLat = 4, wLat = 4;
        (int pt, int ph, int pw) = cfg.PatchSize;
        int gt = tLat / pt, gh = hLat / ph, gw = wLat / pw;
        Tensor latent = Rand([1L, cfg.InChannels, tLat, hLat, wLat], seed: 7);
        Tensor prompt = Rand([3L, cfg.TextDim], seed: 1);
        float[] timesteps = perFrameTimesteps ? [0f, 500f, 500f] : [500f];

        Tensor baseline = transformer.Forward(backend1, latent, prompt, timesteps);

        CpSequencePlan plan = CpSequencePlan.Create(gt, gh * gw, [1.0, 1.0]);
        using CpKvExchange exchange = new(plan);
        transformer.PrewarmSequenceCaches(backend1, gt, gh, gw, prompt);
        Tensor rank1Latent = Rand([1L, cfg.InChannels, tLat, hLat, wLat], seed: 7);
        (Tensor local0, Tensor local1) = CfgBranchRunner.Run(
            () => RankForward(transformer, backend1, latent, prompt, timesteps, 0, plan, exchange),
            () => RankForward(transformer, backend2, rank1Latent, prompt, timesteps, 1, plan, exchange));
        Tensor full = WanDitOps.ConcatRows(local0, local1, cfg.OutChannels * pt * ph * pw);
        Tensor cpOut = WanDitOps.Unpatchify(full, cfg.OutChannels, gt, gh, gw, cfg.PatchSize);

        Assert.Equal(baseline.Shape, cpOut.Shape);
        float* a = (float*)baseline.DataPointer, b = (float*)cpOut.DataPointer;
        float max = 0;
        for (long i = 0; i < baseline.ElementCount; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        Assert.True(max < 1e-4f, $"CP-split forward diverged from the single-backend forward (max |diff| = {max:E3}).");
    }

    private static Tensor RankForward(WanVideoTransformer transformer, Core.Backends.IBackend backend, Tensor latent, Tensor prompt,
        float[] timesteps, int rank, CpSequencePlan plan, CpKvExchange exchange)
    {
        try
        {
            return transformer.Forward(backend, latent, prompt, timesteps, null, new CpForwardContext { Rank = rank, Plan = plan, Exchange = exchange });
        }
        catch
        {
            exchange.Abort();
            throw;
        }
    }

    private static Tensor Rand(long[] shape, int seed)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}

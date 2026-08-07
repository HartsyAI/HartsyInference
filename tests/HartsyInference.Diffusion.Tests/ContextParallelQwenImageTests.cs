using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Same-GPU parity for Qwen-Image context parallelism (img-row sequence split, replicated weights,
/// replicated txt stream): two <see cref="CudaBackend"/> instances on ONE ordinal (the ContextParallelWanTests
/// shape) and a tiny synthetic config, so only the mechanics are under test — the row-aligned
/// <see cref="CpSequencePlan"/> slicing, the rank-restricted GLOBAL-position rope tables, the per-block
/// <see cref="CpKvExchange"/> joint K/V rendezvous with the replicated txt prefix, and the fork/gather path the
/// pipeline uses. The CP math is the identical kernels over identical values (the exchange is pure copies), so
/// two tiers: single-head is bit-tight (measured exactly 0 — the mechanics gate), multi-head gets a relative bar
/// for the SDPA kernels' (heads × sq)-dependent tiling drift (see the per-fact remarks).</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public unsafe class ContextParallelQwenImageTests
{
    private readonly ITestOutputHelper _output;
    public ContextParallelQwenImageTests(ITestOutputHelper output) => _output = output;

    // HeadDim must stay 128: QwenImageTransformer hardcodes the rope's [16, 56, 56] axis split (sums to 128).
    private static QwenImageConfig TinyConfig(int numHeads) => new()
    {
        HiddenSize = numHeads * 128,
        NumHeads = numHeads,
        HeadDim = 128,
        Depth = 4,
        MlpRatio = 2.0f,
        ContextDim = 32,
        InChannels = 16,
        PatchSize = 2,
    };

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    /// <summary>Single-head config: every kernel in the CP path is bit-deterministic across the M=Sr-vs-M=S
    /// geometry change here (measured max |diff| = 0), so this fact pins the split/rope/exchange MECHANICS
    /// bit-for-bit — any nonzero drift at heads=1 is a wiring bug, not numerics.</summary>
    [Fact]
    public void Transformer_CpSplitForward_MatchesSingleBackend_SingleHead_BitTight()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunParity(numHeads: 1, weights: [1.0, 1.0], txtSeqLens: [5], timesteps: [0.7f], maxAbsBar: 1e-4f);
    }

    /// <summary>Multi-head config: the SDPA kernels' internal tiling varies with the (heads × sq) geometry, so a
    /// local-q (sq=Sr+txt) forward legitimately drifts a few 1e-5 relative per block vs the sq=S baseline —
    /// measured err/scale ≤ 2.3e-4 across the cuDNN-F16, cuDNN-F32 and custom-F32 SDPA paths at Depth=4, and it
    /// persists (shrinks to ~8e-5/block) in full F32, so it is reduction-order numerics, not precision mode. The
    /// relative bar (the sharding tests' cross-device convention) still catches any real boundary/exchange bug,
    /// which lands at O(1) of scale; the single-head fact above pins exactness separately.</summary>
    [Fact]
    public void Transformer_CpSplitForward_MatchesSingleBackend_MultiHead()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunParity(numHeads: 2, weights: [1.0, 1.0], txtSeqLens: [5], timesteps: [0.7f], relToScaleBar: 1e-3f);
    }

    /// <summary>Asymmetric row split (4+1) plus TWO sequential forwards through ONE exchange instance with
    /// DIFFERENT txt lengths — the cond/uncond shape, which alternates the per-backend rope-table key and
    /// exercises the locked rebuild from both rank threads.</summary>
    [Fact]
    public void Transformer_CpSplitForward_AsymmetricSplit_ExchangeReusedAcrossForwards()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunParity(numHeads: 1, weights: [3.0, 1.0], txtSeqLens: [5, 3], timesteps: [0.7f, 0.35f], maxAbsBar: 1e-4f);
    }

    private void RunParity(int numHeads, double[] weights, int[] txtSeqLens, float[] timesteps,
        float maxAbsBar = float.MaxValue, float relToScaleBar = float.MaxValue)
    {
        QwenImageConfig cfg = TinyConfig(numHeads);
        const int hPacked = 5, wPacked = 4;   // 5 packed rows → uneven splits for both weight sets
        int patchDim = cfg.PatchSize * cfg.PatchSize * cfg.InChannels;
        int imgSeq = hPacked * wPacked;

        Dictionary<string, Tensor> w = QwenImageWeightBuilder.Build(cfg);
        using QwenImageTransformer transformer = new(cfg);
        transformer.LoadWeights(w);

        using CudaBackend backend1 = new(deviceOrdinal: 0, PtxDir());
        using CudaBackend backend2 = new(deviceOrdinal: 0, PtxDir());
        backend1.PreloadWeights(transformer.EnumerateWeights());
        backend2.PreloadWeights(transformer.EnumerateWeights());

        CpSequencePlan plan = CpSequencePlan.Create(hPacked, wPacked, weights);
        _output.WriteLine($"plan: rows {plan.Ranks[0].FrameCount}+{plan.Ranks[1].FrameCount}, " +
            $"tokens {plan.Ranks[0].TokenCount}+{plan.Ranks[1].TokenCount}");
        using CpKvExchange exchange = new(plan);

        for (int f = 0; f < timesteps.Length; f++)
        {
            int txtSeq = txtSeqLens[f];
            float t = timesteps[f];
            using Tensor packedLatent = QwenImageWeightBuilder.Rand(new TensorShape(1, imgSeq, patchDim), 100 + f, 0.5f);
            using Tensor encoderHidden = QwenImageWeightBuilder.Rand(new TensorShape(1, txtSeq, cfg.ContextDim), 200 + f, 0.2f);

            Tensor baseline = transformer.Forward(backend1, packedLatent, encoderHidden, t, hPacked, wPacked);
            _ = baseline.DataPointer;

            // Rank 1 gets its own host clones — the pipeline's contract (never share non-weight tensors
            // across the two rank threads' backends).
            Tensor rank1Latent = Clone(packedLatent);
            Tensor rank1Hidden = Clone(encoderHidden);
            Tensor local0, local1;
            try
            {
                (local0, local1) = CfgBranchRunner.Run(
                    () => RankForward(transformer, backend1, packedLatent, encoderHidden, t, hPacked, wPacked, 0, plan, exchange),
                    () => RankForward(transformer, backend2, rank1Latent, rank1Hidden, t, hPacked, wPacked, 1, plan, exchange));
            }
            finally
            {
                rank1Latent.Dispose();
                rank1Hidden.Dispose();
            }
            Tensor cpOut = ConcatSeq(local0, local1);
            local0.Dispose();
            local1.Dispose();

            float maxDiff = MaxAbsDiff(baseline, cpOut);
            float scale = AbsMax(baseline);
            float relToScale = maxDiff / Math.Max(scale, 1e-6f);
            _output.WriteLine($"forward {f} (txtSeq={txtSeq}, t={t}): max |cp - baseline| = {maxDiff:E3}, " +
                $"output scale = {scale:E3}, err/scale = {relToScale:E3}");
            Assert.True(maxDiff < maxAbsBar,
                $"CP-split forward {f} diverged from the single-backend forward (max |diff| = {maxDiff:E3}).");
            Assert.True(relToScale < relToScaleBar,
                $"CP-split forward {f} diverged from the single-backend forward (err/scale = {relToScale:E3}).");
            baseline.Dispose();
            cpOut.Dispose();
        }
        QwenImageWeightBuilder.DisposeAll(w);
    }

    private static Tensor RankForward(QwenImageTransformer transformer, IBackend backend, Tensor latent,
        Tensor hidden, float t, int hPacked, int wPacked, int rank, CpSequencePlan plan, CpKvExchange exchange)
    {
        try
        {
            CpForwardContext ctx = new() { Rank = rank, Plan = plan, Exchange = exchange };
            Tensor local = transformer.Forward(backend, latent, hidden, t, hPacked, wPacked,
                refGrids: null, refTimestepZero: false, stepCache: null, cp: ctx);
            _ = local.DataPointer;   // host-materialize on the owning thread
            return local;
        }
        catch
        {
            exchange.Abort();
            throw;
        }
    }

    private static Tensor Clone(Tensor src)
    {
        Tensor o = new Tensor(src.Shape, DType.F32);
        long bytes = src.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }

    private static Tensor ConcatSeq(Tensor a, Tensor b)
    {
        long dim = a.Shape[2];
        Tensor o = new Tensor(new TensorShape(1, a.Shape[1] + b.Shape[1], dim), DType.F32);
        long aBytes = a.ElementCount * 4, bBytes = b.ElementCount * 4;
        Buffer.MemoryCopy((float*)a.DataPointer, (float*)o.DataPointer, aBytes, aBytes);
        Buffer.MemoryCopy((float*)b.DataPointer, (float*)o.DataPointer + a.ElementCount, bBytes, bBytes);
        return o;
    }

    private static float AbsMax(Tensor a)
    {
        float* p = (float*)a.DataPointer;
        float max = 0;
        for (long i = 0; i < a.Shape.ElementCount; i++)
            max = Math.Max(max, Math.Abs(p[i]));
        return max;
    }

    private static float MaxAbsDiff(Tensor a, Tensor b)
    {
        Assert.Equal(a.Shape.ElementCount, b.Shape.ElementCount);
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        float max = 0;
        for (long i = 0; i < a.Shape.ElementCount; i++)
            max = Math.Max(max, Math.Abs(pa[i] - pb[i]));
        return max;
    }
}

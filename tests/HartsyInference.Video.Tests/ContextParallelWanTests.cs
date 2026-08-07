using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>Same-GPU parity for Wan context parallelism (sequence split, replicated weights): two
/// <see cref="CudaBackend"/> instances on ONE ordinal (the CfgBranchParallelWanTests shape) and a tiny synthetic
/// config, so only the mechanics are under test — the frame-aligned <see cref="CpSequencePlan"/> slicing, the
/// per-block <see cref="CpKvExchange"/> K/V rendezvous, the G&gt;1 per-frame-timestep range slicing, and the
/// pipeline's fork/gather/unpatchify path. The CP math is the identical kernels over identical values (the
/// exchange is pure copies), so the transformer-level bar is a tiny max|diff| (GEMM row-tiling can differ at
/// M=Sr vs M=S) and the pipeline-level bar is the CFG-parallel test's frame comparison.</summary>
public unsafe class ContextParallelWanTests
{
    private readonly ITestOutputHelper _output;
    public ContextParallelWanTests(ITestOutputHelper output) => _output = output;

    private static WanVideoConfig TinyConfig => new()
    {
        NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
        TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
        VaeSpatialCompression = 16, VaeTemporalCompression = 4,
        NumInferenceSteps = 3, GuidanceScale = 5, FlowShift = 5,
    };

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    [Fact]
    public void Transformer_CpSplitForward_MatchesSingleBackend_SharedTimestep()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunTransformerParity(perFrameTimesteps: false);
    }

    [Fact]
    public void Transformer_CpSplitForward_MatchesSingleBackend_PerFrameTimesteps()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunTransformerParity(perFrameTimesteps: true);
    }

    private void RunTransformerParity(bool perFrameTimesteps)
    {
        WanVideoConfig cfg = TinyConfig;
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));

        using CudaBackend backend1 = new(deviceOrdinal: 0, ptxDir: PtxDir());
        using CudaBackend backend2 = new(deviceOrdinal: 0, ptxDir: PtxDir());

        int tLat = 3, hLat = 4, wLat = 4;   // gt=3 frames → uneven 2+1 split; gh=gw=2 → 4 tokens/frame, S=12
        (int pt, int ph, int pw) = cfg.PatchSize;
        int gt = tLat / pt, gh = hLat / ph, gw = wLat / pw;
        Tensor latent = RandTensor([1L, cfg.InChannels, tLat, hLat, wLat], seed: 7);
        Tensor prompt = RandTensor([3L, cfg.TextDim], seed: 1);
        float[] timesteps = perFrameTimesteps ? [0f, 500f, 500f] : [500f];

        backend1.PreloadWeights(transformer.EnumerateWeights());
        backend2.PreloadWeights(transformer.EnumerateWeights());

        Tensor baseline = transformer.Forward(backend1, latent, prompt, timesteps);
        _ = baseline.DataPointer;

        CpSequencePlan plan = CpSequencePlan.Create(gt, gh * gw, [1.0, 1.0]);
        _output.WriteLine($"plan: frames {plan.Ranks[0].FrameCount}+{plan.Ranks[1].FrameCount}, " +
            $"tokens {plan.Ranks[0].TokenCount}+{plan.Ranks[1].TokenCount}");
        using CpKvExchange exchange = new(plan);
        transformer.PrewarmSequenceCaches(backend1, gt, gh, gw, prompt);
        Tensor rank1Latent = CloneTensor(latent);
        (Tensor local0, Tensor local1) = CfgBranchRunner.Run(
            () => RankForward(transformer, backend1, latent, prompt, timesteps, 0, plan, exchange),
            () => RankForward(transformer, backend2, rank1Latent, prompt, timesteps, 1, plan, exchange));
        rank1Latent.Dispose();
        Tensor full = WanDitOps.ConcatRows(local0, local1, cfg.OutChannels * pt * ph * pw);
        local0.Dispose();
        local1.Dispose();
        Tensor cpOut = WanDitOps.Unpatchify(full, cfg.OutChannels, gt, gh, gw, cfg.PatchSize);
        full.Dispose();

        float maxDiff = MaxAbsDiff(baseline, cpOut);
        _output.WriteLine($"max |cp - baseline| = {maxDiff:E3}");
        Assert.True(maxDiff < 1e-4f, $"CP-split forward diverged from single-backend forward (max |diff| = {maxDiff:E3}).");
        baseline.Dispose();
        cpOut.Dispose();
        latent.Dispose();
        prompt.Dispose();
    }

    private static Tensor RankForward(WanVideoTransformer transformer, IBackend backend, Tensor latent, Tensor prompt,
        float[] timesteps, int rank, CpSequencePlan plan, CpKvExchange exchange)
    {
        try
        {
            CpForwardContext ctx = new() { Rank = rank, Plan = plan, Exchange = exchange };
            Tensor local = transformer.Forward(backend, latent, prompt, timesteps, null, ctx);
            _ = local.DataPointer;   // host-materialize on the owning thread
            return local;
        }
        catch
        {
            exchange.Abort();
            throw;
        }
    }

    [Fact]
    public void GenerateFromEmbeddings_ContextParallel_MatchesSequential_SameGpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        WanVideoConfig cfg = TinyConfig;
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));

        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));

        using CudaBackend backend1 = new(deviceOrdinal: 0, ptxDir: PtxDir());
        using CudaBackend backend2 = new(deviceOrdinal: 0, ptxDir: PtxDir());

        Tensor promptEmbeds = RandTensor([3L, cfg.TextDim], seed: 1);
        Tensor negEmbeds = RandTensor([2L, cfg.TextDim], seed: 2);
        int numFrames = 5;   // (5-1)/4 + 1 = 2 latent frames → the minimum splittable gt
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 3, CfgScale = 5, Seed = 42 };

        WanVideoPipeline sequential = new(backend1, transformer, vae, cfg);
        (byte[][] baseline, int wBase, int hBase, _) = sequential.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames);
        _output.WriteLine($"sequential: {baseline.Length} frames, {wBase}x{hBase}");

        WanVideoPipeline parallel = new(backend1, transformer, vae, cfg) { CpBackends = [backend1, backend2] };
        (byte[][] cpFrames, int wCp, int hCp, _) = parallel.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames);
        _output.WriteLine($"context-parallel: {cpFrames.Length} frames, {wCp}x{hCp}, decision={parallel.LastCpDecision}");

        Assert.NotNull(parallel.LastCpDecision);
        Assert.StartsWith("active", parallel.LastCpDecision);
        Assert.Equal(wBase, wCp);
        Assert.Equal(hBase, hCp);
        Assert.Equal(baseline.Length, cpFrames.Length);
        for (int f = 0; f < baseline.Length; f++)
        {
            // Same kernels over the same values; the only wiggle is GEMM row-tiling at M=Sr vs M=S, which after
            // the VAE's u8 quantization is at most one grey level.
            int maxByteDiff = 0;
            for (int i = 0; i < baseline[f].Length; i++)
                maxByteDiff = Math.Max(maxByteDiff, Math.Abs(baseline[f][i] - cpFrames[f][i]));
            _output.WriteLine($"frame {f}: max byte diff = {maxByteDiff}");
            Assert.True(maxByteDiff <= 1, $"frame {f} differs from sequential output by {maxByteDiff} grey levels.");
        }
    }

    private static Tensor RandTensor(long[] shape, int seed)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor CloneTensor(Tensor src)
    {
        Tensor o = new Tensor(src.Shape, DType.F32);
        long bytes = src.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
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

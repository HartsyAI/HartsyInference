using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>Same-GPU bit-parity for CFG-branch parallelism (Phase 7, ROADMAP.md §1): two <see cref="CudaBackend"/>
/// instances on ONE device ordinal (the Phase 1 same-GPU dual-backend infra), a tiny synthetic Wan config so the
/// real weights' VRAM footprint never enters into it — the real TI2V-5B checkpoint needs ~10 GB replicated per
/// backend, which doesn't fit twice on either card on this box (see the ROADMAP.md §1 CFG-parallel note).
/// Proves the mechanics — <c>CfgBranchRunner</c> dispatch, per-backend weight caching after preload, the
/// worker thread's <c>[ThreadStatic]</c> ambient binding via its first backend call, per-step latent cloning —
/// produce byte-identical output to the sequential path, deterministically, on real CUDA kernels (not just
/// structurally on CPU).
///
/// <para><b>Run with HARTSY_ASSERT_AMBIENT=1</b> for the tripwire to be meaningful: a green run without it only
/// proves the output matched, not that every op bound its ambient correctly — a missed <c>EnterOp</c> on a
/// same-GPU (shared-context) setup could still silently resolve to the wrong backend's state and, worst case,
/// coincidentally still produce matching bytes since both backends share one physical device.</para></summary>
public unsafe class CfgBranchParallelWanTests
{
    private readonly ITestOutputHelper _output;
    public CfgBranchParallelWanTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GenerateFromEmbeddings_CfgParallel_MatchesSequential_SameGpu()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 3, GuidanceScale = 5, FlowShift = 5,
        };
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));

        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));

        using CudaBackend backend1 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend backend2 = new(deviceOrdinal: 0, ptxDir: ptxDir);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        int numFrames = 5;   // (5-1)/4 + 1 = 2 latent frames
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 3, CfgScale = 5, Seed = 42 };

        WanVideoPipeline sequential = new(backend1, transformer, vae, cfg);
        (byte[][] baseline, int wBase, int hBase, _) = sequential.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames);
        _output.WriteLine($"sequential: {baseline.Length} frames, {wBase}x{hBase}");

        WanVideoPipeline parallel = new(backend1, transformer, vae, cfg) { CfgParallelBackend = backend2 };
        (byte[][] parallelFrames, int wPar, int hPar, _) = parallel.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, numFrames);
        _output.WriteLine($"cfg-parallel: {parallelFrames.Length} frames, {wPar}x{hPar}");

        Assert.Equal(wBase, wPar);
        Assert.Equal(hBase, hPar);
        Assert.Equal(baseline.Length, parallelFrames.Length);
        for (int f = 0; f < baseline.Length; f++)
        {
            Assert.True(baseline[f].AsSpan().SequenceEqual(parallelFrames[f]),
                $"frame {f} differs between sequential and CFG-parallel output.");
        }
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}

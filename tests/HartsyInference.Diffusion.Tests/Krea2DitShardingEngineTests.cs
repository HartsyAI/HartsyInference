using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Phase 8 pipeline-wiring verification (ROADMAP.md §1): drives a REAL Krea 2 generation through the full
/// <c>InferenceEngine</c> → <c>Krea2Recipe</c> → <c>Krea2Pipeline</c> path with <see cref="PlacementConfig.EnableDitSharding"/>
/// set, not the raw <c>Krea2Transformer.ForwardSharded</c> primitive <see cref="Krea2DitShardingVramTests"/> already
/// covers. That test proved the split itself is correct and pools VRAM; this one proves the placement plumbing
/// (<c>RecipeContext.DitShardBackend</c>, <c>PlacementPlanner.DitSplitPlan</c>, the asymmetric preload/free in
/// <c>Krea2Pipeline</c>) is actually wired end to end from a <c>PlacementConfig</c> an operator would set.
/// <para><b>Tolerance, not bit-parity</b>: the two backends may take different fp8 GEMM paths (the 3060 on this box
/// is SM 8.6; <c>Fp8Executor.IsSupported</c> gates native fp8 GEMM at SM 8.9+, so whichever ordinal lands there falls
/// back to dequant) — sharded vs unsharded can legitimately diverge numerically even though both are correct. The
/// gate here is SSIM, matching how this repo already treats cross-backend/cross-precision real-weight comparisons
/// (see <c>Helpers/Ssim.cs</c>), not the exact-equality bar <c>Krea2DitShardingTests</c> uses for its synthetic F32
/// same-precision config.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class Krea2DitShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public Krea2DitShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }

        string turboDir = TestPaths.Krea2.TurboDir;
        string transformerFile = Path.Combine(turboDir, "krea2_turbo_fp8_scaled.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, transformerFile)) return;

        // Pre-flight VRAM probe on BOTH CUDA ordinals via throwaway backends — per-backend State (Phase 1A) makes
        // these safe to construct and dispose alongside whatever the InferenceEngine builds later; they never touch
        // the engine's own backends. CUDA's own ordinal numbering does NOT match nvidia-smi's PCI order on this box
        // (documented in hartsy-multigpu-overhaul memory) — probe both live rather than assume which physical card
        // is which.
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Krea2DitShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        double freeGb0, freeGb1, totalGb0, totalGb1;
        using (CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir))
        using (CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir))
        {
            (nuint free0, nuint total0) = probe0.Context.GetMemoryInfo();
            (nuint free1, nuint total1) = probe1.Context.GetMemoryInfo();
            freeGb0 = free0 / (1024.0 * 1024.0 * 1024.0); totalGb0 = total0 / (1024.0 * 1024.0 * 1024.0);
            freeGb1 = free1 / (1024.0 * 1024.0 * 1024.0); totalGb1 = total1 / (1024.0 * 1024.0 * 1024.0);
        }
        _output.WriteLine($"Ordinal 0: {freeGb0:F2}/{totalGb0:F1} GB free/total. Ordinal 1: {freeGb1:F2}/{totalGb1:F1} GB free/total.");
        if (freeGb0 < 8.0 || freeGb1 < 8.0)
        {
            _output.WriteLine("SKIPPED: insufficient free VRAM on one or both cards for a real Krea2 run.");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("krea2", modelPathArg: null, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: krea2 checkpoint not resolvable via the catalog."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 1024,
            Height = 1024,
            Steps = 8,
            CfgScale = 1.0f,
            Seed = 42,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/2] UNSHARDED baseline (single GPU, ordinal 0)...");
        ImageResult baseline;
        using (InferenceEngine unshardedEngine = new InferenceEngine("cuda", 0))
        {
            baseline = await unshardedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Width}x{baseline.Height}, seed={baseline.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(baseline.Rgb, "baseline");

        sw.Restart();
        _output.WriteLine("[2/2] SHARDED (DiT block loop split cuda:0 + cuda:1)...");
        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };
        ImageResult sharded;
        using (InferenceEngine shardedEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            sharded = await shardedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  sharded: {sharded.Width}x{sharded.Height}, seed={sharded.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(sharded.Rgb, "sharded");

        Assert.Equal(baseline.Width, sharded.Width);
        Assert.Equal(baseline.Height, sharded.Height);
        double ssim = Ssim.Compute(baseline.Rgb, sharded.Rgb, sharded.Width, sharded.Height);
        _output.WriteLine($"SSIM(baseline, sharded) = {ssim:F4}");
        // Loose bound, not a tight parity gate: see the class doc for why bit/near-bit parity isn't the right bar
        // here (mixed fp8 GEMM paths across an SM 8.6 / SM 8.9+ pair). This asserts "the same image", not "the
        // same image to numerical precision" — a wiring regression (wrong split point, corrupted hand-off) would
        // produce a visibly different or corrupted image, which this WOULD catch.
        Assert.True(ssim > 0.75, $"sharded output diverged too far from the unsharded baseline (SSIM={ssim:F4}) — " +
            "check the block-range hand-off (CopyFromPeer for the joint activation / tembMod) rather than assuming " +
            "this is expected fp8-path drift.");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    /// <summary>Guards the teardown symmetry bug an advisor review caught before this shipped: freeing the DiT after
    /// a sharded generation must free BOTH backends' block ranges, not just the primary's — an asymmetric free
    /// leaves <see cref="PlacementConfig.EnableDitSharding"/>'s shard backend accumulating a fresh block range every
    /// generation. Forces the non-resident path (<c>HARTSY_KEEP_MODELS=0</c>) via the process environment BEFORE
    /// touching <c>Krea2Pipeline</c> — its <c>KeepModelsResident</c> flag is a <c>static readonly</c> field read once
    /// per process, so this only works because the env var is set as the first line of the test and the test runs
    /// alone in its own process (<c>--filter</c> to just this method, or accept it may be stale if run alongside
    /// other tests that touch <c>Krea2Pipeline</c> first in the same process).</summary>
    [Fact]
    public async Task DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations()
    {
        Environment.SetEnvironmentVariable("HARTSY_KEEP_MODELS", "0");

        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string transformerFile = Path.Combine(TestPaths.Krea2.TurboDir, "krea2_turbo_fp8_scaled.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, transformerFile)) return;
        ModelSpec spec = ModelResolver.Resolve("krea2", modelPathArg: null, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: krea2 checkpoint not resolvable via the catalog."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 1024,
            Height = 1024,
            Steps = 8,
            CfgScale = 1.0f,
            Seed = 42,
        };
        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };

        using InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement });
        _output.WriteLine("[1/2] First sharded generation (non-resident: weights freed after)...");
        await engine.Images.GenerateAsync(spec, request);
        (double free0AfterFirst, double free1AfterFirst) = ProbeFreeGb();
        _output.WriteLine($"  after gen 1 — ordinal 0: {free0AfterFirst:F2} GB free, ordinal 1: {free1AfterFirst:F2} GB free");

        _output.WriteLine("[2/2] Second sharded generation (re-loads + re-frees both backends' ranges)...");
        await engine.Images.GenerateAsync(spec, request);
        (double free0AfterSecond, double free1AfterSecond) = ProbeFreeGb();
        _output.WriteLine($"  after gen 2 — ordinal 0: {free0AfterSecond:F2} GB free, ordinal 1: {free1AfterSecond:F2} GB free");

        // A leaked shard-backend free would show ordinal 1's free VRAM monotonically shrinking generation over
        // generation (an extra ~5-6 GB block range never released). 0.5 GB covers allocator fragmentation/pool
        // retention noise without masking a real multi-GB leak.
        double drift0 = free0AfterFirst - free0AfterSecond;
        double drift1 = free1AfterFirst - free1AfterSecond;
        _output.WriteLine($"Drift after gen 2 vs gen 1 — ordinal 0: {drift0:F2} GB, ordinal 1: {drift1:F2} GB");
        Assert.True(Math.Abs(drift0) < 0.5, $"ordinal 0 free VRAM drifted {drift0:F2} GB between generations — possible leak.");
        Assert.True(Math.Abs(drift1) < 0.5, $"ordinal 1 (shard backend) free VRAM drifted {drift1:F2} GB between " +
            "generations — the asymmetric free (EnumerateBlockRangeWeights on the shard backend) may not be firing.");
    }

    private static (double free0Gb, double free1Gb) ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Krea2DitShardingEngineTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }
}

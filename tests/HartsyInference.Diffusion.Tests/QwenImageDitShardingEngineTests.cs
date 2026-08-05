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

/// <summary>Qwen-Image DiT sharding through the FULL engine path (<c>InferenceEngine</c> → <c>QwenImageRecipe</c> →
/// <c>QwenImagePipeline</c>) with <see cref="PlacementConfig.EnableDitSharding"/> — the placement plumbing, the
/// asymmetric preload/free, and the per-step <c>ForwardSharded</c> routing, not the raw transformer primitive
/// <see cref="QwenImageDitShardingVramTests"/> covers. Uses the on-disk Edit-2511 fp8 checkpoint via an explicit
/// model path (the recipe loads Edit checkpoints as T2I). The gate is SSIM vs the unsharded baseline, not
/// bit-parity — the two cards legitimately take different fp8 GEMM paths (SM 8.9 native vs SM 8.6 dequant), the
/// same cross-hardware bar <c>Krea2DitShardingEngineTests</c> documents.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class QwenImageDitShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageDitShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageDitShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        (double free0, double free1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 < 20.0 || free1 < 9.0)
        {
            _output.WriteLine("SKIPPED: the ~19 GB DiT needs most of both cards free (baseline may stream, sharded needs both shares resident).");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("qwen-image", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen-image not resolvable with the explicit path."); return; }

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
        _output.WriteLine("[1/2] UNSHARDED baseline (single GPU, ordinal 0; may block-stream)...");
        ImageResult baseline;
        using (InferenceEngine unshardedEngine = new InferenceEngine("cuda", 0))
        {
            baseline = await unshardedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Width}x{baseline.Height}, seed={baseline.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(baseline.Rgb, "baseline");

        sw.Restart();
        _output.WriteLine("[2/2] SHARDED (60-block loop split cuda:0 + cuda:1)...");
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
        Assert.True(ssim > 0.75, $"sharded output diverged too far from the unsharded baseline (SSIM={ssim:F4}) — " +
            "check the block-range hand-off (img/txt/temb CopyFromPeer) rather than assuming fp8-path drift.");
    }

    /// <summary>Teardown-symmetry regression: with <c>HARTSY_KEEP_MODELS=0</c> two consecutive sharded generations
    /// must not accumulate VRAM on either ordinal — the shard backend's block range is freed by the asymmetric
    /// <c>FreeTransformerWeights</c>, not the unsharded whole-DiT free that would silently no-op there. Same
    /// static-init caveat as the Krea2 twin: run this fact filter-isolated in its own process.</summary>
    [Fact]
    public async Task DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations()
    {
        Environment.SetEnvironmentVariable("HARTSY_KEEP_MODELS", "0");

        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        ModelSpec spec = ModelResolver.Resolve("qwen-image", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen-image not resolvable with the explicit path."); return; }
        (double preFree0, double preFree1) = ProbeFreeGb();
        if (preFree0 < 20.0 || preFree1 < 9.0) { _output.WriteLine("SKIPPED: insufficient free VRAM."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 1024,
            Height = 1024,
            Steps = 4,
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

        double drift0 = free0AfterFirst - free0AfterSecond;
        double drift1 = free1AfterFirst - free1AfterSecond;
        _output.WriteLine($"Drift after gen 2 vs gen 1 — ordinal 0: {drift0:F2} GB, ordinal 1: {drift1:F2} GB");
        Assert.True(Math.Abs(drift0) < 0.5, $"ordinal 0 free VRAM drifted {drift0:F2} GB between generations — possible leak.");
        Assert.True(Math.Abs(drift1) < 0.5, $"ordinal 1 (shard backend) free VRAM drifted {drift1:F2} GB between " +
            "generations — the asymmetric free (EnumerateBlockRangeWeights on the shard backend) may not be firing.");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    private static (double free0Gb, double free1Gb) ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageDitShardingEngineTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }
}

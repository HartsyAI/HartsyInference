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

/// <summary>SD3.5 DiT sharding through the FULL engine path (<c>InferenceEngine</c> → <c>Sd3Recipe</c> →
/// <c>Sd3Pipeline</c>) with <see cref="PlacementConfig.EnableDitSharding"/> — the placement plumbing, the
/// asymmetric preload/free, and the per-step <c>ForwardSharded</c> routing, mirroring
/// <c>QwenImageDitShardingEngineTests</c>/<c>Krea2DitShardingEngineTests</c>. The gate is SSIM vs the unsharded
/// baseline, not bit-parity — the two cards legitimately take different F32 GEMM/SDPA kernel paths cross-SM
/// (measured directly in <c>Sd3DitShardingTests</c>'s doc: same-GPU is bit-exact, cross-GPU is not), the same
/// cross-hardware bar the sibling sharded-model engine tests document.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class Sd3DitShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public Sd3DitShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Sd35.MediumTransformerOnly;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Sd3DitShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        (double free0, double free1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 < 6.0 || free1 < 6.0)
        {
            _output.WriteLine("SKIPPED: the ~5 GB transformer plus text-encoder/VAE headroom needs free space on both cards.");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("sd3", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: sd3 not resolvable with the explicit path."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 768,
            Height = 768,
            Steps = 12,
            CfgScale = 4.5f,
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
        _output.WriteLine("[2/2] SHARDED (JointBlock loop split cuda:0 + cuda:1)...");
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
        Assert.True(ssim > 0.75, $"sharded output diverged too far from the unsharded baseline (SSIM={ssim:F4}) — "
            + "check the block-range hand-off (image/context/temb CopyFromPeer) rather than assuming GEMM-path drift.");
    }

    /// <summary>Teardown-symmetry regression: with <c>HARTSY_KEEP_MODELS=0</c> two consecutive sharded generations
    /// must not accumulate VRAM on either ordinal — mirrors the Qwen-Image/Krea2 twin.</summary>
    [Fact]
    public async Task DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations()
    {
        Environment.SetEnvironmentVariable("HARTSY_KEEP_MODELS", "0");

        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Sd35.MediumTransformerOnly;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        ModelSpec spec = ModelResolver.Resolve("sd3", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: sd3 not resolvable with the explicit path."); return; }
        (double preFree0, double preFree1) = ProbeFreeGb();
        if (preFree0 < 6.0 || preFree1 < 6.0) { _output.WriteLine("SKIPPED: insufficient free VRAM."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 768,
            Height = 768,
            Steps = 6,
            CfgScale = 4.5f,
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
        Assert.True(Math.Abs(drift1) < 0.5, $"ordinal 1 (shard backend) free VRAM drifted {drift1:F2} GB between "
            + "generations — the asymmetric free (EnumerateBlockRangeWeights on the shard backend) may not be firing.");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    private static (double free0Gb, double free1Gb) ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Sd3DitShardingEngineTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }
}

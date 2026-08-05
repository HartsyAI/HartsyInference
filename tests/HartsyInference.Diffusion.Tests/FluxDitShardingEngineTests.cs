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

/// <summary>Flux DiT sharding v1 through the full engine path with the on-disk dev fp8 checkpoint: baseline vs
/// sharded generation SSIM-gated (the two cards take different fp8 GEMM paths — native SM 8.9 vs SM 8.6 dequant —
/// so the bar is same-image, not bit-parity; same rationale as the Krea2/Qwen twins). A genuine pooling demo on
/// this box: the ~11-12 GB fp8 DiT cannot sit on the 3060 at all, yet sharded it holds a multi-GB share.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class FluxDitShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public FluxDitShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Flux.Dev;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(FluxDitShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        (double free0, double free1) = ProbeFreeGb(ptxDir);
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 < 16.0 || free1 < 7.0)
        {
            _output.WriteLine("SKIPPED: insufficient free VRAM for baseline + sharded runs.");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("flux1", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: flux1 not resolvable with the explicit path."); return; }

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
        _output.WriteLine($"  baseline: {baseline.Width}x{baseline.Height}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(baseline.Rgb, "baseline");

        sw.Restart();
        _output.WriteLine("[2/2] SHARDED (57-block flat loop split cuda:0 + cuda:1)...");
        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };
        ImageResult sharded;
        using (InferenceEngine shardedEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            sharded = await shardedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  sharded: {sharded.Width}x{sharded.Height}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(sharded.Rgb, "sharded");

        Assert.Equal(baseline.Width, sharded.Width);
        Assert.Equal(baseline.Height, sharded.Height);
        double ssim = Ssim.Compute(baseline.Rgb, sharded.Rgb, sharded.Width, sharded.Height);
        _output.WriteLine($"SSIM(baseline, sharded) = {ssim:F4}");
        Assert.True(ssim > 0.75, $"sharded output diverged too far from the unsharded baseline (SSIM={ssim:F4}) — " +
            "check the double/single-region hand-offs (img/txt/x/temb CopyFromPeer) rather than assuming fp8-path drift.");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    private static (double free0Gb, double free1Gb) ProbeFreeGb(string ptxDir)
    {
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }
}

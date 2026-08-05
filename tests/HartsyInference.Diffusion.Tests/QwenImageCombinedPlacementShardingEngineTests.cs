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

/// <summary>Component placement COMPOSED with DiT sharding through the full engine path — the user-facing combo
/// "text encoder on GPU B + denoiser split across A+B". Phases are sequential (encode → denoise → decode), so the
/// text encoder and the DiT shard range share GPU B at different times; <c>PlacementPlanner.ValidatePlacement</c>
/// allows this combination (only CfgParallel+sharding is rejected). The gate is SSIM vs a sharded-only run of the
/// same request, not bit-parity — moving the fp8 text encoder to the other card changes its GEMM path (SM 8.9
/// native vs SM 8.6 dequant), the same cross-hardware bar <c>QwenImageDitShardingEngineTests</c> documents.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class QwenImageCombinedPlacementShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageCombinedPlacementShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task CombinedTeVaePlacementAndSharding_RealEngine_MatchesShardedOnlyRun()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageCombinedPlacementShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        (double free0, double free1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 < 16.0 || free1 < 8.0)
        {
            // TE runs on B before B's shard range preloads — sequential phases, so the shares don't add.
            _output.WriteLine("SKIPPED: each card needs its DiT split share free (plus TE/VAE transiently on B).");
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
        _output.WriteLine("[1/2] SHARDED-ONLY reference (DiT split cuda:0 + cuda:1, TE/VAE on primary)...");
        PlacementConfig shardedOnly = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };
        ImageResult shardedResult;
        using (InferenceEngine shardedEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = shardedOnly }))
        {
            shardedResult = await shardedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  sharded-only: {shardedResult.Width}x{shardedResult.Height}, seed={shardedResult.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(shardedResult.Rgb, "sharded-only");

        sw.Restart();
        _output.WriteLine("[2/2] COMBINED (same shard split + TE and VAE placed on cuda:1)...");
        PlacementConfig combined = new PlacementConfig
        {
            ShardDevices = ["cuda:0", "cuda:1"],
            EnableDitSharding = true,
            TextEncoderDevice = "cuda:1",
            VaeDevice = "cuda:1",
        };
        ImageResult combinedResult;
        using (InferenceEngine combinedEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = combined }))
        {
            combinedResult = await combinedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  combined: {combinedResult.Width}x{combinedResult.Height}, seed={combinedResult.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(combinedResult.Rgb, "combined");

        Assert.Equal(shardedResult.Width, combinedResult.Width);
        Assert.Equal(shardedResult.Height, combinedResult.Height);
        double ssim = Ssim.Compute(shardedResult.Rgb, combinedResult.Rgb, combinedResult.Width, combinedResult.Height);
        _output.WriteLine($"SSIM(sharded-only, combined) = {ssim:F4}");
        Assert.True(ssim > 0.75, $"combined placement+sharding diverged too far from the sharded-only run (SSIM={ssim:F4}) — " +
            "the text encoder moved GPUs between runs, so fp8 encoder cross-SM drift is the only expected difference; " +
            "anything larger points at the TE/VAE placement handoff, not the shard split.");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    private static (double free0Gb, double free1Gb) ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageCombinedPlacementShardingEngineTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }
}

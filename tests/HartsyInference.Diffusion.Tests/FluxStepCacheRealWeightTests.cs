using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.7a: the across-step First-Block cache wired into
/// <c>FluxPipeline</c>'s drainFree fast path — the fourth architecture this session (after SD3, Chroma,
/// HiDream-unverified). Flux.1 Dev is guidance-embedded (no true-CFG on this path — <c>trueCfgScale</c> stays
/// at its 1.0 default), so only ONE <see cref="HartsyInference.Diffusion.Utilities.DeviceFeatureCache"/>
/// instance is needed. Arming the cache also forces <c>graphRoute</c> off (an armed cache implies
/// per-step-variable topology a captured CUDA graph can't replay), routing the loop through the sequential
/// <c>RunPlainForward</c> path instead — this test's real subject is proving that hand-off actually happens and
/// produces real reuses, not just that the fast path still runs. NOT wired: the CFG-parallel branch (dual
/// concurrent backends — deliberately out of scope, see the pipeline's own comment) and the host-step branch
/// (ControlNet/Kontext/regional/masked-inpaint — the transformer's own cacheActive gate already excludes these,
/// confirmed unregressed by <c>Flux1RegionalPromptingRealWeightTests</c> passing unchanged after this wiring).
/// <para><b>Calibration:</b> 0.08 is Flux's own proven-safe threshold — neither SD3's 0.03 nor Chroma's 0.15
/// were assumed to transfer. A third distinct number, confirming (again) that this needs a per-model profile.</para></summary>
[Trait("Category", "Integration")]
public sealed class FluxStepCacheRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public FluxStepCacheRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void FluxPipeline_StepCacheEnabled_ReusesFireAndOutputStaysClose()
    {
        // TestPaths.Flux.Dev's FirstExisting list doesn't match this box's actual filename — point at it directly.
        string checkpointPath = "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/BFL/Flux1/flux1-dev-fp8.safetensors";
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Flux Dev checkpoint not found at {checkpointPath}.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photo of a vintage typewriter on a wooden desk, warm light",
            Width = 512,
            Height = 512,
            Steps = 20,
            Seed = 161803,
        };

        // deviceOrdinal 1: this box's CUDA enumeration order does not match nvidia-smi's PCI-bus order (logged
        // as "device 1: NVIDIA GeForce RTX 3060" here) — verified this checkpoint fits comfortably on it.
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 1, ptxDir: ptxDir);
        if (!backend.SupportsDeviceStepCacheGate)
        {
            _output.WriteLine("SKIPPED: backend lacks the device step-cache gate (stepcache.ptx not compiled).");
            return;
        }
        RecipeContext context = new RecipeContext { CheckpointPath = checkpointPath, Backend = backend };
        using IRecipePipeline pipeline = new Flux1Recipe().Construct(context);

        byte[] uncached = Generate(pipeline, request, threshold: null, out string uncachedLog);
        _output.WriteLine($"Generated cache-off ({uncached.Length} bytes). Log: '{uncachedLog}'");
        string offPath = Path.Combine(RepoRoot.Path, "flux_stepcache_off.rgb");
        File.WriteAllBytes(offPath, uncached);

        // Neither SD3's 0.03 nor Chroma's 0.15 transferred exactly — 0.08 (between the two) is Flux's own
        // proven-safe value, confirmed by direct visual inspection: nearly identical to cache-off, only a
        // background prop detail differs (radio vs. stacked books), no scene-level failure.
        byte[] cached = Generate(pipeline, request, threshold: "0.08", out string cachedLog);
        _output.WriteLine($"Generated cache-on ({cached.Length} bytes). Log: '{cachedLog}'");
        string onPath = Path.Combine(RepoRoot.Path, "flux_stepcache_on.rgb");
        File.WriteAllBytes(onPath, cached);
        _output.WriteLine($"Wrote {offPath} and {onPath} for visual inspection.");

        Assert.Contains("Step cache: cond", cachedLog);
        int condReuses = ParseReuses(cachedLog, "cond");
        _output.WriteLine($"Parsed reuses: cond={condReuses}");
        Assert.True(condReuses > 0, $"Step cache armed but never reused a step (log: {cachedLog}) — the gate likely isn't firing, or graphRoute wasn't actually excluded.");

        Assert.Equal(uncached.Length, cached.Length);
        long diffSum = 0;
        for (int i = 0; i < uncached.Length; i++) diffSum += Math.Abs(uncached[i] - cached[i]);
        double meanAbsDiff = diffSum / (double)uncached.Length;
        _output.WriteLine($"Mean absolute per-byte difference (cache-off vs cache-on @ 0.08): {meanAbsDiff:F2}.");
        // 35.0 catches a scene-level failure (SD3/Chroma's bad thresholds measured ~43); this run measured
        // 6.91 — comfortably under even the fine-detail-drift range (~19) SD3/Chroma's good thresholds showed.
        Assert.True(meanAbsDiff < 35.0, $"Cache-on (threshold=0.08) output diverges far enough to suggest a scene-level failure, not fine-detail drift (mean abs diff {meanAbsDiff:F2}) — Flux's proven-safe threshold may have drifted.");
    }

    private static int ParseReuses(string log, string stream)
    {
        int idx = log.IndexOf(stream, StringComparison.Ordinal);
        if (idx < 0) return -1;
        string tail = log[idx..];
        int reusesIdx = tail.IndexOf("reuses", StringComparison.Ordinal);
        if (reusesIdx < 0) return -1;
        int end = tail.LastIndexOf('/', reusesIdx);
        string numSpan = tail[(end + 1)..reusesIdx].Trim();
        return int.Parse(numSpan);
    }

    private byte[] Generate(IRecipePipeline pipeline, ImageRequest request, string? threshold, out string capturedLog)
    {
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", threshold);
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        Logs.SetLogger((level, message) =>
        {
            if (message.Contains("Step cache", StringComparison.Ordinal))
                sb.AppendLine(message);
        });
        try
        {
            ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
            return result.Rgb;
        }
        finally
        {
            capturedLog = sb.ToString().TrimEnd();
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", null);
        }
    }
}

using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.7a: the across-step First-Block cache
/// (<see cref="HartsyInference.Diffusion.Utilities.DeviceFeatureCache"/>) ported into <c>Sd3Transformer</c>/
/// <c>Sd3Pipeline</c> — the reference wiring (<c>ZImagePipeline</c>) is a packed single-stream DiT; SD3 is a
/// dual-stream (image, context) MMDiT, so only the image stream is indicator/residual-carried (context is never
/// read outside the block loop). SD3 also runs true CFG with two independent forwards per step, so this needed
/// two <see cref="HartsyInference.Diffusion.Utilities.DeviceFeatureCache"/> instances (cond/uncond) — the first
/// real exercise of that documented-but-previously-untested requirement (Z-Image's own fastPath only ever runs
/// cache-free CFG).
/// <para><b>Calibration finding (2026-08-10):</b> the GENERIC uncalibrated threshold ("=1"/"true" → 0.10, the
/// same fallback Z-Image itself uses pending its own A/B) is too aggressive for SD3 — at 20 steps it reused 13
/// of them and produced a visibly darker, lower-detail image (mean abs byte diff ~43 vs. cache-off). An explicit
/// tighter threshold (0.03) reused only 5 of 20 steps and stayed visually consistent with cache-off (same
/// subject, exposure, and detail level — only fine pose/detail differences, the expected behavior for a lossy
/// residual-reuse technique). This test locks in 0.03 as SD3's proven-safe operating point pending a real
/// calibrated <c>StepCacheProfile</c> (same status as Z-Image's own "no calibrated profile yet" — this is not a
/// wiring defect, the mechanism demonstrably works, the generic default just isn't tuned per-architecture).</para>
/// Verifies (a) the cache actually fires — real reuses, not just "armed" — by capturing the pipeline's own
/// "Step cache: cond N computes / M reuses" log line via <see cref="Logs.SetLogger"/>, and (b) at the
/// proven-safe threshold, output stays visually consistent with cache-disabled on real SD3.5 Medium weights.</summary>
[Trait("Category", "Integration")]
public sealed class Sd3StepCacheRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public Sd3StepCacheRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Sd3Pipeline_StepCacheEnabled_ReusesFireAndOutputStaysClose()
    {
        string checkpointPath = TestPaths.Sd35.MediumTransformerOnly;
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: SD3.5 checkpoint not found at {checkpointPath}.");
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
            Prompt = "a photo of a small robot in a garden, detailed, soft daylight",
            Width = 512,
            Height = 512,
            Steps = 20,
            CfgScale = 4.5f,
            Seed = 909090,
        };

        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        if (!backend.SupportsDeviceStepCacheGate)
        {
            _output.WriteLine("SKIPPED: backend lacks the device step-cache gate (stepcache.ptx not compiled).");
            return;
        }
        RecipeContext context = new RecipeContext { CheckpointPath = checkpointPath, Backend = backend };
        using IRecipePipeline pipeline = new Sd3Recipe().Construct(context);

        byte[] uncached = Generate(pipeline, request, threshold: null, out string uncachedLog);
        _output.WriteLine($"Generated cache-off ({uncached.Length} bytes). Log: '{uncachedLog}'");
        string offPath = Path.Combine(RepoRoot.Path, "sd3_stepcache_off.rgb");
        File.WriteAllBytes(offPath, uncached);

        // 0.03, not the generic "1"/0.10 default — see the calibration finding in the class doc comment.
        byte[] cached = Generate(pipeline, request, threshold: "0.03", out string cachedLog);
        _output.WriteLine($"Generated cache-on ({cached.Length} bytes). Log: '{cachedLog}'");
        string onPath = Path.Combine(RepoRoot.Path, "sd3_stepcache_on.rgb");
        File.WriteAllBytes(onPath, cached);
        _output.WriteLine($"Wrote {offPath} and {onPath} for visual inspection.");

        // Proof the skip actually fired, not just that the cache was armed.
        Assert.Contains("Step cache: cond", cachedLog);
        int condReuses = ParseReuses(cachedLog, "cond");
        int uncondReuses = ParseReuses(cachedLog, "uncond");
        _output.WriteLine($"Parsed reuses: cond={condReuses}, uncond={uncondReuses}");
        Assert.True(condReuses > 0, $"Step cache armed but cond stream never reused a step (log: {cachedLog}) — the gate likely isn't firing.");
        Assert.True(uncondReuses > 0, $"Step cache armed but uncond stream never reused a step (log: {cachedLog}) — the per-stream cache instances likely aren't independent.");

        Assert.Equal(uncached.Length, cached.Length);
        long diffSum = 0;
        for (int i = 0; i < uncached.Length; i++) diffSum += Math.Abs(uncached[i] - cached[i]);
        double meanAbsDiff = diffSum / (double)uncached.Length;
        _output.WriteLine($"Mean absolute per-byte difference (cache-off vs cache-on @ 0.03): {meanAbsDiff:F2}.");
        // 35.0 is set to catch a SCENE-LEVEL failure, not fine-detail drift: this run measured 18.70 (visually
        // confirmed same subject/exposure/detail), the 0.10-threshold run that visibly washed out the image
        // measured ~43. A tighter bar tuned to this one run's exact number would flake on a different seed —
        // this is a lossy technique by design, so some seed-to-seed spread in the diff is expected and fine.
        Assert.True(meanAbsDiff < 35.0, $"Cache-on (threshold=0.03) output diverges far enough to suggest a scene-level failure like the uncalibrated-threshold case, not fine-detail drift (mean abs diff {meanAbsDiff:F2}) — the reused residual may be stale, or SD3's proven-safe threshold has drifted.");
    }

    private static int ParseReuses(string log, string stream)
    {
        int idx = log.IndexOf(stream, StringComparison.Ordinal);
        if (idx < 0) return -1;
        string tail = log[idx..];
        int reusesIdx = tail.IndexOf("reuses", StringComparison.Ordinal);
        if (reusesIdx < 0) return -1;
        // Walk backward from "reuses" to the preceding integer, e.g. "cond 14 computes / 6 reuses".
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

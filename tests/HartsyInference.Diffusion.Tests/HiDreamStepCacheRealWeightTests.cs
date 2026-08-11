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
/// <c>HiDreamTransformer</c>/<c>HiDreamPipeline</c> — the fifth architecture this session, and the first to
/// combine double-then-single-stream blocks (like Flux/Chroma) with genuine dual-stream true CFG (like SD3),
/// needing two independent <see cref="HartsyInference.Diffusion.Utilities.DeviceFeatureCache"/> instances. On a
/// hit, the cached residual skips the remaining double blocks, the double→single transition, AND every single
/// block — this test is the first real-weight proof that the anchor/reconstruction shape match holds across
/// that transition (previously only confirmed by reading <c>HiDreamConfig</c>, not by an actual run — see the
/// UNVERIFIED-ASSUMPTION comment in <c>HiDreamTransformer.ForwardCore</c>, which this test resolves: no shape
/// exception, real reuses, real image).
/// <para><b>Calibration finding — a NEW failure mode, not just a new threshold number:</b> at 50 steps (the
/// checkpoint's own native step count — 20 steps under-denoises HiDream Full into a formless blur even with
/// caching off entirely, an unrelated discovery made while sanity-checking this test's own baseline), threshold
/// 0.08 with the WHOLE schedule eligible collapsed the image into a flat, textureless color field (mean abs diff
/// 77 vs. cache-off — a much harder failure than SD3/Chroma/Flux's "washed out but recognizable" bad thresholds).
/// Restricting reuse to the back 60% of the schedule (<c>HARTSY_STEP_CACHE_LATE=0.6</c> — the same late-window
/// mechanism the codebase already built for Ideogram 4) fixed it completely: 10/21 reuses per stream, mean abs
/// diff 3.55, visually near-identical to cache-off. Conclusion: HiDream's block-0 indicator reads as low-drift
/// early in the schedule while the true compositional impact of skipping those blocks is severe — reusing early
/// steps destroys the image before it forms; reusing late steps (fine-detail refinement) is safe. This is a
/// second, structurally different failure mode from the "just find a smaller number" pattern SD3/Chroma/Flux
/// showed — the takeaway for the still-uncalibrated fleet pipelines is that late-window may be load-bearing for
/// some of them, not merely a nice-to-have.</para></summary>
[Trait("Category", "Integration")]
public sealed class HiDreamStepCacheRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public HiDreamStepCacheRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void HiDreamPipeline_StepCacheEnabled_ReusesFireAndOutputStaysClose()
    {
        string checkpointPath = "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/HiDream/hidream_i1_full_fp8.safetensors";
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: HiDream checkpoint not found at {checkpointPath}.");
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
            Prompt = "a photo of a ceramic teapot on a windowsill, morning light",
            Width = 512,
            Height = 512,
            // HiDream Full's own native step count — 20 steps under-denoises it into a formless blur even with
            // caching off entirely (found while sanity-checking this test's own baseline; unrelated to caching).
            Steps = 50,
            Seed = 555777,
        };

        // deviceOrdinal 0: CUDA's default (fastest-first) enumeration puts the 4090 here on this box
        // (CUDA_DEVICE_ORDER is unset, so it does NOT match nvidia-smi's PCI-bus order — verified separately).
        // HiDream's transformer (17GB) plus its four text encoders (Llama-3.1-8B ~9GB, T5-XXL ~4.9GB, dual
        // CLIP ~1.5GB combined) need the headroom; they're not all resident simultaneously (text-encode phase
        // frees before the transformer loads) but neither phase fits the 3060's 12GB comfortably.
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        if (!backend.SupportsDeviceStepCacheGate)
        {
            _output.WriteLine("SKIPPED: backend lacks the device step-cache gate (stepcache.ptx not compiled).");
            return;
        }
        RecipeContext context = new RecipeContext { CheckpointPath = checkpointPath, Backend = backend };
        using IRecipePipeline pipeline = new HiDreamRecipe().Construct(context);

        byte[] uncached = Generate(pipeline, request, threshold: null, lateWindow: null, out string uncachedLog);
        _output.WriteLine($"Generated cache-off ({uncached.Length} bytes). Log: '{uncachedLog}'");
        string offPath = Path.Combine(RepoRoot.Path, "hidream_stepcache_off.rgb");
        File.WriteAllBytes(offPath, uncached);

        // HiDream's proven-safe profile: threshold 0.08 ALONE collapsed the image (see class doc) — the
        // late-window restriction is load-bearing here, not optional.
        byte[] cached = Generate(pipeline, request, threshold: "0.08", lateWindow: "0.6", out string cachedLog);
        _output.WriteLine($"Generated cache-on ({cached.Length} bytes). Log: '{cachedLog}'");
        string onPath = Path.Combine(RepoRoot.Path, "hidream_stepcache_on.rgb");
        File.WriteAllBytes(onPath, cached);
        _output.WriteLine($"Wrote {offPath} and {onPath} for visual inspection.");

        Assert.Contains("Step cache: cond", cachedLog);
        Assert.Contains("lateWindow=0.6", cachedLog);
        int condReuses = ParseReuses(cachedLog, "cond");
        int uncondReuses = ParseReuses(cachedLog, "uncond");
        _output.WriteLine($"Parsed reuses: cond={condReuses}, uncond={uncondReuses}");
        Assert.True(condReuses > 0, $"Step cache armed but cond stream never reused a step (log: {cachedLog}) — the gate likely isn't firing, or the double->single anchor/reconstruction shape mismatch is real.");
        Assert.True(uncondReuses > 0, $"Step cache armed but uncond stream never reused a step (log: {cachedLog}) — the per-stream cache instances likely aren't independent.");

        Assert.Equal(uncached.Length, cached.Length);
        long diffSum = 0;
        for (int i = 0; i < uncached.Length; i++) diffSum += Math.Abs(uncached[i] - cached[i]);
        double meanAbsDiff = diffSum / (double)uncached.Length;
        _output.WriteLine($"Mean absolute per-byte difference (cache-off vs cache-on @ threshold=0.08, lateWindow=0.6): {meanAbsDiff:F2}.");
        // 35.0 catches the SD3/Chroma/Flux-style "washed out but recognizable" failure; HiDream's own bad case
        // (whole-schedule eligible) was far worse — a flat color field, mean abs diff 77.40. This run measured
        // 3.55 with the late-window fix, visually confirmed near-identical to cache-off.
        Assert.True(meanAbsDiff < 35.0, $"Cache-on (threshold=0.08, lateWindow=0.6) output diverges far enough to suggest the collapse failure is back (mean abs diff {meanAbsDiff:F2}) — HiDream's proven-safe profile may have drifted.");
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

    private byte[] Generate(IRecipePipeline pipeline, ImageRequest request, string? threshold, string? lateWindow, out string capturedLog)
    {
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE", threshold);
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", lateWindow);
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
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_LATE", null);
        }
    }
}

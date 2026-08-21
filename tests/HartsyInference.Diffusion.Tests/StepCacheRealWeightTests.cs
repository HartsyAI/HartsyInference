using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression for Tier 1.7a: the across-step First-Block cache
/// (<see cref="HartsyInference.Diffusion.Utilities.DeviceFeatureCache"/>) wired into each architecture's pipeline.
/// One shared harness parameterized per model instead of a near-identical file per model (was:
/// Chroma/Flux/HiDream/Sd3, 568 lines combined, differing only in the recipe type, checkpoint, request shape, the
/// calibrated threshold/late-window, whether the model runs true dual-stream CFG (cond+uncond both need to show
/// reuses) or is guidance-embedded (cond alone), and the proven-safe mean-abs-diff ceiling.
///
/// <para><b>The calibration itself does NOT generalize</b> — that's the actual finding this test class carries
/// forward, one row at a time: SD3's proven-safe threshold is 0.03, Chroma's is 0.15, Flux's is 0.08, and
/// HiDream's 0.08 alone collapses the image into a flat color field (mean abs diff 77) unless restricted to the
/// back 60% of the schedule via <c>HARTSY_STEP_CACHE_LATE=0.6</c> — a structurally different failure mode from
/// "just find a smaller number." Each model needs its own real-weight run to prove reuses actually fire (not just
/// "armed") and that the output stays visually consistent with cache-disabled, not a shared assumption.</para></summary>
[Trait("Category", "Integration")]
public sealed class StepCacheRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public StepCacheRealWeightTests(ITestOutputHelper output) => _output = output;

    public sealed record Case(
        string Name, string CheckpointPath, Func<IArchitectureRecipe> RecipeFactory, int DeviceOrdinal,
        string Prompt, int Width, int Height, int Steps, float? Cfg, int Seed,
        string Threshold, string? LateWindow, bool DualStreamCfg, double MeanAbsDiffBound);

    public static TheoryData<Case> Cases() => new()
    {
        // deviceOrdinal 1: this box's CUDA enumeration order does not match nvidia-smi's PCI-bus order — verified
        // this checkpoint fits comfortably on it (the 3060). CfgScale omitted (null): Flux Dev is
        // guidance-embedded, no true CFG on this path.
        new Case("Flux", "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/BFL/Flux1/flux1-dev-fp8.safetensors",
            () => new Flux1Recipe(), DeviceOrdinal: 1,
            "a photo of a vintage typewriter on a wooden desk, warm light", 512, 512, 20, null, 161803,
            Threshold: "0.08", LateWindow: null, DualStreamCfg: false, MeanAbsDiffBound: 35.0),
        // deviceOrdinal 1 (the 4090, more headroom): Chroma's 9.2GB fp8 transformer + T5-XXL is tighter than the
        // 3060's 12GB comfortably allows.
        new Case("Chroma", TestPaths.Chroma.V1,
            () => new ChromaRecipe(), DeviceOrdinal: 1,
            "a photo of a red bicycle leaning against a brick wall", 512, 512, 20, 4.0f, 271828,
            Threshold: "0.15", LateWindow: null, DualStreamCfg: true, MeanAbsDiffBound: 35.0),
        // deviceOrdinal 0: CUDA's default (fastest-first) enumeration puts the 4090 here on this box. HiDream's
        // transformer (17GB) plus its four text encoders need the headroom. CfgScale omitted (null): matches the
        // original test, which never set it.
        new Case("HiDream", "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/HiDream/hidream_i1_full_fp8.safetensors",
            () => new HiDreamRecipe(), DeviceOrdinal: 0,
            // HiDream Full's own native step count — 20 steps under-denoises it into a formless blur even with
            // caching off entirely (unrelated to caching).
            "a photo of a ceramic teapot on a windowsill, morning light", 512, 512, 50, null, 555777,
            Threshold: "0.08", LateWindow: "0.6", DualStreamCfg: true, MeanAbsDiffBound: 35.0),
        new Case("Sd3", TestPaths.Sd35.MediumTransformerOnly,
            () => new Sd3Recipe(), DeviceOrdinal: 0,
            "a photo of a small robot in a garden, detailed, soft daylight", 512, 512, 20, 4.5f, 909090,
            Threshold: "0.03", LateWindow: null, DualStreamCfg: true, MeanAbsDiffBound: 35.0),
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Pipeline_StepCacheEnabled_ReusesFireAndOutputStaysClose(Case c)
    {
        if (!File.Exists(c.CheckpointPath))
        {
            _output.WriteLine($"SKIPPED: {c.Name} checkpoint not found at {c.CheckpointPath}.");
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
            Prompt = c.Prompt,
            Width = c.Width,
            Height = c.Height,
            Steps = c.Steps,
            CfgScale = c.Cfg,
            Seed = c.Seed,
        };

        using CudaBackend backend = new CudaBackend(deviceOrdinal: c.DeviceOrdinal, ptxDir: ptxDir);
        if (!backend.SupportsDeviceStepCacheGate)
        {
            _output.WriteLine("SKIPPED: backend lacks the device step-cache gate (stepcache.ptx not compiled).");
            return;
        }
        RecipeContext context = new RecipeContext { CheckpointPath = c.CheckpointPath, Backend = backend };
        using IRecipePipeline pipeline = c.RecipeFactory().Construct(context);

        byte[] uncached = Generate(pipeline, request, threshold: null, lateWindow: null, out string uncachedLog);
        _output.WriteLine($"[{c.Name}] Generated cache-off ({uncached.Length} bytes). Log: '{uncachedLog}'");
        string offPath = Path.Combine(RepoRoot.Path, $"{c.Name.ToLowerInvariant()}_stepcache_off.rgb");
        File.WriteAllBytes(offPath, uncached);

        byte[] cached = Generate(pipeline, request, threshold: c.Threshold, lateWindow: c.LateWindow, out string cachedLog);
        _output.WriteLine($"[{c.Name}] Generated cache-on ({cached.Length} bytes). Log: '{cachedLog}'");
        string onPath = Path.Combine(RepoRoot.Path, $"{c.Name.ToLowerInvariant()}_stepcache_on.rgb");
        File.WriteAllBytes(onPath, cached);
        _output.WriteLine($"[{c.Name}] Wrote {offPath} and {onPath} for visual inspection.");

        Assert.Contains("Step cache: cond", cachedLog);
        if (c.LateWindow is not null) Assert.Contains($"lateWindow={c.LateWindow}", cachedLog);
        int condReuses = ParseReuses(cachedLog, "cond");
        _output.WriteLine($"[{c.Name}] Parsed reuses: cond={condReuses}" + (c.DualStreamCfg ? $", uncond={ParseReuses(cachedLog, "uncond")}" : ""));
        Assert.True(condReuses > 0, $"[{c.Name}] Step cache armed but cond stream never reused a step (log: {cachedLog}) — the gate likely isn't firing.");
        if (c.DualStreamCfg)
        {
            int uncondReuses = ParseReuses(cachedLog, "uncond");
            Assert.True(uncondReuses > 0, $"[{c.Name}] Step cache armed but uncond stream never reused a step (log: {cachedLog}) — the per-stream cache instances likely aren't independent.");
        }

        Assert.Equal(uncached.Length, cached.Length);
        long diffSum = 0;
        for (int i = 0; i < uncached.Length; i++) diffSum += Math.Abs(uncached[i] - cached[i]);
        double meanAbsDiff = diffSum / (double)uncached.Length;
        _output.WriteLine($"[{c.Name}] Mean absolute per-byte difference (cache-off vs cache-on @ threshold={c.Threshold}" +
            (c.LateWindow is null ? "" : $", lateWindow={c.LateWindow}") + $"): {meanAbsDiff:F2}.");
        // The bound catches a scene-level failure (washed-out/collapsed image), not fine-detail drift — each
        // model's proven-safe threshold above was calibrated to land comfortably under it.
        Assert.True(meanAbsDiff < c.MeanAbsDiffBound,
            $"[{c.Name}] Cache-on output diverges far enough to suggest a scene-level failure, not fine-detail " +
            $"drift (mean abs diff {meanAbsDiff:F2}) — this model's proven-safe threshold may have drifted.");
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

    private static byte[] Generate(IRecipePipeline pipeline, ImageRequest request, string? threshold, string? lateWindow, out string capturedLog)
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

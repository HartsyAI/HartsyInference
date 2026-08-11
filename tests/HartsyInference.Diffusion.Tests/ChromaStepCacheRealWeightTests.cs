using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.7a: the across-step First-Block cache ported into
/// <c>ChromaTransformer</c>/<c>ChromaPipeline</c> — the third distinct shape this session (after SD3's separate
/// dual-stream Forward and Flux's shared ForwardWithTemb): Chroma's <c>ForwardDoubleRange</c>/<c>ForwardSingleRange</c>
/// take <c>ref</c> params and unconditionally dispose their input on every call, so unlike Sd3/Flux's
/// guarded-dispose loops the cache anchor can't just alias the loop's own tensor — block 0 is run alone via
/// <c>ForwardDoubleRange</c>'s own partial-range support (the DiT-sharding primitive), then a device-side
/// snapshot copy of its output becomes the indicator/residual anchor before the remaining blocks run. Chroma
/// runs true CFG through <c>ForwardPaired</c> (two independent <c>ForwardOnePass</c> calls sharing one modTable),
/// needing the same two-cache-instances-per-stream pattern as SD3/HiDream.
/// <para><b>Calibration finding:</b> neither SD3's proven-safe value (0.03) nor the generic uncalibrated
/// default (0.10) transfer to Chroma — 0.03 produced ZERO reuses in 20 steps (Chroma's block-0 indicator drifts
/// faster than SD3's per step), while 0.3 reused 14/20 but visibly washed out the image (mean abs diff 43,
/// matching SD3's scene-level-failure signature). 0.15 reused 11/20 and stayed visually consistent with
/// cache-disabled (same subject/exposure/quality, only fine geometry differs) — locked in here as Chroma's own
/// proven-safe value. Confirms per-model calibration is a real, separate need, not a one-time SD3 fluke.</para>
/// Verifies (a) real reuses fire (log capture), (b) output stays visually consistent with cache-disabled.</summary>
[Trait("Category", "Integration")]
public sealed class ChromaStepCacheRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public ChromaStepCacheRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ChromaPipeline_StepCacheEnabled_ReusesFireAndOutputStaysClose()
    {
        string checkpointPath = TestPaths.Chroma.V1;
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Chroma checkpoint not found at {checkpointPath}.");
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
            Prompt = "a photo of a red bicycle leaning against a brick wall",
            Width = 512,
            Height = 512,
            Steps = 20,
            CfgScale = 4.0f,
            Seed = 271828,
        };

        // deviceOrdinal 1 (the 4090, more headroom): Chroma's 9.2GB fp8 transformer + T5-XXL is tighter than
        // the 3060's 12GB comfortably allows (SD3.5-medium's transformer alone was already this card's
        // documented OOM edge for a smaller transformer — see Sd3Pipeline.cs's own text-encoding-phase comment).
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 1, ptxDir: ptxDir);
        if (!backend.SupportsDeviceStepCacheGate)
        {
            _output.WriteLine("SKIPPED: backend lacks the device step-cache gate (stepcache.ptx not compiled).");
            return;
        }
        RecipeContext context = new RecipeContext { CheckpointPath = checkpointPath, Backend = backend };
        using IRecipePipeline pipeline = new ChromaRecipe().Construct(context);

        byte[] uncached = Generate(pipeline, request, threshold: null, out string uncachedLog);
        _output.WriteLine($"Generated cache-off ({uncached.Length} bytes). Log: '{uncachedLog}'");
        string offPath = Path.Combine(RepoRoot.Path, "chroma_stepcache_off.rgb");
        File.WriteAllBytes(offPath, uncached);

        byte[] cached = Generate(pipeline, request, threshold: "0.15", out string cachedLog);
        _output.WriteLine($"Generated cache-on ({cached.Length} bytes). Log: '{cachedLog}'");
        string onPath = Path.Combine(RepoRoot.Path, "chroma_stepcache_on.rgb");
        File.WriteAllBytes(onPath, cached);
        _output.WriteLine($"Wrote {offPath} and {onPath} for visual inspection.");

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
        _output.WriteLine($"Mean absolute per-byte difference (cache-off vs cache-on @ 0.15): {meanAbsDiff:F2}.");
        // 35.0 catches a SCENE-LEVEL failure, not fine-detail drift: this run measured 18.91 (visually confirmed
        // same subject/exposure/quality, only bike-frame/spoke geometry differs), the threshold=0.3 probe that
        // visibly washed out the image measured 43.17 — same shape as the SD3 finding, different numeric
        // threshold (Chroma needed >0.03 to reuse ANYTHING at all — 0.03 gave 0/20 reuses — but 0.3 was already
        // too loose; 0.15 is the proven-safe middle this run locks in).
        Assert.True(meanAbsDiff < 35.0, $"Cache-on (threshold=0.15) output diverges far enough to suggest a scene-level failure like the threshold=0.3 case, not fine-detail drift (mean abs diff {meanAbsDiff:F2}) — Chroma's proven-safe threshold may have drifted.");
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

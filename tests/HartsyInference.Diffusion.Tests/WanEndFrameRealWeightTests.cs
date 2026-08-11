using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 3.3: Wan's <c>expand_timesteps</c> TI2V path now honors
/// <see cref="VideoRequest.VideoEndFrame"/> symmetrically with <see cref="VideoRequest.InitImage"/> — both get
/// VAE-encoded into a per-frame-pinned latent (<c>WriteFirstFrame</c>/<c>WriteLastFrame</c> in
/// <see cref="HartsyInference.Video.Pipelines.WanVideoPipeline.RunDenoise"/>), each re-imposed at per-frame
/// timestep 0 every denoise step. Previously <c>VideoEndFrame</c> was silently dropped on this (non-concat) path —
/// the gap Tier 0.2 could only paper over with an honest refusal, not fix.
/// <para>Verification approach: use two solid, maximally-distinct synthetic colors (init=red, end=blue) rather than
/// real photos — cheap to construct, and the pass/fail signal (does frame 0 lean red, does the LAST frame lean
/// blue) is unambiguous from simple per-channel means, no perceptual judgment call needed. Still followed by an
/// actual visual PNG inspection per the plan's hard rule, not just the numeric channel check.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanEndFrameRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public WanEndFrameRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task WanVideoRecipe_WithEndFrame_LastFrameLeansTowardSuppliedEndColor()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl)) return;

        // "wan" (the catalog slug), not "wan-22-5b" (the compat-class constant) — ModelResolver.Resolve only
        // reaches a real family id via ModelCatalog.Find, and the catalog has no "wan-22-5b" entry (confirmed:
        // its one Wan video entry is Id="wan"); "wan-22-5b" as modelArg resolves to family "unknown" and every
        // conditioning request gets rejected. This matches every other Wan real-weight test in this file's
        // neighborhood. The actual DiT config still resolves correctly through WanConfigDetector's weight-derived
        // detection (WanVideoRecipe.ResolveConfig falls back to it for any non-compat-class familyId) — this is
        // an engine-service-layer routing quirk, not a gap in the checkpoint-detection itself. The compat-class-
        // specific Supports assertion above is unaffected — it constructs WanVideoRecipe directly, no resolver
        // involved.
        ModelSpec spec = ModelResolver.Resolve("wan", TestPaths.WanVideo.Ti2V5B, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: wan not resolvable with the explicit path."); return; }

        Assert.True(new WanVideoRecipe(WanVideoRecipe.Wan22_5BCompatClassId).Supports.HasFlag(VideoFeatures.EndFrame),
            "WanVideoRecipe should now declare VideoFeatures.EndFrame for wan-22-5b (Tier 3.3).");

        const int size = 480;
        ImageData redInit = SolidColor(size, size, r: 220, g: 20, b: 20);
        ImageData blueEnd = SolidColor(size, size, r: 20, g: 20, b: 220);

        VideoRequest request = new VideoRequest
        {
            Prompt = "a smooth color gradient transition, cinematic",
            Width = size,
            Height = size,
            Frames = 9,
            Steps = 6,
            CfgScale = 5.0f,
            Seed = 1234,
            InitImage = redInit,
            VideoEndFrame = blueEnd,
        };

        VideoGenerationResult result;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            result = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"Generated {result.Frames.Count} frames.");
        Assert.True(result.Frames.Count >= 2, "Need at least 2 frames to compare first vs. last.");

        VideoFrame first = result.Frames[0];
        VideoFrame last = result.Frames[^1];

        (double meanR0, double meanG0, double meanB0) = ChannelMeans(first.Rgb);
        (double meanRN, double meanGN, double meanBN) = ChannelMeans(last.Rgb);
        _output.WriteLine($"Frame 0 channel means: R={meanR0:F1} G={meanG0:F1} B={meanB0:F1}");
        _output.WriteLine($"Frame {result.Frames.Count - 1} channel means: R={meanRN:F1} G={meanGN:F1} B={meanBN:F1}");

        string firstPath = Path.Combine(RepoRoot.Path, "wan_endframe_first.rgb");
        string lastPath = Path.Combine(RepoRoot.Path, "wan_endframe_last.rgb");
        File.WriteAllBytes(firstPath, first.Rgb);
        File.WriteAllBytes(lastPath, last.Rgb);
        _output.WriteLine($"Wrote {firstPath} and {lastPath} ({first.Width}x{first.Height}) for visual inspection.");

        // Frame 0 should lean red (R > B); the last frame should lean blue (B > R) — the model is free to
        // interpolate in between, this only pins the two ends the conditioning actually constrains.
        Assert.True(meanR0 > meanB0, $"Frame 0 should lean toward the red init color (R={meanR0:F1}, B={meanB0:F1}) — firstFrameLatent may not be reaching the denoiser.");
        Assert.True(meanBN > meanRN, $"Last frame should lean toward the blue end color (B={meanBN:F1}, R={meanRN:F1}) — lastFrameLatent/WriteLastFrame may not be reaching the denoiser.");
    }

    private static ImageData SolidColor(int width, int height, byte r, byte g, byte b)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int i = 0; i < rgb.Length; i += 3) { rgb[i] = r; rgb[i + 1] = g; rgb[i + 2] = b; }
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }

    private static (double r, double g, double b) ChannelMeans(byte[] rgb)
    {
        long sumR = 0, sumG = 0, sumB = 0;
        int pixels = rgb.Length / 3;
        for (int i = 0; i < rgb.Length; i += 3) { sumR += rgb[i]; sumG += rgb[i + 1]; sumB += rgb[i + 2]; }
        return (sumR / (double)pixels, sumG / (double)pixels, sumB / (double)pixels);
    }
}

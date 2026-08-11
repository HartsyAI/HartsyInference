using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight test for Tier 3.2's first vertical slice: <c>&lt;segment:X&gt;</c> post-hoc refinement
/// through the real <see cref="HartsyInference.Engine.Services.ImagesService.GenerateAsync"/> orchestration path
/// (not a direct pipeline call — <see cref="HartsyInference.Engine.Features.SegmentRefinement"/> is wired at the
/// SERVICE layer, above any specific recipe, so this has to go through the same entry point a real request does).
/// Covers the CLIPSeg free-text matching path only (no YOLO detector, no <c>&lt;clear:&gt;</c>, single segment) —
/// see the class doc on <see cref="HartsyInference.Engine.Features.SegmentRefinement"/> for the full scope note.
///
/// <para>Verification approach: generate the SAME scene/seed twice on Flux.1 Schnell (already declares
/// <see cref="ImageFeatures.Regional"/> + Img2Img + Inpaint) — once with a plain prompt (baseline), once with a
/// <c>&lt;segment:the red apple&gt;a bright blue apple&lt;segment:end&gt;</c> tag appended. Pass condition: the
/// segment pass actually ran (<c>Meta["segments_refined"]</c> present), and the two images differ by a real,
/// substantial amount (not near-zero, which would mean the segment silently no-op'd) — followed by an actual visual
/// inspection of both, since a substantial diff alone doesn't prove the CHANGE was color-correct or spatially
/// confined, only that something happened.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class SegmentRefinementRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public SegmentRefinementRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ImagesService_WithSegmentTag_RefinesOnlyTheMatchedRegion()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.Flux.Schnell)) return;

        ModelSpec spec = ModelResolver.Resolve("flux1", TestPaths.Flux.Schnell, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: flux1 not resolvable with the explicit path."); return; }

        const string basePrompt = "a red apple and a green pear on a wooden table, studio photography, plain background";
        const int size = 512;

        ImageRequest baselineRequest = new ImageRequest
        {
            Prompt = basePrompt,
            Width = size,
            Height = size,
            Steps = 8,
            CfgScale = 1.0f,
            Seed = 777,
        };

        // Unlike <region:>, a <segment:> part has no "<segment:end>" closer (confirmed by reading
        // PromptRegionParser.ParseSegmentData — there's no special-case for "end" the way region's TryParseRect
        // path has). A trailing <segment:end> would parse as a SECOND segment whose matcher text is literally
        // "end", which correctly matches nothing and is skipped — harmless, but pointless log noise. A segment
        // part just runs until the next tag or end of string.
        //
        // Explicit ",0.95,0.5" overrides (ParseSegmentData: trailing comma value = Strength [mask threshold],
        // second-to-last = Strength2 [creativity/denoise]) — the plain default (Strength2=0.6) round-tripped
        // through Schnell's 4-step schedule only ran 2 real denoise steps (startStep = steps - round(steps*0.6)),
        // which changed the apple's texture/shading (proving the crop/composite mechanism works) but not enough
        // to swap its color — the discriminating signal this test actually wants. Strength2=0.95 at Steps=8 gives
        // the segment pass a real denoise budget (startStep = 8 - round(8*0.95) = 0..1, i.e. nearly full re-denoise).
        string segmentPrompt = basePrompt + "<segment:the red apple,0.95,0.5>a bright blue apple";
        ImageRequest segmentRequest = baselineRequest with
        {
            Prompt = segmentPrompt,
            Regional = new Regional { Plan = segmentPrompt, MaskOversize = 24, MaskGrow = 4, MaskBlur = 6 },
        };

        ImageResult baseline, refined;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Images.GenerateAsync(spec, baselineRequest);
            refined = await engine.Images.GenerateAsync(spec, segmentRequest);
        }

        _output.WriteLine($"Baseline {baseline.Width}x{baseline.Height}, refined {refined.Width}x{refined.Height}.");
        Assert.True(refined.Meta.TryGetValue("segments_refined", out string? countStr) && int.Parse(countStr) >= 1,
            $"Expected refined.Meta['segments_refined'] >= 1 (segment pass should have matched and refined the apple) — Meta had: {string.Join(", ", refined.Meta.Select(kv => $"{kv.Key}={kv.Value}"))}.");

        double diff = MeanAbsDiff(baseline.Rgb, refined.Rgb);
        _output.WriteLine($"Mean absolute per-byte difference (baseline vs segment-refined): {diff:F2}.");

        string basePath = Path.Combine(RepoRoot.Path, "segment_refinement_baseline.rgb");
        string refPath = Path.Combine(RepoRoot.Path, "segment_refinement_refined.rgb");
        File.WriteAllBytes(basePath, baseline.Rgb);
        File.WriteAllBytes(refPath, refined.Rgb);
        _output.WriteLine($"Wrote {basePath} and {refPath} for visual inspection.");

        // A near-zero diff would mean the segment pass matched nothing (or matched but the composite silently
        // no-op'd) despite Meta claiming a refinement happened — that combination is worse than an honest skip.
        Assert.True(diff > 3.0, $"Baseline and segment-refined images are nearly identical (diff {diff:F2}) despite Meta claiming a segment was refined — the composite may be a no-op.");
    }

    private static double MeanAbsDiff(byte[] a, byte[] b)
    {
        long sum = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) sum += Math.Abs(a[i] - b[i]);
        return sum / (double)n;
    }
}

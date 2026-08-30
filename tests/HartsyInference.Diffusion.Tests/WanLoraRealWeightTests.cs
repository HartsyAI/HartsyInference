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

/// <summary>Real-weight regression test for Tier 1.6: <see cref="WanVideoRecipe"/> now merges a requested LoRA
/// stack into the transformer weight dict before <c>LoadWeights</c> (mirroring the existing
/// <c>Sd3Recipe</c>/<c>SdxlRecipe</c>/<c>Flux1Recipe</c> call shape) and declares <see cref="VideoFeatures.Lora"/> —
/// full engine path (<c>InferenceEngine</c> → <c>VideoService</c> → <c>WanVideoRecipe</c>), not a unit test in
/// isolation, so this also proves plan-driven feature validation accepts a LoRA-bearing Wan request.
/// Same-seed A/B (LoRA off vs. on) on real TI2V-5B weights. Only covers the single (dense, non-MoE)
/// expert — this box has no local Wan 2.2 A14B dual-expert checkpoint pair (2x ~14GB) and downloading one just
/// for this test would violate the download→test→delete discipline for something this session already proved
/// the mechanism for; the MoE second-expert merge (<c>WanVideoRecipe.ConstructBase</c>'s
/// <c>loraStack?.ApplyTo(convLow.Transformer, ...)</c> call) is the identical <c>MergedLoraStack.ApplyTo</c> call
/// already exercised here against the primary expert, just against a second dict — not independently verified
/// against real A14B weights. The double-apply itself is sound by inspection: <c>convLow.Transformer</c> is a
/// distinct dictionary from a separate <c>LoadAndConvert</c> call (not the same instance as the primary expert's),
/// and <c>LoraStack.AccumulateDelta</c> only ever reads <c>layer.LoraUp</c>/<c>LoraDown</c> via non-mutating
/// <c>CastTo</c> copies (disposing the copies, never the source layer) — so the parsed <c>LoraLayer</c>s
/// are untouched between the two <c>ApplyTo</c> calls and the second call's grouping is rebuilt fresh from them.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanLoraRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public WanLoraRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task WanVideoRecipe_WithLora_ProducesVisiblyDifferentOutputThanWithout()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl)) return;
        if (!File.Exists(TestPaths.WanVideo.LoraPath))
        {
            _output.WriteLine($"SKIPPED: Wan LoRA not found at {TestPaths.WanVideo.LoraPath}.");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("wan", TestPaths.WanVideo.Ti2V5B, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: wan not resolvable with the explicit path."); return; }

        Assert.True(new WanVideoRecipe().Supports.HasFlag(VideoFeatures.Lora), "WanVideoRecipe should now declare VideoFeatures.Lora.");

        VideoRequest baseRequest = new VideoRequest
        {
            Prompt = "a red ball bouncing on a wooden table",
            Width = 480,
            Height = 480,
            Frames = 9,
            Steps = 6,
            CfgScale = 5.0f,
            Seed = 42,
        };

        VideoGenerationResult withoutLora;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            withoutLora = await engine.Video.GenerateAsync(spec, baseRequest);
        }
        _output.WriteLine($"Generated without LoRA ({withoutLora.Frames.Count} frames).");

        VideoRequest withLoraRequest = baseRequest with
        {
            Loras = new LoraStack { Entries = [new LoraEntry { Model = TestPaths.WanVideo.LoraPath, Weight = 1.0 }] },
        };
        VideoGenerationResult withLora;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            withLora = await engine.Video.GenerateAsync(spec, withLoraRequest);
        }
        _output.WriteLine($"Generated with LoRA ({withLora.Frames.Count} frames).");

        Assert.Equal(withoutLora.Frames.Count, withLora.Frames.Count);
        VideoFrame off0 = withoutLora.Frames[0];
        VideoFrame on0 = withLora.Frames[0];
        Assert.Equal(off0.Rgb.Length, on0.Rgb.Length);

        long diffSum = 0;
        for (int i = 0; i < off0.Rgb.Length; i++)
        {
            diffSum += Math.Abs(off0.Rgb[i] - on0.Rgb[i]);
        }
        double meanAbsDiff = diffSum / (double)off0.Rgb.Length;
        _output.WriteLine($"Mean absolute per-byte difference (frame 0): {meanAbsDiff:F2} (0 would mean byte-identical, i.e. the LoRA had no effect).");

        Directory.CreateDirectory(TestPaths.OutputDir);
        string offPath = Path.Combine(TestPaths.OutputDir, "wan_lora_off_frame0.rgb");
        string onPath = Path.Combine(TestPaths.OutputDir, "wan_lora_on_frame0.rgb");
        File.WriteAllBytes(offPath, off0.Rgb);
        File.WriteAllBytes(onPath, on0.Rgb);
        _output.WriteLine($"Wrote {offPath} and {onPath} ({off0.Width}x{off0.Height}) for visual inspection.");

        Assert.True(meanAbsDiff > 1.0, $"LoRA-on vs. LoRA-off outputs are nearly identical (mean abs diff {meanAbsDiff:F2}) — the merge likely isn't reaching the transformer.");
    }
}

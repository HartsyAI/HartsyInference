using HartsyInference.Cuda;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Vision;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight test for Tier 3.8's <c>&lt;refcrop:N,query[,threshold]&gt;</c> — MiniMax-H3 region-targeted
/// reference conditioning. Two parts, per the design doc's own verification shape
/// (<c>docs/Research/MINIMAX_H3.md</c>): (1) the crop itself, dumped and actually looked at — cheap, and the real
/// direct check of "did CLIPSeg find the right region"; (2) a same-seed A/B through the full ref2va pipeline
/// confirming the crop actually changes conditioning, not just that it runs without throwing.</summary>
[Trait("Category", "Integration")]
public sealed class MiniMaxH3ReferenceCropRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public MiniMaxH3ReferenceCropRealWeightTests(ITestOutputHelper output) => _output = output;

    /// <summary>A neutral gray canvas with a solid red square in its left half — CLIPSeg tends to do better on
    /// color+shape descriptions than flat 50/50 color splits (no shape to key on), matching how the segment
    /// refinement real-weight tests used a real photographed subject rather than a flat color block.</summary>
    private static ImageData RedSquareOnGray(int size)
    {
        byte[] rgb = new byte[size * size * 3];
        for (int i = 0; i < rgb.Length; i += 3) { rgb[i] = 120; rgb[i + 1] = 120; rgb[i + 2] = 120; }
        int squareSize = size / 3;
        int top = size / 3, left = size / 8;
        for (int y = top; y < top + squareSize; y++)
        {
            for (int x = left; x < left + squareSize; x++)
            {
                int px = (y * size + x) * 3;
                rgb[px] = 220; rgb[px + 1] = 20; rgb[px + 2] = 20;
            }
        }
        return new ImageData { Rgb = rgb, Width = size, Height = size };
    }

    [Fact]
    public void RefCrop_IsolatesTheRedSquare_NotTheWholeReference()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }
        string? clipSegDir = VisionModelPaths.FindClipSegDirectory(null);
        if (clipSegDir is null)
        {
            _output.WriteLine("SKIPPED: CLIPSeg weights not found.");
            return;
        }

        const int size = 256;
        ImageData reference = RedSquareOnGray(size);
        VideoRequest request = new VideoRequest
        {
            Prompt = "a calm ocean at dusk <refcrop:1,the red square,0.5>",
            ReferenceImages = [reference],
        };

        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        using ClipSegSegmenter segmenter = new ClipSegSegmenter();
        VideoRequest result = ReferenceCropResolver.Apply(request, backend, segmenter, cancel: default);

        // The tag must not leak into the base prompt (Tier 3.2's own lesson).
        Assert.DoesNotContain("refcrop", result.Prompt, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("a calm ocean at dusk", result.Prompt.Trim());

        Assert.NotNull(result.ReferenceImages);
        ImageData cropped = result.ReferenceImages![0];
        _output.WriteLine($"Reference cropped: {reference.Width}x{reference.Height} -> {cropped.Width}x{cropped.Height}.");

        // Real crop, not a no-op — and small relative to the source (the match is a small square, not "everything").
        Assert.True(cropped.Width < reference.Width || cropped.Height < reference.Height,
            "Reference was not cropped at all — <refcrop:> had no effect.");
        Assert.True((long)cropped.Width * cropped.Height < (long)reference.Width * reference.Height / 2,
            $"Crop {cropped.Width}x{cropped.Height} is more than half the source area — CLIPSeg likely matched too broadly.");

        byte[] croppedRgb = cropped.Rgb;
        double meanR = 0, meanG = 0, meanB = 0;
        int pixels = cropped.Width * cropped.Height;
        for (int i = 0; i < croppedRgb.Length; i += 3) { meanR += croppedRgb[i]; meanG += croppedRgb[i + 1]; meanB += croppedRgb[i + 2]; }
        meanR /= pixels; meanG /= pixels; meanB /= pixels;
        _output.WriteLine($"Cropped region mean RGB: ({meanR:F1},{meanG:F1},{meanB:F1}).");

        Directory.CreateDirectory(TestPaths.OutputDir);
        string outPath = Path.Combine(TestPaths.OutputDir, "h3_refcrop_isolated.rgb");
        File.WriteAllBytes(outPath, croppedRgb);
        _output.WriteLine($"Wrote {outPath} for visual inspection ({cropped.Width}x{cropped.Height}, RGB24 raw).");

        // The crop should be dominated by red, not gray — direct confirmation CLIPSeg found the square, not some
        // gray background patch. (Oversize padding around the match means some gray survives at the edges.)
        Assert.True(meanR > meanG + 30 && meanR > meanB + 30,
            $"Cropped region isn't red-dominated (R={meanR:F1} G={meanG:F1} B={meanB:F1}) — likely matched the wrong region.");
    }

    [Fact]
    public void RefCrop_SameSeedAB_CroppedVsWholeReference_ProducesDifferentVideo()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.MiniMaxH3.DitRef2VaFp8, TestPaths.MiniMaxH3.VideoVae, TestPaths.MiniMaxH3.AudioVae))
        {
            return;
        }
        if (VisionModelPaths.FindClipSegDirectory(null) is null)
        {
            _output.WriteLine("SKIPPED: CLIPSeg weights not found.");
            return;
        }

        const int size = 256, frames = 5, steps = 6;
        ImageData reference = RedSquareOnGray(size);

        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext { CheckpointPath = TestPaths.MiniMaxH3.DitRef2VaFp8, Backend = backend };
        using IVideoRecipePipeline pipeline = new MiniMaxH3Recipe().Construct(context);

        VideoRequest baseRequest = new VideoRequest
        {
            Prompt = "the reference subject appears in a calm ocean scene at dusk",
            ReferenceImages = [reference],
            Width = size,
            Height = size,
            Frames = frames,
            Steps = steps,
            Seed = 4242,
        };
        VideoGenerationResult whole = pipeline.Generate(baseRequest, progress: null, cancel: default);
        _output.WriteLine($"Whole-reference: {whole.Frames[0].Width}x{whole.Frames[0].Height}, {whole.Frames.Count} frame(s).");

        VideoRequest croppedRequest = baseRequest with
        {
            Prompt = "<refcrop:1,the red square,0.5>the reference subject appears in a calm ocean scene at dusk",
        };
        VideoGenerationResult cropped = pipeline.Generate(croppedRequest, progress: null, cancel: default);
        _output.WriteLine($"Cropped-reference: {cropped.Frames[0].Width}x{cropped.Frames[0].Height}, {cropped.Frames.Count} frame(s).");

        Assert.Equal(whole.Frames[0].Width, cropped.Frames[0].Width);
        Assert.Equal(whole.Frames[0].Height, cropped.Frames[0].Height);
        Assert.Equal(whole.Frames.Count, cropped.Frames.Count);

        byte[] wholeRgb = whole.Frames[0].Rgb;
        byte[] croppedRgb = cropped.Frames[0].Rgb;
        long diffSum = 0;
        for (int i = 0; i < wholeRgb.Length; i++) diffSum += Math.Abs(wholeRgb[i] - croppedRgb[i]);
        double meanAbsDiff = diffSum / (double)wholeRgb.Length;
        _output.WriteLine($"Frame 0 mean abs diff (whole vs. cropped reference): {meanAbsDiff:F2}.");

        Directory.CreateDirectory(TestPaths.OutputDir);
        File.WriteAllBytes(Path.Combine(TestPaths.OutputDir, "h3_refcrop_whole_frame0.rgb"), wholeRgb);
        File.WriteAllBytes(Path.Combine(TestPaths.OutputDir, "h3_refcrop_cropped_frame0.rgb"), croppedRgb);
        _output.WriteLine("Wrote both frame-0 outputs for visual inspection.");

        Assert.True(meanAbsDiff > 0.5, $"Cropped vs. whole reference produced nearly identical output (mean abs diff {meanAbsDiff:F2}) — <refcrop:> likely isn't reaching the conditioning.");
    }
}

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

/// <summary>Real-weight regression test for Tier 3.4's remaining scope beyond the VAE-encoder-only round trip
/// (<see cref="LtxVideoVaeEncoderRoundTripRealWeightTests"/>): the full I2V path through
/// <see cref="HartsyInference.Video.Pipelines.LtxVideoPipeline"/>'s per-token AdaLN conditioning
/// (<see cref="HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.LtxVideoBlock"/>'s row-indexed modulation,
/// reusing the SAME <c>AffineBroadcastRowIndexed</c>/<c>GatedResidualRowIndexed</c> primitive
/// <c>MiniMaxH3Transformer</c> already proved for multi-segment/multi-timestep modulation), the recipe-level
/// wiring (<see cref="LtxVideoRecipe.SupportsFor"/>/<see cref="LtxVideoRecipePipeline"/>), through the real
/// engine-service path (<c>InferenceEngine.Video.GenerateAsync</c>) — not a direct pipeline call.
/// <para>Verification approach, matching <c>WanEndFrameRealWeightTests</c>: a solid, maximally-distinct synthetic
/// init color (red) rather than a real photo — the pass/fail signal (does frame 0 read red, do later frames
/// diverge while staying coherent) needs no perceptual judgment call for the numeric half, followed by an actual
/// visual PNG inspection per the plan's hard rule.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LtxVideoI2VRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideoI2VRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task LtxVideoRecipe_WithInitImage_FirstFrameLeansTowardSuppliedColorAndClipStaysCoherent()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.LtxVideo.SingleFile)) return;

        ModelSpec spec = ModelResolver.Resolve("ltx-video", TestPaths.LtxVideo.SingleFile, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: ltx-video not resolvable with the explicit path."); return; }

        Assert.True(new LtxVideoRecipe().SupportsFor(TestPaths.LtxVideo.SingleFile).HasFlag(VideoFeatures.InitImage),
            "LtxVideoRecipe should declare VideoFeatures.InitImage for the base 0.9 checkpoint (Tier 3.4).");

        const int size = 480;   // divisible by patch(4)*2^3=32
        ImageData redInit = SolidColor(size, size, r: 220, g: 20, b: 20);

        // Frames=25/Steps=20, not a smaller/faster pair: LTX's temporal compression is 8, so
        // tLat=(numFrames-1)/8+1 — at 9 frames that's ONLY 2 latent frames, meaning frame-0 conditioning pins
        // half the entire latent and there is no room left to observe denoising progress at all (a real run at
        // 9f/8 steps produced every output frame identically flat-red — indistinguishable from a conditioning-
        // leak bug until frames/steps were raised and a genuine progressive trajectory appeared). 25 frames
        // gives 4 latent frames (3 unconditioned) and 20 steps is enough budget for them to actually denoise.
        // Lowering either number for speed will likely reproduce the false failure, not a real regression.
        VideoRequest request = new VideoRequest
        {
            Prompt = "a red apple slowly rotating on a wooden table, cinematic lighting",
            Width = size,
            Height = size,
            Frames = 25,
            Steps = 20,
            CfgScale = 3.0f,
            Seed = 1234,
            InitImage = redInit,
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

        double frameDiff = MeanAbsDiff(first.Rgb, last.Rgb);
        _output.WriteLine($"Mean absolute per-byte difference (frame 0 vs last frame): {frameDiff:F2}.");

        string firstPath = Path.Combine(RepoRoot.Path, "ltx_i2v_first.rgb");
        string lastPath = Path.Combine(RepoRoot.Path, "ltx_i2v_last.rgb");
        File.WriteAllBytes(firstPath, first.Rgb);
        File.WriteAllBytes(lastPath, last.Rgb);
        _output.WriteLine($"Wrote {firstPath} and {lastPath} ({first.Width}x{first.Height}) for visual inspection.");

        for (int idx = 0; idx < result.Frames.Count; idx += Math.Max(1, result.Frames.Count / 6))
        {
            VideoFrame f = result.Frames[idx];
            string p = Path.Combine(RepoRoot.Path, $"ltx_i2v_frame{idx:D2}.rgb");
            File.WriteAllBytes(p, f.Rgb);
            (double r, double g, double b) = ChannelMeans(f.Rgb);
            _output.WriteLine($"Frame {idx}: R={r:F1} G={g:F1} B={b:F1} -> {p}");
        }

        // Frame 0 must read red — the conditioning mask pins it to the encoded init image every step (see
        // LtxVideoPipeline.RunDenoise's per-step re-pin). Not "leans red" like Wan's blended I2V — LTX's
        // conditioning_mask is exactly 0/1 (diffusers pipeline_ltx_image2video.py), so frame 0 is the DECODED
        // init image, near-exactly, same as the encoder round-trip test's own tolerance.
        Assert.True(meanR0 > meanB0 && meanR0 > 150, $"Frame 0 should read as the red init color (R={meanR0:F1}, G={meanG0:F1}, B={meanB0:F1}) — firstFrameLatent/conditioningMask may not be reaching the denoiser.");
        // Later frames must actually denoise, not just replay frame 0 forever (that would mean the conditioning
        // mask is pinning EVERY token, not just frame 0's — a real bug this test is specifically able to catch).
        Assert.True(frameDiff > 5.0, $"Frame 0 and the last frame are nearly identical (diff {frameDiff:F2}) — every token may be conditioned/pinned instead of just frame 0's (denoising may not be running past frame 0).");
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

    private static double MeanAbsDiff(byte[] a, byte[] b)
    {
        long sum = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) sum += Math.Abs(a[i] - b[i]);
        return sum / (double)n;
    }
}

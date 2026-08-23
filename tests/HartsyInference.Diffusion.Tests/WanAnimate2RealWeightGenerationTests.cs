using HartsyInference.Core.Exceptions;
using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Codec;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight, real-generation coverage for the codex-review fixes touching Wan-Animate-2: the CUDA
/// tiled-SDPA per-head bias fix (finding 1) and the driving-cache preflight dtype-ordering fix (finding 4). Unit
/// tests already pin the mechanism in isolation (<c>KeyOnlySdpaBiasTests</c>,
/// <c>WanAnimate2DrivingCachePolicyTests</c>); this exercises the actual recipe pipeline end to end against the
/// distilled checkpoint and looks at the decoded frame, per this repo's own hard rule that a code-only review is
/// not sufficient sign-off for a video model. Reference and driving content are both real photographic footage
/// (JFK's inaugural address) rather than a synthetic test pattern — a flat-color reference produced a degenerate,
/// prompt-ignoring output the first time this test was written, which a byte-range check alone did not catch.</summary>
[Trait("Category", "Integration")]
public sealed class WanAnimate2RealWeightGenerationTests
{
    private readonly ITestOutputHelper _output;
    public WanAnimate2RealWeightGenerationTests(ITestOutputHelper output) => _output = output;

    private static string DistilledCheckpoint => Path.Combine(
        Path.GetDirectoryName(TestPaths.WanVideo.Animate2)!, "wan_animate_2_distill_int8_convrot.safetensors");

    private static readonly string DrivingVideoPath =
        Path.Combine(TestPaths.ModelsDir, "TestAssets", "restore", "matrix", "jfk.mp4");

    /// <summary>The driving clip's own first frame, decoded to RGB24 — a real single-subject photo (JFK at the
    /// podium) instead of a synthetic flat-color canvas, so a degenerate or prompt-ignoring output is visible
    /// rather than masked by the reference already being trivial.</summary>
    private static ImageData ReferenceImage()
    {
        string framePath = Path.Combine(Path.GetTempPath(), $"hartsy-animate2-ref-{Guid.NewGuid():N}.png");
        int exit = RunFfmpeg($"-y -i \"{DrivingVideoPath}\" -frames:v 1 -update 1 \"{framePath}\"");
        if (exit != 0 || !File.Exists(framePath))
        {
            throw new InvalidOperationException($"ffmpeg failed to extract a reference frame from '{DrivingVideoPath}' (exit {exit}).");
        }
        try
        {
            (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(framePath);
            return new ImageData { Rgb = rgb, Width = width, Height = height };
        }
        finally
        {
            File.Delete(framePath);
        }
    }

    private static int RunFfmpeg(string args)
    {
        System.Diagnostics.ProcessStartInfo psi = new System.Diagnostics.ProcessStartInfo("ffmpeg", args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        using System.Diagnostics.Process proc = System.Diagnostics.Process.Start(psi)!;
        proc.WaitForExit();
        return proc.ExitCode;
    }

    private static VideoGenerationResult RunGeneration(
        (string Key, string? Value)[] env, int width = 384, int height = 384, int frames = 21)
    {
        (string Key, string? Value)[] saved = env.Select(e => (e.Key, Environment.GetEnvironmentVariable(e.Key))).ToArray();
        foreach ((string key, string? value) in env) Environment.SetEnvironmentVariable(key, value);
        try
        {
            string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
            using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
            RecipeContext context = new RecipeContext { CheckpointPath = DistilledCheckpoint, Backend = backend };
            using IVideoRecipePipeline pipeline = new WanAnimate2Recipe().Construct(context);

            VideoClip driving = new VideoClip { Data = File.ReadAllBytes(DrivingVideoPath), Format = "mp4" };
            VideoRequest request = new VideoRequest
            {
                Prompt = "a man in a suit speaking at a podium with microphones, black and white archival footage, a crowd seated behind him",
                Steps = 6,
                CfgScale = 1.0f,
                Width = width,
                Height = height,
                Frames = frames,
                Fps = 16,
                Seed = 1,
                DrivingVideo = driving,
                Extra = new Dictionary<string, object>
                {
                    [WanAnimate2RecipePipeline.ReferenceImageKey] = ReferenceImage(),
                },
            };
            return pipeline.Generate(request, progress: null, cancel: default);
        }
        finally
        {
            foreach ((string key, string? value) in saved) Environment.SetEnvironmentVariable(key, value);
        }
    }

    /// <summary>Baseline generation with default dispatch (whatever the SDPA size heuristic and driving-cache auto
    /// policy pick on this card) — the checkpoint loads, the driving video decodes, and a real frame comes out.
    /// The first frame is dumped for visual inspection, per the repo rule that a scalar diff is not enough
    /// evidence for a video model.</summary>
    [Fact]
    public void Generate_Baseline_ProducesRealFrames()
    {
        if (!RealWeightGate.Require(_output.WriteLine, DistilledCheckpoint, DrivingVideoPath)) return;
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        VideoGenerationResult result = RunGeneration([]);
        Assert.NotEmpty(result.Frames);
        VideoFrame first = result.Frames[0];
        Assert.True(first.Rgb.Length == first.Width * first.Height * 3);

        foreach (int idx in new[] { 0, result.Frames.Count / 2, result.Frames.Count - 1 }.Distinct())
        {
            VideoFrame frame = result.Frames[idx];
            string path = TestImage.SaveBmpDated($"animate2-baseline-frame{idx}", frame.Rgb, frame.Width, frame.Height);
            _output.WriteLine($"Baseline frame {idx} -> {path}");
        }

        // Not a degenerate (all-black/all-gray) output — a real generation has some tonal spread.
        int min = 255, max = 0;
        foreach (byte b in first.Rgb) { if (b < min) min = b; if (b > max) max = b; }
        Assert.True(max - min > 10, $"Frame 0 looks degenerate (byte range {min}-{max}).");
    }

    /// <summary>Finding 1's regression check: the real Wan-Animate-2 log_scale bias forced through the CUDA
    /// tiled-SDPA path (<c>HARTSY_SDPA_FORCE_TILED=1</c>) must produce the same output as the non-tiled path,
    /// same seed. Both runs also pin <c>HARTSY_SDPA_NO_F16=1</c> so the comparison isolates dispatch choice: the
    /// TRUE default dispatch uses Wan's F16 fast path for unmasked self-attention (RMS-normed Q/K, bounded scores),
    /// while <c>SdpaTiledF32</c> is F32-only, so comparing against the unpinned default conflates a real (and
    /// harmless, pre-existing) precision-mode gap with the bias fix under test — an earlier version of this test
    /// did exactly that and mis-read the F16/F32 gap as a divergence. This caller's bias is <c>[1,Skv]</c>
    /// (<c>biasBlocks == 1</c>), which is the one case the fix could have regressed by changing the accumulate
    /// from one batched call to a per-head loop; <c>KeyOnlySdpaBiasTests.TiledPath_PerHeadBias_SelectsTheOwningHeadsRow</c>
    /// is what proves the fix itself, on a synthetic per-head bias no live caller currently sends.</summary>
    [Fact]
    public void Generate_ForcedTiledSdpa_MatchesNonTiled_BothF32()
    {
        if (!RealWeightGate.Require(_output.WriteLine, DistilledCheckpoint, DrivingVideoPath)) return;
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        VideoGenerationResult nonTiledF32 = RunGeneration([("HARTSY_SDPA_NO_F16", "1")]);
        VideoGenerationResult tiledF32 = RunGeneration([("HARTSY_SDPA_NO_F16", "1"), ("HARTSY_SDPA_FORCE_TILED", "1")]);

        Assert.Equal(nonTiledF32.Frames.Count, tiledF32.Frames.Count);
        VideoFrame a = nonTiledF32.Frames[0], b = tiledF32.Frames[0];
        Assert.Equal(a.Width, b.Width);
        Assert.Equal(a.Height, b.Height);

        long sumAbsDiff = 0;
        int worst = 0;
        for (int i = 0; i < a.Rgb.Length; i++)
        {
            int d = Math.Abs(a.Rgb[i] - b.Rgb[i]);
            sumAbsDiff += d;
            if (d > worst) worst = d;
        }
        double meanAbsDiff = (double)sumAbsDiff / a.Rgb.Length;
        _output.WriteLine($"non-tiled-F32 vs forced-tiled-F32: mean|Δ|={meanAbsDiff:F3}, max|Δ|={worst} (of {a.Rgb.Length} bytes)");

        string pathA = TestImage.SaveBmpDated("animate2-nontiled-f32-frame0", a.Rgb, a.Width, a.Height);
        string pathB = TestImage.SaveBmpDated("animate2-forced-tiled-f32-frame0", b.Rgb, b.Width, b.Height);
        _output.WriteLine($"Non-tiled F32 frame 0 -> {pathA}");
        _output.WriteLine($"Forced-tiled F32 frame 0 -> {pathB}");

        // Both paths are F32 GEMM throughout now, so the remaining gap is float accumulation-order noise (the
        // tiled path sums per-tile via separate cuBLAS calls instead of one batched call) — visually confirmed
        // identical (see the two dumped frames), measured at mean|Δ|~1.4/255 on this checkpoint/geometry. A
        // regression from a genuinely wrong bias lookup reads orders of magnitude larger, as the synthetic
        // per-head test (mean masked-region shift ~e^1.3, not noise) demonstrates.
        Assert.True(meanAbsDiff < 5.0, $"Forced-tiled SDPA diverged from the non-tiled F32 path: mean|Δ|={meanAbsDiff:F3}.");
    }

    /// <summary>Finding 4's regression check: pinning the driving cache to F32 must still produce a real, correct
    /// generation at a geometry that comfortably fits — proving the reordered preflight (dtype resolved before the
    /// feasibility floor is sized) does not itself misfire and wrongly refuse a request that fits fine.</summary>
    [Fact]
    public void Generate_PinnedF32DrivingCache_StillGenerates()
    {
        if (!RealWeightGate.Require(_output.WriteLine, DistilledCheckpoint, DrivingVideoPath)) return;
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        VideoGenerationResult result = RunGeneration([("HARTSY_ANIMATE2_BF16_DRIVING_CACHE", "off")]);
        Assert.NotEmpty(result.Frames);
        VideoFrame first = result.Frames[0];

        string path = TestImage.SaveBmpDated("animate2-f32-cache-frame0", first.Rgb, first.Width, first.Height);
        _output.WriteLine($"F32-pinned frame 0 -> {path}");

        int min = 255, max = 0;
        foreach (byte b in first.Rgb) { if (b < min) min = b; if (b > max) max = b; }
        Assert.True(max - min > 10, $"Frame 0 looks degenerate (byte range {min}-{max}).");
    }

    /// <summary>Finding 4's actual refusal path: a geometry too large for the F32 driving cache, with the cache
    /// pinned to F32, must refuse at preflight (before any heavy compute) and NAME F32 in the message — the bug
    /// was that this refusal used to size itself off the smaller BF16 cache regardless of the pin, so a pinned-F32
    /// request that only BF16 could fit passed preflight and OOM'd later with a generic CUDA error instead of
    /// this actionable one.</summary>
    [Fact]
    public void Generate_PinnedF32DrivingCache_RefusesTooLargeAGeometry_NamingF32()
    {
        if (!RealWeightGate.Require(_output.WriteLine, DistilledCheckpoint, DrivingVideoPath)) return;
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        OutOfVramException ex = Assert.Throws<OutOfVramException>(() => RunGeneration(
            [("HARTSY_ANIMATE2_BF16_DRIVING_CACHE", "off")], width: 1280, height: 1280, frames: 161));

        _output.WriteLine(ex.Message);
        Assert.Contains("F32 driving cache", ex.Message, StringComparison.Ordinal);
        Assert.Contains("HARTSY_ANIMATE2_BF16_DRIVING_CACHE", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("BF16 driving cache", ex.Message, StringComparison.Ordinal);
    }
}

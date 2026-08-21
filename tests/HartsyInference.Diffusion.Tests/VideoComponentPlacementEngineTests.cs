using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>TE/VAE component placement through the full engine path, video models — the video-modality twin of
/// <see cref="FluxComponentPlacementEngineTests"/> (which stays separate: it's the lone image-modality case, with
/// a different request/result shape, not worth a one-row Theory). One shared harness parameterized per model
/// instead of a near-identical file per model (was: LtxVideo2/LtxVideo/Wan/WanVae, 336 lines combined, differing
/// only in which of TE/VAE gets placed, the checkpoint(s), the request shape, and whether an audio side-channel
/// exists to verify). Placing a component on the second GPU must reproduce the single-GPU baseline video (and,
/// where present, its audio) — on this box's mismatched SM pair (3060/4090) that's a first-frame SSIM bar, not
/// bit-exactness, since the placed component legitimately takes a different GEMM/conv path cross-hardware.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class VideoComponentPlacementEngineTests
{
    private readonly ITestOutputHelper _output;
    public VideoComponentPlacementEngineTests(ITestOutputHelper output) => _output = output;

    public sealed record Case(
        string Name, string ModelId, string ResolverCheckpoint, string[] GatePaths,
        string Prompt, string NegativePrompt, int Width, int Height, int Frames, int? Fps, int Steps, float Cfg, int Seed,
        bool PlaceTe, bool PlaceVae, bool CheckAudio);

    public static TheoryData<Case> Cases() => new()
    {
        new Case("LtxVideo2", "ltx-2", TestPaths.LtxVideo2.SingleFile,
            [TestPaths.LtxVideo2.SingleFile, TestPaths.LtxVideo2.GemmaEncoder, TestPaths.LtxVideo2.GemmaTokenizer],
            "a cinematic shot of a cat walking through a sunlit garden", "blurry, low quality, distorted, watermark",
            512, 320, 17, 24, 6, 3.0f, 42, PlaceTe: true, PlaceVae: true, CheckAudio: true),
        new Case("LtxVideo", "ltx-video", TestPaths.LtxVideo.SingleFile, [TestPaths.LtxVideo.SingleFile],
            "a cinematic shot of a cat walking through a sunlit garden", "blurry, low quality, distorted, watermark",
            512, 320, 17, 24, 6, 3.0f, 42, PlaceTe: true, PlaceVae: true, CheckAudio: false),
        new Case("Wan (TE placement)", "wan", TestPaths.WanVideo.Ti2V5B, [TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl],
            "a red ball bouncing on a wooden table", "",
            480, 480, 9, null, 6, 5.0f, 42, PlaceTe: true, PlaceVae: false, CheckAudio: false),
        new Case("Wan (VAE placement)", "wan", TestPaths.WanVideo.Ti2V5B, [TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl],
            "a red ball bouncing on a wooden table", "",
            480, 480, 9, null, 6, 5.0f, 42, PlaceTe: false, PlaceVae: true, CheckAudio: false),
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task ComponentPlacement_RealEngine_MatchesSingleGpuBaseline(Case c)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, c.GatePaths)) return;

        ModelSpec spec = ModelResolver.Resolve(c.ModelId, c.ResolverCheckpoint, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine($"SKIPPED: {c.ModelId} not resolvable with the explicit path."); return; }

        VideoRequest request = new VideoRequest
        {
            Prompt = c.Prompt,
            NegativePrompt = c.NegativePrompt,
            Width = c.Width,
            Height = c.Height,
            Frames = c.Frames,
            Fps = c.Fps,
            Steps = c.Steps,
            CfgScale = c.Cfg,
            Seed = c.Seed,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[{c.Name}] [1/2] Baseline (everything on ordinal 0)...");
        VideoGenerationResult baseline;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Frames.Count} frames, " +
            $"{(baseline.Audio is null || baseline.Audio.IsEmpty ? "no audio" : $"{baseline.Audio.Channels!.Length}ch @ {baseline.Audio.SampleRate} Hz")}, " +
            $"{sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        string placedWhat = (c.PlaceTe, c.PlaceVae) switch
        {
            (true, true) => "TE + VAE",
            (true, false) => "TE",
            (false, true) => "VAE",
            _ => "(nothing — case is misconfigured)",
        };
        _output.WriteLine($"[{c.Name}] [2/2] {placedWhat} placed on ordinal 1 (rest stays on ordinal 0)...");
        PlacementConfig placement = new PlacementConfig
        {
            TextEncoderDevice = c.PlaceTe ? "cuda:1" : null,
            VaeDevice = c.PlaceVae ? "cuda:1" : null,
        };
        VideoGenerationResult placed;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            placed = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  placed: {placed.Frames.Count} frames, " +
            $"{(placed.Audio is null || placed.Audio.IsEmpty ? "no audio" : $"{placed.Audio.Channels!.Length}ch @ {placed.Audio.SampleRate} Hz")}, " +
            $"{sw.Elapsed.TotalSeconds:F1}s");

        Assert.Equal(baseline.Frames.Count, placed.Frames.Count);
        VideoFrame b0 = baseline.Frames[0];
        VideoFrame p0 = placed.Frames[0];
        double ssim = Ssim.Compute(b0.Rgb, p0.Rgb, b0.Width, b0.Height);
        _output.WriteLine($"[{c.Name}] first-frame SSIM(baseline, {placedWhat} placed) = {ssim:F4}");
        Assert.True(ssim > 0.75, $"[{c.Name}] {placedWhat}-placed output diverged from baseline (SSIM={ssim:F4}) — " +
            "check the backend routing and the LOAD-BEARING host materialization sweeps.");

        if (!c.CheckAudio) return;

        // Audio-VAE + vocoder path: only asserted when the checkpoint's side models actually produced a soundtrack,
        // so this stays a real check, not a vacuous one.
        bool baselineHasAudio = baseline.Audio is not null && !baseline.Audio.IsEmpty;
        bool placedHasAudio = placed.Audio is not null && !placed.Audio.IsEmpty;
        Assert.Equal(baselineHasAudio, placedHasAudio);
        if (baselineHasAudio)
        {
            Assert.Equal(baseline.Audio!.SampleRate, placed.Audio!.SampleRate);
            Assert.Equal(baseline.Audio.Channels!.Length, placed.Audio.Channels!.Length);
            float[] b = baseline.Audio.Channels[0];
            float[] p = placed.Audio.Channels[0];
            int n = Math.Min(b.Length, p.Length);
            double sumSq = 0, refSumSq = 0;
            for (int i = 0; i < n; i++) { double d = b[i] - p[i]; sumSq += d * d; refSumSq += (double)b[i] * b[i]; }
            double relL2 = refSumSq > 0 ? Math.Sqrt(sumSq / refSumSq) : 0;
            _output.WriteLine($"[{c.Name}] audio channel-0 relative L2(baseline, placed) = {relL2:F4} ({n} samples)");
            Assert.True(relL2 < 0.25, $"[{c.Name}] {placedWhat}-placed audio diverged from baseline (relL2={relL2:F4}) — " +
                "check the audio-VAE+vocoder VaeBackend routing.");
        }
        else
        {
            _output.WriteLine($"[{c.Name}] (checkpoint has no audio VAE/vocoder side models — audio path not exercised)");
        }
    }
}

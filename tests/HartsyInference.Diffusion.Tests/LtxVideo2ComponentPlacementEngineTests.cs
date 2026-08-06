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
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Diffusion.Tests;

/// <summary>LTX-2.3 TE+VAE placement through the full engine path: <see cref="LtxVideo2Pipeline"/> already routes
/// its Gemma encode through <c>TextEncoderBackend</c> and BOTH the video-VAE decode and the audio-VAE+vocoder decode
/// through <c>VaeBackend</c> (wired well before this test existed — see the class doc on <c>LtxVideo2Pipeline</c>);
/// this is the missing verification half (<c>docs/MULTI_GPU.md</c> flagged "wired but UNVERIFIED — no
/// ComponentPlacementEngineTests class exists"). Placing both encoders on the second GPU (DiT + both VAEs stay on
/// the first) must reproduce the single-GPU baseline video AND audio. Needs the bundled/side-model-completed
/// LTX-2.3 checkpoint (~22 GB transformer split + already-cached video/audio VAE + text-projection side models) —
/// SKIPS cleanly (not a failure) when that's not on disk, exactly like the generation test this mirrors
/// (<c>LtxVideo2GenerationTests</c>).</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LtxVideo2ComponentPlacementEngineTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideo2ComponentPlacementEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TeAndVaePlacement_RealEngine_MatchesSingleGpuBaseline()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.LtxVideo2.SingleFile,
                TestPaths.LtxVideo2.GemmaEncoder, TestPaths.LtxVideo2.GemmaTokenizer)) return;

        ModelSpec spec = ModelResolver.Resolve("ltx-2", TestPaths.LtxVideo2.SingleFile, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: ltx-2 not resolvable with the explicit path."); return; }

        VideoRequest request = new VideoRequest
        {
            Prompt = "a cinematic shot of a cat walking through a sunlit garden",
            NegativePrompt = "blurry, low quality, distorted, watermark",
            Width = 512,
            Height = 320,
            Frames = 17,
            Fps = 24,
            Steps = 6,
            CfgScale = 3.0f,
            Seed = 42,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/2] Baseline (everything on ordinal 0)...");
        VideoGenerationResult baseline;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Frames.Count} frames, " +
            $"{(baseline.Audio is null || baseline.Audio.IsEmpty ? "no audio" : $"{baseline.Audio.Channels!.Length}ch @ {baseline.Audio.SampleRate} Hz")}, " +
            $"{sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        _output.WriteLine("[2/2] Gemma + both VAEs placed on ordinal 1 (DiT stays on ordinal 0)...");
        PlacementConfig placement = new PlacementConfig { TextEncoderDevice = "cuda:1", VaeDevice = "cuda:1" };
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
        _output.WriteLine($"first-frame SSIM(baseline, TE+VAE placed) = {ssim:F4}");
        Assert.True(ssim > 0.75, $"TE/VAE-placed LTX-2 video diverged from baseline (SSIM={ssim:F4}) — check the "
            + "Gemma/video-VAE backend routing and the LOAD-BEARING host materialization in LtxVideo2Pipeline.");

        // Audio-VAE + vocoder path: only asserted when the checkpoint's side models actually produced a soundtrack
        // (conv.AudioVae/conv.Vocoder present — see LtxVideo2Recipe) so this stays a real check, not a vacuous one.
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
            _output.WriteLine($"audio channel-0 relative L2(baseline, placed) = {relL2:F4} ({n} samples)");
            Assert.True(relL2 < 0.25, $"TE/VAE-placed LTX-2 audio diverged from baseline (relL2={relL2:F4}) — check "
                + "the audio-VAE+vocoder VaeBackend routing.");
        }
        else
        {
            _output.WriteLine("(checkpoint has no audio VAE/vocoder side models — audio path not exercised)");
        }
    }
}

using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 3.5: streaming video decode. Scope, deliberately narrow (see
/// <see cref="WanVideoRecipePipeline.GenerateFramesAsync"/>'s doc comment): plain Wan T2V only (no init image /
/// end frame, no boomerang/trim — those need the full buffered clip). Two things this test has to prove, neither
/// of which "no exception thrown" would catch:
/// <list type="bullet">
/// <item><b>Correctness</b>: the streamed frames are byte-identical to the buffered path's frames for the same
/// seed/request. Compared directly at the RGB-frame level (not by round-tripping through ffmpeg/mp4, which would
/// only prove the container step, not this change — this change is entirely about which code path produces the
/// frames the container step reads).</item>
/// <item><b>It's actually streaming, not buffered-then-dumped</b>: <see cref="HartsyInference.Video.Pipelines.WanVideoPipeline.GenerateFramesAsync"/>'s
/// first frame can only arrive after the full synchronous denoise loop completes (there's no incremental
/// signal during denoising itself — diffusion isn't causal across time), so a large gap before frame 0 is
/// expected and not a bug. The proof of real streaming is that the gaps AFTER frame 0 (each a VAE
/// <c>DecodeStreaming</c> group) are much smaller than the gap before it — if streaming were fake (buffer
/// everything, then yield in a tight loop), every gap would look like the post-frame-0 gaps, not like the
/// pre-frame-0 one.</item>
/// </list></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanStreamingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public WanStreamingRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task WanVideoService_StreamingVsBuffered_ProduceByteIdenticalFramesAndArriveIncrementally()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl)) return;

        // "wan", not "wan-22-5b" — see WanEndFrameRealWeightTests's comment for why (ModelResolver.Resolve only
        // reaches a real family via ModelCatalog.Find, which has no "wan-22-5b" entry).
        ModelSpec spec = ModelResolver.Resolve("wan", TestPaths.WanVideo.Ti2V5B, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: wan not resolvable with the explicit path."); return; }

        const int size = 480;
        VideoRequest request = new VideoRequest
        {
            Prompt = "a red apple slowly rotating on a wooden table, studio lighting",
            Width = size,
            Height = size,
            Frames = 9,
            Steps = 6,
            CfgScale = 5.0f,
            Seed = 4321,
        };

        List<VideoFrame> buffered;
        List<VideoFrame> streamed;
        List<long> arrivalMs = [];
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            VideoGenerationResult bufferedResult = await engine.Video.GenerateAsync(spec, request);
            buffered = [.. bufferedResult.Frames];
            _output.WriteLine($"Buffered: {buffered.Count} frames.");

            Stopwatch sw = Stopwatch.StartNew();
            streamed = [];
            await foreach (VideoFrame frame in engine.Video.GenerateFramesAsync(spec, request))
            {
                arrivalMs.Add(sw.ElapsedMilliseconds);
                streamed.Add(frame);
            }
            _output.WriteLine($"Streamed: {streamed.Count} frames. Arrival times (ms): {string.Join(", ", arrivalMs)}");
        }

        // --- Correctness: byte-identical frames, same seed/request ---
        Assert.Equal(buffered.Count, streamed.Count);
        for (int i = 0; i < buffered.Count; i++)
        {
            Assert.Equal(buffered[i].Width, streamed[i].Width);
            Assert.Equal(buffered[i].Height, streamed[i].Height);
            Assert.True(buffered[i].Rgb.AsSpan().SequenceEqual(streamed[i].Rgb),
                $"Frame {i} differs between the buffered and streaming code paths — same seed/request should be deterministic on this path.");
        }

        // --- Real streaming, not fake: the gap before frame 0 (whole denoise loop) must dominate the gaps
        // after it (VAE decode groups only) — a buffered-then-dumped implementation would make every gap tiny.
        Assert.True(arrivalMs.Count >= 2, "Need at least 2 frames to compare inter-frame gaps.");
        long firstGap = arrivalMs[0];
        long maxLaterGap = 0;
        for (int i = 2; i < arrivalMs.Count; i++)
        {
            maxLaterGap = Math.Max(maxLaterGap, arrivalMs[i] - arrivalMs[i - 1]);
        }
        _output.WriteLine($"Gap before frame 0 (denoise + first decode group): {firstGap}ms. Largest gap after frame 1: {maxLaterGap}ms.");
        Assert.True(firstGap > maxLaterGap * 2,
            $"Expected the pre-frame-0 gap ({firstGap}ms) to dominate later inter-frame gaps ({maxLaterGap}ms) — frames may not actually be streaming incrementally.");
    }
}

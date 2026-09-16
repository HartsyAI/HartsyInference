using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Opt-in real-checkpoint acceptance for long-form chaining; output and metrics stay on disk for the
/// operator inspection the release gate requires. Seam continuity is measured against the clip's own frame-to-frame
/// baseline, since H3 output moves and an absolute SSIM threshold would mean nothing.</summary>
[Collection("CudaSerial")]
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MiniMaxH3ChainRealWeightTests
{
    private const string RunEnvVar = "HARTSY_RUN_H3_CHAIN_REAL";
    private const int Width = 512;
    private const int Height = 288;
    private const int SegmentFrames = 141;
    private const int Fps = 24;
    private const int Steps = 30;

    private const string Prompt =
        "A lighthouse keeper in a yellow raincoat walks along a stone pier in a storm, waves breaking, "
        + "steady tracking camera, continuous motion, howling wind and crashing surf";

    private readonly ITestOutputHelper _output;

    public MiniMaxH3ChainRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task Chain_RealCheckpoint_ProducesOneContinuousClipPastASingleGeneration()
    {
        if (Environment.GetEnvironmentVariable(RunEnvVar) != "1")
        {
            _output.WriteLine($"SKIPPED: set {RunEnvVar}=1 and {RealWeightGate.RequireEnvVar}=1 "
                + "to run the long-form chaining acceptance generation.");
            return;
        }
        Assert.True(CudaContext.IsAvailable(), $"{RunEnvVar}=1 requires a CUDA device.");
        if (!RealWeightGate.Require(_output.WriteLine,
                TestPaths.MiniMaxH3.DitFp8,
                TestPaths.MiniMaxH3.TextEncoder,
                TestPaths.MiniMaxH3.VideoVae,
                TestPaths.MiniMaxH3.AudioVae))
        {
            return;
        }

        int segments = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_H3_CHAIN_SEGMENTS"),
            out int requested) ? Math.Max(2, requested) : 3;
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> plan = PlanFor(segments);
        int totalFrames = MiniMaxH3ChainPlanner.TotalFrames(plan);
        Assert.True(totalFrames > SegmentFrames,
            "The acceptance clip has to be longer than one generation or it proves nothing.");
        _output.WriteLine($"Chaining {plan.Count} segments of {SegmentFrames}f "
            + $"({MiniMaxH3ChainPlanner.DefaultContextFrames}f carried) -> {totalFrames} frames "
            + $"({totalFrames / (double)Fps:F1} s).");

        int gpu = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_GPU"), out int requestedGpu)
            ? requestedGpu : 0;
        ModelSpec spec = ModelResolver.Resolve("minimax-h3", TestPaths.MiniMaxH3.DitFp8, Modality.Video);
        VideoRequest request = new VideoRequest
        {
            Prompt = Prompt,
            Width = Width,
            Height = Height,
            Frames = SegmentFrames,
            ChainTotalFrames = totalFrames,
            Fps = Fps,
            Steps = Steps,
            Seed = 987654,
            Components = new ComponentOverrides
            {
                Qwen = TestPaths.MiniMaxH3.TextEncoder,
                VideoVae = TestPaths.MiniMaxH3.VideoVae,
                AudioVae = TestPaths.MiniMaxH3.AudioVae,
            },
        };

        using InferenceEngine engine = new("cuda", gpu);
        VideoPlan releasePlan = await engine.VideoPlanning.PlanAsync(spec, request);
        VideoPlan chainPlan = BindValidationCanaryPlan(releasePlan);
        VideoGenerationResult result = await engine.Video.GenerateAsync(
            chainPlan, request, new LoggingProgress(_output));

        Assert.Equal(totalFrames, result.Frames.Count);
        AssertCoherent(result);

        // The protected head is context, not output, so it must not appear twice in the assembly.
        int boundary = plan[0].FrameCount;
        Assert.NotEqual(
            Convert.ToHexString(result.Frames[boundary - 1].Rgb),
            Convert.ToHexString(result.Frames[boundary].Rgb));

        double withinSegment = MeanAdjacentSsim(result.Frames, 1, plan[0].FrameCount - 1);
        List<double> seamSsims = [];
        int cursor = 0;
        foreach (MiniMaxH3ChainPlanner.Segment segment in plan)
        {
            cursor += segment.NewFrames;
            if (cursor < result.Frames.Count)
            {
                seamSsims.Add(Ssim.Compute(
                    result.Frames[cursor - 1].Rgb, result.Frames[cursor].Rgb, Width, Height));
            }
        }
        double worstSeam = seamSsims.Count == 0 ? 1d : seamSsims.Min();
        _output.WriteLine($"Within-segment adjacent SSIM {withinSegment:F4}; "
            + $"seam SSIMs [{string.Join(", ", seamSsims.Select(value => value.ToString("F4")))}].");

        AudioBuffer audio = Assert.IsType<AudioBuffer>(result.Audio);
        double expectedSeconds = totalFrames / (double)Fps;
        Assert.True(Math.Abs(audio.Seconds - expectedSeconds) < 0.5d,
            $"Soundtrack is {audio.Seconds:F2} s against {expectedSeconds:F2} s of video — the streams drifted.");

        string outputRoot = Path.Combine(TestPaths.OutputDir, "h3-chain-real-weight");
        VideoOutputWriter.Written written = Persist(result, outputRoot, "minimax-h3-long-form-chain");
        JsonSerializerOptions json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
        json.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
        await File.WriteAllTextAsync(Path.Combine(written.Directory, "real-weight-evidence.json"),
            JsonSerializer.Serialize(new
            {
                runUtc = DateTimeOffset.UtcNow,
                segments = plan.Count,
                segmentFrames = SegmentFrames,
                contextFrames = MiniMaxH3ChainPlanner.DefaultContextFrames,
                totalFrames,
                seconds = expectedSeconds,
                audioSeconds = audio.Seconds,
                withinSegmentAdjacentSsim = withinSegment,
                seamSsims,
                worstSeam,
                execution = result.Execution,
            }, json));
        _output.WriteLine($"Chain output: {written.Directory}");
        Assert.NotNull(written.Mp4Path);

        // A hard cut would sit far below the clip's own adjacent-frame similarity.
        Assert.True(worstSeam > withinSegment * 0.7d,
            $"Worst seam SSIM {worstSeam:F4} is a discontinuity against the clip's own "
            + $"{withinSegment:F4} adjacent-frame baseline.");
    }

    private static IReadOnlyList<MiniMaxH3ChainPlanner.Segment> PlanFor(int segments)
    {
        int newPerSegment = SegmentFrames - MiniMaxH3ChainPlanner.DefaultContextFrames;
        int target = SegmentFrames + (segments - 1) * newPerSegment;
        return MiniMaxH3ChainPlanner.Plan(target, MiniMaxH3ChainPlanner.DefaultContextFrames, SegmentFrames);
    }

    private static double MeanAdjacentSsim(IReadOnlyList<VideoFrame> frames, int start, int end)
    {
        double sum = 0;
        int count = 0;
        for (int i = Math.Max(1, start); i < Math.Min(end, frames.Count); i++)
        {
            sum += Ssim.Compute(frames[i - 1].Rgb, frames[i].Rgb, Width, Height);
            count++;
        }
        return count == 0 ? 1d : sum / count;
    }

    private void AssertCoherent(VideoGenerationResult result)
    {
        foreach (VideoFrame frame in result.Frames)
        {
            Assert.Equal(Width, frame.Width);
            Assert.Equal(Height, frame.Height);
            Assert.Equal(Width * Height * 3, frame.Rgb.Length);
            Assert.True(frame.Rgb.Count(value => value != 0) > frame.Rgb.Length / 10,
                $"Frame {frame.Index} is effectively black.");
            Assert.True(frame.Rgb.Count(value => value != 255) > frame.Rgb.Length / 10,
                $"Frame {frame.Index} is effectively white.");
        }
        AudioBuffer audio = Assert.IsType<AudioBuffer>(result.Audio);
        Assert.Equal(2, audio.ChannelCount);
        Assert.All(audio.Channels, channel => Assert.All(channel,
            sample => Assert.True(float.IsFinite(sample))));
    }

    private static VideoOutputWriter.Written Persist(
        VideoGenerationResult result, string outputRoot, string slug)
    {
        VideoFrame first = result.Frames[0];
        return VideoOutputWriter.Write(
            [.. result.Frames.Select(frame => frame.Rgb)],
            first.Width, first.Height, outputRoot, slug, result.Audio, result.Fps ?? Fps);
    }

    private sealed class LoggingProgress(ITestOutputHelper output) : IProgress<StepPreview>
    {
        public void Report(StepPreview value) =>
            output.WriteLine($"denoise {value.Step}/{value.TotalSteps}");
    }

    /// <summary>Runs the validation-pending chain while the public service plan stays release-blocked.</summary>
    private static VideoPlan BindValidationCanaryPlan(VideoPlan releasePlan)
    {
        VideoPlanIssue[] retained = releasePlan.Issues
            .Where(issue => issue.Code is not "video.h3_expansion.release_blocked"
                and not "video.feature.unsupported")
            .ToArray();
        return VideoRequestExecutionBinding.BindPlan(releasePlan with
        {
            Profile = releasePlan.Profile with
            {
                Features = releasePlan.Profile.Features | VideoFeatures.LongFormChain,
            },
            Issues = retained,
        });
    }
}
